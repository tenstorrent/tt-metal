# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.tt.model.row_pipelined_model import RowPipelinedModel
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_weight_config
from models.demos.deepseek_v3.utils.hf_model_utils import load_model_weights
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict
from models.perf.benchmarking_utils import BenchmarkProfiler


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 0.0


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a copy of the HF state_dict with leading 'model.' stripped.

    Deepseek TT modules expect keys like 'embed_tokens.', 'layers.', 'norm.',
    but HF weights are under 'model.'.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            out[k[len("model.") :]] = v
        else:
            out[k] = v
    return out


_WORD_RE = re.compile(r"\w+('\w+)?|\S", flags=re.UNICODE)


def _normalize_text(s: str, ignore_case: bool = False, strip_punct: bool = False):
    if ignore_case:
        s = s.lower()
    toks = _WORD_RE.findall(s)
    if strip_punct:
        toks = [t for t in toks if re.search(r"\w", t)]
    return toks


def _compare_texts(gen: str, ref: str):
    """Strict comparison: exact match required, case- and punctuation-sensitive."""
    gen_toks = _normalize_text(gen)
    ref_toks = _normalize_text(ref)
    sm = difflib.SequenceMatcher(a=ref_toks, b=gen_toks, autojunk=False)
    opcodes = sm.get_opcodes()

    subs = sum(1 for tag, *_ in opcodes if tag == "replace")
    dels = sum((i2 - i1) for tag, i1, i2, *_ in opcodes if tag == "delete")
    ins = sum((j2 - j1) for tag, *_, j1, j2 in opcodes if tag == "insert")
    N = max(1, len(ref_toks))
    wer = (subs + dels + ins) / N

    exact = (wer == 0.0) and (len(gen_toks) == len(ref_toks))
    if exact:
        return True, ""

    # compact diff report
    diff_lines = []
    for i, (tag, i1, i2, j1, j2) in enumerate(opcodes[:10]):
        if tag == "equal":
            continue
        diff_lines.append(
            f"  [{i}] {tag.upper()}: ref='{ ' '.join(ref_toks[i1:i2]) }', gen='{ ' '.join(gen_toks[j1:j2]) }'"
        )

    report = (
        f"Exact match required (WER==0). Got WER={wer:.4f} "
        f"(ref_tokens={len(ref_toks)}, gen_tokens={len(gen_toks)})\n" + "\n".join(diff_lines)
    )
    return False, report


class DeepseekGenerator:
    """
    Simple generator that wires RowPipelinedModel + LMHead for decode-only inference.

    Notes:
    - Prefill at the model level is not fully implemented in RowPipelinedModel; we emulate
      prefill by iterating decode steps over the prompt tokens (updates caches).
    - Batch size in configs is tied to USERS_PER_ROW; for simplicity we decode
      up to that many sequences. If fewer are provided, we pad/ignore extras.

    Usage:
    - Context manager (recommended):
      ```python
      with DeepseekGenerator(...) as gen:
          output = gen.generate(...)
      ```
    - Manual cleanup:
      ```python
      gen = DeepseekGenerator(...)
      output = gen.generate(...)
      gen.cleanup_all()  # Cleanup is mandatory
      ```
    """

    def __init__(
        self,
        hf_config: AutoConfig | None = None,
        mesh_device: ttnn.MeshDevice | None = None,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = USERS_PER_ROW,
        tokenizer=None,
        random_weights: bool = False,
        dense_layers: int | None = None,
        override_num_layers: int | None = None,
        single_layer: str | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.model_path = str(model_path)
        self.batch_size = min(USERS_PER_ROW, batch_size)
        self.cache_dir = cache_dir

        # Load HF config + tokenizer
        self.hf_config = (
            hf_config if hf_config is not None else AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        )
        # self._ensure_max_seq_len(self.hf_config)
        self.hf_config.max_seq_len = 4096  # TODO: Change this when needed?
        # Optional overrides for layer counts before building states
        if override_num_layers is not None:
            try:
                self.hf_config.num_hidden_layers = int(override_num_layers)
            except Exception:
                pass
        if dense_layers is not None:
            try:
                self.hf_config.first_k_dense_replace = int(dense_layers)
            except Exception:
                pass
        # Tokenizer is optional; caller can pass a tokenizer or handle failure.
        self.tokenizer = tokenizer

        # Runtime helpers
        self.ccl = CCL(mesh_device)
        mesh_shape = list(mesh_device.shape)
        self.dp_factor = mesh_shape[1]
        # Weight cache to avoid loading weights multiple times
        self._weight_ttnn_cache: dict[str, ttnn.Tensor] = {}
        # Paged attention setup
        self.paged_config = MLA1D.get_valid_paged_config(self.hf_config.max_seq_len, USERS_PER_ROW, self.dp_factor)
        self.page_tables_tt = [
            MLA1D.create_page_table(
                paged_config=self.paged_config,
                mesh_device=mesh_device,
            )
            for _ in range(self.hf_config.num_hidden_layers)
        ]
        self.rope = RotarySetup(device=mesh_device, batch_size_per_row=USERS_PER_ROW, hf_config=self.hf_config)

        # Prepare weights/configs
        self.random_weights = random_weights
        self.single_layer = single_layer
        self._prepare_weight_configs(cache_dir)

    @staticmethod
    def _ensure_max_seq_len(hf_config) -> None:
        if getattr(hf_config, "max_seq_len", None) is not None:
            return
        try:
            if getattr(hf_config, "rope_scaling", None):
                factor = hf_config.rope_scaling.get("factor")
                orig = hf_config.rope_scaling.get("original_max_position_embeddings")
                if factor and orig:
                    hf_config.max_seq_len = int(factor * orig)
                    return
            if getattr(hf_config, "max_position_embeddings", None):
                hf_config.max_seq_len = int(hf_config.max_position_embeddings)
                return
        except Exception:
            pass
        hf_config.max_seq_len = 4096

    def _prepare_weight_configs(self, cache_dir: str | Path | None) -> None:
        weight_cache_path = Path(cache_dir) if cache_dir is not None else Path("generated/deepseek_v3")
        weight_cache_path.mkdir(parents=True, exist_ok=True)

        if self.random_weights:
            if self.single_layer and self.single_layer.lower() == "moe":
                raise NotImplementedError(
                    "Random weights with 'moe' single layer is not supported by RowPipelinedModel demo yet. Use 'mlp' or disable random mode."
                )
            logger.info("Building random weights from HF reference model (ForCausalLM)...")
            ref_model = DeepseekV3ForCausalLM(self.hf_config).eval()
            # Ensure parameter/buffer dtype matches downstream expectations (bfloat16)
            ref_model = ref_model.to(dtype=torch.bfloat16)
            torch_state = ref_model.state_dict()
            # Quantize MLP weights as expected by TT converters
            torch_state = add_inv_scale_to_state_dict(
                torch_state,
                block_shape=self.hf_config.quantization_config["weight_block_size"],
            )
            model_state = {
                k: v
                for k, v in torch_state.items()
                if k.startswith("model.embed_tokens.")
                or k.startswith("model.layers.")
                or k.startswith("model.norm.")
                or k.startswith("lm_head.")
            }
        else:
            logger.info(f"Loading HF weights from {self.model_path} (this may take a while)...")
            hf_weights = load_model_weights(self.model_path)
            logger.info("HF weights loaded")

            if "lm_head.weight" not in hf_weights:
                raise RuntimeError(
                    "No HF safetensors found in model path or missing 'lm_head.weight'. "
                    "Set DEEPSEEK_V3_HF_MODEL to a directory containing DeepSeek-V3 safetensors, or pass --model-path."
                )
            model_state = {
                k: v
                for k, v in hf_weights.items()
                if k.startswith("model.embed_tokens.")
                or k.startswith("model.layers.")
                or k.startswith("model.norm.")
                or k.startswith("lm_head.")
            }
        # Convert weights to TT tensors-on-disk and build weight_config
        logger.info("Converting weights to TTNN SavedWeight format (RowPipelinedModel)...")
        self.model_weight_config = get_weight_config(
            ModuleClass=RowPipelinedModel,
            hf_config=self.hf_config,
            state_dicts=(model_state,),
            weight_cache_path=weight_cache_path,
            mesh_device=self.mesh_device,
            force_recalculate=False,
        )

    def _prepare_model_states(self) -> None:
        logger.info("Creating model states...")
        self.model_state = RowPipelinedModel.create_state(
            hf_config=self.hf_config, mesh_device=self.mesh_device, paged_config=self.paged_config, ccl=self.ccl
        )
        logger.info("Creating model shared states...")
        self.model_shared_state = RowPipelinedModel.create_shared_state(
            hf_config=self.hf_config, mesh_device=self.mesh_device
        )

    def _prepare_run_configs(self, mode: str) -> None:
        if mode == "prefill":
            logger.info("Creating model prefill config...")
            self.model_prefill_cfg = RowPipelinedModel.prefill_model_config(
                hf_config=self.hf_config, mesh_device=self.mesh_device
            )
            self._prepare_model_states()
            self.model_run_config_prefill = create_run_config(
                self.model_prefill_cfg,
                self.model_weight_config,
                self.model_state,
                self.model_shared_state,
                cached_ttnn_weights=self._weight_ttnn_cache,
            )
        elif mode == "decode":
            logger.info("Creating model decode config...")
            assert (
                hasattr(self, "model_state") and self.model_state is not None
            ), "Model state must be prepared before creating decode run config. Run _prepare_run_configs('prefill') first."
            assert (
                hasattr(self, "model_shared_state") and self.model_shared_state is not None
            ), "Model shared state must be prepared before creating decode run config. Run _prepare_run_configs('prefill') first."
            self.model_decode_cfg = RowPipelinedModel.decode_model_config(
                hf_config=self.hf_config, mesh_device=self.mesh_device
            )
            self.model_run_config_decode = create_run_config(
                self.model_decode_cfg,
                self.model_weight_config,
                self.model_state,
                self.model_shared_state,
                cached_ttnn_weights=self._weight_ttnn_cache,
            )
        else:
            raise ValueError(f"Unknown run config mode: {mode}")

        logger.info(f"Model run config created for {mode}...")

    def _cleanup_run_configs(self, mode: str) -> None:
        if mode == "prefill":
            if hasattr(self, "model_run_config_prefill") and self.model_run_config_prefill is not None:
                del self.model_run_config_prefill
            else:
                logger.info("No prefill run config to cleanup")
        elif mode == "decode":
            if hasattr(self, "model_run_config_decode") and self.model_run_config_decode is not None:
                del self.model_run_config_decode
            else:
                logger.info("No decode run config to cleanup")
        else:
            raise ValueError(f"Unknown run config mode: {mode}")

    def clear_ttnn_weight_cache(self) -> None:
        """Clear the TTNN weight cache to free up memory."""
        # Deallocate all TTNN tensors before clearing the cache
        for tensor_path, tensor in self._weight_ttnn_cache.items():
            try:
                ttnn.deallocate(tensor)
            except Exception as e:
                logger.warning(f"Failed to deallocate tensor {tensor_path}: {e}")
        self._weight_ttnn_cache.clear()

    def cleanup_all(self) -> None:
        """Comprehensive cleanup of all resources managed by the generator."""
        # Clear TTNN weight cache
        try:
            self.clear_ttnn_weight_cache()
        except Exception as e:
            logger.warning(f"Failed to clear weight cache: {e}")

        # Clean up run configs
        try:
            self._cleanup_run_configs("prefill")
        except Exception as e:
            logger.warning(f"Failed to cleanup prefill run config: {e}")

        try:
            self._cleanup_run_configs("decode")
        except Exception as e:
            logger.warning(f"Failed to cleanup decode run config: {e}")

        # Clean up model states
        try:
            if hasattr(self, "model_state") and self.model_state is not None:
                del self.model_state
        except Exception as e:
            logger.warning(f"Failed to cleanup model state: {e}")

        try:
            if hasattr(self, "model_shared_state") and self.model_shared_state is not None:
                del self.model_shared_state
        except Exception as e:
            logger.warning(f"Failed to cleanup model shared state: {e}")

        # Clean up page tables (TTNN tensors)
        try:
            if hasattr(self, "page_tables_tt") and self.page_tables_tt is not None:
                for i, page_table in enumerate(self.page_tables_tt):
                    try:
                        ttnn.deallocate(page_table)
                    except Exception as e:
                        logger.warning(f"Failed to deallocate page table {i}: {e}")
                del self.page_tables_tt
        except Exception as e:
            logger.warning(f"Failed to cleanup page tables: {e}")

        # Clean up RoPE setup
        try:
            if hasattr(self, "rope") and self.rope is not None:
                del self.rope
        except Exception as e:
            logger.warning(f"Failed to cleanup RoPE setup: {e}")

        # Clean up CCL
        try:
            if hasattr(self, "ccl") and self.ccl is not None:
                del self.ccl
        except Exception as e:
            logger.warning(f"Failed to cleanup CCL: {e}")

        # Clean up configs
        try:
            if hasattr(self, "model_prefill_cfg") and self.model_prefill_cfg is not None:
                del self.model_prefill_cfg
            if hasattr(self, "model_decode_cfg") and self.model_decode_cfg is not None:
                del self.model_decode_cfg
            if hasattr(self, "model_weight_config") and self.model_weight_config is not None:
                del self.model_weight_config

        except Exception as e:
            logger.warning(f"Failed to cleanup model configs: {e}")

        # Clean up paged config
        try:
            if hasattr(self, "paged_config") and self.paged_config is not None:
                del self.paged_config
        except Exception as e:
            logger.warning(f"Failed to cleanup paged config: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_all()

    def _tt_from_tokens_step(self, tokens_step: torch.Tensor) -> ttnn.Tensor:
        """Tokens step: [B] -> TTNN tensor [1, 1, B] uint32, replicated to mesh."""
        assert tokens_step.dim() == 1
        x = tokens_step.view(1, 1, -1).to(torch.int32)
        return ttnn.from_torch(
            x,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _tt_from_positions(self, positions: torch.Tensor) -> Tuple[dict, ttnn.Tensor]:
        """Return rope tensors dict and TTNN positions shard for decode.

        positions: [B] int tensor
        returns: (rope_tensors, tt_positions)
        """
        # Build RoPE tensors for current positions
        rope_mats = self.rope.get_rot_mats(positions.to(torch.int32))
        rope_tensors = {
            "cos_matrix": rope_mats["cos_matrix"],
            "sin_matrix": rope_mats["sin_matrix"],
            "trans_matrix": rope_mats["trans_matrix"],
        }

        # Create TTNN position tensor as INT32 with the same sharding pattern used in tests
        mesh_shape = list(self.mesh_device.shape)
        tt_positions = ttnn.from_torch(
            positions.to(torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=mesh_shape),
            dtype=ttnn.int32,
        )
        return rope_tensors, tt_positions

    def _decode_step(self, tokens_step: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Run a single decode step and return logits on host as torch tensor [1, 1, B, V]."""
        # Prepare TT inputs
        tt_tokens = self._tt_from_tokens_step(tokens_step)
        rope_tensors, tt_positions = self._tt_from_positions(positions)

        # RowPipelinedModel forward
        logits_tt = RowPipelinedModel.forward_decode(
            tt_tokens,
            tt_positions,
            self.model_run_config_decode,
            rope_tensors,
            self.page_tables_tt,
        )
        # Gather to host
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))

        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(logits_tt)

        return logits  # [1, 1, B, V]

    def _prefill(self, tokens: torch.Tensor, user_id: int) -> torch.Tensor:
        """Run prefill for the full prompt sequence and return logits for the last position.

        Args:
            tokens: [1, 1, seq_len] padded token sequences
            user_id: user id for the prefill

        Returns:
            logits: [1, 1, seq_len, V] logits for the full sequence
        """

        tokens = tokens.view(1, 1, -1)
        seq_len = tokens.shape[2]

        # Prepare TT inputs for prefill - reshape to [1, 1, actual_seq_len]
        tt_tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # RoPE setup for prefill
        rope_setup = RotarySetup(
            device=self.mesh_device,
            batch_size_per_row=1,
            hf_config=self.hf_config,
        )

        rot_mats = rope_setup.get_rot_mats_table(seq_len)
        rope_tensors = {
            "cos_matrix": rot_mats["cos_matrix"],
            "sin_matrix": rot_mats["sin_matrix"],
            "trans_matrix": rot_mats["trans_matrix"],
        }

        # RowPipelinedModel forward prefill
        logits_tt = RowPipelinedModel.forward_prefill(
            tt_tokens, user_id, self.model_run_config_prefill, rope_tensors, self.page_tables_tt
        )

        # Gather to host
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))

        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(logits_tt)
        return logits  # [1, 1, seq_len, V]

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)  # [B]

    def _pad_batch(self, tokens_list: List[List[int]]) -> Tuple[torch.Tensor, List[int]]:
        """Pad/pack a list of token id sequences to batch of size USERS_PER_ROW.

        Returns
            tokens_packed: torch.LongTensor [USERS_PER_ROW, S]
            valid_counts: list of actual sequence lengths for first N sequences
        """
        assert len(tokens_list) > 0 and len(tokens_list) <= USERS_PER_ROW
        max_len = max(len(t) for t in tokens_list)
        # Round up to nearest multiple of TILE_SIZE
        max_len = ((max_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        out = torch.full((USERS_PER_ROW, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        lengths = torch.zeros((USERS_PER_ROW,), dtype=torch.int32)
        for i, seq in enumerate(tokens_list):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            lengths[i] = len(seq)
        return out, lengths

    def generate(
        self,
        prompts: Iterable[str],
        max_new_tokens: int = 32,
        sampling: SamplingParams | None = None,
        teacher_forcing=None,
        early_print_first_user: bool = True,
        repeat_batches: int = 1,
    ) -> Tuple[List[List[int]], dict]:
        """Generate tokens for the given prompts using greedy decode by default.

        early_print_first_user: If True, prints generated tokens for the first user
                                at each step. Better for demo visibility.

        Returns: (list of generated token id lists for the provided prompts (order preserved), statistics dictionary)
        """
        # Initialize profiler
        profiler = BenchmarkProfiler()
        profiler.start("run")

        prompts = list(prompts)
        num_of_prompts = len(prompts)
        assert 1 <= num_of_prompts <= USERS_PER_ROW, f"Supports 1..{USERS_PER_ROW} prompts"

        logger.info("Creating model run configs...")
        profiler.start("preparing_prefill_config")
        self._prepare_run_configs("prefill")
        profiler.end("preparing_prefill_config")

        profiler.start("preparing_decode_config")
        self._prepare_run_configs("decode")
        profiler.end("preparing_decode_config")

        # Tokenize using HF chat template
        profiler.start("tokenizing")
        encoded: List[List[int]] = [self._encode_prompt(p) for p in prompts]
        tokens_batched, lengths = self._pad_batch(encoded)  # [USERS_PER_ROW, seq_len]
        profiler.end("tokenizing")

        logger.info(f"Lengths of (encoded) prompts: {lengths}")

        # Average prompt length for prefill calculation
        avg_prompt_len = float(lengths[0] if len(lengths) > 0 else 0)

        # Accumulators for per-run stats so we can average later
        all_prefill_time: List[float] = []
        all_total_decode_time: List[float] = []
        all_decode_tps_per_user: List[float] = []
        all_decode_tps: List[float] = []
        all_prefill_tps: List[float] = []
        all_time_to_first_token: List[float] = []

        for run_idx in range(repeat_batches):
            logger.info(f"Starting generation run {run_idx + 1}/{repeat_batches}...")

            # Prefill
            profiler.start("inference_prefill")
            num_of_users = tokens_batched.shape[0]
            last_logits = []
            for user_id in range(num_of_users):
                if lengths[user_id] == 0:
                    logger.info(f"Skipping prefill for user_id: {user_id} as prompt length is 0")
                    last_logits.append(torch.zeros(self.hf_config.vocab_size))
                    continue
                logger.info(f"Running prefill for user_id: {user_id}")
                logger.info(
                    f"Input to the prefill: {self.tokenizer.decode(tokens_batched[user_id].tolist(), skip_special_tokens=True)}"
                )
                user_out = self._prefill(tokens_batched[user_id], user_id)
                user_out = user_out[0, 0, -1:, :].squeeze(0)  # [ 1, 1, seq_len, V] -> [V]
                last_logits.append(user_out)
            last_logits = torch.stack(last_logits)
            profiler.end("inference_prefill")

            assert len(last_logits) == num_of_users

            logger.info(f"Finished prefill for all users...")

            # First sampled token after prompt
            next_tokens = self._sample_greedy(last_logits)

            # Decode
            positions = torch.zeros(USERS_PER_ROW, dtype=torch.int32) + lengths

            # If teacher forcing is enabled, collect the model's predicted token and force GT for next step (single prompt)
            if teacher_forcing is not None:
                # Only enforce for the first user to keep scope minimal
                forced = teacher_forcing.collect_predicted_tokens(int(next_tokens[0].item()))
                next_tokens[0] = int(forced)

            generations: List[List[int]] = [[] for _ in range(num_of_prompts)]
            logger.info(f"Generating {max_new_tokens} tokens for {num_of_prompts} user(s)...")
            if early_print_first_user:
                logger.info("===== Generation for first user =====")

            profiler.start("inference_decode")
            for gen_idx in range(max_new_tokens):
                # Decode one step with previous next_tokens
                profiler.start(f"decode_time_{gen_idx}")
                logits = self._decode_step(next_tokens, positions).squeeze(0).squeeze(0)
                profiler.end(f"decode_time_{gen_idx}")

                pred_tokens = self._sample_greedy(logits)
                if teacher_forcing is not None:
                    forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                    pred_tokens[0] = int(forced)
                next_tokens = pred_tokens
                positions += 1

                # Collect only for the original batch size
                for i in range(num_of_prompts):
                    token_value = int(next_tokens[i].item())
                    generations[i].append(token_value)
                    if early_print_first_user and i == 0:
                        print(self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True)

            profiler.end("inference_decode")

            if teacher_forcing is not None:
                acc = teacher_forcing.compute_accuracy()
                logger.info(
                    f"[Teacher-Forced Accuracy] top1={acc.get('top1', 0.0)*100:.2f}% "
                    f"top5={(acc.get('top5')*100 if acc.get('top5') is not None else float('nan')):.2f}%"
                )

            if early_print_first_user:
                logger.info("\n===== Done =====")

            # Per-run statistics collection
            prefill_time = profiler.get_duration("inference_prefill")
            decode_times = [profiler.get_duration(f"decode_time_{i}") for i in range(max_new_tokens)]

            # Get config preparation times once we exit the loop later, like before
            # Calculate statistics for this run
            # prefill throughput
            prefill_tokens_per_sec = (avg_prompt_len / prefill_time) * num_of_prompts if prefill_time > 0 else 0

            # decode throughput (excluding first iteration compile time)
            if len(decode_times) > 1:
                total_decode_time = sum(decode_times[1:])  # Exclude iteration 0 (compile time)
                decode_tokens_per_sec_per_user = (
                    (max_new_tokens - 1) / total_decode_time if total_decode_time > 0 else 0
                )
            else:
                total_decode_time = sum(decode_times)
                decode_tokens_per_sec_per_user = 0
            decode_tokens_per_sec = decode_tokens_per_sec_per_user * num_of_prompts

            # latency to first token
            avg_time_to_first_token = prefill_time / num_of_prompts if num_of_prompts > 0 else 0

            # Accumulate so we can average across repeat_batches
            all_prefill_time.append(prefill_time)
            all_total_decode_time.append(total_decode_time)
            all_decode_tps_per_user.append(decode_tokens_per_sec_per_user)
            all_decode_tps.append(decode_tokens_per_sec)
            all_prefill_tps.append(prefill_tokens_per_sec)
            all_time_to_first_token.append(avg_time_to_first_token)

        profiler.end("run")

        # Get config preparation times
        prefill_config_time = profiler.get_duration("preparing_prefill_config")
        decode_config_time = profiler.get_duration("preparing_decode_config")

        # Average the per-run stats we collected
        denom = float(repeat_batches) if repeat_batches > 0 else 1.0

        avg_prefill_time = sum(all_prefill_time) / denom if all_prefill_time else 0
        avg_total_decode_time = sum(all_total_decode_time) / denom if all_total_decode_time else 0
        avg_decode_tps_per_user = sum(all_decode_tps_per_user) / denom if all_decode_tps_per_user else 0
        avg_decode_tps = sum(all_decode_tps) / denom if all_decode_tps else 0
        avg_prefill_tps = sum(all_prefill_tps) / denom if all_prefill_tps else 0
        avg_time_to_first_token = sum(all_time_to_first_token) / denom if all_time_to_first_token else 0

        statistics = {
            "preparing_prefill_config": prefill_config_time / denom,
            "preparing_decode_config": decode_config_time / denom,
            "inference_prefill": avg_prefill_time,
            "inference_decode": avg_total_decode_time,
            "prefill_time_to_token": avg_time_to_first_token,
            "prefill_t/s": avg_prefill_tps,
            "decode_t/s/u": avg_decode_tps_per_user,
            "decode_t/s": avg_decode_tps,
            "Full demo runtime": profiler.get_duration("run") / denom,
        }

        # Add teacher forcing accuracy to statistics if applicable
        if teacher_forcing is not None:
            final_acc = teacher_forcing.compute_accuracy()
            statistics["token_accuracy_top1"] = final_acc.get("top1")
            statistics["token_accuracy_top5"] = final_acc.get("top5")

        return generations, statistics

    def _encode_prompt(self, prompt: str) -> List[int]:
        # Use HF chat template if a tokenizer is provided; otherwise synthesize simple token ids
        if self.tokenizer is not None:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
            return list(ids)

        # Fallback: deterministic dummy tokenization for random-weights mode
        vocab = int(getattr(self.hf_config, "vocab_size", 32768))
        # Start with a BOS-like token 0
        out: List[int] = [0]
        data = prompt.encode("utf-8")[:16]
        for i, b in enumerate(data):
            out.append(int((b + i) % vocab))
        # Append an EOS-like token 1 if within vocab
        if vocab > 1:
            out.append(1)
        return out


__all__ = ["DeepseekGenerator", "SamplingParams"]
