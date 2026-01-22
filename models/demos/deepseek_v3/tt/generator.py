# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.weight_config import get_weight_config
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


class DeepseekGenerator:
    """
    Simple generator that wires RowBatchedModel + LMHead for decode-only inference.

    Notes:
    - Prefill at the model level is not fully implemented in RowBatchedModel; we emulate
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
        enable_trace: bool = False,
    ) -> None:
        self.mesh_device = mesh_device
        self.model_path = str(model_path)
        self.cache_dir = cache_dir

        # Load HF config + tokenizer
        self.hf_config = (
            hf_config if hf_config is not None else AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        )
        # self._ensure_max_seq_len(self.hf_config)
        self.hf_config.max_seq_len = 4096
        # Optional overrides for layer counts before building states
        if override_num_layers is not None:
            try:
                self.hf_config.num_hidden_layers = int(override_num_layers)
            except Exception as e:
                logger.warning(f"Failed to override num_hidden_layers with value '{override_num_layers}': {e}")
        if dense_layers is not None:
            try:
                self.hf_config.first_k_dense_replace = int(dense_layers)
            except Exception as e:
                logger.warning(f"Failed to override first_k_dense_replace with value '{dense_layers}': {e}")
        # Tokenizer is optional; caller can pass a tokenizer or handle failure.
        self.tokenizer = tokenizer

        # Runtime helpers
        self.ccl = CCL(mesh_device)
        mesh_shape = list(mesh_device.shape)
        self.dp_factor = mesh_shape[1]
        # Weight cache to avoid loading weights multiple times
        self._weight_ttnn_cache: dict[str, ttnn.Tensor] = {}
        # Paged attention setup
        self.batch_size_per_row = USERS_PER_ROW
        self.batch_size = self.batch_size_per_row * self.mesh_device.shape[0]
        self.paged_config = MLA2D.get_valid_paged_config(self.hf_config.max_seq_len, self.batch_size, self.dp_factor)

        self.random_weights = random_weights
        self.single_layer = single_layer

        # Trace state (decode)
        self._trace_id: int | None = None
        self._trace_tokens: ttnn.Tensor | None = None
        self._trace_positions: ttnn.Tensor | None = None
        self._trace_rot_idxs: ttnn.Tensor | None = None
        self._trace_output: ttnn.Tensor | None = None
        self.enable_trace = enable_trace
        logger.info(f"Enable trace: {self.enable_trace}")

        # Initialize rope_setup once
        self.rope_setup = RotarySetup(
            device=self.mesh_device, batch_size_per_row=self.batch_size_per_row, hf_config=self.hf_config
        )

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

        self.model_weight_config = get_weight_config(
            ModuleClass=RowBatchedModel,
            hf_config=self.hf_config,
            weight_cache_path=weight_cache_path,
            mesh_device=self.mesh_device,
            force_recalculate=False,
            random_weights=self.random_weights,
            model_path=self.model_path,
            single_layer=self.single_layer,
        )

    def _prepare_model_states(self, kv_cache_override: KvCacheConfig | None = None) -> None:
        logger.info("Creating model states...")
        self.model_state = RowBatchedModel.create_state(
            hf_config=self.hf_config,
            mesh_device=self.mesh_device,
            paged_config=self.paged_config,
            ccl=self.ccl,
            kv_cache_override=kv_cache_override,
        )
        logger.info("Creating model shared states...")
        self.model_shared_state = RowBatchedModel.create_shared_state(
            hf_config=self.hf_config, mesh_device=self.mesh_device
        )

    def _prepare_run_configs(self, mode: str, kv_cache_override: KvCacheConfig | None = None) -> None:
        if mode == "prefill":
            logger.info("Creating model prefill config...")
            self.model_prefill_cfg = RowBatchedModel.prefill_model_config(
                hf_config=self.hf_config, mesh_device=self.mesh_device
            )
            self._prepare_model_states(kv_cache_override=kv_cache_override)
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
            self.model_decode_cfg = RowBatchedModel.decode_model_config(
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
            if hasattr(self, "rope_setup") and self.rope_setup is not None:
                del self.rope_setup
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

        # Clean up trace state
        if self.enable_trace:
            try:
                if hasattr(self, "_trace_id") and self._trace_id is not None:
                    ttnn.release_trace(self.mesh_device, self._trace_id)
            except Exception as e:
                logger.warning(f"Failed to release trace: {e}")

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
        rope_setup = RotarySetup(
            device=self.mesh_device, batch_size_per_row=self.batch_size_per_row, hf_config=self.hf_config
        )
        rope_mats = rope_setup.get_rot_mats_table(seq_len=1)
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
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        return rope_tensors, tt_positions

    def _get_page_tables(self) -> tuple[ttnn.Tensor, ...]:
        if hasattr(self, "page_tables_tt") and self.page_tables_tt is not None:
            return self.page_tables_tt

        assert hasattr(self, "paged_config") and self.paged_config is not None
        assert hasattr(self, "mesh_device") and self.mesh_device is not None
        assert hasattr(self, "batch_size_per_row") and self.batch_size_per_row is not None
        assert hasattr(self, "hf_config") and self.hf_config is not None
        self.page_tables_tt = tuple(
            MLA2D.create_page_table(
                paged_config=self.paged_config,
                mesh_device=self.mesh_device,
                batch_size_per_row=int(self.batch_size_per_row / self.mesh_device.shape[0]),
            )
            for _ in range(self.hf_config.num_hidden_layers)
        )
        return self.page_tables_tt

    def _decode_step(
        self,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_table: torch.Tensor | None = None,
        return_rot_idxs: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, ttnn.Tensor]:
        """Run a single decode step and return logits on host as torch tensor [1, 1, B, V].

        Args:
            tokens_step: Input tokens
            positions: Position indices
            batch_size_per_row: Batch size per row

        Returns:
            logits tensor
        """
        # Prepare TT inputs
        tt_tokens = self._tt_from_tokens_step(tokens_step)

        # Get rot_idxs from positions (this uses ttnn.as_tensor, which is like from_torch)
        rot_idxs = self.rope_setup.get_rot_idxs(positions)

        # Generate rotation matrices from rot_idxs (all ttnn ops)
        rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(rot_idxs)

        # Create TTNN position tensor
        tt_positions = ttnn.from_torch(
            positions,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.int32,
        )

        if page_table is not None:
            page_tables_to_use = self._convert_vllm_page_table_for_batch(page_table)
        else:
            page_tables_to_use = self._get_page_tables()
        # RowBatchedModel forward
        logits_tt = RowBatchedModel.forward_decode(
            tt_tokens,
            tt_positions,
            self.model_run_config_decode,
            rope_tensors,
            page_tables=page_tables_to_use,
        )
        # Gather to host
        logits = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )
        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(logits_tt)

        return logits  # [1, 1, B, V]

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)  # [B]

    def _pad_batch(self, tokens_list: List[List[int]], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad/pack a list of token id sequences to batch of size batch_size.

        Returns
            tokens_packed: torch.LongTensor [batch_size, S]
            lengths: torch.IntTensor [batch_size] with actual sequence lengths for first N sequences, zeros otherwise
        """
        assert len(tokens_list) > 0 and len(tokens_list) <= batch_size
        max_len = max(len(t) for t in tokens_list)
        # Round up to nearest multiple of TILE_SIZE
        max_len = ((max_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

        pad_id = 0
        if self.tokenizer is not None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                # Some tokenizers don't define pad_token_id; fall back to EOS/config EOS.
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is None:
                    eos_id = getattr(self.hf_config, "eos_token_id", None)
                    if isinstance(eos_id, (list, tuple)):
                        eos_id = eos_id[0] if eos_id else None
                pad_id = eos_id if eos_id is not None else 0
            pad_id = int(pad_id)

        out = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        lengths = torch.zeros((batch_size,), dtype=torch.int32)
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
        pre_tokenized: List[List[int]] | None = None,
    ) -> Tuple[List[List[int]], dict]:
        """Generate tokens for the given prompts using greedy decode by default.

        early_print_first_user: If True, prints generated tokens for the first user
                                at each step. Better for demo visibility.

        repeat_batches: Number of times to repeat the prefill+decode pass. Only the
                        last pass's tokens are returned; timings aggregate.

        Returns: (list of generated token id lists for the provided prompts (order preserved), statistics dictionary)
        """
        # Initialize profiler
        profiler = BenchmarkProfiler()
        profiler.start("run")

        prompts = list(prompts)
        num_of_prompts = len(prompts)
        assert 1 <= num_of_prompts <= self.batch_size, f"Supports 1..{self.batch_size} prompts"

        logger.info("Creating model run configs...")
        profiler.start("preparing_prefill_config")
        self._prepare_run_configs("prefill")
        profiler.end("preparing_prefill_config")

        profiler.start("preparing_decode_config")
        self._prepare_run_configs("decode")
        profiler.end("preparing_decode_config")

        # Tokenize using HF chat template (or use pre-tokenized prompt ids for teacher-forcing / exact alignment)
        profiler.start("tokenizing")
        if pre_tokenized is not None:
            if len(pre_tokenized) != num_of_prompts:
                raise ValueError(
                    f"pre_tokenized length ({len(pre_tokenized)}) must match number of prompts ({num_of_prompts})"
                )
            encoded: List[List[int]] = [list(map(int, seq)) for seq in pre_tokenized]
        else:
            encoded = [self._encode_prompt(p) for p in prompts]
        tokens_batched, lengths = self._pad_batch(encoded, self.batch_size)  # [batch_size, seq_len]
        profiler.end("tokenizing")

        logger.info(f"Lengths of {lengths.shape} (encoded) prompts: {lengths}")

        # Run one or more prefill+decode batches
        for _ in range(repeat_batches):
            # Reset teacher-forcing state per batch.
            if teacher_forcing is not None:
                teacher_forcing.reset()

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
                prompt_len = int(lengths[user_id].item())
                logger.info(
                    "Input to the prefill: "
                    + (
                        self.tokenizer.decode(
                            tokens_batched[user_id][:prompt_len].tolist(),
                            skip_special_tokens=True,
                        )
                        if self.tokenizer is not None
                        else str(tokens_batched[user_id][:prompt_len].tolist())
                    )
                )
                user_out = self._prefill(tokens_batched[user_id], user_id=user_id)
                # Use logits at the *actual* last prompt token (not the padded tail).
                last_logits.append(user_out[0, 0, prompt_len - 1, :])
                self.ccl.reset_sem_counters()
            last_logits = torch.stack(last_logits)
            profiler.end("inference_prefill")

            assert len(last_logits) == num_of_users

            logger.info(f"Finished prefill for all users...")

            generations: List[List[int]] = [[] for _ in range(num_of_prompts)]
            if max_new_tokens <= 0:
                logger.info("max_new_tokens <= 0, skipping decode loop.")
            else:
                logger.info(f"Generating {max_new_tokens} tokens for {num_of_prompts} user(s)...")
                if early_print_first_user:
                    logger.info("===== Generation for first user =====")

                # First generated token comes from prefill's last-position logits
                next_tokens = self._sample_greedy(last_logits)
                if teacher_forcing is not None:
                    # Record user-0 prediction for accuracy, but force teacher token for alignment.
                    forced0 = teacher_forcing.collect_predicted_tokens(int(next_tokens[0].item()))
                    next_tokens[0] = int(forced0)

                # Positions for the first generated token are the prompt lengths
                positions = lengths.clone()

                # Record token 0
                for i in range(num_of_prompts):
                    token_value = int(next_tokens[i].item())
                    generations[i].append(token_value)
                    if early_print_first_user and i == 0:
                        if self.tokenizer is not None:
                            print(self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True)
                        else:
                            print(f"{token_value} ", end="", flush=True)

                # Generate remaining tokens with decode (each decode call produces the next token)
                decode_steps = max_new_tokens - 1
                profiler.start("inference_decode")
                for gen_idx in range(decode_steps):
                    logger.info(f"Decoding step {gen_idx} for {num_of_prompts} user(s)...")
                    profiler.start(f"decode_time_{gen_idx}")
                    logits = self.decode_forward(
                        next_tokens,
                        positions,
                        self.batch_size_per_row,
                        profiler,
                        gen_idx,
                        enable_trace=self.enable_trace,
                    )
                    profiler.end(f"decode_time_{gen_idx}")
                    self.ccl.reset_sem_counters()
                    pred_tokens = self._sample_greedy(logits)
                    if teacher_forcing is not None:
                        # Record user-0 prediction for accuracy, then force teacher token.
                        forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                        pred_tokens[0] = int(forced)
                    next_tokens = pred_tokens
                    positions += 1

                    for i in range(num_of_prompts):
                        token_value = int(next_tokens[i].item())
                        generations[i].append(token_value)
                        if early_print_first_user and i == 0:
                            if self.tokenizer is not None:
                                print(self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True)
                            else:
                                print(f"{token_value} ", end="", flush=True)

                profiler.end("inference_decode")

            if early_print_first_user:
                logger.info("\n===== Done =====")

        profiler.end("run")
        # Calculate statistics
        prefill_time = profiler.get_duration("inference_prefill")
        decode_steps = max(max_new_tokens - 1, 0)
        decode_times = [profiler.get_duration(f"decode_time_{i}") for i in range(decode_steps)]

        # Get config preparation times
        prefill_config_time = profiler.get_duration("preparing_prefill_config")
        decode_config_time = profiler.get_duration("preparing_decode_config")

        # Average prompt length for prefill calculation
        avg_prompt_len = float(lengths[0] if len(lengths) > 0 else 0)

        # Calculate statistics
        if prefill_time > 0:
            prefill_tokens_per_sec = (avg_prompt_len * num_of_prompts * repeat_batches) / prefill_time
        else:
            prefill_tokens_per_sec = 0

        # Calculate decode throughput excluding the first iteration (compile time).
        # We run (max_new_tokens - 1) decode calls (token 0 comes from prefill logits).
        if len(decode_times) > 1:
            total_decode_time = sum(decode_times[1:])  # Exclude iteration 0 (compile time)
            effective_decode_tokens = max(len(decode_times) - 1, 0)
            decode_tokens_per_sec_per_user = (
                (effective_decode_tokens * repeat_batches) / total_decode_time if total_decode_time > 0 else 0
            )
        elif len(decode_times) == 1:
            total_decode_time = decode_times[0]
            decode_tokens_per_sec_per_user = 0
        else:
            total_decode_time = 0
            decode_tokens_per_sec_per_user = 0
        decode_tokens_per_sec = decode_tokens_per_sec_per_user * num_of_prompts
        avg_time_to_first_token = prefill_time / (num_of_prompts * repeat_batches) if num_of_prompts > 0 else 0

        if self.enable_trace and profiler.contains_step("trace_execution_127"):
            trace_execution_time_for_128th_token = profiler.get_duration("trace_execution_127")
            trace_execution_tokens_per_sec_per_user_128th_token = (
                repeat_batches / trace_execution_time_for_128th_token if trace_execution_time_for_128th_token > 0 else 0
            )
        else:
            trace_execution_tokens_per_sec_per_user_128th_token = None

        statistics = {
            "preparing_prefill_config": prefill_config_time,
            "preparing_decode_config": decode_config_time,
            "inference_prefill": prefill_time,
            "inference_decode": total_decode_time,
            "prefill_time_to_token": avg_time_to_first_token,
            "prefill_t/s": prefill_tokens_per_sec,
            "decode_t/s/u": decode_tokens_per_sec_per_user,
            "trace_execution_t/s/u @128th token": trace_execution_tokens_per_sec_per_user_128th_token,
            "decode_t/s": decode_tokens_per_sec,
            "Full demo runtime": profiler.get_duration("run"),
        }

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

    def _prefill(
        self,
        tokens: torch.Tensor,
        user_id: int,
        page_table: torch.Tensor | None = None,
        local_user_id: int | None = None,
    ) -> torch.Tensor:
        """Run prefill for the full prompt sequence and return logits for the last position.

        Args:
            tokens: [1, 1, seq_len] padded token sequences
            user_id: user id for the prefill
            local_user_id: local user id for page table lookup

        Returns:
            logits: [1, 1, seq_len, V] logits for the full sequence
        """

        tokens = tokens.view(1, 1, -1)
        seq_len = tokens.shape[-1]

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

        if page_table is not None:
            page_tables_to_use = self._convert_vllm_page_table_for_user(page_table, user_id, local_user_id)
        else:
            page_tables_to_use = self._get_page_tables()

        # RowBatchedModel forward prefill
        logits_tt = RowBatchedModel.forward_prefill(
            x=tt_tokens,
            user_id=user_id,
            cfg=self.model_run_config_prefill,
            rope_tensors=rope_tensors,
            page_tables=page_tables_to_use,
        )

        # Gather to host
        logits = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )

        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(logits_tt)
        return logits  # [1, 1, seq_len, V]

    def _capture_decode_trace(
        self, init_tokens: torch.Tensor, positions: torch.Tensor, batch_size_per_row: int
    ) -> None:
        """Allocate persistent inputs, capture trace for one decode iteration, and store trace state."""
        assert self._trace_id is None, "Trace already captured"

        # 1) Warm-up compile run (no trace) to keep compilation out of capture
        logger.info("Running warm-up decode step (no trace)...")
        _ = self._decode_step(init_tokens, positions, batch_size_per_row=batch_size_per_row)
        ttnn.synchronize_device(self.mesh_device)

        # 2) Allocate persistent device inputs
        self._trace_tokens = self._tt_from_tokens_step(init_tokens)
        self._trace_positions = ttnn.from_torch(
            positions,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.int32,
        )

        self._trace_rot_idxs = self.rope_setup.get_rot_idxs(positions)
        ttnn.synchronize_device(self.mesh_device)

        # 3) Capture decode graph
        self.ccl.reset_sem_counters()
        logger.info("Begin capturing decode trace...")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Only capture the rot_mats generation from rot_idxs (all ttnn ops, no from_torch)
        rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(self._trace_rot_idxs)
        logger.info(f"Rope tensors done")

        # TODO: Fix this for vLLM
        self._trace_output = RowBatchedModel.forward_decode(
            x=self._trace_tokens,
            position_idxs=self._trace_positions,
            cfg=self.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_tables=self.page_tables_tt,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Decode trace capture complete.")
        self._trace_id = trace_id

    def decode_forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        profiler: BenchmarkProfiler,
        gen_idx: int,
        enable_trace: bool = False,
    ) -> torch.Tensor:
        if not enable_trace:
            return self._decode_step(tokens, positions, batch_size_per_row).squeeze(0).squeeze(0)
        else:
            # Capture trace and return trace output
            if self._trace_id is None:
                self._capture_decode_trace(tokens, positions, batch_size_per_row)
                # First call: return the captured run's output
                assert self._trace_output is not None
                logits = ttnn.to_torch(
                    self._trace_output,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                    ),
                )
                return logits.squeeze(0).squeeze(0)

            # Update persistent inputs and execute
            assert (
                self._trace_tokens is not None
                and self._trace_positions is not None
                and self._trace_rot_idxs is not None
                and self._trace_id is not None
            )
            torch_input = tokens.view(1, 1, -1).to(torch.int32)
            host_tokens = ttnn.from_torch(
                torch_input,
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            ttnn.copy_host_to_device_tensor(host_tokens, self._trace_tokens)

            host_positions = ttnn.from_torch(
                positions,
                device=None,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.int32,
            )

            ttnn.copy_host_to_device_tensor(host_positions, self._trace_positions)

            host_rot_idxs = self.rope_setup.get_rot_idxs(positions, on_host=True)
            ttnn.copy_host_to_device_tensor(host_rot_idxs, self._trace_rot_idxs)

            self.ccl.reset_sem_counters()
            profiler.start(f"trace_execution_{gen_idx}")
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)
            profiler.end(f"trace_execution_{gen_idx}")
            logger.info(
                f"Trace execution t/s/user @ {gen_idx}th token: {1/profiler.get_duration(f'trace_execution_{gen_idx}')}"
            )
            assert self._trace_output is not None
            logits = ttnn.to_torch(
                self._trace_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                ),
            )
            return logits.squeeze(0).squeeze(0)

    def warmup_model_prefill(self, kv_cache, enable_trace, sampling_params) -> None:
        logger.warning("Warmup model prefill not implemented for DeepseekGenerator")
        logger.warning("Tracing in prefill mode is not supported for DeepseekGenerator")

    def get_kv_cache(self):
        assert self.model_state is not None, "Model state is not initialized"

        kv_cache_list = []
        for decoder_type in ["mlp_decoder_block", "moe_decoder_block"]:
            if decoder_type in self.model_run_config_prefill:
                decoder_blocks = self.model_run_config_prefill[decoder_type]
                for block_cfg in decoder_blocks:
                    if "mla" in block_cfg and "mla1d" in block_cfg["mla"] and "kvpe_cache" in block_cfg["mla"]["mla1d"]:
                        kvpe_cache = block_cfg["mla"]["mla1d"]["kvpe_cache"]
                        kv_cache_list.append(kvpe_cache)
                    else:
                        raise ValueError(f"KVPE cache not found for decoder block {decoder_type}")

        return kv_cache_list

    def set_kv_cache(self, kv_cache_list: list[ttnn.Tensor]) -> None:
        """
        Set the kvpe_cache values in block configs from the provided kv_cache_list.
        This is the inverse operation of get_kv_cache().

        Args:
            kv_cache_list: List of TTNN tensors to set as kvpe_cache, one per decoder block
        """
        assert self.model_run_config_prefill is not None, "Model run config prefill is not initialized"
        assert len(kv_cache_list) > 0, "kv_cache_list cannot be empty"

        cache_idx = 0
        for decoder_type in ["mlp_decoder_block", "moe_decoder_block"]:
            if decoder_type in self.model_run_config_prefill:
                decoder_blocks = self.model_run_config_prefill[decoder_type]
                for block_cfg in decoder_blocks:
                    if "mla" in block_cfg and "mla1d" in block_cfg["mla"] and "kvpe_cache" in block_cfg["mla"]["mla1d"]:
                        if cache_idx >= len(kv_cache_list):
                            raise ValueError(
                                f"Not enough kv_cache entries. Expected at least {cache_idx + 1}, got {len(kv_cache_list)}"
                            )
                        block_cfg["mla"]["mla1d"]["kvpe_cache"] = kv_cache_list[cache_idx]
                        cache_idx += 1
                    else:
                        raise ValueError(f"MLA structure not found for decoder block {decoder_type}")

        if cache_idx < len(kv_cache_list):
            logger.warning(
                f"set_kv_cache: More kv_cache entries provided ({len(kv_cache_list)}) than decoder blocks ({cache_idx})"
            )

    def _convert_vllm_page_table_for_user(
        self, page_table: torch.Tensor, user_id: int, local_user_id: int | None = None
    ) -> tuple[ttnn.Tensor, ...]:
        """
        Convert vLLM's block_tables (page_table) to TTNN tensor format for a specific user.
        Creates one page table per layer as expected by the model.

        Args:
            page_table: torch.Tensor of shape [batch_size, max_num_blocks_per_req] from vLLM
            user_id: The user index to extract the page table for
            local_user_id: The local user index to extract the page table for

        Returns:
            Tuple of TTNN tensors, one per layer
        """
        # Calculate expected shape: [batch_per_shard, blocks_per_user]
        batch_per_shard = even_int_div(self.batch_size_per_row, self.dp_factor)
        blocks_per_user = even_int_div(self.paged_config.max_num_blocks, batch_per_shard)

        # Extract the user's block table row
        idx = local_user_id if local_user_id is not None else user_id
        user_blocks = page_table[
            idx, : min(blocks_per_user, page_table.shape[1])
        ].clone()  # [max_num_blocks_per_req] or less

        max_num_blocks = batch_per_shard * blocks_per_user
        full_page_table = torch.randperm(max_num_blocks, dtype=torch.int32)
        full_page_table = full_page_table.reshape(batch_per_shard, blocks_per_user)

        local_user_idx = user_id % batch_per_shard
        num_user_blocks = min(user_blocks.shape[0], blocks_per_user)
        full_page_table[local_user_idx, :num_user_blocks] = user_blocks[:num_user_blocks]

        # Convert to TTNN format using the model's helper
        page_table_tt = MLA2D.create_page_table(
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            page_table=full_page_table,
            batch_size_per_row=self.batch_size_per_row // self.mesh_device.shape[0],
        )

        num_layers = self.hf_config.num_hidden_layers
        return tuple(ttnn.clone(page_table_tt) for _ in range(num_layers))

    def _convert_vllm_page_table_for_batch(self, page_table: torch.Tensor) -> tuple[ttnn.Tensor, ...]:
        """
        Convert vLLM's block_tables (page_table) to TTNN tensor format for the entire batch.
        Creates one page table per layer as expected by the model.

        Args:
            page_table: torch.Tensor of shape [batch_size, max_num_blocks_per_req] from vLLM

        Returns:
            Tuple of TTNN tensors, one per layer
        """
        # Use vLLM page table directly, but shard it across devices to match the sharded batch size
        # in paged_update_cache.
        # page_table shape: [batch_size, max_blocks_per_req]

        page_table_tt = ttnn.from_torch(
            page_table,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return tuple(page_table_tt for _ in range(self.hf_config.num_hidden_layers))


__all__ = ["DeepseekGenerator", "SamplingParams"]
