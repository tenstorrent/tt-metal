# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.model_1d import Model1D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import MAX_BATCH_SIZE
from models.demos.deepseek_v3.utils.hf_model_utils import load_model_weights
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict


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
    Simple generator that wires Model1D + LMHead for decode-only inference.

    Notes:
    - Prefill at the model level is not fully implemented in Model1D; we emulate
      prefill by iterating decode steps over the prompt tokens (updates caches).
    - Batch size in configs is tied to MAX_BATCH_SIZE; for simplicity we decode
      up to that many sequences. If fewer are provided, we pad/ignore extras.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        model_path: str | Path,
        cache_dir: str | Path | None = None,
        batch_size: int = MAX_BATCH_SIZE,
        tokenizer=None,
        random_weights: bool = False,
        dense_layers: int | None = None,
        override_num_layers: int | None = None,
        single_layer: str | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        # self.mesh_device.disable_and_clear_program_cache()
        self.model_path = str(model_path)
        self.batch_size = min(MAX_BATCH_SIZE, batch_size)

        # Load HF config + tokenizer
        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._ensure_max_seq_len(self.hf_config)

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
        self.ccl = CCL1D(mesh_device)
        mesh_shape = list(mesh_device.shape)
        self.dp_factor = mesh_shape[1]

        self.hf_config.num_hidden_layers = 5
        # logger.info(f"hf_config: {self.hf_config}")
        # Paged attention setup
        self.paged_config = MLA1D.get_valid_paged_config(self.hf_config.max_seq_len, MAX_BATCH_SIZE, self.dp_factor)
        torch_page_tables = [
            MLA1D.create_rand_page_table(
                MAX_BATCH_SIZE,
                dp_factor=self.dp_factor,
                config=self.paged_config,
                mesh_device=mesh_device,
            )
            for _ in range(self.hf_config.num_hidden_layers)
        ]
        self.page_tables_tt = tuple(
            MLA1D.create_page_table(torch_page_table, self.paged_config, mesh_device)
            for torch_page_table in torch_page_tables
        )

        self.rope = RotarySetup(device=mesh_device, batch_size=MAX_BATCH_SIZE, hf_config=self.hf_config)

        # Prepare weights/configs
        self.random_weights = random_weights
        self.single_layer = single_layer
        self._prepare_run_configs(cache_dir)

        # Trace state (decode)
        self._trace_id: int | None = None
        self._trace_tokens: ttnn.Tensor | None = None
        self._trace_positions: ttnn.Tensor | None = None
        self._trace_output: ttnn.Tensor | None = None

    @staticmethod
    def _ensure_max_seq_len(hf_config) -> None:
        # if getattr(hf_config, "max_seq_len", None) is not None:
        #     return
        # try:
        #     if getattr(hf_config, "rope_scaling", None):
        #         factor = hf_config.rope_scaling.get("factor")
        #         orig = hf_config.rope_scaling.get("original_max_position_embeddings")
        #         if factor and orig:
        #             hf_config.max_seq_len = int(factor * orig)
        #             return
        #     if getattr(hf_config, "max_position_embeddings", None):
        #         hf_config.max_seq_len = int(hf_config.max_position_embeddings)
        #         return
        # except Exception:
        #     pass
        hf_config.max_seq_len = 4096

    def _prepare_run_configs(self, cache_dir: str | Path | None) -> None:
        cache_root = Path(cache_dir) if cache_dir is not None else Path("generated/deepseek_v3")
        weights_out = cache_root / "weights"
        weights_out.mkdir(parents=True, exist_ok=True)

        if self.random_weights:
            if self.single_layer and self.single_layer.lower() == "moe":
                raise NotImplementedError(
                    "Random weights with 'moe' single layer is not supported by Model1D demo yet. Use 'mlp' or disable random mode."
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
            stripped = _strip_model_prefix(torch_state)
            model1d_state = {
                k: v
                for k, v in stripped.items()
                if k.startswith("embed_tokens.")
                or k.startswith("layers.")
                or k.startswith("norm.")
                or k.startswith("lm_head.")
            }
        else:
            # logger.info(f" self.hf_config = {self.hf_config}")
            logger.info("Loading HF weights (this may take a while)...")
            hf_weights = load_model_weights(self.model_path)
            logger.info("HF weights loaded")

            if "lm_head.weight" not in hf_weights:
                raise RuntimeError(
                    "No HF safetensors found in model path or missing 'lm_head.weight'. "
                    "Set DEEPSEEK_V3_HF_MODEL to a directory containing DeepSeek-V3 safetensors, or pass --model-path."
                )
            # logger.info("Dequantizing HF weights...")
            # hf_weights = dict(dequantize_state_dict(hf_weights, self.hf_config))
            # logger.info("HF weights dequantized.")
            model1d_state = {
                k: v
                for k, v in hf_weights.items()
                if k.startswith("model.embed_tokens.")
                or k.startswith("model.layers.")
                or k.startswith("model.norm.")
                or k.startswith("lm_head.")
            }

        # Convert weights to TT tensors-on-disk and build weight_config
        deepseek_cache_path = Path(os.getenv("DEEPSEEK_V3_CACHE", "/proj_sw/user_dev/deepseek-v3-cache"))
        weights_type = "random" if self.random_weights else "hf_real"
        cache_dir = deepseek_cache_path / f"model_{self.hf_config.num_hidden_layers}_layers" / weights_type
        tensor_cache_path = cache_dir / "ttnn_tensors_cache"
        weight_config_path = cache_dir / "weight_config.json"
        # save this weight config to json file if it doesn't exist
        if not weight_config_path.exists():
            logger.info(f"Weight config not found at {weight_config_path}, creating new one.")
            self.model1d_weight_config = Model1D.convert_weights(
                self.hf_config, [model1d_state], tensor_cache_path, self.mesh_device
            )
            with open(weight_config_path, "w") as f:
                json.dump(self.model1d_weight_config, f)
            logger.info(f"Saved weight config to {weight_config_path}")
        else:
            logger.info(f"Weight config found at {weight_config_path}, loading existing one.")
            with open(weight_config_path, "r") as f:
                self.model1d_weight_config = json.load(f)
            logger.info(f"Loaded weight config from {weight_config_path}")

        model_decode_cfg = Model1D.decode_model_config(self.hf_config, self.mesh_device)
        logger.info(f"Model decode config done")
        model_state = Model1D.create_state(
            hf_config=self.hf_config,
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            ccl=self.ccl,
            mla_caches=None,
        )
        logger.info(f"create_state done")
        model_shared_state = Model1D.create_shared_state(hf_config=self.hf_config, mesh_device=self.mesh_device)
        logger.info(f"create_shared_state done")
        self.model1d_run_config = create_run_config(
            model_decode_cfg, self.model1d_weight_config, model_state, model_shared_state
        )
        logger.info(f"Run config done")

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
        rope_tensors = {"cos_matrix": rope_mats[0], "sin_matrix": rope_mats[1], "trans_matrix": rope_mats[2]}

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

        # Model forward
        logits_tt = Model1D.forward_decode(
            tt_tokens,
            tt_positions,
            self.model1d_run_config,
            rope_tensors,
            self.page_tables_tt,
        )
        # Gather to host
        logits = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))

        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(logits_tt)

        return logits  # [1, 1, B, V]

    def _read_logits_host(self, tt_logits: ttnn.Tensor) -> torch.Tensor:
        """Convert device logits [1, 1, B, V] to torch [B, V]."""
        pt_logits = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
        return pt_logits.squeeze(0).squeeze(0)

    def _capture_decode_trace(self, init_tokens: torch.Tensor, positions: torch.Tensor) -> None:
        """Allocate persistent inputs, capture trace for one decode iteration, and store trace state."""
        assert self._trace_id is None, "Trace already captured"

        # 1) Warm-up compile run (no trace) to keep compilation out of capture
        logger.info("Running warm-up decode step (no trace)...")
        _ = self._decode_step(init_tokens, positions)

        # 2) Allocate persistent device inputs
        # Tokens buffer shape [1, 1, B]
        torch_input = init_tokens.view(1, 1, -1).to(torch.int32)
        self._trace_tokens = ttnn.from_torch(
            torch_input,
            device=self.mesh_device,  # set None?
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Positions buffer shape [B]
        self._trace_positions = ttnn.from_torch(
            positions.to(torch.int32),
            device=self.mesh_device,  # set None?
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=self.mesh_device.shape),
            dtype=ttnn.int32,
        )

        # 3) Capture decode graph
        # rope_tensors = self.rope.get_rot_mats(self._trace_positions)
        rope_tensors = self.rope.get_rot_mats(positions.to(torch.int32))
        logger.info("Begin capturing decode trace...")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logger.info("Begin capturing decode trace...returned from ttnn api")
        # breakpoint()
        logger.info("calling model1d forward decode")
        self._trace_output = Model1D.forward_decode(
            x=self._trace_tokens,
            position_idxs=self._trace_positions,
            rope_tensors={
                "cos_matrix": rope_tensors[0],
                "sin_matrix": rope_tensors[1],
                "trans_matrix": rope_tensors[2],
            },
            page_tables=self.page_tables_tt,
            cfg=self.model1d_run_config,
        )
        logger.info("calling end_trace_capture")
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Decode trace capture complete.")
        self._trace_id = trace_id

    # Align with tt_transformers: provide decode_forward_text and compatible helpers
    def read_decode_output(self, tt_out: ttnn.Tensor, async_read: bool = False):
        if not async_read:
            return self._read_logits_host(tt_out)
        # Async path: best-effort stub (returns host tensor, event None)
        return self._read_logits_host(tt_out), None

    def process_decode_output_host(self, to_host: torch.Tensor, is_tokens: bool = False):
        # Our host tensors are already [B, V] logits; return as-is
        return to_host

    def decode_forward_text(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params: SamplingParams | None = None,
    ):
        # Delegate to our decode_forward; ignore page_table/kv_cache which are not used in Model1D path
        logits_or_tt = self.decode_forward(
            tokens, start_pos, enable_trace=enable_trace, read_from_device=read_from_device
        )
        if read_from_device:
            # If requested tokens on device (temperature==0), return tokens instead of logits
            if sampling_params is not None and getattr(sampling_params, "temperature", 0) == 0:
                return torch.argmax(logits_or_tt, dim=-1)
            return logits_or_tt
        return logits_or_tt

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens: torch.Tensor | None = None,
        empty_slots=None,
        **kwargs,
    ) -> torch.Tensor:
        """Emulate prefill by iterating decode steps over the prompt tokens.

        tokens: [B, S]
        prompt_lens: optional [B] lengths; if None, uses full S
        Returns logits for the last token: [B, V]
        """
        B, S = tokens.shape
        positions = torch.zeros(B, dtype=torch.int32)
        last_logits = None
        if prompt_lens is None:
            prompt_lens = torch.full((B,), S, dtype=torch.int32)

        max_len = int(prompt_lens.max().item())
        for step in range(max_len):
            # Use token at 'step' for all users; if step exceeds S-1, clamp to last column
            step_tokens = tokens[:, min(step, S - 1)]
            last_logits = self.decode_forward(step_tokens, positions, enable_trace=False, read_from_device=True)
            positions += 1
        assert last_logits is not None
        return last_logits

    def decode_forward(
        self,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        *,
        enable_trace: bool = True,
        read_from_device: bool = True,
    ) -> torch.Tensor | ttnn.Tensor:
        """Run a single decode iteration.

        - If enable_trace is True, lazily capture a trace on first call and then execute the trace.
        - If enable_trace is False, run without trace (allocates per call).

        Inputs:
          tokens_step: torch int tensor [B]
          positions: torch int tensor [B]
        Returns logits as torch [B, V] if read_from_device else device tensor handle.
        """
        if not enable_trace:
            logits_torch = self._decode_step(tokens_step, positions)
            # return self._read_logits_host(logits_tt_or) if read_from_device else logits_tt_or
            return logits_torch

        # Trace path
        if self._trace_id is None:
            logger.info(f"Capturing decode trace...")
            self._capture_decode_trace(tokens_step, positions)
            logger.info(f"Capturing decode trace...done")
            # First call: return the captured run's output
            assert self._trace_output is not None
            return self._read_logits_host(self._trace_output) if read_from_device else self._trace_output

        # Update persistent inputs and execute
        assert self._trace_tokens is not None and self._trace_positions is not None and self._trace_id is not None
        # Update tokens [1, 1, B]
        torch_input = tokens_step.view(1, 1, -1).to(torch.int32)
        ttnn.copy_host_to_device_tensor(torch_input, self._trace_tokens)
        # Update positions [B]
        ttnn.copy_host_to_device_tensor(positions.to(torch.int32), self._trace_positions)

        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        assert self._trace_output is not None
        return self._read_logits_host(self._trace_output) if read_from_device else self._trace_output

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [1, 1, B, V]
        return torch.argmax(logits[0, 0], dim=-1)  # [B]

    def _pad_batch(self, tokens_list: List[List[int]]) -> Tuple[torch.Tensor, List[int]]:
        """Pad/pack a list of token id sequences to batch of size MAX_BATCH_SIZE.

        Returns
            tokens_packed: torch.LongTensor [MAX_BATCH_SIZE, S]
            valid_counts: list of actual sequence lengths for first N sequences
        """
        assert len(tokens_list) > 0 and len(tokens_list) <= MAX_BATCH_SIZE
        max_len = max(len(t) for t in tokens_list)
        B = MAX_BATCH_SIZE
        out = torch.full((B, max_len), 0, dtype=torch.long)
        valid = []
        for i, seq in enumerate(tokens_list):
            out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            valid.append(len(seq))
        return out, valid

    def generate(
        self,
        prompts: Iterable[str],
        max_new_tokens: int = 32,
        sampling: SamplingParams | None = None,
        teacher_forcing=None,
        enable_trace: bool = False,
        profiler=None,
    ) -> List[List[int]]:
        """Generate tokens for the given prompts using greedy decode by default.

        Returns: list of generated token id lists for the provided prompts (order preserved).
        """
        prompts = list(prompts)
        assert 1 <= len(prompts) <= MAX_BATCH_SIZE, f"Supports 1..{MAX_BATCH_SIZE} prompts"

        # Tokenize using HF chat template
        encoded: List[List[int]] = [self._encode_prompt(p) for p in prompts]
        tokens_batched, lengths = self._pad_batch(encoded)  # [MAX_BATCH_SIZE, S]

        # Prefill via repeated decode steps over prompt tokens (do not trace prefill)
        if profiler is not None:
            # Placeholders to keep summary uniform with traced path
            try:
                profiler.start("compile_prefill")
                profiler.end("compile_prefill")
            except Exception:
                pass
            try:
                profiler.start("inference_prefill")
            except Exception:
                pass
        B = MAX_BATCH_SIZE
        positions = torch.zeros(B, dtype=torch.int32)
        last_logits = None
        for step in range(tokens_batched.shape[1]):
            step_tokens = tokens_batched[:, step]  # [B]
            # Always avoid tracing for the prefill emulation
            last_logits = self.decode_forward(step_tokens, positions, enable_trace=False, read_from_device=True)
            positions += 1

        if profiler is not None:
            try:
                profiler.end("inference_prefill")
            except Exception:
                pass

        assert last_logits is not None
        # First sampled token after prompt
        next_tokens = self._sample_greedy(last_logits)
        # If teacher forcing is enabled, collect the model's predicted token and force GT for next step (single prompt)
        if teacher_forcing is not None:
            # Only enforce for the first user to keep scope minimal
            forced = teacher_forcing.collect_predicted_tokens(int(next_tokens[0].item()))
            next_tokens[0] = int(forced)

        generations: List[List[int]] = [[] for _ in range(len(prompts))]
        logger.info("Generating: ")
        if profiler is not None:
            try:
                profiler.start("inference_decode")
            except Exception:
                pass
        for gen_idx in range(max_new_tokens):
            # Decode one step with previous next_tokens
            if profiler is not None:
                try:
                    if enable_trace and gen_idx == 0:
                        profiler.start("compile_decode")
                    else:
                        profiler.start(f"inference_decode_time_{gen_idx + 1}")
                except Exception:
                    pass
            logits = self.decode_forward(next_tokens, positions, enable_trace=enable_trace, read_from_device=True)
            if profiler is not None:
                try:
                    if enable_trace and gen_idx == 0:
                        profiler.end("compile_decode")
                    else:
                        profiler.end(f"inference_decode_time_{gen_idx + 1}")
                except Exception:
                    pass
            pred_tokens = self._sample_greedy(logits)
            if teacher_forcing is not None:
                forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                pred_tokens[0] = int(forced)
            next_tokens = pred_tokens
            positions += 1

            # Collect only for the original batch size
            for i in range(len(prompts)):
                generations[i].append(int(next_tokens[i].item()))
                logger.info(
                    f"new token generated = {self.tokenizer.decode(int(next_tokens[i].item()), skip_special_tokens=True)}"
                )
        print("\n", flush=True)
        if profiler is not None:
            try:
                profiler.end("inference_decode")
            except Exception:
                pass
        logger.info("Done generating tokens")

        return generations

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
