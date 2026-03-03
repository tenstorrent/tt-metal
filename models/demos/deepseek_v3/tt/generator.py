# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from loguru import logger
from tracy import signpost
from transformers import AutoConfig

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.warmup import WarmupForwardMixin
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div
from models.demos.deepseek_v3.utils.debug_utils import dump_ttnn_meminfo
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.weight_config import get_weight_config
from models.perf.benchmarking_utils import BenchmarkProfiler

MAX_SEQ_LEN = 2048


@dataclass(frozen=True)
class SamplingModuleArgs:
    vocab_size: int
    padded_vocab_size: int
    max_top_k: int
    max_batch_size: int
    sampling_dp: int
    cluster_shape: tuple[int, int]
    sampling_all_gather_axis: int = 0
    sub_core_grids: ttnn.CoreRangeSet | None = None
    sub_core_grid_topk: ttnn.CoreRangeSet | None = None
    start_core: ttnn.CoreCoord = field(default_factory=lambda: ttnn.CoreCoord(0, 0))
    model_config: dict = field(default_factory=dict)


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


class DeepseekGenerator(WarmupForwardMixin):
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
        max_seq_len: int | None = None,
        enable_trace: bool = False,
        enable_mem_profile: bool = False,
        signpost: bool = False,
        prefill_max_tokens: int | None = None,
        force_recalculate: bool = False,
        profile_decode: bool = False,
        sample_on_device: bool = True,
        dump_host_logits: bool = False,
        dump_host_logits_dir: str | Path | None = None,
        dump_sampling_compare: bool = False,
    ) -> None:
        self.mesh_device = mesh_device
        self.model_path = str(model_path)
        self.cache_dir = cache_dir
        self.sample_on_device = sample_on_device
        self.dump_host_logits = dump_host_logits
        self.dump_sampling_compare = dump_sampling_compare
        self._logits_dump_dir = Path(dump_host_logits_dir) if dump_host_logits_dir is not None else None
        self._rank = os.getenv("OMPI_COMM_WORLD_RANK") or os.getenv("PMI_RANK") or os.getenv("RANK") or "0"

        # Load HF config + tokenizer
        self.hf_config = (
            hf_config if hf_config is not None else AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        )
        # Hard-code the context length to keep KV cache + RoPE tables bounded.
        # (Avoid env var overrides; long-context runs should change this constant in code.)
        if max_seq_len is not None and int(max_seq_len) != MAX_SEQ_LEN:
            logger.warning(f"Ignoring requested max_seq_len={max_seq_len}; using MAX_SEQ_LEN={MAX_SEQ_LEN}.")
        self.hf_config.max_seq_len = MAX_SEQ_LEN
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
        # Ensure first_k_dense_replace doesn't exceed num_hidden_layers
        if hasattr(self.hf_config, "first_k_dense_replace") and hasattr(self.hf_config, "num_hidden_layers"):
            if self.hf_config.first_k_dense_replace > self.hf_config.num_hidden_layers:
                logger.warning(
                    f"Clamping first_k_dense_replace from {self.hf_config.first_k_dense_replace} "
                    f"to {self.hf_config.num_hidden_layers} (num_hidden_layers)"
                )
                self.hf_config.first_k_dense_replace = self.hf_config.num_hidden_layers
        # Tokenizer is optional; caller can pass a tokenizer or handle failure.
        self.tokenizer = tokenizer
        self.hf_config.num_hidden_layers = 5
        logger.info(f"num_hidden_layers: {self.hf_config.num_hidden_layers}")

        # Runtime helpers
        self.ccl = CCL(mesh_device)
        mesh_shape = list(mesh_device.shape)
        self.dp_factor = mesh_shape[1]
        self.batch_size_per_row = USERS_PER_ROW
        self.batch_size = self.batch_size_per_row * self.mesh_device.shape[0]

        self.sampling_args = SamplingModuleArgs(
            vocab_size=self.hf_config.vocab_size,
            padded_vocab_size=self.hf_config.vocab_size,  # Pratik: Check! Need to pad?
            max_top_k=32,
            max_batch_size=USERS_PER_ROW,
            sampling_dp=mesh_shape[0],
            cluster_shape=tuple(mesh_shape),
            sampling_all_gather_axis=1,  # Pratik: Check!
        )

        self.sampling_generator = SamplingGenerator(
            args=self.sampling_args, mesh_device=self.mesh_device, tt_ccl=self.ccl, enable_internal_trace=False
        )

        self.sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        self._reset_sampling_state(self.sampling_params)

        # Weight cache to avoid loading weights multiple times
        self._weight_ttnn_cache: dict[str, ttnn.Tensor] = {}
        # Paged attention setup

        self.paged_config = MLA2D.get_valid_paged_config(
            self.hf_config.max_seq_len, self.batch_size_per_row, self.dp_factor
        )

        self.random_weights = random_weights
        self.single_layer = single_layer

        # Log sampling mode
        logger.info(f"Sampling mode: {'device' if self.sample_on_device else 'host'}")
        if self.dump_host_logits:
            if self._logits_dump_dir is None:
                self._logits_dump_dir = Path(self.cache_dir) / "debug_host_logits"
            self._logits_dump_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Host logits dump enabled: {self._logits_dump_dir} (rank={self._rank})")
        if self.dump_sampling_compare:
            if self._logits_dump_dir is None:
                self._logits_dump_dir = Path(self.cache_dir) / "debug_host_logits"
                self._logits_dump_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Sampling compare dump enabled: {self._logits_dump_dir} (rank={self._rank})")

        # Model runtime state
        self.model_state = None
        self.model_shared_state = None
        self.model_prefill_cfg = None
        self.model_decode_cfg = None
        self.model_weight_config = None
        self.page_tables_tt = None

        # Trace state (decode)
        self._trace_id: int | None = None
        self._trace_tokens: ttnn.Tensor | None = None
        self._trace_positions: ttnn.Tensor | None = None
        self._trace_rot_idxs: ttnn.Tensor | None = None
        self._trace_logits: ttnn.Tensor | None = None
        self._trace_output: ttnn.Tensor | None = None
        self._trace_page_tables_to_use: tuple[ttnn.Tensor, ...] | None = None
        self.enable_trace = enable_trace
        self.enable_mem_profile = enable_mem_profile
        self.signpost = signpost
        self.prefill_max_tokens = prefill_max_tokens
        self.force_recalculate = force_recalculate
        self.profile_decode = profile_decode  # Profile decode: skip prefill, run only 1st dense + 1st MoE layer
        logger.info(f"Enable trace: {self.enable_trace}")
        if self.enable_trace and not self.sample_on_device:
            raise ValueError("Trace mode requires device sampling. Set sample_on_device=True or disable trace.")
        if self.profile_decode:
            logger.info("profile_decode=True: Prefill skipped, decode runs only 1st dense layer + 1st MoE layer")

        # Initialize rope_setup once
        self.rope_setup = RotarySetup(
            device=self.mesh_device, batch_size_per_row=self.batch_size_per_row, hf_config=self.hf_config
        )

        self._prepare_weight_configs(cache_dir)

    def _dump_meminfo(self, header: str) -> None:
        if self.enable_mem_profile:
            dump_ttnn_meminfo(self.mesh_device, header=header)

    @staticmethod
    def _ensure_max_seq_len(hf_config) -> None:
        if getattr(hf_config, "max_seq_len", None) is not None:
            return
        try:
            max_pos = getattr(hf_config, "max_position_embeddings", None)
            scaled = None
            if getattr(hf_config, "rope_scaling", None):
                factor = hf_config.rope_scaling.get("factor")
                orig = hf_config.rope_scaling.get("original_max_position_embeddings")
                if factor and orig:
                    scaled = int(factor * orig)
            if max_pos is not None and scaled is not None:
                # Prefer the larger of the declared max_position_embeddings and the rope-scaled length.
                hf_config.max_seq_len = int(max(max_pos, scaled))
                return
            if scaled is not None:
                hf_config.max_seq_len = int(scaled)
                return
            if max_pos is not None:
                hf_config.max_seq_len = int(max_pos)
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
            force_recalculate=self.force_recalculate,
            random_weights=self.random_weights,
            model_path=self.model_path,
            single_layer=self.single_layer,
        )

    def _prepare_model_states(self, kv_cache_override: KvCacheConfig | None = None) -> None:
        logger.info("Creating model states...")
        self._dump_meminfo("Before creating model states...")
        self.model_state = RowBatchedModel.create_state(
            hf_config=self.hf_config,
            mesh_device=self.mesh_device,
            paged_config=self.paged_config,
            ccl=self.ccl,
            kv_cache_override=kv_cache_override,
        )
        self._dump_meminfo("After creating model states...")
        logger.info("Creating model shared states...")
        self._dump_meminfo("Before creating model shared states...")
        self.model_shared_state = RowBatchedModel.create_shared_state(
            hf_config=self.hf_config, mesh_device=self.mesh_device
        )
        self._dump_meminfo("After creating model shared states...")

    def _prepare_run_configs(self, mode: str, kv_cache_override: KvCacheConfig | None = None) -> None:
        if mode == "prefill":
            logger.info("Creating model prefill config...")
            self._dump_meminfo("Before creating model prefill config...")
            self.model_prefill_cfg = RowBatchedModel.prefill_model_config(
                hf_config=self.hf_config, mesh_device=self.mesh_device
            )
            self._dump_meminfo("After creating model prefill config...")
            self._prepare_model_states(kv_cache_override=kv_cache_override)
            self._dump_meminfo("Before creating model run config for prefill...")
            self.model_run_config_prefill = create_run_config(
                self.model_prefill_cfg,
                self.model_weight_config,
                self.model_state,
                self.model_shared_state,
                cached_ttnn_weights=self._weight_ttnn_cache,
            )
            self._dump_meminfo("After creating model run config for prefill...")
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
            self._dump_meminfo("Before creating model run config for decode...")
            self.model_run_config_decode = create_run_config(
                self.model_decode_cfg,
                self.model_weight_config,
                self.model_state,
                self.model_shared_state,
                cached_ttnn_weights=self._weight_ttnn_cache,
            )
            self._dump_meminfo("After creating model run config for decode...")
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
            if self.model_state is not None:
                del self.model_state
        except Exception as e:
            logger.warning(f"Failed to cleanup model state: {e}")

        try:
            if self.model_shared_state is not None:
                del self.model_shared_state
        except Exception as e:
            logger.warning(f"Failed to cleanup model shared state: {e}")

        # Clean up trace state
        try:
            if self._trace_id is not None:
                ttnn.release_trace(self.mesh_device, self._trace_id)
                del self._trace_id
            if self._trace_tokens is not None:
                ttnn.deallocate(self._trace_tokens)
                del self._trace_tokens
            if self._trace_positions is not None:
                ttnn.deallocate(self._trace_positions)
                del self._trace_positions
            if self._trace_rot_idxs is not None:
                ttnn.deallocate(self._trace_rot_idxs)
                del self._trace_rot_idxs
            if self._trace_output is not None:
                ttnn.deallocate(self._trace_output)
                del self._trace_output
            if self._trace_page_tables_to_use is not None and self._trace_page_tables_to_use is not self.page_tables_tt:
                for i, page_table in enumerate(self._trace_page_tables_to_use):
                    try:
                        ttnn.deallocate(page_table)
                    except Exception as e:
                        logger.warning(f"Failed to deallocate trace page table {i}: {e}")
                del self._trace_page_tables_to_use
        except Exception as e:
            logger.warning(f"Failed to cleanup trace state: {e}")

        # Clean up page tables (TTNN tensors)
        try:
            if self.page_tables_tt is not None:
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
            if self.rope_setup is not None:
                del self.rope_setup
        except Exception as e:
            logger.warning(f"Failed to cleanup RoPE setup: {e}")

        # Clean up CCL
        try:
            if self.ccl is not None:
                del self.ccl
        except Exception as e:
            logger.warning(f"Failed to cleanup CCL: {e}")

        # Clean up configs
        try:
            if self.model_prefill_cfg is not None:
                del self.model_prefill_cfg
            if self.model_decode_cfg is not None:
                del self.model_decode_cfg
            if self.model_weight_config is not None:
                del self.model_weight_config

        except Exception as e:
            logger.warning(f"Failed to cleanup model configs: {e}")

        # Clean up paged config
        try:
            if self.paged_config is not None:
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
        assert tokens_step.dim() == 1, f"Expected 1D tensor, got shape {tokens_step.shape}"
        x = tokens_step.view(1, 1, -1).to(torch.int32)
        return ttnn.from_torch(
            x,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _reset_sampling_state(self, sampling_params: SamplingParams) -> None:
        sampling_params = format_sampling_params(sampling_params, self.batch_size_per_row)
        sampling_dp = int(self.sampling_args.sampling_dp)
        if sampling_dp > 1:
            total_param_size = self.batch_size_per_row * sampling_dp
            expanded_fields = {}
            for param_field in fields(sampling_params):
                value = getattr(sampling_params, param_field.name)
                if value is None:
                    expanded_fields[param_field.name] = None
                elif isinstance(value, list):
                    if len(value) == total_param_size:
                        expanded_fields[param_field.name] = value
                    elif len(value) == self.batch_size_per_row:
                        expanded_fields[param_field.name] = value * sampling_dp
                    elif len(value) == 1:
                        expanded_fields[param_field.name] = value * total_param_size
                    else:
                        raise ValueError(
                            f"Sampling param '{param_field.name}' has length {len(value)}; "
                            f"expected 1, {self.batch_size_per_row}, or {total_param_size}."
                        )
                else:
                    expanded_fields[param_field.name] = [value] * total_param_size
            sampling_params = replace(sampling_params, **expanded_fields)
        self.sampling_generator.reset_sampling_params(sampling_params)
        self.sampling_generator.seed_manager.get_new_values()

    def _sample_tokens_device(
        self, logits: ttnn.Tensor, tt_out_tok: ttnn.Tensor | None = None, enable_trace: bool = False
    ) -> ttnn.Tensor:
        logger.info(f"logits.shape before sample: {logits.shape}")
        breakpoint()
        tt_out = self.sampling_generator.sample(logits, enable_trace=enable_trace, tt_out_tok=tt_out_tok)
        if isinstance(tt_out, tuple):
            tt_tokens, tt_log_probs = tt_out
            logger.info(f"tuple tt_tokens.shape after sample: {tt_tokens.shape}")
            logger.info(f"tuple tt_log_probs.shape after sample: {tt_log_probs.shape}")
            if tt_log_probs is not None:
                ttnn.deallocate(tt_log_probs)
            tt_out = tt_tokens
        logger.info(f"tt_out.shape after sample return: {tt_out.shape}")
        return tt_out

    def _tokens_from_device(self, tt_tokens: ttnn.Tensor, batch_size: int) -> torch.Tensor:
        mesh_shape = tuple(self.mesh_device.shape)
        composed = ttnn.to_torch(
            tt_tokens,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, -1), mesh_shape=mesh_shape),
        )
        if composed.ndim == 4:
            if tt_tokens.shape[-2] == self.batch_size_per_row:
                tokens = composed[:, :, :, 0]
            elif tt_tokens.shape[-1] == self.batch_size_per_row:
                tokens = composed[:, :, 0, : self.batch_size_per_row]
            else:
                tokens = composed
            tokens = tokens.reshape(-1)
        else:
            tokens = composed.reshape(-1)
        return tokens[:batch_size]

    def _normalize_sampled_tokens(self, tt_tokens: ttnn.Tensor) -> ttnn.Tensor:
        if len(tt_tokens.shape) == 4 and tt_tokens.shape[-1] == 1:
            return ttnn.reshape(tt_tokens, [1, 1, tt_tokens.shape[-2]])
        if len(tt_tokens.shape) == 4 and tt_tokens.shape[-2] == 1:
            return ttnn.reshape(tt_tokens, [1, 1, tt_tokens.shape[-1]])
        return tt_tokens

    def _reset_decode_trace_state(self) -> None:
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._trace_id)
        for tensor in (self._trace_tokens, self._trace_positions, self._trace_rot_idxs, self._trace_logits):
            if tensor is not None:
                ttnn.deallocate(tensor)
        self._trace_id = None
        self._trace_tokens = None
        self._trace_positions = None
        self._trace_rot_idxs = None
        self._trace_logits = None
        self._trace_output = None

    def _reset_trace_inputs(self, tokens: torch.Tensor, positions: torch.Tensor) -> None:
        assert self._trace_tokens is not None and self._trace_positions is not None and self._trace_rot_idxs is not None
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

    def _increment_decode_positions_device(self) -> None:
        ttnn.plus_one(self._trace_positions, skip_negative_entries=True)
        ttnn.plus_one(self._trace_rot_idxs)

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

    def _get_page_tables(self) -> ttnn.Tensor | Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return page tables as TTNN tensor or tuple of TTNN tensors [1, 1, B, N]."""
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
                batch_size=self.batch_size_per_row,
            )
            for _ in range(self.hf_config.num_hidden_layers)
        )
        return self.page_tables_tt

    def _decode_step(
        self,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_tables: torch.Tensor | None = None,
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

        if page_tables is not None:
            page_tables_to_use = self._convert_vllm_page_table_for_batch(page_tables, device=self.mesh_device)
        else:
            page_tables_to_use = self._get_page_tables()
        # RowBatchedModel forward
        logits_tt = RowBatchedModel.forward_decode(
            tt_tokens,
            tt_positions,
            self.model_run_config_decode,
            rope_tensors,
            page_tables=page_tables_to_use,
            profile_decode=self.profile_decode,
        )

        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)

        return logits_tt  # [1, 1, B, V]

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample tokens greedily from logits. Handles both [B, V] and [1, 1, B, V] shapes."""
        sampled = torch.argmax(logits, dim=-1)  # [..., B]
        # Squeeze out any leading dimensions to get [B]
        while sampled.dim() > 1:
            sampled = sampled.squeeze(0)
        return sampled  # [B]

    def _logits_to_host(self, logits_tt: ttnn.Tensor) -> torch.Tensor:
        """Convert logits from device to host for sampling. Returns [1, 1, B, V] tensor."""
        logger.info(f"logits_tt.shape before to_torch: {logits_tt.shape}")
        logits = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )
        logger.info(f"logits.shape after to_torch: {logits.shape}")
        return logits  # [1, 1, B, V]

    def _maybe_dump_host_logits(
        self,
        logits_host: torch.Tensor,
        *,
        stage: str,
        repeat_idx: int,
        mode: str,
        user_id: int | None = None,
        gen_idx: int | None = None,
    ) -> None:
        if not self.dump_host_logits or self._logits_dump_dir is None:
            return

        filename_parts = [f"rank{self._rank}", f"mode_{mode}", stage, f"repeat_{repeat_idx:03d}"]
        if user_id is not None:
            filename_parts.append(f"user_{user_id:03d}")
        if gen_idx is not None:
            filename_parts.append(f"step_{gen_idx:04d}")
        filename = "__".join(filename_parts) + ".pt"
        out_path = self._logits_dump_dir / filename

        payload = {
            "stage": stage,
            "repeat_idx": repeat_idx,
            "user_id": user_id,
            "gen_idx": gen_idx,
            "mode": mode,
            "shape": tuple(logits_host.shape),
            "dtype": str(logits_host.dtype),
            "logits": logits_host.detach().cpu(),
        }
        torch.save(payload, out_path)

    def _maybe_dump_sampling_compare(
        self,
        *,
        repeat_idx: int,
        gen_idx: int,
        host_argmax: torch.Tensor,
        device_tokens: torch.Tensor,
    ) -> None:
        if not self.dump_sampling_compare or self._logits_dump_dir is None:
            return

        host_argmax_cpu = host_argmax.detach().cpu().reshape(-1).to(torch.int64)
        device_tokens_cpu = device_tokens.detach().cpu().reshape(-1).to(torch.int64)
        n = min(host_argmax_cpu.numel(), device_tokens_cpu.numel())
        host_argmax_cpu = host_argmax_cpu[:n]
        device_tokens_cpu = device_tokens_cpu[:n]
        mismatch_mask = host_argmax_cpu != device_tokens_cpu
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False).reshape(-1).to(torch.int64)

        filename = (
            f"rank{self._rank}__mode_device__decode_sampling_compare__repeat_{repeat_idx:03d}__step_{gen_idx:04d}.pt"
        )
        out_path = self._logits_dump_dir / filename
        payload = {
            "repeat_idx": repeat_idx,
            "gen_idx": gen_idx,
            "num_users": int(n),
            "num_mismatches": int(mismatch_indices.numel()),
            "mismatch_indices": mismatch_indices,
            "host_argmax_tokens": host_argmax_cpu,
            "device_sampled_tokens": device_tokens_cpu,
        }
        torch.save(payload, out_path)

    def _decode_step_tt(
        self,
        tt_tokens: ttnn.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_table: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Run a single decode step using device tokens and return logits on device."""
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
            page_tables_to_use = self._convert_vllm_page_table_for_batch(page_table, device=self.mesh_device)
        else:
            page_tables_to_use = self._get_page_tables()

        logits_tt = RowBatchedModel.forward_decode(
            tt_tokens,
            tt_positions,
            self.model_run_config_decode,
            rope_tensors,
            page_tables=page_tables_to_use,
            profile_decode=self.profile_decode,
        )
        return logits_tt

    def _pad_batch(self, tokens_list: List[List[int]], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad/pack a list of token id sequences to batch of size batch_size.

        Returns
            tokens_packed: torch.LongTensor [batch_size, S]
            lengths: torch.IntTensor [batch_size] with actual sequence lengths for first N sequences, zeros otherwise
        """
        assert len(tokens_list) > 0 and len(tokens_list) <= batch_size
        max_len = max(len(t) for t in tokens_list)
        if self.prefill_max_tokens is not None:
            max_len = min(self.prefill_max_tokens, max_len)  # truncate all sequences to the prefill_max_tokens
            logger.info(f"Truncating sequences to {max_len} tokens")
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
            len_seq = len(seq) if self.prefill_max_tokens is None else min(self.prefill_max_tokens, len(seq))
            out[i, :len_seq] = torch.tensor(seq[:len_seq], dtype=torch.long)
            lengths[i] = len_seq
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
        if len(prompts) > self.batch_size:
            logger.warning(f"Supports 1..{self.batch_size} prompts. Cutton off additional prompts.")
            prompts = prompts[: self.batch_size]
        num_of_prompts = len(prompts)

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

        # if sampling is None:
        #     sampling = self.sampling_generator

        if self.enable_trace and teacher_forcing is not None:
            logger.warning("Teacher forcing is disabled when enable_trace=True.")
            teacher_forcing = None

        # Run one or more prefill+decode batches
        for repeat_idx in range(repeat_batches):
            # Reset teacher-forcing state per batch.
            if teacher_forcing is not None:
                teacher_forcing.reset()

            # Prefill (can be skipped for decode-only profiling)
            num_of_users = tokens_batched.shape[0]
            if self.profile_decode:
                logger.info("Skipping prefill (profile_decode=True) - using random tokens for decode profiling")
                # Generate random starting token IDs directly instead of
                # allocating a full [num_users, vocab_size] logits tensor.
                vocab_size = int(getattr(self.hf_config, "vocab_size", 32768))
                next_tokens_override = torch.randint(0, vocab_size, (num_of_users,))
                # Set lengths to 0 so positions start at 0
                lengths = torch.zeros((num_of_users,), dtype=torch.int32)
            else:
                if self.signpost:
                    signpost(header="prefill")
                profiler.start("inference_prefill")
                last_tokens = []
                for user_id in range(num_of_users):
                    if lengths[user_id] == 0:
                        logger.info(f"Skipping prefill for user_id: {user_id} as prompt length is 0")
                        pad_token = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
                        last_tokens.append(torch.tensor(int(pad_token), dtype=torch.int64))
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
                    prefill_logits = self._prefill(tokens_batched[user_id], user_id=user_id, return_logits=True)
                    logger.info(f"prefill_logits.shape: {prefill_logits.shape}")
                    assert prefill_logits is not None
                    last_logits = self._slice_last_token_logits(prefill_logits, prompt_len)
                    logger.info(f"last_logits.shape after slice: {last_logits.shape}")
                    last_logits = self._expand_prefill_logits(last_logits)
                    logger.info(f"last_logits.shape after expand: {last_logits.shape}")
                    host_last_logits_for_debug = None
                    if self.dump_host_logits:
                        host_last_logits_for_debug = self._logits_to_host(last_logits)
                        self._maybe_dump_host_logits(
                            host_last_logits_for_debug,
                            stage="prefill",
                            repeat_idx=repeat_idx,
                            mode="device" if self.sample_on_device else "host",
                            user_id=user_id,
                            gen_idx=0,
                        )

                    if self.sample_on_device:
                        # Device sampling (new way)
                        tt_pred = self._sample_tokens_device(last_logits)
                        host_pred = self._tokens_from_device(tt_pred, batch_size=1)
                        pred_token = host_pred[0]
                        ttnn.deallocate(tt_pred)
                    else:
                        # Host sampling (old way)
                        host_logits = (
                            host_last_logits_for_debug
                            if host_last_logits_for_debug is not None
                            else self._logits_to_host(last_logits)
                        )
                        sampled_tokens = self._sample_greedy(host_logits)  # [B]
                        pred_token = sampled_tokens[0]  # Get first token

                    ttnn.deallocate(last_logits)
                    ttnn.deallocate(prefill_logits)
                    # Keep a stable host dtype across host/device sampling paths so torch.stack succeeds.
                    if isinstance(pred_token, torch.Tensor):
                        pred_token = int(pred_token.item())
                    last_tokens.append(torch.tensor(pred_token, dtype=torch.int64))
                    self.ccl.reset_sem_counters()
                last_tokens = torch.stack(last_tokens)
                profiler.end("inference_prefill")
                if self.signpost:
                    signpost(header="prefill")
                ttnn.ReadDeviceProfiler(self.mesh_device)

            if not self.profile_decode:
                assert len(last_tokens) == num_of_users

            logger.info(
                f"Finished prefill for all users..."
                if not self.profile_decode
                else "Skipped prefill, starting decode..."
            )
            generations: List[List[int]] = [[] for _ in range(num_of_prompts)]
            if max_new_tokens <= 0:
                logger.info("max_new_tokens <= 0, skipping decode loop.")
            else:
                logger.info(f"Generating {max_new_tokens} tokens for {num_of_prompts} user(s)...")
                if early_print_first_user:
                    logger.info("===== Generation for first user =====")

                # First generated token comes from prefill's last-position logits,
                # or from random IDs when profiling decode only.
                if self.profile_decode:
                    next_tokens = next_tokens_override
                else:
                    next_tokens = last_tokens
                if teacher_forcing is not None:
                    # Record user-0 prediction for accuracy, but force teacher token for alignment.
                    forced0 = teacher_forcing.collect_predicted_tokens(int(next_tokens[0].item()))
                    next_tokens[0] = int(forced0)

                # Non-trace decode path consumes device-resident tokens.
                tt_next_tokens = None
                if not self.enable_trace and self.sample_on_device:
                    tt_next_tokens = self._tt_from_tokens_step(next_tokens)

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
                read_events = []
                tt_out_toks_cpu = []
                trace_exec_offset = 1
                for gen_idx in range(decode_steps):
                    logger.info(f"Decoding step {gen_idx} for {num_of_prompts} user(s)...")
                    profiler.start(f"decode_time_{gen_idx}")
                    host_decode_logits = None

                    if self.enable_trace:
                        # Trace mode (always uses device sampling)
                        logits = self.decode_forward(
                            next_tokens,
                            positions,
                            self.batch_size_per_row,
                            profiler,
                            gen_idx,
                            enable_trace=True,
                        )
                        tt_next_tokens = self._sample_tokens_device(
                            logits, tt_out_tok=self._trace_tokens, enable_trace=True
                        )
                    elif self.sample_on_device:
                        # Device sampling (new way)
                        logits = self.decode_forward_tt(
                            tt_next_tokens,
                            positions,
                            self.batch_size_per_row,
                            enable_trace=False,
                        )
                    else:
                        # Host sampling (old way)
                        logits_tt = self._decode_step(next_tokens, positions, self.batch_size_per_row)
                        logits = self._logits_to_host(logits_tt)
                        ttnn.deallocate(logits_tt)

                    if self.dump_host_logits:
                        host_decode_logits = (
                            logits if isinstance(logits, torch.Tensor) else self._logits_to_host(logits)
                        )
                        self._maybe_dump_host_logits(
                            host_decode_logits,
                            stage="decode",
                            repeat_idx=repeat_idx,
                            mode="device" if self.sample_on_device else "host",
                            gen_idx=gen_idx,
                        )

                    profiler.end(f"decode_time_{gen_idx}")
                    self.ccl.reset_sem_counters()

                    if self.enable_trace:
                        tt_out_toks_cpu.append(tt_next_tokens.cpu(blocking=False, cq_id=0))
                        read_events.append(ttnn.record_event(self.mesh_device, 0))
                        ready_idx = gen_idx - trace_exec_offset
                        if ready_idx >= 0:
                            ttnn.event_synchronize(read_events[ready_idx])
                            tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out_toks_cpu[ready_idx])[0])
                            pred_tokens = tt_output_torch.reshape(-1)[: self.batch_size]
                            for i in range(num_of_prompts):
                                token_value = int(pred_tokens[i].item())
                                generations[i].append(token_value)
                                if early_print_first_user and i == 0:
                                    print(
                                        self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True
                                    )
                    elif self.sample_on_device:
                        # Device sampling path
                        self._sample_tokens_device(logits, tt_out_tok=tt_next_tokens)
                        if self.dump_sampling_compare:
                            host_decode_logits_cmp = (
                                host_decode_logits if host_decode_logits is not None else self._logits_to_host(logits)
                            )
                            host_argmax_tokens = self._sample_greedy(host_decode_logits_cmp)
                        ttnn.deallocate(logits)
                        pred_tokens = self._tokens_from_device(tt_next_tokens, self.batch_size)
                        if self.dump_sampling_compare:
                            self._maybe_dump_sampling_compare(
                                repeat_idx=repeat_idx,
                                gen_idx=gen_idx,
                                host_argmax=host_argmax_tokens,
                                device_tokens=pred_tokens,
                            )
                        if teacher_forcing is not None:
                            forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                            pred_tokens[0] = int(forced)
                            ttnn.deallocate(tt_next_tokens)
                            tt_next_tokens = self._tt_from_tokens_step(pred_tokens)
                        next_tokens = pred_tokens
                        positions += 1
                        for i in range(num_of_prompts):
                            token_value = int(pred_tokens[i].item())
                            generations[i].append(token_value)
                            if early_print_first_user and i == 0:
                                print(self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True)
                    else:
                        # Host sampling path (old way)
                        pred_tokens = self._sample_greedy(logits)
                        if teacher_forcing is not None:
                            forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                            pred_tokens[0] = int(forced)
                        next_tokens = pred_tokens
                        positions += 1
                        for i in range(num_of_prompts):
                            token_value = int(next_tokens[i].item())
                            generations[i].append(token_value)
                            if early_print_first_user and i == 0:
                                if self.tokenizer is not None:
                                    print(
                                        self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True
                                    )
                                else:
                                    print(f"{token_value} ", end="", flush=True)

                if self.enable_trace:
                    for trailing_idx in range(max(0, max_new_tokens - trace_exec_offset), max_new_tokens):
                        ttnn.event_synchronize(read_events[trailing_idx])
                        tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out_toks_cpu[trailing_idx])[0])
                        pred_tokens = tt_output_torch.reshape(-1)[: self.batch_size]
                        for i in range(num_of_prompts):
                            token_value = int(pred_tokens[i].item())
                            generations[i].append(token_value)
                            if early_print_first_user and i == 0:
                                print(self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True)

                profiler.end("inference_decode")

            if early_print_first_user:
                logger.info("\n===== Done =====")

        profiler.end("run")
        # Calculate statistics
        prefill_time = profiler.get_duration("inference_prefill") if not self.profile_decode else 0
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
        return_logits: bool = False,
    ) -> ttnn.Tensor | None:
        """Run prefill for the full prompt sequence to populate caches.

        Args:
            tokens: [1, 1, seq_len] padded token sequences
            user_id: user id for the prefill
            local_user_id: local user id for page table lookup

        Returns:
            logits tensor if return_logits is True, otherwise None
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
        out_tt = RowBatchedModel.forward_prefill(
            x=tt_tokens,
            user_id=user_id,
            cfg=self.model_run_config_prefill,
            rope_tensors=rope_tensors,
            page_tables=page_tables_to_use,
        )

        ttnn.deallocate(tt_tokens)
        if return_logits:
            return out_tt
        ttnn.deallocate(out_tt)
        return None

    def _slice_last_token_logits(self, logits: ttnn.Tensor, prompt_len: int) -> ttnn.Tensor:
        last_idx = max(prompt_len - 1, 0)
        shard_len = logits.shape[2]
        local_idx = last_idx % shard_len
        row_idx = last_idx // shard_len

        last_logits = ttnn.slice(
            logits,
            [0, 0, local_idx, 0],
            [1, 1, local_idx + 1, logits.shape[-1]],
        )

        if self.mesh_device.shape[0] > 1:
            gather_cfg = self.ccl.populate_all_gather_runtime_args(
                {
                    "cluster_axis": 0,
                    "dim": 2,
                    "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                    "topology": ttnn.Topology.Linear,
                }
            )
            gathered = ttnn.experimental.all_gather_async(last_logits, **gather_cfg)
            ttnn.deallocate(last_logits)
            last_logits = ttnn.slice(
                gathered,
                [0, 0, row_idx, 0],
                [1, 1, row_idx + 1, gathered.shape[-1]],
            )
            ttnn.deallocate(gathered)

        return last_logits

    def _expand_prefill_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        if logits.shape[2] == self.batch_size_per_row:
            return logits
        if logits.shape[2] == 1:
            expanded = ttnn.repeat(logits, (1, 1, self.batch_size_per_row, 1))
            ttnn.deallocate(logits)
            return expanded
        return logits

    def _capture_decode_trace(
        self,
        init_tokens: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_tables: torch.Tensor | None = None,
    ) -> None:
        """Allocate persistent inputs, capture trace for one decode iteration, and store trace state."""
        assert self._trace_id is None, "Trace already captured"

        # 1) Allocate persistent device inputs
        self._trace_tokens = self._tt_from_tokens_step(init_tokens)
        self._trace_positions = ttnn.from_torch(
            positions,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.int32,
        )

        self._trace_rot_idxs = self.rope_setup.get_rot_idxs(positions)

        if page_tables is not None:
            self._trace_page_tables_to_use = self._convert_vllm_page_table_for_batch(
                page_tables, device=self.mesh_device
            )
        else:
            self._trace_page_tables_to_use = self._get_page_tables()
        ttnn.synchronize_device(self.mesh_device)

        # 2) Warm-up compile run (no trace) using persistent buffers
        logger.info("Running warm-up decode step (no trace)...")
        self._reset_trace_inputs(init_tokens, positions)
        warmup_rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(self._trace_rot_idxs)
        warmup_logits = RowBatchedModel.forward_decode(
            x=self._trace_tokens,
            position_idxs=self._trace_positions,
            cfg=self.model_run_config_decode,
            rope_tensors=warmup_rope_tensors,
            page_tables=self.page_tables_tt,
        )
        self._increment_decode_positions_device()
        _ = self.sampling_generator.sample(warmup_logits, enable_trace=False, tt_out_tok=self._trace_tokens)
        ttnn.deallocate(warmup_logits)
        ttnn.synchronize_device(self.mesh_device)

        # 3) Capture decode graph
        self._reset_trace_inputs(init_tokens, positions)
        self.ccl.reset_sem_counters()
        logger.info("Begin capturing decode trace...")
        if self.signpost:
            signpost(header="decode_trace_capture")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Only capture the rot_mats generation from rot_idxs (all ttnn ops, no from_torch)
        rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(self._trace_rot_idxs)
        logits = RowBatchedModel.forward_decode(
            x=self._trace_tokens,
            position_idxs=self._trace_positions,
            cfg=self.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_tables=self._trace_page_tables_to_use,
            profile_decode=self.profile_decode,
        )
        self._trace_logits = logits
        self._increment_decode_positions_device()
        self._trace_output = logits
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        if self.signpost:
            signpost(header="decode_trace_capture")
        logger.info("Decode trace capture complete.")
        self._trace_id = trace_id

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        batch_size_per_row: int = USERS_PER_ROW,
        gen_idx: int = 0,
        profiler: BenchmarkProfiler | None = None,
        enable_trace: bool = False,
        page_table: torch.Tensor | None = None,
        kv_cache: None = None,
        read_from_device: bool = None,
        sampling_params: SamplingParams = None,
    ) -> ttnn.Tensor:
        # vLLM does not pass enable_trace param while initializing the model.
        # vLLM sets it in decode/prefill calls only, so we need to set it here too.
        self.enable_trace = enable_trace
        if not enable_trace:
            # _decode_step returns logits on device
            return self._decode_step(tokens, start_pos, batch_size_per_row)
        else:
            # Capture trace and return trace output
            if self._trace_id is None:
                self._capture_decode_trace(tokens, start_pos, batch_size_per_row, page_table)
                # First call: return the captured run's output (logits)
                assert self._trace_output is not None
                return self._trace_output

            # Update persistent inputs and execute
            assert (
                self._trace_tokens is not None
                and self._trace_positions is not None
                and self._trace_rot_idxs is not None
                and self._trace_id is not None
                and self._trace_page_tables_to_use is not None
            )

            if gen_idx == 0:
                self._reset_trace_inputs(tokens, start_pos)

            if page_table is not None:
                page_tables_to_use = self._convert_vllm_page_table_for_batch(page_table, device=None)
                for i, page_table in enumerate(page_tables_to_use):
                    ttnn.copy_host_to_device_tensor(page_table, self._trace_page_tables_to_use[i])

            self.ccl.reset_sem_counters()
            if self.signpost:
                signpost(header="decode_execute_trace")
            if profiler is not None:
                profiler.start(f"trace_execution_{gen_idx}")
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)
            if profiler is not None:
                profiler.end(f"trace_execution_{gen_idx}")
                logger.info(
                    f"Trace execution t/s/user @ {gen_idx}th token: {1/profiler.get_duration(f'trace_execution_{gen_idx}')}"
                )
            assert self._trace_output is not None

            if self.signpost:
                signpost(header="decode_execute_trace")
            if self.profile_decode:
                # trigger the profiler to read the device side data each iteration to not miss any data
                ttnn.ReadDeviceProfiler(self.mesh_device)
            return self._trace_output

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device) -> None:
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

    def decode_forward_tt(
        self,
        tt_tokens: ttnn.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        enable_trace: bool = False,
    ) -> ttnn.Tensor:
        if enable_trace:
            raise NotImplementedError("Trace mode uses host token updates; call decode_forward instead.")
        return self._decode_step_tt(tt_tokens, positions, batch_size_per_row)

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
        if getattr(self, "kv_cache_shape", None) is not None:
            expected_blocks = int(self.kv_cache_shape[0])
            expected_block_size = int(self.kv_cache_shape[2])
            if (
                self.paged_config.max_num_blocks != expected_blocks
                or self.paged_config.block_size != expected_block_size
            ):
                logger.warning(
                    "KVDBG paged_config mismatch with kv_cache_shape: paged_max_blocks={} kv_max_blocks={} "
                    "paged_block_size={} kv_block_size={}",
                    self.paged_config.max_num_blocks,
                    expected_blocks,
                    self.paged_config.block_size,
                    expected_block_size,
                )
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
            batch_size=self.batch_size_per_row,
        )

        num_layers = self.hf_config.num_hidden_layers
        return tuple(ttnn.clone(page_table_tt) for _ in range(num_layers))

    def _convert_vllm_page_table_for_batch(
        self, page_table: torch.Tensor, device: ttnn.Device | ttnn.MeshDevice | None
    ) -> tuple[ttnn.Tensor, ...]:
        """
        Convert vLLM's block_tables (page_table) to TTNN tensor format for the entire batch.
        Creates one page table per layer as expected by the model.

        Args:
            page_table: torch.Tensor of shape [batch_size, max_num_blocks_per_req] from vLLM
            device: ttnn.Device, ttnn.MeshDevice, or None. If provided, creates device tensors on the specified device.
                   If None, creates host tensors instead of device tensors.

        Returns:
            Tuple of TTNN tensors, one per layer
        """
        # Use vLLM page table directly, but shard it across devices to match the sharded batch size
        # in paged_update_cache.
        # page_table shape: [batch_size, max_blocks_per_req]

        page_table_tt = ttnn.from_torch(
            page_table,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return tuple(page_table_tt for _ in range(self.hf_config.num_hidden_layers))


__all__ = ["DeepseekGenerator", "SamplingParams"]
