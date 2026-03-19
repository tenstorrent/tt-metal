# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, NamedTuple, Tuple

import torch
from loguru import logger
from tracy import signpost
from transformers import AutoConfig

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, chunk_sampling_params
from models.common.warmup import WarmupForwardMixin
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_K,
    DEFAULT_SAMPLING_TOP_P,
    USERS_PER_ROW,
    even_int_div,
    make_deepseek_sampling_args,
)
from models.demos.deepseek_v3.utils.debug_utils import dump_ttnn_meminfo
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.weight_config import get_weight_config
from models.perf.benchmarking_utils import BenchmarkProfiler

DEFAULT_MAX_SEQ_LEN = 2048


def _build_verify_alias_page_table_host(
    base_page_table: torch.Tensor,
    num_prompts: int,
    verify_offset: int,
    prompt_indices: List[int] | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    """Build a host-side aliased page table for verify batching."""
    if num_prompts <= 0:
        return base_page_table.clone()
    if base_page_table.dim() != 2:
        raise RuntimeError(f"Unexpected page table rank for MTP verify aliasing: {tuple(base_page_table.shape)}")

    alias_page_table = base_page_table.clone().to(torch.int32)
    num_rows = int(alias_page_table.shape[0])
    if num_rows <= 0:
        raise RuntimeError("Page table has zero rows; cannot build MTP verify aliasing.")

    prompt_indices_for_alias = prompt_indices
    if not interleaved and prompt_indices_for_alias is None:
        prompt_indices_for_alias = list(range(num_prompts))

    if interleaved:
        if prompt_indices_for_alias is None:
            for row in range(1, num_rows, 2):
                alias_page_table[row] = alias_page_table[row - 1]
        else:
            for i in prompt_indices_for_alias:
                if i < 0:
                    continue
                src_row = (2 * i) % num_rows
                dst_row = (src_row + 1) % num_rows
                alias_page_table[dst_row] = alias_page_table[src_row]
    else:
        for i in prompt_indices_for_alias:
            if i < 0 or i >= num_prompts:
                continue
            src_row = i % num_rows
            dst_row = (verify_offset + i) % num_rows
            alias_page_table[dst_row] = alias_page_table[src_row]

    return alias_page_table


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


class _MtpPromptLayout(NamedTuple):
    use_mtp_path: bool
    tokens_batched: torch.Tensor
    lengths: torch.Tensor
    prompt_user_ids: torch.Tensor | None
    spec_user_ids: torch.Tensor | None


class _MtpDecodeBootstrap(NamedTuple):
    next_tokens: torch.Tensor
    positions: torch.Tensor
    spec_tokens: torch.Tensor | None
    decode_page_tables: tuple[ttnn.Tensor, ...]


class _MtpDecodeLoopResult(NamedTuple):
    mtp_accept_rate: float
    mtp_accepts: int
    mtp_verifies: int
    decode_step_idx: int
    decode_forward_passes: int
    decode_step_active_masks: List[List[bool]]
    decode_step_user_tokens: List[List[int]]


class DeepseekGenerator(WarmupForwardMixin):
    """
    Simple generator that wires RowBatchedModel + LMHead for decode-only inference.

    Notes:
    - Prefill at the model level is not fully implemented in RowBatchedModel; we emulate
      prefill by iterating decode steps over the prompt tokens (updates caches).
    - Decode runs are configured for up to a fixed number of users per row.
      If fewer prompts are provided, we pad/ignore extras.

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
        batch_size_per_row: int = USERS_PER_ROW,
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
        sample_on_device: bool = False,
        enable_mtp: bool = False,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.model_path = str(model_path)
        self.cache_dir = cache_dir

        # Load HF config + tokenizer
        self.hf_config = (
            hf_config if hf_config is not None else AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        )
        model_max_seq_len = int(self.hf_config.max_position_embeddings)
        requested_max_seq_len = DEFAULT_MAX_SEQ_LEN if max_seq_len is None else int(max_seq_len)
        if requested_max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {requested_max_seq_len}")
        if requested_max_seq_len % ttnn.TILE_SIZE != 0:
            raise ValueError(f"max_seq_len {requested_max_seq_len} must be divisible by TILE_SIZE={ttnn.TILE_SIZE}")
        if requested_max_seq_len > model_max_seq_len:
            raise ValueError(
                f"max_seq_len {requested_max_seq_len} exceeds model-supported context length {model_max_seq_len}"
            )
        if requested_max_seq_len != DEFAULT_MAX_SEQ_LEN:
            logger.warning(
                "Using overridden max_seq_len={} (default={}, model supports up to {}).",
                requested_max_seq_len,
                DEFAULT_MAX_SEQ_LEN,
                model_max_seq_len,
            )
        self.hf_config.max_seq_len = requested_max_seq_len
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
        self.enable_mtp = bool(enable_mtp)

        if not self.enable_mtp and hasattr(self.hf_config, "num_nextn_predict_layers"):
            self.hf_config.num_nextn_predict_layers = 0
        logger.info(f"MTP enabled: {self.enable_mtp}")
        # Tokenizer is optional; caller can pass a tokenizer or handle failure.
        self.tokenizer = tokenizer

        # Runtime helpers
        self.ccl = CCL(mesh_device)
        mesh_shape = list(mesh_device.shape)
        self.dp_factor = mesh_shape[1]
        batch_size_per_row = int(batch_size_per_row)
        if batch_size_per_row <= 0:
            raise ValueError(f"batch_size_per_row must be > 0, got {batch_size_per_row}")
        if batch_size_per_row > USERS_PER_ROW:
            raise ValueError(f"batch_size_per_row {batch_size_per_row} exceeds the supported maximum {USERS_PER_ROW}")
        if batch_size_per_row % self.dp_factor != 0:
            raise ValueError(f"batch_size_per_row {batch_size_per_row} must be divisible by dp_factor={self.dp_factor}")
        self.batch_size_per_row = batch_size_per_row
        self.batch_size = self.batch_size_per_row * self.mesh_device.shape[0]

        # Configure sampling
        # sampling values of all users are assumed to be the same default values if not provided in constructor.
        self.sample_on_device = sample_on_device
        self.sampling_params = (
            sampling_params
            if sampling_params is not None
            else SamplingParams(
                temperature=[DEFAULT_SAMPLING_TEMPERATURE] * self.batch_size,
                top_p=[DEFAULT_SAMPLING_TOP_P] * self.batch_size,
                top_k=[DEFAULT_SAMPLING_TOP_K] * self.batch_size,
            )
        )
        if self._get_sampling_value(self.sampling_params.top_k, 0) == 0 and self.sample_on_device:
            raise SystemExit(
                "top-k=0 is not supported when sampling on device. Sampling on host instead. See https://github.com/tenstorrent/tt-metal/issues/40236"
            )
        if self.sample_on_device:
            enable_internal_trace_sampling = enable_trace and self.sample_on_device
            self.sampling_args = make_deepseek_sampling_args(
                mesh_device,
                self.hf_config.vocab_size,
                max_batch_size=self.batch_size_per_row,
            )
            self.sampling_generator = SamplingGenerator(
                args=self.sampling_args,
                mesh_device=self.mesh_device,
                tt_ccl=self.ccl,
                enable_internal_trace=enable_internal_trace_sampling,
            )

            self._reset_sampling_state(self.sampling_params, self.batch_size, self.batch_size_per_row)

        logger.info(f"Sampling mode: {'device' if self.sample_on_device else 'host'}")
        logger.info(
            f"Sampling parameters for first user (other users may have different values): "
            + f"temperature={self._get_sampling_value(self.sampling_params.temperature, 0)}, "
            + f"top_p={self._get_sampling_value(self.sampling_params.top_p, 0)}, "
            + f"top_k={self._get_sampling_value(self.sampling_params.top_k, 0)}"
        )

        if enable_mtp and sample_on_device:
            raise SystemExit("MTP with sampling on device is not supported. Disable MTP or sample on host.")

        # Weight cache to avoid loading weights multiple times
        self._weight_ttnn_cache: dict[str, ttnn.Tensor] = {}
        # Paged attention setup
        self.paged_config = MLA2D.get_valid_paged_config(
            self.hf_config.max_seq_len, self.batch_size_per_row, self.dp_factor
        )

        self.random_weights = random_weights
        self.single_layer = single_layer

        # Model runtime state
        self.model_state = None
        self.model_shared_state = None
        self.model_prefill_cfg = None
        self.model_decode_cfg = None
        self.model_weight_config = None
        self.page_tables_tt = None
        self.mtp_page_table_tt = None
        self.base_page_table_host = None
        self.mtp_page_table_host = None

        # Trace state (decode)
        self._trace_id: int | None = None
        self._trace_tokens: ttnn.Tensor | None = None
        self._trace_positions: ttnn.Tensor | None = None
        self._trace_rot_idxs: ttnn.Tensor | None = None
        self._trace_output: ttnn.Tensor | None = None
        self._trace_page_tables_to_use: tuple[ttnn.Tensor, ...] | None = None
        self._mtp_verify_trace_id: int | None = None
        self._mtp_verify_trace_tokens: ttnn.Tensor | None = None
        self._mtp_verify_trace_positions: ttnn.Tensor | None = None
        self._mtp_verify_trace_rot_idxs: ttnn.Tensor | None = None
        self._mtp_verify_trace_output: tuple[ttnn.Tensor, ttnn.Tensor] | None = None
        self._mtp_verify_trace_page_tables: tuple[ttnn.Tensor, ...] | None = None
        self._mtp_predict_trace_id: int | None = None
        self._mtp_predict_trace_hidden: ttnn.Tensor | None = None
        self._mtp_predict_trace_tokens: ttnn.Tensor | None = None
        self._mtp_predict_trace_positions: ttnn.Tensor | None = None
        self._mtp_predict_trace_rot_idxs: ttnn.Tensor | None = None
        self._mtp_predict_trace_output: ttnn.Tensor | None = None
        self._mtp_predict_trace_page_table: ttnn.Tensor | None = None
        self.enable_trace = enable_trace
        self.enable_mem_profile = enable_mem_profile
        self.signpost = signpost
        self.prefill_max_tokens = prefill_max_tokens
        self.force_recalculate = force_recalculate
        self.profile_decode = profile_decode  # Profile decode: skip prefill, run only 1st dense + 1st MoE layer
        logger.info(f"Enable trace: {self.enable_trace}")
        if self.profile_decode:
            logger.info("profile_decode=True: Prefill skipped, decode runs only 1st dense layer + 1st MoE layer")

        # Initialize rope_setup once
        self.rope_setup = RotarySetup(
            device=self.mesh_device, batch_size_per_row=self.batch_size_per_row, hf_config=self.hf_config
        )

        self._prepare_weight_configs(cache_dir)
        self._assert_mtp_available()

    def _dump_meminfo(self, header: str) -> None:
        if self.enable_mem_profile:
            dump_ttnn_meminfo(self.mesh_device, header=header)

    def _prepare_weight_configs(self, cache_dir: str | Path | None) -> None:
        weight_cache_base = Path(cache_dir) if cache_dir is not None else Path("generated/deepseek_v3")
        weight_cache_base.mkdir(parents=True, exist_ok=True)

        cache_subdir_name = f"{self.hf_config.num_hidden_layers}_layers"
        if self.enable_mtp:
            cache_subdir_name = f"{cache_subdir_name}_mtp"

        self.model_weight_config = get_weight_config(
            ModuleClass=RowBatchedModel,
            hf_config=self.hf_config,
            weight_cache_path=weight_cache_base,
            mesh_device=self.mesh_device,
            force_recalculate=self.force_recalculate,
            random_weights=self.random_weights,
            model_path=self.model_path,
            single_layer=self.single_layer,
            cache_subdir_name=cache_subdir_name,
        )

    def _assert_mtp_available(self) -> None:
        if not self.enable_mtp:
            return

        if self.random_weights:
            raise ValueError("MTP cannot be enabled with --random-weights.")

        requested_mtp_layers = int(getattr(self.hf_config, "num_nextn_predict_layers", 0))
        if requested_mtp_layers <= 0:
            raise RuntimeError(
                "MTP was enabled, but the model config does not include a valid MTP layer "
                "(num_nextn_predict_layers <= 0)."
            )

        mtp_cfg = self.model_weight_config.get("mtp") if isinstance(self.model_weight_config, dict) else None
        if not isinstance(mtp_cfg, dict) or not mtp_cfg:
            raise RuntimeError(
                "MTP was enabled, but the resolved weight config does not contain MTP tensors. "
                "Regenerate the DeepSeek cache with MTP weights before running."
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
                hf_config=self.hf_config,
                mesh_device=self.mesh_device,
                batch_size_per_row=self.batch_size_per_row,
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
            if self.enable_mtp and (
                "mtp" not in self.model_run_config_prefill or self.model_run_config_prefill["mtp"] is None
            ):
                raise RuntimeError("MTP was enabled, but the prefill run config has no MTP block.")
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
                hf_config=self.hf_config,
                mesh_device=self.mesh_device,
                batch_size_per_row=self.batch_size_per_row,
            )
            self._dump_meminfo("Before creating model run config for decode...")
            self.model_run_config_decode = create_run_config(
                self.model_decode_cfg,
                self.model_weight_config,
                self.model_state,
                self.model_shared_state,
                cached_ttnn_weights=self._weight_ttnn_cache,
            )
            if self.enable_mtp and (
                "mtp" not in self.model_run_config_decode or self.model_run_config_decode["mtp"] is None
            ):
                raise RuntimeError("MTP was enabled, but the decode run config has no MTP block.")
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

        # Clean up sampling trace state
        try:
            if hasattr(self, "sampling_generator") and self.sampling_generator is not None:
                self.sampling_generator.reset_trace()
        except Exception as e:
            logger.warning(f"Failed to reset sampling trace state: {e}")

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
            if self._mtp_verify_trace_id is not None:
                ttnn.release_trace(self.mesh_device, self._mtp_verify_trace_id)
                del self._mtp_verify_trace_id
            if self._mtp_verify_trace_tokens is not None:
                ttnn.deallocate(self._mtp_verify_trace_tokens)
                del self._mtp_verify_trace_tokens
            if self._mtp_verify_trace_positions is not None:
                ttnn.deallocate(self._mtp_verify_trace_positions)
                del self._mtp_verify_trace_positions
            if self._mtp_verify_trace_rot_idxs is not None:
                ttnn.deallocate(self._mtp_verify_trace_rot_idxs)
                del self._mtp_verify_trace_rot_idxs
            if self._mtp_verify_trace_output is not None:
                verify_logits_tt, verify_hidden_tt = self._mtp_verify_trace_output
                ttnn.deallocate(verify_logits_tt)
                ttnn.deallocate(verify_hidden_tt)
                del self._mtp_verify_trace_output
            if (
                self._mtp_verify_trace_page_tables is not None
                and self._mtp_verify_trace_page_tables is not self.page_tables_tt
            ):
                for i, page_table in enumerate(self._mtp_verify_trace_page_tables):
                    try:
                        ttnn.deallocate(page_table)
                    except Exception as e:
                        logger.warning(f"Failed to deallocate MTP verify trace page table {i}: {e}")
                del self._mtp_verify_trace_page_tables
            if self._mtp_predict_trace_id is not None:
                ttnn.release_trace(self.mesh_device, self._mtp_predict_trace_id)
                del self._mtp_predict_trace_id
            if self._mtp_predict_trace_hidden is not None:
                ttnn.deallocate(self._mtp_predict_trace_hidden)
                del self._mtp_predict_trace_hidden
            if self._mtp_predict_trace_tokens is not None:
                ttnn.deallocate(self._mtp_predict_trace_tokens)
                del self._mtp_predict_trace_tokens
            if self._mtp_predict_trace_positions is not None:
                ttnn.deallocate(self._mtp_predict_trace_positions)
                del self._mtp_predict_trace_positions
            if self._mtp_predict_trace_rot_idxs is not None:
                ttnn.deallocate(self._mtp_predict_trace_rot_idxs)
                del self._mtp_predict_trace_rot_idxs
            if self._mtp_predict_trace_output is not None:
                ttnn.deallocate(self._mtp_predict_trace_output)
                del self._mtp_predict_trace_output
            if (
                self._mtp_predict_trace_page_table is not None
                and self._mtp_predict_trace_page_table is not self.mtp_page_table_tt
            ):
                ttnn.deallocate(self._mtp_predict_trace_page_table)
                del self._mtp_predict_trace_page_table
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
        self.base_page_table_host = None
        self.mtp_page_table_host = None
        try:
            if self.mtp_page_table_tt is not None:
                ttnn.deallocate(self.mtp_page_table_tt)
                del self.mtp_page_table_tt
        except Exception as e:
            logger.warning(f"Failed to cleanup MTP page table: {e}")

        try:
            self._release_alias_runtime_caches()
        except Exception as e:
            logger.warning(f"Failed to cleanup MTP alias runtime caches: {e}")

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

    def _reset_sampling_state(self, sampling_params: SamplingParams, batch_size: int, batch_size_per_row: int) -> None:
        sampling_dp = self.sampling_generator.tt_sampling._sampling_dp
        sampling_param_chunks = chunk_sampling_params(sampling_params, sampling_dp)
        seed = getattr(sampling_params, "seed", None)
        if seed is not None:
            user_ids = list(range(batch_size))
            self.sampling_generator.seed_manager.reset_seed(seed, user_ids)
        self.sampling_generator.apply_decode_state(
            sampling_param_chunks,
            reset_batch=True,
            prompt_tokens=torch.zeros((batch_size_per_row, 1), dtype=torch.int64),
            output_tokens=torch.zeros((batch_size_per_row, 1), dtype=torch.int64),
        )

    def _sample_tokens_device(
        self, logits: ttnn.Tensor, enable_trace: bool = False, user_slots: list[int] | None = None
    ) -> ttnn.Tensor:
        sampling_batch_size = self.sampling_generator.tt_sampling.max_batch_size
        sampling_logits = logits
        if logits.shape[2] != sampling_batch_size:
            if enable_trace:
                raise ValueError(
                    f"Device sampling trace requires logits batch {sampling_batch_size}, got {logits.shape[2]}"
                )
            if logits.shape[2] <= 0 or logits.shape[2] > sampling_batch_size:
                raise ValueError(
                    f"Device sampling expects logits batch in [1, {sampling_batch_size}], got {logits.shape[2]}"
                )
            # Sampling kernels operate on the padded per-row batch size. Append filler rows so
            # smaller decode/prefill batches can reuse the same device sampling path.
            filler_row = ttnn.slice(
                logits,
                [0, 0, logits.shape[2] - 1, 0],
                [1, 1, logits.shape[2], logits.shape[-1]],
            )
            filler = ttnn.repeat(filler_row, (1, 1, sampling_batch_size - logits.shape[2], 1))
            ttnn.deallocate(filler_row)
            sampling_logits = ttnn.concat([logits, filler], dim=2)
            ttnn.deallocate(filler)

        self.sampling_generator.seed_manager.get_new_values(user_slots)
        self.sampling_generator.enable_internal_trace = enable_trace
        try:
            tt_out = self.sampling_generator.sample(sampling_logits, enable_trace=enable_trace)
        finally:
            if sampling_logits is not logits:
                ttnn.deallocate(sampling_logits)

        if isinstance(tt_out, tuple):
            tt_tokens, tt_log_probs = tt_out
            if tt_log_probs is not None:
                ttnn.deallocate(tt_log_probs)
            tt_out = tt_tokens

        return tt_out

    def _tokens_from_device(self, tt_out_tok, mesh_device, batch_size_per_row: int) -> torch.Tensor:
        composed = ttnn.to_torch(
            tt_out_tok,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        if composed.ndim == 4:
            if tt_out_tok.shape[-2] == batch_size_per_row:
                tokens = composed[:, :, :, 0]
            elif tt_out_tok.shape[-1] == batch_size_per_row:
                tokens = composed[:, :, 0, :batch_size_per_row]
            else:
                tokens = composed
            tokens = tokens.reshape(-1)
        else:
            tokens = composed.reshape(-1)
        batch_size = batch_size_per_row * int(mesh_device.shape[0])
        return tokens[:batch_size].to(torch.int64)

    def _tt_from_hidden_states_step(
        self,
        hidden_states: torch.Tensor,
        device: ttnn.Device | ttnn.MeshDevice | None,
    ) -> ttnn.Tensor:
        """Hidden states step: [B, H] -> TTNN tensor [1, 1, B, H] sharded across rows/cols."""
        assert hidden_states.dim() == 2, "hidden_states must be [B, H]"
        x = hidden_states.view(1, 1, hidden_states.shape[0], hidden_states.shape[1]).to(torch.bfloat16).contiguous()
        return ttnn.from_torch(
            x,
            device=device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(-2, -1),
                mesh_shape=tuple(self.mesh_device.shape),
            ),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    def _normalize_hidden_host_for_mtp(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden_size = int(self.hf_config.hidden_size)

        if hidden.dim() == 4:
            hidden = hidden.squeeze(0).squeeze(0)
        if hidden.dim() == 3 and hidden.shape[1] == 1:
            hidden = hidden[:, 0, :]
        if hidden.dim() == 3 and hidden.shape[-1] == hidden_size:
            hidden = hidden.reshape(-1, hidden_size)

        if hidden.dim() != 2 or hidden.shape[-1] != hidden_size:
            raise RuntimeError(f"Unexpected hidden shape for MTP trace path: {tuple(hidden.shape)}")

        return hidden.to(torch.bfloat16).contiguous()

    def _hidden_tt_to_host_for_mtp(self, hidden_tt: ttnn.Tensor) -> torch.Tensor:
        hidden = ttnn.to_torch(
            hidden_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )
        return self._normalize_hidden_host_for_mtp(hidden)

    def _tt_from_positions(self, positions: torch.Tensor) -> Tuple[dict, ttnn.Tensor]:
        """Return rope tensors dict and TTNN positions shard for decode.

        positions: [B] int tensor
        returns: (rope_tensors, tt_positions)
        """
        rope_mats = self.rope_setup.get_rot_mats_table(seq_len=1)
        rope_tensors = {
            "cos_matrix": rope_mats["cos_matrix"],
            "sin_matrix": rope_mats["sin_matrix"],
            "trans_matrix": rope_mats["trans_matrix"],
        }

        tt_positions = ttnn.from_torch(
            positions.to(torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        return rope_tensors, tt_positions

    def _iter_decode_mla_cfgs(self):
        for block_cfg in self.model_run_config_decode["mlp_decoder_block"]:
            yield block_cfg["mla"]["mla1d"]
        for block_cfg in self.model_run_config_decode["moe_decoder_block"]:
            yield block_cfg["mla"]["mla1d"]

    def _iter_alias_runtime_mla_cfgs(self):
        model_run_config_decode = getattr(self, "model_run_config_decode", None)
        if model_run_config_decode is None:
            return
        yield from self._iter_decode_mla_cfgs()
        if "mtp" in model_run_config_decode:
            yield model_run_config_decode["mtp"]["decoder_block"]["mla"]["mla1d"]

    def _update_decode_page_table_alias_masks(self, page_table_host: torch.Tensor | None) -> None:
        for mla_cfg in self._iter_decode_mla_cfgs():
            MLA2D.update_page_table_alias_mask(mla_cfg, page_table_host)

    def _release_alias_runtime_caches(self) -> None:
        for mla_cfg in self._iter_alias_runtime_mla_cfgs():
            MLA2D.release_page_table_alias_runtime_cache(mla_cfg)

    def _get_page_tables(self) -> tuple[ttnn.Tensor, ...]:
        if hasattr(self, "page_tables_tt") and self.page_tables_tt is not None:
            return self.page_tables_tt

        assert hasattr(self, "paged_config") and self.paged_config is not None
        assert hasattr(self, "mesh_device") and self.mesh_device is not None
        assert hasattr(self, "batch_size_per_row") and self.batch_size_per_row is not None
        assert hasattr(self, "hf_config") and self.hf_config is not None
        batch_per_shard = even_int_div(self.batch_size_per_row, self.dp_factor)
        blocks_per_user = even_int_div(self.paged_config.max_num_blocks, batch_per_shard)
        self.base_page_table_host = torch.arange(self.paged_config.max_num_blocks, dtype=torch.int32).reshape(
            batch_per_shard, blocks_per_user
        )
        self.page_tables_tt = tuple(
            MLA2D.create_page_table(
                paged_config=self.paged_config,
                mesh_device=self.mesh_device,
                page_table=self.base_page_table_host,
                batch_size=self.batch_size_per_row,
            )
            for _ in range(self.hf_config.num_hidden_layers)
        )
        self._update_decode_page_table_alias_masks(self.base_page_table_host)
        return self.page_tables_tt

    def _get_mtp_page_table(self) -> ttnn.Tensor:
        if hasattr(self, "mtp_page_table_tt") and self.mtp_page_table_tt is not None:
            return self.mtp_page_table_tt

        assert hasattr(self, "paged_config") and self.paged_config is not None
        assert hasattr(self, "mesh_device") and self.mesh_device is not None
        assert hasattr(self, "batch_size_per_row") and self.batch_size_per_row is not None

        batch_per_shard = even_int_div(self.batch_size_per_row, self.dp_factor)
        blocks_per_user = even_int_div(self.paged_config.max_num_blocks, batch_per_shard)
        self.mtp_page_table_host = torch.arange(self.paged_config.max_num_blocks, dtype=torch.int32).reshape(
            batch_per_shard, blocks_per_user
        )
        self.mtp_page_table_tt = MLA2D.create_page_table(
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            page_table=self.mtp_page_table_host,
            batch_size=self.batch_size_per_row,
        )
        return self.mtp_page_table_tt

    def _decode_step(
        self,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        page_tables: torch.Tensor | None = None,
        sample_on_device: bool = False,
        return_hidden: bool = False,
    ) -> torch.Tensor | ttnn.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run a single decode step."""
        decode_out = self._decode_step_tt(
            tokens_step=tokens_step,
            positions=positions,
            batch_size_per_row=self.batch_size_per_row,
            page_tables=page_tables,
            return_hidden=return_hidden,
        )

        if return_hidden:
            logits_tt, hidden_tt = decode_out
            logits = ttnn.to_torch(
                logits_tt,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                ),
            )
            hidden = ttnn.to_torch(
                hidden_tt,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                ),
            )
            ttnn.deallocate(logits_tt)
            ttnn.deallocate(hidden_tt)
            return logits, hidden  # [1, 1, B, V], [1, 1, B, H]

        logits_tt = decode_out
        if sample_on_device:
            return logits_tt
        logits = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )
        ttnn.deallocate(logits_tt)
        return logits  # [1, 1, B, V]

    def _decode_step_tt(
        self,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_tables: torch.Tensor | tuple[ttnn.Tensor, ...] | None = None,
        return_hidden: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run a single decode step and return TT tensors."""
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

        if isinstance(page_tables, tuple):
            page_tables_to_use = page_tables
        elif page_tables is not None:
            page_tables_to_use = self._convert_vllm_page_table_for_batch(page_tables, device=self.mesh_device)
            self._update_decode_page_table_alias_masks(page_tables)
        else:
            page_tables_to_use = self._get_page_tables()
        decode_out = RowBatchedModel.forward_decode(
            tt_tokens,
            tt_positions,
            self.model_run_config_decode,
            rope_tensors,
            page_tables=page_tables_to_use,
            profile_decode=self.profile_decode,
            return_hidden=return_hidden,
        )
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(tt_positions)
        return decode_out

    def _build_mtp_verify_page_tables(
        self,
        num_prompts: int,
        verify_offset: int,
        prompt_indices: List[int] | None = None,
        interleaved: bool = False,
    ) -> tuple[ttnn.Tensor, ...]:
        """Build per-layer page tables where selected verify lanes alias prompt lanes."""
        if num_prompts <= 0:
            return self._get_page_tables()

        _ = self._get_page_tables()
        if self.base_page_table_host is None:
            raise RuntimeError("Base page table host tensor is not initialized.")
        base_page_table = self.base_page_table_host.to(torch.int32)
        alias_page_table = _build_verify_alias_page_table_host(
            base_page_table=base_page_table,
            num_prompts=num_prompts,
            verify_offset=verify_offset,
            prompt_indices=prompt_indices,
            interleaved=interleaved,
        )

        aliased_tt = MLA2D.create_page_table(
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            page_table=alias_page_table,
            batch_size=self.batch_size_per_row,
        )
        out = tuple(ttnn.clone(aliased_tt) for _ in range(self.hf_config.num_hidden_layers))
        self._update_decode_page_table_alias_masks(alias_page_table)
        ttnn.deallocate(aliased_tt)
        return out

    def _mtp_predict_logits(
        self,
        hidden_states: torch.Tensor | ttnn.Tensor,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        page_table: ttnn.Tensor | None = None,
        use_trace: bool | None = None,
    ) -> torch.Tensor:
        assert self.enable_mtp, "MTP path requested while MTP is disabled"
        assert tokens_step.dim() == 1, "tokens_step must be [B]"
        assert positions.dim() == 1, "positions must be [B]"

        if use_trace is None:
            use_trace = self.enable_trace
        if use_trace:
            return self._mtp_predict_logits_traced(hidden_states, tokens_step, positions, page_table)

        hidden_from_host = not isinstance(hidden_states, ttnn.Tensor)
        if hidden_from_host:
            assert hidden_states.dim() == 2, "hidden_states must be [B, H]"
            tt_hidden = ttnn.from_torch(
                hidden_states.view(1, 1, hidden_states.shape[0], hidden_states.shape[1]).to(torch.bfloat16),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(-2, -1),
                    mesh_shape=tuple(self.mesh_device.shape),
                ),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            tt_hidden = hidden_states

        tt_tokens = self._tt_from_tokens_step(tokens_step)
        tt_positions = ttnn.from_torch(
            positions.to(torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.int32,
        )

        rot_idxs = self.rope_setup.get_rot_idxs(positions)
        rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(rot_idxs)
        mtp_page_table = page_table if page_table is not None else self._get_mtp_page_table()
        logits_tt = RowBatchedModel.forward_mtp_decode(
            hidden_states=tt_hidden,
            token_ids=tt_tokens,
            position_idxs=tt_positions,
            cfg=self.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_table=mtp_page_table,
        )
        logits = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )

        if hidden_from_host:
            ttnn.deallocate(tt_hidden)
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(tt_positions)
        ttnn.deallocate(rot_idxs)
        self._deallocate_rope_tensors(rope_tensors)
        ttnn.deallocate(logits_tt)

        return logits.squeeze(0).squeeze(0)  # [B, V]

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        while logits.dim() > 2 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        return torch.argmax(logits, dim=-1)  # [B]

    @staticmethod
    def _get_sampling_value(value, index: int):
        if isinstance(value, list):
            if not value:
                return None
            if index < len(value):
                return value[index]
            return value[-1]
        return value

    def _sample_greedy_on_host(self, logits: torch.Tensor) -> torch.Tensor:
        return self._sample_greedy(logits)

    def _get_stop_token_ids(self) -> set[int]:
        eos_token_id = self.hf_config.eos_token_id
        if isinstance(eos_token_id, int):
            return {int(eos_token_id)}
        return {int(token_id) for token_id in eos_token_id}

    @staticmethod
    def _deallocate_rope_tensors(rope_tensors: dict[str, ttnn.Tensor] | None) -> None:
        if rope_tensors is None:
            return
        for key in ("cos_matrix", "sin_matrix"):
            tensor = rope_tensors.get(key)
            if tensor is not None:
                ttnn.deallocate(tensor)

    @staticmethod
    def _tensor_buffer_key(tensor: ttnn.Tensor) -> int:
        try:
            return int(tensor.buffer_address())
        except Exception:
            return id(tensor)

    def _release_page_table_tuple(self, page_tables: tuple[ttnn.Tensor, ...] | None) -> None:
        if (
            page_tables is None
            or page_tables is self.page_tables_tt
            or self._mtp_verify_trace_page_tables is page_tables
        ):
            return
        released_keys: set[int] = set()
        for page_table in page_tables:
            key = self._tensor_buffer_key(page_table)
            if key in released_keys:
                continue
            released_keys.add(key)
            ttnn.deallocate(page_table)

    def _get_pad_id(self) -> int:
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
        return int(pad_id)

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
        # Round up to nearest multiple of TILE_SIZE.
        alignment = ttnn.TILE_SIZE
        max_len = ((max_len + alignment - 1) // alignment) * alignment

        pad_id = self._get_pad_id()

        out = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        lengths = torch.zeros((batch_size,), dtype=torch.int32)
        for i, seq in enumerate(tokens_list):
            len_seq = len(seq) if self.prefill_max_tokens is None else min(self.prefill_max_tokens, len(seq))
            out[i, :len_seq] = torch.tensor(seq[:len_seq], dtype=torch.long)
            lengths[i] = len_seq
        return out, lengths

    def _configure_mtp_prompt_layout(
        self,
        tokens_batched: torch.Tensor,
        lengths: torch.Tensor,
        num_of_prompts: int,
        max_new_tokens: int,
        teacher_forcing,
    ) -> _MtpPromptLayout:
        num_of_users = tokens_batched.shape[0]
        use_mtp_path = self.enable_mtp and teacher_forcing is None and max_new_tokens > 1 and not self.profile_decode
        if use_mtp_path and 2 * num_of_prompts > num_of_users:
            logger.warning(
                f"MTP verify batching needs 2x prompt lanes ({2 * num_of_prompts}) but only {num_of_users} are available; "
                "falling back to regular decode path."
            )
            use_mtp_path = False

        prompt_user_ids: torch.Tensor | None = None
        spec_user_ids: torch.Tensor | None = None
        if use_mtp_path and num_of_prompts > 0:
            prompts_per_row = self.batch_size_per_row // 2
            max_prompts_per_batch = prompts_per_row * self.mesh_device.shape[0]
            if num_of_prompts > max_prompts_per_batch:
                logger.warning(
                    f"MTP interleaved layout supports up to {max_prompts_per_batch} prompts, got {num_of_prompts}; "
                    "falling back to regular decode path."
                )
                use_mtp_path = False
            else:
                prompt_user_ids_list: list[int] = []
                spec_user_ids_list: list[int] = []
                for i in range(num_of_prompts):
                    row = i // prompts_per_row
                    col = i % prompts_per_row
                    base = row * self.batch_size_per_row
                    prompt_uid = base + 2 * col
                    spec_uid = prompt_uid + 1
                    prompt_user_ids_list.append(prompt_uid)
                    spec_user_ids_list.append(spec_uid)
                prompt_user_ids = torch.tensor(prompt_user_ids_list, dtype=torch.long)
                spec_user_ids = torch.tensor(spec_user_ids_list, dtype=torch.long)

                tokens_batched_mtp = torch.zeros_like(tokens_batched)
                lengths_mtp = torch.zeros_like(lengths)
                tokens_batched_mtp[prompt_user_ids] = tokens_batched[:num_of_prompts]
                lengths_mtp[prompt_user_ids] = lengths[:num_of_prompts]
                tokens_batched = tokens_batched_mtp
                lengths = lengths_mtp

            if prompt_user_ids is not None and spec_user_ids is not None:
                tokens_batched[spec_user_ids] = tokens_batched[prompt_user_ids]
                lengths[spec_user_ids] = lengths[prompt_user_ids]

        return _MtpPromptLayout(
            use_mtp_path=use_mtp_path,
            tokens_batched=tokens_batched,
            lengths=lengths,
            prompt_user_ids=prompt_user_ids,
            spec_user_ids=spec_user_ids,
        )

    def _bootstrap_mtp_decode_state(
        self,
        last_logits: torch.Tensor,
        lengths: torch.Tensor,
        prefill_last_hidden: list[torch.Tensor | None] | None,
        num_of_prompts: int,
        num_of_users: int,
        prompt_user_ids: torch.Tensor | None,
        spec_user_ids: torch.Tensor | None,
        temp_decode_page_tables: list[tuple[ttnn.Tensor, ...]],
    ) -> _MtpDecodeBootstrap:
        next_tokens_all = self._sample_greedy(last_logits)
        next_tokens = torch.zeros_like(next_tokens_all)
        if prompt_user_ids is not None:
            next_tokens[prompt_user_ids] = next_tokens_all[prompt_user_ids]
        else:
            next_tokens = next_tokens_all

        positions = lengths.clone()
        use_interleaved = prompt_user_ids is not None and spec_user_ids is not None
        verify_offset_default = self.batch_size_per_row // 2
        verify_offset = verify_offset_default if use_interleaved else num_of_prompts
        if not use_interleaved and (verify_offset < num_of_prompts or verify_offset + num_of_prompts > num_of_users):
            raise RuntimeError(
                f"Invalid verify offset {verify_offset} for num_prompts={num_of_prompts}, num_users={num_of_users}"
            )

        decode_page_tables = self._build_mtp_verify_page_tables(
            num_prompts=num_of_prompts,
            verify_offset=verify_offset,
            interleaved=use_interleaved,
        )
        temp_decode_page_tables.append(decode_page_tables)

        spec_tokens: torch.Tensor | None = None
        if prefill_last_hidden is not None:
            hidden_size = int(self.hf_config.hidden_size)
            hidden_tail = torch.zeros((num_of_users, hidden_size), dtype=torch.bfloat16)
            for i, last_hidden in enumerate(prefill_last_hidden):
                if last_hidden is not None:
                    hidden_tail[i] = last_hidden
            positions_tail = lengths.clone()
            bootstrap_mtp_traces = (
                self.enable_trace and self._mtp_predict_trace_id is None and self._mtp_verify_trace_id is None
            )
            spec_logits = self._mtp_predict_logits(
                hidden_states=hidden_tail,
                tokens_step=next_tokens,
                positions=positions_tail,
                use_trace=not bootstrap_mtp_traces,
            )
            self.ccl.reset_sem_counters()
            spec_all = self._sample_greedy(spec_logits)
            if prompt_user_ids is not None:
                spec_tokens = spec_all[prompt_user_ids]
            else:
                spec_tokens = spec_all[:num_of_prompts]

            if bootstrap_mtp_traces:
                mtp_page_table = self._get_mtp_page_table()
                batched_tokens = next_tokens.clone()
                batched_positions = positions.clone()
                if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                    batched_tokens[spec_user_ids] = spec_tokens
                    batched_positions[spec_user_ids] = positions[prompt_user_ids] + 1
                else:
                    batched_tokens[verify_offset : verify_offset + num_of_prompts] = spec_tokens
                    batched_positions[verify_offset : verify_offset + num_of_prompts] = positions[:num_of_prompts] + 1

                self._ensure_mtp_predict_trace_buffers(hidden_tail, next_tokens, positions_tail, mtp_page_table)
                self._ensure_mtp_verify_trace_buffers(batched_tokens, batched_positions, decode_page_tables)
                self._capture_mtp_verify_trace(
                    batched_tokens,
                    batched_positions,
                    batch_size_per_row=self.batch_size_per_row,
                    page_tables=decode_page_tables,
                    compile_run=True,
                )
                self._capture_mtp_predict_trace(
                    hidden_tail,
                    next_tokens,
                    positions_tail,
                    mtp_page_table,
                    compile_run=False,
                )

        return _MtpDecodeBootstrap(
            next_tokens=next_tokens,
            positions=positions,
            spec_tokens=spec_tokens,
            decode_page_tables=decode_page_tables,
        )

    def _run_mtp_decode_loop(
        self,
        num_of_prompts: int,
        num_of_users: int,
        max_new_tokens: int,
        prompt_user_ids: torch.Tensor | None,
        spec_user_ids: torch.Tensor | None,
        next_tokens: torch.Tensor,
        positions: torch.Tensor,
        spec_tokens: torch.Tensor | None,
        decode_page_tables: tuple[ttnn.Tensor, ...] | None,
        generations: List[List[int]],
        finished: list[bool] | None,
        stop_token_ids: set[int],
        notify_finished,
        profiler: BenchmarkProfiler,
        token_trace: bool,
        early_print_first_user: bool,
    ) -> _MtpDecodeLoopResult:
        use_interleaved = prompt_user_ids is not None and spec_user_ids is not None
        verify_offset_default = self.batch_size_per_row // 2
        verify_offset = verify_offset_default if use_interleaved else num_of_prompts
        skip_accept_decode = True
        generated_counts = torch.zeros((num_of_prompts,), dtype=torch.int32)
        if num_of_prompts > 0:
            generated_counts += 1

        if spec_tokens is None:
            raise RuntimeError("MTP spec tokens were not initialized; prefill hidden states missing.")
        if decode_page_tables is None:
            raise RuntimeError("MTP verify page tables were not initialized.")

        total_accepts = 0
        total_verifies = 0
        skipped_decode_tokens = 0
        accepted_committed_second_token = 0
        decode_step_idx = 0
        decode_forward_passes = 0
        decode_step_active_masks: List[List[bool]] = []
        decode_step_user_tokens: List[List[int]] = []

        while any(
            generated_counts[i] < max_new_tokens and (finished is None or not finished[i])
            for i in range(num_of_prompts)
        ):
            step_active_mask = [
                generated_counts[i] < max_new_tokens and (finished is None or not finished[i])
                for i in range(num_of_prompts)
            ]
            step_user_tokens = [0 for _ in range(num_of_prompts)]
            batched_tokens = next_tokens.clone()
            batched_positions = positions.clone()
            if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                batched_tokens[spec_user_ids] = spec_tokens
                batched_positions[spec_user_ids] = positions[prompt_user_ids] + 1
            else:
                batched_tokens[verify_offset : verify_offset + num_of_prompts] = spec_tokens
                batched_positions[verify_offset : verify_offset + num_of_prompts] = positions[:num_of_prompts] + 1

            logger.info(f"Decoding step {decode_step_idx} for {num_of_prompts} user(s)...")
            trace_replay_step = (
                self.enable_trace and self._mtp_verify_trace_id is not None and self._mtp_predict_trace_id is not None
            )
            profiler.start(f"decode_time_{decode_step_idx}")
            if trace_replay_step:
                profiler.start(f"trace_execution_{decode_step_idx}")
            if self.enable_trace:
                logits_2b, hidden_2b = self._mtp_verify_decode_traced(
                    tokens_step=batched_tokens,
                    positions=batched_positions,
                    batch_size_per_row=self.batch_size_per_row,
                    page_tables=decode_page_tables,
                )
            else:
                logits_2b_tt, hidden_2b_tt = self._decode_step_tt(
                    tokens_step=batched_tokens,
                    positions=batched_positions,
                    batch_size_per_row=self.batch_size_per_row,
                    page_tables=decode_page_tables,
                    return_hidden=True,
                )
                logits_2b = ttnn.to_torch(
                    logits_2b_tt,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                    ),
                )
                ttnn.deallocate(logits_2b_tt)
            profiler.end(f"decode_time_{decode_step_idx}")
            decode_step_idx += 1
            decode_forward_passes += 1
            self.ccl.reset_sem_counters()

            logits_2b = logits_2b.squeeze(0).squeeze(0)
            pred_all = self._sample_greedy(logits_2b)
            if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                pred_next = pred_all[prompt_user_ids]
                pred_after_spec = pred_all[spec_user_ids]
            else:
                pred_next = pred_all[:num_of_prompts]
                pred_after_spec = pred_all[verify_offset : verify_offset + num_of_prompts]
            accepted_prompt_mask = torch.zeros((num_of_prompts,), dtype=torch.bool)

            for i in range(num_of_prompts):
                if not step_active_mask[i]:
                    continue

                next_value = int(pred_next[i].item())
                total_verifies += 1
                accepted = next_value == int(spec_tokens[i].item())
                if use_interleaved and prompt_user_ids is not None:
                    prompt_uid = int(prompt_user_ids[i].item())
                else:
                    prompt_uid = i
                next_tokens[prompt_uid] = next_value
                positions[prompt_uid] = positions[prompt_uid] + 1
                generated_counts[i] += 1
                if accepted:
                    total_accepts += 1
                if finished is not None and next_value in stop_token_ids:
                    finished[i] = True
                    notify_finished(i)
                    continue

                generations[i].append(next_value)
                step_user_tokens[i] += 1
                if token_trace:
                    logger.info(f"TOKTRACE prompt={i} gen_idx={int(generated_counts[i].item())-1} token={next_value}")
                if early_print_first_user and i == 0:
                    if self.tokenizer is not None:
                        print(self.tokenizer.decode(next_value, skip_special_tokens=True), end="", flush=True)
                    else:
                        print(f"{next_value} ", end="", flush=True)

                if accepted and generated_counts[i] < max_new_tokens:
                    if skip_accept_decode:
                        accepted_committed_second_token += 1
                        next_after_spec_value = int(pred_after_spec[i].item())
                        next_tokens[prompt_uid] = next_after_spec_value
                        positions[prompt_uid] = positions[prompt_uid] + 1
                        generated_counts[i] += 1
                        if finished is not None and next_after_spec_value in stop_token_ids:
                            finished[i] = True
                            notify_finished(i)
                            continue
                        generations[i].append(next_after_spec_value)
                        step_user_tokens[i] += 1
                        if token_trace:
                            logger.info(
                                f"TOKTRACE prompt={i} gen_idx={int(generated_counts[i].item())-1} token={next_after_spec_value}"
                            )
                        if early_print_first_user and i == 0:
                            if self.tokenizer is not None:
                                print(
                                    self.tokenizer.decode(next_after_spec_value, skip_special_tokens=True),
                                    end="",
                                    flush=True,
                                )
                            else:
                                print(f"{next_after_spec_value} ", end="", flush=True)
                    if finished is None or not finished[i]:
                        accepted_prompt_mask[i] = True

            accepted_indices = [
                i
                for i in range(num_of_prompts)
                if accepted_prompt_mask[i]
                and generated_counts[i] < max_new_tokens
                and (finished is None or not finished[i])
            ]
            if skip_accept_decode and accepted_indices:
                skipped_decode_tokens += len(accepted_indices)

            non_prompt_mask = torch.ones((num_of_users,), dtype=torch.bool)
            if use_interleaved and prompt_user_ids is not None:
                non_prompt_mask[prompt_user_ids] = False
            else:
                non_prompt_mask[:num_of_prompts] = False
            next_tokens[non_prompt_mask] = pred_all[non_prompt_mask]
            positions[non_prompt_mask] = positions[non_prompt_mask] + 1

            tokens_for_spec = next_tokens.clone()
            positions_for_spec = positions.clone()

            hidden_for_spec = hidden_2b if self.enable_trace else hidden_2b_tt
            if self.enable_trace or (skip_accept_decode and accepted_indices):
                hidden_2b_host = hidden_2b if self.enable_trace else self._hidden_tt_to_host_for_mtp(hidden_2b_tt)
                hidden_for_spec = hidden_2b_host.clone()
                accept_mask = accepted_prompt_mask.to(torch.bool)
                if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                    max_idx = int(torch.max(spec_user_ids).item()) if spec_user_ids.numel() > 0 else -1
                    if hidden_2b_host.shape[0] <= max_idx:
                        raise RuntimeError(
                            "Hidden batch smaller than max spec user id: " f"{hidden_2b_host.shape[0]} <= {max_idx}"
                        )
                    hidden_verify = hidden_2b_host[spec_user_ids]
                    if accept_mask.any():
                        accept_prompt_ids = prompt_user_ids[accept_mask]
                        hidden_for_spec[accept_prompt_ids] = hidden_verify[accept_mask]
                else:
                    if hidden_2b_host.shape[0] < verify_offset + num_of_prompts:
                        raise RuntimeError(
                            "Hidden batch smaller than verify offset + num_prompts: "
                            f"{hidden_2b_host.shape[0]} < {verify_offset + num_of_prompts}"
                        )
                    hidden_verify = hidden_2b_host[verify_offset : verify_offset + num_of_prompts]
                    if accept_mask.any():
                        hidden_for_spec[:num_of_prompts][accept_mask] = hidden_verify[accept_mask]

            spec_logits_full = self._mtp_predict_logits(
                hidden_states=hidden_for_spec,
                tokens_step=tokens_for_spec,
                positions=positions_for_spec,
            )
            self.ccl.reset_sem_counters()
            spec_all = self._sample_greedy(spec_logits_full)
            if not self.enable_trace:
                ttnn.deallocate(hidden_2b_tt)
            if trace_replay_step:
                profiler.end(f"trace_execution_{decode_step_idx - 1}")
                logger.info(
                    f"Trace execution t/s/user @ token {decode_step_idx - 1}: "
                    f"{1/profiler.get_duration(f'trace_execution_{decode_step_idx - 1}')}"
                )

            if use_interleaved and prompt_user_ids is not None:
                spec_tokens = spec_all[prompt_user_ids]
            else:
                spec_tokens = spec_all[:num_of_prompts]
            decode_step_active_masks.append(step_active_mask)
            decode_step_user_tokens.append(step_user_tokens)

        mtp_accept_rate = total_accepts / total_verifies if total_verifies > 0 else 0.0
        if total_verifies > 0:
            logger.info(f"MTP accept rate: {total_accepts}/{total_verifies} = {mtp_accept_rate:.3f}")
            logger.info(
                "MTP skip-path summary: skipped_decode_tokens={} "
                "accepted_committed_second_token={}".format(
                    skipped_decode_tokens,
                    accepted_committed_second_token,
                )
            )

        return _MtpDecodeLoopResult(
            mtp_accept_rate=mtp_accept_rate,
            mtp_accepts=total_accepts,
            mtp_verifies=total_verifies,
            decode_step_idx=decode_step_idx,
            decode_forward_passes=decode_forward_passes,
            decode_step_active_masks=decode_step_active_masks,
            decode_step_user_tokens=decode_step_user_tokens,
        )

    def _sample_on_host(self, logits: torch.Tensor, start_user_idx: int = 0) -> torch.Tensor | int:
        """Sample on host using top-k/top-p/temperature from sampling_params."""
        if self.sampling_params is None:
            return torch.argmax(logits, dim=-1)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        elif logits.ndim > 2:
            # Normalize to [batch, vocab] so each sampled row maps to one user lane.
            logits = logits.reshape(-1, logits.shape[-1])

        sampled_tokens = torch.argmax(logits, dim=-1)
        batch_size = logits.shape[0]

        for row_idx in range(batch_size):
            user_idx = start_user_idx + row_idx
            temperature = self._get_sampling_value(self.sampling_params.temperature, user_idx)
            top_k = self._get_sampling_value(self.sampling_params.top_k, user_idx)
            top_p = self._get_sampling_value(self.sampling_params.top_p, user_idx)
            seed = self._get_sampling_value(getattr(self.sampling_params, "seed", None), user_idx)

            temperature = float(temperature) if temperature is not None else 1.0
            top_k = int(top_k) if top_k is not None else 0
            top_p = float(top_p) if top_p is not None else 1.0

            if temperature <= 0:
                continue

            scores = logits[row_idx : row_idx + 1] / temperature

            if top_k > 0:
                top_k = min(top_k, scores.shape[-1])
                kth_values = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
                scores = scores.masked_fill(scores < kth_values, float("-inf"))

            if 0.0 < top_p < 1.0:
                sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_scores, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                sorted_scores = sorted_scores.masked_fill(remove_mask, float("-inf"))

                filtered_scores = torch.full_like(scores, float("-inf"))
                filtered_scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
                scores = filtered_scores

            probs = torch.softmax(scores, dim=-1)
            row_sum = probs.sum(dim=-1)
            valid_row = torch.isfinite(probs).all(dim=-1) & torch.isfinite(row_sum) & (row_sum > 0)
            if not bool(valid_row.all().item()):
                continue

            generator: torch.Generator | None = None
            if seed is not None:
                generator = torch.Generator(device=probs.device).manual_seed(int(seed))

            if generator is None:
                sampled_tokens[row_idx] = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                sampled_tokens[row_idx] = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

        return sampled_tokens

    def generate(
        self,
        prompts: Iterable[str],
        max_new_tokens: int = 32,
        teacher_forcing=None,
        early_print_first_user: bool = True,
        repeat_batches: int = 1,
        pre_tokenized: List[List[int]] | None = None,
        stop_at_eos: bool = True,
        on_user_finished=None,
    ) -> Tuple[List[List[int]], dict]:
        """Generate tokens for the given prompts using greedy decode by default.

        early_print_first_user: If True, prints generated tokens for the first user
                                at each step. Better for demo visibility.

        repeat_batches: Number of times to repeat the prefill+decode pass. Only the
                        last pass's tokens are returned; timings aggregate.

        stop_at_eos: Defaults to True. When enabled and teacher_forcing is not
                     active, stop recording output tokens for a user after EOS.
                     When teacher_forcing is active, EOS-based early stopping is
                     disabled and all teacher-forced tokens are recorded.
        on_user_finished: Optional callback with signature (user_index, output_tokens).

        Returns: (list of generated token id lists for the provided prompts (order preserved), statistics dictionary)
        """
        if teacher_forcing is not None and self.sample_on_device:
            raise ValueError("teacher_forcing is not supported when sample_on_device is True")
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
        decode_steps_for_stats = 0
        decode_forward_passes = 0
        decode_step_active_masks_for_stats: List[List[bool]] = []
        decode_step_user_tokens_for_stats: List[List[int]] = []
        mtp_accept_rate = None
        mtp_accepts = None
        mtp_verifies = None
        temp_decode_page_tables: list[tuple[ttnn.Tensor, ...]] = []
        token_trace = bool(int(os.getenv("DEEPSEEK_TOKEN_TRACE", "0")))
        mtp_layout = self._configure_mtp_prompt_layout(
            tokens_batched=tokens_batched,
            lengths=lengths,
            num_of_prompts=num_of_prompts,
            max_new_tokens=max_new_tokens,
            teacher_forcing=teacher_forcing,
        )
        use_mtp_path = mtp_layout.use_mtp_path
        tokens_batched = mtp_layout.tokens_batched
        lengths = mtp_layout.lengths
        prompt_user_ids = mtp_layout.prompt_user_ids
        spec_user_ids = mtp_layout.spec_user_ids
        num_of_users = tokens_batched.shape[0]

        # Run one or more prefill+decode batches
        stop_token_ids = self._get_stop_token_ids() if stop_at_eos and teacher_forcing is None else set()
        for batch_idx in range(repeat_batches):
            if self.sample_on_device:
                # reset sampling state for each repeat batch, o/p tokens will be different for each repeat batch
                assert self.sampling_params is not None, "sampling_params must be set when sampling on device"
                if self.enable_trace and batch_idx > 0:
                    # Previous batch deallocates trace-owned sampling output tensors.
                    # Reset trace so the next batch captures fresh outputs.
                    self.sampling_generator.reset_trace()
                self._reset_sampling_state(
                    self.sampling_params,
                    self.batch_size,
                    self.batch_size_per_row,
                )
            # Reset teacher-forcing state per batch.
            if teacher_forcing is not None:
                teacher_forcing.reset()

            # Prefill (can be skipped for decode-only profiling)
            num_of_users = tokens_batched.shape[0]
            prefill_last_hidden = [None] * num_of_users if use_mtp_path else None
            if self.profile_decode:
                logger.info("Skipping prefill (profile_decode=True) - using random tokens for decode profiling")
                # Generate random starting token IDs directly instead of
                # allocating a full [num_users, vocab_size] logits tensor.
                vocab_size = int(getattr(self.hf_config, "vocab_size", 32768))
                next_tokens_override = torch.randint(0, vocab_size, (num_of_users,))
                # Set lengths to 0 so positions start at 0
                lengths = torch.zeros((num_of_users,), dtype=torch.int32)
                last_logits = None
            else:
                if self.signpost:
                    signpost(header="prefill")
                profiler.start("inference_prefill")
                skipped_prefill_users = 0
                prefill_tokens = [] if not use_mtp_path else None
                last_logits = [] if use_mtp_path else None
                for user_id in range(num_of_users):
                    if lengths[user_id] == 0:
                        skipped_prefill_users += 1
                        if use_mtp_path:
                            assert last_logits is not None
                            last_logits.append(torch.zeros(self.hf_config.vocab_size))
                        else:
                            assert prefill_tokens is not None
                            pad_token = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
                            prefill_tokens.append(torch.tensor(int(pad_token), dtype=torch.int64))
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
                    if use_mtp_path:
                        assert last_logits is not None
                        user_out, last_hidden = self._prefill(
                            tokens_batched[user_id],
                            user_id=user_id,
                            prompt_len=prompt_len,
                            return_last_hidden=True,
                        )
                        if prefill_last_hidden is not None:
                            prefill_last_hidden[user_id] = last_hidden
                        # Use logits at the actual last prompt token.
                        last_logits.append(user_out[0, 0, prompt_len - 1, :])
                    else:
                        assert prefill_tokens is not None
                        prefill_logits = self._prefill(
                            tokens_batched[user_id], user_id=user_id, sample_on_device=self.sample_on_device
                        )
                        assert prefill_logits is not None
                        if self.sample_on_device:
                            prefill_logits = self._slice_last_token_logits(
                                prefill_logits, prompt_len, expand_to_batch=True
                            )
                            prefill_logits_sampled_device = self._sample_tokens_device(
                                prefill_logits, user_slots=[user_id]
                            )
                            prefill_logits_sampled_host = self._tokens_from_device(
                                prefill_logits_sampled_device, self.mesh_device, batch_size_per_row=1
                            )
                            pred_token = int(prefill_logits_sampled_host[0].item())
                            ttnn.deallocate(prefill_logits)
                            ttnn.deallocate(prefill_logits_sampled_device)
                        else:
                            assert isinstance(
                                prefill_logits, torch.Tensor
                            ), "prefill_logits should be a torch.Tensor on host"
                            last_token_logits = prefill_logits[0, 0, max(prompt_len - 1, 0), :]
                            pred_token = int(
                                self._sample_on_host(last_token_logits.unsqueeze(0), start_user_idx=user_id).item()
                            )
                        prefill_tokens.append(torch.tensor(pred_token, dtype=torch.int64))
                    self.ccl.reset_sem_counters()
                if use_mtp_path:
                    assert last_logits is not None
                    last_logits = torch.stack(last_logits)
                else:
                    assert prefill_tokens is not None
                    prefill_tokens = torch.stack(prefill_tokens)
                if skipped_prefill_users > 0:
                    logger.info(f"Skipped prefill for {skipped_prefill_users} user(s) with empty prompts.")
                profiler.end("inference_prefill")
                if self.signpost:
                    signpost(header="prefill")

            if not self.profile_decode:
                if use_mtp_path:
                    assert last_logits is not None and len(last_logits) == num_of_users
                else:
                    assert prefill_tokens is not None and len(prefill_tokens) == num_of_users

            logger.info(
                f"Finished prefill for all users..."
                if not self.profile_decode
                else "Skipped prefill, starting decode..."
            )

            generations: List[List[int]] = [[] for _ in range(num_of_prompts)]
            finished = [False] * num_of_prompts if stop_token_ids else None
            notified = [False] * num_of_prompts if stop_token_ids else None
            callback = on_user_finished if batch_idx == repeat_batches - 1 else None

            def notify_finished(user_idx: int) -> None:
                if callback is None or notified is None or notified[user_idx]:
                    return
                try:
                    callback(user_idx, list(generations[user_idx]))
                except Exception as exc:
                    logger.warning(f"on_user_finished callback failed for user {user_idx}: {exc}")
                notified[user_idx] = True

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
                    positions = lengths.clone()
                else:
                    if use_mtp_path:
                        assert last_logits is not None
                        mtp_bootstrap = self._bootstrap_mtp_decode_state(
                            last_logits=last_logits,
                            lengths=lengths,
                            prefill_last_hidden=prefill_last_hidden,
                            num_of_prompts=num_of_prompts,
                            num_of_users=num_of_users,
                            prompt_user_ids=prompt_user_ids,
                            spec_user_ids=spec_user_ids,
                            temp_decode_page_tables=temp_decode_page_tables,
                        )
                        next_tokens = mtp_bootstrap.next_tokens
                        positions = mtp_bootstrap.positions
                        spec_tokens = mtp_bootstrap.spec_tokens
                        decode_page_tables = mtp_bootstrap.decode_page_tables
                    else:
                        assert prefill_tokens is not None
                        next_tokens = prefill_tokens
                        positions = lengths.clone()
                if teacher_forcing is not None:
                    # Record user-0 prediction for accuracy, but force teacher token for alignment.
                    tf_idx = int(prompt_user_ids[0].item()) if (prompt_user_ids is not None) else 0
                    forced0 = teacher_forcing.collect_predicted_tokens(int(next_tokens[tf_idx].item()))
                    next_tokens[tf_idx] = int(forced0)

                # Record token 0
                for i in range(num_of_prompts):
                    prompt_uid = int(prompt_user_ids[i].item()) if prompt_user_ids is not None else i
                    token_value = int(next_tokens[prompt_uid].item())
                    if finished is not None and token_value in stop_token_ids:
                        finished[i] = True
                        notify_finished(i)
                        continue
                    generations[i].append(token_value)
                    if token_trace:
                        logger.info(f"TOKTRACE prompt={i} gen_idx=0 token={token_value}")
                    if early_print_first_user and i == 0:
                        if self.tokenizer is not None:
                            print(self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True)
                        else:
                            print(f"{token_value} ", end="", flush=True)

                # Generate remaining tokens with decode (each decode call produces the next token)
                profiler.start("inference_decode")
                pred_tokens_device: ttnn.Tensor | None = None
                decode_step_idx = 0
                decode_step_active_masks: List[List[bool]] = []
                decode_step_user_tokens: List[List[int]] = []
                if use_mtp_path:
                    mtp_decode_result = self._run_mtp_decode_loop(
                        num_of_prompts=num_of_prompts,
                        num_of_users=num_of_users,
                        max_new_tokens=max_new_tokens,
                        prompt_user_ids=prompt_user_ids,
                        spec_user_ids=spec_user_ids,
                        next_tokens=next_tokens,
                        positions=positions,
                        spec_tokens=spec_tokens,
                        decode_page_tables=decode_page_tables,
                        generations=generations,
                        finished=finished,
                        stop_token_ids=stop_token_ids,
                        notify_finished=notify_finished,
                        profiler=profiler,
                        token_trace=token_trace,
                        early_print_first_user=early_print_first_user,
                    )
                    mtp_accept_rate = mtp_decode_result.mtp_accept_rate
                    mtp_accepts = mtp_decode_result.mtp_accepts
                    mtp_verifies = mtp_decode_result.mtp_verifies
                    decode_step_idx = mtp_decode_result.decode_step_idx
                    decode_forward_passes += mtp_decode_result.decode_forward_passes
                    decode_step_active_masks = mtp_decode_result.decode_step_active_masks
                    decode_step_user_tokens = mtp_decode_result.decode_step_user_tokens
                else:
                    decode_steps = max_new_tokens - 1
                    for gen_idx in range(decode_steps):
                        step_active_mask = [finished is None or not finished[i] for i in range(num_of_prompts)]
                        if not any(step_active_mask):
                            break
                        logger.info(f"Decoding step {gen_idx} for {num_of_prompts} user(s)...")
                        profiler.start(f"decode_time_{gen_idx}")
                        decode_logits = self.decode_forward(
                            tokens=next_tokens,
                            start_pos=positions,
                            profiler=profiler,
                            gen_idx=gen_idx,
                            enable_trace=self.enable_trace,
                            sample_on_device=self.sample_on_device,
                        )
                        profiler.end(f"decode_time_{gen_idx}")
                        decode_step_idx = gen_idx + 1
                        decode_forward_passes += 1
                        self.ccl.reset_sem_counters()
                        if self.sample_on_device:
                            pred_tokens_device = self._sample_tokens_device(
                                decode_logits, enable_trace=self.enable_trace
                            )
                            pred_tokens = self._tokens_from_device(
                                pred_tokens_device, self.mesh_device, batch_size_per_row=self.batch_size_per_row
                            )
                            if not self.enable_trace:
                                ttnn.deallocate(decode_logits)
                                ttnn.deallocate(pred_tokens_device)
                        else:
                            pred_tokens = self._sample_on_host(decode_logits)
                        if teacher_forcing is not None:
                            # Record user-0 prediction for accuracy, then force teacher token.
                            forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                            pred_tokens[0] = int(forced)
                        next_tokens = pred_tokens
                        positions += 1

                        step_user_tokens = [0 for _ in range(num_of_prompts)]
                        for i in range(num_of_prompts):
                            if finished is not None and finished[i]:
                                continue
                            token_value = int(next_tokens[i].item())
                            if finished is not None and token_value in stop_token_ids:
                                finished[i] = True
                                notify_finished(i)
                                continue
                            generations[i].append(token_value)
                            step_user_tokens[i] += 1
                            if token_trace:
                                logger.info(
                                    f"TOKTRACE prompt={i} gen_idx={len(generations[i]) - 1} token={token_value}"
                                )
                            if early_print_first_user and i == 0:
                                if self.tokenizer is not None:
                                    print(
                                        self.tokenizer.decode(token_value, skip_special_tokens=True), end="", flush=True
                                    )
                                else:
                                    print(f"{token_value} ", end="", flush=True)
                        decode_step_active_masks.append(step_active_mask)
                        decode_step_user_tokens.append(step_user_tokens)

                # Trace path: deallocate once after replay loop completes.
                if self.sample_on_device and self.enable_trace and pred_tokens_device is not None:
                    ttnn.deallocate(pred_tokens_device)
                profiler.end("inference_decode")
                decode_steps_for_stats = decode_step_idx
                decode_step_active_masks_for_stats = decode_step_active_masks
                decode_step_user_tokens_for_stats = decode_step_user_tokens

            if early_print_first_user:
                logger.info("\n===== Done =====")

        profiler.end("run")
        # Calculate statistics
        prefill_time = profiler.get_duration("inference_prefill") if not self.profile_decode else 0
        decode_steps = max(decode_steps_for_stats, 0)
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
            per_user_tokens = [0 for _ in range(num_of_prompts)]
            per_user_active_time = [0.0 for _ in range(num_of_prompts)]

            max_steps = min(
                len(decode_times), len(decode_step_active_masks_for_stats), len(decode_step_user_tokens_for_stats)
            )
            for step_idx in range(1, max_steps):
                step_time = decode_times[step_idx]
                step_active_mask = decode_step_active_masks_for_stats[step_idx]
                step_user_tokens = decode_step_user_tokens_for_stats[step_idx]
                for i in range(num_of_prompts):
                    tokens_i = step_user_tokens[i] if i < len(step_user_tokens) else 0
                    per_user_tokens[i] += tokens_i
                    if i < len(step_active_mask) and step_active_mask[i]:
                        per_user_active_time[i] += step_time

            if repeat_batches > 1:
                per_user_tokens = [t * repeat_batches for t in per_user_tokens]

            per_user_tps = [
                (per_user_tokens[i] / per_user_active_time[i]) if per_user_active_time[i] > 0 else 0
                for i in range(num_of_prompts)
            ]
            decode_tokens_per_sec_per_user = (sum(per_user_tps) / num_of_prompts) if num_of_prompts > 0 else 0
            decode_tokens_per_sec = (decode_tokens_per_sec_per_user * num_of_prompts) if total_decode_time > 0 else 0
        elif len(decode_times) == 1:
            total_decode_time = decode_times[0]
            decode_tokens_per_sec_per_user = 0
            decode_tokens_per_sec = 0
        else:
            total_decode_time = 0
            decode_tokens_per_sec_per_user = 0
            decode_tokens_per_sec = 0
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
            "decode_forward_passes": decode_forward_passes,
            "Full demo runtime": profiler.get_duration("run"),
        }
        statistics["mtp_accept_rate"] = mtp_accept_rate
        if mtp_accepts is not None:
            statistics["mtp_accepts"] = mtp_accepts
        if mtp_verifies is not None:
            statistics["mtp_verifies"] = mtp_verifies

        model_params = {
            "mesh_device": f"{self.mesh_device.shape[0]}x{self.mesh_device.shape[1]}",
            "model_path": str(self.model_path),
            "cache_dir": str(self.cache_dir),
            "batch_size": self.batch_size,
            "repeat_batches": repeat_batches,
            "enable_trace": self.enable_trace,
            "sample_on_device": self.sample_on_device,
            "num_hidden_layers": self.hf_config.num_hidden_layers,
            "random_weights": self.random_weights,
            "sampling": {
                "temperature": self.sampling_params.temperature,
                "top_k": self.sampling_params.top_k,
                "top_p": self.sampling_params.top_p,
            },
        }

        for page_tables in temp_decode_page_tables:
            self._release_page_table_tuple(page_tables)

        return generations, statistics, model_params

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
        sample_on_device: bool = False,
        prompt_len: int | None = None,
        return_last_hidden: bool = False,
    ) -> ttnn.Tensor | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run prefill for the full prompt sequence.

        Args:
            tokens: [1, 1, seq_len] padded token sequences
            user_id: user id for the prefill
            local_user_id: local user id for page table lookup

        Returns:
            logits ttnn.Tensor on device if sample_on_device is True, otherwise logits torch.Tensor on host
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

        rot_mats = self.rope_setup.get_rot_mats_table(seq_len)
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
        last_hidden = None
        if self.enable_mtp:
            logits_tt, hidden_tt = RowBatchedModel.forward_prefill(
                x=tt_tokens,
                user_id=user_id,
                cfg=self.model_run_config_prefill,
                rope_tensors=rope_tensors,
                page_tables=page_tables_to_use,
                return_hidden=True,
            )
        else:
            logits_tt = RowBatchedModel.forward_prefill(
                x=tt_tokens,
                user_id=user_id,
                cfg=self.model_run_config_prefill,
                rope_tensors=rope_tensors,
                page_tables=page_tables_to_use,
            )
            hidden_tt = None

        if sample_on_device and return_last_hidden:
            raise ValueError("sample_on_device=True and return_last_hidden=True is not supported.")

        if sample_on_device:
            logits = logits_tt
        else:
            logits = ttnn.to_torch(
                logits_tt,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                ),
            )

        if self.enable_mtp:
            # Prime MTP cache for this user using prompt tokens.
            mtp_page_table = self._get_mtp_page_table()
            full_seq_len = int(hidden_tt.shape[2])
            if full_seq_len > 0 and prompt_len is not None and prompt_len > 1:
                ring_size = max(int(self.mesh_device.shape[0]), 1)
                global_seq_len = int(tokens.shape[2])
                if ring_size > 1:
                    assert (
                        global_seq_len % ring_size == 0
                    ), f"MTP prefill seq_len {global_seq_len} not divisible by ring_size {ring_size}"
                    expected_local_len = global_seq_len // ring_size
                    assert (
                        full_seq_len == expected_local_len
                    ), f"MTP prefill hidden/token length mismatch: hidden_len={full_seq_len} expected={expected_local_len}"
                else:
                    assert (
                        full_seq_len == global_seq_len
                    ), f"MTP prefill hidden/token length mismatch: hidden_len={full_seq_len} tokens_len={global_seq_len}"

                aligned_global_len = global_seq_len
                if aligned_global_len > 0:
                    hidden_shifted = hidden_tt
                    pad_id = self._get_pad_id()
                    tokens_shifted_host = torch.full((1, 1, aligned_global_len), pad_id, dtype=tokens.dtype)
                    if aligned_global_len > 1:
                        tokens_shifted_host[:, :, : aligned_global_len - 1] = tokens[:, :, 1:aligned_global_len]
                    tokens_shifted = ttnn.from_torch(
                        tokens_shifted_host,
                        device=self.mesh_device,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        dtype=ttnn.uint32,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )

                    mtp_rot_mats = self.rope_setup.get_rot_mats_table(aligned_global_len)
                    mtp_rope_tensors = {
                        "cos_matrix": mtp_rot_mats["cos_matrix"],
                        "sin_matrix": mtp_rot_mats["sin_matrix"],
                        "trans_matrix": mtp_rot_mats["trans_matrix"],
                    }

                    mtp_logits_tt = RowBatchedModel.forward_mtp_prefill(
                        hidden_states=hidden_shifted,
                        token_ids=tokens_shifted,
                        user_id=user_id,
                        cfg=self.model_run_config_prefill,
                        rope_tensors=mtp_rope_tensors,
                        page_table=mtp_page_table,
                    )
                    if hidden_shifted is not hidden_tt:
                        ttnn.deallocate(hidden_shifted)
                    ttnn.deallocate(tokens_shifted)
                    self._deallocate_rope_tensors(mtp_rope_tensors)
                    ttnn.deallocate(mtp_logits_tt)
                    self.ccl.reset_sem_counters()

            if return_last_hidden:
                if prompt_len is None:
                    raise ValueError("prompt_len is required when return_last_hidden=True")
                if prompt_len <= 0:
                    last_hidden = torch.zeros((self.hf_config.hidden_size,), dtype=torch.bfloat16)
                else:
                    global_hidden_idx = prompt_len - 1
                    hidden_row_idx = 0
                    hidden_idx = global_hidden_idx
                    if full_seq_len <= 0:
                        raise RuntimeError("Cannot extract last prefill hidden state from an empty hidden tensor.")
                    if self.mesh_device.shape[0] > 1:
                        hidden_row_idx = global_hidden_idx // full_seq_len
                        hidden_idx = global_hidden_idx % full_seq_len
                    hidden_slice = ttnn.slice(
                        hidden_tt, [0, 0, hidden_idx, 0], [1, 1, hidden_idx + 1, hidden_tt.shape[3]]
                    )
                    last_hidden = ttnn.to_torch(
                        hidden_slice,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                        ),
                    )
                    last_hidden = last_hidden.squeeze(0).squeeze(0)
                    if last_hidden.dim() == 3 and last_hidden.shape[1] == 1:
                        last_hidden = last_hidden[:, 0, :]
                    if last_hidden.dim() == 2 and last_hidden.shape[-1] == self.hf_config.hidden_size:
                        if hidden_row_idx >= last_hidden.shape[0]:
                            raise RuntimeError(
                                f"Last hidden row index {hidden_row_idx} out of range for shape {tuple(last_hidden.shape)}"
                            )
                        last_hidden = last_hidden[hidden_row_idx]
                    elif last_hidden.dim() != 1 or last_hidden.shape[0] != self.hf_config.hidden_size:
                        raise RuntimeError(
                            f"Unexpected last_hidden shape after prefill gather: {tuple(last_hidden.shape)}"
                        )
                    ttnn.deallocate(hidden_slice)

            ttnn.deallocate(hidden_tt)

        ttnn.deallocate(tt_tokens)
        self._deallocate_rope_tensors(rope_tensors)
        if not sample_on_device:
            ttnn.deallocate(logits_tt)
        if return_last_hidden:
            return logits, last_hidden
        return logits

    def _slice_last_token_logits(
        self, logits: ttnn.Tensor, prompt_len: int, *, expand_to_batch: bool = False
    ) -> ttnn.Tensor:
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

        if expand_to_batch and last_logits.shape[2] == 1 and self.batch_size_per_row > 1:
            expanded = ttnn.repeat(last_logits, (1, 1, self.batch_size_per_row, 1))
            ttnn.deallocate(last_logits)
            last_logits = expanded

        return last_logits

    def _capture_mtp_verify_trace(
        self,
        init_tokens: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_tables: tuple[ttnn.Tensor, ...],
        compile_run: bool = True,
    ) -> None:
        assert self._mtp_verify_trace_id is None, "MTP verify trace already captured"

        self._ensure_mtp_verify_trace_buffers(init_tokens, positions, page_tables)
        if compile_run:
            logger.info("Running warm-up MTP verify decode step (no trace)...")
            _ = self._decode_step_tt(
                init_tokens,
                positions,
                batch_size_per_row=batch_size_per_row,
                page_tables=page_tables,
                return_hidden=True,
            )
        ttnn.synchronize_device(self.mesh_device)

        self.ccl.reset_sem_counters()
        logger.info("Begin capturing MTP verify decode trace...")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(self._mtp_verify_trace_rot_idxs)
        self._mtp_verify_trace_output = RowBatchedModel.forward_decode(
            x=self._mtp_verify_trace_tokens,
            position_idxs=self._mtp_verify_trace_positions,
            cfg=self.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_tables=self._mtp_verify_trace_page_tables,
            profile_decode=self.profile_decode,
            return_hidden=True,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("MTP verify decode trace capture complete.")
        self._mtp_verify_trace_id = trace_id

    def _ensure_mtp_verify_trace_buffers(
        self,
        init_tokens: torch.Tensor,
        positions: torch.Tensor,
        page_tables: tuple[ttnn.Tensor, ...],
    ) -> None:
        if self._mtp_verify_trace_tokens is None:
            self._mtp_verify_trace_tokens = self._tt_from_tokens_step(init_tokens)
        if self._mtp_verify_trace_positions is None:
            self._mtp_verify_trace_positions = ttnn.from_torch(
                positions,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.int32,
            )
        if self._mtp_verify_trace_rot_idxs is None:
            self._mtp_verify_trace_rot_idxs = self.rope_setup.get_rot_idxs(positions)
        if self._mtp_verify_trace_page_tables is None:
            self._mtp_verify_trace_page_tables = page_tables

    def _mtp_verify_decode_traced(
        self,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_tables: tuple[ttnn.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mtp_verify_trace_id is None:
            self._capture_mtp_verify_trace(tokens_step, positions, batch_size_per_row, page_tables)
        else:
            assert (
                self._mtp_verify_trace_tokens is not None
                and self._mtp_verify_trace_positions is not None
                and self._mtp_verify_trace_rot_idxs is not None
                and self._mtp_verify_trace_output is not None
            )
            host_tokens = ttnn.from_torch(
                tokens_step.view(1, 1, -1).to(torch.int32),
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_tokens, self._mtp_verify_trace_tokens)

            host_positions = ttnn.from_torch(
                positions.to(torch.int32),
                device=None,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.int32,
            )
            ttnn.copy_host_to_device_tensor(host_positions, self._mtp_verify_trace_positions)

            host_rot_idxs = self.rope_setup.get_rot_idxs(positions, on_host=True)
            ttnn.copy_host_to_device_tensor(host_rot_idxs, self._mtp_verify_trace_rot_idxs)

            self.ccl.reset_sem_counters()
            ttnn.execute_trace(self.mesh_device, self._mtp_verify_trace_id, cq_id=0, blocking=True)

        assert self._mtp_verify_trace_output is not None
        logits_tt, hidden_tt = self._mtp_verify_trace_output
        logits = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )
        hidden = self._hidden_tt_to_host_for_mtp(hidden_tt)
        return logits.squeeze(0).squeeze(0), hidden

    def _capture_mtp_predict_trace(
        self,
        hidden_states: torch.Tensor,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        page_table: ttnn.Tensor,
        compile_run: bool = True,
    ) -> None:
        assert self._mtp_predict_trace_id is None, "MTP predictor trace already captured"

        self._ensure_mtp_predict_trace_buffers(hidden_states, tokens_step, positions, page_table)
        if compile_run:
            logger.info("Running warm-up MTP predictor step (no trace)...")
            _ = self._mtp_predict_logits(
                hidden_states=hidden_states,
                tokens_step=tokens_step,
                positions=positions,
                page_table=page_table,
                use_trace=False,
            )
        ttnn.synchronize_device(self.mesh_device)

        self.ccl.reset_sem_counters()
        logger.info("Begin capturing MTP predictor trace...")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(self._mtp_predict_trace_rot_idxs)
        self._mtp_predict_trace_output = RowBatchedModel.forward_mtp_decode(
            hidden_states=self._mtp_predict_trace_hidden,
            token_ids=self._mtp_predict_trace_tokens,
            position_idxs=self._mtp_predict_trace_positions,
            cfg=self.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_table=self._mtp_predict_trace_page_table,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("MTP predictor trace capture complete.")
        self._mtp_predict_trace_id = trace_id

    def _ensure_mtp_predict_trace_buffers(
        self,
        hidden_states: torch.Tensor,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        page_table: ttnn.Tensor,
    ) -> None:
        if self._mtp_predict_trace_hidden is None:
            self._mtp_predict_trace_hidden = self._tt_from_hidden_states_step(hidden_states, device=self.mesh_device)
        if self._mtp_predict_trace_tokens is None:
            self._mtp_predict_trace_tokens = self._tt_from_tokens_step(tokens_step)
        if self._mtp_predict_trace_positions is None:
            self._mtp_predict_trace_positions = ttnn.from_torch(
                positions.to(torch.int32),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.int32,
            )
        if self._mtp_predict_trace_rot_idxs is None:
            self._mtp_predict_trace_rot_idxs = self.rope_setup.get_rot_idxs(positions)
        if self._mtp_predict_trace_page_table is None:
            self._mtp_predict_trace_page_table = page_table

    def _mtp_predict_logits_traced(
        self,
        hidden_states: torch.Tensor | ttnn.Tensor,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        page_table: ttnn.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_host = (
            self._hidden_tt_to_host_for_mtp(hidden_states) if isinstance(hidden_states, ttnn.Tensor) else hidden_states
        )
        hidden_host = self._normalize_hidden_host_for_mtp(hidden_host)

        mtp_page_table = page_table if page_table is not None else self._get_mtp_page_table()
        if self._mtp_predict_trace_id is None:
            self._capture_mtp_predict_trace(hidden_host, tokens_step, positions, mtp_page_table)
        else:
            assert (
                self._mtp_predict_trace_hidden is not None
                and self._mtp_predict_trace_tokens is not None
                and self._mtp_predict_trace_positions is not None
                and self._mtp_predict_trace_rot_idxs is not None
                and self._mtp_predict_trace_output is not None
            )
            host_hidden = self._tt_from_hidden_states_step(hidden_host, device=None)
            ttnn.copy_host_to_device_tensor(host_hidden, self._mtp_predict_trace_hidden)

            host_tokens = ttnn.from_torch(
                tokens_step.view(1, 1, -1).to(torch.int32),
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_tokens, self._mtp_predict_trace_tokens)

            host_positions = ttnn.from_torch(
                positions.to(torch.int32),
                device=None,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.int32,
            )
            ttnn.copy_host_to_device_tensor(host_positions, self._mtp_predict_trace_positions)

            host_rot_idxs = self.rope_setup.get_rot_idxs(positions, on_host=True)
            ttnn.copy_host_to_device_tensor(host_rot_idxs, self._mtp_predict_trace_rot_idxs)

            self.ccl.reset_sem_counters()
            ttnn.execute_trace(self.mesh_device, self._mtp_predict_trace_id, cq_id=0, blocking=True)

        assert self._mtp_predict_trace_output is not None
        logits = ttnn.to_torch(
            self._mtp_predict_trace_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )
        return logits.squeeze(0).squeeze(0)

    def _capture_decode_trace(
        self,
        init_tokens: torch.Tensor,
        positions: torch.Tensor,
        page_tables: torch.Tensor | None = None,
    ) -> None:
        """Allocate persistent inputs, capture trace for one decode iteration, and store trace state."""
        assert self._trace_id is None, "Trace already captured"

        # 1) Warm-up compile run (no trace) to keep compilation out of capture
        logger.info("Running warm-up decode step (no trace)...")
        if self.signpost:
            signpost(header="decode_warmup")
        _ = self._decode_step(init_tokens, positions, page_tables=page_tables, sample_on_device=False)
        ttnn.synchronize_device(self.mesh_device)
        if self.signpost:
            signpost(header="decode_warmup")

        # 2) Allocate persistent device inputs
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

        # 3) Capture decode graph
        self.ccl.reset_sem_counters()
        logger.info("Begin capturing decode trace...")
        if self.signpost:
            signpost(header="decode_trace_capture")
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Only capture the rot_mats generation from rot_idxs (all ttnn ops, no from_torch)
        rope_tensors = self.rope_setup.get_rot_mats_from_rot_idxs(self._trace_rot_idxs)
        self._trace_output = RowBatchedModel.forward_decode(
            x=self._trace_tokens,
            position_idxs=self._trace_positions,
            cfg=self.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_tables=self._trace_page_tables_to_use,
            profile_decode=self.profile_decode,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        if self.signpost:
            signpost(header="decode_trace_capture")
        logger.info("Decode trace capture complete.")
        self._trace_id = trace_id

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        gen_idx: int = 0,
        profiler: BenchmarkProfiler | None = None,
        enable_trace: bool = False,
        page_table: torch.Tensor | None = None,
        kv_cache: None = None,
        read_from_device: bool = None,
        sampling_params: SamplingParams = None,
        sample_on_device: bool = False,
    ) -> ttnn.Tensor | torch.Tensor:
        # vLLM does not pass enable_trace param while initializing the model.
        # vLLM sets it in decode/prefill calls only, so we need to set it here too.
        self.enable_trace = enable_trace

        if not enable_trace:
            return self._decode_step(tokens, start_pos, page_table, sample_on_device)
        else:
            # Capture trace and return trace output
            if self._trace_id is None:
                self._capture_decode_trace(tokens, start_pos, page_table)
                # First call: return the captured run's output
                assert self._trace_output is not None

                if sample_on_device:
                    # return trace output for sampling on device, no need to get logits on host
                    return self._trace_output

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
                and self._trace_page_tables_to_use is not None
            )
            torch_input = tokens.view(1, 1, -1).to(torch.int32)

            if self.signpost:
                signpost(header="decode_execute_trace")

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
                start_pos,
                device=None,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=ttnn.int32,
            )

            ttnn.copy_host_to_device_tensor(host_positions, self._trace_positions)

            host_rot_idxs = self.rope_setup.get_rot_idxs(start_pos, on_host=True)
            ttnn.copy_host_to_device_tensor(host_rot_idxs, self._trace_rot_idxs)

            if page_table is not None:
                page_tables_to_use = self._convert_vllm_page_table_for_batch(page_table, device=None)
                for i, host_page_table in enumerate(page_tables_to_use):
                    ttnn.copy_host_to_device_tensor(host_page_table, self._trace_page_tables_to_use[i])
                self._update_decode_page_table_alias_masks(page_table)

            self.ccl.reset_sem_counters()
            if profiler is not None:
                profiler.start(f"trace_execution_{gen_idx}")
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)
            if profiler is not None:
                profiler.end(f"trace_execution_{gen_idx}")
                logger.info(
                    f"Trace execution t/s/user @ token {gen_idx}: "
                    f"{1/profiler.get_duration(f'trace_execution_{gen_idx}')}"
                )
            assert self._trace_output is not None

            if sample_on_device:
                # return trace output for sampling on device, no need to get logits on host
                if self.signpost:
                    signpost(header="decode_execute_trace_sample_on_device")
                if self.profile_decode:
                    # trigger the profiler to read the device side data each iteration to not miss any data
                    ttnn.ReadDeviceProfiler(self.mesh_device)
                return self._trace_output

            assert sample_on_device == False, "sample_on_device should be False at this point"

            logits = ttnn.to_torch(
                self._trace_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                ),
            )
            if self.signpost:
                signpost(header="decode_execute_trace")
            if self.profile_decode:
                # trigger the profiler to read the device side data each iteration to not miss any data
                ttnn.ReadDeviceProfiler(self.mesh_device)
            return logits.squeeze(0).squeeze(0)

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
