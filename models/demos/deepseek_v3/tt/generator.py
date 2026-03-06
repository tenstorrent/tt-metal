# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from loguru import logger
from safetensors import safe_open
from tracy import signpost
from transformers import AutoConfig

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.common.warmup import WarmupForwardMixin
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.tt.mtp import MTP2D
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig, SavedWeight
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div
from models.demos.deepseek_v3.utils.debug_utils import dump_ttnn_meminfo
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.weight_config import (
    WeightConfigEncoder,
    get_weight_config,
    locked_file,
    try_decode_saved_weight,
    validate_weight_config_paths,
)
from models.perf.benchmarking_utils import BenchmarkProfiler

MAX_SEQ_LEN = 2048


def _debug_mtp_enabled() -> bool:
    return os.getenv("DEBUG_MTP", "0") == "1"


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
        mtp_mode: str = "auto",
        min_mtp_accept_rate: float | None = None,
        mtp_skip_on_accept: bool | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.model_path = str(model_path)
        self.cache_dir = cache_dir

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
        requested_mtp_layers = int(getattr(self.hf_config, "num_nextn_predict_layers", 0))
        has_mtp = (
            (not random_weights)
            and requested_mtp_layers > 0
            and int(getattr(self.hf_config, "num_hidden_layers", 0)) >= 61
        )
        mtp_mode = (mtp_mode or "auto").lower()
        if mtp_mode not in {"auto", "on", "off"}:
            raise ValueError(f"Invalid mtp_mode '{mtp_mode}'. Expected one of: auto, on, off.")
        if mtp_mode == "on":
            if random_weights:
                raise ValueError("MTP cannot be forced on with --random-weights.")
            if not has_mtp:
                raise ValueError(
                    "MTP was forced on, but the model config does not include a valid MTP layer "
                    "(num_nextn_predict_layers=0 or num_hidden_layers < 61)."
                )
            self.enable_mtp = True
        elif mtp_mode == "off":
            self.enable_mtp = False
        else:
            self.enable_mtp = has_mtp

        if not self.enable_mtp and hasattr(self.hf_config, "num_nextn_predict_layers"):
            self.hf_config.num_nextn_predict_layers = 0
        self.mtp_mode = mtp_mode
        self.min_mtp_accept_rate = min_mtp_accept_rate
        self.mtp_skip_on_accept = mtp_skip_on_accept
        logger.info(f"MTP enabled: {self.enable_mtp}")
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
        weight_cache_root = Path(cache_dir) if cache_dir is not None else Path("generated/deepseek_v3")
        weight_cache_root.mkdir(parents=True, exist_ok=True)

        self.model_weight_config = get_weight_config(
            ModuleClass=RowBatchedModel,
            hf_config=self.hf_config,
            weight_cache_path=weight_cache_root,
            mesh_device=self.mesh_device,
            force_recalculate=self.force_recalculate,
            random_weights=self.random_weights,
            model_path=self.model_path,
            single_layer=self.single_layer,
        )

        if self.enable_mtp and self._mtp_cache_needs_refresh(weight_cache_root, self.model_weight_config):
            if not self._has_cached_mtp_weights(self.model_weight_config):
                logger.warning(
                    "MTP is enabled but cached model weights do not include MTP tensors; augmenting cache with MTP tensors."
                )
            else:
                logger.warning(
                    "MTP cached tensors are incompatible with current runtime expectations; refreshing MTP cache."
                )
            try:
                self._augment_mtp_weights_in_cache(weight_cache_root)
            except Exception as e:
                logger.warning(f"Fast MTP cache augmentation failed ({e}); falling back to full cache refresh.")
                self.model_weight_config = get_weight_config(
                    ModuleClass=RowBatchedModel,
                    hf_config=self.hf_config,
                    weight_cache_path=weight_cache_root,
                    mesh_device=self.mesh_device,
                    force_recalculate=True,
                    random_weights=self.random_weights,
                    model_path=self.model_path,
                    single_layer=self.single_layer,
                )
            else:
                self.model_weight_config = get_weight_config(
                    ModuleClass=RowBatchedModel,
                    hf_config=self.hf_config,
                    weight_cache_path=weight_cache_root,
                    mesh_device=self.mesh_device,
                    force_recalculate=False,
                    random_weights=self.random_weights,
                    model_path=self.model_path,
                    single_layer=self.single_layer,
                )

    @staticmethod
    def _has_cached_mtp_weights(weight_config: dict | None) -> bool:
        if not isinstance(weight_config, dict):
            return False
        mtp_cfg = weight_config.get("mtp")
        return isinstance(mtp_cfg, dict) and len(mtp_cfg) > 0

    def _mtp_cache_needs_refresh(self, weight_cache_root: Path, weight_config: dict | None) -> bool:
        if not self._has_cached_mtp_weights(weight_config):
            return True

        try:
            eh_proj_weight = weight_config["mtp"]["eh_proj"]["linear"]["input_tensor_b"]
            if not isinstance(eh_proj_weight, SavedWeight):
                return True

            eh_proj_path = self._get_weight_cache_leaf_path(weight_cache_root) / eh_proj_weight.path
            if not eh_proj_path.exists():
                return True

            eh_proj_tensor = ttnn.load_tensor(str(eh_proj_path))
            expected_shard_width = even_int_div(int(self.hf_config.hidden_size), int(self.mesh_device.shape[1]))
            actual_shard_width = int(eh_proj_tensor.shape[-1])
            if actual_shard_width != expected_shard_width:
                logger.warning(
                    f"MTP eh_proj cached shard width is {actual_shard_width}, expected {expected_shard_width}; cache refresh required."
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to validate cached MTP tensors ({e}); refreshing MTP cache.")
            return True

        return False

    def _get_weight_cache_leaf_path(self, weight_cache_root: Path) -> Path:
        return (
            weight_cache_root
            / f"{self.hf_config.num_hidden_layers}_layers"
            / f"mesh_{self.mesh_device.shape[0]}x{self.mesh_device.shape[1]}"
        )

    def _load_mtp_layer_state_dict(self, skip_tied_weights: bool) -> dict[str, torch.Tensor]:
        if self.model_path is None:
            raise RuntimeError("Cannot augment MTP cache without model_path")

        model_dir = Path(self.model_path)
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise RuntimeError(f"Missing safetensors index file: {index_path}")

        with index_path.open("r") as f:
            weight_map = json.load(f)["weight_map"]

        mtp_layer_idx = int(self.hf_config.num_hidden_layers)
        mtp_prefix = f"model.layers.{mtp_layer_idx}."
        mtp_full_keys = sorted(k for k in weight_map.keys() if k.startswith(mtp_prefix))
        if len(mtp_full_keys) == 0:
            raise RuntimeError(f"No MTP keys found under prefix {mtp_prefix}")

        if skip_tied_weights:
            tied_keys = {
                f"{mtp_prefix}embed_tokens.weight",
                f"{mtp_prefix}shared_head.head.weight",
            }
            mtp_full_keys = [k for k in mtp_full_keys if k not in tied_keys]

        state_dict: dict[str, torch.Tensor] = {}
        keys_per_shard: dict[str, list[str]] = {}
        for key in mtp_full_keys:
            keys_per_shard.setdefault(weight_map[key], []).append(key)

        for shard_file, shard_keys in keys_per_shard.items():
            with safe_open(str(model_dir / shard_file), framework="pt", device="cpu") as shard:
                for key in shard_keys:
                    state_dict[key[len(mtp_prefix) :]] = shard.get_tensor(key)

        if "eh_proj.weight" not in state_dict:
            raise RuntimeError("Loaded MTP state dict is missing required key 'eh_proj.weight'")

        return state_dict

    def _augment_mtp_weights_in_cache(self, weight_cache_root: Path) -> None:
        if self.random_weights:
            raise RuntimeError("Random-weights mode is not supported for MTP cache augmentation")

        cache_leaf = self._get_weight_cache_leaf_path(weight_cache_root)
        config_path = cache_leaf / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Base cache config does not exist at {config_path}")

        with locked_file(config_path, "r", exclusive=False) as f:
            base_weight_cfg = json.load(f, object_hook=try_decode_saved_weight)

        use_tied_weights = bool(getattr(self.hf_config, "tie_word_embeddings", False))
        mtp_state_dict = self._load_mtp_layer_state_dict(skip_tied_weights=use_tied_weights)
        mtp_weight_cfg = MTP2D.convert_weights(
            hf_config=self.hf_config,
            state_dicts=(mtp_state_dict,),
            output_path=cache_leaf / "mtp",
            mesh_device=self.mesh_device,
            reuse_embedding_weight_cfg=base_weight_cfg.get("embedding") if use_tied_weights else None,
            reuse_head_weight_cfg=base_weight_cfg.get("lm_head") if use_tied_weights else None,
        )

        base_weight_cfg["mtp"] = mtp_weight_cfg
        validate_weight_config_paths(cache_leaf, base_weight_cfg)

        with locked_file(config_path, "w", exclusive=True) as f:
            json.dump(base_weight_cfg, f, cls=WeightConfigEncoder)

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
            if self.enable_mtp and (
                "mtp" not in self.model_run_config_decode or self.model_run_config_decode["mtp"] is None
            ):
                logger.warning("Requested MTP path but decode run config has no MTP block; disabling MTP for this run.")
                self.enable_mtp = False
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
        self.base_page_table_host = None
        self.mtp_page_table_host = None
        try:
            if self.mtp_page_table_tt is not None:
                ttnn.deallocate(self.mtp_page_table_tt)
                del self.mtp_page_table_tt
        except Exception as e:
            logger.warning(f"Failed to cleanup MTP page table: {e}")

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
        batch_size_per_row: int,
        page_tables: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run a single decode step and return logits on host as torch tensor [1, 1, B, V]."""
        decode_out = self._decode_step_tt(
            tokens_step=tokens_step,
            positions=positions,
            batch_size_per_row=batch_size_per_row,
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
        else:
            page_tables_to_use = self._get_page_tables()
        # RowBatchedModel forward
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
        if base_page_table.dim() != 2:
            raise RuntimeError(f"Unexpected page table rank for MTP verify aliasing: {tuple(base_page_table.shape)}")

        alias_page_table = base_page_table.clone()
        num_rows = int(alias_page_table.shape[0])
        if num_rows <= 0:
            raise RuntimeError("Page table has zero rows; cannot build MTP verify aliasing.")

        if prompt_indices is None:
            prompt_indices = list(range(num_prompts))
        debug_page_table = _debug_mtp_enabled()
        if debug_page_table:
            logger.info(
                "MTP base page table (shape={}): {}",
                tuple(int(dim) for dim in base_page_table.shape),
                base_page_table.tolist(),
            )
        if interleaved:
            for row in range(1, num_rows, 2):
                alias_page_table[row] = alias_page_table[row - 1]
                if debug_page_table:
                    logger.info(
                        "MTP interleaved alias: row={} -> row={}",
                        row,
                        row - 1,
                    )
        else:
            for i in prompt_indices:
                if i < 0 or i >= num_prompts:
                    continue
                src_row = i % num_rows
                dst_row = (verify_offset + i) % num_rows
                alias_page_table[dst_row] = alias_page_table[src_row]
                if debug_page_table:
                    logger.info(
                        "MTP verify alias: prompt_idx={} src_row={} -> verify_row={} (verify_offset={})",
                        i,
                        src_row,
                        dst_row,
                        verify_offset,
                    )

        if debug_page_table:
            logger.info(
                "MTP aliased page table (shape={}): {}",
                tuple(int(dim) for dim in alias_page_table.shape),
                alias_page_table.tolist(),
            )

        aliased_tt = MLA2D.create_page_table(
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            page_table=alias_page_table,
            batch_size=self.batch_size_per_row,
        )
        out = tuple(ttnn.clone(aliased_tt) for _ in range(self.hf_config.num_hidden_layers))
        ttnn.deallocate(aliased_tt)
        return out

    def _build_mtp_verify_mtp_page_table(
        self,
        num_prompts: int,
        verify_offset: int,
        prompt_indices: List[int] | None = None,
        interleaved: bool = False,
    ) -> ttnn.Tensor:
        """Build an aliased MTP page table where selected verify lanes alias prompt lanes."""
        if num_prompts <= 0:
            return ttnn.clone(self._get_mtp_page_table())

        _ = self._get_mtp_page_table()
        if self.mtp_page_table_host is None:
            raise RuntimeError("MTP base page table host tensor is not initialized.")
        base_page_table = self.mtp_page_table_host.to(torch.int32)
        if base_page_table.dim() != 2:
            raise RuntimeError(f"Unexpected MTP page table rank: {tuple(base_page_table.shape)}")

        alias_page_table = base_page_table.clone()
        num_rows = int(alias_page_table.shape[0])
        if num_rows <= 0:
            raise RuntimeError("MTP page table has zero rows; cannot build verify aliasing.")

        if prompt_indices is None:
            prompt_indices = list(range(num_prompts))
        debug_page_table = _debug_mtp_enabled()
        if debug_page_table:
            logger.info(
                "MTP base MTP page table (shape={}): {}",
                tuple(int(dim) for dim in base_page_table.shape),
                base_page_table.tolist(),
            )
        if interleaved:
            for row in range(1, num_rows, 2):
                alias_page_table[row] = alias_page_table[row - 1]
                if debug_page_table:
                    logger.info(
                        "MTP interleaved MTP alias: row={} -> row={}",
                        row,
                        row - 1,
                    )
        else:
            for i in prompt_indices:
                if i < 0 or i >= num_prompts:
                    continue
                src_row = i % num_rows
                dst_row = (verify_offset + i) % num_rows
                alias_page_table[dst_row] = alias_page_table[src_row]
                if debug_page_table:
                    logger.info(
                        "MTP verify MTP alias: prompt_idx={} src_row={} -> verify_row={} (verify_offset={})",
                        i,
                        src_row,
                        dst_row,
                        verify_offset,
                    )

        if debug_page_table:
            logger.info(
                "MTP aliased MTP page table (shape={}): {}",
                tuple(int(dim) for dim in alias_page_table.shape),
                alias_page_table.tolist(),
            )

        return MLA2D.create_page_table(
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            page_table=alias_page_table,
            batch_size=self.batch_size_per_row,
        )

    def _mtp_predict_logits(
        self,
        hidden_states: torch.Tensor | ttnn.Tensor,
        tokens_step: torch.Tensor,
        positions: torch.Tensor,
        page_table: ttnn.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.enable_mtp, "MTP path requested while MTP is disabled"
        assert tokens_step.dim() == 1, "tokens_step must be [B]"
        assert positions.dim() == 1, "positions must be [B]"

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
        ttnn.deallocate(logits_tt)

        return logits.squeeze(0).squeeze(0)  # [B, V]

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)  # [B]

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
        decode_steps_for_stats = 0
        decode_forward_passes = 0
        decode_step_active_masks_for_stats: List[List[bool]] = []
        decode_step_user_tokens_for_stats: List[List[int]] = []
        num_of_users = tokens_batched.shape[0]
        token_trace = bool(int(os.getenv("DEEPSEEK_TOKEN_TRACE", "0")))
        use_mtp_path = self.enable_mtp and teacher_forcing is None and max_new_tokens > 1 and not self.profile_decode
        if use_mtp_path and 2 * num_of_prompts > num_of_users:
            logger.warning(
                f"MTP verify batching needs 2x prompt lanes ({2 * num_of_prompts}) but only {num_of_users} are available; "
                "falling back to regular decode path."
            )
            use_mtp_path = False
        stats_max_prompts_per_batch = self.batch_size // 2 if use_mtp_path else self.batch_size
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
                stats_max_prompts_per_batch = self.batch_size
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

            # Prefill verify lanes with the same prompt contexts as active lanes so verify decode
            # reads from matching caches instead of empty padded users.
            if prompt_user_ids is not None and spec_user_ids is not None:
                tokens_batched[spec_user_ids] = tokens_batched[prompt_user_ids]
                lengths[spec_user_ids] = lengths[prompt_user_ids]

        # Run one or more prefill+decode batches
        for _ in range(repeat_batches):
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
                    if use_mtp_path:
                        user_out, last_hidden = self._prefill(
                            tokens_batched[user_id],
                            user_id=user_id,
                            prompt_len=prompt_len,
                            return_last_hidden=True,
                        )
                        if prefill_last_hidden is not None:
                            prefill_last_hidden[user_id] = last_hidden
                    else:
                        user_out = self._prefill(tokens_batched[user_id], user_id=user_id)
                    # Use logits at the *actual* last prompt token (not the padded tail).
                    last_logits.append(user_out[0, 0, prompt_len - 1, :])
                    self.ccl.reset_sem_counters()
                last_logits = torch.stack(last_logits)
                profiler.end("inference_prefill")
                if self.signpost:
                    signpost(header="prefill")

            if not self.profile_decode:
                assert len(last_logits) == num_of_users

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
                    next_tokens_all = self._sample_greedy(last_logits)
                    if use_mtp_path and prompt_user_ids is not None:
                        next_tokens = torch.zeros_like(next_tokens_all)
                        next_tokens[prompt_user_ids] = next_tokens_all[prompt_user_ids]
                    else:
                        next_tokens = next_tokens_all
                if teacher_forcing is not None:
                    # Record user-0 prediction for accuracy, but force teacher token for alignment.
                    tf_idx = int(prompt_user_ids[0].item()) if (prompt_user_ids is not None) else 0
                    forced0 = teacher_forcing.collect_predicted_tokens(int(next_tokens[tf_idx].item()))
                    next_tokens[tf_idx] = int(forced0)

                # Positions for the first generated token are the prompt lengths
                positions = lengths.clone()

                spec_tokens = None
                if use_mtp_path and prefill_last_hidden is not None:
                    hidden_size = int(self.hf_config.hidden_size)
                    hidden_tail = torch.zeros((num_of_users, hidden_size), dtype=torch.bfloat16)
                    for i, last_hidden in enumerate(prefill_last_hidden):
                        if last_hidden is not None:
                            hidden_tail[i] = last_hidden
                    bootstrap_pos_delta = int(os.getenv("DEEPSEEK_MTP_BOOTSTRAP_POS_DELTA", "0"))
                    positions_tail = lengths.clone() + bootstrap_pos_delta
                    spec_logits = self._mtp_predict_logits(
                        hidden_states=hidden_tail,
                        tokens_step=next_tokens,
                        positions=positions_tail,
                    )
                    self.ccl.reset_sem_counters()
                    spec_all = self._sample_greedy(spec_logits)
                    if prompt_user_ids is not None:
                        spec_tokens = spec_all[prompt_user_ids]
                    else:
                        spec_tokens = spec_all[:num_of_prompts]

                # Record token 0
                for i in range(num_of_prompts):
                    prompt_uid = int(prompt_user_ids[i].item()) if prompt_user_ids is not None else i
                    token_value = int(next_tokens[prompt_uid].item())
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
                decode_step_idx = 0
                decode_step_active_masks: List[List[bool]] = []
                decode_step_user_tokens: List[List[int]] = []
                mtp_accept_rate = None
                mtp_accepts = None
                debug_mtp = bool(int(os.getenv("DEBUG_MTP", "0")))
                debug_mtp_steps = 3
                debug_mtp_step_idx = 0
                mtp_step_trace = debug_mtp
                if use_mtp_path:
                    if self.mtp_skip_on_accept is None:
                        skip_accept_decode = os.getenv("DEEPSEEK_MTP_DISABLE_SKIP_ACCEPT", "0") != "1"
                    else:
                        skip_accept_decode = bool(self.mtp_skip_on_accept)
                    logger.info(f"MTP skip-on-accept path enabled: {skip_accept_decode}")
                    generated_counts = torch.zeros((num_of_prompts,), dtype=torch.int32)
                    if num_of_prompts > 0:
                        generated_counts += 1
                    use_interleaved = prompt_user_ids is not None and spec_user_ids is not None
                    verify_offset_default = self.batch_size_per_row // 2
                    verify_offset = int(
                        os.getenv(
                            "DEEPSEEK_MTP_VERIFY_OFFSET",
                            str(verify_offset_default if use_interleaved else num_of_prompts),
                        )
                    )
                    if not use_interleaved:
                        if verify_offset < num_of_prompts or verify_offset + num_of_prompts > num_of_users:
                            raise RuntimeError(
                                f"Invalid DEEPSEEK_MTP_VERIFY_OFFSET={verify_offset} for "
                                f"num_prompts={num_of_prompts}, num_users={num_of_users}"
                            )
                    if _debug_mtp_enabled():
                        batch_per_shard = even_int_div(self.batch_size_per_row, self.dp_factor)
                        mesh_rows = int(self.mesh_device.shape[0])
                        logger.info(
                            "MTP lane assignment: num_prompts={} num_users={} verify_offset={} batch_size_per_row={} "
                            "batch_per_shard={} dp_factor={} mesh_shape={}",
                            num_of_prompts,
                            num_of_users,
                            verify_offset,
                            self.batch_size_per_row,
                            batch_per_shard,
                            self.dp_factor,
                            tuple(int(dim) for dim in self.mesh_device.shape),
                        )
                        for i in range(num_of_prompts):
                            if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                                prompt_user = int(prompt_user_ids[i].item())
                                spec_user = int(spec_user_ids[i].item())
                                local_user_in_row = i % (self.batch_size_per_row // 2)
                                dp_col = local_user_in_row // 2
                                batch_row = local_user_in_row % 2
                                logger.info(
                                    "MTP interleaved map: prompt_idx={} local_user_in_row={} dp_col={} batch_row={} "
                                    "prompt_user_id={} spec_user_id={}",
                                    i,
                                    local_user_in_row,
                                    dp_col,
                                    batch_row,
                                    prompt_user,
                                    spec_user,
                                )
                            else:
                                prompt_user = i
                                spec_user = verify_offset + i
                            for lane_label, user_id in (("prompt", prompt_user), ("spec", spec_user)):
                                mesh_row = int(user_id // self.batch_size_per_row) if self.batch_size_per_row else 0
                                batch_row = int(user_id % batch_per_shard) if batch_per_shard else 0
                                logger.info(
                                    "MTP lane: prompt_idx={} lane={} user_id={} mesh_row={}/{} batch_row={}/{}",
                                    i,
                                    lane_label,
                                    user_id,
                                    mesh_row,
                                    mesh_rows,
                                    batch_row,
                                    batch_per_shard,
                                )
                    decode_page_tables = self._build_mtp_verify_page_tables(
                        num_prompts=num_of_prompts, verify_offset=verify_offset, interleaved=use_interleaved
                    )

                    if spec_tokens is None:
                        raise RuntimeError("MTP spec tokens were not initialized; prefill hidden states missing.")
                    if debug_mtp:
                        debug_prompt_uid = (
                            int(prompt_user_ids[0].item()) if use_interleaved and prompt_user_ids is not None else 0
                        )
                        logger.info(
                            "MTP bootstrap: curr[0]={} spec[0]={} pos[0]={}".format(
                                int(next_tokens[debug_prompt_uid].item()),
                                int(spec_tokens[0].item()),
                                int(positions[debug_prompt_uid].item()),
                            )
                        )
                    total_accepts = 0
                    total_verifies = 0
                    total_accepts_alt = 0
                    total_verifies_alt = 0
                    skipped_decode_tokens = 0
                    accepted_committed_second_token = 0

                    while any(generated_counts[i] < max_new_tokens for i in range(num_of_prompts)):
                        step_active_mask = [generated_counts[i] < max_new_tokens for i in range(num_of_prompts)]
                        step_user_tokens = [0 for _ in range(num_of_prompts)]
                        # Pack verification batch into available decode lanes:
                        # [0:N) holds next-token verification, [N:2N) holds next-next-token verification.
                        batched_tokens = next_tokens.clone()
                        batched_positions = positions.clone()
                        if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                            batched_tokens[spec_user_ids] = spec_tokens
                            batched_positions[spec_user_ids] = positions[prompt_user_ids] + 1
                        else:
                            batched_tokens[verify_offset : verify_offset + num_of_prompts] = spec_tokens
                            batched_positions[verify_offset : verify_offset + num_of_prompts] = (
                                positions[:num_of_prompts] + 1
                            )

                        logger.info(f"Decoding step {decode_step_idx} for {num_of_prompts} user(s)...")
                        profiler.start(f"decode_time_{decode_step_idx}")
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
                        positions_before = positions.clone()
                        accepted_prompt_mask = torch.zeros((num_of_prompts,), dtype=torch.bool)

                        for i in range(num_of_prompts):
                            if generated_counts[i] >= max_new_tokens:
                                continue

                            next_value = int(pred_next[i].item())
                            generations[i].append(next_value)
                            step_user_tokens[i] += 1
                            generated_counts[i] += 1
                            if token_trace:
                                logger.info(
                                    f"TOKTRACE prompt={i} gen_idx={int(generated_counts[i].item())-1} token={next_value}"
                                )
                            if early_print_first_user and i == 0:
                                if self.tokenizer is not None:
                                    print(
                                        self.tokenizer.decode(next_value, skip_special_tokens=True),
                                        end="",
                                        flush=True,
                                    )
                                else:
                                    print(f"{next_value} ", end="", flush=True)

                            total_verifies += 1
                            accepted = next_value == int(spec_tokens[i].item())
                            if use_interleaved and prompt_user_ids is not None:
                                prompt_uid = int(prompt_user_ids[i].item())
                            else:
                                prompt_uid = i
                            next_tokens[prompt_uid] = next_value
                            positions[prompt_uid] = positions[prompt_uid] + 1

                            if accepted and generated_counts[i] < max_new_tokens:
                                accepted_prompt_mask[i] = True
                                total_accepts += 1
                                if skip_accept_decode:
                                    accepted_committed_second_token += 1
                                    # True skip: commit the verified next-next token and advance positions by +2.
                                    next_after_spec_value = int(pred_after_spec[i].item())
                                    generations[i].append(next_after_spec_value)
                                    step_user_tokens[i] += 1
                                    generated_counts[i] += 1
                                    if token_trace:
                                        logger.info(
                                            f"TOKTRACE prompt={i} gen_idx={int(generated_counts[i].item())-1} "
                                            f"token={next_after_spec_value}"
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
                                    next_tokens[prompt_uid] = next_after_spec_value
                                    positions[prompt_uid] = positions[prompt_uid] + 1

                        accepted_indices = [
                            i
                            for i in range(num_of_prompts)
                            if accepted_prompt_mask[i] and generated_counts[i] < max_new_tokens
                        ]
                        if mtp_step_trace:
                            for i in range(num_of_prompts):
                                prev_spec_token = int(spec_tokens[i].item())
                                next_pred_token = int(pred_next[i].item())
                                verify_lane_pred_token = int(pred_after_spec[i].item())
                                prev_spec_accepted = next_pred_token == prev_spec_token
                                if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                                    prompt_uid = int(prompt_user_ids[i].item())
                                    spec_uid = int(spec_user_ids[i].item())
                                else:
                                    prompt_uid = i
                                    spec_uid = verify_offset + i
                                next_lane_pos = int(batched_positions[prompt_uid].item())
                                verify_lane_pos = int(batched_positions[spec_uid].item())
                                updated_pos = int(positions[prompt_uid].item())
                                logger.info(
                                    "MTP_STEP user={} step={} "
                                    "next_lane(pred_token={}, pos={}) "
                                    "spec_lane(input_spec_token={}, verify_pred_token={}, pos={}) "
                                    "prev_spec_accepted={} updated_pos={} next_verify_pos={}".format(
                                        i,
                                        decode_step_idx - 1,
                                        next_pred_token,
                                        next_lane_pos,
                                        prev_spec_token,
                                        verify_lane_pred_token,
                                        verify_lane_pos,
                                        prev_spec_accepted,
                                        updated_pos,
                                        updated_pos + 1,
                                    )
                                )
                        if skip_accept_decode and accepted_indices:
                            skipped_decode_tokens += len(accepted_indices)
                            logger.info(
                                f"MTP true-skip committed second token at step {decode_step_idx - 1}: {accepted_indices}"
                            )

                        if debug_mtp and debug_mtp_step_idx < debug_mtp_steps:
                            debug_mtp_step_idx += 1
                            debug_prompt_uid = (
                                int(prompt_user_ids[0].item()) if use_interleaved and prompt_user_ids is not None else 0
                            )
                            logger.info(
                                "MTP debug step {}: curr[0]={} pos[0]={} pred_next[0]={} spec[0]={} pred_after_spec[0]={}".format(
                                    debug_mtp_step_idx,
                                    int(next_tokens[debug_prompt_uid].item()),
                                    int(positions_before[debug_prompt_uid].item()),
                                    int(pred_next[0].item()),
                                    int(spec_tokens[0].item()),
                                    int(pred_after_spec[0].item()),
                                )
                            )

                        if debug_mtp:
                            for i in range(num_of_prompts):
                                if generated_counts[i] >= max_new_tokens:
                                    continue
                                total_verifies_alt += 1
                                if int(pred_after_spec[i].item()) == int(spec_tokens[i].item()):
                                    total_accepts_alt += 1
                            accepted_indices = torch.nonzero(accepted_prompt_mask).flatten().tolist()
                            if accepted_indices:
                                if use_interleaved and prompt_user_ids is not None:
                                    accepted_pos = {
                                        i: int(positions_before[int(prompt_user_ids[i].item())].item())
                                        for i in accepted_indices
                                    }
                                else:
                                    accepted_pos = {i: int(positions_before[i].item()) for i in accepted_indices}
                                accepted_next = {i: int(pred_next[i].item()) for i in accepted_indices}
                                accepted_spec = {i: int(spec_tokens[i].item()) for i in accepted_indices}
                                accepted_next_text = {}
                                accepted_spec_text = {}
                                if self.tokenizer is not None:
                                    for i in accepted_indices:
                                        accepted_next_text[i] = repr(self.tokenizer.decode([accepted_next[i]]))
                                        accepted_spec_text[i] = repr(self.tokenizer.decode([accepted_spec[i]]))
                                logger.info(
                                    "MTP accepts at step {}: idx={} pos={} pred_next={} spec={} pred_next_text={} spec_text={}".format(
                                        decode_step_idx - 1,
                                        accepted_indices,
                                        accepted_pos,
                                        accepted_next,
                                        accepted_spec,
                                        accepted_next_text if self.tokenizer is not None else "n/a",
                                        accepted_spec_text if self.tokenizer is not None else "n/a",
                                    )
                                )
                            rejected_indices = [
                                i
                                for i in range(num_of_prompts)
                                if (not accepted_prompt_mask[i]) and generated_counts[i] < max_new_tokens
                            ]
                            if rejected_indices:
                                if use_interleaved and prompt_user_ids is not None:
                                    rejected_pos = {
                                        i: int(positions_before[int(prompt_user_ids[i].item())].item())
                                        for i in rejected_indices
                                    }
                                else:
                                    rejected_pos = {i: int(positions_before[i].item()) for i in rejected_indices}
                                rejected_next = {i: int(pred_next[i].item()) for i in rejected_indices}
                                rejected_spec = {i: int(spec_tokens[i].item()) for i in rejected_indices}
                                rejected_after = {i: int(pred_after_spec[i].item()) for i in rejected_indices}
                                rejected_next_text = {}
                                rejected_spec_text = {}
                                rejected_after_text = {}
                                if self.tokenizer is not None:
                                    for i in rejected_indices:
                                        rejected_next_text[i] = repr(self.tokenizer.decode([rejected_next[i]]))
                                        rejected_spec_text[i] = repr(self.tokenizer.decode([rejected_spec[i]]))
                                        rejected_after_text[i] = repr(self.tokenizer.decode([rejected_after[i]]))
                                logger.info(
                                    "MTP rejects at step {}: idx={} pos={} pred_next={} spec={} pred_after_spec={} pred_next_text={} spec_text={} pred_after_text={}".format(
                                        decode_step_idx - 1,
                                        rejected_indices,
                                        rejected_pos,
                                        rejected_next,
                                        rejected_spec,
                                        rejected_after,
                                        rejected_next_text if self.tokenizer is not None else "n/a",
                                        rejected_spec_text if self.tokenizer is not None else "n/a",
                                        rejected_after_text if self.tokenizer is not None else "n/a",
                                    )
                                )

                        # Keep non-prompt lanes advancing to preserve tensor shapes.
                        non_prompt_mask = torch.ones((num_of_users,), dtype=torch.bool)
                        if use_interleaved and prompt_user_ids is not None:
                            non_prompt_mask[prompt_user_ids] = False
                        else:
                            non_prompt_mask[:num_of_prompts] = False
                        next_tokens[non_prompt_mask] = pred_all[non_prompt_mask]
                        positions[non_prompt_mask] = positions[non_prompt_mask] + 1

                        tokens_for_spec = next_tokens.clone()
                        positions_for_spec = positions.clone()

                        hidden_for_spec = hidden_2b_tt
                        if skip_accept_decode and accepted_indices:
                            # Select hidden[t] for rejects (prompt lane) and hidden[t+1] for accepts (verify lane).
                            hidden_2b = ttnn.to_torch(
                                hidden_2b_tt,
                                mesh_composer=ttnn.ConcatMesh2dToTensor(
                                    self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                                ),
                            )
                            hidden_2b = hidden_2b.squeeze(0).squeeze(0)
                            hidden_size = int(self.hf_config.hidden_size)
                            if hidden_2b.dim() == 3 and hidden_2b.shape[-1] == hidden_size:
                                hidden_2b = hidden_2b.reshape(-1, hidden_size)
                            if hidden_2b.dim() != 2:
                                raise RuntimeError(
                                    f"Unexpected hidden_2b shape for MTP skip selection: {tuple(hidden_2b.shape)}"
                                )

                            hidden_for_spec = hidden_2b.clone()
                            accept_mask = accepted_prompt_mask.to(torch.bool)
                            if use_interleaved and prompt_user_ids is not None and spec_user_ids is not None:
                                max_idx = int(torch.max(spec_user_ids).item()) if spec_user_ids.numel() > 0 else -1
                                if hidden_2b.shape[0] <= max_idx:
                                    raise RuntimeError(
                                        "Hidden batch smaller than max spec user id: "
                                        f"{hidden_2b.shape[0]} <= {max_idx}"
                                    )
                                hidden_prompt = hidden_2b[prompt_user_ids]
                                hidden_verify = hidden_2b[spec_user_ids]
                                if accept_mask.any():
                                    accept_prompt_ids = prompt_user_ids[accept_mask]
                                    hidden_for_spec[accept_prompt_ids] = hidden_verify[accept_mask]
                            else:
                                if hidden_2b.shape[0] < verify_offset + num_of_prompts:
                                    raise RuntimeError(
                                        "Hidden batch smaller than verify offset + num_prompts: "
                                        f"{hidden_2b.shape[0]} < {verify_offset + num_of_prompts}"
                                    )
                                hidden_prompt = hidden_2b[:num_of_prompts]
                                hidden_verify = hidden_2b[verify_offset : verify_offset + num_of_prompts]
                                if accept_mask.any():
                                    hidden_for_spec[:num_of_prompts][accept_mask] = hidden_verify[accept_mask]

                        spec_logits_full = self._mtp_predict_logits(
                            hidden_states=hidden_for_spec,
                            tokens_step=tokens_for_spec,
                            positions=positions_for_spec,
                        )
                        self.ccl.reset_sem_counters()
                        spec_all = self._sample_greedy(spec_logits_full)
                        ttnn.deallocate(hidden_2b_tt)

                        if use_interleaved and prompt_user_ids is not None:
                            spec_tokens_next = spec_all[prompt_user_ids]
                        else:
                            spec_tokens_next = spec_all[:num_of_prompts]
                        spec_tokens = spec_tokens_next
                        if mtp_step_trace:
                            for i in range(num_of_prompts):
                                logger.info(
                                    "MTP_STEP_NEXT_SPEC user={} step={} next_spec_token={}".format(
                                        i, decode_step_idx - 1, int(spec_tokens[i].item())
                                    )
                                )
                        decode_step_active_masks.append(step_active_mask)
                        decode_step_user_tokens.append(step_user_tokens)

                    mtp_accepts = total_accepts
                    if total_verifies > 0:
                        mtp_accept_rate = total_accepts / total_verifies
                        logger.info(f"MTP accept rate: {total_accepts}/{total_verifies} = {mtp_accept_rate:.3f}")
                        logger.info(
                            "MTP skip-path summary: enabled={} skipped_decode_tokens={} "
                            "accepted_committed_second_token={}".format(
                                skip_accept_decode,
                                skipped_decode_tokens if skip_accept_decode else 0,
                                accepted_committed_second_token if skip_accept_decode else 0,
                            )
                        )
                        if self.min_mtp_accept_rate is not None and mtp_accept_rate < self.min_mtp_accept_rate:
                            raise RuntimeError(
                                f"MTP accept rate {mtp_accept_rate:.3f} below required minimum "
                                f"{self.min_mtp_accept_rate:.3f}"
                            )
                        if debug_mtp and total_verifies_alt > 0:
                            alt_rate = total_accepts_alt / total_verifies_alt
                            logger.info(
                                f"MTP alt accept rate (spec vs pred_after_spec): {total_accepts_alt}/{total_verifies_alt} = {alt_rate:.3f}"
                            )
                    elif self.min_mtp_accept_rate is not None:
                        raise RuntimeError("MTP verification produced zero samples; cannot validate accept rate.")
                    else:
                        mtp_accept_rate = 0.0
                else:
                    decode_steps = max_new_tokens - 1
                    for gen_idx in range(decode_steps):
                        logger.info(f"Decoding step {gen_idx} for {num_of_prompts} user(s)...")
                        profiler.start(f"decode_time_{gen_idx}")
                        logits = self.decode_forward(
                            tokens=next_tokens,
                            start_pos=positions,
                            batch_size_per_row=self.batch_size_per_row,
                            profiler=profiler,
                            gen_idx=gen_idx,
                            enable_trace=self.enable_trace,
                        )
                        profiler.end(f"decode_time_{gen_idx}")
                        decode_step_idx = gen_idx + 1
                        decode_forward_passes += 1
                        self.ccl.reset_sem_counters()
                        pred_tokens = self._sample_greedy(logits)
                        if teacher_forcing is not None:
                            # Record user-0 prediction for accuracy, then force teacher token.
                            forced = teacher_forcing.collect_predicted_tokens(int(pred_tokens[0].item()))
                            pred_tokens[0] = int(forced)
                        next_tokens = pred_tokens
                        positions += 1

                        step_active_mask = [True for _ in range(num_of_prompts)]
                        step_user_tokens = [0 for _ in range(num_of_prompts)]
                        for i in range(num_of_prompts):
                            token_value = int(next_tokens[i].item())
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
            total_generated_tokens = 0
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
                    total_generated_tokens += tokens_i
                    if i < len(step_active_mask) and step_active_mask[i]:
                        per_user_active_time[i] += step_time

            if repeat_batches > 1:
                per_user_tokens = [t * repeat_batches for t in per_user_tokens]

            per_user_tps = [
                (per_user_tokens[i] / per_user_active_time[i]) if per_user_active_time[i] > 0 else 0
                for i in range(num_of_prompts)
            ]
            decode_tokens_per_sec_per_user = (sum(per_user_tps) / num_of_prompts) if num_of_prompts > 0 else 0
            decode_tokens_per_sec = (
                (decode_tokens_per_sec_per_user * stats_max_prompts_per_batch) if total_decode_time > 0 else 0
            )
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
        prompt_len: int | None = None,
        return_last_hidden: bool = False,
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

        # Gather to host
        logits = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )

        if self.enable_mtp:
            # Prime MTP cache for this user using prompt tokens.
            mtp_page_table = self._get_mtp_page_table()
            full_seq_len = int(hidden_tt.shape[2])
            if os.getenv("DEEPSEEK_MTP_SKIP_PREFILL", "0") == "1":
                logger.info("Skipping MTP prefill priming (DEEPSEEK_MTP_SKIP_PREFILL=1).")
            elif full_seq_len > 0 and prompt_len is not None and prompt_len > 1:
                # Prime MTP cache with aligned pairs: hidden[t] + token[t+1].
                # Keep base prefill parity-safe, then pad only the MTP priming sequence so
                # reduce_scatter sees a ring-compatible tile count.
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
                    # Keep hidden aligned to cache positions; shift tokens left to represent token[t+1].
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

                    rope_setup_mtp = RotarySetup(
                        device=self.mesh_device,
                        batch_size_per_row=1,
                        hf_config=self.hf_config,
                    )
                    mtp_rot_mats = rope_setup_mtp.get_rot_mats_table(aligned_global_len)
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
                    ttnn.deallocate(mtp_logits_tt)
                    self.ccl.reset_sem_counters()

            if return_last_hidden:
                if prompt_len is None:
                    raise ValueError("prompt_len is required when return_last_hidden=True")
                if prompt_len <= 0:
                    last_hidden = torch.zeros((self.hf_config.hidden_size,), dtype=torch.bfloat16)
                else:
                    hidden_idx = min(prompt_len - 1, full_seq_len - 1)
                    hidden_slice = ttnn.slice(
                        hidden_tt, [0, 0, hidden_idx, 0], [1, 1, hidden_idx + 1, hidden_tt.shape[3]]
                    )
                    last_hidden = ttnn.to_torch(
                        hidden_slice,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape
                        ),
                    )
                    last_hidden = last_hidden.squeeze(0).squeeze(0).squeeze(0)
                    if last_hidden.dim() == 2 and last_hidden.shape[-1] == self.hf_config.hidden_size:
                        # Some mesh composers leave an extra mesh-row dimension; take the first row.
                        last_hidden = last_hidden[0]
                    ttnn.deallocate(hidden_slice)

            ttnn.deallocate(hidden_tt)

        # Free device tensors for this step
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(logits_tt)
        if return_last_hidden:
            return logits, last_hidden
        return logits  # [1, 1, seq_len, V]

    def _capture_decode_trace(
        self,
        init_tokens: torch.Tensor,
        positions: torch.Tensor,
        batch_size_per_row: int,
        page_tables: torch.Tensor | None = None,
    ) -> None:
        """Allocate persistent inputs, capture trace for one decode iteration, and store trace state."""
        assert self._trace_id is None, "Trace already captured"

        # 1) Warm-up compile run (no trace) to keep compilation out of capture
        logger.info("Running warm-up decode step (no trace)...")
        if self.signpost:
            signpost(header="decode_warmup")
        _ = self._decode_step(init_tokens, positions, batch_size_per_row=batch_size_per_row, page_tables=page_tables)
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
        batch_size_per_row: int = USERS_PER_ROW,
        gen_idx: int = 0,
        profiler: BenchmarkProfiler | None = None,
        enable_trace: bool = False,
        page_table: torch.Tensor | None = None,
        kv_cache: None = None,
        read_from_device: bool = None,
        sampling_params: SamplingParams = None,
    ) -> torch.Tensor:
        # vLLM does not pass enable_trace param while initializing the model.
        # vLLM sets it in decode/prefill calls only, so we need to set it here too.
        self.enable_trace = enable_trace
        if not enable_trace:
            return self._decode_step(tokens, start_pos, batch_size_per_row, page_table).squeeze(0).squeeze(0)
        else:
            # Capture trace and return trace output
            if self._trace_id is None:
                self._capture_decode_trace(tokens, start_pos, batch_size_per_row, page_table)
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
                for i, page_table in enumerate(page_tables_to_use):
                    ttnn.copy_host_to_device_tensor(page_table, self._trace_page_tables_to_use[i])

            self.ccl.reset_sem_counters()
            if profiler is not None:
                profiler.start(f"trace_execution_{gen_idx}")
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)
            if profiler is not None:
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
