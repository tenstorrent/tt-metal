# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DeepSeek-V4 Pro / Flash prefill adapters.

Subclasses ``MLAPrefillAdapter`` for the shared plumbing and replaces what V4 does differently:

* the config is built from the model-config class (``v4_hf_config``), not read from the checkpoint;
* there is no engine-owned KV cache. V4 attention reads its weights off torch modules and keeps its own
  state, so ``build_runtime`` loads the native checkpoint on host, one layer at a time, and
  ``allocate_kv_cache`` hands back an empty handle;
* the ttnn weight cache is opt-in (set the variant's ``ttnn_cache_env``) and covers only what V4 caches:
  the norms and the MoE (routed + shared experts, gate). A hit saves the host conversion of those
  weights; it does not skip reading the checkpoint, since attention still comes off the torch modules;
* a pipeline rank other than the first may not start inside the hash-routed layers, which need host
  token ids that only the first rank has.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches

_CHECKPOINT_ROOT = "/mnt/models/blaze/deepseek-ai"
_TRACE_ROOT = "/mnt/models/deepseek-prefill-cache/golden/structured_traces"


class _DeepSeekV4Adapter(MLAPrefillAdapter):
    default_gate_mode = "DEVICE_FP32"  # applies to the top-k layers; hash layers always take the hash gate
    # Opt-in: each variant names the env var that points at its cache root. CI sets it to the weka paths
    # (see tests/pipeline_reorg/blaze_models_prefill_tests.yaml); with it unset nothing is read or written.
    ttnn_cache_default = ""
    config_builder_overrides_checkpoint = True
    prefill_trace_layout = "chunked_group_a_v1"

    @property
    def config_builder(self) -> Callable:
        from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config

        return partial(v4_hf_config, self.model_config)

    @property
    def transformer_cls(self):
        from models.demos.deepseek_v3_d_p.tt.v4.transformer import TtV4Transformer

        return TtV4Transformer

    def load_hf_config(self):
        """The full-depth config from the model constants; the runner overwrites ``max_seq_len``."""
        return self.config_builder()

    def weight_cache_path(self, mesh_shape: tuple, checkpoint: Optional[Path] = None) -> Optional[Path]:
        """The ttnn weight-cache dir, or None when no cache root is configured (or it cannot be created).

        ``<root>/{name}_{arch}_{N}dev/<checkpoint-dir-name>/{sp}x{tp}``: the same ``{name}_{arch}_{N}dev``
        / ``{sp}x{tp}`` frame as the other adapters, with the checkpoint's directory name between them so
        that a different checkpoint can never silently load another's tensors (the Kimi-K3 convention).
        The root is the variant's ``ttnn_cache_env``, else ``PREFILL_TTNN_CACHE``, else ``ttnn_cache_default``.

        ``checkpoint`` defaults to ``checkpoint_path()``; a test that resolves its own checkpoint passes it so
        the cache is keyed by the weights it actually loads.
        """
        env_name = getattr(self, "ttnn_cache_env", None) or ""
        root = os.environ.get(env_name) or os.environ.get("PREFILL_TTNN_CACHE") or self.ttnn_cache_default
        if not root:
            return None
        checkpoint = Path(checkpoint) if checkpoint is not None else self.checkpoint_path()
        arch = "bh" if is_blackhole() else "wh"
        sp, tp = mesh_shape
        path = Path(root) / f"{self.name}_{arch}_{ttnn.get_num_devices()}dev" / checkpoint.name / f"{sp}x{tp}"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # e.g. a user outside the cache-writers group: run uncached rather than fail
            logger.warning(f"DeepSeek-V4 weight cache disabled: cannot create {path}: {exc}")
            return None
        logger.info(f"DeepSeek-V4 ttnn weight cache: {path}")
        return path

    def checkpoint_path(self) -> Path:
        from models.demos.deepseek_v3_d_p.tt.v4.weights import CHECKPOINT_INDEX

        path = Path(os.environ.get("PREFILL_HF_MODEL") or self.hf_model_default)
        if not (path / CHECKPOINT_INDEX).is_file():
            raise FileNotFoundError(f"no DeepSeek-V4 checkpoint index at {path}; set PREFILL_HF_MODEL")
        return path

    def allocate_kv_cache(self, *, mesh_device, hf_config, params) -> MlaKvCaches:
        """Empty: the model allocates its own caches, migratable ones included (``tt/v4/kv_caches.py``)."""
        return MlaKvCaches(kvpe=None)

    def cache_kind(self, config_id: int) -> str:
        """Table config ``config_id``'s kind; none is a ``"kvpe"`` or DSA ``"index"`` cache a generic consumer
        could decode, so the producer must not infer one from the config count."""
        from models.demos.deepseek_v3_d_p.tt.v4.kv_caches import CACHE_KINDS

        return CACHE_KINDS[config_id] if 0 <= config_id < len(CACHE_KINDS) else "other"

    def pipeline_activation_planes(self, boundary_layer_idx: int) -> int:
        """Every boundary carries the residual streams, one plane each (``TtV4Runtime.activation_planes``)."""
        return self.model_config.HC_MULT

    def layer_split_boundaries(self, num_layers: int) -> set:
        """Layer 0, or any layer past the hash-routed ones."""
        return {0} | set(range(self.model_config.NUM_HASH_LAYERS, num_layers + 1))

    def build_runtime(self, *, mesh_device, hf_config, params):
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
        from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntimeConfig
        from models.demos.deepseek_v3_d_p.tt.v4.runtime import TtV4Runtime
        from models.demos.deepseek_v3_d_p.tt.v4.weights import V4CheckpointLayers, v4_model_from_checkpoint

        topology = per_axis_topology()
        checkpoint = self.checkpoint_path()
        logger.info(f"DeepSeek-V4 per-axis CCL topology (sp, tp) = {topology}; checkpoint {checkpoint}")

        state_dict = v4_model_from_checkpoint(
            hf_config, checkpoint, load_embed=params.is_first_rank, load_tail=params.is_last_rank
        )
        state_dict["layers"] = V4CheckpointLayers(hf_config, checkpoint, params.first_layer_idx, params.num_layers)

        runtime_config = TtPrefillRuntimeConfig(
            num_layers=params.num_layers,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            chunk_size=params.chunk_size,
            num_users=params.num_users,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            num_links=params.num_links,
            topology=topology,
            capacity_factor=params.capacity_factor,
            gate_fallback_mode=GateComputeMode[params.gate_mode_name],
            weight_cache_path=params.weight_cache_path,
            model_cfg=self.model_config,
            first_layer_idx=params.first_layer_idx,
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
            # The runner asks the last rank for a KV-only last layer; V4 has none, so it runs that layer whole.
            kv_only_last_layer=False,
            dflash_enabled=params.dflash_enabled,
            mtp_levels=params.mtp_levels,
            routing_use_l1_small_for_semaphores=self.routing_use_l1_small_for_semaphores,
            sparse_kv_cache_format=None,
            use_trace=params.use_trace,
            overlap_shared_expert_with_dispatch=params.overlap_shared_expert_with_dispatch,
        )
        return TtV4Runtime(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict, config=runtime_config)


class DeepSeekV4ProAdapter(_DeepSeekV4Adapter):
    name = "deepseek_v4_pro"
    model_config = DeepSeekV4ProConfig
    hf_model_default = f"{_CHECKPOINT_ROOT}/DeepSeek-V4-Pro-0813-dequantized"
    prefill_trace_default = f"{_TRACE_ROOT}/v4_pro_55K_partial_trace/trace_v4_pro_full"
    env_var = "V4_PRO_HF_MODEL"
    ttnn_cache_env = "TT_DSV4_PRO_PREFILL_TTNN_CACHE"


class DeepSeekV4FlashAdapter(_DeepSeekV4Adapter):
    name = "deepseek_v4_flash"
    model_config = DeepSeekV4FlashConfig
    hf_model_default = f"{_CHECKPOINT_ROOT}/DeepSeek-V4-Flash-0731-dequantized"
    prefill_trace_default = f"{_TRACE_ROOT}/v4_flash_55K_partial_trace/trace_v4_flash_full"
    env_var = "V4_FLASH_HF_MODEL"
    ttnn_cache_env = "TT_DSV4_FLASH_PREFILL_TTNN_CACHE"
