# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Hy4-preview (tencent/Hy4-preview, ``hy_v4``) prefill adapter: GLM-5.3 plus the Hy4 deltas.

Hy4's attention and MoE have GLM-5.3's dimensions, so this subclasses ``GLM53Adapter`` and inherits
its KV layout unchanged: the scaled-FP8 MLA KVPE cache on every layer, plus the block-cyclic,
TP-deduplicated index-key cache sized to the FULL-indexer layer count. Hy4 has 21 full layers, read
from its explicit ``indexer_types`` by ``full_indexer_rank``. What it changes:

  * config: ``Hy4Config`` / ``hy4_hf_config`` (``reference/hy4_config.py``), built by hand because
    ``hy_v4`` is not AutoConfig-loadable on our transformers pin. When the checkpoint's config.json is
    readable it is validated against the constants and supplies the per-layer maps.
  * weights: ``TtHy4Runtime`` rebuilds a missing weight cache through the Hy4 checkpoint reader
    (``tt/runners/hy4_checkpoint.py``: fused experts, MXFP8 scales, ``linear_gate`` -> ``g_proj``,
    ``learnable_sink_param`` -> ``attn_sink``).
  * mesh layout: TP32 / DP1 (``tp32``, the SC1 placement of #65884: hidden over the 4-chip mesh axis 1, heads and
    sequence over the 8-chip axis 0, residual ``[S / 8, 6144 / 4]``), selected by default. The shared prefill
    stack cannot run it yet, so it fails at runner start naming the blocking issues; GLM-5.3's SP 8 x TP 4
    stays as the explicitly selected DEV / NUMERICS layout (``PREFILL_HY4_LAYOUT=glm_prefill_sp8_tp4``) on the same
    8x4 mesh. See ``tt/hy4/layout.py``. The weight cache is keyed by mesh and layout.
  * deltas the stack does not implement yet are refused unless a bring-up override is set (below).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.hy4_config import (
    Hy4Config,
    Hy4TP32Config,
    hy4_hf_config,
    load_hy4_config_json,
)
from models.demos.deepseek_v3_d_p.tt.hy4.layout import resolve_layout
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import mesh_cache_key

# Bring-up override: run with the deltas below MISSING (GLM-5.3 numerics for those parts). For smoke runs
# and for checks that do not depend on them; KV PCC past the first layer will not match HF hy_v4.
ALLOW_PENDING_DELTAS_ENV = "PREFILL_HY4_ALLOW_PENDING_DELTAS"

# Hy4 deltas the prefill stack does not implement yet: config attribute -> (what, owning issue). An issue
# that implements one deletes its entry here.
PENDING_DELTAS = {
    "enable_ihc": ("iHC hyper-connection residual (hc_mult=4, fp32 streams)", "#65891"),
}


def pending_deltas(hf_config) -> list[str]:
    """Descriptions of the Hy4 deltas ``hf_config`` turns on that the TT prefill stack cannot run yet."""
    return [f"{what} ({issue})" for attr, (what, issue) in PENDING_DELTAS.items() if getattr(hf_config, attr, False)]


class Hy4Adapter(GLM53Adapter):
    # --- identity & runner defaults ---
    name = "hy4_preview"
    model_config = Hy4Config
    # BF16 release. The MXFP8 release (tencent/Hy4-preview-FP8) reads through the same path.
    hf_model_default = "/mnt/tt-data/Hy4-Preview"
    ttnn_cache_default = "/mnt/tt-data/Hy4-Preview-Cache/prefill"
    # Golden traces come from #65919 (HF hy_v4); PREFILL_TRACE_DIR points the producer at one.
    prefill_trace_default = "/mnt/tt-data/Hy4-Preview-Cache/golden_traces/hy4-prefill"
    # GLM-5.3's MTP predictor reads DeepSeek-layout model.layers.78; Hy4's MTP layer is model.mtp_layers.0 (#65925).
    supports_mtp = False

    # --- test metadata ---
    hf_repo_id = "tencent/Hy4-preview"
    env_var = "HY4_HF_MODEL"
    mla_ref_cache_env = "HY4_MLA_REF_CACHE"
    ref_cache_env = "TT_HY4_PREFILL_HOST_REF_CACHE"
    ttnn_cache_env = "TT_HY4_PREFILL_TTNN_CACHE"
    # No HF hy_v4 reference is wired into the pytest fixtures yet (#65919 / #65893).
    supports_pretrained = False
    pretrained_mla_layer = None
    mla_trace_defaults = ()
    test_prefill_trace_default = prefill_trace_default
    # GLM-5.3's thresholds until the sibling issues measure Hy4's (#65921 MLA, #65923 MoE).
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971

    def _checkpoint_dir(self) -> str:
        return os.environ.get("PREFILL_HF_MODEL") or os.environ.get(self.env_var, "") or self.hf_model_default

    def load_hf_config(self):
        """Hand-built ``hy_v4`` config. If the checkpoint's config.json is readable it is validated against
        ``Hy4Config`` (a mismatch raises) and supplies ``indexer_types``; otherwise the constants do. The
        runner overwrites ``max_seq_len`` afterwards; seed the builder with it so the rope config agrees."""
        max_seq = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 8192))
        cfg_json = Path(self._checkpoint_dir()) / "config.json"
        if cfg_json.is_file():
            logger.info(f"Hy4: building the hy_v4 config from {cfg_json} (validated against Hy4Config)")
            return hy4_hf_config(max_seq=max_seq, config_dict=load_hy4_config_json(cfg_json))
        logger.warning(f"Hy4: no config.json at {cfg_json}; using the Hy4Config constants")
        return hy4_hf_config(max_seq=max_seq)

    @property
    def config_builder(self) -> Callable:
        return hy4_hf_config

    # --- mesh layout (TP32 / DP1 by default; tt/hy4/layout.py) ---
    def _resolve_layout(self, sp: int, tp: int, sp_axis: int):
        return resolve_layout(sp, tp, sp_axis, Hy4Config.NUM_ATTENTION_HEADS)

    def weight_cache_path(self, mesh_shape: tuple, sp_axis: int = 0) -> Optional[Path]:
        """``$PREFILL_TTNN_CACHE/{name}_{arch}_{N}dev/{rows}x{cols}[_sp1]{layout suffix}``. The runner calls this
        before it opens the mesh, so an unrunnable layout (``tp32`` today) fails here, at startup."""
        layout = self._resolve_layout(mesh_shape[sp_axis], mesh_shape[1 - sp_axis], sp_axis)
        root = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        if not root:
            return None
        key = mesh_cache_key(mesh_shape, sp_axis)
        if layout is not None:
            key = layout.cache_key(key)
        path = Path(root) / f"{self.name}_{'bh' if is_blackhole() else 'wh'}_{ttnn.get_num_devices()}dev" / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _check_layout(self, params) -> None:
        layout = self._resolve_layout(params.sp_factor, params.tp_factor, params.sp_axis)
        heads = (
            layout.heads_per_chip(Hy4Config.NUM_ATTENTION_HEADS)
            if layout is not None
            else Hy4Config.NUM_ATTENTION_HEADS // params.tp_factor
        )
        if layout is None:
            logger.info(
                f"Hy4: test mesh SP={params.sp_factor} x TP={params.tp_factor} ({heads} heads per chip), no layout"
            )
        elif layout.role == "dev":
            logger.warning(f"Hy4: layout {layout.name} ({layout.note}); the Hy4 layout is TP32 / DP1 (#65884)")
        else:
            logger.info(f"Hy4: layout {layout.name} ({layout.note})")

    # --- pending deltas ---
    def _check_pending_deltas(self, hf_config) -> None:
        pending = pending_deltas(hf_config)
        if not pending:
            return
        if os.environ.get(ALLOW_PENDING_DELTAS_ENV, "0") != "1":
            raise NotImplementedError(
                "Hy4-preview prefill cannot run these deltas yet: "
                + "; ".join(pending)
                + f". Set {ALLOW_PENDING_DELTAS_ENV}=1 to run without them (bring-up only: KV after the first layer "
                "will not match HF hy_v4)."
            )
        logger.warning(
            f"Hy4: {ALLOW_PENDING_DELTAS_ENV}=1, running WITHOUT: " + "; ".join(pending) + ". Outputs are not Hy4's."
        )

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        self._check_layout(params)
        self._check_pending_deltas(hf_config)  # before the cache takes DRAM, not after
        return super().allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def model_config_for(self, params) -> type:
        """The model-constants class for this mesh: ``Hy4TP32Config`` (2D-TP shared expert and dense FFN, #65890) on a
        layout that asks for it (tp32), ``Hy4Config`` otherwise (the dev layout and test meshes)."""
        layout = self._resolve_layout(params.sp_factor, params.tp_factor, params.sp_axis)
        return Hy4TP32Config if layout is not None and layout.ffn_2d_tp else Hy4Config

    def build_runtime(self, *, mesh_device, hf_config, params):
        self._check_layout(params)
        self._check_pending_deltas(hf_config)
        # Shadows the class attribute for this adapter instance only; the runtime config reads it.
        self.model_config = self.model_config_for(params)
        return super().build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)

    def _runtime_cls(self) -> type:
        from models.demos.deepseek_v3_d_p.tt.hy4.runtime import TtHy4Runtime

        return TtHy4Runtime
