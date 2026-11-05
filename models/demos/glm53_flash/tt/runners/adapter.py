# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 Flash adapter for the common prefill runner (tt-blaze G53F-0256 G4b): the disaggregated prefill of the
tt-blaze 48-ring, served by tt-d-gen's prefill worker. One rank (the 8x4 mesh), one slot, 45 decoder layers + the MTP
row 45 on the table's layer axis. The KV the runner publishes is the ring's byte format (``kv_export.py``), so the KV
Manager moves it chunk for chunk into the ring's slot. Import-light: everything heavy is imported inside the methods."""

from __future__ import annotations

import os
from pathlib import Path

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter


class GLM53FlashModelConfig:
    NUM_LAYERS = 46  # 45 decoder layers + the MTP row (table layer 45)

    @staticmethod
    def fabric_payload() -> int:
        from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_max_payload_size

        return get_max_payload_size()


class GLM53FlashExport(KvCaches):
    """Marker base: ``kv_export.GlmRingExport`` instances are registered as this model's ``KvCaches``."""


class GLM53FlashAdapter(PrefillModelAdapter):
    name = "glm53_flash"
    model_config = GLM53FlashModelConfig
    hf_model_default = ""  # the HF cache snapshot (reference.checkpoint.default_snapshot_dir)
    ttnn_cache_default = "/mnt/tt-data/sdawle/glm53f/prefill/mesh_cache"
    prefill_trace_default = ""
    l1_small_size = 32768
    supports_mtp = False

    def fabric_payload_size(self) -> int:
        return GLM53FlashModelConfig.fabric_payload()

    def cache_kind(self, config_id: int) -> str:
        return ("glm_dsa_latent", "glm_dsa_pool", "kda_recurrent", "kda_convolution")[config_id]

    def layer_position_range(self, layer_idx: int, real_len: int) -> tuple[int, int]:
        from models.demos.glm53_flash.tt.runners.kv_export import DSA_LAYERS, MTP_LAYER, kda_window_range

        if layer_idx in DSA_LAYERS or layer_idx == MTP_LAYER:
            return 0, real_len
        return kda_window_range(real_len, int(os.environ.get("GLM53F_KDA_VERSIONS", "2")))

    def load_hf_config(self):
        """``GLM53FlashConfig`` from the checkpoint, as a subclass instance so the runner may set ``max_seq_len`` (the
        dataclass is frozen; a non-field attribute on a subclass instance is allowed)."""
        from dataclasses import fields

        from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
        from models.demos.glm53_flash.tt.config import GLM53FlashConfig

        snap = Path(os.environ.get("PREFILL_HF_MODEL") or default_snapshot_dir())
        base = GLM53FlashConfig.from_hf_config_json(snap / "config.json")

        class GLM53FlashRunnerConfig(GLM53FlashConfig):
            pass

        return GLM53FlashRunnerConfig(**{f.name: getattr(base, f.name) for f in fields(base)})

    def weight_cache_path(self, mesh_shape: tuple):
        root = os.environ.get("PREFILL_TTNN_CACHE", self.ttnn_cache_default)
        return None if root.lower() == "none" else Path(root)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        from models.demos.glm53_flash.tt.runners.kv_export import GlmRingExport

        GLM53FlashExport.register(GlmRingExport)
        return GlmRingExport.allocate(
            mesh_device,
            max_seq_len=params.max_seq_len,
            versions=int(os.environ.get("GLM53F_KDA_VERSIONS", "2")),
        )

    def build_runtime(self, *, mesh_device, hf_config, params):
        from models.demos.glm53_flash.tt.runners.runtime import GLM53FlashPrefillRuntime

        return GLM53FlashPrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, params=params)
