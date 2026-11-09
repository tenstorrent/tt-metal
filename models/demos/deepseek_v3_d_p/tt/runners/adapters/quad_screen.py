# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Weight-free adapters for the quad prefill-runner hardware screen (gwdt --quad).

Both reuse Kimi-K2.7's dimensions, KV cache, fabric payload and L1_SMALL carve-out, and its in-repo config, so a
run needs no weights, weight cache, golden trace or mounted share. Each pipeline rank is one stage whose body is
fixed (quad_screen_runtime.py); every stage checks what it received and what it sent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_7 import KimiK27Adapter

_IN_REPO_CONFIG = Path(__file__).resolve().parents[3] / "reference" / "kimi_k2_7"


class _QuadScreenAdapter(KimiK27Adapter):
    hf_model_default = str(_IN_REPO_CONFIG)
    ttnn_cache_default = ""
    prefill_trace_default = ""
    supports_dflash = False

    def weight_cache_path(self, mesh_shape: tuple):
        return None


class QuadBlockRandomAdapter(_QuadScreenAdapter):
    """Every stage: K2.7 layer 0 (dense) + layer 1 (MoE), block-random's random weights and input."""

    name = "quad_block_random"

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.deepseek_v3_d_p.tt.runners.quad_screen_runtime import (
            QuadBlockRandomRuntime,
            random_block_state_dict,
        )

        assert params.num_layers == 2, f"each quad screen stage runs 2 layers, got {params.num_layers}"
        # Built as a middle stage so it takes a hidden state and returns one on every rank. Global layer
        # first_layer_idx is made the dense one and the next the MoE one, so every stage runs block-random's
        # L0 + L1 while its layer acks keep their own global indices.
        inner = dataclasses.replace(
            params, is_first_rank=False, is_last_rank=False, kv_only_last_layer=False, weight_cache_path=None
        )
        stage_cfg = type(
            "QuadStageKimiK27Config", (self.model_config,), {"NUM_DENSE_LAYERS": params.first_layer_idx + 1}
        )
        config = dataclasses.replace(self.runtime_config(inner), model_cfg=stage_cfg)
        return QuadBlockRandomRuntime(
            mesh_device,
            hf_config,
            config,
            random_block_state_dict(self, hf_config),
            is_first_rank=params.is_first_rank,
            is_last_rank=params.is_last_rank,
        )


class QuadPassthroughAdapter(_QuadScreenAdapter):
    """Every stage forwards its input through ttnn.clone "layers"; no weights at all."""

    name = "quad_passthrough"

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.deepseek_v3_d_p.tt.runners.quad_screen_runtime import QuadPassthroughRuntime

        config = self.runtime_config(dataclasses.replace(params, kv_only_last_layer=False, weight_cache_path=None))
        return QuadPassthroughRuntime(mesh_device=mesh_device, hf_config=hf_config, state_dict={}, config=config)
