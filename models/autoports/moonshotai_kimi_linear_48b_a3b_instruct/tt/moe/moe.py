# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE block = sigmoid router -> sparse routed experts (+ shared expert), one all-reduce."""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.ccl import KimiCCL
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.mlp import TPSwiGLU
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.moe.experts import KimiExperts
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.moe.router import KimiRouter


class KimiMoE:
    def __init__(
        self,
        mesh_device,
        cfg: KimiLinearConfig,
        sd: dict | None,
        *,
        layer_idx: int,
        ccl: KimiCCL,
        cache_path: Path | None,
        expert_dtype=ttnn.bfloat8_b,
        shared_dtype=ttnn.bfloat8_b,
    ):
        self.ccl = ccl
        name = f"layer_{layer_idx}.moe"
        get = (lambda k: None) if sd is None else (lambda k: sd.get(k))
        self.router = KimiRouter(
            mesh_device,
            cfg,
            get("moe.gate.weight"),
            get("moe.gate.e_score_correction_bias"),
            name=f"{name}.router",
            cache_path=cache_path,
        )
        self.experts = KimiExperts(
            mesh_device, cfg, sd, name=f"{name}.experts", cache_path=cache_path, dtype=expert_dtype
        )
        self.shared = None
        if cfg.num_shared_experts:
            self.shared = TPSwiGLU(
                mesh_device,
                get("moe.shared.gate_proj.weight"),
                get("moe.shared.up_proj.weight"),
                get("moe.shared.down_proj.weight"),
                name=f"{name}.shared",
                cache_path=cache_path,
                dtype=shared_dtype,
            )

    def forward(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """x [1,1,S,H] replicated (post-norm) -> [1,1,S,H] replicated."""
        routing = self.router(x)
        routed = (
            self.experts.forward_decode(x, routing) if mode == "decode" else self.experts.forward_prefill(x, routing)
        )
        ttnn.deallocate(routing)
        if self.shared is not None:
            sh = self.shared.forward(x)
            routed = ttnn.add(routed, sh)
            ttnn.deallocate(sh)
        return self.ccl.all_reduce(routed)


class KimiDenseMLP:
    def __init__(
        self,
        mesh_device,
        cfg: KimiLinearConfig,
        sd: dict | None,
        *,
        layer_idx: int,
        ccl: KimiCCL,
        cache_path: Path | None,
        dtype=ttnn.bfloat8_b,
    ):
        self.ccl = ccl
        get = (lambda k: None) if sd is None else (lambda k: sd.get(k))
        self.mlp = TPSwiGLU(
            mesh_device,
            get("mlp.gate_proj.weight"),
            get("mlp.up_proj.weight"),
            get("mlp.down_proj.weight"),
            name=f"layer_{layer_idx}.mlp",
            cache_path=cache_path,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor, mode: str = "decode") -> ttnn.Tensor:
        return self.ccl.all_reduce(self.mlp.forward(x))
