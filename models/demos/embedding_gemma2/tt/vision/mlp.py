# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionMLP (HF `Gemma4VisionMLP`): down( gelu_tanh(gate x) * up x ), no biases, 768 -> 3072 -> 768.

Same math as the text `MLP`; E5b gives the vision tower its own module so its policy can differ:
    weights      dtype / fidelity / output dtype from the precision policy (E6b, `vision.weights.mlp_gate_up`,
                 `mlp_down`; doc/datatype_sweep/README.md)
    projections  `vision_linear` (2D multicast configs at M <= 4096, text rule above)
    gelu_tanh    applied as an SFPU pre-activation of the gate * up multiply, not in the gate
                 matmul's packer: -0.3 ms at 1 x 2560, -6.9 ms at 4 x 2560 and 8 x 1280 per forward
                 (E5b A/B; the packer gelu cost 52 us per gate matmul at M = 2560, Tracy)
    per image    batches above VISION_L1_WIDE_MAX_TOKENS["mlp"] patches run one image / frame at a
                 time (slice -> the one-image path with up / gate in L1 -> concat): -5.6 ms at
                 4 x 2560 (121.63 -> 116.02 ms), -6.5 ms at 8 x 1280 (94.67 -> 88.14 ms) per forward
                 (E5b review fix, bench_vision burst medians, doc/optimized_vision/README.md)

Contract
    input   x [B, 1, S, 768] bf16 TILE (DRAM or L1 interleaved)
    output  [B, 1, S, 768] bf16 TILE, `memory_config` (DRAM when None)
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import MLPConfig
from models.demos.embedding_gemma2.tt.precision import VisionPrecision, resolve_precision
from models.demos.embedding_gemma2.tt.vision.config import VISION_L1_WIDE_MAX_TOKENS, vision_act_memory_config
from models.demos.embedding_gemma2.tt.vision.linear import vision_linear
from models.demos.embedding_gemma2.tt.weights import as_weight, linear_weight

_GELU = [ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)]


class VisionMLP(LightweightModule):
    def __init__(
        self,
        config: MLPConfig,
        w_gate: ttnn.Tensor,
        w_up: ttnn.Tensor,
        w_down: ttnn.Tensor,
        *,
        layer_idx: int = 0,
        precision: VisionPrecision | None = None,
    ):
        super().__init__()
        precision = precision or resolve_precision(None).vision
        self.config = config
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.gate_up_policy = precision.matmul("mlp_gate_up", layer_idx)
        self.down_policy = precision.matmul("mlp_down", layer_idx)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: MLPConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "vision.mlp",
        layer_idx: int = 0,
        precision: VisionPrecision | None = None,
    ) -> "VisionMLP":
        """`state_dict` holds the HF `encoder.layers.<i>.mlp.` sub-state (`.linear.` folded);
        `precision` the vision policy (None: the default precision file), applied as layer `layer_idx`."""
        precision = precision or resolve_precision(None).vision
        H, I = config.hidden_size, config.intermediate_size
        shapes = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
        group = {"gate_proj": "mlp_gate_up", "up_proj": "mlp_gate_up", "down_proj": "mlp_down"}
        ws = {}
        for k, shp in shapes.items():
            w = state_dict[f"{k}.weight"]
            if tuple(w.shape) != shp:
                raise ValueError(f"{name}.{k}: expected {shp}, got {tuple(w.shape)}")
            dtype = precision.matmul(group[k], layer_idx).ttnn_dtype
            ws[k] = as_weight(linear_weight(w), device, cache_dir=cache_dir, name=f"{name}.{k}.weight", dtype=dtype)
        return cls(config, ws["gate_proj"], ws["up_proj"], ws["down_proj"], layer_idx=layer_idx, precision=precision)

    def forward(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        B, S = x.shape[0], x.shape[2]
        wide_max = VISION_L1_WIDE_MAX_TOKENS["mlp"]
        if B > 1 and B * S > wide_max and S <= wide_max:
            return self._forward_per_image(x, memory_config)
        return self._forward(x, memory_config)

    def _forward_per_image(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        """Batch too large for the L1 up / gate outputs: one image at a time on the one-image path.

        At 4 x 2560 the M = 10240 projections run minimal_matmul with DRAM outputs; per image they
        get the M = 2560 2D configs and L1 up / gate outputs (121.63 -> 116.02 ms per forward; 8 x 1280:
        94.67 -> 88.14 ms; bench_vision burst medians). Each image's input slice and MLP output are
        [1, 1, S, 768] in L1.
        """
        B, S, D = x.shape[0], x.shape[2], x.shape[3]
        one = vision_act_memory_config(x.device(), 1, S, D)
        outs = []
        for b in range(B):
            xb = ttnn.slice(x, [b, 0, 0, 0], [b + 1, 1, S, D], memory_config=one)
            # tuned=False: batched forwards keep the E5b configs (the E10.1 tuned table is batch 1 only)
            outs.append(self._forward(xb, memory_config=one, tuned=False))
            ttnn.deallocate(xb)
        out = ttnn.concat(outs, dim=0, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            ttnn.deallocate(t)
        return out

    def _forward(self, x: ttnn.Tensor, memory_config=None, tuned: bool = True) -> ttnn.Tensor:
        """`tuned`: use the per-grid tuned entries (roles, the "h" placement); False for the per-image
        slices of a batched forward."""
        B, S = x.shape[0], x.shape[2]
        I = self.config.intermediate_size
        dev = x.device()
        # up / gate outputs in L1 for one image (-1.3 ms at 1 x 2560); h in L1 gave nothing (E5b screen)
        mem_ug = vision_act_memory_config(dev, B, S, I, wide="mlp")
        # h: the "h" wide cap is a per-grid tuned entry (VISION_L1_WIDE_TUNED, with the 2D down config of
        # VISION_TUNED); without one the plain rule (DRAM at 2560 patches)
        mem = vision_act_memory_config(dev, B, S, I, wide="h" if tuned else None)

        def role(name):
            return name if tuned else None

        up = vision_linear(x, self.w_up, memory_config=mem_ug, policy=self.gate_up_policy, role=role("up"))
        gate = vision_linear(x, self.w_gate, memory_config=mem_ug, policy=self.gate_up_policy, role=role("gate"))
        h = ttnn.multiply(gate, up, memory_config=mem, input_tensor_a_activations=_GELU)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = vision_linear(
            h,
            self.w_down,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            policy=self.down_policy,
            role=role("down"),
        )
        ttnn.deallocate(h)
        return out
