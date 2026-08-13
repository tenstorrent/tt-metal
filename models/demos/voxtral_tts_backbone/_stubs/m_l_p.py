# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `m_l_p` (`MistralMLP`) for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

    down_proj(silu(gate_proj(x)) * up_proj(x))     # 3072 -> 9216 -> 3072

This is the canonical home of the layer's SwiGLU: `_stubs/decoder_layer.py`
imports `TtMLP` from here rather than keeping a second copy, so the composite
and the standalone component can't drift.

The plan's reuse target, `models/tt_transformers/tt/mlp.py::MLP`, is not
constructible from this per-component harness — it needs the full `ModelArgs`
plumbing (`tt_ccl`, `weight_cache_path`, `model_config`) and its forward is
`forward(x, mode)` with its own sharded memory configs. The forward below uses
the same ttnn primitives that module does.

`build` stages weights with torch (they come from an HF checkpoint); `__call__`
is pure ttnn — `models/common/native_probe.py` counts what actually executes.
"""
from __future__ import annotations

import ttnn

from models.demos.voxtral_tts_backbone._stubs.attention import _stage
from models.demos.voxtral_tts_backbone._stubs.decode_matmul import build_plan


class TtMLP:
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, device, weights, compute_kernel_config=None):
        self.device = device
        self.w_gate, self.w_up, self.w_down = weights
        self.compute_kernel_config = compute_kernel_config
        # down_proj is the widest read in the block (9216 -> 3072) and the one
        # `ttnn.linear` routes worst on its own, so its decode call gets an
        # explicit full-grid plan. The weight layout is untouched.
        self.down_plan = build_plan(device, int(self.w_down.shape[-2]), int(self.w_down.shape[-1]))

    @classmethod
    def build(cls, device, torch_module, compute_kernel_config=None):
        if torch_module is None:
            raise RuntimeError("m_l_p build needs the HF MistralMLP module to read weights from")
        act = getattr(torch_module, "act_fn", None)
        act_name = type(act).__name__.lower() if act is not None else ""
        if "silu" not in act_name and "swish" not in act_name:
            raise RuntimeError(
                f"MLP activation is {act_name or '<unknown>'}, not SiLU; this port implements SwiGLU only"
            )
        if compute_kernel_config is None:
            compute_kernel_config = _default_compute_kernel_config(device)
        weights = (
            _stage(torch_module.gate_proj.weight, device),
            _stage(torch_module.up_proj.weight, device),
            _stage(torch_module.down_proj.weight, device),
        )
        return cls(device, weights, compute_kernel_config)

    def __call__(self, x, *_args, **_ignored):
        mm = {"compute_kernel_config": self.compute_kernel_config} if self.compute_kernel_config else {}
        gate = ttnn.silu(ttnn.linear(x, self.w_gate, **mm))
        up = ttnn.linear(x, self.w_up, **mm)
        hidden = ttnn.multiply(gate, up)
        if self.down_plan is not None and self.down_plan.matches(hidden):
            return self.down_plan(hidden, self.w_down, self.compute_kernel_config)
        return ttnn.linear(hidden, self.w_down, **mm)


def _default_compute_kernel_config(device):
    """HiFi4 + fp32 accumulate: the 9216-deep down_proj reduction is where bf16
    LoFi accumulation would cost the PCC digits this gate asks for."""
    try:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    except Exception:  # noqa: BLE001 - accuracy tuning is best-effort
        return None


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtMLP.build(device, torch_module)


# Legacy slug-named shim.
def m_l_p(device, torch_module=None):
    return TtMLP.build(device, torch_module)
