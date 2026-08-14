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
        # explicit plan. The weight layout is untouched.
        #
        # 24 cores, NOT the whole grid. This op is not bandwidth-bound -- it
        # takes the same 0.159 ms whether its weight is bf16 or bf8_b -- so what
        # limits it is the shape of the k-reduction, and spreading it wider makes
        # that worse: at 96 cores each core walks 96 sequential k-blocks
        # (in0_block_w=3) to produce ONE output tile, with nothing to overlap the
        # reduction against. Fewer, fatter shards win by a lot. Measured per call
        # at bf8_b: 96c 0.1589 / 48c 0.0960 / 32c 0.1137 / 24c 0.0934 / 16c
        # 0.1084 / 12c 0.1342 / 8c 0.1964 ms.
        self.down_plan = build_plan(
            device, int(self.w_down.shape[-2]), int(self.w_down.shape[-1]), max_cores=24
        )
        # gate/up (3072 -> 9216) previously kept the DEFAULT routing: its N is 288
        # tiles, so `ttnn.linear` already spreads it well, and a sweep of seven
        # core counts lost every time -- "the plan's two reshard ops cost more
        # than they buy".
        #
        # That accounting no longer holds. The norm ahead of this block now emits
        # its result IN a 48-core width shard, and gate and up read the SAME
        # activation, so the plan's input reshard is not two ops or even one: it
        # is zero. What is left is the matmul routing on its own, which is the
        # part the old sweep could never see separately.
        self.gate_up_plan = build_plan(
            device, int(self.w_gate.shape[-2]), int(self.w_gate.shape[-1]), max_cores=48
        )

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
        # All three MLP weights are STORED as bfloat8_b: together they are 170 MB
        # of the ~233 MB a layer streams per decoded token, and decode is bound on
        # exactly those bytes. down_proj is the accuracy-sensitive one (a
        # 9216-deep reduction), so it is walked down after gate/up and gated on
        # full-model PCC rather than assumed safe.
        weights = (
            _stage(torch_module.gate_proj.weight, device, dtype=ttnn.bfloat8_b),
            _stage(torch_module.up_proj.weight, device, dtype=ttnn.bfloat8_b),
            _stage(torch_module.down_proj.weight, device, dtype=ttnn.bfloat8_b),
        )
        return cls(device, weights, compute_kernel_config)

    def __call__(self, x, *_args, **_ignored):
        mm = {"compute_kernel_config": self.compute_kernel_config} if self.compute_kernel_config else {}
        # SiLU rides on the MULTIPLY, not on the gate projection. The product is
        # the activation's only consumer, and the binary op can apply a unary to
        # an input as it reads it -- so `silu(gate) * up` is one launch, and the
        # standalone unary, which fetched 9216 values back out of DRAM and wrote
        # them again, is gone. 26 launches per token.
        #
        # NOT via `ttnn.linear(activation="silu")`: with no program config to put
        # it in, ttnn appends a `unary_chain` op -- the same launch renamed --
        # and naming a core grid to reach the fused path costs far more than the
        # unary did (gate/up 43.35 -> 57.50 ms on the same 96 cores, measured).
        if self.gate_up_plan is not None and self.gate_up_plan.matches(x):
            # ONE shard for both: gate and up read the same activation, so the
            # conversion is opened once -- and when the norm already emitted
            # this exact layout it is not opened at all.
            shared = self.gate_up_plan.shard_input(x)
            gate = self.gate_up_plan.run_presharded_raw(shared, self.w_gate, self.compute_kernel_config)
            up = self.gate_up_plan.run_presharded_raw(shared, self.w_up, self.compute_kernel_config)
        else:
            gate = ttnn.linear(x, self.w_gate, **mm)
            up = ttnn.linear(x, self.w_up, **mm)
        swiglu = {"input_tensor_a_activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]}
        if self.down_plan is not None and self.down_plan.is_decode_row(x):
            # The SwiGLU product is down_proj's only consumer, so it is written
            # STRAIGHT into the layout down_proj wants. Producing it interleaved
            # and converting after is the same bytes plus a whole extra op.
            hidden = ttnn.multiply(gate, up, memory_config=self.down_plan.input_memory_config, **swiglu)
            return self.down_plan.run_presharded(hidden, self.w_down, self.compute_kernel_config)
        hidden = ttnn.multiply(gate, up, **swiglu)
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
