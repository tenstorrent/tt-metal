# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in accuracy knobs for Kimi-K3 / K3.1 prefill: a numerically accurate reference ("teacher") run.

Every knob is off by default, and off is production: no op, dtype, config or cache key changes unless a knob is set.
They are read from the environment each time a module asks, so set them before the model is built.

    K3_ACC=1                 every knob below at its "on" value (each K3_ACC_<NAME> still overrides it)
    K3_ACC_ROUTED_BF8        routed experts in bfloat8_b, read from the bf16-dequantized checkpoint (bf8 holds the
                             MXFP4 values exactly where a 16-row block's scales span < 4 octaves): "1" every MoE layer,
                             or a comma list of layer indices
    K3_ACC_BF16_WEIGHTS      bf16 for every weight that is bf16 in the published checkpoint: KDA input / output
                             projections, MLA q_a / q_b / kv_a / kv_b / o / g projections, indexer wq_b / wk, the
                             MoE head's shared expert + latent down and the tail's latent up, layer 0's dense MLP
    K3_ACC_KV                "bf16": the DSA layers' kvpe cache in bf16 (BF16_RM) instead of scaled FP8
    K3_ACC_FIDELITY          HiFi4 where the op takes a compute config: KDA input / output projections, scan and
                             affine prefix, gated norm output in fp32; MLA projections (fp32 DEST only where the
                             tuned config already had it: ttMLA's default config keeps bf16 DEST) with bf16 instead
                             of bf8 outputs; sparse SDPA with fp32 DEST (the approximate exp kept); the indexer's
                             score matmul and key LayerNorm; layer 0's dense MLP (fp32 DEST); the MoE head's
                             shared expert and the tail's latent up; routed experts' x / h in bf16 (not bf8); the
                             MoE combine without its 2^-14 snap of every routed term (fp32 sums)
    K3_ACC_ROUTED_FIDELITY   the routed experts' matmul fidelity (on: HiFi3; or a MathFidelity name). Their default
                             LoFi truncates the 8th significant bit of the activation operand, exact for production's
                             bfp8 x / h but biased low (~0.27 % a matmul, ~0.9 % on the routed sum) for the bf16 x / h
                             of K3_ACC_FIDELITY; HiFi3 multiplies that bit too (the MXFP4 weights fit SrcA's high part)
    K3_ACC_COMBINE_SNAP      under K3_ACC_FIDELITY, the MoE combine's snap step instead of none: "1" or a step (on:
                             2^-20). The routed terms then sum as an int32 count of the step, bit-identical in any
                             arrival order, so K3_ACC runs are deterministic; at 2^-20 it costs nothing (A2: every
                             score equal to 0.01 dB with the snap off). "0" turns it off (fp32 sums, arrival order)
    K3_ACC_MLA_FP32          MLA (and DSA indexer) accumulation in fp32 everywhere: ttMLA's default compute config with
                             fp32 DEST at HiFi4 (tuned matmul subblocks clipped to <= 4 tiles), the fused matmul +
                             reduce-scatter ops (q_a, o_proj, indexer wk / weights_proj; fp32 already) at HiFi4 instead
                             of HiFi2, and kv_a's TP partials in fp32 through the gather and the reduction

The weight-dtype knobs change the TTNN cache file names (as_tensor encodes the dtype), so a layer built with them
needs a writable weight cache (it is built from the checkpoint the first time); the production cache is never
overwritten, only missed.
"""

from __future__ import annotations

import os

import ttnn

_ALL = "K3_ACC"
_ON = {
    "ROUTED_BF8": "1",
    "BF16_WEIGHTS": "1",
    "KV": "bf16",
    "FIDELITY": "1",
    "MLA_FP32": "1",
    "COMBINE_SNAP": "9.5367431640625e-07",
    "ROUTED_FIDELITY": "HiFi3",
}


def knob(name: str) -> str:
    """K3_ACC_<name>'s value: "" when off (unset, empty or "0"), else its value; K3_ACC=1 turns unset knobs on."""
    assert name in _ON, f"no accuracy knob {name!r} ({sorted(_ON)})"
    v = os.environ.get(f"{_ALL}_{name}")
    if v is None and os.environ.get(_ALL, "") not in ("", "0"):
        v = _ON[name]
    return "" if v in (None, "", "0") else v


def any_on() -> bool:
    """Whether any accuracy knob is set (a run that is not production)."""
    return any(knob(n) for n in _ON)


def summary() -> dict:
    """The knobs that are on, {name: value} (for logs and result files)."""
    return {n: knob(n) for n in _ON if knob(n)}


def routed_bf8(layer_idx: int) -> bool:
    """Whether this MoE layer's routed experts are bfloat8_b (K3_ACC_ROUTED_BF8)."""
    v = knob("ROUTED_BF8")
    return bool(v) and (v == "1" or str(int(layer_idx)) in v.split(","))


def bf16_weights() -> bool:
    """K3_ACC_BF16_WEIGHTS: the checkpoint's bf16 weights stay bf16 on device."""
    return bool(knob("BF16_WEIGHTS"))


def weight_dtype(production: ttnn.DataType) -> ttnn.DataType:
    """A bf16-origin weight's device dtype: bf16 under K3_ACC_BF16_WEIGHTS, else the production dtype."""
    return ttnn.bfloat16 if bf16_weights() else production


def fidelity() -> bool:
    """K3_ACC_FIDELITY: HiFi4 + fp32 DEST where wired."""
    return bool(knob("FIDELITY"))


def math_fidelity(production: ttnn.MathFidelity) -> ttnn.MathFidelity:
    """HiFi4 under K3_ACC_FIDELITY, else the production fidelity."""
    return ttnn.MathFidelity.HiFi4 if fidelity() else production


def combine_snap() -> float:
    """K3_ACC_COMBINE_SNAP's step (0.0: no snap); "1" means the on value, 2^-20."""
    v = knob("COMBINE_SNAP")
    return float(_ON["COMBINE_SNAP"] if v == "1" else v) if v else 0.0


def routed_fidelity():
    """K3_ACC_ROUTED_FIDELITY's ttnn.MathFidelity for the routed experts (None: off, the op's LoFi); "1" means HiFi3."""
    v = knob("ROUTED_FIDELITY")
    if not v:
        return None
    return getattr(ttnn.MathFidelity, _ON["ROUTED_FIDELITY"] if v == "1" else v)


def mla_fp32() -> bool:
    """K3_ACC_MLA_FP32: fp32 accumulation (and HiFi4) throughout ttMLA and the DSA indexer."""
    return bool(knob("MLA_FP32"))


def fp32_dest_program_config(pc):
    """A matmul program config whose output subblock fits fp32 DEST (h * w <= 4 tiles): a copy with the subblock
    shrunk to divisors of the original (so it still divides the output block), or pc itself when it fits / is None."""
    h, w = getattr(pc, "out_subblock_h", None), getattr(pc, "out_subblock_w", None)
    if pc is None or h is None or w is None or h * w <= 4:
        return pc
    best = max(((a, b) for a in range(1, h + 1) if h % a == 0 for b in range(1, w + 1) if w % b == 0 and a * b <= 4),
               key=lambda ab: (ab[0] * ab[1], ab[1]))
    out = type(pc).from_json(pc.to_json())
    out.out_subblock_h, out.out_subblock_w = best
    return out


def mmrs_kwargs(arch) -> dict:
    """matmul_reduce_scatter_prefill's compute_kernel_config under K3_ACC_MLA_FP32 (HiFi4; the op keeps fp32 DEST and
    fp32 partial sums itself), else nothing (its HiFi2 default)."""
    if not mla_fp32():
        return {}
    return {"compute_kernel_config": ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)}


def kv_bf16() -> bool:
    """K3_ACC_KV=bf16: bf16 kvpe cache on the DSA layers."""
    v = knob("KV")
    assert v in ("", "bf16"), f"K3_ACC_KV={v!r}: only 'bf16'"
    return v == "bf16"
