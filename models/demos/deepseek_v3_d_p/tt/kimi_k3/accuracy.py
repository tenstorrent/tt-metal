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
                             MoE head's shared expert + latent down and the tail's latent up
    K3_ACC_KV                "bf16": the DSA layers' kvpe cache in bf16 (BF16_RM) instead of scaled FP8
    K3_ACC_FIDELITY          HiFi4 with fp32 DEST accumulation where the op takes a compute config: KDA input /
                             output projections and gated norm output (fp32), MLA projections, sparse SDPA (the
                             approximate exp kept), the indexer's score matmul and key LayerNorm, the MoE head's
                             shared expert and the tail's latent up; routed experts' x / h in bf16 (not bf8); the
                             MoE combine without its 2^-14 snap of every routed term (fp32 sums)

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


def kv_bf16() -> bool:
    """K3_ACC_KV=bf16: bf16 kvpe cache on the DSA layers."""
    v = knob("KV")
    assert v in ("", "bf16"), f"K3_ACC_KV={v!r}: only 'bf16'"
    return v == "bf16"
