# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Name/shape mapping between speculators DFlash state dicts and the ttml drafter.

ttml parameter names (``AbstractModuleBase.named_parameters``) equal speculators names
except that each layer's ``self_attn.k_proj.weight`` and ``self_attn.v_proj.weight`` are
stacked into ``self_attn.kv_proj.weight`` (K rows first), and every tensor is rank 4:
linear ``[out, in] -> [1, 1, out, in]``, norm ``[H] -> [1, 1, 1, H]``, embedding
``[V, H] -> [1, 1, V, H]``.

The numpy-only helpers have no ttnn dependency; device helpers import ttml lazily.
"""

from __future__ import annotations

import re
from typing import Mapping

import numpy as np

# speculators keys with no ttml counterpart. verifier_lm_head is a frozen copy of
# lm_head (speculators load_verifier_weights); the ttml teacher reuses lm_head.
IGNORED_SPECULATORS_KEYS = ("verifier_lm_head.weight", "t2d", "d2t")
_K = re.compile(r"^(layers\.\d+\.self_attn)\.k_proj\.weight$")
_V = re.compile(r"^(layers\.\d+\.self_attn)\.v_proj\.weight$")
_KV = re.compile(r"^(layers\.\d+\.self_attn)\.kv_proj\.weight$")


def _rank4(x: np.ndarray) -> np.ndarray:
    return x.reshape((1,) * (4 - x.ndim) + x.shape)


def speculators_to_ttml(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """speculators state dict (numpy) -> ttml names, rank-4 float32 arrays."""
    if "verifier_lm_head.weight" in state and "lm_head.weight" in state:
        if not np.array_equal(state["verifier_lm_head.weight"], state["lm_head.weight"]):
            raise ValueError("verifier_lm_head differs from lm_head; the ttml teacher reuses lm_head")
    for key in ("input_proj.weight", "output_proj.weight"):
        if key in state:
            raise NotImplementedError(f"{key}: hidden-size projections are not ported")
    out: dict[str, np.ndarray] = {}
    kv: dict[str, dict[str, np.ndarray]] = {}
    for name, value in state.items():
        if name in IGNORED_SPECULATORS_KEYS:
            continue
        value = np.asarray(value, dtype=np.float32)
        if m := _K.match(name):
            kv.setdefault(m.group(1), {})["k"] = value
        elif m := _V.match(name):
            kv.setdefault(m.group(1), {})["v"] = value
        else:
            out[name] = _rank4(value)
    for prefix, parts in kv.items():
        if set(parts) != {"k", "v"}:
            raise ValueError(f"{prefix}: need both k_proj and v_proj")
        out[f"{prefix}.kv_proj.weight"] = _rank4(np.concatenate([parts["k"], parts["v"]], axis=0))
    return out


def ttml_to_speculators(
    named: Mapping[str, np.ndarray], spec_shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    """ttml-named arrays (weights or grads) -> speculators names and shapes."""
    out: dict[str, np.ndarray] = {}
    for name, value in named.items():
        value = np.asarray(value)
        if m := _KV.match(name):
            k_shape = spec_shapes[f"{m.group(1)}.k_proj.weight"]
            v_shape = spec_shapes[f"{m.group(1)}.v_proj.weight"]
            flat = value.reshape(-1, value.shape[-1])
            out[f"{m.group(1)}.k_proj.weight"] = flat[: k_shape[0]].reshape(k_shape)
            out[f"{m.group(1)}.v_proj.weight"] = flat[k_shape[0] :].reshape(v_shape)
        elif name in spec_shapes:
            out[name] = value.reshape(spec_shapes[name])
        else:
            raise KeyError(f"{name}: no speculators counterpart")
    return out


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def named_parameters(model) -> dict:
    """``{dotted name: ttml.autograd.Tensor}`` for every parameter of ``model``."""
    return dict(model.named_parameters())


def load_speculators_weights(model, state: Mapping[str, np.ndarray]) -> None:
    """Assign a speculators state dict (numpy) to the ttml drafter.

    Must run before the optimizer is built: ``AdamWFullPrecision`` snapshots its fp32
    master weights from the bf16 parameters at construction.
    """
    from ttml.autograd.numpy_assign import assign_numpy  # noqa: PLC0415

    params = named_parameters(model)
    mapped = speculators_to_ttml(state)
    missing = sorted(set(params) - set(mapped))
    unexpected = sorted(set(mapped) - set(params))
    if missing or unexpected:
        raise KeyError(f"weight mapping mismatch: missing={missing} unexpected={unexpected}")
    for name, tensor in params.items():
        assign_numpy(tensor, mapped[name])


def read_parameters(model) -> dict[str, np.ndarray]:
    """Current (bf16) parameter values as float32 numpy, ttml names."""
    import ttnn  # noqa: PLC0415

    return {n: np.asarray(t.to_numpy(ttnn.DataType.FLOAT32), np.float32) for n, t in named_parameters(model).items()}


def read_gradients(model, strict: bool = True) -> dict[str, np.ndarray]:
    """Accumulated gradients of trainable parameters as float32 numpy, ttml names.

    A trainable parameter with no gradient means it is disconnected from the loss:
    ``strict`` raises; otherwise it is left out of the result for the caller to report.
    """
    import ttnn  # noqa: PLC0415

    import ttml  # noqa: PLC0415

    out = {}
    for name, t in named_parameters(model).items():
        if not t.get_requires_grad():
            continue
        if not t.is_grad_initialized():
            if strict:
                raise RuntimeError(f"{name}: trainable parameter received no gradient")
            continue
        grad = ttml.autograd.Tensor(t.get_grad(), False)
        out[name] = np.asarray(grad.to_numpy(ttnn.DataType.FLOAT32), np.float32)
    return out


def read_master_weights(optimizer) -> dict[str, np.ndarray]:
    """fp32 master weights of ``AdamWFullPrecision``, ttml dotted names."""
    import ttnn  # noqa: PLC0415

    masters = optimizer.get_state_dict()["master_weights"]
    return {
        name.split("/", 1)[-1].replace("/", "."): np.asarray(t.to_numpy(ttnn.DataType.FLOAT32), np.float32)
        for name, t in masters.items()
    }
