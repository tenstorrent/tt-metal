# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Determinism wrappers for the reduction op tests.

Each ``ttnn_<op>`` helper runs ``ttnn.<op>`` twice with the same inputs, asserts
the two runs produce identical outputs, and returns the first one (a drop-in
replacement for the original ``ttnn.<op>`` call).

The equality check is deliberately stricter / more tolerant than ``torch.equal``:
  * tuples/lists are compared element-wise (e.g. ``ttnn.topk`` -> (values, indices)),
  * ``None`` is accepted (ops invoked with a preallocated ``output_tensor=`` /
    ``out=`` may return ``None``),
  * integer outputs (e.g. ``argmax`` / ``topk`` indices) use exact equality,
  * floating outputs treat NaNs in matching positions as equal, since reduction
    outputs may legitimately contain NaN (e.g. ``prod`` over NaN tile padding) and
    ``torch.equal`` reports NaN == NaN as False.
"""

import torch
import ttnn


def _assert_outputs_equal(output1, output2):
    if output1 is None or output2 is None:
        assert output1 is None and output2 is None, "one run returned None, the other did not"
        return
    if isinstance(output1, (tuple, list)):
        assert len(output1) == len(output2)
        for a, b in zip(output1, output2):
            _assert_outputs_equal(a, b)
        return
    a = output1 if isinstance(output1, torch.Tensor) else ttnn.to_torch(output1)
    b = output2 if isinstance(output2, torch.Tensor) else ttnn.to_torch(output2)
    assert a.shape == b.shape, f"shape mismatch between the two runs: {a.shape} vs {b.shape}"
    if a.is_floating_point():
        both_nan = a.isnan() & b.isnan()
        assert bool(torch.all((a == b) | both_nan)), "the two runs produced different outputs"
    else:
        assert torch.equal(a, b), "the two runs produced different outputs"


def _snapshot(output):
    """Deep-copy ``output`` to host so it survives a later in-place overwrite.

    Returns ``None`` for ``None``, recurses into tuples/lists, and otherwise converts the
    (ttnn) tensor to a cloned torch tensor.
    """
    if output is None:
        return None
    if isinstance(output, (tuple, list)):
        return type(output)(_snapshot(x) for x in output)
    return ttnn.to_torch(output).clone()


def _run_twice(op, *args, **kwargs):
    output1 = op(*args, **kwargs)
    output2 = op(*args, **kwargs)
    _assert_outputs_equal(output1, output2)
    return output1


def _run_twice_into(op, prealloc, out_kwarg, *args, **kwargs):
    """Run ``op`` twice writing into the preallocated buffer(s) ``prealloc`` (passed via the
    ``out_kwarg`` keyword, e.g. ``output_tensor`` or ``out``), asserting both runs leave
    identical contents in the buffer.

    Both runs target the same memory, so comparing the live buffer to itself (or comparing
    return values that alias it) would be vacuous. Instead the buffer is snapshotted to host
    after the first run, before the second run overwrites it, and the snapshot is compared
    against the buffer's contents after the second run.
    """
    op(*args, **{out_kwarg: prealloc}, **kwargs)
    snapshot1 = _snapshot(prealloc)
    op(*args, **{out_kwarg: prealloc}, **kwargs)
    _assert_outputs_equal(snapshot1, prealloc)
    return prealloc


def ttnn_sum(*args, **kwargs):
    return _run_twice(ttnn.sum, *args, **kwargs)


def ttnn_mean(*args, **kwargs):
    return _run_twice(ttnn.mean, *args, **kwargs)


def ttnn_max(*args, **kwargs):
    return _run_twice(ttnn.max, *args, **kwargs)


def ttnn_min(*args, **kwargs):
    return _run_twice(ttnn.min, *args, **kwargs)


def ttnn_prod(*args, **kwargs):
    return _run_twice(ttnn.prod, *args, **kwargs)


def ttnn_std(*args, **kwargs):
    return _run_twice(ttnn.std, *args, **kwargs)


def ttnn_var(*args, **kwargs):
    return _run_twice(ttnn.var, *args, **kwargs)


def ttnn_cumsum(*args, **kwargs):
    return _run_twice(ttnn.cumsum, *args, **kwargs)


def ttnn_cumprod(*args, **kwargs):
    return _run_twice(ttnn.cumprod, *args, **kwargs)


def ttnn_topk(*args, **kwargs):
    return _run_twice(ttnn.topk, *args, **kwargs)


def ttnn_argmax(*args, **kwargs):
    return _run_twice(ttnn.argmax, *args, **kwargs)


def ttnn_moe(*args, **kwargs):
    return _run_twice(ttnn.moe, *args, **kwargs)


def ttnn_sampling(*args, **kwargs):
    return _run_twice(ttnn.sampling, *args, **kwargs)


def ttnn_deepseek_grouped_gate(*args, **kwargs):
    return _run_twice(ttnn.experimental.deepseek_grouped_gate, *args, **kwargs)


# Preallocated-output variants: each runs the op twice into the same preallocated buffer and
# asserts determinism by snapshotting the buffer between runs. ``prealloc`` is the buffer (or
# tuple of buffers, e.g. ``ttnn.topk`` -> (values, indices)) and is returned after both runs.
def ttnn_topk_preallocated(prealloc, *args, **kwargs):
    return _run_twice_into(ttnn.topk, prealloc, "output_tensor", *args, **kwargs)


def ttnn_argmax_preallocated(prealloc, *args, **kwargs):
    return _run_twice_into(ttnn.argmax, prealloc, "output_tensor", *args, **kwargs)


def ttnn_cumsum_preallocated(prealloc, *args, **kwargs):
    return _run_twice_into(ttnn.cumsum, prealloc, "out", *args, **kwargs)


def ttnn_cumprod_preallocated(prealloc, *args, **kwargs):
    return _run_twice_into(ttnn.cumprod, prealloc, "out", *args, **kwargs)


def ttnn_moe_preallocated(prealloc, *args, **kwargs):
    return _run_twice_into(ttnn.moe, prealloc, "output_tensor", *args, **kwargs)


def ttnn_sampling_preallocated(prealloc, *args, **kwargs):
    return _run_twice_into(ttnn.sampling, prealloc, "output_tensor", *args, **kwargs)


# Maps the parametrized ``op`` name to its determinism wrapper, for tests that
# dispatch dynamically (previously via ``getattr(ttnn, op)``).
TTNN_REDUCTION_WRAPPERS = {
    "sum": ttnn_sum,
    "mean": ttnn_mean,
    "max": ttnn_max,
    "min": ttnn_min,
    "prod": ttnn_prod,
    "std": ttnn_std,
    "var": ttnn_var,
    "cumsum": ttnn_cumsum,
    "cumprod": ttnn_cumprod,
}

# Same mapping for the preallocated-output (``out=``) accumulation variants.
TTNN_REDUCTION_PREALLOCATED_WRAPPERS = {
    "cumsum": ttnn_cumsum_preallocated,
    "cumprod": ttnn_cumprod_preallocated,
}
