# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Determinism wrappers for the matmul op tests.

Each ``ttnn_<op>`` helper runs ``ttnn.<op>`` (or the ``ttnn.experimental.<op>``
equivalent) twice with the same inputs, asserts the two runs produce identical
outputs, and returns the first one (a drop-in replacement for the original call).

The equality check handles single tensors, tuples/lists of tensors, ``None``
(ops invoked with a preallocated ``output_tensor=`` may return ``None``), integer
outputs, and treats NaNs in matching positions as equal (``torch.equal`` reports
NaN == NaN as False, so a re-run producing bit-identical output still compares
equal).
"""

from tt_py_test_utils_common.matmul_determinism import _assert_outputs_equal, _run_twice, ttnn_linear

import torch
import ttnn


def ttnn_matmul(*args, **kwargs):
    return _run_twice(ttnn.matmul, *args, **kwargs)


def ttnn_attn_matmul(*args, **kwargs):
    return _run_twice(ttnn.experimental.attn_matmul, *args, **kwargs)


def ttnn_group_attn_matmul(*args, **kwargs):
    return _run_twice(ttnn.experimental.group_attn_matmul, *args, **kwargs)


def ttnn_attn_matmul_from_cache(*args, **kwargs):
    return _run_twice(ttnn.experimental.attn_matmul_from_cache, *args, **kwargs)
