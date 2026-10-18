# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the fused (norm / softmax) op tests.

Determinism wrappers: each ``ttnn_<op>`` helper runs ``ttnn.<op>`` twice with the
same inputs, asserts the two runs produce identical outputs, and returns the first
one (a drop-in replacement for the original ``ttnn.<op>`` call).

In-place ops mutate their input tensor, so the ``*_in_place`` helpers clone the
input before each run to give both runs identical inputs.

Also holds the shared ``test_id`` x gamma/beta dtype grid used by the mix-precision
layernorm tests (see ``MIX_PRECISION_TEST_IDS``).
"""

import torch
import ttnn

# ---------------------------------------------------------------------------
# Shared test_id x gamma/beta dtype grid for the mix-precision layernorm tests.
#
# The dtype is fused into ``test_id``, alternating across the residual / no-residual pair so that
# both formats appear in every {LN, RMSN} x {G, GB} category (each test_id runs exactly one dtype):
#
#   test_id   0     1      2       3      4       5        6    7     8      9     10     11
#   name      LN    LN_G   LN_GB   RMSN   RMSN_G  RMSN_GB  LN   LN_G  LN_GB  RMSN  RMSN_G RMSN_GB
#             |<---------- residual (add_*) ---------->|   |<--------- no residual --------->|
#   gamma     bf16  bf16   fp32    fp32   fp32    bf16     bf16 fp32  bf16   fp32  bf16   fp32
#
# Down the residual pairs: LN_G bf16/fp32, LN_GB fp32/bf16, RMSN_G fp32/bf16, RMSN_GB bf16/fp32.
#
# ---------------------------------------------------------------------------

_MIX_PRECISION_TEST_ID_NAMES = (
    "add_LN",
    "add_LN_G",
    "add_LN_GB",
    "add_RMSN",
    "add_RMSN_G",
    "add_RMSN_GB",
    "LN",
    "LN_G",
    "LN_GB",
    "RMSN",
    "RMSN_G",
    "RMSN_GB",
)

_MIX_PRECISION_GAMMA_DTYPES = (
    ttnn.bfloat16,  # 0  add_LN     (no gamma passed)
    ttnn.bfloat16,  # 1  add_LN_G
    ttnn.float32,  # 2  add_LN_GB
    ttnn.float32,  # 3  add_RMSN   (no gamma passed)
    ttnn.float32,  # 4  add_RMSN_G
    ttnn.bfloat16,  # 5  add_RMSN_GB
    ttnn.bfloat16,  # 6  LN        (no gamma passed)
    ttnn.float32,  # 7  LN_G
    ttnn.bfloat16,  # 8  LN_GB
    ttnn.float32,  # 9  RMSN      (no gamma passed)
    ttnn.bfloat16,  # 10 RMSN_G
    ttnn.float32,  # 11 RMSN_GB
)


def _mix_precision_id(name, dtype):
    """Test id for one (test_id, dtype) pair; cases that never pass gamma/beta are labelled no_gb."""
    if not name.endswith(("_G", "_GB")):
        return f"{name}-no_gb"
    return f"{name}-{'gb_fp32' if dtype == ttnn.float32 else 'gb_bf16'}"


MIX_PRECISION_TEST_IDS = tuple(enumerate(_MIX_PRECISION_GAMMA_DTYPES))
MIX_PRECISION_TEST_ID_NAMES = tuple(
    _mix_precision_id(name, dtype) for name, dtype in zip(_MIX_PRECISION_TEST_ID_NAMES, _MIX_PRECISION_GAMMA_DTYPES)
)


def _run_twice(op, *args, **kwargs):
    output1 = op(*args, **kwargs)
    output2 = op(*args, **kwargs)
    assert torch.equal(ttnn.to_torch(output1), ttnn.to_torch(output2))
    return output1


def _run_twice_in_place(op, input_tensor, *args, **kwargs):
    output1 = op(ttnn.clone(input_tensor), *args, **kwargs)
    output2 = op(ttnn.clone(input_tensor), *args, **kwargs)
    assert torch.equal(ttnn.to_torch(output1), ttnn.to_torch(output2))
    return output1


def ttnn_softmax(*args, **kwargs):
    return _run_twice(ttnn.softmax, *args, **kwargs)


def ttnn_scale_mask_softmax(*args, **kwargs):
    return _run_twice(ttnn.scale_mask_softmax, *args, **kwargs)


def ttnn_softmax_in_place(input_tensor, *args, **kwargs):
    return _run_twice_in_place(ttnn.softmax_in_place, input_tensor, *args, **kwargs)


def ttnn_scale_mask_softmax_in_place(input_tensor, *args, **kwargs):
    return _run_twice_in_place(ttnn.scale_mask_softmax_in_place, input_tensor, *args, **kwargs)


def ttnn_layer_norm(*args, **kwargs):
    return _run_twice(ttnn.layer_norm, *args, **kwargs)


def ttnn_layer_norm_in_place(input_tensor, *args, **kwargs):
    # A sharded program_config with ``inplace=True`` makes ttnn.layer_norm write its result back
    # into ``input_tensor``; clone the input before each run so both runs see identical inputs.
    return _run_twice_in_place(ttnn.layer_norm, input_tensor, *args, **kwargs)


def ttnn_rms_norm(*args, **kwargs):
    return _run_twice(ttnn.rms_norm, *args, **kwargs)


def ttnn_rms_norm_in_place(input_tensor, *args, **kwargs):
    # A sharded program_config with ``inplace=True`` makes ttnn.rms_norm write its result back
    # into ``input_tensor``; clone the input before each run so both runs see identical inputs.
    return _run_twice_in_place(ttnn.rms_norm, input_tensor, *args, **kwargs)


def ttnn_group_norm(*args, **kwargs):
    return _run_twice(ttnn.group_norm, *args, **kwargs)


def ttnn_group_norm_in_place(input_tensor, *args, **kwargs):
    # ``ttnn.group_norm`` defaults to ``inplace=True``, which mutates ``input_tensor``; clone
    # the input before each run so both runs see identical (unmutated) inputs.
    return _run_twice_in_place(ttnn.group_norm, input_tensor, *args, **kwargs)


def ttnn_layer_norm_pre_all_gather(*args, **kwargs):
    return _run_twice(ttnn.layer_norm_pre_all_gather, *args, **kwargs)


def ttnn_layer_norm_post_all_gather(*args, **kwargs):
    return _run_twice(ttnn.layer_norm_post_all_gather, *args, **kwargs)


def ttnn_rms_norm_pre_all_gather(*args, **kwargs):
    return _run_twice(ttnn.rms_norm_pre_all_gather, *args, **kwargs)


def ttnn_rms_norm_post_all_gather(*args, **kwargs):
    return _run_twice(ttnn.rms_norm_post_all_gather, *args, **kwargs)
