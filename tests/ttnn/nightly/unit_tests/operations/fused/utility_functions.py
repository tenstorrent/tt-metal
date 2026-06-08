# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Determinism wrappers for the fused (norm / softmax) op tests.

Each ``ttnn_<op>`` helper runs ``ttnn.<op>`` twice with the same inputs, asserts
the two runs produce identical outputs, and returns the first one (a drop-in
replacement for the original ``ttnn.<op>`` call).

In-place ops mutate their input tensor, so the ``*_in_place`` helpers clone the
input before each run to give both runs identical inputs.
"""

import torch
import ttnn


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


def ttnn_rms_norm(*args, **kwargs):
    return _run_twice(ttnn.rms_norm, *args, **kwargs)


def ttnn_group_norm(*args, **kwargs):
    return _run_twice(ttnn.group_norm, *args, **kwargs)


def ttnn_layer_norm_pre_all_gather(*args, **kwargs):
    return _run_twice(ttnn.layer_norm_pre_all_gather, *args, **kwargs)


def ttnn_layer_norm_post_all_gather(*args, **kwargs):
    return _run_twice(ttnn.layer_norm_post_all_gather, *args, **kwargs)


def ttnn_rms_norm_pre_all_gather(*args, **kwargs):
    return _run_twice(ttnn.rms_norm_pre_all_gather, *args, **kwargs)


def ttnn_rms_norm_post_all_gather(*args, **kwargs):
    return _run_twice(ttnn.rms_norm_post_all_gather, *args, **kwargs)
