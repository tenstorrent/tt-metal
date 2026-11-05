# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# A head count of 0, or one large enough that the uint32 head-count sum wraps to 0, used to reach an
# integer division on the host, which killed the process with SIGFPE instead of raising.

_UINT32_MAX = 2**32 - 1


def _tile(device, shape):
    return ttnn.from_torch(torch.randn(shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)


def _split_qkv(device, **kwargs):
    ttnn.transformer.split_query_key_value_and_split_heads(_tile(device, (1, 32, 192)), **kwargs)


def _split_qkv_with_kv(device, **kwargs):
    ttnn.transformer.split_query_key_value_and_split_heads(
        _tile(device, (1, 32, 64)), _tile(device, (1, 32, 128)), **kwargs
    )


def _nlp_create_qkv_heads(device, **kwargs):
    ttnn.experimental.nlp_create_qkv_heads(_tile(device, (1, 1, 32, 192)), **kwargs)


def _nlp_create_qkv_heads_with_kv(device, **kwargs):
    ttnn.experimental.nlp_create_qkv_heads(_tile(device, (1, 1, 32, 64)), _tile(device, (1, 1, 32, 128)), **kwargs)


def _nlp_create_qkv_heads_boltz(device, **kwargs):
    ttnn.experimental.nlp_create_qkv_heads_boltz(_tile(device, (1, 1, 32, 192)), **kwargs)


def _nlp_create_qkv_heads_boltz_with_kv(device, **kwargs):
    ttnn.experimental.nlp_create_qkv_heads_boltz(
        _tile(device, (1, 1, 32, 64)), _tile(device, (1, 1, 32, 128)), **kwargs
    )


def _create_qkv_heads(device, **kwargs):
    ttnn.experimental.create_qkv_heads(_tile(device, (1, 1, 32, 192)), **kwargs)


def _create_qkv_heads_from_separate_tensors(device, **kwargs):
    ttnn.experimental.create_qkv_heads_from_separate_tensors(
        _tile(device, (1, 1, 32, 64)), _tile(device, (1, 1, 32, 128)), **kwargs
    )


@pytest.mark.parametrize(
    "run, kwargs, message",
    [
        (_split_qkv, {"num_heads": 0}, "num_heads must be greater than 0"),
        (_split_qkv, {"num_heads": 1, "num_kv_heads": 0}, "num_kv_heads must be greater than 0"),
        (_split_qkv, {"num_heads": 2, "num_kv_heads": _UINT32_MAX}, "heads do not fit in the hidden dimension"),
        (_split_qkv_with_kv, {"num_heads": 1, "num_kv_heads": 2**31}, r"2 \* num_kv_heads .* exceeds"),
        (_nlp_create_qkv_heads, {"num_heads": 0}, "num_q_heads must be greater than 0"),
        (_nlp_create_qkv_heads, {"num_heads": 2, "num_kv_heads": _UINT32_MAX}, "exceeds the fused width"),
        (_nlp_create_qkv_heads_with_kv, {"num_heads": 1, "num_kv_heads": 0}, "num_kv_heads must be greater than 0"),
        (_nlp_create_qkv_heads_with_kv, {"num_heads": 1, "num_kv_heads": 2**31}, "exceeds the KV width"),
        (_nlp_create_qkv_heads, {"num_heads": 0, "kv_tied": True}, "num_q_heads must be greater than 0"),
        (
            _nlp_create_qkv_heads,
            {"num_heads": 2, "num_kv_heads": _UINT32_MAX, "kv_tied": True},
            r"num_q_heads \(2\) \+ num_kv_heads .* exceeds the fused width",
        ),
        (
            _nlp_create_qkv_heads_with_kv,
            {"num_heads": 1, "num_kv_heads": 0, "kv_tied": True},
            "num_kv_heads must be greater than 0",
        ),
        (
            _nlp_create_qkv_heads_with_kv,
            {"num_heads": 1, "num_kv_heads": _UINT32_MAX, "kv_tied": True},
            r"1 \* num_kv_heads .* exceeds the KV width",
        ),
        (_nlp_create_qkv_heads_boltz, {"num_heads": 0}, "num_q_heads must be greater than 0"),
        (_nlp_create_qkv_heads_boltz, {"num_heads": 2, "num_kv_heads": _UINT32_MAX}, "exceeds the fused width"),
        (
            _nlp_create_qkv_heads_boltz_with_kv,
            {"num_heads": 1, "num_kv_heads": 0},
            "num_kv_heads must be greater than 0",
        ),
        (_nlp_create_qkv_heads_boltz_with_kv, {"num_heads": 1, "num_kv_heads": 2**31}, "exceeds the KV width"),
        (_create_qkv_heads, {"num_heads": 0}, "num_q_heads must be greater than 0"),
        (_create_qkv_heads, {"num_heads": 1, "num_kv_heads": 0}, "num_kv_heads must be greater than 0"),
        (_create_qkv_heads, {"num_heads": 2, "num_kv_heads": _UINT32_MAX}, "exceeds the flattened hidden dimension"),
        (_create_qkv_heads_from_separate_tensors, {"num_heads": 0}, "num_q_heads must be greater than 0"),
        (
            _create_qkv_heads_from_separate_tensors,
            {"num_heads": 1, "num_kv_heads": 0},
            "num_kv_heads must be greater than 0",
        ),
        (
            _create_qkv_heads_from_separate_tensors,
            {"num_heads": 1, "num_kv_heads": 2**31},
            "exceeds the KV hidden dimension",
        ),
    ],
    ids=[
        "split_qkv_zero_q",
        "split_qkv_zero_kv",
        "split_qkv_wrap",
        "split_qkv_with_kv_wrap",
        "nlp_create_qkv_heads_zero_q",
        "nlp_create_qkv_heads_wrap",
        "nlp_create_qkv_heads_with_kv_zero_kv",
        "nlp_create_qkv_heads_with_kv_wrap",
        "nlp_create_qkv_heads_tied_zero_q",
        "nlp_create_qkv_heads_tied_wrap",
        "nlp_create_qkv_heads_tied_with_kv_zero_kv",
        "nlp_create_qkv_heads_tied_with_kv_wrap",
        "nlp_create_qkv_heads_boltz_zero_q",
        "nlp_create_qkv_heads_boltz_wrap",
        "nlp_create_qkv_heads_boltz_with_kv_zero_kv",
        "nlp_create_qkv_heads_boltz_with_kv_wrap",
        "create_qkv_heads_zero_q",
        "create_qkv_heads_zero_kv",
        "create_qkv_heads_wrap",
        "create_qkv_heads_from_separate_tensors_zero_q",
        "create_qkv_heads_from_separate_tensors_zero_kv",
        "create_qkv_heads_from_separate_tensors_wrap",
    ],
)
def test_head_count_rejected(device, expect_error, run, kwargs, message):
    with expect_error(RuntimeError, message):
        run(device, **kwargs)
