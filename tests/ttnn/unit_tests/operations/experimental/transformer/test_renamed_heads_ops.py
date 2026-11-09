# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The heads operations lost their nlp_ prefix; the old names stay as deprecated aliases."""

import pytest
import torch

import ttnn
from ttnn.experimental_renames import RENAMED_EXPERIMENTAL_OPERATIONS


@pytest.mark.parametrize("old_name, new_name", sorted(RENAMED_EXPERIMENTAL_OPERATIONS.items()))
def test_old_name_warns_and_forwards(monkeypatch, old_name, new_name):
    calls = []
    monkeypatch.setattr(ttnn.experimental, new_name, lambda *args, **kwargs: calls.append((args, kwargs)) or "result")
    with pytest.warns(DeprecationWarning, match=f"renamed to ttnn.experimental.{new_name}"):
        result = getattr(ttnn.experimental, old_name)(1, 2, key="value")
    assert result == "result"
    assert calls == [((1, 2), {"key": "value"})]


def test_new_names_are_operations():
    for new_name in RENAMED_EXPERIMENTAL_OPERATIONS.values():
        assert callable(getattr(ttnn.experimental, new_name)), new_name


def _to_device(device, shape):
    return ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)


def test_old_names_match_new_names_on_device(device):
    heads = _to_device(device, (1, 8, 32, 64))
    with pytest.warns(DeprecationWarning):
        old = ttnn.experimental.nlp_concat_heads(heads)
    new = ttnn.experimental.concat_heads(heads)
    assert torch.equal(ttnn.to_torch(old), ttnn.to_torch(new))

    fused_qkv = _to_device(device, (1, 1, 32, 3 * 8 * 64))
    with pytest.warns(DeprecationWarning):
        old_q, old_k, old_v = ttnn.experimental.nlp_create_qkv_heads(fused_qkv, num_heads=8)
    new_q, new_k, new_v = ttnn.experimental.create_qkv_heads(fused_qkv, num_heads=8)
    for old, new in ((old_q, new_q), (old_k, new_k), (old_v, new_v)):
        assert torch.equal(ttnn.to_torch(old), ttnn.to_torch(new))
