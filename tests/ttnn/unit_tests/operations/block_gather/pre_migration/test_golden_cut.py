# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of block_gather (golden spec: ttnn/ttnn/bringup/block_select_ttnn/tests/unit/test_block_select.py).

The golden suite has no TARGET / SUPPORTED table: its axes are the (W, K, block) triples it runs on 96 rows of the
whole system mesh, tensors replicated, compared byte-exact against a torch gather. Every block value it uses (1, 2, 8,
16, 32) appears here, with its smallest (512, 16, 2) and largest (65536, 512, 8) inputs and the golden data patterns
(ties, a -inf tail, none / just-out-of-range ids, a repeated block). Each case is checked on every chip of the mesh,
then again through the caller's output buffer (a prepared-program hit), as the golden suite does.
"""

import importlib
import os

import pytest
import torch
import ttnn

from block_gather_pm_ref import bits_equal, mismatch, ref_block_gather, to_dev, to_host

SENT = 0xFFFFFFFF
ROWS = 96  # the golden suite's row count


def _op():
    """The entry under test, as the acceptance suite selects it (source: the Python package; native: the registered
    op's name in TT_PRE_MIGRATION_ENTRY)."""
    mode = os.environ.get("TT_PRE_MIGRATION_MODE", "source") or "source"
    spec = os.environ.get("TT_PRE_MIGRATION_ENTRY", "") if mode == "native" else ""
    spec = spec or "ttnn.bringup.block_select_ttnn:block_gather"
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


GOLDEN = [
    pytest.param(512, 16, 2, id="w512_k16_b2_smallest"),
    pytest.param(1024, 64, 16, id="w1024_k64_b16"),
    pytest.param(2080, 64, 1, id="w2080_k64_b1"),
    pytest.param(4096, 512, 8, id="w4096_k512_b8"),
    pytest.param(8192, 128, 32, id="w8192_k128_b32"),
    pytest.param(65536, 512, 8, id="w65536_k512_b8_largest"),
]


@pytest.mark.parametrize(("W", "K", "B"), GOLDEN)
def test_golden_cut(full_mesh, W, K, B):
    op = _op()
    g = torch.Generator().manual_seed(W * 7 + K + B)
    s = torch.randn(1, 1, ROWS, W, generator=g).to(torch.bfloat16)
    s[..., :, ::7] = 0.5  # ties
    s[..., 3, W // 2 :] = float("-inf")  # a causal tail
    nb = W // B
    ids = torch.randint(0, nb, (1, 1, ROWS, K), generator=g)
    ids[..., 0, -5:] = SENT  # none
    ids[..., 1, 3] = nb  # just out of range
    ids[..., 2, :] = 0  # one block repeated
    st = to_dev(s, full_mesh, ttnn.bfloat16, mesh=True)
    it = to_dev(ids, full_mesh, ttnn.uint32, mesh=True)
    want = ref_block_gather(s, ids, B)

    got = op(st, it, block=B)
    assert list(got.shape) == [1, 1, ROWS, K * B]
    assert got.dtype == ttnn.bfloat16 and got.layout == ttnn.ROW_MAJOR_LAYOUT
    for chip, d in enumerate(to_host(got, mesh=True)):
        assert bits_equal(d, want), (chip, mismatch(d, want))

    entries = full_mesh.num_program_cache_entries()
    again = op(st, it, block=B, output_tensor=got)  # prepared hit, the caller's buffer
    assert full_mesh.num_program_cache_entries() == entries
    assert again is got
    for chip, d in enumerate(to_host(again, mesh=True)):
        assert bits_equal(d, want), (chip, mismatch(d, want))
    for t in (st, it, got):
        ttnn.deallocate(t)
