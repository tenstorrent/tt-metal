# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of block_keys (golden: ttnn/ttnn/bringup/block_select_ttnn/tests/unit/test_block_select.py).

The golden suite has no TARGET / SUPPORTED tables; its axes are its block_keys cases (W, K, block) with 96 rows and
KP = min(512, K block), its data pattern (none and just-out-of-range block ids, a row of one repeated block, a row of
none positions) and a byte-exact comparison against its ``ref_keys``. Every golden (W, K, block) appears here, which
covers its smallest (512, 16, 2) and largest (65536, 512, 8) inputs, plus block 4 (the README's "a power of two <= 32",
the one block value the golden cases skip), each on a first call (a miss) and again into the caller's output buffer.
The golden's replicated whole-mesh topology is test_golden_cut_mesh.py.
"""

import pytest
import torch

pytestmark = pytest.mark.use_module_device

ROWS = 96
GOLDEN = [(4096, 512, 8), (2080, 64, 1), (8192, 128, 32), (1024, 64, 16), (65536, 512, 8), (512, 16, 2)]
EXTRA = [(4096, 256, 4)]


def golden_inputs(W, K, B, seed, sent):
    """The golden suite's block_keys data pattern (its ids are block_gather's ids)."""
    g = torch.Generator().manual_seed(seed)
    nb = W // B
    ids = torch.randint(0, nb, (1, 1, ROWS, K), generator=g)
    ids[..., 0, -5:] = sent
    ids[..., 1, 3] = nb
    ids[..., 2, :] = 0
    pos = torch.randint(0, K * B, (1, 1, ROWS, min(512, K * B)), generator=g)
    pos[..., 4, :7] = sent
    return pos, ids


@pytest.mark.parametrize(("W", "K", "B"), [pytest.param(*c, id=f"w{c[0]}_k{c[1]}_b{c[2]}") for c in GOLDEN + EXTRA])
def test_golden_cut(device, block_keys_op, bk, W, K, B):
    pos, ids = golden_inputs(W, K, B, seed=W + K + B, sent=bk.SENT)
    pt, it = bk.up(pos, device), bk.up(ids, device)
    want = bk.ref_keys(pos, ids, B, W)
    keys = block_keys_op(pt, it, block=B, width=W)
    bk.assert_dram_rm_u32(keys, list(pos.shape))
    bk.assert_exact(bk.down(keys), want, f"golden W={W} K={K} B={B}")
    again = block_keys_op(pt, it, block=B, width=W, output_tensor=keys)  # a hit into the caller's buffer
    assert bk.addr(again) == bk.addr(keys)
    bk.assert_exact(bk.down(again), want, f"golden W={W} K={K} B={B} again")
