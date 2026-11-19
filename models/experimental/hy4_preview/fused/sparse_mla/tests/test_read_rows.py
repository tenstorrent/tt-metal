# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The sparse row reader: rows selected by id, read from the packed KV storage in DRAM and delivered as BF16
tiles, must equal the prefill stack's host unpack of the same rows bit for bit, except what a Tensix register cannot
hold: latent values whose FP8 code is subnormal (below 2**-6 of the block's scale) are read as zero, and a zero
is +0 whatever its sign in storage.

Mock Galaxy (HY4_MOCK=1): the program is compiled and its CBs / kernel sizes checked, nothing is compared."""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.sparse_mla.layout import SparseMlaConfig, kv_row_location
from models.experimental.hy4_preview.fused.sparse_mla.op import SparseMla
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs

LAYERS = 2  # slots per user in the test storage; the reader addresses slot = user * LAYERS + layer

# One device open for the module: every test takes the plain `device` fixture.
pytestmark = pytest.mark.use_module_device


def _counts(name, cfg):
    if name == "mixed":
        return [1, 37, 512, 2048, 0, 64, 129, 300]
    return [int(name)] * cfg.users


@pytest.mark.parametrize("rows_per_slot", [4096])
@pytest.mark.parametrize(
    "counts, fidelity", [("mixed", "hifi4"), ("1", "hifi4"), ("37", "hifi4"), ("0", "hifi4"), ("mixed", "lofi")]
)
def test_read_rows(device, rows_per_slot, counts, fidelity):
    """``fidelity`` is the converter's (the scale multiply): a power-of-two scale times a 4-bit significand is
    exact at any fidelity, LoFi is one FPU pass instead of four."""
    fidelity = {"hifi4": ttnn.MathFidelity.HiFi4, "lofi": ttnn.MathFidelity.LoFi}[fidelity]
    cfg = SparseMlaConfig(rows_per_slot=rows_per_slot, num_layers=LAYERS)
    layer = 1
    slots = cfg.users * LAYERS
    latent, rope = hs.random_latents(slots, cfg.rows_per_slot, cfg, seed=3)
    kv, rows = hs.make_storage(device, cfg, slots, latent, rope)
    lists = [hs.spread_ids(cfg, n, seed=10 + u) for u, n in enumerate(_counts(counts, cfg))]
    sel = hs.make_selection(device, cfg, lists)
    out = hs.to_dev(torch.zeros(1, 1, cfg.users * cfg.topk, cfg.width), device, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    op = SparseMla(cfg, device)
    if hs.MOCK:
        ttnn.experimental.prepare_generic_op(
            [kv, sel, out], op.read_rows_program(kv, sel, out, layer, fidelity=fidelity)
        )
        return
    ttnn.generic_op([kv, sel, out], op.read_rows_program(kv, sel, out, layer, fidelity=fidelity))
    got = ttnn.to_torch(out)[0, 0].to(torch.bfloat16)

    subnormal = hs.fp8_subnormal(kv, device, cfg)
    pages, banks, total, signed_zeros, zeros, flushed, flushed_max = set(), set(), 0, 0, 0, 0, 0.0
    for u, ids in enumerate(lists):
        n = ids.numel()
        want = rows[cfg.slot(u, layer)][ids]
        have = got[u * cfg.topk : u * cfg.topk + n]
        sub = subnormal[cfg.slot(u, layer)][ids]
        bits = (have.view(torch.int16) != want.view(torch.int16)) & ~sub
        values = (have.float() != want.float()) & ~sub
        assert not values.any(), (
            f"user {u}: {n} rows, {values.sum().item()} values differ, "
            f"max abs {(have.float() - want.float()).abs().max().item():.3e}; first rows {values.any(-1).nonzero()[:4].flatten().tolist()}"
        )
        assert (have.float()[sub] == 0).all(), f"user {u}: an FP8 subnormal is neither decoded nor flushed to zero"
        flushed += int(sub.sum())
        flushed_max = max(flushed_max, want.float()[sub].abs().max().item() if sub.any() else 0.0)
        signed_zeros += int(bits.sum())  # equal values with different bits can only be +0 against -0
        zeros += int((want.float() == 0).sum())
        total += n
        for i in ids.tolist():
            bank, off = kv_row_location(cfg, cfg.slot(u, layer), i)
            pages.add((bank, off))
            banks.add(bank)
    logger.info(
        f"HY4_MK3 read_rows counts={counts} converter {fidelity} R={rows_per_slot}: {total} rows equal to the host unpack "
        f"({total * cfg.width} values, {zeros} zeros, {signed_zeros} with the other sign of zero; {flushed} "
        f"FP8-subnormal latent values read as 0, largest {flushed_max:.3e}), "
        f"{len(pages)} DRAM pages over {len(banks)} banks"
    )
    # A zero comes out as +0 whatever its sign in storage (a negative FP8 zero times the scale is -0 on the host).
    assert signed_zeros <= zeros
