# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-row append: one new packed row per user at a given position, written into the packed KV storage by the
chip that owns the position, read back through the prefill stack's host reader.

Mock Galaxy (HY4_MOCK=1): the program is compiled, nothing is compared."""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.hy4.kv_cache import KvStripe
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, reconstruct_scaled_fp8_kv_cache
from models.experimental.hy4_preview.fused.sparse_mla.layout import SparseMlaConfig
from models.experimental.hy4_preview.fused.sparse_mla.op import SparseMla
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs

# One device open for the module: every test takes the plain `device` fixture.
pytestmark = pytest.mark.use_module_device


def test_append(device):
    cfg = SparseMlaConfig(rows_per_slot=4096, num_layers=2)  # a 131072-token cache: 4096 rows per chip
    stripe = KvStripe(sp=8, tp=4)  # the prefill stack's placement: 128-token blocks over 32 chips
    layer, chip = 1, 5
    slots = cfg.users * cfg.num_layers
    latent, rope = hs.random_latents(slots, cfg.rows_per_slot, cfg, seed=5)
    kv, before = hs.make_storage(device, cfg, slots, latent, rope)

    # users 0, 2, 3, 6, 7 are at positions this chip owns (first, middle, last row of a block, several periods);
    # users 1 and 4 on other chips; user 5 beyond the cache.
    own = [chip * 128, 3 * 4096 + chip * 128 + 77, 31 * 4096 + chip * 128 + 127, 17 * 4096 + chip * 128 + 1, 640 + 5]
    positions = [own[0], 6 * 128 + 3, own[1], own[2], 4096 + 9, 32 * 4096 + chip * 128, own[3], own[4]]
    gen = torch.Generator().manual_seed(11)
    new_latent = torch.randn(cfg.users, 1, 1, cfg.latent, generator=gen) * 3.0
    new_rope = torch.randn(cfg.users, 1, 1, cfg.rope, generator=gen)
    rows = hs.pack_rows(device, new_latent, new_rope)  # the prefill stack's pack of the new tokens
    pos = hs.to_dev(torch.tensor(positions, dtype=torch.int32).reshape(1, 1, 1, -1), device, ttnn.uint32)

    op = SparseMla(cfg, device)
    if hs.MOCK:
        ttnn.experimental.prepare_generic_op([rows, pos, kv], op.append_program(kv, rows, pos, layer, chip))
        return
    op.append(kv, rows, pos, layer, chip)

    after = hs.read_storage(kv, device, cfg)
    packed = ttnn.to_torch(rows, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    want = reconstruct_scaled_fp8_kv_cache(packed, MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope))
    expected = before.clone()
    written = 0
    for u, p in enumerate(positions):
        if int(stripe.chip(p)) == chip and int(stripe.row(p)) < cfg.rows_per_slot:
            expected[cfg.slot(u, layer), int(stripe.row(p))] = want[u, 0, 0]
            written += 1
    assert written == 5
    same = after.view(torch.int16) == expected.view(torch.int16)
    assert same.all(), f"{(~same).any(-1).sum().item()} storage rows differ from the expected storage"
    changed = (after.view(torch.int16) != before.view(torch.int16)).any(-1).sum().item()
    logger.info(
        f"HY4_MK3 append: {written} rows written by chip {chip} (3 skipped), read back bit-exact through the prefill "
        f"host reader; {changed} of {after.shape[0] * after.shape[1]} storage rows changed"
    )
    assert changed == written
