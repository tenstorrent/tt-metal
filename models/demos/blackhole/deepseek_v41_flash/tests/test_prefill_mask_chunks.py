# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Column-chunked prefill masks (tt/prefill_dyn.py: latent_mask / latent_mask_parts / visibility_mask, LATENT_MASK_COLS = 32768) against the host formula, for L above the chunk width
(70k = 2 full chunks + a remainder, 98304 = 3 chunks), several chunk offsets and ratios; the chunked masks must be IDENTICAL to the host values (0 / NEG, 1 / 0).
Prints MC lines with the device time of one build (eager, synced) so the per-chunk cost of the chunking is visible."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_dyn import NEG, latent_mask, visibility_mask


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="plain")], indirect=True)
@torch.no_grad()
def test_chunked_masks(mesh_device):
    md = mesh_device
    up = lambda t: ttnn.from_torch(
        t,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    C = 256
    for L in (4096, 69632, 98304):
        for r, s0 in ((1, 0), (1, L // 2), (2, L), (2, 2 * L - 4096)):
            pos = s0 + torch.arange(C)
            lim = ((pos + 1) // r).float().reshape(1, 1, C, 1)
            off = float(L - (s0 + C) // r)
            pcol = up(torch.arange(L, dtype=torch.float32).reshape(1, 1, 1, L))
            offt, limt = up(torch.full((1, 1, 1, 1), off)), up(lim)
            j = torch.arange(L).view(1, -1) - int(off)
            inside = (j >= 0) & (j < lim.reshape(C, 1))
            t0 = time.perf_counter()
            lm = latent_mask(pcol, offt, limt, C, L)
            ttnn.synchronize_device(md)
            t_lm = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            vm = visibility_mask(pcol, offt, limt, C, L)
            ttnn.synchronize_device(md)
            t_vm = (time.perf_counter() - t0) * 1e3
            got_l = ttnn.to_torch(ttnn.get_device_tensors(lm)[0]).float().reshape(C, L)
            got_v = ttnn.to_torch(ttnn.get_device_tensors(vm)[0]).float().reshape(C, L)
            want_l = torch.where(inside, 0.0, NEG)
            bad_l = int((got_l != want_l.to(torch.bfloat16).float()).sum())
            bad_v = int((got_v != inside.float()).sum())
            print(
                f"MC L={L} r={r} s0={s0}: latent_mask mismatches {bad_l} ({t_lm:.1f} ms), visibility_mask mismatches {bad_v} ({t_vm:.1f} ms)",
                flush=True,
            )
            assert bad_l == 0 and bad_v == 0
            for t in (lm, vm, pcol, offt, limt):
                ttnn.deallocate(t)
