# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-built latent mask (tt/prefill_dyn.latent_mask) vs the host formula, for several chunk offsets."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_dyn import NEG, latent_mask


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.timeout(600)
def test_latent_mask(device):
    up0 = lambda t: ttnn.from_torch(t, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    pc = up0(torch.arange(64, dtype=torch.float32).reshape(1, 1, 1, 64))
    of = up0(torch.full((1, 1, 1, 1), 10.0))
    j = ttnn.subtract(pc, of)
    print("j", ttnn.to_torch(j).flatten()[:14].tolist(), flush=True)
    ge0 = ttnn.ge(j, 0.0)
    print("ge0", ttnn.to_torch(ge0).flatten()[8:14].tolist(), ge0.dtype, flush=True)
    lim = up0(torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 32, 1))
    jr = ttnn.repeat(j, [1, 1, 32, 1])
    lr = ttnn.repeat(lim, [1, 1, 1, 64])
    print(
        "jr",
        tuple(jr.shape),
        ttnn.to_torch(jr)[0, 0, :2, 8:14].tolist(),
        "lr",
        tuple(lr.shape),
        ttnn.to_torch(lr)[0, 0, :3, :2].tolist(),
        flush=True,
    )
    lt = ttnn.lt(jr, lr)
    print("lt", ttnn.to_torch(lt)[0, 0, :3, 8:14].tolist(), flush=True)

    C, L = 256, 2048
    for r, s0 in [(1, 0), (1, 256), (1, 1280), (2, 0), (2, 768), (2, 1792)]:
        pos = s0 + torch.arange(C)
        lim = ((pos + 1) // r).float().reshape(1, 1, C, 1)
        off = float(L - (s0 + C) // r)
        up = lambda t: ttnn.from_torch(t, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        got = ttnn.to_torch(
            latent_mask(
                up(torch.arange(L, dtype=torch.float32).reshape(1, 1, 1, L)),
                up(torch.full((1, 1, 1, 1), off)),
                up(lim),
                C,
                L,
            )
        ).float()
        j = torch.arange(L).view(1, -1) - int(off)
        want = torch.where((j >= 0) & (j < lim.reshape(C, 1)), 0.0, NEG).reshape(1, 1, C, L)
        bad = int(((got < -1e8) != (want < -1e8)).sum()) + int((got[want == 0].abs() > 0).sum())
        print(f"latent mask r={r} s0={s0}: mismatches {bad}", flush=True)
        assert bad == 0
