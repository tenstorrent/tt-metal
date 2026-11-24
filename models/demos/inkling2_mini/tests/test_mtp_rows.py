# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""DSparkDrafter.commit_rows (the drafter context of the R tokens of a speculative verify pass, one pass) against
R commit() calls: the context K / V rows of every layer and each row's f_rnn bit-equal, R = 2, 3, bf16 and fp32
taps. Needs the v1 weight cache + ``<cache>/mtp``.

    pytest -q models/demos/inkling2_mini/tests/test_mtp_rows.py
"""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.model import HIDDEN
from models.demos.inkling2_mini.tt.mtp import N_LAYERS
from models.demos.inkling2_mini.tt.spec_generator import TAP_LP, build


@pytest.fixture(scope="module")
def drafter():
    dev = ttnn.open_device(device_id=0)
    _, d = build(dev, 2048, log=lambda s: None)
    yield d
    ttnn.close_device(dev)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("R", [2, 3])
def test_commit_rows(drafter, R, dtype):
    d, dev = drafter, drafter.device
    ntap = len(TAP_LP) + 1
    torch.manual_seed(R)
    p0 = 700 + 31 * R
    taps = torch.randn(R, ntap, HIDDEN)  # [position, tap]
    put = lambda t, **kw: ttnn.from_torch(t, device=dev, **kw)
    caches = lambda: [ttnn.to_torch(c).float()[0, :, p0 - 1 : p0 + R + 1].clone()  # noqa: E731
                      for i in range(N_LAYERS) for c in (d.k_cache[i], d.v_cache[i])]  # fmt: skip

    ref_f = []
    for r in range(R):
        f = torch.zeros(1, 1, 32, HIDDEN)
        f[0, 0, :ntap] = taps[r]
        d.set_commit_pos(p0 + r)
        d.commit(
            put(f, dtype=dtype, layout=ttnn.TILE_LAYOUT),
            put(torch.tensor([p0 + r], dtype=torch.int32), dtype=ttnn.int32),
        )
        ref_f.append(ttnn.to_torch(d.f_rnn).float().reshape(-1).clone())
    ref = caches()
    assert all(c[:, 1 : R + 1].any() for c in ref)
    for i in range(N_LAYERS):  # other values in the rows, so the comparison sees the R-row write
        for c in (d.k_cache[i], d.v_cache[i]):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.full(tuple(c.shape), 0.5), dtype=c.dtype, layout=ttnn.TILE_LAYOUT), c
            )
    assert not any(torch.equal(a, b) for a, b in zip(caches(), ref))

    stack = taps.reshape(1, 1, ntap * R, HIDDEN)  # row ntap * r + j = tap j of position r
    ids = torch.arange(p0, p0 + R, dtype=torch.int32).view(1, R)
    d.commit_rows(
        put(stack, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        put(torch.tensor([p0], dtype=torch.int32), dtype=ttnn.int32),
        put(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
    )
    got = caches()
    for j, (a, b) in enumerate(zip(got, ref)):
        assert torch.equal(a[:, 1 : R + 1], b[:, 1 : R + 1]), ("layer", j // 2, "kv"[j % 2])
        assert (a[:, 0] == 0.5).all() and (a[:, R + 1] == 0.5).all(), "rows outside p0 .. p0+R-1 untouched"
    fr = ttnn.to_torch(d.f_rnn_rows).float()[0, 0]
    for r in range(R):
        assert torch.equal(fr[r], ref_f[r]), r
