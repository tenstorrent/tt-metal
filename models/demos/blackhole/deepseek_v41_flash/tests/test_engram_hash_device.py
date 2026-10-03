# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exactness (>=2000 random histories incl. sequence-start/pad cases) and device time of the Engram hash kernel.
Env DSV41_HASH_MODE=0 (native uint64) | 1 (32-bit limbs)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.engram_hash import DSV41EngramHash

T = 4


def replay_ms(md, fn, n=30):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    dt = (time.perf_counter() - t) / n * 1e3
    ttnn.release_trace(md, tid)
    return dt


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_engram_hash_device(mesh_device):
    from transformers import AutoTokenizer

    md = mesh_device
    mode = int(os.environ.get("DSV41_HASH_MODE", "0"))
    mod = R.load_model_module()
    ndev = md.get_num_devices()
    per = ndev * T
    NH = 2400
    args = R.model_args(NH, 16)
    layout = mod.EngramLayout.from_args(args)
    tok = AutoTokenizer.from_pretrained(R.CKPT_DIR)
    state = mod.NgramHashState(args, layout, tok)
    W = layout.max_ngram_size
    print(
        f"HASH W={W} NL={len(layout.layer_ids)} H={layout.n_heads} pad={int(state.pad_id)} V={len(state.token_map)} mode={mode}",
        flush=True,
    )
    dev = DSV41EngramHash(
        md,
        state.token_map.numpy(),
        state.multipliers.numpy(),
        state.primes.numpy(),
        state.offsets.numpy(),
        int(state.pad_id),
        T=T,
        mode=mode,
    )
    V = len(state.token_map)
    g = torch.Generator().manual_seed(1234)
    # histories: lengths 1..W+3 (start/pad cases), mixed with raw token 2 (== pad) and repeated tokens
    lens = [1, 2, 3, 4, 5, 7]
    hist_all, ref_all = [], []
    for L in lens:
        n = NH // len(lens)
        ids = torch.randint(0, V, (n, L), generator=g)
        ids[torch.rand(n, L, generator=g) < 0.1] = 2
        rep = torch.rand(n, L, generator=g) < 0.1
        ids[:, 1:] = torch.where(rep[:, 1:], ids[:, :-1], ids[:, 1:])
        state.cache.zero_()
        ref = state(ids, 0)[:, -1]  # [n, NL, NCOL] int64
        h = torch.full((n, 16), -1, dtype=torch.int32)
        for s in range(min(L, 16)):
            h[:, s] = ids[:, L - 1 - s].to(torch.int32)
        hist_all.append(h)
        ref_all.append(ref)
    hist_all, ref_all = torch.cat(hist_all), torch.cat(ref_all)
    N = hist_all.shape[0]
    pad = (-N) % per
    hist_p = torch.cat([hist_all, torch.full((pad, 16), -1, dtype=torch.int32)])
    got = []
    for i in range(0, hist_p.shape[0], per):
        out = dev(dev.upload_hist(hist_p[i : i + per].contiguous()))
        o = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        got.append(dev.ids(o))
    got = torch.cat(got)[:N].to(torch.int64)
    mism = int((got != ref_all).sum())
    print(
        f"HASH checked {N} histories x {ref_all.shape[1] * ref_all.shape[2]} ids: max mismatches elements = {mism}, "
        f"rows with mismatch = {int((got != ref_all).flatten(1).any(1).sum())}",
        flush=True,
    )
    assert mism == 0

    # device time: chain n calls in a trace
    hist = dev.upload_hist(hist_p[:per].contiguous())

    def many(n):
        def f():
            for _ in range(n):
                dev(hist)

        return f

    t1, t41 = replay_ms(md, many(1)), replay_ms(md, many(41))
    print(
        f"HASH_TIME mode={mode}: single-op trace {t1:.3f} ms | per op in long trace {(t41 - t1) / 40 * 1e3:.1f} us",
        flush=True,
    )
