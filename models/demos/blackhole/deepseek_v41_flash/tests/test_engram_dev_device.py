# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test of DSV41_DEV_ENGRAM (tt/engram_dev.py): hash + gather + combine in a trace vs the shipped host path, bit exact (-0 == +0), many rounds with partial
accepts, T of B=4/8/16 (k=3/5), plus the added device time per stage.  Synthetic hash state and SMALL synthetic tables (spread over all 32 chips, random fp8 / e8m0 bytes), so no
checkpoint is read.  Env: DSV41_HASH_MODE (0 native uint64 | 1 limbs), DSV41_DEVENG_ROUNDS (20), DSV41_DEVENG_GEOMS ("1x4,1x6,2x6,4x4,4x6"  = users-per-mesh-row x n)."""

import os
import time
import types

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_engram_dev_cpu import DEAD, H, NL, W, make_state
from models.demos.blackhole.deepseek_v41_flash.tt import engram_dev as ed
from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash

NROWS = 100000  # synthetic rows per layer (hash modulus < 96k, offsets 0)


class SynthTable:
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        w = torch.randint(0, 256, (n, 256), generator=g, dtype=torch.int32)
        w[(w & 0x7F) == 0x7F] = 0x30  # no NaN encodings
        self.w = w.to(torch.uint8).view(torch.float8_e4m3fn)
        self.s = torch.randint(118, 136, (n, 8), generator=g, dtype=torch.int32).to(torch.uint8).view(torch.float8_e8m0fnu)
        self.in_ram = True

    def rows(self, idx):
        flat = idx.reshape(-1)
        v = self.w[flat].float().unflatten(-1, (-1, 32)) * self.s[flat].float().unsqueeze(-1)
        return v.flatten(-2).to(torch.bfloat16).reshape(*idx.shape, -1)


def norm0(x):
    v = x.view(torch.int16).clone()
    v[v == -32768] = 0
    return v


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


def chain_us(md, op):
    def many(n):
        def f():
            for _ in range(n):
                op()

        return f

    return (replay_ms(md, many(21)) - replay_ms(md, many(1))) / 20 * 1e3


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_engram_dev_device(mesh_device):
    md = mesh_device
    mrows, mcols = tuple(md.shape)
    mode = int(os.environ.get("DSV41_HASH_MODE", "0"))
    rounds = int(os.environ.get("DSV41_DEVENG_ROUNDS", "20"))
    geoms = [tuple(int(x) for x in g.split("x")) for g in os.environ.get("DSV41_DEVENG_GEOMS", "1x4,1x6,2x6,4x4,4x6").split(",")]
    Umax = max(u for u, _ in geoms) * mrows
    ref, mine = make_state(Umax), make_state(Umax)
    for st in (ref, mine):  # ids must stay inside the synthetic tables
        st.offsets = (torch.arange(NL * (W - 1) * H).reshape(NL, (W - 1) * H) * 2400) % 1  # offsets 0: ids = hash % prime (< 1100)
        st.primes = st.primes + (torch.arange(NL * (W - 1) * H).reshape(NL, W - 1, H) * 2000)  # spread over up to ~48k rows
    tabs_h = [SynthTable(NROWS, 11 + l) for l in range(NL)]
    host_rows = types.SimpleNamespace(
        engram=types.SimpleNamespace(hash=mine), tables={1: tabs_h[0], 14: tabs_h[1]}
    )
    tabs = ed.DevEngramTables(md, host_rows, [1, 14], log=lambda s: print(s, flush=True))
    print("DRAM after synthetic tables:", ttnn.get_memory_view(md, ttnn.BufferType.DRAM), flush=True)
    rng = np.random.default_rng(5)
    ragged = RaggedNgramHash(ref)
    st_np = dict(cache=mine.cache.numpy(), tmap=tabs.tmap, dead=DEAD)
    B = Umax
    lens = rng.integers(1, 40, size=B)
    base_all = torch.tensor(lens, dtype=torch.long)
    for b in range(B):
        toks = torch.tensor(rng.integers(0, 2000, size=(1, int(lens[b]))))
        for st in (ref, mine):
            st.cache[b, : int(lens[b])] = st.token_map[toks]
    mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(mrows, mcols))
    report = []
    for U, n in geoms:
        T = U * n
        dv = ed.DevEngramRows(tabs, T, n, mode=mode)
        Bg = U * mrows
        phys = torch.tensor([(i // U) * (Umax // mrows) + i % U for i in range(Bg)], dtype=torch.long)  # users of the geometry inside the full cache
        ctrl_h = torch.zeros(Bg * n, dv.CW, dtype=torch.int32)
        ctrl_dev = None
        bad = 0
        for rnd in range(rounds):
            X = torch.tensor(rng.integers(0, 2000, size=(Bg, n)))
            base = base_all[phys].clone()
            want_ids = ragged(X, base, rows=phys)  # [Bg, n, NL, 24]
            ed.fill_ctrl(ctrl_h, X, base, st_np, rows=phys, w=W)
            ctrl_h[:, 0] = X.reshape(-1).to(torch.int32)
            ctrl_h[:, 1] = (base.reshape(-1, 1) + torch.arange(n).reshape(1, n)).reshape(-1).to(torch.int32)
            host_c = ttnn.from_torch(ctrl_h, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp)
            if ctrl_dev is None:
                ctrl_dev = ttnn.to_device(host_c, md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                ttnn.copy_host_to_device_tensor(host_c, ctrl_dev)
            out = dv.rows(ctrl_dev)
            ttnn.synchronize_device(md)
            ids_dev = ttnn.to_torch(ttnn.get_device_tensors(dv.ids)[0]).reshape(Bg * n, 64)[:, : NL * 24]
            ids_ok = (ids_dev.reshape(Bg * n, NL, 24) == want_ids.reshape(Bg * n, NL, 24).to(torch.int32)).all()
            exp = torch.cat([tabs_h[l].rows(want_ids[:, :, l, :]).flatten(-2).reshape(Bg * n, -1) for l in range(NL)], -1)
            R = ttnn.get_device_tensors(out)
            for r in range(mrows):
                want = norm0(exp[r * T : (r + 1) * T])
                for c in range(mcols):
                    got = norm0(ttnn.to_torch(R[r * mcols + c]).reshape(T, -1))
                    bad += int((got != want).sum())
            assert bool(ids_ok), f"geometry U={U} n={n} round {rnd}: ids differ"
            m = torch.tensor(rng.integers(0, n, size=Bg))
            base_all[phys] = base + 1 + m
        print(f"EXACT U={U} n={n} (T={T}, TTOT={dv.TTOT}) rounds={rounds}: mismatching bf16 elements = {bad}", flush=True)
        assert bad == 0
        # device time per stage (ops back to back in one trace)
        ids = dv.hash(ctrl_dev)
        part = dv.gather(ids)
        t_h = chain_us(md, lambda: dv.hash(ctrl_dev))
        t_g = chain_us(md, lambda: dv.gather(ids))
        def comb():
            p_ = ttnn.to_layout(part, ttnn.TILE_LAYOUT) if (T * 48) % 32 == 0 else part
            r_ = ttnn.all_reduce(
                ttnn.reduce_scatter(p_, dim=2, cluster_axis=0, num_links=2, topology=ttnn.Topology.Linear),
                cluster_axis=1,
                num_links=2,
                topology=ttnn.Topology.Linear,
            )
            return ttnn.to_layout(r_, ttnn.ROW_MAJOR_LAYOUT) if (T * 48) % 32 == 0 else r_

        t_c = chain_us(md, comb)
        t_all = chain_us(md, lambda: dv.rows(ctrl_dev))
        line = f"DEVENG_TIME U={U} n={n} T={T} mode={mode}: hash {t_h:.1f} us, gather {t_g:.1f} us, combine {t_c:.1f} us, whole rows() {t_all:.1f} us"
        print(line, flush=True)
        report.append(line)
    print("DRAM at end:", ttnn.get_memory_view(md, ttnn.BufferType.DRAM), flush=True)
    print("\n".join(report), flush=True)
