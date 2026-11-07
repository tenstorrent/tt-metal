# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Persistent L1 cost of the load-based decode buckets (the server's prefill programs need the lowest persistent L1 buffer above their static circular buffer region, ~685968 B: 'Statically allocated
circular buffers ... clash with L1 buffers'): total allocated L1 bytes per bank after the full B=16 runner set, after the drafter view of every bucket, and after each bucket runner (compile pass
included). Env: TT_OPT_B=16, TT_OPT_K, BUCKET_UBS (default "2,1"). Prints 'L1ALLOC <stage> <bytes per bank> largest_free <bytes>'."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.tt_opt import test_tt_opt_decode as T
from models.demos.blackhole.deepseek_v41_flash.tt.spec_model import SpecRunner

_PREV = {}


def _l1(md, tag):
    mv = ttnn.get_memory_view(md, ttnn.BufferType.L1)
    T._log(
        f"L1ALLOC {tag}: allocated/bank {mv.total_bytes_allocated_per_bank} largest_free {mv.largest_contiguous_bytes_free_per_bank}"
    )
    for c in ((1, 2), (0, 0), (5, 5), (11, 9)):
        try:
            r = ttnn.experimental_get_l1_occupied_ranges(md, ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*c))
            now = sorted((int(a), int(b)) for a, b in r)
            old = _PREV.get(c, [])
            new = [x for x in now if x not in old]
            gone = [x for x in old if x not in now]
            T._log(
                f"L1MAP {tag} core {c}: persistent ranges >= 500000 {[x for x in now if x[1] > 500000][:14]}; new {[(a, b, b - a) for a, b in new][:10]}; gone {[(a, b, b - a) for a, b in gone][:6]}"
            )
            _PREV[c] = now
        except Exception as e:  # noqa: BLE001
            T._log(f"L1MAP unavailable core {c}: {type(e).__name__}: {str(e)[:100]}")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", T.PARAMS, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_bucket_l1(mesh_device):
    md = mesh_device
    _l1(md, "device open")
    gen, runner, X, base, _ = T._setup(md, minimal=True, tracy_layers=False)
    _l1(md, "model built + runner k=%d" % runner.k)
    runner.prepare(torch.zeros(runner.B, runner.n, dtype=torch.long), torch.full((runner.B,), 130, dtype=torch.long))
    _l1(md, "full runner prepared")

    def probe(obj, name, tag, md=md):
        f = getattr(obj, name, None)
        if f is None:
            return

        def w(*a, **k):
            b = ttnn.get_memory_view(md, ttnn.BufferType.L1).total_bytes_allocated_per_bank
            r = f(*a, **k)
            ttnn.synchronize_device(md)
            e = ttnn.get_memory_view(md, ttnn.BufferType.L1).total_bytes_allocated_per_bank
            if e != b:
                T._log(f"L1PROBE {tag}.{name}: {b} -> {e} ({e - b:+d})")
            return r

        setattr(obj, name, w)

    for Ub in [int(x) for x in os.environ.get("BUCKET_UBS", "2,1").split(",")]:
        dv = runner.drafter_root.view_users(Ub)
        _l1(md, f"drafter view Ub={Ub}")
        br = SpecRunner(gen.m, runner.k, drafter=dv, Ub=Ub)
        _l1(md, f"bucket runner Ub={Ub} built")
        if os.environ.get("PROBE", "1") == "1":
            for L, lv, _k in br.dec.layers:
                probe(lv, "forward", f"U{Ub}.layer{L}")
                probe(lv.moe, "forward", f"U{Ub}.layer{L}.moe")
                probe(lv.attention, "forward", f"U{Ub}.layer{L}.attn")
            probe(br.dec.head, "forward", f"U{Ub}.head")
            probe(br.dec.head, "sample_global", f"U{Ub}.head")
            probe(br.dec, "_engram_rows", f"U{Ub}.dec")
            probe(br.dec, "_engram_fwd", f"U{Ub}.dec")
            probe(br.dec.drafter, "draft", f"U{Ub}.drafter")
            probe(br.dec.drafter, "forward", f"U{Ub}.drafter")
        import collections
        import traceback

        refs = []
        agg = collections.defaultdict(lambda: [0, 0])
        orig = {
            n: getattr(ttnn, n) for n in ("from_torch", "allocate_tensor_on_device", "empty", "zeros", "full", "ones")
        }

        def mk(n):
            def w(*a, **k):
                b = ttnn.get_memory_view(md, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
                r = orig[n](*a, **k)
                d = ttnn.get_memory_view(md, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank - b
                if d:
                    fr = [f for f in traceback.extract_stack()[:-1] if "deepseek_v41_flash/tt/" in f.filename][-2:]
                    key = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in reversed(fr))
                    try:
                        refs.append((int(r.buffer_address()), d, n, key))
                    except Exception:  # noqa: BLE001
                        pass
                    agg[(n, key)][0] += d
                    agg[(n, key)][1] += 1
                return r

            return w

        if os.environ.get("CENSUS", "1") == "1":
            for n in orig:
                setattr(ttnn, n, mk(n))
        prof = {"_l1_trace": [], "_dram": True}
        br.dec.profile = prof
        br.prepare(torch.zeros(br.B, br.n, dtype=torch.long), torch.full((br.B,), 130, dtype=torch.long))
        br.dec.profile = None
        for n in orig:
            setattr(ttnn, n, orig[n])
        import gc

        gc.collect()
        ttnn.synchronize_device(md)
        probe_t = orig["allocate_tensor_on_device"](
            ttnn.Shape([1, 1, 32, 32 * 8 * 777]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        probe_addr = int(probe_t.buffer_address())
        ttnn.synchronize_device(md)
        try:
            d0_ = md.get_devices()[0] if hasattr(md, "get_devices") else md
            ttnn.dump_device_memory_state(d0_, "/tmp/dms_")
            rows = open("/tmp/dms_detailed_memory_usage.csv").read().splitlines()
            h = next(i for i, l in enumerate(rows) if l.split()[:3] == ["Block", "Address", "Size"])
            blocks = {}
            for l in rows[h + 1 :]:
                c = l.split()
                if len(c) != 6:
                    break
                if c[5] == "yes":
                    blocks[int(c[1])] = int(c[2])
            T._log(
                f"DRAMLIVE refs {len(refs)} sample {[(a, d) for a, d, _, _ in refs[-6:]]}; blocks sample {sorted(blocks.items())[-6:]}; nblocks {len(blocks)}"
            )
            psz = 32 * 32 * 8 * 777 * 2 // 8
            cand = [a for a, sz in blocks.items() if sz == psz]
            off = probe_addr - cand[0] if len(cand) == 1 else None
            T._log(f"DRAMLIVE probe addr {probe_addr} size {psz} candidates {cand} offset {off}")
            if off is not None:
                blocks = {a + off: sz for a, sz in blocks.items()}
            latest = {}
            for a, d, n, key in refs:
                latest[a] = (d, n, key)
            by = collections.defaultdict(lambda: [0, 0])
            for a, sz in blocks.items():
                if a in latest:
                    by[latest[a][2]][0] += sz
                    by[latest[a][2]][1] += 1
            tot = sum(v[0] for v in by.values())
            T._log(
                f"DRAMLIVE Ub={Ub} live blocks in bank 0: {len(blocks)}; attributed to prepare-window creations {tot / 2**20:.3f} MiB/bank"
            )
            for key, (sz, c) in sorted(by.items(), key=lambda kv: -kv[1][0])[:22]:
                T._log(f"DRAMLIVE Ub={Ub} {sz / 2**20:.3f} MiB/bank x{c} {key}")
        except Exception as e:  # noqa: BLE001
            T._log(f"DRAMLIVE dump failed {type(e).__name__}: {str(e)[:200]}")
        agg = {}
        dr = lambda: ttnn.get_memory_view(md, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank / 2**20
        d0 = dr()
        T._log(f"DRAMDROP Ub={Ub} after prepare {d0:.2f}")
        for L, lv, _k in br.dec.layers:
            for obj, names in (
                (lv.attention, ("last_lat", "_last_qr")),
                (lv.moe, ("last_routing",)),
                (lv, ("last_h",)),
            ):
                for nme in names:
                    if getattr(obj, nme, None) is not None:
                        try:
                            setattr(obj, nme, None)
                        except Exception:  # noqa: BLE001
                            pass
            d1 = dr()
        T._log(f"DRAMDROP Ub={Ub} after dropping last_lat/_last_qr/last_routing {dr():.2f} (delta {dr() - d0:+.3f})")
        for (n, key), (d, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]:
            T._log(f"DRAMCENSUS Ub={Ub} {d / 2**20:+.3f} MiB/bank x{c} {n} {key}")
        last = None
        lastd = None
        for name, a, lf, dr in prof["_l1_trace"]:
            if last is not None and a != last:
                T._log(f"L1SEC Ub={Ub} {name}: {last} -> {a} ({a - last:+d})")
            if lastd is not None and dr != lastd:
                T._log(f"DRAMSEC Ub={Ub} {name}: {(dr - lastd) / 2**20:+.3f} MiB/bank")
            last, lastd = a, dr
        _l1(md, f"bucket runner Ub={Ub} prepared")
