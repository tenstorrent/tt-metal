# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Load time of one layer's routed experts (tt/experts.py:TtExperts, FIX.4: the laid-out weight cache and the miss
path). Not a gate: numbers for BREADCRUMBS.

    PYTHONPATH=$PWD nice -n 19 scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tt/perf_experts_load.py

test_host_weights_identical (CPU): _host_weights (each slot's 32-chip stack dequantized and checked at once) gives the
same bf16 tensors, chip / slot order and bfp4_stats as the per-expert loop it replaced (_host_weights_per_expert).

test_load_cache: one layer built with no cache (off, under cProfile: EXPERTS_LOAD_PROFILE), then with the run's cache
from scratch (miss: packs and dumps), then again (hit: loads the dumps); the hit's laid-out device tensors equal the
miss's (EXPERTS_LOAD_TIMES).
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt import experts as E
from models.demos.deepseek_v41_flash_d_p.tt.perf_experts import _weighted_params

LAYER = 0


def _host_weights_per_expert(mod, ck, gtab):
    """The miss path before FIX.4 (tt/experts.py:_load_block at 89e622744f9): one expert at a time, a bf16 check per
    matrix, the checks / stats on the bf16 stacks. Returns (flat_w, bfp4 stats or None)."""
    rows, cols, layer, H, I = mod.dgs, mod.ngroups, mod.layer, mod.H, mod.I
    flat_w = [[None] * mod.epc for _ in range(rows * cols)]
    changed = total = 0
    worst = (0.0, None)
    for le in range(mod.epc):
        w13 = torch.empty(cols, rows, 2 * I, H, dtype=torch.bfloat16)
        w2 = torch.empty(cols, rows, H, I, dtype=torch.bfloat16)
        for c in range(cols):
            for r in range(rows):
                p = ck.expert(layer, int(gtab[c, r, le]))
                w = {nm: E.fp4_dequant(*p[nm]) for nm in E.NAMES}
                w13[c, r, :I] = w["w1"]
                w13[c, r, I:] = w["w3"]
                w2[c, r] = w["w2"]
                for t in w.values():
                    assert bool(torch.equal(t.to(torch.bfloat16).float(), t))
                flat_w[r * cols + c][le] = (w13[c, r, :I], w13[c, r, I:], w2[c, r])
        if mod.impl == "bfp8":
            assert E._bfp8_exact(w13) and E._bfp8_exact(w2)
        else:
            for mats in ((w13[:, :, :I], "w1"), (w13[:, :, I:], "w3"), (w2, "w2")):
                n_ch, rel = E._bfp4_change(mats[0])
                changed, total = changed + n_ch, total + mats[0].numel()
                v, i = rel.flatten().max(dim=0)
                if float(v) > worst[0]:
                    c, r = divmod(int(i), rows)
                    worst = (float(v), (int(gtab[c, r, le]), mats[1]))
    stats = None if mod.impl == "bfp8" else {"changed": changed, "total": total, "worst": worst}
    return flat_w, stats


@pytest.mark.timeout(3600)
def test_host_weights_identical():
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    rows, cols = tuple(S.mesh)
    mod = E.TtExperts.__new__(E.TtExperts)  # (the host part only: no device)
    mod.dgs, mod.ngroups, mod.layer, mod.H, mod.I = rows, cols, LAYER, cfg.dim, cfg.inter
    mod.epc, mod.impl, mod.bfp4_stats = cfg.n_experts // (rows * cols), E.settings.get("EXPERTS_IMPL"), None
    gtab = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=mod.epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    new = mod._host_weights(ck, gtab)
    pr.disable()
    t_new = time.time() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
    print(f"EXPERTS_HOST_PROFILE\n{s.getvalue()}")
    t0 = time.time()
    old, st = _host_weights_per_expert(mod, ck, gtab)
    t_old = time.time() - t0
    for d in range(rows * cols):
        for le in range(mod.epc):
            for a, b in zip(new[d][le], old[d][le]):
                assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape and torch.equal(a, b), (d, le)
    if st is not None:
        ns = mod.bfp4_stats
        assert (ns["changed"], ns["total"]) == (st["changed"], st["total"]), (ns, st)
        assert tuple(ns["worst_matrix"]) == st["worst"][1], (ns, st)
        assert abs(ns["worst_rel"] - st["worst"][0]) <= 1e-6 * st["worst"][0], (ns, st)
    print(
        f"EXPERTS_HOST_IDENTICAL layer={LAYER} mesh={rows}x{cols} impl={mod.impl} host_weights new={t_new:.1f} s "
        f"per-expert={t_old:.1f} s stats={mod.bfp4_stats}"
    )


def _dev0_last(t):
    ts = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(ts[i]) for i in (0, len(ts) - 1)]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_load_cache(mesh_device):
    S = spec()
    ck, cfg, chunk = hooks._checkpoint(S), hooks._config(S), hooks._chunk(S)
    times = {}
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    mod = E.build_experts(mesh_device, ck, cfg, LAYER, chunk, cache_dir=None)
    pr.disable()
    times["off"] = time.time() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(f"EXPERTS_LOAD_PROFILE layer={LAYER} off wall={times['off']:.1f} s\n{s.getvalue()}")
    mod.free()

    cdir = E.expert_cache_dir(S)
    prefix = E.expert_cache_prefix(cdir, ck, LAYER, E.settings.get("EXPERTS_IMPL"), mesh_device, cfg.n_experts)
    stem = Path(prefix)
    for f in list(stem.parent.glob(f"{stem.name}.*")) if stem.parent.is_dir() else []:
        f.unlink()  # (a clean miss)
    t0 = time.time()
    miss = E.build_experts(mesh_device, ck, cfg, LAYER, chunk, cache_dir=cdir)
    times["miss"] = time.time() - t0
    t0 = time.time()
    hit = E.build_experts(mesh_device, ck, cfg, LAYER, chunk, cache_dir=cdir)
    times["hit"] = time.time() - t0
    assert hit._host_s is None and miss._host_s is not None
    assert hit.bfp4_stats == miss.bfp4_stats, (hit.bfp4_stats, miss.bfp4_stats)
    for nm in ("w_gu", "w_d", "w_rd"):
        a, b = getattr(miss.fe, nm), getattr(hit.fe, nm)
        assert (a is None) == (b is None), nm
        if a is None:
            continue
        assert a.dtype == b.dtype and a.memory_config() == b.memory_config(), nm
        for x, y in zip(_dev0_last(a), _dev0_last(b)):
            assert torch.equal(x, y), nm
    size = sum(f.stat().st_size for f in stem.parent.glob(f"{stem.name}.*"))
    print(
        f"EXPERTS_LOAD_TIMES layer={LAYER} off={times['off']:.1f} s miss={times['miss']:.1f} s "
        f"(host_weights {miss._host_s:.1f} s) hit={times['hit']:.1f} s cache={size / 2**30:.2f} GiB at {stem}"
    )
    miss.free()
    hit.free()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_cache_hits(mesh_device):
    """The seed rung's layers' experts (W.1's) from the run's cache (as the ladder builds them, hooks.device_model):
    all hits; the per-layer load time (EXPERTS_CACHE_HITS). Run after a ladder or model run filled the cache."""
    S = spec()
    ck, cfg, chunk = hooks._checkpoint(S), hooks._config(S), hooks._chunk(S)
    cdir, times = E.expert_cache_dir(S), {}
    rung = next(r for r in S.data["ladder"] if r["name"] == "seed")
    for layer in rung["layers"]:
        t0 = time.time()
        mod = E.build_experts(mesh_device, ck, cfg, int(layer), chunk, cache_dir=cdir)
        times[int(layer)] = round(time.time() - t0, 2)
        assert mod._host_s is None, f"layer {layer}: cache miss"
        mod.free()
    print(f"EXPERTS_CACHE_HITS layers={len(times)} total={sum(times.values()):.1f} s per layer={times}")
