# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert placement bench: the production MoE chain (all_to_all_dispatch_metadata -> moe_compute -> fused moe_tail -> reduce_scatter) on the 4x8 mesh,
random bfp8 weights, for the SEQUENTIAL expert layout (``MP_KIND=seq``) or a static PLACEMENT (``MP_KIND=placed``: expert e -> chip ``assign[e]``, 12 per chip)
on REAL routing samples (captured expert ids of test_batch_scaling.py, DSV41_ROUTE_CAPTURE=1).

One layout per process (two layouts in one process hang trace replay). Per T (tokens per mesh row, N = 4 T tokens) and sample it measures the per-call time of the
chain (trace of 4 calls minus a trace of 1 call) and writes JSON lines ``MPR {...}``.

Env: MP_KIND (seq|placed), MP_ASSIGN (json {layer: [384 chips]}, placed), MP_LAYERS (comma list of MoE layers whose routes are sampled; placed: ONE layer),
MP_ROUTES (comma list of routes.pt), MP_TS (default 2,4,12), MP_NS (samples per T and layer, default 16), MP_PCC (1: golden check at the first T), MP_OUT (dir),
MP_PCC_LAYER_SEQ (unused). Sample shapes (users x positions): T=2: 8x1, 4: 16x1, 8: 8x4, 12: 8x6, 16: 16x4 (B=8 plain, B=16 plain, B=8 k=3, B=8 k=5, B=16 k=3)."""
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ttnn

os.environ.setdefault("MOE_COMPUTE_BFP8_WEIGHTS", "1")
os.environ.setdefault("MOE_COMPUTE_FP32_ACC", "1")
os.environ.setdefault("MOE_COMPUTE_PIN_EAST_X", "7")

from models.common.moe.tt_moe_decode import _TTMoEDecodeExpertState as ES
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock

H, NE, I, ROWS, COLS, K = 5120, 384, 2304, 4, 8, 6
OUT = os.environ.get("MP_OUT", "/tmp")
TS = [int(x) for x in os.environ.get("MP_TS", "2,4,12").split(",")]
SHAPES = {2: (8, 1), 4: (16, 1), 8: (8, 4), 12: (8, 6), 16: (16, 4)}


def gen(shape, seed, scale=0.02, chunk=32):
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(shape, dtype=torch.bfloat16)
    for i in range(0, shape[0], chunk):
        out[i : i + chunk] = (torch.randn((min(chunk, shape[0] - i),) + tuple(shape[1:]), generator=g) * scale).to(
            torch.bfloat16
        )
    return out


def upload(md, t):
    return ttnn.from_torch(
        t, device=md, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensorToMesh(md, dim=1)
    )


def mapping_list(kind, assign):
    if kind == "seq":
        # the production sequential mapping of the YAML (cluster_axis 0): expert e -> chip (row (e % 48) // 12) * 8 + e // 48
        return [((e % 48) // 12) * COLS + e // 48 for e in range(NE)]
    return [int(x) for x in assign]


def build_state(md, mapping, W0, W1, W2):
    """weights are in expert-id order; the permutation that makes ShardTensorToMesh(dim=experts) land each expert on mapping[e] is argsort(mapping, stable)."""
    m = torch.tensor(mapping, dtype=torch.int)
    assert all(int((m == d).sum()) == NE // 32 for d in range(32)), "mapping must hold 12 experts per chip"
    perm = torch.argsort(m, stable=True)
    tt = [upload(md, w[perm][None]) for w in (W0, W1, W2)]
    mp = ES._init_expert_mapping(m, None, md, (ROWS, COLS), 0)
    w01, w2 = ES._init_total_expert_weights_impl(tt[0], tt[1], tt[2], md, 1, NE // 32, H, I, False)
    return SimpleNamespace(tt_expert_mapping=mp, tt_w0_w1=w01, tt_w2=w2)


def build_dec(md, state, T):
    cfg, _ = DSV41MoEBlock._configs(md, T, ttnn.Topology.Linear, False)
    if (
        os.environ.get("MP_DRAM_IN") == "1"
    ):  # glue experiment: dispatch reads its inputs from DRAM (no 3 x to_memory_config to L1)
        cfg = cfg.model_copy(
            update={
                "dispatch_input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "dispatch_input_expert_scores_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            }
        )
    dec = DSV41MoEBlock._weightless_decode(md, cfg, None, expert_state=state)
    dec._mesh = md
    return dec


def load_routes(paths):
    arrs = []
    for p in paths:
        r = torch.load(p)
        a = np.stack([np.stack([t.numpy() for t in st]) for st in r]).transpose(1, 0, 2, 3)  # [L, S, B, 6]
        if (
            os.environ.get("MP_UHALF") == "1"
        ):  # held-out users: the second half of each file (sim2.py SPLITALL trains on the first half)
            a = a[:, :, a.shape[2] // 2 :]
        arrs.append(a)
    S = min(a.shape[1] for a in arrs)
    return np.concatenate([a[:, :S] for a in arrs], axis=2)


def sample_ids(R, T, rng):
    """R [S, B, 6] of one layer -> [4T, 6] ids: users x consecutive steps of the shape of this T (token order: user-major)."""
    u, p = SHAPES[T]
    S, B, _ = R.shape
    p = min(p, S)
    us = rng.choice(B, size=min(u, B), replace=False)
    s0 = int(rng.integers(0, S - p + 1))
    ids = R[s0 : s0 + p][:, us].reshape(-1, K)
    n = 4 * T
    if ids.shape[0] < n:  # fewer users than the shape asks for: repeat
        ids = np.concatenate([ids] * (-(-n // ids.shape[0])))
    return torch.from_numpy(ids[:n].copy())


def stats(ids, mapping):
    cnt = {}
    for e in set(ids.flatten().tolist()):
        cnt.setdefault(mapping[e], set()).add(e)
    return dict(
        distinct=len(set(ids.flatten().tolist())),
        n_max=max(len(v) for v in cnt.values()),
        n_mean=sum(len(v) for v in cnt.values()) / 32,
    )


def golden(x, ids, sc, W0, W1, W2):
    N = x.shape[0]
    out = torch.zeros(N, H)
    for t in range(N):
        for k in range(K):
            e = int(ids[t, k])
            g = torch.clamp(x[t].float() @ W0[e].float(), max=10.0)
            u = torch.clamp(x[t].float() @ W1[e].float(), -10.0, 10.0)
            out[t] += float(sc[t, k]) * ((torch.nn.functional.silu(g) * u) @ W2[e].float())
    return out


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def capture(md, fn, k):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(k):
        o = fn()
        del o
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    return tid


def run_us(md, tid, reps=10):
    for _ in range(2):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    return (time.perf_counter() - t) / reps * 1e6


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(5400)
@torch.no_grad()
def test_moe_place_bench(mesh_device):
    md = mesh_device
    kind = os.environ.get("MP_KIND", "seq")
    layers = [int(x) for x in os.environ.get("MP_LAYERS", "4").split(",")]
    routes = load_routes(os.environ["MP_ROUTES"].split(","))
    nsamp = int(os.environ.get("MP_NS", "16"))
    assign = None
    if kind == "placed":
        assert len(layers) == 1
        assign = json.load(open(os.environ["MP_ASSIGN"]))[str(layers[0])]
    mapping = mapping_list(kind, assign)
    rep_rows = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    if os.environ.get("MP_RS_MIN") == "1":  # chain with the lean reduce_scatter (DSV41_MOE_RS_MIN=1)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean
        from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import set_rs_ccl
        from models.demos.gpt_oss.tt.ccl import CCLManager

        os.environ["DSV41_MOE_RS_MIN"] = "1"
        _ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ccl_lean.make_lean_sems(_ccl)
        set_rs_ccl(_ccl)
    t0 = time.time()
    W0, W1, W2 = gen((NE, H, I), 1), gen((NE, H, I), 2), gen((NE, I, H), 3)
    print(f"MPLOG weights generated {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    state = build_state(md, mapping, W0, W1, W2)
    print(f"MPLOG state {kind} built {time.time() - t0:.0f} s", flush=True)
    results = []

    def emit(rec):
        results.append(rec)
        print("MPR " + json.dumps(rec), flush=True)

    for ti, T in enumerate(TS):
        N = ROWS * T
        x = ttnn.from_torch(
            torch.randn(N, 1, 1, H).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep_rows,
        )
        mk = lambda ids, dev: ttnn.from_torch(
            ids.to(torch.int32).reshape(N, 1, 1, K),
            device=md if dev else None,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if dev else None,
            mesh_mapper=rep_rows,
        )
        sc_t = torch.rand(N, K) + 0.1
        sc_t = (sc_t / sc_t.sum(1, keepdim=True)).to(torch.bfloat16)
        sc = ttnn.from_torch(
            sc_t.reshape(N, 1, 1, K),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep_rows,
        )
        rng = np.random.default_rng(100 + T)
        ids0 = sample_ids(routes[layers[0]], T, rng)
        ids_dev = mk(ids0, True)
        dec = build_dec(md, state, T)
        fn = lambda d=dec: d.forward(x, sc, ids_dev, layer_id=0)
        if os.environ.get("MP_NOWARM") != "1":
            # DSV41MoEBlock.warmup(): leave a free hole under the persistent L1 allocations before the first compile so moe_compute's persistent semaphore lands at the top of L1
            # (otherwise the L1 copies of the dispatch inputs overlap it: wrong results at T >= 4 in this harness)
            nb = ttnn.get_memory_view(md, ttnn.BufferType.L1).num_banks
            tile_row = lambda n: ttnn.empty(
                [1, 1, 32, 32 * nb * n],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            hole, fence = tile_row(1), tile_row(2)
            ttnn.deallocate(hole)
            o_ = fn()
            ttnn.synchronize_device(md)
            ttnn.deallocate(o_)
            ttnn.deallocate(fence)
        tr1, tr4 = capture(md, fn, 1), capture(md, fn, 4)
        if os.environ.get("MP_PCC", "1") == "all" or (os.environ.get("MP_PCC", "1") == "1" and ti == 0):
            xh = torch.cat([ttnn.to_torch(ttnn.get_device_tensors(x)[r * COLS]) for r in range(ROWS)]).reshape(N, H)
            ref = golden(xh, ids0, sc_t, W0, W1, W2)
            o = fn()
            ttnn.synchronize_device(md)
            devs = ttnn.get_device_tensors(o)
            y = torch.zeros(N, H)
            for r in range(ROWS):
                for c in range(COLS):
                    tt = ttnn.to_torch(devs[r * COLS + c]).float().reshape(-1, H // COLS)
                    y[r * T : (r + 1) * T, c * (H // COLS) : (c + 1) * (H // COLS)] = tt[:T]
            emit(dict(T=T, kind=kind, pcc=pcc(y, ref), ref_norm=float(ref.norm()), out_norm=float(y.norm())))
        for layer in layers:
            lrng = np.random.default_rng(100 + T + 1000 * layer)  # same samples for every layout of the same (T, layer)
            for s in range(nsamp):
                ids = sample_ids(routes[layer], T, lrng)
                ttnn.copy_host_to_device_tensor(mk(ids, False), ids_dev)
                st = stats(ids, mapping)
                t1 = run_us(md, tr1)
                t4 = run_us(md, tr4)
                emit(dict(T=T, kind=kind, layer=layer, s=s, call_us=(t4 - t1) / 3, t1_us=t1, **st))
        ttnn.release_trace(md, tr1)
        ttnn.release_trace(md, tr4)
    json.dump(results, open(os.path.join(OUT, "mp_results.json"), "w"), indent=1)
