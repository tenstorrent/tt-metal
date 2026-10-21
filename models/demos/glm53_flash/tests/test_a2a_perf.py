# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cost of an expert-parallel MoE exchange at GLM-5.3-Flash decode shapes: ttnn.all_to_all_dispatch +
all_to_all_combine (the DeepSeek-V3 Galaxy decode scheme) vs the hybrid layout's all_gather_rows +
reduce_scatter_rows + all_reduce_cols, traced replay on a (2,4) block (8 devices; 288 experts = 36 per
device) at 4 and 32 tokens per device (= 32 and 256 users on the 8x4 at 1 and 8 per chip).

    GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_a2a_perf.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
H, K, E = 4096, 8, 288
N_REP = 30


def _timed(mesh_device, fn, label):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    logger.info(f"A2A {label}: {us:.1f} us per call (traced replay)")
    ttnn.release_trace(mesh_device, tid)
    return out, us


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=80_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_expert_exchange(mesh_device, B):
    rows, cols = MESH
    D = rows * cols
    E_loc = E // D
    mesh = GLMMesh.from_device(mesh_device)
    torch.manual_seed(0)
    # tokens [B*D, 1, 1, H] sharded over all devices (B per device); indices [B*D, 1, 1, K]; mapping expert -> device
    tokens = torch.randn(B * D, 1, 1, H, dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(B * D)]).reshape(B * D, 1, 1, K).to(torch.int16)
    mapping = torch.zeros(1, 1, E, D, dtype=torch.int16)
    for e in range(E):
        mapping[0, 0, e, e // E_loc] = 1
    shard_all = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(0, None))  # rows get B*cols, then cols split
    # row-major flattening of the mesh: shard dim 0 over rows, then over cols -> use a two-level split
    tok_rows = tokens.reshape(rows, cols * B, 1, 1, H)
    tokens_dev = ttnn.from_torch(
        tokens,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    idx_dev = ttnn.from_torch(
        idx,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    map_dev = ttnn.from_torch(
        mapping,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    results = {}
    for axis in (0, 1):  # cluster_axis=None (full-mesh dispatch) is not exercised on Blackhole and hung on a 2x4 block
        try:
            out, us = _timed(
                mesh_device,
                lambda: ttnn.all_to_all_dispatch(tokens_dev, idx_dev, map_dev, cluster_axis=axis, num_links=1),
                f"B={B}/dev dispatch cluster_axis={axis} [{B * D} tokens, H={H}, K={K}, E={E}]",
            )
            out_tokens, out_meta = out
            logger.info(f"   dispatch out shapes {out_tokens.shape} {out_meta.shape}")
            results[("dispatch", axis)] = us
            # combine: expert contributions [E_loc(per device), T, 1, H]: one row per (expert, token) -> back to owners
            T = out_tokens.shape[0]
            contrib = ttnn.from_torch(
                torch.randn(E, T, 1, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
            _, us2 = _timed(
                mesh_device,
                lambda: ttnn.all_to_all_combine(contrib, out_meta, map_dev, cluster_axis=axis, num_links=1),
                f"B={B}/dev combine cluster_axis={axis}",
            )
            results[("combine", axis)] = us2
            ttnn.deallocate(contrib)
        except Exception as e:  # noqa: BLE001
            logger.info(f"A2A B={B} cluster_axis={axis} failed: {str(e).splitlines()[0][:200]}")
    # the hybrid layout's three collectives at the same token count per row
    x = ttnn.from_torch(
        torch.randn(1, 1, rows * B * cols, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(2, None)),
    )
    _, g = _timed(mesh_device, lambda: mesh.all_gather_rows(x, dim=2), f"B={B}/dev hybrid all_gather_rows")
    big = ttnn.from_torch(
        torch.randn(1, 1, rows * rows * B * cols, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(2, None)),
    )
    _, rs = _timed(mesh_device, lambda: mesh.reduce_scatter_rows(big, dim=2), f"B={B}/dev hybrid reduce_scatter_rows")
    _, ar = _timed(mesh_device, lambda: mesh.all_reduce_cols(x), f"B={B}/dev hybrid all_reduce_cols")
    logger.info(
        f"SUMMARY B={B}/dev: hybrid 3 collectives {g + rs + ar:.0f} us; a2a "
        + ", ".join(f"{k[0]}@{k[1]}={v:.0f}" for k, v in results.items())
    )


# ------------------------------------------------------------------------------------------------ correctness
def golden_exchange(tokens, idx, contrib, rows, cols, E):
    """Torch reference of the EP exchange at the model's placement (tokens of row r replicated over its columns,
    chip d = r * cols + c owns experts [d E/D, (d+1) E/D)). tokens [R*T, H] (row r's tokens at [rT, (r+1)T)),
    idx [R*T, K] global ids, contrib [D, R*T, H] = the contribution device d would compute for gathered token g.
    Returns (present [D, R*T] bool: token g reaches device d; combined [D, T, H]: the per-device sum over its
    column's devices of the contributions for its own tokens; slot [D, R*T] int: the k-slot device d writes for
    token g, -1 if none)."""
    D = rows * cols
    E_loc = E // D
    G, K = idx.shape
    T = G // rows
    owner = idx // E_loc  # [G, K] device of each pick
    present = torch.zeros(D, G, dtype=torch.bool)
    slot = torch.full((D, G), -1, dtype=torch.long)
    for d in range(D):
        for g in range(G):
            ks = [k for k in range(K) if owner[g, k] == d]
            if ks:
                present[d, g] = True
                # the combine writes the slot of the device's FIRST local expert (ascending id) in the token's list
                first = min(ks, key=lambda k: idx[g, k])
                slot[d, g] = first
    combined = torch.zeros(D, T, contrib.shape[-1])
    for r in range(rows):
        for c in range(cols):
            d = r * cols + c
            for b in range(T):
                g = r * T + b
                for r2 in range(rows):
                    d2 = r2 * cols + c
                    if present[d2, g]:
                        combined[d, b] += contrib[d2, g].float()
    return present, combined, slot


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("T", [4, 32])
def test_dispatch_combine_golden(mesh_device, T):
    """ttnn.all_to_all_dispatch + all_to_all_combine(local_reduce=True) along the mesh rows (cluster_axis=0) at the
    EP path's shapes and dtypes vs a torch golden: tokens [T,1,1,H] bf16 RM per device (row-sharded, column-
    replicated), ids [T,1,1,K] uint16, mapping [1,1,E,D] uint16; the dispatched rows of every device, the gathered
    metadata, and the K-slot sum of the combine into a zeroed output (the untouched slots must stay 0).
    Fabric from GLM53_FABRIC (1d = the Blackhole CI config of these ops, 2d = the model's block fabric)."""
    rows, cols = MESH
    D = rows * cols
    E_loc = E // D
    mesh = GLMMesh.from_device(mesh_device)
    torch.manual_seed(1)
    G = rows * T
    tokens = torch.randn(G, 1, 1, H, dtype=torch.bfloat16)
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(G)]).to(torch.int64)  # [G, K] distinct global ids
    mapping = torch.eye(D, dtype=torch.int32).repeat_interleave(E_loc, dim=0).reshape(1, 1, E, D)
    contrib = torch.randn(D, G, H, dtype=torch.bfloat16)
    present, combined_ref, slot = golden_exchange(tokens.reshape(G, H), idx, contrib, rows, cols, E)
    logger.info(
        f"T={T}/row: tokens reach {present.sum(0).float().mean():.2f} of {D} devices on average; rows per device "
        f"max {present.sum(1).max()} / mean {present.sum(1).float().mean():.1f}; links axis0 "
        f"{ttnn.get_num_links(mesh_device, 0)} axis1 {ttnn.get_num_links(mesh_device, 1)}"
    )
    rm = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tok_dev = ttnn.from_torch(tokens, dtype=ttnn.bfloat16, mesh_mapper=mesh.shard_rows(0), **rm)
    idx_dev = ttnn.from_torch(
        idx.reshape(G, 1, 1, K).to(torch.int32), dtype=ttnn.uint16, mesh_mapper=mesh.shard_rows(0), **rm
    )
    map_dev = ttnn.from_torch(
        mapping,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kw = dict(cluster_axis=0, num_links=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    for it in range(2):  # twice: the second call exercises the program cache / semaphore reuse
        t0 = time.perf_counter()
        out, meta = ttnn.all_to_all_dispatch(tok_dev, idx_dev, map_dev, **kw)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"dispatch {it} done in {time.perf_counter() - t0:.2f} s: out {out.shape} meta {meta.shape}")
        assert tuple(out.shape) == (1, G, 1, H) and tuple(meta.shape) == (1, G, 1, K), (out.shape, meta.shape)
        outs = [ttnn.to_torch(d).reshape(G, H) for d in mesh.device_tensors(out)]
        metas = [ttnn.to_torch(d).reshape(G, K).to(torch.int64) for d in mesh.device_tensors(meta)]
        bad = []
        for d in range(D):
            if not torch.equal(metas[d], idx):
                bad.append(("meta", d))
            for g in range(G):
                if present[d, g] and not torch.equal(outs[d][g], tokens[g, 0, 0]):
                    bad.append(("tok", d, g))
        assert not bad, bad[:10]
        logger.info(f"dispatch {it}: every dispatched row and the gathered metadata byte-exact on all {D} devices")
        # combine: contributions in dispatch-row order (one per (device, gathered token)), zeroed output, sum over K
        contrib_dev = ttnn.from_torch(
            contrib.reshape(D, G, 1, H),
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            **rm,
        )  # [1, G, 1, H] per device
        zero = ttnn.zeros((K, T, 1, H), dtype=ttnn.bfloat16, **rm)
        t0 = time.perf_counter()
        comb = ttnn.all_to_all_combine(contrib_dev, meta, map_dev, local_reduce=True, output_tensor=zero, **kw)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"combine {it} done in {time.perf_counter() - t0:.2f} s: {comb.shape}")
        assert tuple(comb.shape) == (K, T, 1, H), comb.shape
        combs = [ttnn.to_torch(d).reshape(K, T, H).float() for d in mesh.device_tensors(comb)]
        worst = 1.0
        for r in range(rows):
            for c in range(cols):
                d = r * cols + c
                got = combs[d]
                # slot check: device d2 in column c writes slot[d2, g] for its tokens; untouched slots stay 0
                for b in range(T):
                    g = r * T + b
                    written = {int(slot[r2 * cols + c, g]) for r2 in range(rows) if present[r2 * cols + c, g]}
                    for k in range(K):
                        if k not in written:
                            assert torch.equal(got[k, b], torch.zeros(H)), (d, b, k, "slot not zero")
                p = pcc(combined_ref[d].reshape(1, T, H), got.sum(0).reshape(1, T, H))
                worst = min(worst, p)
        assert worst > 0.9999, worst
        logger.info(
            f"combine {it}: K-slot sum == sum of the column's device contributions on all devices, pcc {worst:.6f}"
        )
        for t in (out, meta, contrib_dev, comb):
            ttnn.deallocate(t)


# ------------------------------------------------------------------------------------------- decode step cost
from models.demos.glm53_flash.tests.test_moe import LAYER, realistic_mlp_input, weights  # noqa: E402,F401 (fixture)
from models.demos.glm53_flash.tt.config import GLM53FlashConfig  # noqa: E402
from models.demos.glm53_flash.tt.moe import TtMoE  # noqa: E402


def _time_stage(mesh_device, fn, reps=20):
    """Compile, capture one call as a trace, warm replay, ``reps`` replays -> (the trace-time output, us per call).
    The output tensors of the capture call persist (the replays write into them) and serve as the next stage's
    inputs."""
    out = fn()
    ttnn.synchronize_device(mesh_device)
    _dealloc(out)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / reps * 1e6
    ttnn.release_trace(mesh_device, tid)
    return out, us


def _dealloc(out):
    for t in out if isinstance(out, (tuple, list)) else (out,):
        if t is not None:
            ttnn.deallocate(t)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=200_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("T", [4, 32])
def test_moe_decode_components(mesh_device, weights, T):
    """The MoE decode step of one layer (real layer-4 weights and routing) on the block at T users per row: the EP
    path (router -> [x|w] rows + ids dispatch -> expert ops -> combine + K-sum) vs the hybrid stream path (dense
    router -> all_gather_rows x2 -> local expert slices -> reduce_scatter_rows), every stage as a traced replay,
    plus the shared expert + the column all-reduce and the whole forward of each layout; the two layouts' outputs
    are compared. Logs a markdown table (doc/mesh/orchestration_v2.md keeps the numbers)."""
    rows, cols = MESH
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mlp = {k[len("mlp.") :]: v for k, v in raw.items() if "experts." not in k or "shared_experts" in k}
    mlp["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp["experts.down_proj"] = sd["experts.down_proj"]
    mesh = GLMMesh.from_device(mesh_device)
    t0 = time.perf_counter()
    moe_ep = TtMoE(mesh_device, cfg, mlp, mesh=mesh, layout="ep")
    t1 = time.perf_counter()
    moe_h = TtMoE(mesh_device, cfg, mlp, mesh=mesh, layout="hybrid", stream_experts=True)
    logger.info(
        f"layer {LAYER} on {rows}x{cols}: EP built in {t1 - t0:.0f} s, hybrid (stream) in {time.perf_counter() - t1:.0f} s"
    )
    x_all = realistic_mlp_input(snapshot, text, raw, rows * 32).reshape(1, 1, rows * 32, -1)
    xs = torch.cat([x_all[:, :, 32 * r : 32 * r + T] for r in range(rows)], dim=2)  # row r: T tokens
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, rows * T, -1)).reshape(1, 1, rows * T, -1)
    x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)  # [1,1,T,H] per chip, replicated over the columns
    res = {}
    # ---- EP stages
    (w16, idx, _), res["ep router"] = _time_stage(mesh_device, lambda: moe_ep.router(x, ep_rows=True))
    (tok, meta), res["ep dispatch (+6 ops)"] = _time_stage(
        mesh_device, lambda: moe_ep.experts.dispatch_rows(x, w16, idx)
    )
    out, res["ep experts (2 ops)"] = _time_stage(mesh_device, lambda: moe_ep.experts.run_experts(tok, meta))
    y_ep, res["ep combine + K-sum (+4 ops)"] = _time_stage(
        mesh_device, lambda: moe_ep.experts.combine_rows(out, meta, T)
    )
    rows_dev = [
        int((ttnn.to_torch(d).reshape(-1, meta.shape[-1]).to(torch.int64) // moe_ep.experts.E_local == i).any(-1).sum())
        for i, d in enumerate(mesh.device_tensors(meta))
    ]
    logger.info(f"T={T}/row: dispatched rows per device {rows_dev} (of {rows * T} gathered)")
    # ---- hybrid stages
    (_, _, dense), res["hybrid router (dense)"] = _time_stage(mesh_device, lambda: moe_h.router(x, dense_only=True))
    (x_g, d_g), res["hybrid all_gather_rows x2"] = _time_stage(
        mesh_device, lambda: (mesh.all_gather_rows(x, dim=2), mesh.all_gather_rows(dense, dim=2))
    )
    y_g, res["hybrid local experts"] = _time_stage(mesh_device, lambda: moe_h.experts._local_experts(x_g, d_g))
    y_h, res["hybrid reduce_scatter_rows"] = _time_stage(mesh_device, lambda: mesh.reduce_scatter_rows(y_g, dim=2))
    # ---- common: shared expert, column all-reduce
    sh, res["shared expert (TP)"] = _time_stage(mesh_device, lambda: moe_h.shared(x, reduce=False))
    _, res["all_reduce_cols"] = _time_stage(mesh_device, lambda: mesh.all_reduce_cols(y_h))
    # ---- whole forwards
    y_ep_full, res["EP forward total"] = _time_stage(mesh_device, lambda: moe_ep(x))
    y_h_full, res["hybrid forward total"] = _time_stage(mesh_device, lambda: moe_h(x))
    ep_sh = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y_ep_full)]
    h_sh = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y_h_full)]
    worst_ep = worst_h = worst_x = 1.0
    for r in range(rows):
        ref = y_hf[:, :, r * T : (r + 1) * T]
        for c in range(cols):
            worst_ep = min(worst_ep, pcc(ref, ep_sh[r * cols + c][:, :, :T]))
            worst_h = min(worst_h, pcc(ref, h_sh[r * cols + c][:, :, :T]))
            worst_x = min(worst_x, pcc(h_sh[r * cols + c][:, :, :T], ep_sh[r * cols + c][:, :, :T]))
    logger.info(f"T={T}/row: worst pcc vs HF: EP {worst_ep:.6f}, hybrid {worst_h:.6f}; EP vs hybrid {worst_x:.6f}")
    logger.info(f"| stage (T={T}/row, {rows}x{cols} block, layer {LAYER}) | us per step |")
    logger.info("|---|---|")
    for k, v in res.items():
        logger.info(f"| {k} | {v:.0f} |")
    ep_sum = (
        res["ep router"] + res["ep dispatch (+6 ops)"] + res["ep experts (2 ops)"] + res["ep combine + K-sum (+4 ops)"]
    )
    h_sum = (
        res["hybrid router (dense)"]
        + res["hybrid all_gather_rows x2"]
        + res["hybrid local experts"]
        + res["hybrid reduce_scatter_rows"]
    )
    logger.info(
        f"SUMMARY T={T}/row: EP stages {ep_sum:.0f} us (forward {res['EP forward total']:.0f}); hybrid stages {h_sum:.0f} us (forward {res['hybrid forward total']:.0f})"
    )
    assert worst_ep > 0.995 and worst_h > 0.995, (worst_ep, worst_h)
