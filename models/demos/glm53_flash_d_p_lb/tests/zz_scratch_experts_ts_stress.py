# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: hang stress of the K3 token-split block's routed experts at GLM-5.3-Flash's shape (tt/experts_ts.py's
block, routed() only: routing gather, planner, dispatch prep, arena zero, sparse dispatch, the fused flat expert +
combine, tilize), the same inputs every call. Per routing (GLM_TSS_ROUTINGS: uniform, hot = expert 0 in every token,
golden = layer GLM_TSG_LAYER's real top-8 from the component golden): a few eager calls, then one call captured as a
trace and replayed GLM_TSS_REPLAYS times (GLM_TSS_BATCH replays between syncs; the host is out of the loop), or, with
GLM_TSS_EAGER=1, the call itself in a loop. A hang trips the device timeout (TT_METAL_OPERATION_TIMEOUT_SECONDS).
Fake weights (control flow does not depend on their values).

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 BRINGUP_TRACE_REGION_BYTES=100000000 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_experts_ts_stress.py -s

GLM_TSS_CHUNK (2048 tokens: 256 a chip), GLM_TSS_REPLAYS (20000), GLM_TSS_BATCH (500), GLM_TSS_ROTATE (step: the trace also rotates every expert id by step mod 288 before each
call - a different routing each replay, hot experts moving across chips), GLM_TSS_REBUILD (n: instead,
n fresh blocks, two calls each - the first call after a build)."""

import os
import re
import time

import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import component_golden, device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)
H, I, E, K = 4096, 2048, 288, 8


def _routing(name, T, g):
    if name.startswith("file:"):  # (a saved real routing, e.g. hang/std_hooks.py save_routing: ids / w [T, K])
        r = torch.load(name[5:])
        rep = -(-T // r["ids"].shape[0])
        return r["ids"].repeat(rep, 1)[:T], r["w"].repeat(rep, 1)[:T]
    if name == "golden":
        gl = component_golden(S_)
        r = gl[0].layer(gl[1], int(os.environ.get("GLM_TSG_LAYER", "4")))["router"].float()
        r = r.repeat(-(-T // r.shape[0]), 1)[:T]
        w, ids = torch.topk(r, K, dim=-1)
        return ids, w
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    if name == "hot":
        has = (ids == 0).any(1)
        ids[~has, 0] = 0
    w = torch.rand(T, K, generator=g) + 0.1
    return ids, w / w.sum(1, keepdim=True) * 2.5


@mesh_parametrize
def test_experts_ts_stress(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.tt.common import split_from_host
    from models.demos.glm53_flash_d_p.tt.experts_ts import TtExpertsTs

    hooks.apply_device_settings(S_)
    T = int(os.environ.get("GLM_TSS_CHUNK", "2048"))
    n = mesh_device.get_num_devices()
    s = T // n
    replays = int(os.environ.get("GLM_TSS_REPLAYS", "20000"))
    batch = int(os.environ.get("GLM_TSS_BATCH", "500"))
    eager = os.environ.get("GLM_TSS_EAGER") == "1"
    ts = TtExpertsTs(
        mesh_device, 4, "fake", num_experts=E, emb_dim=H, hidden_dim=I, top_k=K, max_seq_len=T, cache=False, num_links=2
    )
    g = torch.Generator().manual_seed(0)
    for name in os.environ.get("GLM_TSS_ROUTINGS", "uniform,hot,golden").split(","):  # (file:<path> too)
        label = re.sub(r"\W", "_", "file_" + os.path.basename(name[5:])) if name.startswith("file:") else name
        ids, w = _routing(name, T, g)
        x = torch.randn(T, H, generator=g).to(torch.bfloat16)
        xd = split_from_host(mesh_device, x.view(1, 1, T, H))
        idd = split_from_host(mesh_device, ids.view(1, 1, T, K).to(torch.int32), dtype=ttnn.uint16)
        wd = split_from_host(mesh_device, w.view(1, 1, T, K).to(torch.bfloat16))
        for _ in range(3):  # (eager: the block built, rebuilt once if the arena moved, every program compiled)
            ttnn.deallocate(ts(xd, idx=idd, wts=wd, split=True, full_mesh=True))
        ttnn.synchronize_device(mesh_device)
        rebuilds = int(os.environ.get("GLM_TSS_REBUILD", "0"))
        if rebuilds:  # (a fresh block each time: its first call is a program-cache miss, as at the original hang)
            from models.demos.glm53_flash_d_p.tt import experts_ts

            t0 = time.time()
            for i in range(rebuilds):
                experts_ts._BLOCKS.clear()
                for _ in range(2):
                    ttnn.deallocate(ts(xd, idx=idd, wts=wd, split=True, full_mesh=True))
                ttnn.synchronize_device(mesh_device)
                if (i + 1) % 10 == 0:
                    logger.info(f"TSS {label}: {i + 1} / {rebuilds} rebuilds, {time.time() - t0:.0f} s")
            logger.info(f"TSS {label}: {rebuilds} rebuilds (first call a cache miss each), no hang")
            for t_ in (xd, idd, wd):
                ttnn.deallocate(t_)
            continue
        blk = ts._block(s)
        x2 = ttnn.reshape(xd, [s, H])
        i2, w2 = ttnn.reshape(idd, [s, K]), ttnn.reshape(wd, [s, K])
        ttnn.deallocate(blk.routed(x2, i2, w2, ts.flat))
        ttnn.synchronize_device(mesh_device)
        cnt = torch.bincount(ids.flatten(), minlength=E).view(n, -1).sum(1).tolist()
        mx = int(torch.bincount(ids.flatten(), minlength=E).max())
        logger.info(
            f"TSS {label}: rows per chip {cnt}, max rows/expert {mx}, {'eager' if eager else 'traced'}, {replays} calls"
        )
        tr = None
        step = int(os.environ.get("GLM_TSS_ROTATE", "0"))  # (in the trace: every expert id += step mod E, each replay)
        fids = ttnn.typecast(idd, ttnn.float32) if step else None

        def rotate():  # (in place: the float ids, then the routing buffer routed() reads; distinct ids stay distinct)
            f2 = ttnn.add(fids, float(step))
            m = ttnn.ge(f2, float(E))
            f3 = ttnn.subtract(f2, ttnn.multiply(m, float(E)))
            ttnn.copy(f3, fids)
            u = ttnn.typecast(fids, ttnn.uint16)
            ttnn.copy(u, idd)
            for t_ in (f2, m, f3, u):
                ttnn.deallocate(t_)

        if not eager:
            if step:
                rotate()  # (once eagerly: its programs compiled before the capture)
                ttnn.deallocate(blk.routed(x2, i2, w2, ts.flat))
                ttnn.synchronize_device(mesh_device)
            tr = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            if step:
                rotate()
            yt = blk.routed(x2, i2, w2, ts.flat)
            ttnn.end_trace_capture(mesh_device, tr, cq_id=0)
        t0, done = time.time(), 0
        while done < replays:
            k = min(batch, replays - done)
            for _ in range(k):
                if eager:
                    ttnn.deallocate(blk.routed(x2, i2, w2, ts.flat))
                else:
                    ttnn.execute_trace(mesh_device, tr, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            done += k
            if done % (batch * 4) == 0 or done == replays:
                el = time.time() - t0
                logger.info(f"TSS {label}: {done} / {replays} calls, {el:.0f} s ({el / done * 1e3:.2f} ms a call)")
        if tr is not None:
            ttnn.release_trace(mesh_device, tr)
            ttnn.deallocate(yt)
        if step:  # (the routing after the replays: it did move)
            got = torch.cat([ttnn.to_torch(t_).reshape(-1, K) for t_ in ttnn.get_device_tensors(idd)]).long()
            # (the rotations the device applied: eager ones + one per replay - the capture itself only records)
            found = [k for k in range(E) if torch.equal(got, (ids.long() + step * k) % E)]
            applied = [k for k in found if k == (1 + replays) % E]
            logger.info(
                f"TSS {label}: routing after the replays = start rotated by {found} x {step} (mod {E}); expected "
                f"{(1 + replays) % E}: {'OK' if applied else 'MISMATCH'}"
            )
            ttnn.deallocate(fids)
        for t_ in (xd, idd, wd):
            ttnn.deallocate(t_)
        logger.info(f"TSS {label}: {replays} calls, no hang")
