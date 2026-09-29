# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side cost of the MoE block and of a whole decoder layer (warm calls, real weights, real-token input).

Per shape: the host time of the Python call (device idle before it, not synchronized inside: the time until every
command is enqueued), the synchronized wall time, and a traced replay (device only, no host in the loop). A cProfile
of warm calls lists the host functions (ttnn ops, descriptor builds) by own time. Run without the profiler.

``MIMO_HOST_DUMP=<file>`` saves the layer / MoE outputs (torch.save) to compare implementations bit for bit.
"""

import cProfile
import io
import os
import pstats
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.attention.attention import cache_v_dim
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager
from models.demos.mimo_v2_d_p.tt.decoder import TtDecoderLayer
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions
from models.demos.mimo_v2_d_p.tt.rope import build_indexed_rope, build_transformation_mat

ITERS = int(os.environ.get("MIMO_HOST_ITERS", "10"))


def _med(v):
    s = sorted(v)
    return s[len(s) // 2]


def _host_vs_wall(fn, mesh_device, n):
    """Per call: host time until fn returns (device idle before) and the synchronized wall time."""
    host, wall = [], []
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        host.append((t1 - t0) * 1e3)
        wall.append((t2 - t0) * 1e3)
        out.deallocate(True)
    return _med(host), _med(wall)


def _traced(fn, mesh_device, n):
    """Median synchronized replay time (ms) of fn captured into a trace, and its output (host, per device)."""
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        t.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh_device, tid)
    out.deallocate(True)
    return _med(t)


def _profile(fn, mesh_device, n, top=25):
    ttnn.synchronize_device(mesh_device)
    pr = cProfile.Profile()
    for _ in range(n):
        pr.enable()
        out = fn()
        pr.disable()
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
    return "\n".join(l for l in s.getvalue().splitlines() if l.strip())


def _per_op(fn, mesh_device, n):
    """Median host time (us) of each ttnn op call of fn, in call order, and the rest of the call (Python glue)."""
    cls = ttnn.decorators.FastOperation
    orig = cls.__call__
    seq = []

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        seq.append((self.python_fully_qualified_name, time.perf_counter() - t0))
        return r

    runs, totals = [], []
    cls.__call__ = timed
    try:
        for _ in range(n):
            ttnn.synchronize_device(mesh_device)
            seq.clear()
            t0 = time.perf_counter()
            out = fn()
            totals.append(time.perf_counter() - t0)
            runs.append(list(seq))
            ttnn.synchronize_device(mesh_device)
            out.deallocate(True)
    finally:
        cls.__call__ = orig
    rows = [(runs[0][i][0], _med([r[i][1] for r in runs]) * 1e6) for i in range(len(runs[0]))]
    ops_us = sum(us for _, us in rows)
    lines = [f"  {i:2d} {name:60s} {us:8.1f} us" for i, (name, us) in enumerate(rows)]
    lines.append(
        f"  ops {ops_us:.1f} us, whole call {_med(totals) * 1e6:.1f} us (glue {_med(totals) * 1e6 - ops_us:.1f} us)"
    )
    return "\n".join(lines)


def _host(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [int(x) for x in os.environ.get("MIMO_HOST_LAYERS", "1").split(",")])
@pytest.mark.parametrize(
    "chunk_local", [int(c) for c in os.environ.get("MIMO_HOST_CHUNK_LOCAL", "640,2048").split(",")]
)
def test_host_overhead(mesh_device, device_params, layer_idx, chunk_local):
    cfg = MiMoTextConfig.from_json()
    assert cfg.is_moe(layer_idx), layer_idx
    spec = cfg.layer_attn(layer_idx)
    sp, tp = tuple(mesh_device.shape)
    chunk = chunk_local * sp
    max_seq = 2 * chunk  # the ring SDPA's chunked path needs a cache longer than the chunk
    kv_actual = chunk
    opts = MiMoRuntimeOptions.from_env()
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=opts.num_links, topology=sp_topo)
    layer = TtDecoderLayer(
        mesh_device,
        cfg,
        layer_idx,
        layer_state(layer_idx, cfg),
        ccl=ccl,
        sp_topology=sp_topo,
        seq_len_per_chip=chunk_local,
        options=opts,
    )
    kv = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=max_seq,
        n_kv_local=layer.attn.nkv_l,
        k_dim=spec.head_dim,
        v_dim=cache_v_dim(spec),
    )
    rope = build_indexed_rope(mesh_device, spec, max_seq_len=max_seq, chunk_size=chunk)
    trans = build_transformation_mat(mesh_device)
    ids = hf.tokenize_prompt(chunk)
    x = ttnn.from_torch(
        global_state()["embed_tokens.weight"][ids].float()[None, None],
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None)),
    )
    h = layer.post_attn_norm(x)  # an MoE input with the real routing statistics
    run_layer = lambda: layer(x, rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
    run_moe = lambda: layer.ffn(h)
    tag = f"L{layer_idx}_C{chunk_local}"

    outs = {}
    for name, fn in (("moe", run_moe), ("layer", run_layer)):
        for _ in range(2):  # compile + first-address warm-up
            fn().deallocate(True)
        ttnn.synchronize_device(mesh_device)
        o = fn()
        outs[name] = _host(o)
        o.deallocate(True)
        host, wall = _host_vs_wall(fn, mesh_device, ITERS)
        o = fn()
        again = _host(o)
        o.deallocate(True)
        assert all(torch.equal(a, b) for a, b in zip(outs[name], again)), f"{tag} {name}: not deterministic"
        replay = _traced(fn, mesh_device, ITERS)
        logger.info(
            f"HOST {tag} {name}: host call {host:.3f} ms, synced wall {wall:.3f} ms, traced replay {replay:.3f} ms "
            f"(exposed host {wall - replay:+.3f} ms)"
        )
        logger.info(f"OPS {tag} {name} (host per ttnn call, median of {ITERS}):\n{_per_op(fn, mesh_device, ITERS)}")
        if os.environ.get("MIMO_HOST_CPROFILE"):
            logger.info(f"PROFILE {tag} {name} ({ITERS} warm calls):\n{_profile(fn, mesh_device, ITERS)}")
    if os.environ.get("MIMO_HOST_DUMP"):
        path = f"{os.environ['MIMO_HOST_DUMP']}.{tag}.pt"
        torch.save(outs, path)
        logger.info(f"saved {path}")
