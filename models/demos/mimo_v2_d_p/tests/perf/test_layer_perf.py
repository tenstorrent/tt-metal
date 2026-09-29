# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full decoder-layer device perf (attention + FFN), real weights, input = real-token embeddings (realistic
routing). Signposts ``L{idx}_{kind}_C{chunk_local}_ctx{ctx}``; analyze with analyze_tags.py --ops."""

import os
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

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

CTX = int(os.environ.get("MIMO_PERF_CTX", "32768"))


def _report(msg):
    """Result lines: logged, and appended to MIMO_PERF_OUT (a run under LOGURU_LEVEL=ERROR still gets them)."""
    logger.info(msg)
    if os.environ.get("MIMO_PERF_OUT"):
        with open(os.environ["MIMO_PERF_OUT"], "a") as f:
            f.write(msg + "\n")


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [int(x) for x in os.environ.get("MIMO_PERF_LAYERS", "0,1,5").split(",")])
@pytest.mark.parametrize(
    "chunk_local", [int(c) for c in os.environ.get("MIMO_PERF_CHUNK_LOCAL", "640,2048").split(",")]
)
def test_layer_perf(mesh_device, device_params, layer_idx, chunk_local):
    cfg = MiMoTextConfig.from_json()
    spec = cfg.layer_attn(layer_idx)
    sp, tp = tuple(mesh_device.shape)
    chunk = chunk_local * sp
    max_seq = (CTX + chunk - 1) // chunk * chunk
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    opts = MiMoRuntimeOptions.from_env()
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
    x_host = global_state()["embed_tokens.weight"][ids].float()[None, None]
    # MIMO_PERF_KV_ACTUAL: the chunk's start (its prior context; default the last chunk of CTX). The cache must be longer
    # than one chunk: the ring SDPA's chunked path (the only one sliding-window supports) needs Q shorter than K
    kv_actual = int(os.environ["MIMO_PERF_KV_ACTUAL"]) if os.environ.get("MIMO_PERF_KV_ACTUAL") else max_seq - chunk
    assert kv_actual % chunk == 0 and kv_actual + chunk <= max_seq and max_seq > chunk, (kv_actual, chunk, max_seq)
    kind = "GA" if spec.window is None else "SWA"
    tag = f"L{layer_idx}_{kind}_C{chunk_local}_ctx{max_seq}" + (
        f"_kv{kv_actual}" if os.environ.get("MIMO_PERF_KV_ACTUAL") else ""
    )
    host = []
    wall = []  # MIMO_PERF_WALL=N: N timed iterations (run without --profile: host + device wall time per layer)
    n_it = 1 + int(os.environ.get("MIMO_PERF_WALL", "2"))
    for it in range(n_it):
        x = ttnn.from_torch(
            x_host,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None)),
        )
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        t0 = time.perf_counter()
        out = layer(x, rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
        host.append((time.perf_counter() - t0) * 1e3)  # host: every command enqueued (device still running)
        ttnn.synchronize_device(mesh_device)
        wall.append((time.perf_counter() - t0) * 1e3)
        if it:
            signpost(f"{tag}_end")
        out.deallocate(True)
        x.deallocate(True)
    if int(os.environ.get("MIMO_PERF_TRACE", "0")):  # the layer traced: capture once, replay (no host in the loop)
        n_tr = int(os.environ["MIMO_PERF_TRACE"])
        x = ttnn.from_torch(
            x_host,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None)),
        )
        eager = layer(x, rope, trans, kv, cache_layer=0, kv_actual=kv_actual)  # (also warms this x's addresses)
        eager_h = [ttnn.to_torch(t_) for t_ in ttnn.get_device_tensors(eager)]
        eager.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = layer(x, rope, trans, kv, cache_layer=0, kv_actual=kv_actual)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        synced = []
        for _ in range(n_tr):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            synced.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        for _ in range(n_tr):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        b2b = (time.perf_counter() - t0) * 1e3 / n_tr
        s_ = sorted(synced)
        _report(
            f"TRACE {tag}: replay synced median {s_[len(s_) // 2]:.2f} ms (min {s_[0]:.2f}), back to back {b2b:.2f} ms/layer over {n_tr}"
        )
        traced_h = [ttnn.to_torch(t_) for t_ in ttnn.get_device_tensors(out)]
        assert all(torch.equal(a_, b_) for a_, b_ in zip(eager_h, traced_h)), f"{tag}: traced output differs from eager"
        ttnn.release_trace(mesh_device, tid)
        out.deallocate(True)
        x.deallocate(True)
    h_ = sorted(host[1:])
    _report(f"HOST {tag}: layer call (enqueue) median {h_[len(h_) // 2]:.2f} ms, min {h_[0]:.2f} ms")
    w_ = sorted(wall[1:])
    _report(
        f"WALL {tag}: median {w_[len(w_) // 2]:.2f} ms, min {w_[0]:.2f} ms over {len(w_)} (all {[round(v, 2) for v in wall]})"
    )
    logger.info(f"ran {tag}")
