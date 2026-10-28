# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device tests of the cross-op weight prefetch (QWEN36_WEIGHT_PREFETCH, tt/weight_prefetch.py) on a Blackhole mesh.

1. test_prefetch_matmul_bit_exact: every production TP decode matmul shape (gate/up with fused SiLU, down, GDN in/out,
   attn qkv/wo) through the Tensor prefetcher equals the plain DRAM-reading 1D matmul BIT FOR BIT, at 8 and 32 rows,
   on a cold and a program-cache-hit call.
2. test_prefetch_mlp_trace: an MLP-shaped decode sequence (rms_norm -> gate/up -> mul -> down per layer, distinct
   weights per layer) captured into a trace with the prefetch requests, replayed many times: every replay equals the
   eager plain run bit for bit (no hang, no stale ring data), and the replay time is logged against the same
   sequence traced without prefetch. QWEN36_WP_ASSERT_PERF=1 makes "prefetch is not slower" a hard check.

Needs Blackhole firmware >= 19.12.0.0 (skips otherwise). No checkpoint needed (random weights).

Run (Galaxy, one 1x8 TP submesh):
    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_weight_prefetch.py -v -s
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import weight_prefetch as wp

# Per-device (K, N) at TP=8 for Qwen3.8-27B (dim 5120, hidden 17408), with the model_config.py 1D decode progcfg
# arguments: (K, N, shard_dim, num_cores, max_in0_block_w, fused SiLU). shard_dim -1 = column-parallel, 0 = row-parallel.
_TP8_SHAPES = {
    "gateup": (5120, 2176, -1, 44, 16, True),
    "down": (2176, 5120, 0, 84, 8, False),
    "gdn_in": (5120, 2080, -1, 84, 8, False),
    "gdn_out": (768, 5120, 0, 84, 8, False),
    "attn_qkv": (5120, 2048, -1, 60, 8, False),
    "attn_wo": (768, 5120, 0, 84, 8, False),
}

# The MLP up projection: gate's shape and grid without the fused SiLU (model_config mlp_w3_decode_1d_progcfg).
_UP = (5120, 2176, -1, 44, 16, False)
_TARGET = {"up": "gateup"}


def _shape(name):
    return _UP if name == "up" else _TP8_SHAPES[name]


_DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 268435456}


def _mesh_params(fn):
    shape = _resolve_mesh_shape(8)
    fn = pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(fn)


@pytest.fixture
def prefetch_mesh(mesh_device, monkeypatch):
    if not hasattr(ttnn.experimental, "is_tensor_prefetcher_supported") or not (
        ttnn.experimental.is_tensor_prefetcher_supported(mesh_device)
    ):
        pytest.skip("Tensor prefetcher unsupported (needs Blackhole firmware >= 19.12.0.0)")
    monkeypatch.setenv(wp.ENV, "all")
    wp._reset_for_tests()
    yield mesh_device
    ttnn.synchronize_device(mesh_device)
    ttnn.experimental.stop_tensor_prefetcher(mesh_device)
    wp._reset_for_tests()


def _progcfg(mesh, name):
    k, n, _, num_cores, ibw, silu = _shape(name)
    return tpc.create_matmul_1d_decode_progcfg(
        1,
        k,
        n,
        num_cores=num_cores,
        fused_activation=ttnn.UnaryOpType.SILU if silu else None,
        grid_w=mesh.compute_with_storage_grid_size().x,
        max_in0_block_w=ibw,
    )


def _torch_weight(name, tp, seed):
    """Full [out, in] checkpoint-layout weight whose per-device shard is the TP=8 decode shape."""
    k, n, dim, *_ = _shape(name)
    g = torch.Generator().manual_seed(seed)
    out_f, in_f = (n * tp, k) if dim == -1 else (n, k * tp)
    return (torch.randn(out_f, in_f, generator=g) * (in_f**-0.5)).to(torch.bfloat16)


def _load_pair(mesh, name, seed, dtype=ttnn.bfloat8_b):
    """(interleaved primary, progcfg) with a receiver-contiguous decode copy registered for it."""
    tp = mesh.get_num_devices()
    _, _, dim, *_ = _shape(name)
    pc = _progcfg(mesh, name)
    tw = _torch_weight(name, tp, seed)
    primary = tpc.shard_w(tw, mesh, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=None, dtype=dtype)
    target = _TARGET.get(name, name)
    twin = wp.load_prefetched(target, f"test.{name}", mesh, tw, dim, dtype, None, pc, tp)
    assert twin is not None, f"{name}: weight geometry does not fit the prefetch transport"
    assert wp.attach_twin(mesh, primary, twin, pc), f"{name}: registration failed (see [weight_prefetch] log)"
    return primary, pc


def _x(mesh, rows, k, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, 1, rows, k, generator=g).to(torch.bfloat16)
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _host(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _assert_bit_exact(ref, got, what):
    d = (ref - got).abs()
    assert torch.equal(ref, got), f"{what}: prefetch output differs (max|d|={d.max().item():.3e}, n={(d > 0).sum()})"


@torch.no_grad()
@_mesh_params
@pytest.mark.parametrize("rows", [8, 32])
@pytest.mark.parametrize("name", list(_TP8_SHAPES))
def test_prefetch_matmul_bit_exact(prefetch_mesh, name, rows, reset_seeds, ensure_gc):
    mesh = prefetch_mesh
    k = _TP8_SHAPES[name][0]
    primary, pc = _load_pair(mesh, name, seed=1)
    x = _x(mesh, rows, k, seed=2)
    kw = dict(compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=ttnn.L1_MEMORY_CONFIG)
    ref = _host(mesh, ttnn.linear(x, primary, program_config=pc, **kw))  # what QWEN36_WEIGHT_PREFETCH=0 runs
    for call in range(3):  # cold, then program-cache hits
        got = _host(mesh, wp.linear(x, primary, pc, **kw))
        _assert_bit_exact(ref, got, f"{name} rows={rows} call={call}")
    logger.info(f"[weight_prefetch] {name} rows={rows}: bit-exact over 3 calls")


def _mlp_layers(mesh, n_layers):
    layers = []
    for i in range(n_layers):
        layers.append({n: _load_pair(mesh, n, seed=100 * i + j) for j, n in enumerate(("gateup", "up", "down"))})
    return layers


def _mlp_step(x, layers, linear):
    kw = dict(compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=ttnn.L1_MEMORY_CONFIG)
    h = x
    for L in layers:
        hn = ttnn.rms_norm(h, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG)  # non-matmul work to prefetch under
        (wg, pg), (wu, pu), (wd, pd) = L["gateup"], L["up"], L["down"]
        g = linear(hn, wg, pg, **kw)
        u = linear(hn, wu, pu, **kw)
        ttnn.deallocate(hn)
        a = ttnn.mul(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        d = linear(a, wd, pd, **kw)
        ttnn.deallocate(a)
        h = ttnn.add(h, d, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(d)
    return h


def _plain(x, w, pc, **kw):
    return ttnn.linear(x, w, program_config=pc, **kw)


def _capture(mesh, fn):
    fn()  # compile
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return tid, out


def _time_replays(mesh, tid, n):
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / n * 1e3


@torch.no_grad()
@_mesh_params
@pytest.mark.parametrize("rows", [8, 32])
def test_prefetch_mlp_trace(prefetch_mesh, rows, reset_seeds, ensure_gc):
    mesh = prefetch_mesh
    n_layers = int(os.environ.get("QWEN36_WP_TEST_LAYERS", "8"))
    replays = int(os.environ.get("QWEN36_WP_TEST_REPLAYS", "50"))
    layers = _mlp_layers(mesh, n_layers)
    x = _x(mesh, rows, 5120, seed=7)

    ref = _host(mesh, _mlp_step(x, layers, _plain))
    eager_pf = _host(mesh, _mlp_step(x, layers, wp.linear))
    _assert_bit_exact(ref, eager_pf, "eager MLP stack")

    tid_plain, out_plain = _capture(mesh, lambda: _mlp_step(x, layers, _plain))
    tid_pf, out_pf = _capture(mesh, lambda: _mlp_step(x, layers, wp.linear))
    for r in range(5):
        ttnn.execute_trace(mesh, tid_pf, cq_id=0, blocking=True)
        _assert_bit_exact(ref, _host(mesh, out_pf), f"traced MLP stack replay {r}")
    ttnn.execute_trace(mesh, tid_plain, cq_id=0, blocking=True)
    _assert_bit_exact(ref, _host(mesh, out_plain), "traced plain MLP stack")

    ms_plain = _time_replays(mesh, tid_plain, replays)
    ms_pf = _time_replays(mesh, tid_pf, replays)
    # Interleave once more so clock/thermal drift does not favour the second measurement.
    ms_plain = min(ms_plain, _time_replays(mesh, tid_plain, replays))
    logger.info(
        f"[weight_prefetch] MLP x{n_layers} rows={rows}: plain {ms_plain:.3f} ms, prefetch {ms_pf:.3f} ms per replay "
        f"({(ms_plain - ms_pf) / n_layers * 1e3:+.1f} us/layer, {100 * (ms_plain - ms_pf) / ms_plain:+.1f}%)"
    )
    # Ring state survives the timing burst: one more replay is still exact.
    ttnn.execute_trace(mesh, tid_pf, cq_id=0, blocking=True)
    _assert_bit_exact(ref, _host(mesh, out_pf), "traced MLP stack after timing")
    ttnn.release_trace(mesh, tid_plain)
    ttnn.release_trace(mesh, tid_pf)
    if os.environ.get("QWEN36_WP_ASSERT_PERF") == "1":
        assert ms_pf <= ms_plain * 1.01, f"prefetch slower: {ms_pf:.3f} vs {ms_plain:.3f} ms"
