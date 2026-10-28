# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.matmul_reduce_scatter_prefill on a Blackhole box.

Every device of a group along cluster_axis holds a K-slice A[g] (..., M, K) and W[g] (K, N); the output is the group
sum of A[g] @ W[g], scattered along scatter_dim. Shapes are per-device prefill shapes of the row-parallel linears of
DeepSeek-V3, Kimi K2.7 / K3, GLM 5.x, MiniMax-M3 and MiMo-V2 at TP=4, at each model's production precision.

The Ring cases need the whole box as one ring: a QuietBox 2 opened as 1x4 under FABRIC_2D_TORUS_XY closes its snake
cycle. The Linear cases run on the whole box as a 1x2 (p300) or 1x4 (QuietBox 2) line; other machines skip.
"""

import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0

# PCC targets by weight dtype (the sum of G partials of K-long dot products).
PCC = {ttnn.bfloat16: 0.999, ttnn.bfloat8_b: 0.998, ttnn.bfloat4_b: 0.99}


def router_config():
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024 + 64  # the production Blackhole payload
    return config


def line_device_params():
    # The whole box only: a fabric-enabled submesh smaller than the machine does not come up reliably.
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": router_config(),
        "require_exact_physical_num_devices": True,
    }


def ring_device_params():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": router_config(),
        "require_exact_physical_num_devices": True,
    }


def random_stacked(mesh_shape, shape, seed, scale=1.0):
    """Distinct data per device, as an (R, C, *shape) tensor: device (r, c) holds [r, c]."""
    torch.manual_seed(seed)
    return (torch.randn((*mesh_shape, *shape), dtype=torch.float32) * scale).to(torch.bfloat16)


def to_mesh(stacked, mesh_device, dtype, on_device=True):
    """on_device=False: the same multi-device tensor on the host (for copy_host_to_device_tensor)."""
    rows, cols = stacked.shape[0], stacked.shape[1]
    full = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    placement = dict(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) if on_device else {}
    return ttnn.from_torch(
        full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
        **placement,
    )


def quantize_like_device(w_stacked, weight_dtype):
    """The weight as the device holds it, so the reference measures the op, not the block-float rounding."""
    if weight_dtype == ttnn.bfloat16:
        return w_stacked
    t = ttnn.from_torch(w_stacked.reshape(-1, w_stacked.shape[-1]), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).reshape(w_stacked.shape).to(torch.bfloat16)


def reference(a, w, cluster_axis, scatter_dim):
    """Per device (r, c): its slice of the group sum, as an (R, C) grid of tensors."""
    rows, cols = a.shape[0], a.shape[1]
    group = rows if cluster_axis == 0 else cols
    out = [[None] * cols for _ in range(rows)]
    for line in range(cols if cluster_axis == 0 else rows):
        members = [(g, line) if cluster_axis == 0 else (line, g) for g in range(group)]
        total = sum(a[m].float() @ w[m].float() for m in members)
        for g, (r, c) in enumerate(members):
            out[r][c] = torch.chunk(total, group, dim=scatter_dim)[g]
    return out


def check(out, expected, mesh_device, weight_dtype):
    cols = tuple(mesh_device.shape)[1]
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        ref = expected[idx // cols][idx % cols]
        got = ttnn.to_torch(t)
        assert tuple(got.shape) == tuple(ref.shape), f"device {idx}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
        passed, msg = comp_pcc(ref, got.float(), PCC[weight_dtype])
        assert passed, f"device {idx}: {msg}"


def run_case(
    mesh_device,
    a_shape,
    w_shape,
    *,
    cluster_axis,
    scatter_dim,
    topology,
    dtype=ttnn.bfloat16,
    weight_dtype=ttnn.bfloat8_b,
    num_links=None,
    output_tensor=False,
    calls=1,
):
    mesh_shape = tuple(mesh_device.shape)
    a = random_stacked(mesh_shape, a_shape, seed=0)
    w = quantize_like_device(random_stacked(mesh_shape, w_shape, seed=1, scale=w_shape[0] ** -0.5), weight_dtype)
    ta = to_mesh(a, mesh_device, dtype)
    tw = to_mesh(w, mesh_device, weight_dtype)
    expected = reference(a, w, cluster_axis, scatter_dim)
    kwargs = dict(
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2),
    )
    preallocated = None
    if output_tensor:
        probe = ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **kwargs)
        preallocated = ttnn.zeros_like(probe)
        ttnn.deallocate(probe)
    out = None
    for _ in range(calls):
        if out is not None and preallocated is None:
            ttnn.deallocate(out)
        out = ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, output_tensor=preallocated, **kwargs)
    ttnn.synchronize_device(mesh_device)
    if preallocated is not None:
        assert out.buffer_address() == preallocated.buffer_address(), "the op did not write the preallocated output"
    check(out, expected, mesh_device, weight_dtype)


# (a_shape, w_shape, scatter_dim, dtype, weight_dtype); the op always accumulates in fp32 DEST
MODEL_CASES = {
    "kimi_k27_o_proj": ((1, 1, 640, 2048), (2048, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "minimax_m3_o_proj_bf8b_act": ((1, 1, 640, 2048), (2048, 6144), -1, ttnn.bfloat8_b, ttnn.bfloat8_b),
    "mimo_v2_o_proj": ((1, 1, 2048, 2048), (2048, 4096), -2, ttnn.bfloat16, ttnn.bfloat8_b),
    "kimi_k3_dense_ffn_stream": ((1, 1, 640, 8448), (8448, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "glm5_indexer_wk_narrow_n": ((1, 1, 640, 1536), (1536, 128), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    "kimi_k27_o_proj_rank2": ((640, 2048), (2048, 7168), -1, ttnn.bfloat16, ttnn.bfloat8_b),
    # output chunks (the block misses L1): 1 x 6 and 3 x 2 chunks on a 1x4 ring
    "chunked_n": ((1, 1, 2048, 1024), (1024, 16384), -1, ttnn.bfloat16, ttnn.bfloat16),
    "chunked_mn": ((1, 1, 4096, 1024), (1024, 8192), -1, ttnn.bfloat16, ttnn.bfloat16),
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("case", MODEL_CASES.values(), ids=MODEL_CASES.keys())
def test_ring_model_shapes(mesh_device, case):
    a_shape, w_shape, scatter_dim, dtype, weight_dtype = case
    run_case(
        mesh_device,
        a_shape,
        w_shape,
        cluster_axis=1,
        scatter_dim=scatter_dim,
        topology=ttnn.Topology.Ring,
        dtype=dtype,
        weight_dtype=weight_dtype,
        num_links=2,
    )


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [line_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], ids=["1x2", "1x4"], indirect=True)
@pytest.mark.parametrize("scatter_dim", [-1, -2])
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
def test_line(mesh_device, scatter_dim, num_links):
    run_case(
        mesh_device,
        (1, 1, 640, 2048),
        (2048, 7168),
        cluster_axis=1,
        scatter_dim=scatter_dim,
        topology=ttnn.Topology.Linear,
        num_links=num_links,
    )


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16, ttnn.bfloat4_b], ids=["bf16_w", "bf4b_w"])
def test_weight_dtypes(mesh_device, weight_dtype):
    run_case(
        mesh_device,
        (1, 1, 640, 1536),
        (1536, 7168),
        cluster_axis=1,
        scatter_dim=-1,
        topology=ttnn.Topology.Ring,
        weight_dtype=weight_dtype,
    )


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_output_tensor(mesh_device):
    run_case(
        mesh_device,
        (1, 1, 640, 2048),
        (2048, 7168),
        cluster_axis=1,
        scatter_dim=-1,
        topology=ttnn.Topology.Ring,
        output_tensor=True,
    )


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_program_cache(mesh_device):
    """Calls with new tensors hit one cached program and stay correct (runtime args re-pointed on a hit)."""
    kwargs = dict(cluster_axis=1, scatter_dim=-1, topology=ttnn.Topology.Ring)
    run_case(mesh_device, (1, 1, 640, 1536), (1536, 7168), **kwargs)
    entries = mesh_device.num_program_cache_entries()
    run_case(mesh_device, (1, 1, 640, 1536), (1536, 7168), calls=2, **kwargs)
    assert mesh_device.num_program_cache_entries() == entries, "a repeated call built a new program"


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(scatter_dim=0), "scatter_dim"),
        (dict(cluster_axis=0), "need >= 2"),
        (dict(num_links=3), "num_links"),
        (dict(weight_dtype=ttnn.float32), "weight_dtype"),
        (dict(w_shape=(1024, 7168)), "K mismatch"),
        (dict(memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG), "sharded"),
    ],
    ids=["scatter_dim", "single_device_axis", "num_links", "fp32_weight", "k_mismatch", "sharded_output"],
)
def test_refusals(mesh_device, expect_error, overrides, message):
    shapes = {"a_shape": (1, 1, 640, 2048), "w_shape": (2048, 7168)}
    shapes.update({k: overrides.pop(k) for k in list(overrides) if k in shapes})
    weight_dtype = overrides.pop("weight_dtype", ttnn.bfloat8_b)
    mesh_shape = tuple(mesh_device.shape)
    ta = to_mesh(random_stacked(mesh_shape, shapes["a_shape"], seed=0), mesh_device, ttnn.bfloat16)
    tw = to_mesh(random_stacked(mesh_shape, shapes["w_shape"], seed=1), mesh_device, weight_dtype)
    kwargs = dict(cluster_axis=1, scatter_dim=-1, topology=ttnn.Topology.Ring, num_links=2)
    kwargs.update(overrides)
    with expect_error(RuntimeError, message):
        ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **kwargs)


# Traced calls. Each test compiles eagerly, captures one trace and replays it; every output is checked against the
# host reference after the replays.
TRACE_REGION_BYTES = 1 << 20
TRACE_KWARGS = dict(
    cluster_axis=1,
    scatter_dim=-1,
    topology=ttnn.Topology.Ring,
    num_links=2,
    compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2),
)
# (a_shape, w_shape, weight_dtype): an unchunked plan and a chunked one, which need different staging sizes
TRACE_SHAPES = {
    "kimi_k27_o_proj": ((1, 1, 640, 2048), (2048, 7168), ttnn.bfloat8_b),
    "chunked_n": ((1, 1, 2048, 1024), (1024, 16384), ttnn.bfloat16),
}


def traced_device_params():
    return {**ring_device_params(), "trace_region_size": TRACE_REGION_BYTES}


def host_operands(mesh_shape, name, seed):
    a_shape, w_shape, weight_dtype = TRACE_SHAPES[name]
    a = random_stacked(mesh_shape, a_shape, seed=seed)
    w = quantize_like_device(random_stacked(mesh_shape, w_shape, seed=seed + 1, scale=w_shape[0] ** -0.5), weight_dtype)
    return a, w, reference(a, w, TRACE_KWARGS["cluster_axis"], TRACE_KWARGS["scatter_dim"])


def capture(mesh_device, calls):
    """Capture calls (no-argument callables) into one trace; returns (trace_id, outputs)."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [call() for call in calls]
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    return trace_id, outs


def replay(mesh_device, trace_id, times=3):
    for _ in range(times):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [traced_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "order", [("kimi_k27_o_proj", "chunked_n"), ("chunked_n", "kimi_k27_o_proj")], ids=["small_first", "large_first"]
)
def test_trace_two_shapes(mesh_device, order):
    """Two programs of the op with different plans and staging sizes in one trace, either order."""
    mesh_shape = tuple(mesh_device.shape)
    cases = []
    for i, name in enumerate(order):
        a, w, expected = host_operands(mesh_shape, name, seed=10 * i)
        ta = to_mesh(a, mesh_device, ttnn.bfloat16)
        tw = to_mesh(w, mesh_device, TRACE_SHAPES[name][2])
        cases.append((ta, tw, expected, TRACE_SHAPES[name][2]))
    for ta, tw, _, _ in cases:  # compile both
        ttnn.deallocate(ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **TRACE_KWARGS))
    trace_id, outs = capture(
        mesh_device,
        [
            lambda ta=ta, tw=tw: ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **TRACE_KWARGS)
            for ta, tw, _, _ in cases
        ],
    )
    try:
        replay(mesh_device, trace_id)
        for out, (_, _, expected, weight_dtype) in zip(outs, cases):
            check(out, expected, mesh_device, weight_dtype)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [traced_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_trace_after_warmup(mesh_device):
    """A trace captured after eager calls of the same shape: the captured call is a program-cache hit."""
    mesh_shape = tuple(mesh_device.shape)
    a, w, expected = host_operands(mesh_shape, "kimi_k27_o_proj", seed=0)
    weight_dtype = TRACE_SHAPES["kimi_k27_o_proj"][2]
    ta = to_mesh(a, mesh_device, ttnn.bfloat16)
    tw = to_mesh(w, mesh_device, weight_dtype)
    for _ in range(3):
        out = ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **TRACE_KWARGS)
    ttnn.synchronize_device(mesh_device)
    check(out, expected, mesh_device, weight_dtype)
    ttnn.deallocate(out)
    entries = mesh_device.num_program_cache_entries()
    trace_id, (traced,) = capture(
        mesh_device, [lambda: ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **TRACE_KWARGS)]
    )
    try:
        assert mesh_device.num_program_cache_entries() == entries, "the captured call built a new program"
        replay(mesh_device, trace_id)
        check(traced, expected, mesh_device, weight_dtype)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [traced_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_trace_new_inputs(mesh_device):
    """New A and W copied into the captured input tensors before each replay, as a model feeds a trace."""
    mesh_shape = tuple(mesh_device.shape)
    weight_dtype = TRACE_SHAPES["kimi_k27_o_proj"][2]
    a, w, _ = host_operands(mesh_shape, "kimi_k27_o_proj", seed=0)
    ta = to_mesh(a, mesh_device, ttnn.bfloat16)
    tw = to_mesh(w, mesh_device, weight_dtype)
    ttnn.deallocate(ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **TRACE_KWARGS))
    trace_id, (traced,) = capture(
        mesh_device, [lambda: ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **TRACE_KWARGS)]
    )
    try:
        for seed in (20, 30, 40):
            a, w, expected = host_operands(mesh_shape, "kimi_k27_o_proj", seed=seed)
            ttnn.copy_host_to_device_tensor(to_mesh(a, mesh_device, ttnn.bfloat16, on_device=False), ta)
            ttnn.copy_host_to_device_tensor(to_mesh(w, mesh_device, weight_dtype, on_device=False), tw)
            replay(mesh_device, trace_id, times=1)
            check(traced, expected, mesh_device, weight_dtype)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
