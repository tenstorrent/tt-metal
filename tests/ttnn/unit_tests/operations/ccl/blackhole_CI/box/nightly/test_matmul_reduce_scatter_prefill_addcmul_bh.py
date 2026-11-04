# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.matmul_reduce_scatter_prefill with the fused bias and addcmul epilogue, on a Blackhole box:

    out = addcmul_input_tensor1 + fused_ternary_scalar * (group sum of (A[g] @ W[g] + bias[g]), scattered)
          * addcmul_input_tensor2

(the operation of minimal_matmul_strided_reduce_scatter_async; the bias is per device, [.., 1, N], and summed by the
reduce-scatter like a matmul bias before it).

tensor1 (the residual) has the output's shape; tensor2 (the gate) has the output's shape or is one broadcast row
[.., 1, N_out] (the tt_dit adaLN gate). The reference runs in fp32 from the operands as the device holds them. Besides
the PCC of the whole output, the epilogue's own part (out - tensor1) is checked against scalar * sum * tensor2, so a
residual-dominated output cannot hide a wrong product.
"""

import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_matmul_reduce_scatter_prefill_bh import (
    dram_width_sharded,
    line_device_params,
    quantize_like_device,
    random_stacked,
    reference,
    ring_device_params,
    to_mesh,
)

HIFI2 = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2)
# PCC targets of the output by weight dtype (as the op's own tests); the product alone gets 0.001 less
PCC = {ttnn.bfloat16: 0.999, ttnn.bfloat8_b: 0.998}
# (a_shape, w_shape, scatter_dim, weight_dtype)
SHAPES = {
    "kimi_k27_o_proj": ((1, 1, 640, 2048), (2048, 7168), -1, ttnn.bfloat8_b),
    "mimo_v2_o_proj_scatter_m": ((1, 1, 2048, 2048), (2048, 4096), -2, ttnn.bfloat8_b),
    # output chunks: 1 x 6 and 3 x 2 chunks on a 1x4 ring (chunk edges, several fin entries per core)
    "chunked_n": ((1, 1, 2048, 1024), (1024, 16384), -1, ttnn.bfloat16),
    "chunked_mn": ((1, 1, 4096, 1024), (1024, 8192), -1, ttnn.bfloat16),
    # MiniMax-H3 ff2 at its smallest 4x32 bucket (K = 14336 / 4, N 5376)
    "minimax_h3_ff2_m704": ((1, 1, 704, 3584), (3584, 5376), -1, ttnn.bfloat16),
    # decode (one tile row) and the shapes the folded grids take (test_forced_grid)
    "kimi_k27_o_proj_decode": ((1, 1, 32, 2048), (2048, 7168), -1, ttnn.bfloat8_b),
    "tall_narrow_n": ((1, 1, 1792, 2048), (2048, 128), -1, ttnn.bfloat8_b),
    "glm5_indexer_wk_narrow_n": ((1, 1, 640, 1536), (1536, 128), -1, ttnn.bfloat8_b),
}


def quantize(t, dtype):
    """t as the device holds it in dtype (block-float rounding included)."""
    if dtype in (ttnn.bfloat16, ttnn.float32):
        return t.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    flat = ttnn.from_torch(t.reshape(-1, t.shape[-1]).float(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(flat).reshape(t.shape).float()


def output_grid(mesh_shape, ref):
    """Per-device output shapes of the op, from the reference grid."""
    return [[tuple(ref[r][c].shape) for c in range(mesh_shape[1])] for r in range(mesh_shape[0])]


def make_addcmul(mesh_device, ref, gate_rows, a_dtype, b_dtype, seed, on_device=True):
    """Per-device residual (output shape) and gate (output shape or one row), stacked; their (device) tensors."""
    mesh_shape = tuple(mesh_device.shape)
    shape = output_grid(mesh_shape, ref)[0][0]
    torch.manual_seed(seed)
    res = quantize(torch.randn((*mesh_shape, *shape)) * 0.25, a_dtype)
    b_shape = (*shape[:-2], 1, shape[-1]) if gate_rows == 1 else shape
    gate = quantize(1.0 + 0.5 * torch.randn((*mesh_shape, *b_shape)), b_dtype)
    t_res = to_mesh(res, mesh_device, a_dtype, on_device=on_device)
    t_gate = to_mesh(gate, mesh_device, b_dtype, on_device=on_device)
    return res, gate, t_res, t_gate


def biased_reference(ref, bias, scatter_dim, group):
    """ref (the group sums per device) plus the group's summed bias, at each device's scattered slice."""
    if bias is None:
        return ref
    total = bias.float().sum(dim=(0, 1))  # [*, 1, N]: every device's bias, summed (one ring of the 1-row mesh)
    out = [[None] * len(ref[0]) for _ in ref]
    for r in range(len(ref)):
        for c in range(len(ref[0])):
            piece = total if scatter_dim == -2 else torch.chunk(total, group, dim=-1)[c]
            out[r][c] = ref[r][c].float() + piece
    return out


def check_addcmul(out, ref, res, gate, scalar, mesh_device, weight_dtype):
    pcc = PCC[weight_dtype]
    cols = tuple(mesh_device.shape)[1]
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = idx // cols, idx % cols
        mm = ref[r][c].float()
        if res is None:  # bias only
            got = ttnn.to_torch(t).float()
            passed, msg = comp_pcc(mm, got, pcc)
            assert passed, f"device {idx}: output {msg}"
            continue
        expected = res[r, c].float() + scalar * mm * gate[r, c].float()
        got = ttnn.to_torch(t).float()
        assert tuple(got.shape) == tuple(expected.shape), f"device {idx}: shape {tuple(got.shape)} vs {expected.shape}"
        passed, msg = comp_pcc(expected, got, pcc)
        assert passed, f"device {idx}: output {msg}"
        # the epilogue's product alone: a product that is off cannot hide behind the residual
        passed, msg = comp_pcc(scalar * mm * gate[r, c].float(), got - res[r, c].float(), pcc - 0.001)
        assert passed, f"device {idx}: product {msg}"


def run_addcmul(
    mesh_device,
    name,
    *,
    topology=ttnn.Topology.Ring,
    gate_rows="full",
    scalar=1.0,
    a_dtype=ttnn.bfloat16,
    b_dtype=ttnn.bfloat16,
    num_links=None,
    calls=1,
    use_staging=False,
    seed=0,
    bias=None,
    grid=ttnn.MatmulReduceScatterPrefillGrid.GRID_2D,
    w_dram_sharded=False,
):
    """gate_rows: "full", "bcast" or None (no addcmul). bias: None, "dev0" (device 0 only, the tt_dit row-parallel
    bias) or "all" (a different bias on every device)."""
    a_shape, w_shape, scatter_dim, weight_dtype = SHAPES[name]
    mesh_shape = tuple(mesh_device.shape)
    a = random_stacked(mesh_shape, a_shape, seed=seed)
    w = quantize_like_device(random_stacked(mesh_shape, w_shape, seed=seed + 1, scale=w_shape[0] ** -0.5), weight_dtype)
    ta = to_mesh(a, mesh_device, ttnn.bfloat16)
    w_mem = dram_width_sharded(mesh_device, *w_shape) if w_dram_sharded else ttnn.DRAM_MEMORY_CONFIG
    tw = to_mesh(w, mesh_device, weight_dtype, memory_config=w_mem)
    ref = reference(a, w, 1, scatter_dim)
    t_bias = bias_t = None
    if bias is not None:
        torch.manual_seed(seed + 3)
        bias_t = (0.5 * torch.randn((*mesh_shape, 1, w_shape[-1]))).to(torch.bfloat16)
        if bias == "dev0":
            bias_t[:, 1:] = 0
        t_bias = to_mesh(bias_t, mesh_device, ttnn.bfloat16)
    ref = biased_reference(ref, bias_t, scatter_dim, mesh_shape[1])
    res = gate = t_res = t_gate = None
    if gate_rows is not None:
        res, gate, t_res, t_gate = make_addcmul(
            mesh_device, ref, 1 if gate_rows == "bcast" else None, a_dtype, b_dtype, seed + 2
        )
    kwargs = dict(
        cluster_axis=1, scatter_dim=scatter_dim, topology=topology, num_links=num_links, bias=t_bias, grid=grid
    )
    staging = None
    if use_staging:
        staging = ttnn.experimental.matmul_reduce_scatter_prefill_staging(
            ta, tw, addcmul_input_tensor1=t_res, addcmul_input_tensor2=t_gate, **kwargs
        )
    outs = []
    for _ in range(calls):
        outs.append(
            ttnn.experimental.matmul_reduce_scatter_prefill(
                ta,
                tw,
                compute_kernel_config=HIFI2,
                staging_tensor=staging,
                fused_ternary_scalar=scalar if gate_rows is not None else None,
                addcmul_input_tensor1=t_res,
                addcmul_input_tensor2=t_gate,
                **kwargs,
            )
        )
    ttnn.synchronize_device(mesh_device)
    check_addcmul(outs[-1], ref, res, gate, scalar, mesh_device, weight_dtype)
    return outs


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("gate_rows", ["full", "bcast"])
def test_ring_shapes(mesh_device, name, gate_rows):
    run_addcmul(mesh_device, name, gate_rows=gate_rows)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("name", ["kimi_k27_o_proj", "mimo_v2_o_proj_scatter_m", "chunked_mn", "minimax_h3_ff2_m704"])
@pytest.mark.parametrize("gate_rows", [None, "full", "bcast"], ids=["no_addcmul", "full", "bcast"])
@pytest.mark.parametrize("bias", ["dev0", "all"])
def test_bias(mesh_device, name, gate_rows, bias):
    run_addcmul(mesh_device, name, gate_rows=gate_rows, bias=bias)


# The caller's grid: the folds (one line of cores folded into the rectangle; a fold along M reads A alternately on both
# NoCs, so the epilogue's reads take NoC1 there), and a DRAM width-sharded W (decode weights), with the epilogue.
GRID = ttnn.MatmulReduceScatterPrefillGrid


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "name, grid",
    [
        ("kimi_k27_o_proj_decode", GRID.FOLDED_N),
        ("kimi_k27_o_proj", GRID.FOLDED_N),
        ("chunked_n", GRID.FOLDED_N),
        ("tall_narrow_n", GRID.FOLDED_M),
        ("glm5_indexer_wk_narrow_n", GRID.FOLDED_M),
    ],
    ids=["decode-folded_n", "kimi_k27_o_proj-folded_n", "chunked_n-folded_n", "tall-folded_m", "glm5_wk-folded_m"],
)
@pytest.mark.parametrize("gate_rows", ["full", "bcast"])
@pytest.mark.parametrize("bias", [None, "dev0"], ids=["no_bias", "bias"])
def test_forced_grid(mesh_device, name, grid, gate_rows, bias):
    run_addcmul(mesh_device, name, gate_rows=gate_rows, bias=bias, grid=grid)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "name, grid",
    [("kimi_k27_o_proj_decode", GRID.FOLDED_N), ("kimi_k27_o_proj", GRID.GRID_2D), ("chunked_n", GRID.GRID_2D)],
    ids=["decode-folded_n", "kimi_k27_o_proj", "chunked_n"],
)
@pytest.mark.parametrize("gate_rows", ["full", "bcast"])
def test_dram_sharded_weight(mesh_device, name, grid, gate_rows):
    run_addcmul(mesh_device, name, gate_rows=gate_rows, bias="dev0", grid=grid, w_dram_sharded=True, calls=2)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("scalar", [0.5, -1.25])
@pytest.mark.parametrize("gate_rows", ["full", "bcast"])
def test_scalar(mesh_device, scalar, gate_rows):
    run_addcmul(mesh_device, "kimi_k27_o_proj", gate_rows=gate_rows, scalar=scalar)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "a_dtype, b_dtype, gate_rows",
    [
        (ttnn.bfloat8_b, ttnn.bfloat8_b, "full"),
        (ttnn.float32, ttnn.float32, "full"),
        (ttnn.float32, ttnn.float32, "bcast"),
        (ttnn.bfloat8_b, ttnn.bfloat16, "bcast"),
    ],
    ids=["bf8b_full", "fp32_full", "fp32_bcast", "bf8b_a_bf16_bcast"],
)
def test_dtypes(mesh_device, a_dtype, b_dtype, gate_rows):
    run_addcmul(mesh_device, "kimi_k27_o_proj", gate_rows=gate_rows, a_dtype=a_dtype, b_dtype=b_dtype)


@pytest.mark.parametrize("device_params", [line_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], ids=["1x2", "1x4"], indirect=True)
@pytest.mark.parametrize("name", ["kimi_k27_o_proj", "mimo_v2_o_proj_scatter_m"])
@pytest.mark.parametrize("gate_rows", ["full", "bcast"])
def test_line(mesh_device, name, gate_rows):
    run_addcmul(mesh_device, name, topology=ttnn.Topology.Linear, gate_rows=gate_rows)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("name", ["kimi_k27_o_proj", "chunked_mn"])
def test_deterministic(mesh_device, name):
    """Repeated calls are bitwise identical: a fin entry finished late (separate finish) or fused gives the same bits."""
    outs = run_addcmul(mesh_device, name, gate_rows="full", calls=4)
    first = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(outs[0])]
    for out in outs[1:]:
        for idx, t in enumerate(ttnn.get_device_tensors(out)):
            assert torch.equal(ttnn.to_torch(t), first[idx]), f"device {idx}: a repeated call differs"


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_program_cache_and_staging(mesh_device):
    """New addcmul tensors on a cache hit are re-pointed; a caller-held staging sized with the epilogue serves it."""
    run_addcmul(mesh_device, "kimi_k27_o_proj", seed=0)
    entries = mesh_device.num_program_cache_entries()
    run_addcmul(mesh_device, "kimi_k27_o_proj", seed=5, calls=2)
    assert mesh_device.num_program_cache_entries() == entries, "a repeated call built a new program"
    run_addcmul(mesh_device, "chunked_n", gate_rows="bcast", use_staging=True, calls=2)


TRACE_REGION_BYTES = 1 << 20


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{**ring_device_params(), "trace_region_size": TRACE_REGION_BYTES}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("gate_rows", ["full", "bcast"])
def test_trace(mesh_device, gate_rows):
    """A traced call replays with the residual and gate rewritten in place between replays."""
    a_shape, w_shape, scatter_dim, weight_dtype = SHAPES["kimi_k27_o_proj"]
    mesh_shape = tuple(mesh_device.shape)
    a = random_stacked(mesh_shape, a_shape, seed=0)
    w = quantize_like_device(random_stacked(mesh_shape, w_shape, seed=1, scale=w_shape[0] ** -0.5), weight_dtype)
    ta = to_mesh(a, mesh_device, ttnn.bfloat16)
    tw = to_mesh(w, mesh_device, weight_dtype)
    ref = reference(a, w, 1, scatter_dim)
    rows = 1 if gate_rows == "bcast" else None
    res, gate, t_res, t_gate = make_addcmul(mesh_device, ref, rows, ttnn.bfloat16, ttnn.bfloat16, seed=7)
    kwargs = dict(
        cluster_axis=1,
        scatter_dim=scatter_dim,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=HIFI2,
        fused_ternary_scalar=1.0,
        addcmul_input_tensor1=t_res,
        addcmul_input_tensor2=t_gate,
    )
    ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **kwargs)  # compile, mesh resources
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = ttnn.experimental.matmul_reduce_scatter_prefill(ta, tw, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for step in range(2):
        res2, gate2, h_res, h_gate = make_addcmul(
            mesh_device, ref, rows, ttnn.bfloat16, ttnn.bfloat16, seed=20 + step, on_device=False
        )
        ttnn.copy_host_to_device_tensor(h_res, t_res)
        ttnn.copy_host_to_device_tensor(h_gate, t_gate)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        check_addcmul(out, ref, res2, gate2, 1.0, mesh_device, weight_dtype)
    ttnn.release_trace(mesh_device, trace_id)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "kind, message",
    [
        ("only_one", "given together"),
        ("a_shape", "addcmul_input_tensor1 must match"),
        ("b_shape", "addcmul_input_tensor2 must be"),
        ("bf8b_bcast", "broadcast row"),
        ("bf4b", "addcmul_input_tensor1 dtype"),
    ],
)
def test_refusals(mesh_device, expect_error, kind, message):
    mesh_shape = tuple(mesh_device.shape)
    ta = to_mesh(random_stacked(mesh_shape, (1, 1, 640, 2048), seed=0), mesh_device, ttnn.bfloat16)
    tw = to_mesh(random_stacked(mesh_shape, (2048, 7168), seed=1), mesh_device, ttnn.bfloat8_b)
    out_shape = (1, 1, 640, 7168 // 4)

    def dev(shape, dtype=ttnn.bfloat16):
        return to_mesh(random_stacked(mesh_shape, shape, seed=3), mesh_device, dtype)

    t1, t2 = dev(out_shape), dev(out_shape)
    if kind == "only_one":
        t2 = None
    elif kind == "a_shape":
        t1 = dev((1, 1, 640, 7168))
    elif kind == "b_shape":
        t2 = dev((1, 1, 64, 7168 // 4))
    elif kind == "bf8b_bcast":
        t2 = dev((1, 1, 1, 7168 // 4), ttnn.bfloat8_b)
    elif kind == "bf4b":
        t1 = dev(out_shape, ttnn.bfloat4_b)
    with expect_error(RuntimeError, message):
        ttnn.experimental.matmul_reduce_scatter_prefill(
            ta,
            tw,
            cluster_axis=1,
            scatter_dim=-1,
            topology=ttnn.Topology.Ring,
            addcmul_input_tensor1=t1,
            addcmul_input_tensor2=t2,
        )
