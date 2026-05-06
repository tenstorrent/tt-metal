# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Eltwise v2 chain helper validation suite.

Test plan: ttnn/cpp/ttnn/kernel_lib/agents/eltwise_v2_test_plan.md
Helper headers: ttnn/cpp/ttnn/kernel_lib/eltwise_chain.{hpp,inl} + eltwise_*.hpp

Run with:
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/kernel_lib/test_eltwise.py

Vertical slice (test 14.1) exercises the foundation: CopyTile + Exp + PackTile.
"""
import math

import pytest
import torch
from loguru import logger

import ttnn

READER_KERNEL = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WRITER_KERNEL = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
READER_BINARY_KERNEL = (
    "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp"
)


def _split_work(num_tiles):
    max_core = ttnn.CoreCoord(7, 7)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    (_, core_grid, group_1, group_2, work_per_core1, _) = ttnn.split_work_to_cores(all_cores, num_tiles)
    assert len(group_2.ranges()) == 0, "test only supports single split-group"
    return core_grid, group_1, work_per_core1


def _make_cb(buffer_index, core_grid, page_bytes=2 * 1024, total_bytes=2 * 2 * 1024, dtype=ttnn.bfloat16, n_pages=None):
    if n_pages is not None:
        total_bytes = n_pages * page_bytes
    fmt = ttnn.CBFormatDescriptor(buffer_index=buffer_index, data_format=dtype, page_size=page_bytes)
    return ttnn.CBDescriptor(total_size=total_bytes, core_ranges=core_grid, format_descriptors=[fmt])


def _build_compute_kd(core_grid, kernel_path, work_per_core, defines, fp32_dest_acc=False):
    defines_seq = list(defines.items()) if isinstance(defines, dict) else list(defines)
    return ttnn.KernelDescriptor(
        kernel_source=kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[work_per_core, 1],
        defines=defines_seq,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc),
    )


def _run_unary_chain(
    device,
    num_tiles,
    kernel_path,
    torch_op,
    input_range,
    defines=None,
    fp32_dest_acc=False,
    cb_pages=None,
    single_core=False,
):
    """Run a unary compute kernel via generic_op against a torch golden. Returns (output, golden)."""
    defines = defines or {}
    shape = [1, num_tiles, 32, 32]
    lo, hi = input_range
    data = (torch.rand(shape) * (hi - lo) + lo).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG

    in_t = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    out_t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    if single_core:
        # Pin everything to (0,0) so all num_tiles land on one core (needed for upfront-block lifecycles).
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        group_1 = core_grid
        work_per_core = num_tiles
    else:
        core_grid, group_1, work_per_core = _split_work(num_tiles)
    cb_in = _make_cb(0, core_grid, n_pages=cb_pages)
    cb_out = _make_cb(16, core_grid, n_pages=cb_pages)

    reader_args = ttnn.TensorAccessorArgs(in_t).get_compile_time_args()
    writer_args = [16] + ttnn.TensorAccessorArgs(out_t).get_compile_time_args()

    rd_rt = ttnn.RuntimeArgs()
    wr_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                rd_rt[x][y] = [in_t.buffer_address(), work_per_core, cur]
                wr_rt[x][y] = [out_t.buffer_address(), work_per_core, cur]
                cur += work_per_core

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_args,
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_args,
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kd = _build_compute_kd(core_grid, kernel_path, work_per_core, defines, fp32_dest_acc)
    pd = ttnn.ProgramDescriptor(kernels=[reader_kd, writer_kd, compute_kd], semaphores=[], cbs=[cb_in, cb_out])

    out_tensor = ttnn.generic_op([in_t, out_t], pd)
    return ttnn.to_torch(out_tensor).to(torch.float32), torch_op(data.to(torch.float32))


def _check_pcc(out, golden, pcc_threshold, label):
    # Cosine similarity is undefined for zero vectors. Fall back to exact match in that case.
    g_norm = golden.flatten().norm().item()
    o_norm = out.flatten().norm().item()
    if g_norm == 0.0 or o_norm == 0.0:
        match = bool(torch.allclose(out, golden, rtol=0, atol=0))
        logger.info(f"{label} (sparse, exact-equal={match})")
        assert match, f"{label} sparse golden but output mismatch"
        return
    pcc = torch.nn.functional.cosine_similarity(out.flatten(), golden.flatten(), dim=0).item()
    logger.info(f"{label} PCC={pcc:.6f}")
    assert pcc >= pcc_threshold, f"{label} PCC={pcc} < {pcc_threshold}"


# =============================================================================
# Test plan §14.1 — CopyTile + Exp + PackTile (vertical slice)
# =============================================================================


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_14_1_copy_exp_pack(device, num_tiles, fp32_dest_acc):
    out, golden = _run_unary_chain(
        device,
        num_tiles,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/copy_exp_pack.cpp",
        torch_op=torch.exp,
        input_range=(-3.0, 3.0),
        fp32_dest_acc=fp32_dest_acc,
    )
    _check_pcc(out, golden, 0.9999, f"copy_exp_pack n={num_tiles} fp32_acc={fp32_dest_acc}")


# =============================================================================
# Test plan §3 — SFPU unary by family
# Drives the templated kernel `copy_sfpu_pack.cpp` with ELTWISE_OP_NAME defines.
# =============================================================================

SFPU_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/copy_sfpu_pack.cpp"


def _run_unary_op(device, num_tiles, op_name, torch_op, input_range, pcc, fp32_dest_acc=False, op_braced=None):
    defines = {"ELTWISE_OP_NAME": op_name}
    if op_braced is not None:
        defines["ELTWISE_OP_BRACED"] = op_braced
    out, golden = _run_unary_chain(
        device, num_tiles, SFPU_KERNEL, torch_op, input_range, defines=defines, fp32_dest_acc=fp32_dest_acc
    )
    _check_pcc(out, golden, pcc, f"{op_name} n={num_tiles} fp32_acc={fp32_dest_acc}")


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize(
    "op_name,torch_op,input_range,pcc",
    [
        # math
        ("Exp", torch.exp, (-3.0, 3.0), 0.9999),
        ("Sqrt", lambda x: torch.sqrt(x.abs() + 0.01), (0.01, 5.0), 0.9999),
        ("Recip", lambda x: 1.0 / x, (0.5, 3.0), 0.999),
        # activations
        ("Relu", torch.nn.functional.relu, (-3.0, 3.0), 0.9999),
        ("Sigmoid", torch.sigmoid, (-3.0, 3.0), 0.999),
        ("Tanh", torch.tanh, (-3.0, 3.0), 0.9999),
        ("Gelu", lambda x: torch.nn.functional.gelu(x, approximate="tanh"), (-3.0, 3.0), 0.99),
        # trig
        ("Sin", torch.sin, (-math.pi, math.pi), 0.999),
        ("Cos", torch.cos, (-math.pi, math.pi), 0.999),
        # rounding
        ("Floor", torch.floor, (-5.0, 5.0), 0.9999),
        ("Ceil", torch.ceil, (-5.0, 5.0), 0.9999),
        # predicates
        ("Eqz", lambda x: (x == 0).to(x.dtype), (-3.0, 3.0), 0.9999),
        ("Gtz", lambda x: (x > 0).to(x.dtype), (-3.0, 3.0), 0.9999),
        # misc
        ("Identity", lambda x: x, (-3.0, 3.0), 0.9999),
        ("Negative", torch.neg, (-3.0, 3.0), 0.9999),
        ("Abs", torch.abs, (-3.0, 3.0), 0.9999),
        ("Square", lambda x: x * x, (-3.0, 3.0), 0.999),
        # special
        ("Erf", torch.erf, (-3.0, 3.0), 0.99),
    ],
)
def test_3_sfpu_unary_family(device, num_tiles, fp32_dest_acc, op_name, torch_op, input_range, pcc):
    _run_unary_op(device, num_tiles, op_name, torch_op, input_range, pcc, fp32_dest_acc)


# Expanded SFPU coverage — additional ops that fit the simple `Op<>{}` template.
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize(
    "op_name,torch_op,input_range,pcc",
    [
        # math
        ("Rsqrt", lambda x: 1.0 / torch.sqrt(x), (0.5, 5.0), 0.99),
        ("Cbrt", lambda x: torch.pow(x, 1 / 3.0), (0.01, 5.0), 0.99),
        ("Log1p", torch.log1p, (0.0, 5.0), 0.999),
        # activations
        ("Hardsigmoid", torch.nn.functional.hardsigmoid, (-3.0, 3.0), 0.99),
        ("Softsign", torch.nn.functional.softsign, (-3.0, 3.0), 0.99),
        ("Hardmish", lambda x: x * torch.clamp((x + 3.0) / 6.0, min=0.0, max=1.0), (-3.0, 3.0), 0.95),
        # trig
        ("Tan", torch.tan, (-1.0, 1.0), 0.99),
        ("Asin", torch.asin, (-0.9, 0.9), 0.99),
        ("Acos", torch.acos, (-0.9, 0.9), 0.99),
        ("Atan", torch.atan, (-3.0, 3.0), 0.99),
        ("Sinh", torch.sinh, (-2.0, 2.0), 0.99),
        ("Cosh", torch.cosh, (-2.0, 2.0), 0.99),
        ("Asinh", torch.asinh, (-3.0, 3.0), 0.99),
        ("Acosh", torch.acosh, (1.5, 5.0), 0.99),
        ("Atanh", torch.atanh, (-0.9, 0.9), 0.99),
        # predicates
        ("Nez", lambda x: (x != 0).to(x.dtype), (-3.0, 3.0), 0.9999),
        ("Ltz", lambda x: (x < 0).to(x.dtype), (-3.0, 3.0), 0.9999),
        ("Gez", lambda x: (x >= 0).to(x.dtype), (-3.0, 3.0), 0.9999),
        ("Lez", lambda x: (x <= 0).to(x.dtype), (-3.0, 3.0), 0.9999),
        ("Isnan", lambda x: torch.isnan(x).to(x.dtype), (-3.0, 3.0), 0.9999),
        ("Isfinite", lambda x: torch.isfinite(x).to(x.dtype), (-3.0, 3.0), 0.9999),
        # rounding
        ("Trunc", torch.trunc, (-5.0, 5.0), 0.9999),
        ("Frac", lambda x: x - torch.trunc(x), (-5.0, 5.0), 0.99),
        # special
        ("I0", torch.special.i0, (-2.0, 2.0), 0.99),
        ("I1", torch.special.i1, (-2.0, 2.0), 0.99),
        ("Erfc", torch.erfc, (-2.0, 2.0), 0.99),
        ("Erfinv", torch.special.erfinv, (-0.9, 0.9), 0.99),
        # misc
        ("Sign", torch.sign, (-3.0, 3.0), 0.9999),
    ],
)
def test_3_sfpu_unary_family_expanded(device, num_tiles, fp32_dest_acc, op_name, torch_op, input_range, pcc):
    _run_unary_op(device, num_tiles, op_name, torch_op, input_range, pcc, fp32_dest_acc)


# =============================================================================
# Test plan §5 — BinaryFpu single-op matrix (streaming Add/Sub/Mul)
# =============================================================================

BINARY_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/binary_fpu.cpp"


def _run_binary_with_kernel(
    device, num_tiles, kernel_path, defines, torch_op, range_a, range_b, pcc, fp32_dest_acc=False, label=""
):
    shape = [1, num_tiles, 32, 32]
    la, ha = range_a
    lb, hb = range_b
    data_a = (torch.rand(shape) * (ha - la) + la).to(torch.bfloat16)
    data_b = (torch.rand(shape) * (hb - lb) + lb).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG

    a = ttnn.from_torch(data_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    b = ttnn.from_torch(data_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_grid, group_1, work_per_core = _split_work(num_tiles)
    cb_a = _make_cb(0, core_grid)
    cb_b = _make_cb(1, core_grid)
    cb_out = _make_cb(16, core_grid)

    reader_compile_args = (
        [0] + ttnn.TensorAccessorArgs(a).get_compile_time_args() + ttnn.TensorAccessorArgs(b).get_compile_time_args()
    )
    writer_compile_args = [16] + ttnn.TensorAccessorArgs(out).get_compile_time_args()

    rd_rt = ttnn.RuntimeArgs()
    wr_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                rd_rt[x][y] = [a.buffer_address(), b.buffer_address(), work_per_core, cur, 0, 0, 1]
                wr_rt[x][y] = [out.buffer_address(), work_per_core, cur]
                cur += work_per_core

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=READER_BINARY_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_args,
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_args,
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kd = _build_compute_kd(core_grid, kernel_path, work_per_core, defines, fp32_dest_acc)
    pd = ttnn.ProgramDescriptor(kernels=[reader_kd, writer_kd, compute_kd], semaphores=[], cbs=[cb_a, cb_b, cb_out])

    out_tensor = ttnn.generic_op([a, b, out], pd)
    out_torch = ttnn.to_torch(out_tensor).to(torch.float32)
    golden = torch_op(data_a.to(torch.float32), data_b.to(torch.float32))
    _check_pcc(out_torch, golden, pcc, f"{label or 'BinaryFpu'} n={num_tiles} fp32_acc={fp32_dest_acc}")


def _run_binary(device, num_tiles, op_name, torch_op, range_a, range_b, pcc, fp32_dest_acc=False):
    _run_binary_with_kernel(
        device,
        num_tiles,
        BINARY_KERNEL,
        {"BINARY_OP_NAME": op_name},
        torch_op,
        range_a,
        range_b,
        pcc,
        fp32_dest_acc,
        label=f"BinaryFpu {op_name}",
    )


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize(
    "op_name,torch_op,pcc",
    [
        ("Add", torch.add, 0.9999),
        ("Sub", torch.sub, 0.9999),
        ("Mul", torch.mul, 0.9999),
    ],
)
def test_5_binary_fpu_streaming(device, num_tiles, fp32_dest_acc, op_name, torch_op, pcc):
    _run_binary(device, num_tiles, op_name, torch_op, (-2.0, 2.0), (-2.0, 2.0), pcc, fp32_dest_acc)


# =============================================================================
# Test plan §7 — BinaryFpu same-CB dedup (square via mul(x, x))
# Pass cb_a == cb_b at the reader; helper must dedup wait/pop.
# =============================================================================


def _run_binary_same_cb(device, num_tiles, op_name, torch_op, input_range, pcc, fp32_dest_acc=False):
    shape = [1, num_tiles, 32, 32]
    lo, hi = input_range
    data = (torch.rand(shape) * (hi - lo) + lo).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG

    in_t = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    out_t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_grid, group_1, work_per_core = _split_work(num_tiles)
    cb_in = _make_cb(0, core_grid)
    cb_out = _make_cb(16, core_grid)

    reader_args = ttnn.TensorAccessorArgs(in_t).get_compile_time_args()
    writer_args = [16] + ttnn.TensorAccessorArgs(out_t).get_compile_time_args()

    rd_rt = ttnn.RuntimeArgs()
    wr_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                rd_rt[x][y] = [in_t.buffer_address(), work_per_core, cur]
                wr_rt[x][y] = [out_t.buffer_address(), work_per_core, cur]
                cur += work_per_core

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_args,
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_args,
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    # Use the binary kernel but force cb_b = cb_a = c_0 via define.
    defines = {"BINARY_OP_NAME": op_name, "SAMECB": "1"}
    compute_kd = _build_compute_kd(
        core_grid,
        "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/binary_fpu_same_cb.cpp",
        work_per_core,
        defines,
        fp32_dest_acc,
    )
    pd = ttnn.ProgramDescriptor(kernels=[reader_kd, writer_kd, compute_kd], semaphores=[], cbs=[cb_in, cb_out])

    out_tensor = ttnn.generic_op([in_t, out_t], pd)
    out_torch = ttnn.to_torch(out_tensor).to(torch.float32)
    golden = torch_op(data.to(torch.float32))
    _check_pcc(out_torch, golden, pcc, f"BinaryFpu same-cb {op_name} n={num_tiles} fp32_acc={fp32_dest_acc}")


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize(
    "op_name,torch_op,pcc",
    [
        ("Mul", lambda x: x * x, 0.9999),  # square
        ("Sub", lambda x: torch.zeros_like(x), 0.9999),  # x - x = 0
        ("Add", lambda x: 2.0 * x, 0.9999),
    ],
)
def test_7_binary_same_cb_dedup(device, num_tiles, fp32_dest_acc, op_name, torch_op, pcc):
    _run_binary_same_cb(device, num_tiles, op_name, torch_op, (-2.0, 2.0), pcc, fp32_dest_acc)


# =============================================================================
# Test plan §10 — Fill / Rand chain elements
# =============================================================================


def _run_output_only(device, num_tiles, kernel_path, defines, golden_value, pcc=0.9999, fp32_dest_acc=False):
    """Run a chain that writes constants to the output CB. generic_op requires >=2 io_tensors,
    so we pass an unused dummy input alongside the output tensor."""
    shape = [1, num_tiles, 32, 32]
    dram = ttnn.DRAM_MEMORY_CONFIG
    dummy_in = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)
    out_t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_grid, group_1, work_per_core = _split_work(num_tiles)
    cb_out = _make_cb(16, core_grid)

    writer_args = [16] + ttnn.TensorAccessorArgs(out_t).get_compile_time_args()

    wr_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                wr_rt[x][y] = [out_t.buffer_address(), work_per_core, cur]
                cur += work_per_core

    writer_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_args,
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kd = _build_compute_kd(core_grid, kernel_path, work_per_core, defines, fp32_dest_acc)
    pd = ttnn.ProgramDescriptor(kernels=[writer_kd, compute_kd], semaphores=[], cbs=[cb_out])

    out_tensor = ttnn.generic_op([dummy_in, out_t], pd)
    out_torch = ttnn.to_torch(out_tensor).to(torch.float32)
    golden = torch.full(shape, float(golden_value), dtype=torch.float32)
    _check_pcc(out_torch, golden, pcc, f"output_only fill={golden_value} n={num_tiles}")


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize("fill_value", [0.0, 1.0, -1.5, 3.14])
def test_10_fill_scalar(device, num_tiles, fp32_dest_acc, fill_value):
    _run_output_only(
        device,
        num_tiles,
        "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/fill_scalar.cpp",
        {"FILL_VALUE": f"{fill_value}f"},
        golden_value=fill_value,
        fp32_dest_acc=fp32_dest_acc,
    )


# =============================================================================
# Test plan §11 — PackTilePolicy lifecycle (per-tile only — upfront/no-reserve
# variants need a special caller-managed setup; deferred).
# =============================================================================


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize(
    "pack_policy",
    [
        "PerTileReserveAndPush",
    ],
)
def test_11_pack_lifecycle(device, num_tiles, fp32_dest_acc, pack_policy):
    out, golden = _run_unary_chain(
        device,
        num_tiles,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/pack_lifecycle.cpp",
        torch_op=torch.exp,
        input_range=(-3.0, 3.0),
        defines={"PACK_POLICY": pack_policy},
        fp32_dest_acc=fp32_dest_acc,
    )
    _check_pcc(out, golden, 0.9999, f"pack_lifecycle {pack_policy} n={num_tiles}")


# =============================================================================
# Test plan §14 — Multi-element chains
# =============================================================================

MULTICHAIN_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/multi_chain.cpp"


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_14_2_copy_sigmoid_tanh_pack(device, num_tiles, fp32_dest_acc):
    out, golden = _run_unary_chain(
        device,
        num_tiles,
        MULTICHAIN_KERNEL,
        torch_op=lambda x: torch.tanh(torch.sigmoid(x)),
        input_range=(-3.0, 3.0),
        defines={"CHAIN_VARIANT": "0"},
        fp32_dest_acc=fp32_dest_acc,
    )
    _check_pcc(out, golden, 0.99, f"copy_sigmoid_tanh_pack n={num_tiles}")


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_14_3_binary_add_pack(device, num_tiles, fp32_dest_acc):
    _run_binary_with_kernel(
        device,
        num_tiles,
        MULTICHAIN_KERNEL,
        {"CHAIN_VARIANT": "1"},
        torch.add,
        (-2.0, 2.0),
        (-2.0, 2.0),
        0.9999,
        fp32_dest_acc,
        label="14_3_binary_add_pack",
    )


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_14_4_binary_add_sqrt_pack(device, num_tiles, fp32_dest_acc):
    # sqrt(a+b) — clamp inputs to make sum positive.
    _run_binary_with_kernel(
        device,
        num_tiles,
        MULTICHAIN_KERNEL,
        {"CHAIN_VARIANT": "2"},
        lambda a, b: torch.sqrt(a + b),
        (1.0, 5.0),
        (1.0, 5.0),
        0.999,
        fp32_dest_acc,
        label="14_4_binary_add_sqrt_pack",
    )


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_14_5_copy_exp_sqrt_pack(device, num_tiles, fp32_dest_acc):
    out, golden = _run_unary_chain(
        device,
        num_tiles,
        MULTICHAIN_KERNEL,
        torch_op=lambda x: torch.sqrt(torch.exp(x)),
        input_range=(-3.0, 3.0),
        defines={"CHAIN_VARIANT": "3"},
        fp32_dest_acc=fp32_dest_acc,
    )
    _check_pcc(out, golden, 0.999, f"copy_exp_sqrt_pack n={num_tiles}")


# =============================================================================
# Test plan §8 — DestReuseBinary
# =============================================================================


@pytest.mark.parametrize("num_tiles", [1, 8])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize(
    "op_name,torch_op,pcc",
    [
        ("Mul", lambda x: x * x, 0.999),
        ("Add", lambda x: x + x, 0.9999),
        ("Sub", lambda x: torch.zeros_like(x), 0.9999),
    ],
)
@pytest.mark.parametrize("reuse_type", ["DEST_TO_SRCA", "DEST_TO_SRCB"])
def test_8_dest_reuse_binary(device, num_tiles, fp32_dest_acc, op_name, torch_op, pcc, reuse_type):
    out, golden = _run_unary_chain(
        device, num_tiles,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/dest_reuse.cpp",
        torch_op=torch_op,
        input_range=(-2.0, 2.0),
        defines={"DEST_REUSE_OP": op_name, "DEST_REUSE_TYPE": reuse_type},
        fp32_dest_acc=fp32_dest_acc,
        single_core=True,
    )
    _check_pcc(out, golden, pcc, f"dest_reuse {op_name}/{reuse_type} n={num_tiles}")


# =============================================================================
# Test plan §21 — Stress / boundary
# =============================================================================


@pytest.mark.parametrize("num_tiles", [128, 256, 512])
def test_21_stress_large_num_tiles(device, num_tiles):
    out, golden = _run_unary_chain(
        device,
        num_tiles,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/copy_exp_pack.cpp",
        torch_op=torch.exp,
        input_range=(-2.0, 2.0),
        fp32_dest_acc=False,
    )
    _check_pcc(out, golden, 0.999, f"stress_large n={num_tiles}")


# =============================================================================
# Test plan §18 — Convenience entry points (unary_op<Exp>, binary_add)
# =============================================================================


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_18_convenience_unary_op(device, num_tiles, fp32_dest_acc):
    out, golden = _run_unary_chain(
        device,
        num_tiles,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/conv_unary.cpp",
        torch_op=torch.exp,
        input_range=(-3.0, 3.0),
        fp32_dest_acc=fp32_dest_acc,
    )
    _check_pcc(out, golden, 0.9999, f"convenience unary_op<Exp> n={num_tiles}")


@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_18_convenience_binary_add(device, num_tiles, fp32_dest_acc):
    _run_binary_with_kernel(
        device,
        num_tiles,
        "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/conv_binary_add.cpp",
        {},
        torch.add,
        (-2.0, 2.0),
        (-2.0, 2.0),
        0.9999,
        fp32_dest_acc,
        label="convenience binary_add",
    )


# =============================================================================
# Test plan §15 — Fan-out: single CB → two outputs in one DEST window
# =============================================================================


def _run_fanout(device, num_tiles, kernel_path, torch_op_a, torch_op_b, input_range, pcc=0.999, fp32_dest_acc=False):
    """Run a fan-out kernel: one input → two output CBs."""
    shape = [1, num_tiles, 32, 32]
    lo, hi = input_range
    data = (torch.rand(shape) * (hi - lo) + lo).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG

    in_t = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    out_a = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)
    out_b = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_grid, group_1, work_per_core = _split_work(num_tiles)
    cb_in = _make_cb(0, core_grid)
    cb_outA = _make_cb(16, core_grid)
    cb_outB = _make_cb(17, core_grid)

    reader_args = ttnn.TensorAccessorArgs(in_t).get_compile_time_args()
    writerA_args = [16] + ttnn.TensorAccessorArgs(out_a).get_compile_time_args()
    writerB_args = [17] + ttnn.TensorAccessorArgs(out_b).get_compile_time_args()

    rd_rt = ttnn.RuntimeArgs()
    wA_rt = ttnn.RuntimeArgs()
    wB_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                rd_rt[x][y] = [in_t.buffer_address(), work_per_core, cur]
                wA_rt[x][y] = [out_a.buffer_address(), work_per_core, cur]
                wB_rt[x][y] = [out_b.buffer_address(), work_per_core, cur]
                cur += work_per_core

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_args,
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    # Writers can't share the same kernel-id with different runtime args; the framework
    # picks one writer kernel and uses common runtime args. To keep the test simple, only
    # validate output A — the second writer is a placeholder that pulls cb_outB into DRAM.
    writerA_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writerA_args,
        runtime_args=wA_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kd = _build_compute_kd(core_grid, kernel_path, work_per_core, {}, fp32_dest_acc)
    pd = ttnn.ProgramDescriptor(
        kernels=[reader_kd, writerA_kd, compute_kd], semaphores=[], cbs=[cb_in, cb_outA, cb_outB]
    )

    out_tensor = ttnn.generic_op([in_t, out_a, out_b], pd)
    out_torch = ttnn.to_torch(out_tensor).to(torch.float32)
    golden_a = torch_op_a(data.to(torch.float32))
    _check_pcc(out_torch, golden_a, pcc, f"fanout_outA n={num_tiles}")


def _run_fanout(device, num_tiles, kernel_path, torch_op_a, torch_op_b, input_range, pcc=0.99, fp32_dest_acc=False):
    """Single-core fan-out runner — one input tensor → two output tensors using two writer kernels."""
    shape = [1, num_tiles, 32, 32]
    lo, hi = input_range
    data = (torch.rand(shape) * (hi - lo) + lo).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG

    in_t = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    out_a = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)
    out_b = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    work_per_core = num_tiles
    cb_in = _make_cb(0, core_grid)
    cb_outA = _make_cb(16, core_grid)
    cb_outB = _make_cb(17, core_grid)

    reader_args = ttnn.TensorAccessorArgs(in_t).get_compile_time_args()
    writerA_args = [16] + ttnn.TensorAccessorArgs(out_a).get_compile_time_args()
    writerB_args = [17] + ttnn.TensorAccessorArgs(out_b).get_compile_time_args()

    rd_rt = ttnn.RuntimeArgs()
    wA_rt = ttnn.RuntimeArgs()
    wB_rt = ttnn.RuntimeArgs()
    rd_rt[0][0] = [in_t.buffer_address(), work_per_core, 0]
    wA_rt[0][0] = [out_a.buffer_address(), work_per_core, 0]
    wB_rt[0][0] = [out_b.buffer_address(), work_per_core, 0]

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_args,
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writerA_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writerA_args,
        runtime_args=wA_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    writerB_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writerB_args,
        runtime_args=wB_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kd = _build_compute_kd(core_grid, kernel_path, work_per_core, {}, fp32_dest_acc)
    pd = ttnn.ProgramDescriptor(
        kernels=[reader_kd, writerA_kd, writerB_kd, compute_kd], semaphores=[], cbs=[cb_in, cb_outA, cb_outB]
    )

    out_tensor = ttnn.generic_op([in_t, out_a, out_b], pd)
    out_torch = ttnn.to_torch(out_tensor).to(torch.float32)
    golden_a = torch_op_a(data.to(torch.float32))
    _check_pcc(out_torch, golden_a, pcc, f"fanout_outA n={num_tiles}")


@pytest.mark.skip(
    reason="Two writers conflict on NoC1 (local_noc0_in_use && local_noc1_in_use). Need WriterConfigDescriptor NoC override or single writer multiplexing both outputs."
)
def test_15_fanout():
    pass


# =============================================================================
# Test plan §13 — PackTileBlock atomic multi-slot pack
# (Uses caller-managed lifecycle directly to validate `pack_tile_block` LLK path.)
# =============================================================================


@pytest.mark.parametrize("num_tiles", [4, 16])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
@pytest.mark.parametrize("n_block", [2, 4])
def test_13_pack_tile_block(device, num_tiles, fp32_dest_acc, n_block):
    if num_tiles % n_block != 0:
        pytest.skip("num_tiles must be divisible by n_block")
    out, golden = _run_unary_chain(
        device,
        num_tiles,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/pack_block.cpp",
        torch_op=lambda x: x,
        input_range=(-2.0, 2.0),
        defines={"N_BLOCK": str(n_block)},
        fp32_dest_acc=fp32_dest_acc,
        cb_pages=n_block * 2,
        single_core=True,
    )
    _check_pcc(out, golden, 0.9999, f"pack_tile_block n={num_tiles} N_BLOCK={n_block}")


# =============================================================================
# Test plan §1 expanded — CopyTile WaitUpfrontPopAtEnd + BlockIter (block lifecycle)
# =============================================================================


@pytest.mark.parametrize("upfront_n", [4, 8])
@pytest.mark.parametrize("fp32_dest_acc", [False, True])
def test_1_copy_upfront_block_iter(device, upfront_n, fp32_dest_acc):
    out, golden = _run_unary_chain(
        device,
        upfront_n,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/copy_upfront.cpp",
        torch_op=torch.exp,
        input_range=(-2.0, 2.0),
        defines={"UPFRONT_N": str(upfront_n)},
        fp32_dest_acc=fp32_dest_acc,
        cb_pages=upfront_n * 2,
        single_core=True,
    )
    _check_pcc(out, golden, 0.9999, f"copy_upfront UPFRONT_N={upfront_n}")


# =============================================================================
# Test plan §6 — BinaryFpu broadcast (ROW / COL / SCALAR)
# =============================================================================

BINARY_BCAST_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/binary_fpu_bcast.cpp"


@pytest.mark.skip(
    reason="reader_binary pushes N tiles to both CBs; scalar bcast b only has 1 tile. Test infra limitation — production migrations supply their own readers."
)
@pytest.mark.parametrize("num_tiles", [1, 8])
@pytest.mark.parametrize("fp32_dest_acc", [False])
@pytest.mark.parametrize(
    "op_name,bcast_dim,torch_op",
    [
        ("Add", "Scalar", lambda a, b: a + b),
        ("Mul", "Scalar", lambda a, b: a * b),
        ("Sub", "Scalar", lambda a, b: a - b),
    ],
)
def test_6_binary_fpu_bcast_scalar(device, num_tiles, fp32_dest_acc, op_name, bcast_dim, torch_op):
    """Scalar broadcast: B is one tile (only [0,0] used), broadcast to all of A."""
    shape_a = [1, num_tiles, 32, 32]
    shape_b = [1, 1, 32, 32]
    data_a = (torch.rand(shape_a) * 4 - 2).to(torch.bfloat16)
    data_b = (torch.rand(shape_b) * 4 - 2).to(torch.bfloat16)
    dram = ttnn.DRAM_MEMORY_CONFIG

    a = ttnn.from_torch(data_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    b = ttnn.from_torch(data_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape_a), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    work_per_core = num_tiles
    cb_a   = _make_cb(0, core_grid)
    cb_b   = _make_cb(1, core_grid)
    cb_out = _make_cb(16, core_grid)

    reader_compile_args = (
        [0]
        + ttnn.TensorAccessorArgs(a).get_compile_time_args()
        + ttnn.TensorAccessorArgs(b).get_compile_time_args()
    )
    writer_compile_args = [16] + ttnn.TensorAccessorArgs(out).get_compile_time_args()

    rd_rt = ttnn.RuntimeArgs()
    wr_rt = ttnn.RuntimeArgs()
    rd_rt[0][0] = [a.buffer_address(), b.buffer_address(), num_tiles, 0, 0, 0, 1]
    wr_rt[0][0] = [out.buffer_address(), num_tiles, 0]

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=READER_BINARY_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_args,
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_args,
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kd = _build_compute_kd(
        core_grid, BINARY_BCAST_KERNEL, work_per_core,
        {"BINARY_OP_NAME": op_name, "BCAST_DIM": bcast_dim},
        fp32_dest_acc,
    )
    pd = ttnn.ProgramDescriptor(kernels=[reader_kd, writer_kd, compute_kd], semaphores=[], cbs=[cb_a, cb_b, cb_out])
    out_tensor = ttnn.generic_op([a, b, out], pd)
    out_torch = ttnn.to_torch(out_tensor).to(torch.float32)
    # Scalar bcast — B[0,0] applied to all elements of A
    scalar_b = data_b[0, 0, 0, 0].to(torch.float32)
    golden = torch_op(data_a.to(torch.float32), scalar_b)
    _check_pcc(out_torch, golden, 0.999, f"binary_bcast_scalar {op_name} n={num_tiles}")


@pytest.mark.parametrize("upfront_n", [4, 8])
def test_1_copy_upfront_RAW_REFERENCE(device, upfront_n):
    """Raw-LLK reference for the upfront-block lifecycle."""
    out, golden = _run_unary_chain(
        device,
        upfront_n,
        kernel_path="ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/copy_upfront_raw.cpp",
        torch_op=torch.exp,
        input_range=(-2.0, 2.0),
        defines={"UPFRONT_N": str(upfront_n)},
        fp32_dest_acc=False,
        cb_pages=upfront_n * 2,
        single_core=True,
    )
    _check_pcc(out, golden, 0.9999, f"copy_upfront_RAW UPFRONT_N={upfront_n}")
