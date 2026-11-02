# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-tile M coverage for both DRAM-sharded pipelines.

Check every tile row independently against the product of quantized device operands: a
whole-tensor PCC alone can hide missing rows or incorrect shard/subblock strides.
"""

import math

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_for_slow_dispatch
from tests.ttnn.unit_tests.operations.matmul.test_matmul_dram_sharded_multicore import (
    _check,
    _grid,
    _matmul,
    _no_worse_than_stock,
    _rel_err,
    _operands,
    _reference,
    _width_sharded_l1,
)

# M, K, N, activation cores, K block tiles. Include multiple blocks within a shard,
# one block per shard, several shards per block, padded M/N and tall M.
CASES = [
    pytest.param(64, 1024, 1024, 8, 2, id="split-shard"),
    pytest.param(96, 1024, 1536, 8, 4, id="one-block-per-shard"),
    pytest.param(128, 1024, 3072, 32, 4, id="gather-four-shards"),
    pytest.param(33, 1024, 800, 8, 2, id="padded-m-n"),
    pytest.param(65, 1024, 512, 32, 4, id="padded-m-multishard"),
    pytest.param(256, 2048, 2048, 16, 8, id="eight-m-tiles"),
    pytest.param(512, 1024, 1024, 16, 4, id="sixteen-m-tiles"),
    pytest.param(64, 8192, 1024, 32, 8, id="long-k"),
]


def _run(
    device,
    m,
    k,
    n,
    storage,
    block,
    workers,
    *,
    wd="bfp8",
    xd="bf16",
    od="bf16",
    fp32=True,
    tile_h=32,
    k_logical=None,
    seed=0,
    activation=None,
    bias_enabled=False,
    use_multicore_pipeline=None,
    packer_l1_acc=True,
    output_storage=13,
):
    height = math.ceil(m / tile_h) * tile_h
    x, w = _operands(
        device,
        k,
        n,
        wd,
        xd,
        ("rm", storage),
        k // 32 // storage,
        seed=seed,
        m=m,
        tile_h=tile_h,
        k_logical=k_logical,
    )
    # Deliberately reshard to a different set of cores with shard widths that may split a
    # compute subblock or DRAM bank. This catches writers that assume identical layouts.
    out_cores = min(output_storage, math.ceil(n / 32))
    out_sw = math.ceil(n / (32 * out_cores))
    out_cores = math.ceil(n / (32 * out_sw))
    out_mc = _width_sharded_l1(_grid(device, ("rm", out_cores)), out_sw, height)
    bias = None
    if bias_enabled:
        weight_shard = w.memory_config().shard_spec
        bias_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                weight_shard.grid,
                (32, weight_shard.shape[1]),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        bias = ttnn.from_torch(
            torch.randn(1, 1, 1, n),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=bias_mc,
        )
    kw = dict(
        out_mc=out_mc,
        out_dtype=od,
        per_core_n=out_sw,
        per_core_m=height // tile_h,
        in0_block_w=block,
        cores_per_bank=workers,
        fidelity="LoFi" if wd == "bfp4" else "HiFi2",
        fp32=fp32,
        fused_activation=activation,
        bias=bias,
        use_multicore_pipeline=use_multicore_pipeline,
        packer_l1_acc=packer_l1_acc,
    )
    ref = _reference(x, w, n)
    input_before = ttnn.to_torch(x) if k_logical is not None else None
    output = _matmul(x, w, **kw)
    if input_before is not None:
        assert torch.equal(input_before, ttnn.to_torch(x)), "Matmul modified valid input elements"
    if bias is not None:
        ref += ttnn.to_torch(bias).double().reshape(1, n)
    if activation is not None:
        ref = torch.nn.functional.silu(ref)
    actual = ttnn.to_torch(output).reshape(m, n)
    _check_rows(actual, ref, od, fp32, tile_h)
    return x, w, output, kw, ref


def _check_rows(actual, ref, dtype, fp32, tile_h=32):
    assert torch.isfinite(actual).all()
    _check(actual, ref, dtype, fp32)
    for row in range(0, ref.shape[0], tile_h):
        _check(actual[row : row + tile_h], ref[row : row + tile_h], dtype, fp32)


@pytest.mark.parametrize("m,k,n,storage,block", CASES)
@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
@pytest.mark.parametrize("fp32", [True, False], ids=["fp32", "bf16"])
def test_multirow_layouts(device, m, k, n, storage, block, workers, fp32):
    _run(device, m, k, n, storage, block, workers, fp32=fp32)


@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
@pytest.mark.parametrize(
    "wd,xd,od",
    [("bf16", "bf16", "bf16"), ("bfp4", "bf16", "bf16"), ("bfp8", "bfp8", "bfp8")],
)
@pytest.mark.parametrize("fp32", [True, False])
def test_multirow_formats(device, workers, wd, xd, od, fp32):
    _run(device, 96, 2048, 3072, 32, 4, workers, wd=wd, xd=xd, od=od, fp32=fp32)


@pytest.mark.parametrize("block", [1, 3, 5, 8])
@pytest.mark.parametrize("fp32", [True, False])
def test_multirow_ragged_k_and_split_k(device, block, fp32):
    _run(device, 96, 7168, 256, 56, block, 4, fp32=fp32)


@pytest.mark.parametrize("m", [32, 64])
@pytest.mark.parametrize("k", [32, 64])
@pytest.mark.parametrize("fp32", [True, False])
def test_multicore_short_k_omits_empty_partitions(device, m, k, fp32):
    n = device.dram_grid_size().x * 32
    for seed in range(2):
        # Populate every worker's partial buffer before requesting more K groups
        # than K tiles. An empty group must not contribute stale L1 contents.
        _run(device, m, 128, n, 4, 1, 4, fp32=fp32, seed=seed)
        _run(device, m, k, n, k // 32, 1, 4, fp32=fp32, seed=seed + 2)


@pytest.mark.parametrize("xd", ["bf16", "bfp8"])
@pytest.mark.parametrize("tile_h", [16, 32])
@pytest.mark.parametrize("block", [2, 8])
@pytest.mark.parametrize("k_logical", [1016, 1000])
def test_multicast_multirow_padded_k(device, tile_h, block, k_logical, xd):
    _run(device, 65, 1024, 1024, 8, block, 0, tile_h=tile_h, k_logical=k_logical, xd=xd)


@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("bias_enabled", [False, True])
@pytest.mark.parametrize("activation_enabled", [False, True])
def test_multicast_multirow_fusions(device, bias_enabled, activation_enabled, fp32):
    activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation_enabled else None
    _run(
        device,
        96,
        1024,
        3072,
        32,
        4,
        0,
        activation=activation,
        bias_enabled=bias_enabled,
        fp32=fp32,
    )


@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
def test_multirow_program_cache(device, workers):
    # Keep the first tensors alive so the second invocation must use different addresses.
    first = _run(device, 64, 1024, 1024, 8, 2, workers)
    entries = device.num_program_cache_entries()
    second = _run(device, 64, 1024, 1024, 8, 2, workers, seed=1)
    assert device.num_program_cache_entries() == entries
    assert not torch.equal(ttnn.to_torch(first[2]), ttnn.to_torch(second[2]))


@pytest.mark.skipif(not is_blackhole(), reason="The original multi-reader pipeline requires Blackhole")
@pytest.mark.parametrize("readers", [2, 3])
@pytest.mark.parametrize("fp32", [True, False])
def test_multicast_multirow_multiple_readers(device, readers, fp32):
    n = device.dram_grid_size().x * 32 * 12
    _run(device, 96, 2048, n, 32, 4, readers, fp32=fp32, use_multicore_pipeline=False)


@pytest.mark.skipif(not is_blackhole(), reason="Fused multi-reader fallback requires Blackhole")
@pytest.mark.parametrize("fp32", [True, False])
def test_multirow_fused_multicore_fallback(device, fp32):
    _run(
        device,
        96,
        1024,
        3072,
        32,
        4,
        2,
        fp32=fp32,
        bias_enabled=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        use_multicore_pipeline=True,
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
@pytest.mark.parametrize("m", [64, 65, 128])
@pytest.mark.parametrize("fp32", [True, False])
@skip_for_slow_dispatch("trace capture requires fast dispatch")
def test_multirow_trace_with_input_updates(device, workers, m, fp32):
    x, w, _, kw, _ = _run(device, m, 1024, 1536, 8, 2, workers, fp32=fp32)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = _matmul(x, w, **kw)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for seed in range(12):
            torch.manual_seed(seed + 10)
            host_x = ttnn.from_torch(
                torch.randn(1, 1, m, 1024),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_x, x, cq_id=0)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
            actual = ttnn.to_torch(output).reshape(m, 1536)
            _check_rows(actual, _reference(x, w, 1536), "bf16", fp32)
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("fp32", [True, False])
@skip_for_slow_dispatch("trace capture requires fast dispatch")
def test_multirow_coalesced_reads_under_trace_pressure(device, fp32):
    # KB == storage-shard width coalesces both M tiles into a 16 KiB read. A tagged
    # read must enter the NoC before its zero outstanding count can mean complete.
    # Back-to-back invocations exposed premature publication of the second tile row.
    x, w, output, kw, _ = _run(device, 64, 4096, 4096, 32, 4, 4, fp32=fp32, output_storage=32)
    kw["optional_output_tensor"] = output
    _matmul(x, w, **kw)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(128):
        _matmul(x, w, **kw)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for seed in range(12):
            torch.manual_seed(seed + 100)
            host_x = ttnn.from_torch(torch.randn(1, 1, 64, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_x, x, cq_id=0)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
            _check_rows(ttnn.to_torch(output).reshape(64, 4096), _reference(x, w, 4096), "bf16", fp32)
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize(
    "m,k,n,storage,block",
    [(96, 2048, 1536, 32, 2), (512, 1024, 1024, 16, 4), (64, 16384, 1024, 32, 8)],
    ids=["resident", "streaming-tall", "streaming-long-k"],
)
def test_multirow_accuracy_against_single_row(device, workers, fp32, m, k, n, storage, block):
    x, w, output, kw, ref = _run(device, m, k, n, storage, block, workers, fp32=fp32)
    host_x = ttnn.to_torch(x)
    single_x_mc = _width_sharded_l1(x.memory_config().shard_spec.grid, k // 32 // storage)
    output_spec = output.memory_config().shard_spec
    single_kw = dict(
        kw,
        per_core_m=1,
        cores_per_bank=0,
        use_multicore_pipeline=False,
        out_mc=_width_sharded_l1(output_spec.grid, output_spec.shape[1] // 32),
    )
    single_outputs = []
    for row in range(0, m, 32):
        single_x = ttnn.from_torch(
            host_x[:, :, row : row + 32, :].contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=single_x_mc,
        )
        single_out = _matmul(single_x, w, **single_kw)
        single_outputs.append(ttnn.to_torch(single_out).reshape(32, n))
        ttnn.deallocate(single_x)
        ttnn.deallocate(single_out)
    stock = torch.cat(single_outputs)
    _check_rows(stock, ref, "bf16", fp32)
    _no_worse_than_stock(_rel_err(ttnn.to_torch(output).reshape(m, n), ref), _rel_err(stock, ref), fp32)


@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_multicast_multirow_without_packer_accumulation(device, fp32, bias_enabled):
    _run(device, 96, 1024, 1536, 8, 2, 0, fp32=fp32, bias_enabled=bias_enabled, packer_l1_acc=False)


@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize("m", [17, 33])
def test_multirow_fused_batch(device, workers, fp32, m):
    # Batch padding stays between matrices in each width shard. Include two batch
    # dimensions and a partial final M tile in every matrix.
    k, n, storage = 1024, 512, 8
    unused_x, w = _operands(device, k, n, "bfp8", "bf16", ("rm", storage), 4)
    ttnn.deallocate(unused_x)
    height = 6 * math.ceil(m / 32) * 32
    grid = _grid(device, ("rm", storage))
    x = ttnn.from_torch(
        torch.randn(2, 3, m, k),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_width_sharded_l1(grid, 4, height),
    )
    ref = ttnn.to_torch(x).double() @ ttnn.to_torch(w).double().reshape(k, n)
    output = _matmul(
        x,
        w,
        out_mc=_width_sharded_l1(grid, 2, height),
        out_dtype="bf16",
        per_core_n=2,
        per_core_m=height // 32,
        in0_block_w=2,
        cores_per_bank=workers,
        fidelity="HiFi2",
        fp32=fp32,
    )
    assert tuple(output.shape) == (2, 3, m, n)
    actual = ttnn.to_torch(output).reshape(6, m, n)
    for batch in range(6):
        _check_rows(actual[batch], ref.reshape(6, m, n)[batch], "bf16", fp32)


@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
def test_multirow_rejects_per_batch_weights(device, workers, expect_error):
    k, n = 1024, 512
    grid = _grid(device, ("rm", 8))
    x = ttnn.from_torch(
        torch.randn(1, 2, 32, k),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_width_sharded_l1(grid, 4, 64),
    )
    banks = device.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    weight_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (2 * k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    w = ttnn.from_torch(
        torch.randn(1, 2, k, n),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weight_mc,
    )
    with expect_error(RuntimeError, "one weight matrix"):
        _matmul(
            x,
            w,
            out_mc=_width_sharded_l1(grid, 2, 64),
            out_dtype="bf16",
            per_core_n=2,
            per_core_m=2,
            in0_block_w=2,
            cores_per_bank=workers,
            fidelity="HiFi2",
            fp32=True,
        )


@pytest.mark.parametrize("workers", [3, 5, 7])
@pytest.mark.parametrize("fp32", [True, False])
def test_multirow_uneven_worker_columns(device, workers, fp32):
    grid = device.compute_with_storage_grid_size()
    if workers * device.dram_grid_size().x > grid.x * grid.y:
        pytest.skip("more workers than the device grid")
    _run(device, 96, 1024, 3072, 32, 4, workers, fp32=fp32)


@pytest.mark.parametrize("workers", [0, 2], ids=["multicast", "multicore"])
@pytest.mark.parametrize("packer_l1_acc", [False, True])
def test_multirow_float32_partials(device, workers, packer_l1_acc):
    # BF16-representable operands in Float32 storage isolate partial-sum precision.
    # A one-tile K block maximizes the number of partial updates; Float32 output
    # removes BF16 output rounding that would otherwise conceal a lossy reload.
    torch.manual_seed(0)
    m, k, n = 96, 4096, 768
    grid = _grid(device, ("rm", 8))
    banks = device.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    weight_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, math.ceil(n / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    host_x = torch.rand(1, 1, m, k).to(torch.bfloat16).float()
    host_w = torch.rand(1, 1, k, n).to(torch.bfloat16).float()
    x = ttnn.from_torch(
        host_x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_width_sharded_l1(grid, k // 256, m),
    )
    w = ttnn.from_torch(host_w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=weight_mc)
    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=1,
        per_core_M=m // 32,
        per_core_N=n // 256,
        num_workers_per_dram_bank=max(workers, 1),
        use_multicore_pipeline=workers >= 2,
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )
    output = ttnn.matmul(
        x,
        w,
        program_config=program,
        memory_config=_width_sharded_l1(grid, n // 256, m),
        dtype=ttnn.float32,
        compute_kernel_config=compute,
    )
    actual = ttnn.to_torch(output).double().reshape(m, n)
    reference = (host_x.double() @ host_w.double()).reshape(m, n)
    for row in range(0, m, 32):
        error = (actual[row : row + 32] - reference[row : row + 32]).abs().max() / reference[row : row + 32].abs().max()
        assert error < 2e-4, error


@pytest.mark.parametrize("fp32", [True, False])
@pytest.mark.parametrize(
    "m,k,n,storage,block",
    [(64, 1024, 1024, 8, 32), (64, 1024, 1024, 8, 16), (256, 4096, 4096, 32, 8)],
    ids=["single-k-block", "two-k-blocks", "tall-wide-k-block"],
)
def test_multicast_multirow_weight_ring(device, fp32, m, k, n, storage, block):
    # Exercise the single-block reservation, final drain and a larger working set.
    # Cap each bank at 16 weight columns so the tall case also fits seven-bank devices.
    n = min(n, device.dram_grid_size().x * 16 * 32)
    # Spread the tall output across 64 cores to leave room for weight read lookahead.
    _run(device, m, k, n, storage, block, 0, fp32=fp32, output_storage=64 if m == 256 else 32)
