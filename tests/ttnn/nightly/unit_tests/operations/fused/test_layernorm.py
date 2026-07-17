# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch

import ttnn


from models.common.utility_functions import torch2tt_tensor, run_for_blackhole
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import ttnn_layer_norm, ttnn_rms_norm

TEST_PADDING_VALUE = -42


def ref_layernorm(x, gamma, beta, eps):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)


def ref_rmsnorm(x, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma + beta


def run_layernorm_mix_precision_tests(test_id, in_dtype, gamma_dtype, in0_mem_config, out_mem_config, device):
    epsf = 1e-2

    test_dims = (
        (1, 9, 384, 1024),
        (1, 1, 24, 42),
    )
    for test_shape in test_dims:
        in0 = torch.rand(test_shape) * 2 - 0.95
        in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
        in0_t = ttnn.fill_implicit_tile_padding(in0_t, TEST_PADDING_VALUE)

        if test_id <= 5:
            in1 = torch.rand(test_shape) * 2 - 0.8
            in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)

        if test_id % 3 == 0:
            gamma = torch.ones(test_shape[3])
            beta = torch.zeros(test_shape[3])
        if test_id % 3 == 1:
            gamma = torch.rand(test_shape[3]) * 2 - 1
            beta = torch.zeros(test_shape[3])
        if test_id % 3 == 2:
            gamma = torch.rand(test_shape[3]) * 2 - 1
            beta = torch.rand(test_shape[3]) * 2.0 - 1.1

        gamma_t = torch2tt_tensor(gamma, device, tt_memory_config=in0_mem_config, tt_dtype=gamma_dtype)
        beta_t = torch2tt_tensor(beta, device, tt_memory_config=in0_mem_config, tt_dtype=gamma_dtype)

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=True if in_dtype == ttnn.float32 else False,
        )

        if test_id == 0:
            ttz = ttnn_layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 1:
            ttz = ttnn_layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 2:
            ttz = ttnn_layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 3:
            ttz = ttnn_rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 4:
            ttz = ttnn_rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 5:
            ttz = ttnn_rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 6:
            ttz = ttnn_layer_norm(
                in0_t, epsilon=epsf, memory_config=out_mem_config, compute_kernel_config=compute_kernel_config
            )
        if test_id == 7:
            ttz = ttnn_layer_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 8:
            ttz = ttnn_layer_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 9:
            ttz = ttnn_rms_norm(
                in0_t, epsilon=epsf, memory_config=out_mem_config, compute_kernel_config=compute_kernel_config
            )
        if test_id == 10:
            ttz = ttnn_rms_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )
        if test_id == 11:
            ttz = ttnn_rms_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config,
            )

        # BFLOAT8_B tensors are TILE-only (shared exponent in 16-elem sub-blocks
        # is undefined under ROW_MAJOR). Round-trip via BFLOAT16 for host inspection.
        # Pre-PR #42770, fill_implicit_tile_padding leaked BFLOAT16 on rank-4 BFP8
        # inputs, silently upcasting in0_t and producing BFLOAT16 layer_norm output.
        if ttz.dtype == ttnn.bfloat8_b:
            ttz = ttnn.typecast(ttz, ttnn.bfloat16)
        tt_got_back = ttz.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        pt_in = in0 + in1 if test_id <= 5 else in0
        if test_id <= 2 or 6 <= test_id <= 8:
            ref_fn = ref_layernorm
        else:
            ref_fn = ref_rmsnorm

        ref_lnorm = ref_fn(pt_in, gamma.flatten(), beta.flatten(), epsf)

        # BFLOAT8_B layer_norm uses a shared exponent per 16-element sub-block, so its
        # accuracy on small inner dims (e.g. width=42 padded to tile=64, ~1k elements)
        # is fundamentally lower than BFLOAT16/FLOAT32. Worst observed across the 12
        # (test_id, shape) cases: ATOL=0.86, relFro=0.186, PCC=0.984. Pre-PR #42770
        # these never triggered because the fill_implicit_tile_padding dtype leak
        # silently ran the norm in BFLOAT16.
        if in_dtype == ttnn.bfloat8_b:
            pcc_threshold, atol, frobenius_threshold = 0.98, 1.0, 0.22
        else:
            pcc_threshold, atol, frobenius_threshold = 0.999, 0.098, 0.016

        assert_numeric_metrics(
            ref_lnorm,
            tt_got_back,
            pcc_threshold=pcc_threshold,
            rtol=3.266,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )


@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "gamma_dtype",
    (ttnn.bfloat16, ttnn.float32),
    ids=["BFLOAT16", "FLOAT32"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=["FLOAT32", "BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ids=[
        "add_LN",
        "add_LN_G",
        "add_LN_GB",
        "add_RMSN",
        "add_RMSN_G",
        "add_RMSN_GB",
        "LN",
        "LN_G",
        "LN_GB",
        "RMSN",
        "RMSN_G",
        "RMSN_GB",
    ],
)
def test_layernorm_mix_precision(test_id, in_dtype, gamma_dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(0)
    run_layernorm_mix_precision_tests(test_id, in_dtype, gamma_dtype, in0_mem_config, out_mem_config, device)


@run_for_blackhole("blackhole specific test")
@pytest.mark.parametrize(
    "grid_end, shard_orientation",
    [
        ((7, 9), ttnn.ShardOrientation.ROW_MAJOR),
        ((9, 7), ttnn.ShardOrientation.COL_MAJOR),
    ],
    ids=["row_major_8x10", "col_major_10x8"],
)
def test_layer_norm_block_sharded_height_pad(device, grid_end, shard_orientation):
    """
    Test for feature request issue #43801: block-sharded layer_norm where Mt (height in tiles)
    is not evenly divisible by num_cores in the H direction, so the bottom row of
    cores carries trailing shard-pad tiles.

    Tensor [32, 1, 96, 512] (Mt=96, Kt=16) sharded [320, 64]:
      - ROW_MAJOR on 8x10 grid: num_cores_r=10, block_h=10, 96/10=9 (4 trailing pad tiles).
      - COL_MAJOR on 10x8 grid: num_cores_c=10, block_h=10, same 4-tile pad in H.
    The op previously rejected both with a strict Mt/num_cores == block_h check;
    relaxing to div_up is correct because the kernel processes block_h tile-rows
    per core regardless of overall Mt.
    """
    torch.manual_seed(0)
    channels = 512

    x_torch = torch.randn(32, 1, 96, channels, dtype=torch.bfloat16)
    weight_torch = torch.randn(channels, dtype=torch.bfloat16)
    bias_torch = torch.randn(channels, dtype=torch.bfloat16)
    epsilon = 1e-5

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_end[0], grid_end[1]))})
    shard_spec = ttnn.ShardSpec(shard_grid, [320, 64], shard_orientation)
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)

    x = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem_config
    )
    weight = ttnn.from_torch(
        weight_torch.reshape(1, 1, channels // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        bias_torch.reshape(1, 1, channels // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    out = ttnn_layer_norm(
        x,
        weight=weight,
        bias=bias,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )
    out_torch = ttnn.to_torch(out)

    ref = torch.nn.functional.layer_norm(
        x_torch.to(torch.float32), [channels], weight_torch.to(torch.float32), bias_torch.to(torch.float32), epsilon
    ).to(torch.bfloat16)

    assert_numeric_metrics(
        ref,
        out_torch,
        pcc_threshold=0.999,
        rtol=0.006,
        atol=0.018,
        frobenius_threshold=0.003,
    )


@pytest.mark.parametrize("h", [22, 1632, 8192, 16384])
@pytest.mark.parametrize("w", [45, 1280])
@pytest.mark.parametrize("num_chunks", [1, 4])
def test_layer_norm_4D_llama(device, h, w, num_chunks):
    """
    Test specific shapes for LLama and other LLMs
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, num_chunks, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_layer_norm(input_tensor, weight=weight, bias=bias)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.006,
        atol=0.018,
        frobenius_threshold=0.003,
    )


# ---------------------------------------------------------------------------------------------
# FP32 coverage for the complete (non-distributed) interleaved LayerNorm op
# Spans the full config matrix: {legacy, welford} x {fp32, bf16} input x {TILE, ROW_MAJOR} input
# x {bf16, fp32} gamma/beta x {TILE, ROW_MAJOR} gamma/beta. FP32 requires fp32_dest_acc_en=True.
# Welford requires TILE input (ROW_MAJOR input hangs for every dtype), so welford x rm_in is
# skipped to record the limitation.
# The gamma/beta layout axis matters because it selects the reader kernel (use_row_major_kernel):
# ROW_MAJOR gamma/beta go through reader_unary_interleaved_ln_rm_gb.cpp, which reads them as
# row-major sticks, while TILE gamma/beta are read whole-tile and are datum-size-agnostic.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["gb_tile", "gb_rm"])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["gb_bf16", "gb_fp32"])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile_in", "rm_in"])
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "legacy"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b], ids=["fp32", "bf16", "bf8"])
def test_layernorm_interleaved_all_config(device, dtype, use_welford, input_layout, gamma_dtype, gamma_layout):
    if use_welford and input_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Welford requires TILE input; ROW_MAJOR input hangs (dtype-independent limitation)")
    if gamma_layout == ttnn.ROW_MAJOR_LAYOUT and input_layout == ttnn.ROW_MAJOR_LAYOUT:
        # Pre-existing device hang, reproduces on unpatched main and unrelated to the reader fixes
        # here: the rm_gb reader has no TILIZE_IN path, so it reads a ROW_MAJOR input as tiles and
        # nothing ever fills cb_in_rm. Hangs for bf16 and fp32 input alike. See #49970.
        pytest.skip("ROW_MAJOR input + ROW_MAJOR gamma/beta hangs the device (pre-existing, see #49970)")
    if dtype == ttnn.bfloat8_b and input_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("BFLOAT8_B is TILE-only (shared exponent per 16-elem sub-block is undefined under ROW_MAJOR)")

    torch.manual_seed(0)
    M, K = 256, 1024
    x = torch.rand((1, 1, M, K), dtype=torch.float32)
    w = torch.rand((K,), dtype=torch.float32)
    b = torch.rand((K,), dtype=torch.float32)
    ref = torch.nn.functional.layer_norm(x, (K,), weight=w, bias=b, eps=1e-12)

    xt = ttnn.from_torch(x, dtype=dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # ROW_MAJOR gamma/beta must be shaped [1, 1, K/TILE_WIDTH, TILE_WIDTH]; TILE takes [1, 1, 1, K]
    gb_shape = (1, 1, K // 32, 32) if gamma_layout == ttnn.ROW_MAJOR_LAYOUT else (1, 1, 1, K)
    wt = ttnn.from_torch(w.reshape(gb_shape), dtype=gamma_dtype, layout=gamma_layout, device=device)
    bt = ttnn.from_torch(b.reshape(gb_shape), dtype=gamma_dtype, layout=gamma_layout, device=device)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # required for FP32
        packer_l1_acc=False,
    )
    cfg = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)

    recip = None
    if use_welford:
        grid = device.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        recip = ttnn.create_layer_norm_reciprocals(device, crs, K)

    out = ttnn.layer_norm(
        xt,
        epsilon=1e-12,
        weight=wt,
        bias=bt,
        program_config=cfg,
        compute_kernel_config=compute_kernel_config,
        recip_tensor=recip,
    )
    ot = ttnn.to_torch(ttnn.from_device(out)).float().reshape(ref.shape)

    if dtype == ttnn.bfloat8_b:
        pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.02, 0.05, 0.012
    elif dtype == ttnn.bfloat16:
        pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.006, 0.019, 0.004
    else:
        pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.006, 0.012, 0.003

    assert_numeric_metrics(
        ref,
        ot,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )
