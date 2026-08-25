# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
import pytest
import torch
import math

from models.common.utility_functions import torch2tt_tensor
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import (
    ttnn_layer_norm_in_place,
    ttnn_rms_norm_in_place,
)


def rms_norm(x, dim, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean([-i for i in range(1, len(dim) + 1)], keepdim=True) + eps) * gamma + beta


@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "gb_DRAM",
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
@pytest.mark.parametrize("width_padding", [False, True], ids=["no_padding", "padding"])
def test_layernorm_sharded_mix_precision_rm(
    test_id, in_dtype, gamma_dtype, gamma_beta_mem_config, out_mem_config, device, width_padding
):
    torch.manual_seed(1234)
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    if grid_size[1] > 8:
        grid_size[1] = 8
    fidelity = ttnn.MathFidelity.HiFi4

    epsf = 1e-2
    batch = grid_size[1]

    width = 128 * grid_size[1]
    if grid_size[1] > 1 and width_padding:
        width = 128 * (grid_size[1] - 1) + 96  # 4 tiles per core, except last one that has 3

    in0_shape = (batch, 1, 32 * grid_size[0], width)
    M = in0_shape[2] * batch
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    shard_shape = [M // grid_size[0], math.ceil(K / grid_size[1] / 32) * 32]
    in0_t_shard = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        shard_shape,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    if test_id <= 5:
        in1 = torch.rand(in0_shape) * 2 - 0.8
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
        in1_t_shard = ttnn.interleaved_to_sharded(
            in1_t,
            grid_size,
            shard_shape,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )

    if test_id % 3 == 0:
        gamma = torch.ones(in0_shape[3])
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 1:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 2:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttnn.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        gamma_dtype,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(device, gamma_beta_mem_config)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttnn.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        gamma_dtype,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(device, gamma_beta_mem_config)

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        inplace=True,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
    )

    if test_id == 0:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 1:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 2:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 3:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 4:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 5:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 6:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 7:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 8:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 9:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 10:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 11:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    ttz = ttnn.sharded_to_interleaved(ttz, in0_mem_config)
    tt_got_back = ttz.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()

    pt_in = in0 + in1 if test_id <= 5 else in0
    if test_id <= 2 or 6 <= test_id <= 8:
        ref_fn = torch.nn.functional.layer_norm
    else:
        ref_fn = rms_norm
    ref_lnorm = ref_fn(pt_in, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    assert_numeric_metrics(
        ref_lnorm,
        tt_got_back,
        pcc_threshold=0.999,
        rtol=3.293,
        atol=0.101,
        frobenius_threshold=0.030,
    )


@pytest.mark.parametrize(
    "shard_orientation",
    (ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR),
    ids=["RM", "CM"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "gb_DRAM",
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
    "M, K, subblock_w",
    [
        (64, 8192, 4),
        (64, 8192, 4),  # padding test
        (512, 2048, 1),
    ],
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
def test_layernorm_1d_sharded_mix_precision_rm(
    test_id, M, K, subblock_w, in_dtype, gamma_dtype, gamma_beta_mem_config, out_mem_config, shard_orientation, device
):
    torch.manual_seed(1234)
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    device_grid_size = device.compute_with_storage_grid_size()
    if device_grid_size.x >= 8 and device_grid_size.y >= 4:
        if device_grid_size.y >= 8:
            grid_size = (8, 8)
        else:
            grid_size = (8, 4)
    else:
        pytest.skip("Device grid size is too small for this test")

    fidelity = ttnn.MathFidelity.HiFi2

    epsf = 1e-2

    in0_shape = torch.Size([1, 1, M, K])
    M = in0_shape.numel() // in0_shape[3]
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    shard_shape = [M, math.ceil(K / (grid_size[0] * grid_size[1]) / 32) * 32]
    in0_t_shard = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        shard_shape,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        shard_orientation,
    )

    if test_id <= 5:
        in1 = torch.rand(in0_shape) * 2 - 0.8
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
        in1_t_shard = ttnn.interleaved_to_sharded(
            in1_t,
            grid_size,
            shard_shape,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            shard_orientation,
        )

    if test_id % 3 == 0:
        gamma = torch.ones(in0_shape[3])
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 1:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 2:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttnn.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        gamma_dtype,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(device, gamma_beta_mem_config)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttnn.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        gamma_dtype,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(device, gamma_beta_mem_config)

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=subblock_w,
        block_h=M // 32,
        block_w=shard_shape[1] // 32,
        inplace=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    if test_id == 0:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 1:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 2:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 3:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 4:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 5:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 6:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 7:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 8:
        ttz = ttnn_layer_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 9:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 10:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 11:
        ttz = ttnn_rms_norm_in_place(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    ttz = ttnn.sharded_to_interleaved(ttz, in0_mem_config)
    tt_got_back = ttz.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()

    pt_in = in0 + in1 if test_id <= 5 else in0
    if test_id <= 2 or 6 <= test_id <= 8:
        ref_fn = torch.nn.functional.layer_norm
    else:
        ref_fn = rms_norm
    ref_lnorm = ref_fn(pt_in, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    assert_numeric_metrics(
        ref_lnorm,
        tt_got_back,
        pcc_threshold=0.999,
        rtol=3.158,
        atol=0.087,
        frobenius_threshold=0.016,
    )


# ---------------------------------------------------------------------------------------------
# FP32 coverage for the complete (non-distributed) block-sharded LayerNorm op.
# Spans {legacy, welford} x {fp32, bf16} input x {bf16, fp32} ROW_MAJOR gamma/beta. FP32 requires
# fp32_dest_acc_en=True. Input is TILE (welford requires TILE; ROW_MAJOR input hangs).
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["gb_bf16", "gb_fp32"])
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "legacy"])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b], ids=["fp32", "bf16", "bf8"])
def test_layernorm_block_sharded_all_config(device, dtype, use_welford, gamma_dtype):
    torch.manual_seed(1234)
    g = device.compute_with_storage_grid_size()
    grid_size = [g.x, min(g.y, 8)]
    batch = grid_size[1]
    width = 128 * grid_size[1]
    in0_shape = (batch, 1, 32 * grid_size[0], width)
    M, K = in0_shape[2] * batch, in0_shape[3]

    x = torch.rand(in0_shape, dtype=torch.float32) * 2 - 0.95
    w = torch.rand(K, dtype=torch.float32) * 2 - 1
    b = torch.rand(K, dtype=torch.float32) * 2 - 1.1
    ref = torch.nn.functional.layer_norm(x, (K,), weight=w, bias=b, eps=1e-2)

    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_shape = [M // grid_size[0], math.ceil(K / grid_size[1] / 32) * 32]
    x_shard = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_shape, ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ShardOrientation.COL_MAJOR
    )

    wt = ttnn.from_torch(w.reshape(1, 1, -1, 32), dtype=gamma_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bt = ttnn.from_torch(b.reshape(1, 1, -1, 32), dtype=gamma_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        inplace=False,
        use_welford=use_welford,
    )
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, x_shard.memory_config().shard_spec
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # required for FP32
        packer_l1_acc=False,
    )

    recip = None
    if use_welford:
        sspec = x_shard.memory_config().shard_spec
        recip = ttnn.create_layer_norm_reciprocals(device, sspec.grid, sspec.shape[1])

    out = ttnn.layer_norm(
        x_shard,
        epsilon=1e-2,
        weight=wt,
        bias=bt,
        memory_config=out_mem,
        program_config=cfg,
        compute_kernel_config=compute_kernel_config,
        recip_tensor=recip,
    )
    ot = (
        ttnn.to_torch(ttnn.from_device(ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)))
        .float()
        .reshape(ref.shape)
    )

    if dtype == ttnn.bfloat8_b:
        pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.02, 0.045, 0.011
    elif dtype == ttnn.bfloat16:
        pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.006, 0.019, 0.003
    else:
        pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.006, 0.013, 0.003

    assert_numeric_metrics(
        ref,
        ot,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )
