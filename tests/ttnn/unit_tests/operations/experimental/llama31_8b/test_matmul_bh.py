# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import run_for_blackhole

pytestmark = pytest.mark.skipif(bool(os.getenv("TT_METAL_SIMULATOR")), reason="Llama-3.1-8B shapes need hardware")


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


FIDELITY = {"HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}

PREFILL = [
    (1, 128, 4096, 6144, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", True, (8, 10), 1, 1, 1, 28, True),
    (1, 128, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, "HiFi2", True, (8, 8), 8, 1, 1, 19, True),
    (1, 128, 4096, 14336, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16, "LoFi", False, (8, 8), 8, 4, 1, 64, False),
    (1, 128, 14336, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", False, (8, 8), 8, 1, 1, 19, False),
    (1, 1024, 4096, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, "HiFi2", True, (8, 8), 8, 1, 4, 19, True),
    (2, 512, 4096, 14336, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16, "LoFi", False, (8, 8), 4, 4, 2, 64, False),
    (2, 512, 4096, 14336, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", False, (8, 8), 4, 4, 2, 64, False),
]
MINIMAL = [
    (1, 1024, 4096, 6144, ttnn.bfloat16, True, (8, 10)),
    (2, 512, 14336, 4096, ttnn.bfloat8_b, False, (8, 8)),
]
DECODE = [
    (32, 4096, 16032, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, "HiFi2", False, (8, 8), 2, 8),
    (32, 4096, 14336, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16, "LoFi", False, (8, 8), 2, 7),
    (32, 14336, 4096, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", False, (8, 8), 7, 2),
    (32, 4096, 6144, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", True, (8, 4), 4, 6),
    (32, 4096, 4096, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, "HiFi2", True, (8, 4), 4, 4),
]


def dram_width_sharded(device, k, n):
    g = device.dram_grid_size()
    padded = -(-n // (32 * g.x)) * 32 * g.x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    spec = ttnn.ShardSpec(grid, (k, padded // g.x), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def kernel_config(fidelity, fp32):
    return ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE,
        math_fidelity=FIDELITY[fidelity],
        math_approx_mode=fp32,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


def pcc_for(fidelity, x_dtype):
    return 0.99 if fidelity == "LoFi" or x_dtype == ttnn.bfloat8_b else 0.999


L = ttnn.experimental.llama31_8b


def weight(device, k, n, w_dtype, seed=0):
    torch.manual_seed(seed)
    return ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.02,
        dtype=w_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_width_sharded(device, k, n),
    )


def same(a, b):
    assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b))


def twice(device, make_x, ours, main):
    for i in range(2):
        x = make_x(i)
        entries = device.num_program_cache_entries()
        out = ours(x)
        if i:
            assert device.num_program_cache_entries() == entries
        ref = main(x)
        for a, b in zip(out, ref) if isinstance(out, (tuple, list)) else ((out, ref),):
            same(a, b)


@run_for_blackhole()
@pytest.mark.parametrize(
    "batch, m, k, n, x_dtype, w_dtype, out_dtype, fidelity, fp32, grid, in0_block_w, sbw, per_core_m, per_core_n, fuse_batch",
    PREFILL,
)
def test_prefill_matmul_bh(
    device,
    batch,
    m,
    k,
    n,
    x_dtype,
    w_dtype,
    out_dtype,
    fidelity,
    fp32,
    grid,
    in0_block_w,
    sbw,
    per_core_m,
    per_core_n,
    fuse_batch,
):
    def make_x(i):
        torch.manual_seed(1 + i)
        return ttnn.from_torch(
            torch.randn(1, batch, m, k),
            dtype=x_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = make_x(0)
    args = dict(
        dtype=out_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=kernel_config(fidelity, fp32),
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=sbw,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=fuse_batch,
        ),
    )
    w = weight(device, k, n, w_dtype)
    assert L.matmul_bh_covers(x, w, **args)
    twice(device, make_x, lambda x: L.matmul_bh(x, w, **args), lambda x: ttnn.linear(x, w, **args))


@run_for_blackhole()
@pytest.mark.parametrize("batch, m, k, n, x_dtype, fp32, grid", MINIMAL)
def test_prefill_minimal_matmul_bh(device, batch, m, k, n, x_dtype, fp32, grid):
    def make_x(i):
        torch.manual_seed(1 + i)
        return ttnn.from_torch(
            torch.randn(1, batch, m, k),
            dtype=x_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = make_x(0)
    args = dict(
        config=ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=1,
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        ),
        compute_kernel_config=kernel_config("HiFi2", fp32),
    )
    w = weight(device, k, n, ttnn.bfloat8_b)
    assert L.minimal_matmul_bh_covers(x, w, **args)
    twice(
        device,
        make_x,
        lambda x: L.minimal_matmul_bh(x, w, **args),
        lambda x: ttnn.experimental.minimal_matmul(x, w, **args),
    )


@run_for_blackhole()
@pytest.mark.parametrize(
    "m, k, n, x_dtype, w_dtype, out_dtype, fidelity, fp32, act_grid, in0_block_w, per_core_n", DECODE
)
def test_decode_matmul_bh(
    device, m, k, n, x_dtype, w_dtype, out_dtype, fidelity, fp32, act_grid, in0_block_w, per_core_n
):
    torch.manual_seed(1)
    cores = act_grid[0] * act_grid[1]
    act_mem = ttnn.create_sharded_memory_config(
        (m, k // cores),
        core_grid=ttnn.CoreGrid(y=act_grid[1], x=act_grid[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    def make_x(i):
        torch.manual_seed(1 + i)
        return ttnn.from_torch(
            torch.randn(1, 1, m, k), dtype=x_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=act_mem
        )

    x = make_x(0)
    args = dict(
        dtype=out_dtype,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=kernel_config(fidelity, fp32),
        program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w, per_core_M=1, per_core_N=per_core_n, fused_activation=None
        ),
    )
    w = weight(device, k, n, w_dtype)
    assert L.matmul_bh_covers(x, w, **args)
    twice(device, make_x, lambda x: L.matmul_bh(x, w, **args), lambda x: ttnn.linear(x, w, **args))


@run_for_blackhole()
def test_matmul_bh_rejects_other_shapes(device, expect_error):
    torch.manual_seed(1)
    x = ttnn.from_torch(
        torch.randn(1, 1, 256, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    args = dict(
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=kernel_config("HiFi2", True),
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 10),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=8,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
    )
    w = weight(device, 4096, 6144, ttnn.bfloat8_b)
    assert not L.matmul_bh_covers(x, w, **args)
    with expect_error(RuntimeError, "llama31_8b.matmul_bh"):
        L.matmul_bh(x, w, **args)


def _prefill_case(device, kind):
    if kind == "minimal":
        batch, m, k, n, x_dtype, fp32, grid = MINIMAL[0]
        args = dict(
            config=ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=1,
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
            ),
            compute_kernel_config=kernel_config("HiFi2", fp32),
        )
        w = weight(device, k, n, ttnn.bfloat8_b)
        return (
            batch,
            m,
            k,
            x_dtype,
            w,
            args,
            L.minimal_matmul_bh_covers,
            L.minimal_matmul_bh,
            ttnn.experimental.minimal_matmul,
        )
    row = PREFILL[0 if kind == "1d" else 4]
    (
        batch,
        m,
        k,
        n,
        x_dtype,
        w_dtype,
        out_dtype,
        fidelity,
        fp32,
        grid,
        in0_block_w,
        sbw,
        per_core_m,
        per_core_n,
        fuse_batch,
    ) = row
    args = dict(
        dtype=out_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=kernel_config(fidelity, fp32),
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=sbw,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=fuse_batch,
        ),
    )
    w = weight(device, k, n, w_dtype)
    return batch, m, k, x_dtype, w, args, L.matmul_bh_covers, L.matmul_bh, ttnn.linear


@run_for_blackhole()
@pytest.mark.parametrize("kind", ["1d", "2d", "minimal"])
def test_matmul_bh_falls_back_when_l1_is_occupied(device, kind, expect_error):
    batch, m, k, x_dtype, w, args, covers, ours, main = _prefill_case(device, kind)
    torch.manual_seed(5)
    x = ttnn.from_torch(
        torch.randn(1, batch, m, k),
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert covers(x, w, **args)
    held = []
    while covers(x, w, **args):
        assert len(held) < 32
        held.append(
            ttnn.from_torch(
                torch.zeros(1, 1, 1024, 4096),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
    with expect_error(RuntimeError, "not enough free L1"):
        ours(x, w, **args)
    for t in held:
        assert torch.equal(ttnn.to_torch(t), torch.zeros(1, 1, 1024, 4096, dtype=torch.bfloat16))
        ttnn.deallocate(t)
    assert covers(x, w, **args)
    same(ours(x, w, **args), main(x, w, **args))


@run_for_blackhole()
def test_prefill_qkv_heads_bh(device):
    def make_x(i):
        torch.manual_seed(1 + i)
        return ttnn.from_torch(
            torch.randn(1, 1, 1024, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = make_x(0)
    args = dict(
        config=ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=1,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 10),
        ),
        compute_kernel_config=kernel_config("HiFi2", True),
    )
    w = weight(device, 4096, 6144, ttnn.bfloat8_b)
    assert L.qkv_heads_bh_covers(x, w, **args, num_heads=32, num_kv_heads=8)
    twice(
        device,
        make_x,
        lambda x: tuple(L.qkv_heads_bh(x, w, **args, num_heads=32, num_kv_heads=8)),
        lambda x: ttnn.experimental.nlp_create_qkv_heads(
            ttnn.experimental.minimal_matmul(x, w, **args),
            num_heads=32,
            num_kv_heads=8,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def interleave_tiles(a, b):
    k, n = a.shape[-2], a.shape[-1]
    return torch.stack([a.reshape(k, n // 32, 32), b.reshape(k, n // 32, 32)], dim=2).reshape(1, 1, k, 2 * n)


@run_for_blackhole()
def test_prefill_swiglu_matmul_bh(device):
    torch.manual_seed(1)
    w1_t = torch.randn(1, 1, 4096, 14336) * 0.02
    w3_t = torch.randn(1, 1, 4096, 14336) * 0.02

    def put(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_width_sharded(device, t.shape[-2], t.shape[-1]),
        )

    w1, w3 = put(w1_t), put(w3_t)

    def make_x(i):
        torch.manual_seed(11 + i)
        return ttnn.from_torch(
            torch.randn(1, 2, 512, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = make_x(0)
    w13 = put(interleave_tiles(w1_t[0, 0], w3_t[0, 0]))
    args = dict(
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=kernel_config("LoFi", False),
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=2,
            per_core_N=64,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        ),
    )
    assert L.matmul_bh_covers(x, w13, dtype=ttnn.bfloat8_b, **args)

    def main(x):
        gate = ttnn.linear(x, w1, dtype=ttnn.bfloat16, **args)
        up = ttnn.linear(x, w3, dtype=ttnn.bfloat16, **args)
        return ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=gate.memory_config(),
        )

    twice(device, make_x, lambda x: L.matmul_bh(x, w13, dtype=ttnn.bfloat8_b, **args), main)
