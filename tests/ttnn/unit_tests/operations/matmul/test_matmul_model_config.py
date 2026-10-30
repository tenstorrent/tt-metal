# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The opt-in matmul config model (ttnn.CONFIG.matmul_model_config, #57884).

Needs a Blackhole device. The tests that turn the model on use the blob the build fetched (the default
ttnn.CONFIG.matmul_model_path), or TT_MATMUL_MODEL_BLOB when set; a missing blob fails them.
"""

import os
import shutil

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.comparison import comp_pcc

last_pick = ttnn.operations.matmul.matmul_model_last_pick
MODEL_BLOB = os.environ.get("TT_MATMUL_MODEL_BLOB") or str(ttnn.CONFIG.matmul_model_path)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="the matmul config model covers Blackhole only")


@pytest.fixture
def model_on():
    assert os.path.isfile(MODEL_BLOB), f"the matmul config model blob is missing: {MODEL_BLOB}"
    with ttnn.manage_config("matmul_model_path", MODEL_BLOB), ttnn.manage_config("matmul_model_config", True):
        yield


def run_matmul(device, a_shape, b_shape, dtype_b, a_mem, out_mem, bias=False, transpose_b=False):
    torch.manual_seed(0)
    a = torch.randn(a_shape).bfloat16()
    b = torch.randn(b_shape).bfloat16()
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem)
    tb = ttnn.from_torch(
        b, dtype=dtype_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bt = b.transpose(-1, -2) if transpose_b else b
    expected = a.float() @ bt.float()
    if bias:
        c = torch.randn(1, bt.shape[-1]).bfloat16()
        tc = ttnn.from_torch(
            c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.linear(ta, tb, bias=tc, transpose_b=transpose_b, memory_config=out_mem)
        expected = expected + c.float()
    else:
        out = ttnn.matmul(ta, tb, transpose_b=transpose_b, memory_config=out_mem)
    return expected, ttnn.to_torch(out)


CASES = [
    # a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b
    (
        (1, 1, 1024, 4096),
        (1, 1, 4096, 4096),
        ttnn.bfloat8_b,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        False,
        False,
    ),
    (
        (1, 1, 32, 2048),
        (1, 1, 2048, 8192),
        ttnn.bfloat8_b,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        False,
        False,
    ),
    ((1, 1, 4096, 256), (1, 1, 256, 1024), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, False, False),
    ((1, 1, 256, 1024), (1, 1, 1024, 512), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, False, False),
    ((8, 128, 64), (8, 64, 128), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, False, False),
    (
        (1, 1, 512, 1024),
        (1, 1, 1024, 2048),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        True,
        False,
    ),
    (
        (1, 1, 512, 1024),
        (1, 1, 2048, 1024),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        False,
        True,
    ),
    # Squares with packer L1 accumulation, where other K blocking than the default's changes PCC (README, "Precision")
    (
        (1, 1, 1024, 1024),
        (1, 1, 1024, 1024),
        ttnn.bfloat16,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        False,
        False,
    ),
    (
        (1, 1, 2048, 2048),
        (1, 1, 2048, 2048),
        ttnn.bfloat8_b,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
        False,
        False,
    ),
]


@pytest.mark.parametrize("a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b", CASES)
def test_model_config_pcc(device, model_on, a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b):
    with ttnn.manage_config("matmul_model_config", False):
        _, default = run_matmul(device, a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b)
    expected, got = run_matmul(device, a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b)
    pick = last_pick()
    assert pick.startswith("model: "), f"the model should choose for this interleaved matmul: {pick}"
    assert_with_pcc(expected, got, 0.99 if dtype_b == ttnn.bfloat8_b else 0.999)
    # The model keeps the default config's K blocking, the one part of a config that changes rounding (README,
    # "Precision"), so its result is the default's bit for bit; a PCC margin would let a precision change through
    assert torch.equal(got, default), (
        f"the result differs from the default's: PCC {comp_pcc(expected, got)[1]:.6f} with the model, "
        f"{comp_pcc(expected, default)[1]:.6f} without ({pick})"
    )


def test_model_picks_for_a_plain_matmul(device, model_on):
    run_matmul(
        device, (1, 1, 1024, 4096), (1, 1, 4096, 4096), ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    )
    assert last_pick().startswith("model: "), last_pick()
    # The second call is a cache hit with the same pick
    first = last_pick()
    run_matmul(
        device, (1, 1, 1024, 4096), (1, 1, 4096, 4096), ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    )
    assert last_pick() == first, f"cache hit changed the pick: {last_pick()}"


def test_model_reloads_when_the_path_changes(device, model_on, tmp_path):
    """A C++ program that turns the flag on before setting matmul_model_path gets the default configs until it sets
    the path, and the model from then on: the path is read on every eligible matmul, not only on the first. A blob
    that appears at a path whose load failed is loaded too."""
    args = (
        device,
        (1, 1, 1024, 4096),
        (1, 1, 4096, 4096),
        ttnn.bfloat8_b,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    late = tmp_path / "matmul_model.blob"
    with ttnn.manage_config("matmul_model_path", str(late)):
        run_matmul(*args)
        assert last_pick() == "default: model not loaded", last_pick()
        shutil.copyfile(MODEL_BLOB, late)
        run_matmul(*args)
        assert last_pick().startswith("model: "), last_pick()
    with ttnn.manage_config("matmul_model_path", "/nonexistent/matmul_model.blob"):
        run_matmul(*args)
        assert last_pick() == "default: model not loaded", last_pick()
    run_matmul(*args)
    assert last_pick().startswith("model: "), last_pick()


def test_model_keeps_default_for_sharded(device, model_on):
    a = torch.randn(1, 1, 256, 1024).bfloat16()
    b = torch.randn(1, 1, 1024, 512).bfloat16()
    mem = ttnn.create_sharded_memory_config(
        (256, 1024), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=False
    )
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tb = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.matmul(ta, tb)
    assert last_pick().startswith("default: "), last_pick()
    assert_with_pcc(a.float() @ b.float(), ttnn.to_torch(out), 0.999)


def test_model_keeps_default_for_broadcast_a(device, model_on):
    """A broadcast over a batched B: the model describes one batch count, so it keeps today's default."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, 32, 64).bfloat16()
    b = torch.randn(1, 4, 64, 64).bfloat16()
    ta = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tb = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.matmul(ta, tb)
    assert last_pick().startswith("default: B batched"), last_pick()
    assert_with_pcc(a.float() @ b.float(), ttnn.to_torch(out), 0.999)


def test_explicit_program_config_is_untouched(device, model_on):
    """A user program config bypasses the model (it isn't consulted, so its last pick doesn't change)."""
    run_matmul(
        device, (1, 1, 1024, 4096), (1, 1, 4096, 4096), ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    )
    before = last_pick()
    a = torch.randn(1, 1, 64, 128).bfloat16()
    b = torch.randn(1, 1, 128, 64).bfloat16()
    ta = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tb = ttnn.from_torch(
        b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(2, 2),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    out = ttnn.matmul(ta, tb, program_config=config)
    assert last_pick() == before, f"the model was consulted: {last_pick()}"
    assert_with_pcc(a.float() @ b.float(), ttnn.to_torch(out), 0.999)


def test_block_float_output_keeps_precision(device, model_on):
    """bfp8 output without packer L1 or fp32 accumulation: partial sums are kept in bfp8 between K blocks, so the
    pick must not split K finely (gpt-oss decode LM head, K = 2880 is 90 tiles)."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, 32, 2880).bfloat16()
    b = (torch.randn(1, 1, 2880, 4096) / 2880**0.5).bfloat16()
    ta = ttnn.from_torch(
        a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tb = ttnn.from_torch(
        b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    out = ttnn.matmul(ta, tb, dtype=ttnn.bfloat8_b, compute_kernel_config=config)
    assert last_pick().startswith("model: "), last_pick()
    assert_with_pcc(ttnn.to_torch(ta).float() @ ttnn.to_torch(tb).float(), ttnn.to_torch(out), 0.99)


def test_broadcast_bias_with_preallocated_output(device, model_on):
    """A bias that broadcasts the matmul to a larger final output is added after the matmul, and the preallocated
    output is the final one, not the matmul's: the model must choose for the matmul's own output, as the device op
    gets it (review of #62197). Same result as with the flag off."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, 32, 64).bfloat16()
    b = torch.randn(1, 1, 64, 64).bfloat16()
    bias = torch.randn(2, 1, 32, 64).bfloat16()

    def run():
        tt = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = tt(torch.zeros(2, 1, 32, 64).bfloat16())
        return ttnn.to_torch(ttnn.linear(tt(a), tt(b), bias=tt(bias), optional_output_tensor=out))

    with ttnn.manage_config("matmul_model_config", False):
        default = run()
    got = run()
    assert last_pick().startswith("model: "), last_pick()
    assert torch.equal(got, default)
    assert_with_pcc(a.float() @ b.float() + bias.float(), got, 0.999)


def test_preallocated_l1_output_is_not_counted_twice(device, model_on):
    """A preallocated L1 output is already outside the free L1 the pick's budget starts from. Here it fills about 55%
    of every L1 bank, so counting it a second time would leave no L1 for any candidate and keep the default."""
    grid = device.compute_with_storage_grid_size()
    free_bank = 1461120  # a free Blackhole L1 bank, as in the gtests
    tiles_per_bank = 14 * 28  # 392 bf16 tiles, 0.55 of the bank
    assert tiles_per_bank * 2048 > 0.5 * free_bank
    M, K, N = 32 * 14 * grid.y, 256, 32 * 28 * grid.x
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K).bfloat16()
    b = torch.randn(1, 1, K, N).bfloat16()
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.from_torch(
        torch.zeros(1, 1, M, N).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.matmul(ta, tb, memory_config=ttnn.L1_MEMORY_CONFIG, optional_output_tensor=out)
    assert last_pick().startswith("model: "), last_pick()
    assert_with_pcc(a.float() @ b.float(), ttnn.to_torch(out), 0.99)


@pytest.mark.parametrize("a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b", CASES[:3])
def test_flag_off_is_the_default(device, a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b):
    """Flag off (the default): the model is never consulted and matmul's results are bit-identical run to run."""
    assert not ttnn.CONFIG.matmul_model_config, "ttnn.CONFIG.matmul_model_config should be off by default"
    before = last_pick()
    _, first = run_matmul(device, a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b)
    _, second = run_matmul(device, a_shape, b_shape, dtype_b, a_mem, out_mem, bias, transpose_b)
    assert last_pick() == before, f"the model was consulted: {last_pick()}"
    assert torch.equal(first, second), "flag-off matmul is not deterministic run to run"
