# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Smoke test for ``DramCorePrefetcher`` (Python wrapper) without HF weights.

Validates the class's lifecycle — ``init`` → callback-registered ``insert_tensor`` →
``prefetch`` → ``run`` (builds GCB, starts DRISC, queues request) → ``ttnn.linear``
consuming via ``global_cb`` → ``teardown`` — against a synthetic Llama-3.2-3B-shaped
weight + activation. Compares to ``torch.matmul``.

Complements ``test_dram_core_prefetcher_3b.py`` which exercises the same path with
real HF weights through ``MLP``/``Attention``.
"""

import math
import os
import zlib

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.tt_transformers.tt.common import Mode
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

pytestmark = [
    run_for_blackhole("DRAM-core prefetcher requires Blackhole"),
    pytest.mark.skipif(
        os.environ.get("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0") != "1",
        reason="TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES not set",
    ),
]


def _round_up(n, m):
    return ((n + m - 1) // m) * m


@torch.no_grad()
@pytest.mark.parametrize("recv_per_bank", [1, 2, 4])
def test_dram_core_prefetcher_class_smoke(device, recv_per_bank):
    """Drive a single 1D mcast matmul through ``DramCorePrefetcher`` end-to-end.

    Shape mirrors Llama-3.2-3B FF1: K=3072, N=8192/banks/recv_per_bank.
    HF_MODEL is set programmatically so the divisibility check passes; no weights
    are loaded.
    """
    # Set HF_MODEL so DramCorePrefetcher's support check finds a verified config.
    os.environ["HF_MODEL"] = "Llama-3.2-3B"
    # Force the env flag the factory checks.
    os.environ["TT_METAL_USE_DRAM_CORE_PREFETCHER"] = "1"

    from models.tt_transformers.tt.prefetcher import make_prefetcher

    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * recv_per_bank
    TILE = ttnn.TILE_SIZE

    # Small synthetic shape — this is a wiring smoke test, not a perf-target shape.
    # Sized so the GCB fits the remote-CB page cap (~2 MB) at every ring size we sweep.
    # Match the shapes covered by tests/.../test_prefetcher_BH_dram_core_large.py.
    k_tiles_per_shard = 2 if recv_per_bank >= 4 else 4
    n_tiles_per_recv = 1 if recv_per_bank == 1 else 2
    K = k_tiles_per_shard * ring_size * TILE
    N = ring_size * n_tiles_per_recv * TILE
    M = TILE
    dtype = ttnn.bfloat8_b
    tile_bytes = 1088

    # ---- Build a synthetic DRAM-sharded weight ----
    torch.manual_seed(zlib.crc32(f"smoke_{recv_per_bank}".encode()))
    pt_weight = torch.randn(1, 1, K, N)
    dram_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [K, N // num_dram_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_weight = ttnn.as_tensor(
        pt_weight, device=device, dtype=dtype, memory_config=weight_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # ---- Build the activation: width-sharded over the receiver rectangle ----
    receiver_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, recv_per_bank - 1))}
    )
    pt_act = torch.randn(1, 1, M, K)
    K_per_shard = _round_up(math.ceil(K / ring_size), TILE)
    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, K_per_shard),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_act = ttnn.from_torch(
        pt_act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=act_mem_config
    )

    # ---- Program config matches what get_mlp_ff1_3_prg_config would build ----
    out_block_w = N // ring_size // TILE
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(num_dram_banks, recv_per_bank),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=M // TILE,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=recv_per_bank,
        untilize_out=False,
    )

    # ---- Construct DramCorePrefetcher, drive its lifecycle ----
    prefetcher = make_prefetcher(device, num_tensors=1, num_layers=1, num_receiver_cores=recv_per_bank)
    assert prefetcher.__class__.__name__ == "DramCorePrefetcher", (
        f"Expected DramCorePrefetcher but got {prefetcher.__class__.__name__}; "
        "check TT_METAL_USE_DRAM_CORE_PREFETCHER=1 and TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1."
    )
    prefetcher.init(Mode.DECODE)
    prefetcher.register_callback(lambda: prefetcher.insert_tensor(tt_weight, program_config=program_config))
    prefetcher.prefetch()
    prefetcher.run()

    # ---- Matmul consumes the queued weight via global_cb ----
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, N // ring_size),
        core_grid=receiver_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    tt_out = ttnn.linear(
        tt_act,
        tt_weight,
        program_config=program_config,
        memory_config=output_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        global_cb=prefetcher.global_cb,
    )
    prefetcher.teardown()

    out_torch = ttnn.to_torch(tt_out)
    expected = pt_act.float() @ pt_weight.float()
    pcc_threshold = 0.99
    passing, msg = comp_pcc(expected, out_torch, pcc_threshold)
    logger.info(f"[smoke recv_per_bank={recv_per_bank}] {msg}")
    assert passing, f"PCC failed: {msg}"
