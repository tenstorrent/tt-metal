# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fast standalone repro of the lm_head wide-matmul HANG on the 3 MB-L1 Quasar variant.

The full `test_llama_e2e.py` takes ~2h, and it segfaults deep in prefill at the lm_head
(`lm_head_1d.py:133` -> the monkeypatched `_mm`). The crashing op is ONE lm_head vocab-chunk
matmul: `x[1,1,32,2048] @ w[2048,8192] -> [1,1,32,8192]` (the vocab 128256 = 15*8192 + 5376 is
projected in 8192-wide chunks), run as `matmul_multi_core_reuse_mcast_1d_in0`, grid 2x1,
`mcast_in0=1`, `per_core_N=128`, `in0_block_w=4`. On the 3 MB variant both cores stall forever
(`Not done phys cores: 1-0 0-0`), then the Python faulthandler fires a segfault.

This test runs just that matmul (a few seconds) under a sweep of candidate program configs so the
hang can be bisected WITHOUT the e2e. Each config is a separate parametrized case, so you can run
them one at a time (`-k`) — a hang in one does not block the others across separate invocations.

What each outcome tells us:
  * `e2e_2core_ibw4` hangs, `single_core_ibw1` PASSES
      -> the 2-core MCAST is the problem (its semaphore lands in the non-existent [3MB,4MB) L1 gap);
         fix = route the wide lm_head matmul through a single core (no mcast) on the small-L1 variant.
  * `shrink_2core_ibw2` PASSES
      -> it was a CB-footprint overflow after all; the in0_block_w shrink in _mm is the fix.
  * ALL cases (incl. single_core) hang
      -> not config-avoidable: the mcast/kernel-config semaphore is placed at a fixed HAL address in
         the [3MB,4MB) gap. Fix belongs in the variant's SoC descriptor / HAL L1 size (KERNEL_CONFIG,
         get_dev_size(TENSIX, BASE/DEFAULT_UNRESERVED)), an emulator/runtime-owner change.

Run on the 3 MB variant (the physical part is where the gap exists — a 4 MB device will NOT reproduce
even with worker_l1_size forced to 3 MB, because [3MB,4MB) physically exists there):

    LLAMA_QSR_WORKER_L1_SIZE=3145728 MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_lm_head_matmul_3mb.py

    # bisect a single config:
    LLAMA_QSR_WORKER_L1_SIZE=3145728 MESH_DEVICE=N150 \
        pytest ...::test_lm_head_chunk_matmul_3mb -k single_core_ibw1
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

# One lm_head vocab-chunk matmul, exactly as the e2e drives it through the monkeypatched _mm.
M, K, N = 32, 2048, 8192

# (id, grid_x, in0_block_w). grid_x=2 -> the e2e 1D-mcast (per_core_N = 256/2 = 128); grid_x=1 -> single
# core (per_core_N = 256), which removes the cross-core multicast and its semaphore. in0_block_w sets the
# weights CB size = in0_block_w * per_core_N * 2 tiles (double-buffered); it must divide K/32 = 64.
_CONFIGS = [
    ("e2e_2core_ibw4", 2, 4),  # the exact e2e default -> the repro (expected to hang on 3 MB)
    ("shrink_2core_ibw2", 2, 2),  # the current small-L1 footprint-shrink fix in _mm
    ("single_core_ibw2", 1, 2),  # no mcast
    ("single_core_ibw1", 1, 1),  # no mcast, smallest weights CB
]


@pytest.fixture()
def qsr_device():
    """Open a single-device mesh, honoring LLAMA_QSR_WORKER_L1_SIZE (the 3 MB variant opt-in) so the
    device is configured exactly as the e2e configures it."""
    wl1 = os.environ.get("LLAMA_QSR_WORKER_L1_SIZE", "").strip()
    kwargs = {}
    if wl1:
        kwargs["worker_l1_size"] = int(wl1, 0)
    try:
        num_pcie = ttnn.get_num_pcie_devices()
    except Exception as e:  # pragma: no cover - environment probe
        pytest.skip(f"cannot query TT devices: {e}")
    if isinstance(num_pcie, int) and num_pcie == 0:
        pytest.skip("no TT devices detected")
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **kwargs)
    try:
        yield dev
    finally:
        ttnn.close_mesh_device(dev)


def _dram_bf16(t, dev):
    """ROW_MAJOR upload + Quasar-safe tilize to DRAM (from_torch(TILE) faults on Quasar wide-short)."""
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _out_subblock_w(per_core_n):
    osw = 1
    while osw * 2 <= min(per_core_n, 4) and per_core_n % (osw * 2) == 0:
        osw *= 2
    return osw


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("cfg_id, grid_x, ibw", _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_lm_head_chunk_matmul_3mb(qsr_device, cfg_id, grid_x, ibw):
    dev = qsr_device
    grid = dev.compute_with_storage_grid_size()
    if int(grid.x) < grid_x:
        pytest.skip(f"device compute grid x={grid.x} < required {grid_x}")

    nt = N // 32
    per_core_n = max((nt + grid_x - 1) // grid_x, 1)
    weights_cb_tiles = ibw * per_core_n * 2  # double-buffered
    logger.info(
        f"[lm_head-3mb][{cfg_id}] grid={grid_x}x1 in0_block_w={ibw} per_core_N={per_core_n} "
        f"weights_CB~={weights_cb_tiles} tiles (~{weights_cb_tiles * 2048 / 1024 / 1024:.2f} MB); starting"
    )

    torch.manual_seed(0)
    x = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    w = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    xt = _dram_bf16(x, dev)
    wt = _dram_bf16(w, dev)

    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, 1),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=_out_subblock_w(per_core_n),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=(grid_x > 1),  # single core -> no multicast (and no cross-core semaphore)
    )

    out = ttnn.matmul(xt, wt, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)  # if the device hangs, it hangs HERE (the e2e failure point)
    o = ttnn.to_torch(out)

    ref = x.float().reshape(M, K) @ w.float().reshape(K, N)
    pcc = _pcc(o, ref)
    logger.info(f"[lm_head-3mb][{cfg_id}] DONE finite={torch.isfinite(o).all().item()} PCC={pcc:.4f}")
    assert tuple(o.shape) == (1, 1, M, N), f"{cfg_id}: unexpected shape {tuple(o.shape)}"
    assert pcc > 0.99, f"{cfg_id}: PCC {pcc}"


def _hold_l1(dev, mb):
    """Allocate and return a resident L1 tensor of ~`mb` MB per core, to simulate the e2e's L1 occupancy
    at the lm_head. ROW_MAJOR L1 interleaved (no tilize, avoids the wide-short from_torch(TILE) fault)."""
    if mb <= 0:
        return None
    # bf16 L1 interleaved [1,1,32,W]; 32*W*2 bytes spread over NUM_L1_BANKS=2 cores -> ~mb MB/core at W≈mb*64*2/... keep simple: total bytes = mb*2*1MB (both banks), W = mb*2*1024*1024/(32*2).
    total_bytes = int(mb * 2 * 1024 * 1024)
    w_cols = max(32, (total_bytes // (32 * 2)) // 32 * 32)
    t = torch.zeros(1, 1, 32, w_cols, dtype=torch.bfloat16)
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
    )


# The op PASSES in isolation (empty L1), but the e2e hangs it late in prefill when resident tensors occupy
# L1. This sweep HOLDS `pressure_mb` of L1 while running the lm_head matmul, to reproduce that context and
# measure the footprint margin: ibw=4 CBs ≈2.5 MB, ibw=2 ≈1.5 MB, against ~2.83 MB allocatable. If ibw=4
# hangs at a pressure where ibw=2 still passes, the fix is the in0_block_w shrink (apply it unconditionally
# on small L1 — it was reverted because the empty-L1 standalone couldn't show the margin matters).
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("ibw", [4, 2], ids=["ibw4", "ibw2"])
@pytest.mark.parametrize("pressure_mb", [0.0, 0.5, 1.0, 1.5], ids=lambda p: f"p{p}mb")
def test_lm_head_matmul_under_l1_pressure(qsr_device, pressure_mb, ibw):
    dev = qsr_device
    grid = dev.compute_with_storage_grid_size()
    grid_x = min(int(grid.x), 2)
    nt = N // 32
    per_core_n = max((nt + grid_x - 1) // grid_x, 1)

    held = _hold_l1(dev, pressure_mb)  # kept alive for the duration -> L1 stays occupied
    try:
        torch.manual_seed(0)
        x = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        w = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        xt = _dram_bf16(x, dev)
        wt = _dram_bf16(w, dev)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, 1),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=_out_subblock_w(per_core_n),
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=(grid_x > 1),
        )
        logger.info(
            f"[lm_head-3mb-pressure] held={pressure_mb}MB/core ibw={ibw} per_core_N={per_core_n} "
            f"weights_CB~={ibw * per_core_n * 2} tiles; starting matmul"
        )
        out = ttnn.matmul(xt, wt, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)  # hangs HERE if the held L1 + CBs overflow / trip the e2e failure
        o = ttnn.to_torch(out)
    finally:
        if held is not None:
            ttnn.deallocate(held)

    ref = x.float().reshape(M, K) @ w.float().reshape(K, N)
    pcc = _pcc(o, ref)
    logger.info(f"[lm_head-3mb-pressure] held={pressure_mb}MB ibw={ibw} DONE PCC={pcc:.4f}")
    assert pcc > 0.99, f"pressure={pressure_mb}MB ibw={ibw}: PCC {pcc}"
