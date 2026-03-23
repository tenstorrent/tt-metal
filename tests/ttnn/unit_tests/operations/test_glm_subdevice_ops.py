# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GLM-4.7 SubDevice-aware op tests on Galaxy Wormhole TG mesh.

Tests key TTNN ops with a SubDevice manager active, using two layouts:
  1. "Legacy" layout (glm_sub_device): worker cols 1-6, sender cols 0,7
     - Requires sub_device_id + adjusted grid for matmul/linear
  2. "Origin-worker" layout (origin_sub_device): worker cols 0-5, sender cols 6,7
     - Production layout. Matmul grids from (0,0) stay in worker SubDevice.
     - No sub_device_id needed for matmul/linear (just explicit grid <= 6 cols).

Galaxy WH grid per chip: 8 cols (x=0..7) x 9 rows (y=0..8) = 72 cores.
TG mesh = MeshShape(8,4) = 32 devices.

GLM-4.7 Full decode dimensions (TP=8, per-device):
  - Hidden: 5120
  - QKV out: 1792 (12Q + 1KV heads x 128 dim)
  - O-proj in: 1536
  - Batch: 8
  - MoE intermediate: 192 (1536/8 TP)

Usage:
  cd /tt-metal
  python -m pytest tests/ttnn/unit_tests/operations/test_glm_subdevice_ops.py -v -s
"""

import pytest
import torch
import ttnn

# ── GLM-4.7 decode dimensions (TP=8, per device) ────────────────────────────
HIDDEN = 5120
QKV_OUT = 1792      # per device (12Q+1KV heads x 128 dim)
OPROJ_IN = 1536     # per device
BATCH = 8
MOE_INTER = 192     # per device (1536/8 TP)

# ── Galaxy WH grid constants ────────────────────────────────────────────────
GRID_X = 8   # columns 0..7
GRID_Y = 9   # rows 0..8

# Worker grid: cols 1-6, rows 0-8 = 54 cores
WORKER_CORES = ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, GRID_Y - 1))}
)
# Sender cores: cols 0 and 7
SENDER_CORES = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, GRID_Y - 1)),
        ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(7, GRID_Y - 1)),
    }
)

# For sub_device_id matmul: grid_size must account for start_core offset.
# Worker starts at col 1, so compute_with_storage_grid_size.x = num_worker_cols + 1
NUM_WORKER_COLS = 6
GRID_W_OFFSET = NUM_WORKER_COLS + 1  # 7


# ── SubDevice fixture ───────────────────────────────────────────────────────

@pytest.fixture
def glm_sub_device(mesh_device):
    """
    Create and load the GLM-4.7 SubDevice layout (worker cols 1-6, sender cols 0,7).
    Yields mesh_device with SubDevice active. Tears down on exit.
    """
    worker_sd = ttnn.SubDevice([WORKER_CORES])
    sender_sd = ttnn.SubDevice([SENDER_CORES])
    manager = mesh_device.create_sub_device_manager([worker_sd, sender_sd], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    yield mesh_device

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(manager)


# ── Helpers ──────────────────────────────────────────────────────────────────

def to_tt(tensor, mesh_device, dtype=ttnn.bfloat16):
    """Convert torch tensor to TTNN on mesh, DRAM interleaved, tile layout."""
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def from_tt(tensor, mesh_device):
    """Convert TTNN tensor back to torch, concatenated across mesh."""
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


def check_pcc(result_torch, expected_torch, num_devices, min_pcc=0.99, label="op"):
    """Check PCC (cosine similarity) per device replica."""
    for i in range(num_devices):
        actual = result_torch[i : i + 1].flatten().float()
        expected = expected_torch.flatten().float()
        pcc = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
        assert pcc > min_pcc, f"Device {i}: {label} PCC {pcc:.4f} < {min_pcc}"


def check_allclose(result_torch, expected_torch, num_devices, atol=0.1, label="op"):
    """Check allclose per device replica."""
    for i in range(num_devices):
        actual = result_torch[i : i + 1]
        assert torch.allclose(actual, expected_torch, atol=atol), (
            f"Device {i}: {label} mismatch "
            f"(max diff={torch.max(torch.abs(actual - expected_torch)):.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: ttnn.add — DRAM interleaved, GLM residual add shape
# ══════════════════════════════════════════════════════════════════════════════

def test_add_with_subdevice(glm_sub_device):
    """
    ttnn.add on DRAM-interleaved tensors with SubDevice active.
    Shape mirrors GLM-4.7 residual add: [1, 1, BATCH, HIDDEN].
    Eltwise ops don't multicast — should work with any SubDevice layout.
    """
    mesh_device = glm_sub_device
    shape = [1, 1, BATCH, HIDDEN]

    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = a_torch + b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    c_tt = ttnn.add(a_tt, b_tt)
    c_torch = from_tt(c_tt, mesh_device)

    check_allclose(c_torch, expected, mesh_device.get_num_devices(), atol=0.1, label="add")


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: ttnn.multiply — DRAM interleaved, GLM gating shape
# ══════════════════════════════════════════════════════════════════════════════

def test_multiply_with_subdevice(glm_sub_device):
    """
    ttnn.multiply on DRAM-interleaved tensors with SubDevice active.
    Shape mirrors GLM-4.7 SiLU gating: [1, 1, BATCH, MOE_INTER].
    """
    mesh_device = glm_sub_device
    shape = [1, 1, BATCH, MOE_INTER]

    # MOE_INTER=192 is not tile-aligned (192/32=6 tiles, OK).
    # But BATCH=8 is < 32 tile height. Pad to tile boundary for TILE_LAYOUT.
    # Actually ttnn.from_torch handles padding internally for TILE_LAYOUT.
    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = a_torch * b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    c_tt = ttnn.multiply(a_tt, b_tt)
    c_torch = from_tt(c_tt, mesh_device)

    check_allclose(c_torch, expected, mesh_device.get_num_devices(), atol=0.1, label="multiply")


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: ttnn.rms_norm — DRAM interleaved, GLM hidden dim
# ══════════════════════════════════════════════════════════════════════════════

def test_rms_norm_with_subdevice(glm_sub_device):
    """
    ttnn.rms_norm on DRAM-interleaved tensors with SubDevice active.
    Shape: [1, 1, BATCH, HIDDEN] with weight [HIDDEN].
    Mirrors GLM-4.7 pre-attention and post-attention RMSNorm.
    """
    mesh_device = glm_sub_device
    eps = 1e-5

    x_torch = torch.randn(1, 1, BATCH, HIDDEN, dtype=torch.bfloat16)
    w_torch = torch.ones(HIDDEN, dtype=torch.bfloat16)

    # CPU reference
    x_float = x_torch.float()
    rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = (x_float / rms * w_torch.float()).to(torch.bfloat16)

    x_tt = to_tt(x_torch, mesh_device)
    w_tt = to_tt(w_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0), mesh_device)  # [1,1,1,HIDDEN]

    out_tt = ttnn.rms_norm(x_tt, epsilon=eps, weight=w_tt)
    out_torch = from_tt(out_tt, mesh_device)

    check_pcc(out_torch, expected, mesh_device.get_num_devices(), min_pcc=0.98, label="rms_norm")


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: ttnn.linear (QKV projection) — [1,1,B,HIDDEN] x [HIDDEN,QKV_OUT]
# ══════════════════════════════════════════════════════════════════════════════

def test_linear_qkv_with_subdevice(glm_sub_device):
    """
    ttnn.linear for QKV projection with SubDevice active.
    Input: [1, 1, BATCH, HIDDEN] = [1, 1, 8, 5120]
    Weight: [1, 1, HIDDEN, QKV_OUT] = [1, 1, 5120, 1792]

    Uses sub_device_id=SubDeviceId(0) so the matmul factory offsets
    the grid to start from worker SubDevice bounding box (col 1).

    compute_with_storage_grid_size=(7,1): 7 is the total grid extent,
    giving 6 usable worker columns (7 - start_col_1 = 6).
    per_core_N = ceil(QKV_OUT/32) / 6 = 56/6 — must be integer.
    QKV_OUT=1792 → 1792/32=56 tiles. 56/6 is not integer → need different grid.
    Use (7,2) → 12 worker cores. 56/12 is not integer either.
    Use (7,8) → 48 worker cores. 56/48 not integer.
    Simplest: (7,1) with 6 cores. Need N_tiles divisible by 6.
    Pad QKV_OUT to 1920 (60 tiles, 60/6=10). Or use fewer cores.
    Use 4 cores: (5,1) → 4 worker cols (5-1=4). 56/4=14. Works!
    Or use 7 cores: (8,1) → 7 worker cols (8-1=7). 56/7=8. Works!
    """
    mesh_device = glm_sub_device
    M = BATCH          # 8
    K = HIDDEN         # 5120
    N = QKV_OUT        # 1792

    # Tile counts
    M_tiles = (M + 31) // 32   # 1 (8→padded to 32)
    K_tiles = K // 32           # 160
    N_tiles = (N + 31) // 32   # 56

    # Worker cols 1-6 = 6 usable cols. Grid must NOT extend to col 7 (sender).
    # N_tiles=56. Need N_tiles divisible by num_cores.
    # 56 = 2^3 * 7. Use 4 cores: grid (5,1) → 5-1=4 usable. 56/4=14.
    NUM_CORES = 4
    GRID_X_SIZE = NUM_CORES + 1  # 5 (cols 1-4 used)

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = a_torch @ b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    # in0_block_w must be small enough to fit in L1 (~1.5MB per core).
    # Full K_tiles=160 overflows. Use 8 tiles (256 elements) per block.
    IN0_BLOCK_W = 8

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(GRID_X_SIZE, 1),
        in0_block_w=IN0_BLOCK_W,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M_tiles,
        per_core_N=N_tiles // NUM_CORES,  # 56/4=14
        fuse_batch=True,
        mcast_in0=True,
    )

    c_tt = ttnn.linear(
        a_tt, b_tt,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_device_id=ttnn.SubDeviceId(0),
    )
    c_torch = from_tt(c_tt, mesh_device)

    check_pcc(c_torch, expected, mesh_device.get_num_devices(), min_pcc=0.98,
              label="linear_qkv")


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: ttnn.linear (O-projection) — [1,1,B,OPROJ_IN] x [OPROJ_IN,HIDDEN]
# ══════════════════════════════════════════════════════════════════════════════

def test_linear_oproj_with_subdevice(glm_sub_device):
    """
    ttnn.linear for output projection with SubDevice active.
    Input: [1, 1, BATCH, OPROJ_IN] = [1, 1, 8, 1536]
    Weight: [1, 1, OPROJ_IN, HIDDEN] = [1, 1, 1536, 5120]

    HIDDEN=5120 → 160 tiles. 160/7 is not integer.
    Use 5 worker cores: grid (6,1) → 6-1=5. 160/5=32. Works!
    Or use 4: (5,1) → 4. 160/4=40. Works!
    """
    mesh_device = glm_sub_device
    M = BATCH          # 8
    K = OPROJ_IN       # 1536
    N = HIDDEN         # 5120

    M_tiles = (M + 31) // 32   # 1
    K_tiles = K // 32           # 48
    N_tiles = N // 32           # 160

    # Use 5 worker cores: grid (6,1) → 6-1=5, 160/5=32
    NUM_CORES = 5
    GRID_X_SIZE = NUM_CORES + 1  # 6

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = a_torch @ b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    # in0_block_w must fit in L1. K_tiles=48 overflows. Use 4 tiles per block.
    IN0_BLOCK_W = 4

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(GRID_X_SIZE, 1),
        in0_block_w=IN0_BLOCK_W,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M_tiles,
        per_core_N=N_tiles // NUM_CORES,  # 160/5=32
        fuse_batch=True,
        mcast_in0=True,
    )

    c_tt = ttnn.linear(
        a_tt, b_tt,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sub_device_id=ttnn.SubDeviceId(0),
    )
    c_torch = from_tt(c_tt, mesh_device)

    check_pcc(c_torch, expected, mesh_device.get_num_devices(), min_pcc=0.98,
              label="linear_oproj")


# ══════════════════════════════════════════════════════════════════════════════
# Tests 6-10: ORIGIN-WORKER layout (worker=cols 0-5, sender=cols 6-7)
#
# This layout includes (0,0) in the worker SubDevice, so matmul auto-config
# works WITHOUT explicit program_config or sub_device_id.
# If these pass, the entire SubDevice migration is a 2-line change in
# prefetcher_setup.py: move sender from cols {0,7} to cols {6,7}.
# ══════════════════════════════════════════════════════════════════════════════

# Origin-worker SubDevice layout
ORIGIN_WORKER_CORES = ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, GRID_Y - 1))}
)
ORIGIN_SENDER_CORES = ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, GRID_Y - 1))}
)


@pytest.fixture
def origin_sub_device(mesh_device):
    """
    Origin-worker SubDevice layout: worker=cols 0-5, sender=cols 6-7.
    Worker includes (0,0) — matmul auto-config should just work.
    """
    worker_sd = ttnn.SubDevice([ORIGIN_WORKER_CORES])
    sender_sd = ttnn.SubDevice([ORIGIN_SENDER_CORES])
    manager = mesh_device.create_sub_device_manager([worker_sd, sender_sd], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    yield mesh_device

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(manager)


def test_add_origin_worker(origin_sub_device):
    """ttnn.add with origin-worker layout, GLM residual shape."""
    mesh_device = origin_sub_device
    shape = [1, 1, BATCH, HIDDEN]

    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = a_torch + b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    c_tt = ttnn.add(a_tt, b_tt)
    c_torch = from_tt(c_tt, mesh_device)

    check_allclose(c_torch, expected, mesh_device.get_num_devices(), atol=0.1, label="add_origin")


def test_multiply_origin_worker(origin_sub_device):
    """ttnn.multiply with origin-worker layout, GLM gating shape."""
    mesh_device = origin_sub_device
    shape = [1, 1, BATCH, MOE_INTER]

    a_torch = torch.randn(shape, dtype=torch.bfloat16)
    b_torch = torch.randn(shape, dtype=torch.bfloat16)
    expected = a_torch * b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    c_tt = ttnn.multiply(a_tt, b_tt)
    c_torch = from_tt(c_tt, mesh_device)

    check_allclose(c_torch, expected, mesh_device.get_num_devices(), atol=0.1, label="mul_origin")


def test_rms_norm_origin_worker(origin_sub_device):
    """ttnn.rms_norm with origin-worker layout, GLM hidden dim."""
    mesh_device = origin_sub_device
    eps = 1e-5

    x_torch = torch.randn(1, 1, BATCH, HIDDEN, dtype=torch.bfloat16)
    w_torch = torch.ones(HIDDEN, dtype=torch.bfloat16)

    x_float = x_torch.float()
    rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = (x_float / rms * w_torch.float()).to(torch.bfloat16)

    x_tt = to_tt(x_torch, mesh_device)
    w_tt = to_tt(w_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0), mesh_device)

    out_tt = ttnn.rms_norm(x_tt, epsilon=eps, weight=w_tt)
    out_torch = from_tt(out_tt, mesh_device)

    check_pcc(out_torch, expected, mesh_device.get_num_devices(), min_pcc=0.98,
              label="rms_norm_origin")


def test_linear_qkv_origin_worker_explicit_grid(origin_sub_device):
    """
    ttnn.linear for QKV with origin-worker layout and explicit grid within cols 0-5.
    No sub_device_id needed — grid (6,1) starts from (0,0) and stays in worker.
    """
    mesh_device = origin_sub_device
    M, K, N = BATCH, HIDDEN, QKV_OUT  # 8, 5120, 1792

    M_tiles = (M + 31) // 32   # 1
    N_tiles = (N + 31) // 32   # 56

    # 6 cols available (0-5). 56/4=14 (use 4 cores to divide evenly).
    NUM_CORES = 4
    IN0_BLOCK_W = 8

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = a_torch @ b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(NUM_CORES, 1),
        in0_block_w=IN0_BLOCK_W,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M_tiles,
        per_core_N=N_tiles // NUM_CORES,  # 56/4=14
        fuse_batch=True,
        mcast_in0=True,
    )

    # No sub_device_id needed — grid starts at (0,0) and fits in worker cols 0-3
    c_tt = ttnn.linear(
        a_tt, b_tt,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c_torch = from_tt(c_tt, mesh_device)

    check_pcc(c_torch, expected, mesh_device.get_num_devices(), min_pcc=0.98,
              label="linear_qkv_origin")


def test_linear_oproj_origin_worker_explicit_grid(origin_sub_device):
    """
    ttnn.linear for O-proj with origin-worker layout and explicit grid within cols 0-5.
    No sub_device_id needed.
    """
    mesh_device = origin_sub_device
    M, K, N = BATCH, OPROJ_IN, HIDDEN  # 8, 1536, 5120

    M_tiles = (M + 31) // 32   # 1
    N_tiles = N // 32           # 160

    # 6 cols available (0-5). 160/5=32 (use 5 cores).
    NUM_CORES = 5
    IN0_BLOCK_W = 4

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = a_torch @ b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(NUM_CORES, 1),
        in0_block_w=IN0_BLOCK_W,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=M_tiles,
        per_core_N=N_tiles // NUM_CORES,  # 160/5=32
        fuse_batch=True,
        mcast_in0=True,
    )

    c_tt = ttnn.linear(
        a_tt, b_tt,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c_torch = from_tt(c_tt, mesh_device)

    check_pcc(c_torch, expected, mesh_device.get_num_devices(), min_pcc=0.98,
              label="linear_oproj_origin")


def test_matmul_qkv_origin_worker_autoconfig(origin_sub_device):
    """
    ttnn.matmul with AUTO-CONFIG on origin-worker layout.
    Tests whether auto-config picks a grid that stays within the worker SubDevice.
    If this fails, explicit program_config is always required with SubDevices.
    """
    mesh_device = origin_sub_device
    # Use small dimensions to keep auto-config grid small
    M, K, N = 32, 128, 128

    a_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = a_torch @ b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    # AUTO-CONFIG: no program_config, no sub_device_id
    c_tt = ttnn.matmul(a_tt, b_tt)
    c_torch = from_tt(c_tt, mesh_device)

    check_pcc(c_torch, expected, mesh_device.get_num_devices(), min_pcc=0.99,
              label="matmul_autoconfig_small")


def test_matmul_cols_0_5_worker_cols_6_7_sender(mesh_device):
    """Worker=cols 0-5 (includes origin), sender=cols 6-7.

    GLM QKV dimensions: [1,1,8,5120] x [5120,1792] with explicit program_config
    constraining grid to 6 cols (worker SubDevice). NO sub_device_id needed
    because worker starts at (0,0).

    Self-contained test (creates its own SubDevice manager, no fixture dependency).

    NOTE: Auto-config matmul ALWAYS uses full device grid (8 cols), which extends
    into sender cols 6-7. Explicit program_config is REQUIRED when SubDevices are
    active, regardless of layout.
    """
    worker = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))])
    sender = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(7, 8))])

    mgr = mesh_device.create_sub_device_manager(
        [ttnn.SubDevice([worker]), ttnn.SubDevice([sender])], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    try:
        # GLM QKV dimensions: [1,1,8,5120] x [5120,1792]
        x = ttnn.from_torch(
            torch.randn(1, 1, 8, 5120).bfloat16(), device=mesh_device,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
        w = ttnn.from_torch(
            torch.randn(1, 1, 5120, 1792).bfloat16(), device=mesh_device,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))

        # Explicit program_config: 4 cores in row 0 within worker cols 0-5.
        # M=8 (pad to 32=1 tile), K=5120 (160 tiles), N=1792 (56 tiles).
        # 56/4=14 tiles per core, in0_block_w=8 to fit L1.
        # NO sub_device_id needed — worker starts at (0,0), grid (4,1) stays in cols 0-3.
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(4, 1),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=7,
            per_core_M=1,
            per_core_N=14,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        out = ttnn.matmul(x, w, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # ttnn preserves logical shape [1,1,8,1792] (not tile-padded [1,1,32,1792])
        assert list(out.shape) == [1, 1, 8, 1792], f"Unexpected shape: {out.shape}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(mgr)


# ── Task 4: Easy op migration tests (origin-worker layout) ─────────────────
# These test the 5 target ops for SubDevice migration: add, rms_norm, clone,
# typecast, embedding. All use the production layout (worker=cols 0-5).

def test_add_with_sub_core_grids(origin_sub_device):
    """ttnn.add with explicit sub_core_grids restricting to worker grid.

    Verifies the defensive pattern used in decoder_layer_tt.py decode path:
    sub_core_grids=worker_crs ensures add never touches sender cores.
    """
    mesh_device = origin_sub_device
    worker_scg = ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))
    ])

    a_torch = torch.randn(1, 1, BATCH, HIDDEN, dtype=torch.bfloat16)
    b_torch = torch.randn(1, 1, BATCH, HIDDEN, dtype=torch.bfloat16)
    expected = a_torch + b_torch

    a_tt = to_tt(a_torch, mesh_device)
    b_tt = to_tt(b_torch, mesh_device)

    # With sub_core_grids: restricts compute to worker cols 0-5
    c_tt = ttnn.add(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    sub_core_grids=worker_scg)
    c_torch = from_tt(c_tt, mesh_device)
    check_allclose(c_torch, expected, mesh_device.get_num_devices(), atol=0.1,
                   label="add_sub_core_grids")


@pytest.mark.xfail(reason="ttnn.clone has no sub_core_grids param; auto-grid spans sender cols",
                    strict=True)
def test_clone_origin_worker(origin_sub_device):
    """ttnn.clone on origin-worker layout (no sub_core_grids param available).

    Clone is used for MTP hidden state and embedding buffer copy in decode.
    XFAIL: clone uses full device grid for DRAM interleaved, spanning into
    sender cols 6-7. Needs sub_core_grids support added to clone op.
    """
    mesh_device = origin_sub_device
    x_torch = torch.randn(1, 1, BATCH, HIDDEN, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, mesh_device)

    cloned = ttnn.clone(x_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cloned_torch = from_tt(cloned, mesh_device)
    check_allclose(cloned_torch, x_torch, mesh_device.get_num_devices(), atol=0.01,
                   label="clone")


def test_typecast_origin_worker(origin_sub_device):
    """ttnn.typecast BF16->BF8_B with sub_core_grids on origin-worker layout.

    Typecast is a unary op — supports sub_core_grids to restrict to worker grid.
    Used in KV cache dtype conversion.
    """
    mesh_device = origin_sub_device
    worker_scg = ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))
    ])

    x_torch = torch.randn(1, 1, BATCH, HIDDEN, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, mesh_device)

    # Typecast BF16 -> BF8_B with sub_core_grids restricting to worker
    casted = ttnn.typecast(x_tt, dtype=ttnn.bfloat8_b, sub_core_grids=worker_scg)
    assert casted.dtype == ttnn.bfloat8_b, f"Expected BF8_B, got {casted.dtype}"

    # Cast back and check approximate match (BF8_B is lossy)
    back = ttnn.typecast(casted, dtype=ttnn.bfloat16, sub_core_grids=worker_scg)
    back_torch = from_tt(back, mesh_device)
    check_pcc(back_torch, x_torch, mesh_device.get_num_devices(), min_pcc=0.98,
              label="typecast_roundtrip")


def test_embedding_origin_worker(origin_sub_device):
    """ttnn.embedding on origin-worker layout (no sub_core_grids param).

    Used for token lookup in decode trace. Tests that it works under SubDevice.
    """
    mesh_device = origin_sub_device
    vocab_size = 128256
    embed_dim = HIDDEN

    # Small embedding table for test (full GLM embedding is 128256 x 5120)
    # Use smaller vocab to keep test fast
    small_vocab = 1024
    weight_torch = torch.randn(small_vocab, embed_dim, dtype=torch.bfloat16)
    ids_torch = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])  # batch=8 tokens

    weight_tt = ttnn.from_torch(
        weight_torch, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    ids_tt = ttnn.from_torch(
        ids_torch, dtype=ttnn.uint32, device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))

    result = ttnn.embedding(ids_tt, weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert list(result.shape)[-1] == embed_dim, f"Unexpected embed dim: {result.shape}"
