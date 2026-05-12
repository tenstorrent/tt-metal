# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for models/demos/qwen3_6_galaxy/tt/llama_rope.py.

Tests are written RED-first (before implementation exists) and should fail with
ModuleNotFoundError when the implementation doesn't exist.

Run all hardware tests (needs 32-chip BH GLX mesh):
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_llama_rope.py -x -s

All tests require hardware (8×4 BH GLX mesh).
"""

import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEAD_DIM = 256
_PARTIAL_ROTARY_FACTOR = 0.25
_ROTARY_DIM = int(_HEAD_DIM * _PARTIAL_ROTARY_FACTOR)  # 64
_MROPE_SECTION = [11, 11, 10]
_ROPE_THETA = 10_000_000.0
_MAX_SEQ_LEN = 64
_BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Fixture: full 8×4 BH GLX mesh with FABRIC_1D_RING
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the full 8×4 fabric mesh. Fabric init is mandatory before open."""
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# CPU reference helper: partial RoPE application
# ---------------------------------------------------------------------------


def _apply_partial_rope_cpu(q_or_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply partial RoPE to q or k.

    Args:
        q_or_k: [B, n_heads, T, head_dim]
        cos:    [1, 1, T, rotary_dim] or [1, T, rotary_dim]
        sin:    [1, 1, T, rotary_dim] or [1, T, rotary_dim]

    Returns:
        [B, n_heads, T, head_dim] with first rotary_dim dims rotated
    """
    rotary_dim = cos.shape[-1]

    # Ensure cos/sin are 4D: [1, 1, T, rotary_dim]
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [1, 1, T, rd]
        sin = sin.unsqueeze(1)

    q_rot = q_or_k[..., :rotary_dim]
    q_pass = q_or_k[..., rotary_dim:]

    # rotate_half: [-x2, x1] where x1, x2 = chunk(2, dim=-1)
    x1 = q_rot[..., : rotary_dim // 2]
    x2 = q_rot[..., rotary_dim // 2 :]
    rotate_half = torch.cat([-x2, x1], dim=-1)

    q_rot_out = (q_rot * cos) + (rotate_half * sin)
    return torch.cat([q_rot_out, q_pass], dim=-1)


# ---------------------------------------------------------------------------
# Test 1: cos/sin tables match reference on 8×4 mesh
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_cos_sin_tables_match_reference_on_8x4_mesh(mesh_8x4):
    """Qwen36RopeSetup builds cos/sin that match reference build_mrope_cos_sin on 8×4 mesh.
    PCC > 0.999 for both cos and sin.
    """
    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    # Build config
    args = TtQwen36ModelArgs(mesh_8x4)
    assert args.rope_dim == _ROTARY_DIM, f"Expected rope_dim={_ROTARY_DIM}, got {args.rope_dim}"

    # Build RoPE setup
    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_BATCH_SIZE, max_seq_len=_MAX_SEQ_LEN)

    # Get prefill cos/sin from device (covers positions [0, max_seq_len))
    cos_tt, sin_tt = rope.get_cos_sin_for_prefill(seq_len=_MAX_SEQ_LEN)

    # Gather back to host (replicated tensors: ConcatMeshToTensor(dim=0) → [32, 1, seq, rd])
    # Take first device slice [0:1] to get [1, 1, seq_len, rotary_dim]
    cos_host_all = ttnn.to_torch(cos_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
    sin_host_all = ttnn.to_torch(sin_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
    # cos_host_all: [32, 1, max_seq_len, rotary_dim]
    cos_host = cos_host_all[0:1, 0, :, :]  # [1, seq_len, rotary_dim]
    sin_host = sin_host_all[0:1, 0, :, :]

    # Reference: build_mrope_cos_sin expects positions_3d [3, T]
    positions = torch.arange(_MAX_SEQ_LEN)
    positions_3d = torch.stack([positions, positions, positions], dim=0)  # [3, T]
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=_HEAD_DIM,
        partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
        mrope_section=_MROPE_SECTION,
        theta=_ROPE_THETA,
    )
    # cos_ref, sin_ref: [1, T, rotary_dim]

    cos_host_f32 = cos_host.float()
    sin_host_f32 = sin_host.float()

    passing_cos, pcc_cos = comp_pcc(cos_ref.float(), cos_host_f32, pcc=0.999)
    passing_sin, pcc_sin = comp_pcc(sin_ref.float(), sin_host_f32, pcc=0.999)

    print(f"\n[test_1] cos PCC: {pcc_cos}")
    print(f"[test_1] sin PCC: {pcc_sin}")
    assert passing_cos, f"test_1 cos FAILED: {pcc_cos}"
    assert passing_sin, f"test_1 sin FAILED: {pcc_sin}"

    cos_tt.deallocate(True)
    sin_tt.deallocate(True)


# ---------------------------------------------------------------------------
# Test 2: apply_partial_rope PCC on 8×4 mesh
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_apply_partial_rope_pcc_on_8x4_mesh(mesh_8x4):
    """apply_partial_rope on device matches CPU reference. PCC > 0.999."""
    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_BATCH_SIZE, max_seq_len=_MAX_SEQ_LEN)

    # T = 32 (prefill), n_heads = 24 (native Q heads)
    T = 32
    n_heads = 24
    torch.manual_seed(7)
    q_torch = torch.randn(1, n_heads, T, _HEAD_DIM, dtype=torch.float32)

    # Get cos/sin for this T
    cos_tt, sin_tt = rope.get_cos_sin_for_prefill(seq_len=T)

    # Send q to device as replicated
    q_tt = ttnn.from_torch(
        q_torch,
        device=mesh_8x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )

    # Apply partial RoPE on device
    q_rotated_tt = rope.apply_partial_rope(q_tt, cos_tt, sin_tt)

    # Gather result (replicated — all devices have same result)
    # ConcatMeshToTensor(dim=0) → [32, n_heads, T, head_dim]; take first device
    q_rotated_host_all = ttnn.to_torch(q_rotated_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
    q_rotated_host = q_rotated_host_all[0:1, :, :, :]  # [1, n_heads, T, head_dim]

    # CPU reference cos/sin
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=_HEAD_DIM,
        partial_rotary_factor=_PARTIAL_ROTARY_FACTOR,
        mrope_section=_MROPE_SECTION,
        theta=_ROPE_THETA,
    )
    # cos_ref: [1, T, rotary_dim] → need [1, 1, T, rotary_dim]
    cos_ref_4d = cos_ref.unsqueeze(1)
    sin_ref_4d = sin_ref.unsqueeze(1)

    q_ref = _apply_partial_rope_cpu(q_torch, cos_ref_4d, sin_ref_4d)

    passing, pcc_msg = comp_pcc(q_ref.bfloat16(), q_rotated_host.bfloat16(), pcc=0.999)
    print(f"\n[test_2] apply_partial_rope PCC: {pcc_msg}")
    assert passing, f"test_2 FAILED: {pcc_msg}"

    q_tt.deallocate(True)
    q_rotated_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)


# ---------------------------------------------------------------------------
# Test 3: apply_partial_rope preserves pass-through region
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_apply_partial_rope_preserves_pass_through(mesh_8x4):
    """Pass-through region (dims rotary_dim..head_dim-1) is unchanged after RoPE.
    Rotated region (dims 0..rotary_dim-1) of zero input stays zero.
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_BATCH_SIZE, max_seq_len=_MAX_SEQ_LEN)

    T = 32
    n_heads = 24
    pass_through_val = 3.14159

    # Build q where:
    #   first rotary_dim=64 dims are 0.0 (rotation of zero stays zero)
    #   remaining 192 dims are a specific constant (pass-through)
    q_torch = torch.zeros(1, n_heads, T, _HEAD_DIM, dtype=torch.float32)
    q_torch[..., _ROTARY_DIM:] = pass_through_val

    cos_tt, sin_tt = rope.get_cos_sin_for_prefill(seq_len=T)

    q_tt = ttnn.from_torch(
        q_torch,
        device=mesh_8x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )

    q_rotated_tt = rope.apply_partial_rope(q_tt, cos_tt, sin_tt)

    q_rotated_host_all = ttnn.to_torch(q_rotated_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
    q_rotated_host = q_rotated_host_all[0:1, :, :, :]  # [1, n_heads, T, head_dim]

    # Check: first 64 dims should still be ~0 (bfloat16 tolerance)
    rotary_region = q_rotated_host[..., :_ROTARY_DIM].float()
    assert torch.allclose(rotary_region, torch.zeros_like(rotary_region), atol=1e-3), (
        f"Rotary region (zero input) not zero after rotation. " f"max_abs={rotary_region.abs().max().item():.4f}"
    )

    # Check: last 192 dims should be unchanged (pass-through)
    pass_region = q_rotated_host[..., _ROTARY_DIM:].float()
    expected_pass = torch.full_like(pass_region, pass_through_val, dtype=torch.float32)
    # bfloat16 has ~0.5% relative error; pass_through_val=3.14159 → atol ~0.02
    assert torch.allclose(pass_region, expected_pass, atol=5e-2), (
        f"Pass-through region not preserved. " f"max_diff={( pass_region - expected_pass).abs().max().item():.4f}"
    )

    print(
        f"\n[test_3] pass-through preserved. "
        f"rotary_max_abs={rotary_region.abs().max().item():.2e}, "
        f"pass_max_diff={(pass_region - expected_pass).abs().max().item():.2e}"
    )

    q_tt.deallocate(True)
    q_rotated_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)


# ---------------------------------------------------------------------------
# Test 4: decode position advances
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_decode_position_advances(mesh_8x4):
    """cos/sin differ across decode positions (sin(0)=0, sin(1)≠0, sin(63) far from sin(0))."""
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_BATCH_SIZE, max_seq_len=_MAX_SEQ_LEN)

    def _get_cos_sin_host(cur_pos: int):
        cos_tt, sin_tt = rope.get_cos_sin_for_decode(cur_pos)
        # ConcatMeshToTensor(dim=0) → [32, 1, 1, rotary_dim]; take first device [0]
        cos_h = ttnn.to_torch(cos_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
        sin_h = ttnn.to_torch(sin_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_8x4, dim=0))
        cos_tt.deallocate(True)
        sin_tt.deallocate(True)
        # Shape: [32, 1, 1, rotary_dim] → take [0, 0, 0, :] → [rotary_dim]
        return cos_h[0, 0, 0, :].float(), sin_h[0, 0, 0, :].float()

    cos0, sin0 = _get_cos_sin_host(0)
    cos1, sin1 = _get_cos_sin_host(1)
    cos63, sin63 = _get_cos_sin_host(63)

    # sin(pos=0) should be ~0 (since angle = pos * freq = 0)
    assert sin0.abs().max() < 1e-3, f"sin(pos=0) should be ~0 but max={sin0.abs().max():.4f}"

    # cos(pos=0) should be ~1 (since cos(0) = 1)
    assert (cos0 - 1.0).abs().max() < 1e-2, f"cos(pos=0) should be ~1 but max_dev={(cos0-1.0).abs().max():.4f}"

    # sin(pos=1) should differ from sin(pos=0) by a meaningful amount
    diff_sin_01 = (sin1 - sin0).abs().max().item()
    assert diff_sin_01 > 1e-4, f"sin(pos=1) should differ from sin(pos=0) but max_diff={diff_sin_01:.4f}"

    # sin(pos=63) should differ substantially from sin(pos=0)
    diff_sin_0_63 = (sin63 - sin0).abs().max().item()
    assert diff_sin_0_63 > 1e-3, f"sin(pos=63) should differ from sin(pos=0) but max_diff={diff_sin_0_63:.4f}"

    # All positions should differ from each other
    diff_cos_01 = (cos1 - cos0).abs().max().item()
    assert diff_cos_01 > 1e-4, f"cos(pos=1) should differ from cos(pos=0) but max_diff={diff_cos_01:.4f}"

    print(
        f"\n[test_4] Position advances confirmed: "
        f"sin(0)_max={sin0.abs().max():.2e}, "
        f"sin(1)_max={sin1.abs().max():.3f}, "
        f"diff(sin0,sin1)={diff_sin_01:.3f}, "
        f"diff(sin0,sin63)={diff_sin_0_63:.3f}"
    )
