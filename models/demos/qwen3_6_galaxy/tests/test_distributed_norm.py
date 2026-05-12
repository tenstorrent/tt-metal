# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for models/demos/qwen3_6_galaxy/tt/distributed_norm.py.

Tests are written RED-first (before implementation exists) and should fail with
ModuleNotFoundError when the implementation doesn't exist.

Run all hardware tests (needs 32-chip BH GLX mesh):
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_distributed_norm.py -x -s

Tests 1-4 all require hardware (8×4 BH GLX mesh).
"""

import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = (
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_LAYER_0_INPUT_LN_KEY = "model.language_model.layers.0.input_layernorm.weight"
_DIM = 5120
_EPS = 1e-6
_B, _T, _H = 1, 32, _DIM  # batch=1, seqlen=32, hidden=5120


# ---------------------------------------------------------------------------
# Fixture: full 8×4 BH GLX mesh with FABRIC_1D_RING
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the full 8×4 fabric mesh.  Fabric init is mandatory before open."""
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
# Helpers
# ---------------------------------------------------------------------------


def _load_layer0_input_ln_weight():
    """Load the real layer-0 input_layernorm.weight from safetensors."""
    from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_tensors

    tensors = load_qwen36_tensors([_LAYER_0_INPUT_LN_KEY])
    return tensors[_LAYER_0_INPUT_LN_KEY].float()  # shape [5120]


def _make_random_hidden_state(seed: int = 42) -> torch.Tensor:
    """Random bfloat16 hidden state [1, 32, 5120]."""
    torch.manual_seed(seed)
    return torch.randn(_B, _T, _H, dtype=torch.bfloat16)


def _run_distributed_norm_forward(mesh_device, norm_module, x_torch: torch.Tensor):
    """
    Shard the input across cluster_axis=1 (cols), run distributed norm, gather output.

    The input [B, T, H] is sharded across dim=-1 (width) across 4 columns.
    Each column holds a 1280-wide slice.  This matches the expected input layout
    for tt_distributed_rmsnorm as used in the Qwen3.6 decode path.

    Returns a torch.Tensor of shape [B, T, H] on CPU.
    """
    import ttnn

    cluster_shape = [8, 4]

    # Reshape to 4D for ttnn: [B, 1, T, H]
    x_4d = x_torch.unsqueeze(1)  # [1, 1, 32, 5120]

    tt_x = ttnn.from_torch(
        x_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = norm_module(tt_x)

    # Gather: concat across cluster_axis=1 (dim=3 for 4D tensor: [B,1,T,H/4])
    out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=cluster_shape),
    )
    # out_torch: [8*B, 1, T, H] after concat on dim=0 (rows) and dim=3 (cols)
    # We want [B, T, H] — take the first B slices (row 0 = same data replicated)
    out_torch = out_torch[:_B, 0, :, :]  # [B, T, H]

    tt_x.deallocate(True)
    tt_out.deallocate(True)
    return out_torch


# ---------------------------------------------------------------------------
# Test 1 — standard RMSNorm (zero_centered=False), PCC > 0.999
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_distributed_norm_standard_pcc_on_8x4(mesh_8x4):
    """Standard (non-zero-centered) DistributedNorm on 8×4 mesh. PCC > 0.999."""
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm

    weight_torch = _load_layer0_input_ln_weight()  # [5120], float32
    x_torch = _make_random_hidden_state(seed=1)  # [1, 32, 5120], bfloat16

    # TTNN distributed norm
    norm = DistributedNorm(mesh_8x4, weight_torch, eps=_EPS, zero_centered=False)
    ttnn_out = _run_distributed_norm_forward(mesh_8x4, norm, x_torch)

    # CPU reference: standard RMSNorm (w * norm(x))
    # weight_torch is already in [0,1] range for standard convention
    x_f32 = x_torch.float()
    rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + _EPS)
    ref_out = (x_f32 * rms * weight_torch).to(torch.bfloat16)

    passing, pcc_msg = comp_pcc(ref_out, ttnn_out, pcc=0.999)
    print(f"\n[test_1] standard PCC: {pcc_msg}")
    assert passing, f"test_1 FAILED: {pcc_msg}"


# ---------------------------------------------------------------------------
# Test 2 — zero-centered RMSNorm (zero_centered=True), PCC > 0.999
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_distributed_norm_zero_centered_pcc_on_8x4(mesh_8x4):
    """Zero-centered DistributedNorm (HF Qwen3NextRMSNorm convention) on 8×4 mesh. PCC > 0.999."""
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.reference.qwen36 import RMSNorm as RefRMSNorm
    from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm

    weight_torch = _load_layer0_input_ln_weight()  # [5120], float32 (zero-centred convention: values ≈ 0)
    x_torch = _make_random_hidden_state(seed=2)  # [1, 32, 5120], bfloat16

    # TTNN distributed norm with zero_centered=True
    norm = DistributedNorm(mesh_8x4, weight_torch, eps=_EPS, zero_centered=True)
    ttnn_out = _run_distributed_norm_forward(mesh_8x4, norm, x_torch)

    # CPU reference: (1 + weight) * norm(x)  [HF Qwen3NextRMSNorm convention]
    ref_norm = RefRMSNorm(dim=_DIM, eps=_EPS, zero_centered=True)
    with torch.no_grad():
        ref_norm.weight.data.copy_(weight_torch)
        ref_out = ref_norm(x_torch.float()).to(torch.bfloat16)

    passing, pcc_msg = comp_pcc(ref_out, ttnn_out, pcc=0.999)
    print(f"\n[test_2] zero-centered PCC: {pcc_msg}")
    assert passing, f"test_2 FAILED: {pcc_msg}"


# ---------------------------------------------------------------------------
# Test 3 — compare against HF Qwen3NextRMSNorm directly, PCC > 0.999
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_distributed_norm_matches_hf_qwen3next_rms_norm(mesh_8x4):
    """TTNN distributed norm (zero_centered=True) matches HF Qwen3NextRMSNorm. PCC > 0.999."""
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm

    weight_torch = _load_layer0_input_ln_weight()  # [5120]
    x_torch = _make_random_hidden_state(seed=3)  # [1, 32, 5120], bfloat16

    # TTNN with zero_centered=True
    norm = DistributedNorm(mesh_8x4, weight_torch, eps=_EPS, zero_centered=True)
    ttnn_out = _run_distributed_norm_forward(mesh_8x4, norm, x_torch)

    # HF Qwen3NextRMSNorm reference
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm
    except ImportError:
        pytest.skip("transformers Qwen3NextRMSNorm not available")

    with torch.no_grad():
        hf_norm = Qwen3NextRMSNorm(dim=_DIM, eps=_EPS)
        hf_norm.weight.data.copy_(weight_torch)
        hf_out = hf_norm(x_torch.float()).to(torch.bfloat16)

    passing, pcc_msg = comp_pcc(hf_out, ttnn_out, pcc=0.999)
    print(f"\n[test_3] vs HF Qwen3NextRMSNorm PCC: {pcc_msg}")
    assert passing, f"test_3 FAILED: {pcc_msg}"


# ---------------------------------------------------------------------------
# Test 4 — end-to-end with real layer-0 input_layernorm weight, PCC > 0.999
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_distributed_norm_input_layernorm_layer_0_real_weights(mesh_8x4):
    """Real layer-0 input_layernorm weight + random hidden state, zero_centered=True. PCC > 0.999."""
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.reference.qwen36 import RMSNorm as RefRMSNorm
    from models.demos.qwen3_6_galaxy.tt.distributed_norm import DistributedNorm

    weight_torch = _load_layer0_input_ln_weight()  # [5120], real weight from safetensors
    # Simulate pre-norm input: random bfloat16 hidden state as stand-in for layer-0 input
    x_torch = _make_random_hidden_state(seed=4)  # [1, 32, 5120]

    # TTNN
    norm = DistributedNorm(mesh_8x4, weight_torch, eps=_EPS, zero_centered=True)
    ttnn_out = _run_distributed_norm_forward(mesh_8x4, norm, x_torch)

    # CPU oracle (our reference RMSNorm, zero_centered=True)
    ref_norm = RefRMSNorm(dim=_DIM, eps=_EPS, zero_centered=True)
    with torch.no_grad():
        ref_norm.weight.data.copy_(weight_torch)
        ref_out = ref_norm(x_torch.float()).to(torch.bfloat16)

    passing, pcc_msg = comp_pcc(ref_out, ttnn_out, pcc=0.999)
    print(f"\n[test_4] real weights PCC: {pcc_msg}")
    assert passing, f"test_4 FAILED: {pcc_msg}"
