# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for models/demos/qwen3_6_galaxy/tt/qwen36_model_config.py.

Tests are written RED-first and should fail before the implementation exists.

Run all:
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_qwen36_model_config.py -x -s

Run hardware tests only (need 32-chip BH GLX mesh):
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_qwen36_model_config.py -m hardware -x -s

Run CPU/offline tests only:
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_qwen36_model_config.py -m "not hardware" -x -s
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNAPSHOT = Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

HF_LAYER_TYPES = None  # loaded lazily


def _get_hf_layer_types():
    global HF_LAYER_TYPES
    if HF_LAYER_TYPES is None:
        with open(SNAPSHOT / "config.json") as f:
            cfg = json.load(f)
        HF_LAYER_TYPES = cfg["text_config"]["layer_types"]
    return HF_LAYER_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    """Open the full 8×4 BH GLX mesh for hardware tests.

    Uses FABRIC_1D_RING as required for 32-chip BH. Closes after the module.
    Skip if ttnn or the hardware is unavailable.
    """
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
# Test 1 — instantiate on 8×4 mesh, verify basic scalar dims
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_instantiate_on_8x4_mesh(mesh_device):
    """Open fabric + 8×4 mesh, instantiate TtQwen36ModelArgs, verify basic dims."""
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    cfg = TtQwen36ModelArgs(mesh_device)

    # Native model dimensions from config.json text_config
    assert cfg.dim == 5120, f"Expected dim=5120, got {cfg.dim}"
    assert cfg.n_heads == 24, f"Expected n_heads=24, got {cfg.n_heads}"
    assert cfg.head_dim == 256, f"Expected head_dim=256, got {cfg.head_dim}"
    assert cfg.intermediate_dim == 17408, f"Expected intermediate_dim=17408, got {cfg.intermediate_dim}"
    assert cfg.n_kv_heads == 4, f"Expected n_kv_heads=4, got {cfg.n_kv_heads}"
    assert cfg.num_hidden_layers == 64, f"Expected num_hidden_layers=64, got {cfg.num_hidden_layers}"
    assert cfg.vocab_size == 248320, f"Expected vocab_size=248320, got {cfg.vocab_size}"

    # Cluster/hardware
    assert cfg.num_devices == 32, f"Expected num_devices=32, got {cfg.num_devices}"
    assert cfg.cluster_shape == [8, 4], f"Expected cluster_shape=[8,4], got {cfg.cluster_shape}"


# ---------------------------------------------------------------------------
# Test 2 — padded dims match Qwen3-32B targets (no hardware needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_hw", [False])
def test_padded_dims_match_qwen3_32b_targets(use_hw):
    """Verify pad-and-slice constants without needing a mesh device."""
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    cfg = TtQwen36ModelArgs(mesh_device=None)

    # Q/KV head padding targets from task spec
    assert cfg.n_q_heads_native == 24, f"n_q_heads_native={cfg.n_q_heads_native}"
    assert cfg.n_q_heads_padded == 64, f"n_q_heads_padded={cfg.n_q_heads_padded}"
    assert cfg.n_kv_heads_native == 4, f"n_kv_heads_native={cfg.n_kv_heads_native}"
    assert cfg.n_kv_heads_padded == 8, f"n_kv_heads_padded={cfg.n_kv_heads_padded}"

    # Intermediate padding (24-core aligned, matching olmo_galaxy choice)
    assert cfg.intermediate_dim_native == 17408, f"intermediate_dim_native={cfg.intermediate_dim_native}"
    assert (
        cfg.intermediate_dim_per_tp_native == 2176
    ), f"intermediate_dim_per_tp_native={cfg.intermediate_dim_per_tp_native}"
    assert (
        cfg.intermediate_dim_per_tp_padded == 3840
    ), f"intermediate_dim_per_tp_padded={cfg.intermediate_dim_per_tp_padded}"

    # Vocab padding: ceil(248320 / 1024) * 1024 = 248832 = 32 * 7776
    assert cfg.vocab_size_native == 248320, f"vocab_size_native={cfg.vocab_size_native}"
    assert cfg.padded_vocab_size == 248832, f"padded_vocab_size={cfg.padded_vocab_size}"

    # dim_per_tp: no change needed (5120/4=1280, already tile-aligned)
    assert cfg.dim_per_tp == 1280, f"dim_per_tp={cfg.dim_per_tp}"

    # TP factors
    assert cfg.dim_tp_factor == 4, f"dim_tp_factor={cfg.dim_tp_factor}"
    assert cfg.intermediate_dim_tp_factor == 8, f"intermediate_dim_tp_factor={cfg.intermediate_dim_tp_factor}"


# ---------------------------------------------------------------------------
# Test 3 — qkv_size includes output gate when attn_output_gate=True
# ---------------------------------------------------------------------------


def test_qkv_size_includes_gate():
    """When attn_output_gate=True, qkv_size == head_dim * (2*n_kv + 2*n_q).
    When False, qkv_size == head_dim * (2*n_kv + n_q).
    """
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    # Default: attn_output_gate=True
    cfg_gate = TtQwen36ModelArgs(mesh_device=None)
    assert cfg_gate.attn_output_gate is True
    # 256 * (2*4 + 2*24) = 256 * (8 + 48) = 256 * 56 = 14336
    expected_gate = 256 * (2 * 4 + 2 * 24)
    assert expected_gate == 14336
    assert cfg_gate.qkv_size == expected_gate, f"qkv_size with gate: expected {expected_gate}, got {cfg_gate.qkv_size}"

    # With gate disabled
    cfg_no_gate = TtQwen36ModelArgs(mesh_device=None, attn_output_gate=False)
    assert cfg_no_gate.attn_output_gate is False
    # 256 * (2*4 + 24) = 256 * 32 = 8192
    expected_no_gate = 256 * (2 * 4 + 24)
    assert expected_no_gate == 8192
    assert (
        cfg_no_gate.qkv_size == expected_no_gate
    ), f"qkv_size without gate: expected {expected_no_gate}, got {cfg_no_gate.qkv_size}"


# ---------------------------------------------------------------------------
# Test 4 — linear_attention_pattern matches HF config
# ---------------------------------------------------------------------------


def test_linear_attention_pattern_matches_hf():
    """linear_attention_pattern must have 64 entries repeating [lin,lin,lin,full]×16."""
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    cfg = TtQwen36ModelArgs(mesh_device=None)
    pattern = cfg.linear_attention_pattern
    hf_pattern = _get_hf_layer_types()

    assert len(pattern) == 64, f"Expected 64 entries, got {len(pattern)}"
    assert pattern == hf_pattern, "linear_attention_pattern does not match HF layer_types"

    # Check the [lin, lin, lin, full] × 16 structure
    for i in range(16):
        base = i * 4
        assert pattern[base] == "linear_attention", f"Layer {base} should be linear_attention"
        assert pattern[base + 1] == "linear_attention", f"Layer {base+1} should be linear_attention"
        assert pattern[base + 2] == "linear_attention", f"Layer {base+2} should be linear_attention"
        assert pattern[base + 3] == "full_attention", f"Layer {base+3} should be full_attention"

    # Explicit first group
    assert pattern[0] == "linear_attention"
    assert pattern[1] == "linear_attention"
    assert pattern[2] == "linear_attention"
    assert pattern[3] == "full_attention"


# ---------------------------------------------------------------------------
# Test 5 — BH GLX CCL topology (hardware required)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_bh_glx_ccl_topology(mesh_device):
    """On 8×4 mesh: device_name=='BH_GLX', GALAXY_NUM_LINKS==1, CCL_TOPOLOGY==Linear."""
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    cfg = TtQwen36ModelArgs(mesh_device)

    assert cfg.device_name == "BH_GLX", f"Expected BH_GLX, got {cfg.device_name}"
    assert cfg.GALAXY_NUM_LINKS == 1, f"Expected GALAXY_NUM_LINKS=1, got {cfg.GALAXY_NUM_LINKS}"
    assert cfg.CCL_TOPOLOGY == ttnn.Topology.Linear, f"Expected CCL_TOPOLOGY=Linear, got {cfg.CCL_TOPOLOGY}"


# ---------------------------------------------------------------------------
# Test 6 — divisibility asserts fire for wrong mesh shape (no hardware)
# ---------------------------------------------------------------------------


def test_divisibility_asserts():
    """Instantiating with wrong mesh shape (cluster_shape=[4,4]) raises AssertionError."""
    from unittest.mock import MagicMock

    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    # Create a mock mesh device with 16 devices in 4×4 shape
    mock_mesh = MagicMock()
    mock_mesh.get_num_devices.return_value = 16
    mock_mesh.shape = (4, 4)

    with pytest.raises((AssertionError, ValueError)) as exc_info:
        TtQwen36ModelArgs(mock_mesh)

    err = str(exc_info.value)
    # The error should mention something about the device count or shape
    assert any(
        kw in err.lower() for kw in ["32", "device", "mesh", "unsupported", "galaxy", "num_devices"]
    ), f"AssertionError message not descriptive enough: {err}"
