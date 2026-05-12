# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for models/demos/qwen3_6_galaxy/tt/qwen36_deltanet.py.

Tests are written RED-first.  They cover:
  1. test_deltanet_block_layer0_prefill_pcc_on_8x4
  2. test_deltanet_block_layer0_decode_step_pcc_on_8x4
  3. test_deltanet_sharding_correctness
  4. test_deltanet_state_persistence_across_decode_steps

Hardware required: 32-chip BH GLX 8×4 mesh.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_qwen36_deltanet.py -x -s -v
"""

import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

# Layer 0 is linear_attention — matches GatedDeltaNet
_LAYER_IDX = 0
_B = 1
_T_PREFILL = 32
_T_DECODE = 1
_H = 5120
_PCC_THRESH = 0.99

_LINEAR_ATTN_PREFIX = f"model.language_model.layers.{_LAYER_IDX}.linear_attn"
_WEIGHT_KEYS = [
    f"{_LINEAR_ATTN_PREFIX}.in_proj_qkv.weight",
    f"{_LINEAR_ATTN_PREFIX}.in_proj_z.weight",
    f"{_LINEAR_ATTN_PREFIX}.in_proj_a.weight",
    f"{_LINEAR_ATTN_PREFIX}.in_proj_b.weight",
    f"{_LINEAR_ATTN_PREFIX}.conv1d.weight",
    f"{_LINEAR_ATTN_PREFIX}.A_log",
    f"{_LINEAR_ATTN_PREFIX}.dt_bias",
    f"{_LINEAR_ATTN_PREFIX}.norm.weight",
    f"{_LINEAR_ATTN_PREFIX}.out_proj.weight",
]


# ---------------------------------------------------------------------------
# Fabric mesh fixture (same as T3, T4, T5)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the full 8×4 fabric mesh with FABRIC_1D_RING topology."""
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
# Weight loading helpers
# ---------------------------------------------------------------------------


def _load_layer0_weights():
    """Load layer-0 linear_attention weights from safetensors.

    Returns dict with keys matching GatedDeltaNet.state_dict() naming.
    """
    from safetensors.torch import load_file as load_st

    base = _SNAPSHOT_DIR
    with open(base / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    files_needed = sorted({weight_map[k] for k in _WEIGHT_KEYS if k in weight_map})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(base / fn))
        for k in _WEIGHT_KEYS:
            if k in shard:
                raw[k] = shard[k]

    pfx = _LINEAR_ATTN_PREFIX
    return {
        "in_proj_qkv.weight": raw[f"{pfx}.in_proj_qkv.weight"].float(),
        "in_proj_z.weight": raw[f"{pfx}.in_proj_z.weight"].float(),
        "in_proj_a.weight": raw[f"{pfx}.in_proj_a.weight"].float(),
        "in_proj_b.weight": raw[f"{pfx}.in_proj_b.weight"].float(),
        "conv1d.weight": raw[f"{pfx}.conv1d.weight"].float(),
        "A_log": raw[f"{pfx}.A_log"].float(),
        "dt_bias": raw[f"{pfx}.dt_bias"].float(),
        "norm.weight": raw[f"{pfx}.norm.weight"].float(),
        "out_proj.weight": raw[f"{pfx}.out_proj.weight"].float(),
    }


def _make_hidden_state(T: int, seed: int = 42) -> torch.Tensor:
    """Random bfloat16 hidden state [B=1, T, H=5120]."""
    torch.manual_seed(seed)
    return torch.randn(_B, T, _H, dtype=torch.bfloat16)


def _cpu_reference(sd: dict, x: torch.Tensor):
    """Run the validated GatedDeltaNet CPU reference.

    Returns (output [B, T, H], conv_state_new, recurrent_state_new).
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedDeltaNet, Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    model = GatedDeltaNet(config)
    model.eval()

    # Load weights
    with torch.no_grad():
        model.in_proj_qkv.weight.data.copy_(sd["in_proj_qkv.weight"])
        model.in_proj_z.weight.data.copy_(sd["in_proj_z.weight"])
        model.in_proj_a.weight.data.copy_(sd["in_proj_a.weight"])
        model.in_proj_b.weight.data.copy_(sd["in_proj_b.weight"])
        model.conv1d.weight.data.copy_(sd["conv1d.weight"])
        model.A_log.data.copy_(sd["A_log"])
        model.dt_bias.data.copy_(sd["dt_bias"])
        model.norm.weight.data.copy_(sd["norm.weight"])
        model.out_proj.weight.data.copy_(sd["out_proj.weight"])

        out, cs_new, rs_new = model(x.float())
    return out, cs_new, rs_new


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson Correlation Coefficient between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    cc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return cc


def _gather_from_device(tt_out, mesh_device, T: int = None):
    """Gather a replicated TTNN output back to CPU (first device's slice).

    Args:
        T: If provided, slice the time dimension to T to strip TTNN tile-padding.
           Required when T < 32 (tile size), e.g. T=1 for decode.
    """
    import ttnn

    out_cpu = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # out_cpu: [32, B, T_pad, H] or [32*B, T_pad, H] — take first device
    result = out_cpu[0:1]  # [1, T_pad, H]
    if T is not None:
        result = result[:, :T, :]
    return result


# ---------------------------------------------------------------------------
# Test 1: Prefill PCC > 0.99
# ---------------------------------------------------------------------------


def test_deltanet_block_layer0_prefill_pcc_on_8x4(mesh_8x4):
    """Prefill mode PCC test: TtQwen36DeltaNet vs GatedDeltaNet reference."""
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.qwen36_deltanet import TtQwen36DeltaNet
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    sd = _load_layer0_weights()
    args = TtQwen36ModelArgs(mesh_8x4)

    # Build TTNN DeltaNet block
    deltanet = TtQwen36DeltaNet(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat16,
    )

    # Input: [B=1, T=32, H=5120] replicated
    x_torch = _make_hidden_state(_T_PREFILL, seed=7)
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_8x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )

    # TTNN forward
    tt_out = deltanet.forward(x_tt, mode="prefill")
    tt_cpu = _gather_from_device(tt_out, mesh_8x4)  # [1, T, H]
    tt_cpu = tt_cpu.reshape(_B, _T_PREFILL, _H).float()

    # CPU reference
    ref_out, _, _ = _cpu_reference(sd, x_torch)
    ref_out = ref_out.float()

    pcc = _pcc(tt_cpu, ref_out)
    print(f"\n[prefill] PCC = {pcc:.6f}")
    assert pcc > _PCC_THRESH, f"Prefill PCC {pcc:.6f} < {_PCC_THRESH}"


# ---------------------------------------------------------------------------
# Test 2: Decode step PCC > 0.99
# ---------------------------------------------------------------------------


def test_deltanet_block_layer0_decode_step_pcc_on_8x4(mesh_8x4):
    """Decode (T=1) mode PCC test: TtQwen36DeltaNet vs GatedDeltaNet reference."""
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.qwen36_deltanet import TtQwen36DeltaNet
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    sd = _load_layer0_weights()
    args = TtQwen36ModelArgs(mesh_8x4)

    deltanet = TtQwen36DeltaNet(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat16,
    )

    x_torch = _make_hidden_state(_T_DECODE, seed=13)
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_8x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )

    # TTNN decode forward
    tt_out = deltanet.forward(x_tt, mode="decode")
    tt_cpu = _gather_from_device(tt_out, mesh_8x4, T=_T_DECODE)
    tt_cpu = tt_cpu.reshape(_B, _T_DECODE, _H).float()

    # CPU reference (T=1 triggers recurrent path)
    ref_out, _, _ = _cpu_reference(sd, x_torch)
    ref_out = ref_out.float()

    pcc = _pcc(tt_cpu, ref_out)
    print(f"\n[decode] PCC = {pcc:.6f}")
    assert pcc > _PCC_THRESH, f"Decode PCC {pcc:.6f} < {_PCC_THRESH}"


# ---------------------------------------------------------------------------
# Test 3: Sharding correctness — verify conv1d output per row
# ---------------------------------------------------------------------------


def test_deltanet_sharding_correctness(mesh_8x4):
    """Diagnostic: verify per-row sharding layout for conv1d output.

    Each row should have its own slice of conv Q/K/V heads:
      - Row i: Q_conv[i*256:(i+1)*256], K_conv[i*256:(i+1)*256], V_conv[i*768:(i+1)*768]

    This catches the Bug1 conv1d layout bug if reintroduced.
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.qwen36_deltanet import TtQwen36DeltaNet
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    sd = _load_layer0_weights()
    args = TtQwen36ModelArgs(mesh_8x4)

    deltanet = TtQwen36DeltaNet(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat16,
    )

    # Get the pre-interleaved conv weight on host for validation
    # conv1d.weight: [10240, 1, 4], block layout Q[2048]|K[2048]|V[6144]
    conv_w = sd["conv1d.weight"]  # [10240, 1, 4]
    conv_w_flat = conv_w.squeeze(1)  # [10240, 4]

    mesh_rows = 8
    n_k = 16
    hd_k = 128
    n_v = 48
    hd_v = 128
    n_k_per_row = n_k // mesh_rows  # 2
    n_v_per_row = n_v // mesh_rows  # 6
    qk_per_row = n_k_per_row * hd_k  # 256
    v_per_row = n_v_per_row * hd_v  # 768

    # Reference: what each row should have (from correct pre-interleaving)
    conv_Q = conv_w_flat[: n_k * hd_k]  # [2048, 4]
    conv_K = conv_w_flat[n_k * hd_k : 2 * n_k * hd_k]  # [2048, 4]
    conv_V = conv_w_flat[2 * n_k * hd_k :]  # [6144, 4]

    for row_i in range(mesh_rows):
        expected_q = conv_Q[row_i * qk_per_row : (row_i + 1) * qk_per_row]  # [256, 4]
        expected_k = conv_K[row_i * qk_per_row : (row_i + 1) * qk_per_row]  # [256, 4]
        expected_v = conv_V[row_i * v_per_row : (row_i + 1) * v_per_row]  # [768, 4]
        expected_row = torch.cat([expected_q, expected_k, expected_v], dim=0)  # [1280, 4]

        # Verify against the block stored in deltanet
        actual_row = deltanet.get_conv_weight_row(row_i)  # [1280, 4] float32
        assert (
            actual_row.shape == expected_row.shape
        ), f"Row {row_i}: shape mismatch {actual_row.shape} vs {expected_row.shape}"
        # Use torch.allclose since weights are exact (no dtype conversion)
        assert torch.allclose(actual_row.float(), expected_row.float(), atol=1e-3), (
            f"Row {row_i}: conv weight layout mismatch (max diff "
            f"{(actual_row.float() - expected_row.float()).abs().max():.6f})"
        )

    print("\n[sharding] All 8 row conv1d weight layouts correct")


# ---------------------------------------------------------------------------
# Test 4: State persistence across decode steps
# ---------------------------------------------------------------------------


def test_deltanet_state_persistence_across_decode_steps(mesh_8x4):
    """Verify recurrent state continuity between prefill and decode.

    Protocol:
      1. Prefill T=16: capture final recurrent_state
      2. Run 4 decode steps with that state
      3. Run prefill T=20 from scratch (full context)
      4. Assert decode step 4 matches prefill step T=19 (PCC > 0.99)
    """
    import ttnn
    from models.demos.qwen3_6_galaxy.tt.qwen36_deltanet import TtQwen36DeltaNet
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    sd = _load_layer0_weights()
    args = TtQwen36ModelArgs(mesh_8x4)

    deltanet = TtQwen36DeltaNet(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
        dtype=ttnn.bfloat16,
    )

    T_prefill = 16
    T_extra = 4

    # Build a deterministic input sequence [B=1, T=20, H=5120]
    torch.manual_seed(99)
    x_all = torch.randn(_B, T_prefill + T_extra, _H, dtype=torch.bfloat16)

    # --- CPU reference: full T=20 prefill ---
    ref_out_full, _, _ = _cpu_reference(sd, x_all.float())
    ref_last_4 = ref_out_full[:, T_prefill:, :]  # [B, 4, H]

    # --- TT: prefill T=16, then 4 decode steps ---
    x_pfill_tt = ttnn.from_torch(
        x_all[:, :T_prefill],
        device=mesh_8x4,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )
    _, rs_after_prefill, cs_after_prefill = deltanet.forward(x_pfill_tt, mode="prefill", return_state=True)

    tt_decode_outs = []
    rs = rs_after_prefill
    cs = cs_after_prefill
    for t in range(T_extra):
        x_tok = x_all[:, T_prefill + t : T_prefill + t + 1]  # [B, 1, H]
        x_tok_tt = ttnn.from_torch(
            x_tok,
            device=mesh_8x4,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
        )
        out_tt, rs, cs = deltanet.forward(x_tok_tt, mode="decode", recurrent_state=rs, conv_state=cs, return_state=True)
        out_cpu = _gather_from_device(out_tt, mesh_8x4, T=1).reshape(_B, 1, _H).float()
        tt_decode_outs.append(out_cpu)

    tt_decode_cat = torch.cat(tt_decode_outs, dim=1)  # [B, 4, H]

    pcc = _pcc(tt_decode_cat, ref_last_4.float())
    print(f"\n[state_persistence] PCC decode-after-prefill = {pcc:.6f}")
    assert pcc > _PCC_THRESH, (
        f"State persistence PCC {pcc:.6f} < {_PCC_THRESH} — "
        "recurrent state not correctly threaded between prefill and decode"
    )
