# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for models/demos/qwen3_6_galaxy/tt/llama_attention.py.

Tests written RED-first (before implementation) — should fail with
ImportError/ModuleNotFoundError until TtQwen36GatedAttention is implemented.

The standalone attention module uses ReplicateTensorToMesh for all weights, so
the input tensor must also be replicated (not sharded) in these unit tests.
In the full decoder pipeline, the input will be sharded and the weights will
be appropriately sharded/replicated.

Run all hardware tests (needs 32-chip BH GLX mesh):
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_llama_attention.py -x -s -v

All 4 tests require hardware (8×4 BH GLX mesh).
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

_LAYER_IDX = 3  # full_attention layer (pattern index 3 in [lin,lin,lin,full]×16)
_B = 1
_T_PREFILL = 32  # prefill seqlen (must be ≥ 32, divisible by 32)
_H = 5120  # hidden_size
_N_Q = 24  # native Q heads
_N_KV = 4  # native KV heads
_HEAD_DIM = 256
_PCC_THRESH = 0.99

_SELF_ATTN_PREFIX = f"model.language_model.layers.{_LAYER_IDX}.self_attn"
_WEIGHT_KEYS = [
    f"{_SELF_ATTN_PREFIX}.q_proj.weight",
    f"{_SELF_ATTN_PREFIX}.k_proj.weight",
    f"{_SELF_ATTN_PREFIX}.v_proj.weight",
    f"{_SELF_ATTN_PREFIX}.o_proj.weight",
    f"{_SELF_ATTN_PREFIX}.q_norm.weight",
    f"{_SELF_ATTN_PREFIX}.k_norm.weight",
]


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
# Helpers
# ---------------------------------------------------------------------------


def _load_layer_weights():
    """Load layer-3 self-attention weights from safetensors. Returns dict."""
    from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_tensors

    raw = load_qwen36_tensors(_WEIGHT_KEYS)
    return {
        "q_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.q_proj.weight"].float(),
        "k_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.k_proj.weight"].float(),
        "v_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.v_proj.weight"].float(),
        "o_proj.weight": raw[f"{_SELF_ATTN_PREFIX}.o_proj.weight"].float(),
        "q_norm.weight": raw[f"{_SELF_ATTN_PREFIX}.q_norm.weight"].float(),
        "k_norm.weight": raw[f"{_SELF_ATTN_PREFIX}.k_norm.weight"].float(),
    }


def _make_random_hidden_state(T: int, seed: int = 42) -> torch.Tensor:
    """Random bfloat16 hidden state [B, T, H]."""
    torch.manual_seed(seed)
    return torch.randn(_B, T, _H, dtype=torch.bfloat16)


def _build_causal_mask(T: int) -> torch.Tensor:
    """4D additive causal mask [1, 1, T, T] with -inf in upper triangle."""
    mask = torch.zeros(1, 1, T, T)
    mask = mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))
    return mask


def _cpu_reference_attention(sd, x_torch, cos_ref, sin_ref, mask):
    """Run GatedAttention CPU reference.

    sd: state dict keys q_proj.weight, k_proj.weight, v_proj.weight,
                        o_proj.weight, q_norm.weight, k_norm.weight
    x_torch: [B, T, H] float32 or bfloat16
    cos_ref, sin_ref: [1, T, rotary_dim] float32
    mask: [1, 1, T, T] float32

    Returns: [B, T, H] float32
    """
    import json
    import pathlib

    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config

    cfg_path = pathlib.Path(_SNAPSHOT_DIR) / "config.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    with torch.no_grad():
        attn = GatedAttention(config)
        attn.eval()
        attn.q_proj.weight.data.copy_(sd["q_proj.weight"])
        attn.k_proj.weight.data.copy_(sd["k_proj.weight"])
        attn.v_proj.weight.data.copy_(sd["v_proj.weight"])
        attn.o_proj.weight.data.copy_(sd["o_proj.weight"])
        attn.q_norm.weight.data.copy_(sd["q_norm.weight"])
        attn.k_norm.weight.data.copy_(sd["k_norm.weight"])
        out, _ = attn(x_torch.float(), cos_ref, sin_ref, kv_cache=None, attention_mask=mask)
    return out  # [B, T, H] float32


def _build_rope_cos_sin(T: int):
    """Build MRoPE cos/sin for text-only inference at positions [0, T)."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=_HEAD_DIM,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    return cos, sin  # float32, [1, T, 64]


def _send_to_device(t: torch.Tensor, mesh_device, dtype=None):
    """Send a torch tensor to device, replicated across all 32 chips."""
    import ttnn

    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _gather_replicated(tt_tensor, mesh_device):
    """Gather a replicated TTNN tensor back to host.

    Returns: tensor with leading dim removed (first device's data).
    ConcatMeshToTensor(dim=0) → [32, ...]; take first device.
    """
    import ttnn

    all_devices = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return all_devices[0:1]  # [1, ...]


# ---------------------------------------------------------------------------
# Test 1 — prefill PCC > 0.99 against reference oracle
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_gated_attention_layer3_prefill_pcc_on_8x4(mesh_8x4):
    """TtQwen36GatedAttention prefill PCC > 0.99 vs CPU reference oracle.

    Uses real layer-3 weights and random hidden state.
    Verifies: q/k norm, partial RoPE, SDPA, output gate, WO matmul.
    """
    from models.common.utility_functions import comp_pcc

    # --- Imports of implementation (will fail RED until implemented) ---
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    sd = _load_layer_weights()

    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_B, max_seq_len=512)
    cos_tt, sin_tt = rope.get_cos_sin_for_prefill(seq_len=_T_PREFILL)

    attn = TtQwen36GatedAttention(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
    )

    x_torch = _make_random_hidden_state(_T_PREFILL, seed=10)  # [1, 32, 5120] bfloat16
    # Input: replicated across all 32 chips (standalone test mode)
    x_tt = _send_to_device(x_torch, mesh_8x4)

    # Run forward_prefill
    out_tt = attn.forward_prefill(x_tt, rot_mats=(cos_tt, sin_tt), kv_cache=None)

    # Gather output — all chips have identical result (replicated weights + input)
    out_host = _gather_replicated(out_tt, mesh_8x4)  # [1, T, H]

    # CPU reference
    cos_ref, sin_ref = _build_rope_cos_sin(_T_PREFILL)
    mask = _build_causal_mask(_T_PREFILL)
    ref_out = _cpu_reference_attention(sd, x_torch, cos_ref, sin_ref, mask)  # [1, T, H]

    passing, pcc_msg = comp_pcc(ref_out.bfloat16(), out_host.bfloat16(), pcc=_PCC_THRESH)
    print(f"\n[test_1] prefill PCC: {pcc_msg}")
    assert passing, f"test_1 prefill FAILED — PCC below {_PCC_THRESH}: {pcc_msg}"

    x_tt.deallocate(True)
    out_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)


# ---------------------------------------------------------------------------
# Test 2 — decode step PCC > 0.99 against reference oracle
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_gated_attention_layer3_decode_step_pcc_on_8x4(mesh_8x4):
    """TtQwen36GatedAttention decode step (T=1) PCC > 0.99 vs CPU reference oracle.

    Uses real layer-3 weights. Seeds KV cache with a prefill pass (T=32),
    then runs a single decode step (T=1) and checks the output PCC.
    """
    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    sd = _load_layer_weights()

    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_B, max_seq_len=512)

    attn = TtQwen36GatedAttention(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
    )

    # ------- prefill pass: seed KV cache -------
    T_pre = _T_PREFILL
    x_pre = _make_random_hidden_state(T_pre, seed=20)
    cos_pre, sin_pre = rope.get_cos_sin_for_prefill(seq_len=T_pre)
    x_pre_tt = _send_to_device(x_pre, mesh_8x4)
    out_pre_tt = attn.forward_prefill(x_pre_tt, rot_mats=(cos_pre, sin_pre), kv_cache=None, user_id=0)
    out_pre_tt.deallocate(True)
    x_pre_tt.deallocate(True)
    cos_pre.deallocate(True)
    sin_pre.deallocate(True)

    # ------- decode step at position T_pre -------
    cur_pos = T_pre
    x_dec = _make_random_hidden_state(1, seed=21)  # [1, 1, 5120]
    cos_dec, sin_dec = rope.get_cos_sin_for_decode(cur_pos)
    x_dec_tt = _send_to_device(x_dec, mesh_8x4)

    cur_pos_tensor = ttnn.from_torch(
        torch.tensor([cur_pos] * args.max_batch_size, dtype=torch.int32),
        device=mesh_8x4,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_8x4),
    )

    out_dec_tt = attn.forward_decode(
        x_dec_tt,
        current_pos=cur_pos_tensor,
        rot_mats=(cos_dec, sin_dec),
        page_table=None,
        kv_cache=None,
    )

    out_host = _gather_replicated(out_dec_tt, mesh_8x4)  # [1, 1, H]

    # CPU reference
    import json
    import pathlib

    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config

    cfg_path = pathlib.Path(_SNAPSHOT_DIR) / "config.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    with torch.no_grad():
        ref_attn = GatedAttention(config)
        ref_attn.eval()
        for key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            getattr(ref_attn, key).weight.data.copy_(sd[f"{key}.weight"])
        ref_attn.q_norm.weight.data.copy_(sd["q_norm.weight"])
        ref_attn.k_norm.weight.data.copy_(sd["k_norm.weight"])

        # Prefill to build KV cache reference
        cos_pre_ref, sin_pre_ref = _build_rope_cos_sin(T_pre)
        mask_pre = _build_causal_mask(T_pre)
        _, (k_cache_ref, v_cache_ref) = ref_attn(
            x_pre.float(), cos_pre_ref, sin_pre_ref, kv_cache=None, attention_mask=mask_pre
        )

        # Decode at position T_pre
        cos_dec_ref, sin_dec_ref = _build_rope_cos_sin(cur_pos + 1)
        cos_dec_ref = cos_dec_ref[:, cur_pos : cur_pos + 1, :]  # [1, 1, 64]
        sin_dec_ref = sin_dec_ref[:, cur_pos : cur_pos + 1, :]
        ref_dec_out, _ = ref_attn(
            x_dec.float(), cos_dec_ref, sin_dec_ref, kv_cache=(k_cache_ref, v_cache_ref), attention_mask=None
        )

    passing, pcc_msg = comp_pcc(ref_dec_out.bfloat16(), out_host.bfloat16(), pcc=_PCC_THRESH)
    print(f"\n[test_2] decode step PCC: {pcc_msg}")
    assert passing, f"test_2 decode FAILED — PCC below {_PCC_THRESH}: {pcc_msg}"

    x_dec_tt.deallocate(True)
    out_dec_tt.deallocate(True)
    cos_dec.deallocate(True)
    sin_dec.deallocate(True)


# ---------------------------------------------------------------------------
# Test 3 — output gate is active (zero-gate → output ≈ ref-with-zero-gate × 0.5)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_gated_attention_output_gate_active(mesh_8x4):
    """Sanity test: gate is actually applied.

    We create a state dict where the gate-producing rows of q_proj are zeroed out.
    sigmoid(0) = 0.5, so the output should be ~half of the normal output norm.
    We verify:
    1. TTNN PCC vs CPU reference with same zero-gate weights > 0.99
    2. ||zero-gate output|| < 0.9 × ||real output|| (gate is attenuating)
    """
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    sd_real = _load_layer_weights()

    # Build zero-gate state dict: zero out gate rows in q_proj
    # q_proj.weight: [n_q * 2 * head_dim, H] = [12288, 5120]
    # Per-head interleaved: Q_h0(256) | gate_h0(256) | Q_h1(256) | gate_h1(256) | ...
    # Gate rows for head h: rows [h*2*hd + hd : h*2*hd + 2*hd]
    sd_zero_gate = dict(sd_real)
    q_w = sd_real["q_proj.weight"].clone()
    hd = _HEAD_DIM
    n_q = _N_Q
    for h in range(n_q):
        start = h * 2 * hd + hd
        end = h * 2 * hd + 2 * hd
        q_w[start:end, :] = 0.0
    sd_zero_gate["q_proj.weight"] = q_w

    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_B, max_seq_len=512)
    cos_tt, sin_tt = rope.get_cos_sin_for_prefill(seq_len=_T_PREFILL)

    attn_gz = TtQwen36GatedAttention(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd_zero_gate,
        layer_num=_LAYER_IDX,
    )

    x_torch = _make_random_hidden_state(_T_PREFILL, seed=30)
    x_tt = _send_to_device(x_torch, mesh_8x4)

    out_gz_tt = attn_gz.forward_prefill(x_tt, rot_mats=(cos_tt, sin_tt), kv_cache=None)
    out_gz_host = _gather_replicated(out_gz_tt, mesh_8x4)

    # PCC vs CPU reference with same zero-gate weights
    cos_ref, sin_ref = _build_rope_cos_sin(_T_PREFILL)
    mask = _build_causal_mask(_T_PREFILL)
    ref_gz_out = _cpu_reference_attention(sd_zero_gate, x_torch, cos_ref, sin_ref, mask)

    passing, pcc_msg = comp_pcc(ref_gz_out.bfloat16(), out_gz_host.bfloat16(), pcc=_PCC_THRESH)
    print(f"\n[test_3] zero-gate PCC vs oracle: {pcc_msg}")
    assert passing, f"test_3 gate sanity FAILED — PCC below {_PCC_THRESH}: {pcc_msg}"

    # Verify the gate actually attenuates: ||zero-gate|| < 0.9 × ||real||
    ref_real_out = _cpu_reference_attention(
        sd_real, x_torch, _build_rope_cos_sin(_T_PREFILL)[0], _build_rope_cos_sin(_T_PREFILL)[1], mask
    )
    gz_norm = ref_gz_out.norm().item()
    real_norm = ref_real_out.norm().item()
    print(f"[test_3] ||zero-gate||={gz_norm:.3f}, ||real||={real_norm:.3f}")
    assert (
        gz_norm < real_norm * 0.9
    ), f"Gate does not appear active: ||zero-gate||={gz_norm:.3f} should be < 0.9×||real||={real_norm:.3f}"

    x_tt.deallocate(True)
    out_gz_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)


# ---------------------------------------------------------------------------
# Test 4 — zero-padded heads don't corrupt output (pad-and-slice sanity)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_gated_attention_zero_padded_heads_dont_contribute(mesh_8x4):
    """Pad-and-slice sanity: zero-padded Q/KV heads (heads 24..63) contribute zero.

    This verifies that:
    1. The TTNN output still achieves PCC > 0.99 vs the reference (which uses
       only the native 24 Q heads with no padding).
    2. The padded heads in the WO input don't corrupt the result because the
       WO columns beyond 6144 are zero-padded (zero × anything = zero).
    """
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    sd = _load_layer_weights()

    rope = Qwen36RopeSetup(mesh_8x4, args, batch_size=_B, max_seq_len=512)
    cos_tt, sin_tt = rope.get_cos_sin_for_prefill(seq_len=_T_PREFILL)

    attn = TtQwen36GatedAttention(
        mesh_device=mesh_8x4,
        args=args,
        state_dict=sd,
        layer_num=_LAYER_IDX,
    )

    x_torch = _make_random_hidden_state(_T_PREFILL, seed=40)
    x_tt = _send_to_device(x_torch, mesh_8x4)

    out_tt = attn.forward_prefill(x_tt, rot_mats=(cos_tt, sin_tt), kv_cache=None)
    out_host = _gather_replicated(out_tt, mesh_8x4)  # [1, T, H]

    # Reference uses only native 24 Q heads (no padding)
    cos_ref, sin_ref = _build_rope_cos_sin(_T_PREFILL)
    mask = _build_causal_mask(_T_PREFILL)
    ref_out = _cpu_reference_attention(sd, x_torch, cos_ref, sin_ref, mask)

    passing, pcc_msg = comp_pcc(ref_out.bfloat16(), out_host.bfloat16(), pcc=_PCC_THRESH)
    print(f"\n[test_4] pad-and-slice PCC: {pcc_msg}")
    assert passing, (
        f"test_4 pad-and-slice FAILED — PCC below {_PCC_THRESH}: {pcc_msg}. "
        f"Padded heads may be contributing incorrectly."
    )

    x_tt.deallocate(True)
    out_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)
