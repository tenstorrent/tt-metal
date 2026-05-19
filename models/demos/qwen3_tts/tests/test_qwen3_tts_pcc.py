# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Consolidated per-block PCC gate for qwen3_tts on Blackhole P150.

One test per block. Each test builds the TTNN block from current production
code, runs a small fixed input through both the TTNN and a torch reference,
computes Pearson PCC, and asserts ``pcc >= EXPECTED_PCC[name] - 0.005``.

Run:
    pytest -s -v models/demos/qwen3_tts/tests/test_qwen3_tts_pcc.py
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

# -----------------------------------------------------------------------------
# Hardcoded expected PCC per block. Measured on Blackhole P150 with current
# production code (no TT_QWEN3_* env knobs). Threshold for each test is
# EXPECTED_PCC[name] - 0.005.
# -----------------------------------------------------------------------------
EXPECTED_PCC = {
    "mlp_decode": 0.9997,
    "attention_decode": 0.9998,
    "cp_step": 0.9999,
    "talker_chain": 0.9758,
}
TOLERANCE = 0.005


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation across all elements. Returns 0.0 if either input is constant."""
    a, b = a.flatten().float(), b.flatten().float()
    ac, bc = a - a.mean(), b - b.mean()
    denom = ac.norm() * bc.norm()
    if denom < 1e-8:
        return 0.0
    return (ac * bc).sum().item() / denom.item()


def tt_to_torch(t: ttnn.Tensor, device) -> torch.Tensor:
    """Convert a ttnn tensor to torch, handling multi-device meshes.

    For a (1, N) mesh the tensor is replicated across chips; we extract chip-0's
    view (taking the first slice along the stacked axis from ConcatMeshToTensor).
    For a plain Device (or (1,1) mesh) the result is the usual single-chip conversion.
    """
    if device.__class__.__name__ == "MeshDevice" and device.get_num_devices() > 1:
        stacked = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        return stacked[0:1]  # chip-0 slice; shape unchanged except leading dim=1
    return ttnn.to_torch(t)


def _assert_pcc(name: str, pcc: float) -> None:
    threshold = EXPECTED_PCC[name] - TOLERANCE
    print(f"[{name}] measured PCC = {pcc:.6f}  expected = {EXPECTED_PCC[name]:.4f}  threshold = {threshold:.4f}")
    assert pcc >= threshold, f"[{name}] PCC {pcc:.6f} < threshold {threshold:.4f} (expected {EXPECTED_PCC[name]:.4f})"


# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    """Open a mesh device parameterized by MESH_DEVICE env (N150 / N300 / T3K).

    Default (unset) is (1,1) — N150-style single chip with 8x8 worker grid.
    For TP testing set MESH_DEVICE=N300 (1,2) or T3K (1,8).
    Multi-chip meshes need FABRIC_1D so all_reduce / all_gather can run.
    """
    import os

    mesh_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 1))
    multi_chip = mesh_shape != (1, 1)
    if multi_chip:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    yield d
    ttnn.close_mesh_device(d)
    if multi_chip:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def state_dict():
    """Load full HF state dict (excludes speech_tokenizer) via the production
    ``load_weights`` helper. Returns the main weights dict only."""
    from models.demos.qwen3_tts.tt.server import load_weights

    main_dict, _ = load_weights()
    return main_dict


# -----------------------------------------------------------------------------
# Test 1: MLP decode
# -----------------------------------------------------------------------------
def test_mlp_decode_pcc(device):
    """Talker MLP at seq_len=1 with random weights — full-precision reference."""
    from models.demos.qwen3_tts.reference.functional import swiglu_mlp as torch_swiglu_mlp
    from models.demos.qwen3_tts.tt.mlp import MLP

    torch.manual_seed(0)
    hidden, intermediate = 2048, 6144
    x_torch = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16)
    g = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    u = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    d = torch.randn(hidden, intermediate, dtype=torch.bfloat16)

    ref = torch_swiglu_mlp(x_torch.squeeze(1), g, u, d)

    sd = {
        "test_layer.mlp.gate_proj.weight": g,
        "test_layer.mlp.up_proj.weight": u,
        "test_layer.mlp.down_proj.weight": d,
    }
    mlp = MLP(device, hidden, intermediate, sd, "test_layer")

    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_tt = mlp(x_tt, mode="decode")
    y = tt_to_torch(y_tt, device).squeeze(1)
    pcc = compute_pcc(ref, y)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(y_tt)
    _assert_pcc("mlp_decode", pcc)


# -----------------------------------------------------------------------------
# Test 2: Attention decode (seq_len=1 with KV cache)
# -----------------------------------------------------------------------------
def test_attention_decode_pcc(device):
    """Talker Attention with seq_len=1, random weights, KV cache at start_pos=0.

    Uses identity RoPE so this isolates QKV / norm / SDPA numerics from RoPE
    precision. Compares against the reference attention with seq_len=1.
    """
    from models.demos.qwen3_tts.reference.functional import attention as torch_attention
    from models.demos.qwen3_tts.reference.functional import get_default_talker_config
    from models.demos.qwen3_tts.tt.attention import Attention
    from models.demos.qwen3_tts.tt.kv_cache import create_kv_cache_list
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    torch.manual_seed(42)
    cfg = get_default_talker_config()
    tt_cfg = Qwen3TTSTalkerConfig()
    tt_cfg.num_hidden_layers = 1

    H, NH, NKV, HD = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    seq_len = 1

    x_torch = torch.randn(1, seq_len, H, dtype=torch.bfloat16)
    q_w = torch.randn(NH * HD, H, dtype=torch.bfloat16)
    k_w = torch.randn(NKV * HD, H, dtype=torch.bfloat16)
    v_w = torch.randn(NKV * HD, H, dtype=torch.bfloat16)
    o_w = torch.randn(H, NH * HD, dtype=torch.bfloat16)
    qn_w = torch.ones(HD, dtype=torch.bfloat16)
    kn_w = torch.ones(HD, dtype=torch.bfloat16)

    cos_id = torch.ones(1, seq_len, HD, dtype=torch.bfloat16)
    sin_id = torch.zeros(1, seq_len, HD, dtype=torch.bfloat16)

    ref = torch_attention(
        x_torch,
        q_w,
        k_w,
        v_w,
        o_w,
        qn_w,
        kn_w,
        cos_id,
        sin_id,
        num_heads=NH,
        num_kv_heads=NKV,
        head_dim=HD,
        rms_norm_eps=cfg.rms_norm_eps,
        use_mrope=False,
    )

    sd = {
        "test_layer.self_attn.q_proj.weight": q_w,
        "test_layer.self_attn.k_proj.weight": k_w,
        "test_layer.self_attn.v_proj.weight": v_w,
        "test_layer.self_attn.o_proj.weight": o_w,
        "test_layer.self_attn.q_norm.weight": qn_w,
        "test_layer.self_attn.k_norm.weight": kn_w,
    }
    attn = Attention(
        device=device,
        hidden_size=H,
        num_heads=NH,
        num_kv_heads=NKV,
        head_dim=HD,
        state_dict=sd,
        layer_prefix="test_layer",
        rms_norm_eps=cfg.rms_norm_eps,
    )

    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt = ttnn.from_torch(
        cos_id.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_tt = ttnn.from_torch(
        sin_id.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trans = get_rot_transformation_mat(dhead=HD)
    trans_tt = ttnn.from_torch(
        trans,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kv_caches = create_kv_cache_list(device, tt_cfg, max_batch_size=1, max_seq_len=32)
    y_tt, _ = attn(x_tt, cos_tt, sin_tt, trans_tt, kv_cache=kv_caches[0], start_pos=0, mode="decode")
    y = tt_to_torch(y_tt, device).squeeze(1)
    pcc = compute_pcc(ref, y)

    ttnn.deallocate(x_tt)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
    ttnn.deallocate(trans_tt)
    ttnn.deallocate(y_tt)
    for k, v in kv_caches:
        ttnn.deallocate(k)
        ttnn.deallocate(v)

    _assert_pcc("attention_decode", pcc)


# -----------------------------------------------------------------------------
# Test 3: CodePredictor.forward_single_step
# -----------------------------------------------------------------------------
def test_code_predictor_step_pcc(device, state_dict):
    """CodePredictor at the first CP step (2-token prefill: talker hidden + cb0 embed),
    comparing TTNN hidden state against the reference ``code_predictor_forward``."""
    from models.demos.qwen3_tts.reference.functional import Qwen3TTSCodePredictorConfig as RefCPCfg
    from models.demos.qwen3_tts.reference.functional import code_predictor_forward, extract_code_predictor_weights
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    torch.manual_seed(7)
    talker_cfg = Qwen3TTSTalkerConfig()
    cp_cfg = Qwen3TTSCodePredictorConfig()
    ref_cfg = RefCPCfg()

    # Build a synthetic 2-token CP prefill input in CP hidden_size (1024).
    # The CP applies its own input_proj from talker_hidden→cp_hidden when sizes differ;
    # to keep the reference parity simple we feed an embedding already at cp_hidden_size
    # by routing through the projection on the TT side and applying the same on torch side.
    proj_w = state_dict["talker.code_predictor.small_to_mtp_projection.weight"].to(torch.bfloat16)  # [cp_h, talker_h]
    proj_b_key = "talker.code_predictor.small_to_mtp_projection.bias"
    proj_b = state_dict[proj_b_key].to(torch.bfloat16) if proj_b_key in state_dict else None

    talker_h_in = torch.randn(1, 2, talker_cfg.hidden_size, dtype=torch.bfloat16)

    # Reference: project then run CP forward (no KV cache, full prefill semantics).
    proj_w_f = proj_w.float()
    proj_b_f = proj_b.float() if proj_b is not None else None
    x_cp_ref = F.linear(talker_h_in.float(), proj_w_f, proj_b_f)  # [1, 2, cp_hidden]
    cp_weights = extract_code_predictor_weights(state_dict)
    # code_predictor_forward expects layer-relative keys (layers.{i}.*); the
    # extractor strips talker.code_predictor. but leaves the model. prefix in.
    cp_weights_f = {k.replace("model.", ""): v.float() for k, v in cp_weights.items()}
    ref_hidden = code_predictor_forward(x_cp_ref, cp_weights_f, ref_cfg)  # [1, 2, cp_hidden]

    # TTNN side: build CodePredictor and run forward_single_step in prefill mode,
    # returning the hidden state for direct comparison (no lm_head).
    cp = CodePredictor(device=device, config=cp_cfg, talker_hidden_size=talker_cfg.hidden_size, state_dict=state_dict)

    inp = ttnn.from_torch(
        talker_h_in.unsqueeze(1),  # [1, 1, 2, talker_hidden]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cp_head_dim = cp_cfg.head_dim
    cp_trans_mat = get_transformation_mat(cp_head_dim, device)
    cos_pf, sin_pf = get_rope_tensors(device, cp_head_dim, 2, torch.arange(2), cp_cfg.rope_theta)

    hidden_tt, _ = cp.forward_single_step(
        inp,
        cos_pf,
        sin_pf,
        cp_trans_mat,
        generation_step=1,
        kv_caches=None,
        start_pos=0,
        mode="prefill",
        return_hidden_state=True,
    )
    y = tt_to_torch(hidden_tt, device).squeeze(1).float()  # [1, 2, cp_hidden]
    pcc = compute_pcc(ref_hidden, y)

    ttnn.deallocate(inp)
    ttnn.deallocate(hidden_tt)
    ttnn.deallocate(cp_trans_mat)
    ttnn.deallocate(cos_pf)
    ttnn.deallocate(sin_pf)

    _assert_pcc("cp_step", pcc)


# -----------------------------------------------------------------------------
# Test 4: Full 28-layer Talker chain at decode (seq_len=1, no KV cache)
# -----------------------------------------------------------------------------
def test_talker_chain_pcc(device, state_dict):
    """Full 28-layer Talker chain at seq_len=1 — TTNN DecoderLayer output fed into
    next TTNN DecoderLayer, compared against the corresponding reference chain.

    Uses prefill mode with seq_len=1 (one tile of 32 after padding) to keep the
    test fast while exercising all 28 layers + final norm. This matches the
    chained inference shape and isolates per-block precision compounding.
    """
    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSConfig,
        compute_mrope_frequencies,
        decoder_layer,
        extract_talker_weights,
        rms_norm,
    )
    from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    torch.manual_seed(123)
    ref_cfg = Qwen3TTSConfig()
    tt_cfg = Qwen3TTSTalkerConfig()
    talker_weights = extract_talker_weights(state_dict)

    seq_len = 1
    pad_seq = 32  # tile-aligned
    H = tt_cfg.hidden_size

    inp = torch.randn(1, seq_len, H, dtype=torch.bfloat16)

    # ---- Reference 28-layer chain ----
    cos, sin = compute_mrope_frequencies(ref_cfg.head_dim, seq_len, ref_cfg.rope_theta)
    cos = cos.float()
    sin = sin.float()
    attn_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0).float()

    x = inp.float()
    for layer_idx in range(ref_cfg.num_hidden_layers):
        layer_prefix = f"layers.{layer_idx}."
        lw = {k.replace(layer_prefix, ""): v.float() for k, v in talker_weights.items() if k.startswith(layer_prefix)}
        x = decoder_layer(x, lw, cos, sin, ref_cfg, attention_mask=attn_mask, use_mrope=True)
    x_ref = rms_norm(x, talker_weights["norm.weight"].float(), ref_cfg.rms_norm_eps)

    # ---- TTNN 28-layer chain ----
    position_ids = torch.arange(pad_seq)
    cos_tt, sin_tt = get_rope_tensors(device, tt_cfg.head_dim, pad_seq, position_ids, tt_cfg.rope_theta)
    trans_mat = get_transformation_mat(tt_cfg.head_dim, device)

    layers = []
    for i in range(tt_cfg.num_hidden_layers):
        layer = DecoderLayer(
            device=device,
            hidden_size=tt_cfg.hidden_size,
            num_heads=tt_cfg.num_attention_heads,
            num_kv_heads=tt_cfg.num_key_value_heads,
            head_dim=tt_cfg.head_dim,
            intermediate_size=tt_cfg.intermediate_size,
            state_dict=state_dict,
            layer_idx=i,
            layer_prefix="talker.model",
            rms_norm_eps=tt_cfg.rms_norm_eps,
            weight_dtype=ttnn.bfloat16,
        )
        layers.append(layer)

    TILE = 32
    final_norm_w = state_dict["talker.model.norm.weight"].to(torch.bfloat16)
    final_norm_w_reshaped = final_norm_w.view(1, 1, tt_cfg.hidden_size // TILE, TILE)
    final_norm_tt = ttnn.as_tensor(
        final_norm_w_reshaped,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    inp_padded = F.pad(inp, (0, 0, 0, pad_seq - seq_len))
    inp_4d = inp_padded.unsqueeze(1).to(torch.bfloat16)
    hidden_tt = ttnn.from_torch(
        inp_4d, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for layer in layers:
        hidden_tt, _ = layer(
            hidden_tt, cos_tt, sin_tt, trans_mat, attention_mask=None, kv_cache=None, start_pos=0, mode="prefill"
        )
    hidden_tt = ttnn.rms_norm(hidden_tt, epsilon=tt_cfg.rms_norm_eps, weight=final_norm_tt)
    x_tt = tt_to_torch(hidden_tt, device).squeeze(1).float()[:, :seq_len, :]

    ttnn.deallocate(hidden_tt)

    pcc = compute_pcc(x_ref, x_tt)
    _assert_pcc("talker_chain", pcc)
