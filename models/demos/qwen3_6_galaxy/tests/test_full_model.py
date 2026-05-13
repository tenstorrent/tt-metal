# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD tests for TtQwen36Transformer full model — Task 8.

Tests are written RED-first (before implementation) and run GREEN after
TtQwen36Transformer is implemented.

Hardware required: 32-chip BH GLX 8×4 mesh.

Run all tests:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_full_model.py -x -s -v

Logging:
    All runs use: 2>&1 | tee /tmp/qwen36_logs/t8_<step>.log
"""

import json
import pathlib
import sys
import time

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

_B = 1
_T_PREFILL = 32
_H = 5120
_PCC_THRESH = 0.99

# 4-layer hidden-state PCC threshold on real embedding activations: 0.99 (strict).
# Achieved 0.9934 after promoting DistributedNorm + MLP + attention + DeltaNet
# compute kernels to HiFi4 + fp32_dest_acc_en.  (The previous HiFi2/bf16-acc
# DistributedNorm was the precision bottleneck: rsqrt of low variance on small-
# magnitude embedding activations needs fp32 accumulation.)
_PCC_THRESH_4LAYER = 0.99

# 16-layer hidden-state PCC threshold: 0.995.
# T11b: do not lower this threshold; the precision fix must achieve it.
# The prior agent (T11) blanket-applied HiFi4+fp32_dest_acc and claimed a "BF16 floor"
# without per-op measurement data, then lowered this threshold from 0.995 to 0.98.
# T11b task: per-op PCC diagnostic to identify the specific op causing the major PCC drop,
# then apply a targeted fix so this test passes at 0.995.
_PCC_THRESH_16LAYER = 0.995


# ---------------------------------------------------------------------------
# Fabric mesh fixture
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
# Weight loading helper
# ---------------------------------------------------------------------------


def _load_layer_weights(layer_idx: int) -> dict:
    """Load all weights for a single decoder layer from safetensors.

    Returns a flat dict with keys in the form expected by TtQwen36DecoderLayer:
      input_layernorm.weight, post_attention_layernorm.weight,
      mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight,
      self_attn.*  or  linear_attn.*  depending on layer_type
    """
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    pfx = f"model.language_model.layers.{layer_idx}"
    keys_needed = [k for k in weight_map if k.startswith(pfx + ".")]

    files_needed = sorted({weight_map[k] for k in keys_needed})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in keys_needed:
            if k in shard:
                raw[k] = shard[k].float()

    result = {}
    for k, v in raw.items():
        short = k[len(pfx) + 1 :]  # strip "model.language_model.layers.N."
        result[short] = v
    return result


def _load_embedding_and_norm_and_head_weights() -> dict:
    """Load embedding, final norm, and lm_head weights from safetensors.

    Returns a dict with keys:
      tok_embeddings.weight   [vocab_size, hidden_size]
      norm.weight             [hidden_size]
      output.weight           [vocab_size, hidden_size]
    """
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    needed = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]

    files_needed = sorted({weight_map[k] for k in needed if k in weight_map})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in needed:
            if k in shard:
                raw[k] = shard[k].float()

    return {
        "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
        "norm.weight": raw["model.language_model.norm.weight"],
        "output.weight": raw["lm_head.weight"],
    }


# ---------------------------------------------------------------------------
# Reference model helpers
# ---------------------------------------------------------------------------


def _load_config():
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        d = json.load(f)
    return Qwen36Config(d)


def _ref_forward_hidden(ref_model, x_embed, cos, sin, attention_mask=None):
    """Run the reference model's decoder layers starting from a pre-computed embedding.

    This is the true block-level correctness metric — PCC > 0.99 here proves
    the full transformer stack (decoder layers) is numerically correct, consistent
    with T7's proven 0.999949 PCC.

    By accepting ``x_embed`` (the BF16 embedding gathered from the TTNN device) as
    input, we isolate the decoder layer comparison from embedding precision.  This
    mirrors the T7 decoder test design where both reference and TTNN started from
    the SAME BF16 input tensor.

    Args:
        ref_model: Qwen36TextModel reference model.
        x_embed:   float32 tensor [B, T, H] — embedding output gathered from TTNN device
                   (BF16 values upcast to float32; identical starting point as TTNN decoder).
        cos, sin:  MRoPE tensors.
        attention_mask: optional causal mask.

    Returns:
        float32 tensor [B, T, hidden_size] — pre-final-norm decoder output.
    """
    x = x_embed.clone()  # float32, same values as TTNN BF16 embedding
    with torch.no_grad():
        for layer in ref_model.layers:
            x, _, _, _ = layer(x, cos, sin, attention_mask, None, None, None)
    return x.float()  # [B, T, hidden_size] — pre-final-norm


def _build_rope_cos_sin(T: int):
    """Build MRoPE cos/sin for text positions [0, T)."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    return cos, sin  # [1, T, 64]


def _build_ref_model(config, num_layers: int, global_weights: dict, layers_weights: list):
    """Build a Qwen36TextModel with real weights, sliced to num_layers."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36TextModel

    # Build config slice
    sliced_config = type(config).__new__(type(config))
    sliced_config.__dict__.update(config.__dict__)
    sliced_config.num_hidden_layers = num_layers
    sliced_config.layer_types = config.layer_types[:num_layers]

    model = Qwen36TextModel(sliced_config)
    model.eval()

    with torch.no_grad():
        # Embedding
        model.tok_embeddings.weight.data.copy_(global_weights["tok_embeddings.weight"][: config.vocab_size])
        # Final norm
        model.norm.weight.data.copy_(global_weights["norm.weight"])
        # LM head
        model.lm_head.weight.data.copy_(global_weights["output.weight"][: config.vocab_size])

        # Layer weights
        for i in range(num_layers):
            w = layers_weights[i]
            layer = model.layers[i]
            layer.input_layernorm.weight.data.copy_(w["input_layernorm.weight"])
            layer.post_attention_layernorm.weight.data.copy_(w["post_attention_layernorm.weight"])
            layer.mlp.gate_proj.weight.data.copy_(w["mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight.data.copy_(w["mlp.up_proj.weight"])
            layer.mlp.down_proj.weight.data.copy_(w["mlp.down_proj.weight"])

            lt = config.layer_types[i]
            if lt == "full_attention":
                attn = layer.attention
                attn.q_proj.weight.data.copy_(w["self_attn.q_proj.weight"])
                attn.k_proj.weight.data.copy_(w["self_attn.k_proj.weight"])
                attn.v_proj.weight.data.copy_(w["self_attn.v_proj.weight"])
                attn.o_proj.weight.data.copy_(w["self_attn.o_proj.weight"])
                attn.q_norm.weight.data.copy_(w["self_attn.q_norm.weight"])
                attn.k_norm.weight.data.copy_(w["self_attn.k_norm.weight"])
            else:
                dn = layer.attention
                dn.in_proj_qkv.weight.data.copy_(w["linear_attn.in_proj_qkv.weight"])
                dn.in_proj_z.weight.data.copy_(w["linear_attn.in_proj_z.weight"])
                dn.in_proj_a.weight.data.copy_(w["linear_attn.in_proj_a.weight"])
                dn.in_proj_b.weight.data.copy_(w["linear_attn.in_proj_b.weight"])
                dn.conv1d.weight.data.copy_(w["linear_attn.conv1d.weight"])
                dn.A_log.data.copy_(w["linear_attn.A_log"])
                dn.dt_bias.data.copy_(w["linear_attn.dt_bias"])
                dn.norm.weight.data.copy_(w["linear_attn.norm.weight"])
                dn.out_proj.weight.data.copy_(w["linear_attn.out_proj.weight"])

    return model


# ---------------------------------------------------------------------------
# PCC helper
# ---------------------------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    cc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return cc


# ---------------------------------------------------------------------------
# TTNN model builder helper
# ---------------------------------------------------------------------------


def _build_tt_model(mesh_device, args, num_layers: int, global_weights: dict, layers_weights: list):
    """Build TtQwen36Transformer with real weights and num_layers layers."""
    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer

    return TtQwen36Transformer(
        mesh_device=mesh_device,
        args=args,
        global_weights=global_weights,
        layers_weights=layers_weights,
        num_layers=num_layers,
        dtype=None,  # default bfloat16
    )


# ---------------------------------------------------------------------------
# Test 1: test_full_model_4layer_prefill_pcc_on_8x4
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_full_model_4layer_prefill_pcc_on_8x4(mesh_8x4):
    """TtQwen36Transformer 4-layer prefill: hidden-state PCC > 0.99 + top-1 match.

    Builds a 4-layer model (embedding + 4 decoders + final norm + lm_head),
    runs prefill on random input_ids [B=1, T=32], and asserts:
    - PCC of hidden state (after final norm, before LM head) > 0.99
      (the true block-level correctness metric; logits PCC ~0.965 is a known
      BF16 LM head precision artefact over a 248320-dim vocab, NOT a model bug)
    - Top-1 token at last position matches CPU reference (generation correctness)
    """
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 4
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[Test1] Loading weights for {N_LAYERS} layers + embedding/norm/head...")
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    # Random input_ids [B=1, T=32]
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (_B, _T_PREFILL))

    # --- CPU reference ---
    print("[Test1] Building CPU reference model...")
    ref_model = _build_ref_model(config, N_LAYERS, global_weights, layers_weights)
    cos, sin = _build_rope_cos_sin(_T_PREFILL)
    causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
    )

    # Reference: logits for top-1 check
    with torch.no_grad():
        ref_logits = ref_model(input_ids, cos, sin, attention_mask=causal_mask)  # [B, T, vocab]
    ref_last_logits = ref_logits[0, -1, :]  # [vocab_size]
    ref_top1 = ref_last_logits.argmax().item()
    print(f"[Test1] CPU ref logits computed, top-1 token: {ref_top1}")

    # --- TTNN model ---
    print("[Test1] Building TTNN model...")
    tt_model = _build_tt_model(mesh_8x4, args, N_LAYERS, global_weights, layers_weights)

    # Hidden-state PCC — the primary correctness metric.
    # forward_prefill_hidden returns (emb_cpu, hidden_cpu) where:
    #   emb_cpu    = BF16 embedding output (from device), as float32
    #   hidden_cpu = pre-final-norm hidden state after all decoder layers
    print("[Test1] Running TT forward_prefill_hidden...")
    t0 = time.time()
    tt_emb, tt_hidden = tt_model.forward_prefill_hidden(input_ids)
    t1 = time.time()
    print(f"[Test1] TT hidden done in {t1-t0:.2f}s. Hidden shape: {tt_hidden.shape}")

    # Reference hidden: run decoder layers from the SAME BF16 embedding values
    # (matches T7 decoder test design — identical starting point for fair comparison)
    ref_hidden = _ref_forward_hidden(ref_model, tt_emb, cos, sin, attention_mask=causal_mask)

    hidden_pcc = _pcc(tt_hidden, ref_hidden)
    print(f"[Test1] Hidden-state PCC = {hidden_pcc:.6f}  (thresh={_PCC_THRESH_4LAYER})")

    # Top-1 generation correctness — use the LM head path
    print("[Test1] Running TT forward_prefill for top-1 check...")
    t0 = time.time()
    tt_logits = tt_model.forward_prefill(input_ids)  # [B, T, padded_vocab] torch tensor
    t1 = time.time()
    print(f"[Test1] TT prefill done in {t1-t0:.2f}s. Logits shape: {tt_logits.shape}")

    tt_last_logits = tt_logits[0, -1, : config.vocab_size]  # [vocab_size]
    tt_top1 = tt_last_logits.argmax().item()
    # Also report logits PCC (informational — expected ~0.965 due to BF16 LM head)
    logits_pcc = _pcc(tt_last_logits, ref_last_logits.float())
    print(f"[Test1] Logits PCC = {logits_pcc:.6f}  (informational; BF16 LM head artefact ~0.965)")
    print(f"[Test1] Top-1: TT={tt_top1}, ref={ref_top1}, match={tt_top1 == ref_top1}")

    assert hidden_pcc > _PCC_THRESH_4LAYER, f"4-layer prefill hidden-state PCC {hidden_pcc:.4f} < {_PCC_THRESH_4LAYER}"
    assert tt_top1 == ref_top1, f"4-layer top-1 mismatch: TT={tt_top1} vs ref={ref_top1}"
    print("[Test1] PASSED")


# ---------------------------------------------------------------------------
# Test 2: test_full_model_16layer_prefill_pcc_on_8x4
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_full_model_16layer_prefill_pcc_on_8x4(mesh_8x4):
    """TtQwen36Transformer 16-layer prefill: hidden-state PCC > 0.995 + >= 75% top-1 match.

    Builds a 16-layer model, runs prefill on random input_ids [B=1, T=32],
    asserts:
    - Hidden-state PCC > 0.995 (T11b: targeted precision fix after per-op PCC diagnostic)
    - >= 75% position-wise top-1 token match (generation correctness floor)
    """
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 16
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[Test2] Loading weights for {N_LAYERS} layers + embedding/norm/head...")
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    torch.manual_seed(43)
    input_ids = torch.randint(0, config.vocab_size, (_B, _T_PREFILL))

    # --- CPU reference ---
    print("[Test2] Building CPU reference model...")
    ref_model = _build_ref_model(config, N_LAYERS, global_weights, layers_weights)
    cos, sin = _build_rope_cos_sin(_T_PREFILL)
    causal_mask = torch.zeros(1, 1, _T_PREFILL, _T_PREFILL)
    causal_mask = causal_mask.masked_fill(
        torch.triu(torch.ones(_T_PREFILL, _T_PREFILL), diagonal=1).bool(), float("-inf")
    )

    # Reference: logits for top-1 match rate
    with torch.no_grad():
        ref_logits = ref_model(input_ids, cos, sin, attention_mask=causal_mask)
    ref_top1_all = ref_logits[0, :, : config.vocab_size].argmax(dim=-1)  # [T]
    ref_last_logits = ref_logits[0, -1, :]

    # --- TTNN model ---
    print("[Test2] Building TTNN model...")
    tt_model = _build_tt_model(mesh_8x4, args, N_LAYERS, global_weights, layers_weights)

    # Hidden-state PCC — primary metric
    # forward_prefill_hidden returns (emb_cpu, hidden_cpu) for fair comparison
    print("[Test2] Running TT forward_prefill_hidden...")
    t0 = time.time()
    tt_emb, tt_hidden = tt_model.forward_prefill_hidden(input_ids)
    t1 = time.time()
    print(f"[Test2] TT hidden done in {t1-t0:.2f}s. Hidden shape: {tt_hidden.shape}")

    # Reference hidden: run decoder layers from SAME BF16 embedding values (T7-style)
    ref_hidden = _ref_forward_hidden(ref_model, tt_emb, cos, sin, attention_mask=causal_mask)

    hidden_pcc = _pcc(tt_hidden, ref_hidden)
    print(f"[Test2] Hidden-state PCC = {hidden_pcc:.6f}  (thresh={_PCC_THRESH_16LAYER})")

    # Top-1 generation correctness
    print("[Test2] Running TT forward_prefill for top-1 check...")
    t0 = time.time()
    tt_logits = tt_model.forward_prefill(input_ids)
    t1 = time.time()
    print(f"[Test2] TT prefill done in {t1-t0:.2f}s. Logits shape: {tt_logits.shape}")

    tt_last_logits = tt_logits[0, -1, : config.vocab_size]
    tt_top1_all = tt_logits[0, :, : config.vocab_size].argmax(dim=-1)  # [T]

    # Also report logits PCC (informational)
    logits_pcc = _pcc(tt_last_logits, ref_last_logits.float())
    top1_match_rate = (tt_top1_all == ref_top1_all).float().mean().item()
    print(f"[Test2] Logits PCC = {logits_pcc:.6f}  (informational; BF16 LM head artefact; 16-layer floor ~0.81)")
    print(f"[Test2] Top-1 match rate: {top1_match_rate:.4f} (require >= 0.75)")

    assert (
        hidden_pcc > _PCC_THRESH_16LAYER
    ), f"16-layer prefill hidden-state PCC {hidden_pcc:.4f} < {_PCC_THRESH_16LAYER}"
    # 16-layer BF16 bring-up floor: top-1 match rate around 87% post-norm-fix.
    # 75% (24/32) is a safe floor that proves the model assembly is correct.
    assert top1_match_rate >= 0.75, f"Top-1 match rate {top1_match_rate:.4f} < 0.75"
    print("[Test2] PASSED")


# ---------------------------------------------------------------------------
# Test 3: test_full_model_64layer_paris_generation_on_8x4
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_full_model_64layer_paris_generation_on_8x4(mesh_8x4):
    """TtQwen36Transformer 64-layer (FULL MODEL): generates ' Paris' after prompt.

    Headline test: prefills 'The capital of France is', takes argmax of last
    position's logits → next token, asserts decoded token contains 'paris'
    (case-insensitive substring).
    """
    from transformers import AutoTokenizer

    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 64
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[Test3] Loading weights for {N_LAYERS} layers (FULL MODEL) ...")
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    # Tokenize prompt
    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT_DIR), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # [1, T]
    T_prompt = input_ids.shape[-1]
    print(f"[Test3] Prompt tokens: {T_prompt} tokens: {input_ids.tolist()}")

    # Pad to nearest multiple of 32 for TTNN tile alignment
    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids

    # --- TTNN model ---
    print("[Test3] Building TTNN model (full 64-layer)...")
    tt_model = _build_tt_model(mesh_8x4, args, N_LAYERS, global_weights, layers_weights)

    print("[Test3] Running TT prefill (FULL MODEL)...")
    t0 = time.time()
    tt_logits = tt_model.forward_prefill(input_ids_padded)  # [B, T_padded, vocab_size]
    t1 = time.time()
    wall_clock = t1 - t0
    print(f"[Test3] TT prefill done in {wall_clock:.2f}s")

    # Get next token from last real prompt position
    # Position T_prompt - 1 is the last real token
    last_logits = tt_logits[0, T_prompt - 1, : config.vocab_size]  # [vocab_size]
    next_token_id = last_logits.argmax().item()
    decoded = tokenizer.decode([next_token_id])
    print(f"[Test3] Next token id: {next_token_id}, decoded: '{decoded}'")
    print(f"[Test3] Wall-clock prefill time: {wall_clock:.2f}s")

    assert (
        "paris" in decoded.lower()
    ), f"Expected decoded token to contain 'paris', got '{decoded}' (token_id={next_token_id})"
    print("[Test3] PASSED — generated ' Paris'")


# ---------------------------------------------------------------------------
# Test 4: test_full_model_greedy_decode_5_tokens (optional)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_full_model_greedy_decode_5_tokens(mesh_8x4):
    """TtQwen36Transformer: prefill 5-token prompt then greedily decode 5 more tokens.

    After prefill, runs 5 decode steps each appending the argmax.
    Verifies: no NaN/Inf in logits at each step, and prints the 6-token sequence.
    """
    from transformers import AutoTokenizer

    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 64
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[Test4] Loading weights for {N_LAYERS} layers (FULL MODEL + decode)...")
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT_DIR), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]

    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids

    print("[Test4] Building TTNN model...")
    tt_model = _build_tt_model(mesh_8x4, args, N_LAYERS, global_weights, layers_weights)

    print("[Test4] Running TT prefill...")
    t0 = time.time()
    tt_logits, kv_caches, dn_states, conv_states = tt_model.forward_prefill(input_ids_padded, return_caches=True)
    t1 = time.time()
    print(f"[Test4] Prefill done in {t1-t0:.2f}s")

    # Get first new token
    last_logits = tt_logits[0, T_prompt - 1, : config.vocab_size]
    generated = [last_logits.argmax().item()]
    print(f"[Test4] Token 1: id={generated[-1]}, text='{tokenizer.decode([generated[-1]])}'")

    # Decode loop: 4 more tokens (total 5 new tokens)
    current_pos = T_padded  # next decode position
    for step in range(1, 5):
        next_id = torch.tensor([[generated[-1]]])
        step_logits, kv_caches, dn_states, conv_states = tt_model.forward_decode(
            next_id,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
        )
        # step_logits: [B, 1, vocab_size]
        step_last = step_logits[0, 0, : config.vocab_size]

        # Check for NaN/Inf
        assert not torch.isnan(step_last).any(), f"NaN in step {step+1} logits"
        assert not torch.isinf(step_last).any(), f"Inf in step {step+1} logits"

        next_tok = step_last.argmax().item()
        generated.append(next_tok)
        current_pos += 1
        print(f"[Test4] Token {step+1}: id={next_tok}, text='{tokenizer.decode([next_tok])}'")

    full_text = tokenizer.decode(generated)
    print(f"[Test4] Generated sequence ({len(generated)} tokens): '{full_text}'")
    print("[Test4] PASSED — all tokens valid (no NaN/Inf)")
