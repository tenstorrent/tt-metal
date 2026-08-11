# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for full Gemma4 model.

Uses HuggingFace Gemma4 classes as reference for PCC comparison.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    num_layers_for_full_attention_group,
    parametrize_mesh_with_fabric,
)


def _skip_if_config_only_model(model_path):
    """Skip real-weight smoke tests when HF_MODEL points at bundled configs."""
    path = Path(model_path)
    if not path.is_dir():
        return
    has_weights = (
        any(path.glob("*.safetensors"))
        or (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or (path / "pytorch_model.bin.index.json").exists()
    )
    if not has_weights:
        pytest.skip(f"{model_path} contains only config files; real model weights are required")


# ── Config Tests ───────────────────────────────────────────────────────────


def test_model_config():
    """Test that Gemma4ModelArgs loads correctly from HF_MODEL checkpoint."""
    hf_config = TestFactory.create_hf_config()

    # Core fields must be populated
    assert hf_config.hidden_size > 0
    assert hf_config.num_hidden_layers > 0
    assert hf_config.num_attention_heads > 0
    assert hf_config.num_key_value_heads > 0
    assert hf_config.head_dim > 0
    assert hf_config.vocab_size > 0
    assert hf_config.tie_word_embeddings is True

    # Layer types must cover all layers
    assert len(hf_config.layer_types) == hf_config.num_hidden_layers
    assert all(lt in ("sliding_attention", "full_attention") for lt in hf_config.layer_types)

    # MoE fields: either fully populated or all zero
    if hf_config.enable_moe_block:
        assert hf_config.num_experts > 0
        assert hf_config.top_k_experts > 0
        assert hf_config.moe_intermediate_size > 0
    else:
        # Dense model — MoE fields should be zero/None
        assert hf_config.num_experts == 0 or hf_config.num_experts is None

    logger.info(
        f"Model: hidden={hf_config.hidden_size}, layers={hf_config.num_hidden_layers}, "
        f"heads={hf_config.num_attention_heads}, moe={hf_config.enable_moe_block}"
    )


def test_model_instantiation():
    """Test that Gemma4Model can be imported and config created."""
    hf_config = TestFactory.create_hf_config()
    assert hf_config.hidden_size > 0
    assert hf_config.num_hidden_layers > 0


def test_softcapping(reset_seeds):
    """Test logit softcapping matches tanh(x/cap)*cap."""
    cap = 30.0
    x = torch.randn(1, 1, 32, 100, dtype=torch.float32) * 100
    expected = torch.tanh(x / cap) * cap
    assert expected.abs().max() <= cap + 1e-5
    small = torch.randn(1, 1, 32, 100) * 0.1
    small_capped = torch.tanh(small / cap) * cap
    assert torch.allclose(small_capped, small, atol=1e-3)


# ── HF Reference Helpers ──────────────────────────────────────────────────


def _create_hf_text_config(num_experts=None, top_k=None, vocab_size=256, num_layers=1):
    """Create a Gemma4TextConfig from real model config with overrides for speed."""
    from transformers import AutoConfig

    from ...tests.test_factory import _get_model_path

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = config.text_config
    if num_experts is not None:
        tc.num_experts = num_experts
    if top_k is not None:
        tc.top_k_experts = top_k
    tc.vocab_size = vocab_size
    tc.num_hidden_layers = num_layers
    # Disable per-layer input for now (not yet implemented in TT)
    tc.hidden_size_per_layer_input = 0
    tc._attn_implementation = "eager"
    return tc


def _create_hf_model(hf_text_config):
    """Create HF Gemma4 text model with random weights."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextScaledWordEmbedding,
    )

    # Build a minimal model: embedding + 1 layer + norm + lm_head
    class HFRefModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_tokens = Gemma4TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id if hasattr(config, "pad_token_id") else 0,
                embed_scale=config.hidden_size**0.5,
            )
            self.layers = torch.nn.ModuleList(
                [Gemma4TextDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
            )
            self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # Tied lm_head
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None):
            from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

            x = self.embed_tokens(input_ids)
            seq_len = x.shape[1]
            pli_size = getattr(self.config, "hidden_size_per_layer_input", 0) or 0
            pli = torch.randn(1, seq_len, pli_size) if pli_size else None

            # Create RoPE per layer type (sliding vs global have different head_dim)
            rope = Gemma4TextRotaryEmbedding(self.config)
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            x_dummy = torch.randn(1, seq_len, self.config.hidden_size)
            rope_cache = {}
            for layer_type in set(self.config.layer_types[: self.config.num_hidden_layers]):
                rope_cache[layer_type] = rope(x_dummy, pos_ids, layer_type=layer_type)

            shared_kv_states = {}
            for i, layer in enumerate(self.layers):
                layer_type = self.config.layer_types[i]
                x = layer(
                    x,
                    per_layer_input=pli,
                    position_embeddings=rope_cache[layer_type],
                    attention_mask=attention_mask,
                    shared_kv_states=shared_kv_states,
                )
            x = self.norm(x)
            logits = self.lm_head(x)
            # Softcapping
            cap = self.config.final_logit_softcapping
            if cap and cap > 0:
                logits = torch.tanh(logits / cap) * cap
            return logits

    model = HFRefModel(hf_text_config)
    # Randomize router/expert weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if any(k in name for k in ["router", "experts"]):
                if "scale" in name:
                    param.data.fill_(1.0)
                else:
                    param.data.normal_(0, 0.02)
        # Tie lm_head to embeddings
        model.lm_head.weight = model.embed_tokens.weight
        # Set layer_scalar to 1.0
        for layer in model.layers:
            layer.layer_scalar.fill_(1.0)
    model.eval()
    return model


def _hf_model_state_to_tt_state(hf_model):
    """Convert HF model state_dict to format our TT Gemma4Model expects."""
    state = hf_model.state_dict()
    tt_state = {}
    for k, v in state.items():
        # Map: "layers.0.xxx" -> "model.layers.0.xxx"
        # Map: "embed_tokens.weight" -> "model.embed_tokens.weight"
        # Map: "norm.weight" -> "model.norm.weight"
        tt_state[f"model.{k}"] = v
    return tt_state


# ── Single Layer Model PCC Test ───────────────────────────────────────────


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_group", ["sliding_only", "full_group"])
def test_single_layer_model(mesh_device, layer_group, reset_seeds, request):
    """Test few-layer model with random weights against HF reference.

    layer_group:
      - "sliding_only": 1 layer — exercises just the first (sliding) layer.
      - "full_group":   smallest prefix that includes one full-attention layer.
                        Resolved at runtime from the model's layer_types: E2B
                        repeats (sliding x4, full x1) so it needs 5 layers;
                        the larger variants repeat (sliding x5, full x1) so
                        they need 6.

    Uses small vocab (256) and random weights for fast execution.

        pytest -k "1x1 and sliding_only"   # single card, sliding only
        pytest -k "1x8 and full_group"     # T3K, through first full-attn layer
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    # Resolve layer count from the model — see helper docstring for why this
    # depends on the variant. Doing it inside the test (vs at parametrize time)
    # avoids loading the HF config during pytest collection.
    base_config = _create_hf_text_config(vocab_size=256, num_layers=1)
    if layer_group == "sliding_only":
        num_layers = 1
    else:
        num_layers = num_layers_for_full_attention_group(base_config)

    # full_group includes a full-attention layer (head_dim=512) — skip on single
    # device if that overflows L1 on this model.
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if layer_group == "full_group" and tp == 1:
        hf_config_check = TestFactory.create_hf_config()
        if hf_config_check.hidden_size > 4096:
            pytest.skip("Global attention head_dim=512 overflows L1 on single device for large models")

    base_config = _create_hf_text_config(vocab_size=256, num_layers=num_layers)
    is_moe = getattr(base_config, "enable_moe_block", False)
    if is_moe:
        base_config.num_experts = 4
        base_config.top_k_experts = 2
    hf_text_config = base_config
    hf_model = _create_hf_model(hf_text_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config  # Enables internal per-layer RoPE
    tt_state = _hf_model_state_to_tt_state(hf_model)

    seq_len = 32
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=tt_state,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=seq_len,
        max_local_batch_size=1,
        num_layers=num_layers,
    )

    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)

    # HF reference
    causal_mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        hf_logits = hf_model(tokens, attention_mask=causal_mask)

    # TT forward
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    tt_tokens = ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    tt_embeds = tt_model.embed_tokens(tt_tokens)
    tt_embeds = ttnn.reshape(tt_embeds, (1, 1, seq_len, model_args.hidden_size))
    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)

    # Don't pass rope_mats — let model use internal per-layer-type RoPE caches
    tt_logits = tt_model(tt_embeds, rope_mats=None, position_idx=None, page_table=None, kv_caches=None, is_decode=False)
    tt_logits_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]) if is_mesh else ttnn.to_torch(tt_logits))
        .squeeze(0)
        .float()
    )

    passing, pcc_msg = compare_tensors(tt_logits_torch, hf_logits, pcc_threshold=get_pcc_threshold(request))
    logger.info(f"TP={tp} group={layer_group} ({num_layers}-layer) PCC: {pcc_msg}")
    assert passing, f"Single-layer model (group={layer_group}, layers={num_layers}, tp={tp}) PCC too low: {pcc_msg}"


# ── Full Model PCC Test ─────────────────────────────────────────────────


@pytest.mark.gemma4_hf_direct_parity
@parametrize_mesh_with_fabric()
def test_full_model(mesh_device, reset_seeds, request):
    """Test full model (all layers, real weights) against HuggingFace reference.

    Runs on any mesh where the model fits in DRAM. Smaller models (E2B, E4B)
    fit on single device; larger models (A4B, 31B) require TP>=2.

        pytest -k "1x1"   # E2B/E4B on single card
        pytest -k "1x8"   # all models on T3K
    """
    import os

    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.gemma4.tt.common import create_tt_model

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )

    # Skip if model is too large for this mesh — estimate DRAM from config
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    hf_config_check = TestFactory.create_hf_config()
    is_moe = getattr(hf_config_check, "enable_moe_block", False)
    # MoE experts are replicated: ~764 MB/layer at bf8. Dense MLP: ~3*H*I/TP*2 bytes.
    if is_moe and tp < 8:
        pytest.skip(f"MoE model too large for TP={tp} (expert weights replicated)")
    if hf_config_check.hidden_size > 4096 and tp < 2:
        pytest.skip(f"Model too large for single device (hidden={hf_config_check.hidden_size})")

    # ── HF reference ─────────────────────────────────────────────────
    logger.info(f"Loading HF reference model from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # [1, seq_len]
    seq_len = input_ids.shape[1]
    padded_len = ((seq_len + 31) // 32) * 32
    if padded_len > seq_len:
        input_ids_padded = F.pad(input_ids, (0, padded_len - seq_len), value=0)
    else:
        input_ids_padded = input_ids

    logger.info(f"Prompt: '{prompt}' -> {seq_len} tokens (padded to {padded_len})")

    with torch.no_grad():
        hf_out = hf_model(input_ids_padded)
        hf_logits = hf_out.logits.float()  # [1, padded_len, vocab_size]

    # Note: HF Gemma4ForConditionalGeneration already applies softcapping internally,
    # so no need to apply it again here. TT model also applies it internally.

    logger.info(f"HF logits shape: {hf_logits.shape}, range: [{hf_logits.min():.4f}, {hf_logits.max():.4f}]")

    # Free HF model GPU memory (we only need state_dict for TT)
    del hf_model
    import gc

    gc.collect()

    # ── TT model ─────────────────────────────────────────────────────
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    logger.info(f"Creating TT model with all layers (TP={tp})...")
    model_args, tt_model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max(padded_len, 128),
        model_path=model_path,
        create_kv_cache=True,
    )

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    tokens_tt = ttnn.from_torch(
        input_ids_padded.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    embeds = tt_model.embed_tokens(tokens_tt)
    embeds = ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size))
    embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

    # CPU tensors for per-layer input (E2B/E4B models)
    embeds_torch = (
        F.embedding(
            input_ids_padded.long(),
            state_dict.get(
                "model.language_model.embed_tokens.weight",
                state_dict.get("model.embed_tokens.weight", torch.zeros(1)),
            ),
        )
        * tt_model.embed_scale
    ).float()

    tt_logits = tt_model.ttnn_prefill_forward(
        embeds,
        page_table=None,
        kv_cache=tt_kv_cache,
        input_ids_torch=input_ids_padded,
        embeds_torch=embeds_torch,
    )

    if is_mesh:
        tt_logits_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    else:
        tt_logits_torch = ttnn.to_torch(tt_logits).float()
    tt_logits.deallocate(True)

    # Reshape TT output to match HF: TT is [1, 1, padded_len, vocab] -> [1, padded_len, vocab]
    if tt_logits_torch.dim() == 4:
        tt_logits_torch = tt_logits_torch.squeeze(1)

    logger.info(
        f"TT logits shape: {tt_logits_torch.shape}, range: [{tt_logits_torch.min():.4f}, {tt_logits_torch.max():.4f}]"
    )

    # Compare only up to the real (unpadded) sequence length
    hf_compare = hf_logits[:, :seq_len, :]
    tt_compare = tt_logits_torch[:, :seq_len, :]

    passing, pcc_msg = compare_tensors(tt_compare, hf_compare, pcc_threshold=get_pcc_threshold(request))
    logger.info(f"Full model PCC (seq_len={seq_len}): {pcc_msg}")

    # Per-token PCC — shows which prompt positions drag down the full-sequence metric.
    from models.common.utility_functions import comp_pcc

    for t in range(seq_len):
        _, pcc_t = comp_pcc(hf_compare[0, t], tt_compare[0, t], pcc=0.0)
        hf_tok = int(hf_compare[0, t].argmax().item())
        tt_tok = int(tt_compare[0, t].argmax().item())
        match = "ok" if hf_tok == tt_tok else "MISMATCH"
        logger.info(
            f"  token[{t}] pcc={pcc_t:.6f} argmax HF={hf_tok} TT={tt_tok} ({match}) "
            f"hf='{tokenizer.decode([hf_tok])}' tt='{tokenizer.decode([tt_tok])}'"
        )
    _, pcc_last_only = comp_pcc(hf_compare[0, -1], tt_compare[0, -1], pcc=0.0)
    logger.info(f"Last-token-only PCC: {pcc_last_only:.6f}")

    # Also check that argmax tokens match for the last position
    hf_last_tok = hf_compare[0, -1, :].argmax().item()
    tt_last_tok = tt_compare[0, -1, :].argmax().item()
    logger.info(
        f"Last-position argmax: HF={hf_last_tok} ('{tokenizer.decode([hf_last_tok])}'), "
        f"TT={tt_last_tok} ('{tokenizer.decode([tt_last_tok])}')"
    )

    assert passing, f"Full model PCC too low: {pcc_msg}"


# ── Full Model DECODE PCC Test ───────────────────────────────────────────


@pytest.mark.gemma4_hf_direct_parity
@parametrize_mesh_with_fabric()
def test_full_model_decode(mesh_device, reset_seeds, request):
    """End-to-end full-model DECODE PCC vs HuggingFace.

    test_full_model only checks the prefill path. This exercises the full decode
    path (on-device embedding, embedding-lookup RoPE, sharded RMSNorm,
    nlp_concat_heads_decode, paged/non-paged SDPA decode) by prefilling a prompt
    and comparing the *next* token's decode-step logits TT vs HF (teacher-forced
    with the same input token so the comparison is apples-to-apples).

        pytest -k "1x4" models/demos/gemma4/tests/unit/test_model.py::test_full_model_decode
    """
    import gc
    import os

    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.gemma4.tt.common import create_tt_model

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    hf_config_check = TestFactory.create_hf_config()
    if getattr(hf_config_check, "enable_moe_block", False) and tp < 8:
        pytest.skip(f"MoE model too large for TP={tp}")
    if hf_config_check.hidden_size > 4096 and tp < 2:
        pytest.skip(f"Model too large for single device (hidden={hf_config_check.hidden_size})")

    # ── HF reference: prefill, then one decode step ──────────────────────
    logger.info(f"Loading HF reference from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    with torch.no_grad():
        hf_out = hf_model(input_ids, use_cache=True)
        next_tok = int(hf_out.logits[0, -1].argmax().item())  # teacher-forced decode input
        hf_dec = hf_model(torch.tensor([[next_tok]]), past_key_values=hf_out.past_key_values, use_cache=True)
        hf_decode_logits = hf_dec.logits[0, -1].float()  # [vocab]
    logger.info(f"HF prefill next token: {next_tok} ('{tokenizer.decode([next_tok])}'), decoding at pos={seq_len}")
    del hf_model
    gc.collect()

    # ── TT: prefill (fills KV), then the same teacher-forced decode step ──
    padded_len = ((seq_len + 31) // 32) * 32
    input_ids_padded = F.pad(input_ids, (0, padded_len - seq_len), value=0) if padded_len > seq_len else input_ids

    model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max(padded_len, 128),
        model_path=model_path,
        create_kv_cache=True,
    )
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    tokens_tt = ttnn.from_torch(
        input_ids_padded.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    embeds = ttnn.to_layout(
        ttnn.reshape(tt_model.embed_tokens(tokens_tt), (1, 1, padded_len, model_args.hidden_size)), ttnn.TILE_LAYOUT
    )
    tt_model.ttnn_prefill_forward(
        embeds,
        page_table=None,
        kv_cache=tt_kv_cache,
        input_ids_torch=input_ids_padded,
        embeds_torch=None,
    ).deallocate(True)

    # One decode step at position seq_len with the teacher-forced token.
    device_inputs = tt_model.prepare_inputs_decode(torch.tensor([next_tok]), torch.tensor([seq_len]), page_table=None)
    logits, _ = tt_model.ttnn_decode_forward(
        x=device_inputs[0],
        current_pos=device_inputs[1],
        rot_mat_idxs=device_inputs[2],
        page_table=device_inputs[3],
        kv_cache=tt_kv_cache,
    )
    if is_mesh and tp > 1:
        shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(logits)]
        tt_decode_logits = shards[0] if shards[0].shape[-1] >= model_args.vocab_size else torch.cat(shards, dim=-1)
    else:
        tt_decode_logits = ttnn.to_torch(logits).float()
    tt_decode_logits = tt_decode_logits.reshape(-1)[: model_args.vocab_size]

    passing, pcc_msg = compare_tensors(tt_decode_logits, hf_decode_logits, pcc_threshold=get_pcc_threshold(request))
    hf_argmax = int(hf_decode_logits.argmax().item())
    tt_argmax = int(tt_decode_logits.argmax().item())
    logger.info(f"Full model DECODE PCC: {pcc_msg}")
    logger.info(
        f"Decode argmax: HF={hf_argmax} ('{tokenizer.decode([hf_argmax])}'), "
        f"TT={tt_argmax} ('{tokenizer.decode([tt_argmax])}')"
    )
    assert passing, f"Full model decode PCC too low: {pcc_msg}"


# ── Single Decode (profiling) ────────────────────────────────────────────


def _build_decode_harness(mesh_device, model_path, decode_pos, max_seq_len=8192, page_block_size=64):
    """Create the full model + KV cache + one-decode input builder.

    Shared by the functional single-decode test and the traced perf test.
    Returns (model_args, model, tt_kv_cache, page_table_tt, make_inputs, decode_fn).
    """
    import torch.nn.functional as F

    from models.demos.gemma4.tt.common import create_tt_model
    from models.tt_transformers.tt.common import PagedAttentionConfig

    paged_attention_config = PagedAttentionConfig(
        block_size=page_block_size,
        max_num_blocks=max_seq_len // page_block_size,
    )
    model_args, model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        1, paged_attention_config.max_num_blocks
    )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    def make_inputs(token_id, pos):
        embeds_torch, pli_torch = model.compute_host_embeddings(token_id)
        pos_padded = F.pad(torch.tensor([pos], dtype=torch.int32).reshape(1, 1), (0, 31), "constant", 0)
        inputs = {
            "embeds": ttnn.from_torch(
                embeds_torch,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            ),
            "position": ttnn.from_torch(
                pos_padded,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            ),
            "position_int32": ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            ),
        }
        if pli_torch is not None:
            inputs["pli"] = ttnn.from_torch(
                pli_torch,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
        return inputs

    def decode_fn(inputs):
        logits, _ = model.ttnn_decode_forward(
            x=inputs["embeds"],
            current_pos=inputs["position"],
            rot_mat_idxs=inputs["position_int32"],
            page_table=page_table_tt,
            kv_cache=tt_kv_cache,
            pli_combined=inputs.get("pli"),
        )
        return logits

    return model_args, model, tt_kv_cache, page_table_tt, make_inputs, decode_fn


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000},
            id="1x4",
        ),
    ],
    indirect=True,
)
def test_single_decode_perf(mesh_device, reset_seeds, request):
    """Traced single decode with tracy signposts, for clean steady-state profiling.

    The functional ``test_single_decode`` runs decode *untraced*, so every op
    pays full host-dispatch latency and the captured profile is dominated by
    host gaps + the warmup/compile pass. This test instead:

      1. compiles the decode (warmup, untraced),
      2. captures a metal trace of one decode forward,
      3. replays it once (warm), then replays it again wrapped in
         ``tracy.signpost("start"/"stop")``.

    Trace replay dispatches the recorded op stream back-to-back on device, so
    op-to-op gaps collapse to device reality; filtering the resulting CSV to
    the signposted region (or to rows with a METAL TRACE REPLAY SESSION ID)
    yields steady-state device time uncluttered by compilation.

        HF_MODEL=google/gemma-4-31B-it \
          TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
          python -m tracy -p -r -v -m pytest \
          "models/demos/gemma4/tests/unit/test_model.py::test_single_decode_perf"
    """
    import os

    try:
        from tracy import signpost
    except ModuleNotFoundError:

        def signpost(*_a, **_k):
            pass

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )
    _skip_if_config_only_model(model_path)

    tp = mesh_device.shape[1]
    decode_pos = 256
    logger.info(f"Creating TT model with all layers (TP={tp}) for traced single-decode profiling...")
    _, _, _, _, make_inputs, decode_fn = _build_decode_harness(mesh_device, model_path, decode_pos)

    inputs = make_inputs(token_id=1, pos=decode_pos)

    # 1. Warmup (compile, untraced)
    logger.info("Decode warmup (compiling kernels)...")
    warm = decode_fn(inputs)
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)

    # 2. Capture trace of one decode forward
    logger.info("Capturing decode trace...")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = decode_fn(inputs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    try:
        # 3. Warm replay (no signpost) then measured replay (signposted)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        logger.info("Profiling single traced decode (signposted)...")
        signpost("start")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    out = ttnn.to_torch(ttnn.get_device_tensors(trace_out)[0]).float()
    assert torch.isfinite(out).all(), "traced decode produced non-finite logits"
    logger.info(f"Traced single decode complete (TP={tp}, pos={decode_pos}); logits shape {tuple(out.shape)}")


@parametrize_mesh_with_fabric()
def test_single_prefill_perf(mesh_device, reset_seeds, request):
    """Traced single prefill with tracy signposts, for clean device-time profiling.

    TTFT at higher ISL is dominated by the prefill *body* (the per-layer matmuls
    + SDPA + CCL over the whole sequence), not host dispatch — tracing prefill
    barely moves TTFT because async dispatch already hides host overhead behind
    that compute. This test exposes WHERE the device time goes so it can be
    optimized:

      1. compiles one prefill forward (untraced, lm_head on the last tile),
      2. captures a metal trace of the forward with the lm_head deferred (the
         trace returns post-norm hidden states, matching the Generator's traced
         prefill path — see Gemma4Model._prefill_trace_mode),
      3. warm-replays, then replays once inside ``tracy.signpost("start"/"stop")``.

    Filtering the resulting CSV to the signposted region (or to METAL TRACE
    REPLAY rows) gives the steady-state device-op breakdown of the prefill body.
    Set the sequence length with ``GEMMA4_PREFILL_PERF_SEQ_LEN`` (default 4096).

        HF_MODEL=google/gemma-4-31B-it \\
          TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
          GEMMA4_PREFILL_PERF_SEQ_LEN=4096 \\
          python -m tracy -p -r -v -m pytest \\
          "models/demos/gemma4/tests/unit/test_model.py::test_single_prefill_perf"
    """
    import os

    try:
        from tracy import signpost
    except ModuleNotFoundError:

        def signpost(*_a, **_k):
            pass

    from models.demos.gemma4.tt.common import create_tt_model
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )
    _skip_if_config_only_model(model_path)

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    hf_config_check = TestFactory.create_hf_config()
    is_moe = getattr(hf_config_check, "enable_moe_block", False)
    if is_moe and tp < 8:
        pytest.skip(f"MoE model too large for TP={tp} (expert weights replicated)")
    if hf_config_check.hidden_size > 4096 and tp < 2:
        pytest.skip(f"Model too large for single device (hidden={hf_config_check.hidden_size})")

    seq_len = int(os.getenv("GEMMA4_PREFILL_PERF_SEQ_LEN", "4096"))
    page_block_size = 64
    max_seq_len = max(8192, 1 << (seq_len - 1).bit_length())
    paged_attention_config = PagedAttentionConfig(
        block_size=page_block_size,
        max_num_blocks=max_seq_len // page_block_size,
    )

    logger.info(f"Creating TT model (TP={tp}) for single-prefill profiling, seq_len={seq_len}...")
    model_args, model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        1, paged_attention_config.max_num_blocks
    )
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )

    tokens = (torch.arange(seq_len, dtype=torch.int32) % model_args.vocab_size).reshape(1, seq_len)
    tokens_tt = ttnn.from_torch(
        tokens, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
    )
    # Host stashes for PLI models (no-op for 31B/12B, which have no per-layer inputs).
    model._prefill_input_ids_torch = tokens.long()
    if model._embed_weight_cpu is not None:
        import torch.nn.functional as F

        model._prefill_embeds_torch = F.embedding(tokens.long(), model._embed_weight_cpu).float() * model.embed_scale
    else:
        model._prefill_embeds_torch = None

    prefill_pli_device_tensors = None
    if model.hidden_size_per_layer_input and model.per_layer_input_weights:
        per_layer_inputs = model._compute_per_layer_inputs(model._prefill_input_ids_torch, model._prefill_embeds_torch)
        prefill_pli_device_tensors = [
            ttnn.from_torch(
                pli.unsqueeze(0).unsqueeze(0),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            for pli in per_layer_inputs
        ]

    def _embed():
        emb = model.embed_tokens(tokens_tt)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)
        return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)

    get_last_token = ((seq_len - 1) // 32) * 32

    # ── Warmup prefill (kernel compile — excluded from the profile) ──
    logger.info("Single-prefill warmup (compiling kernels)...")
    warm = model.ttnn_prefill_forward(
        x=_embed(),
        page_table=page_table_tt,
        kv_cache=tt_kv_cache,
        get_last_token=get_last_token,
        user_id=0,
        pli_device_tensors=prefill_pli_device_tensors,
    )
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)

    # ── Capture trace of the prefill body (lm_head deferred outside the trace) ──
    logger.info("Capturing prefill trace (post-norm hidden states)...")
    model._prefill_trace_mode = True
    x = _embed()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = model.ttnn_prefill_forward(
        x=x,
        page_table=page_table_tt,
        kv_cache=tt_kv_cache,
        user_id=0,
        pli_device_tensors=prefill_pli_device_tensors,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    model._prefill_trace_mode = False
    ttnn.synchronize_device(mesh_device)

    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        logger.info("Profiling single traced prefill (signposted)...")
        signpost("start")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    out = ttnn.to_torch(ttnn.get_device_tensors(trace_out)[0]).float()
    assert torch.isfinite(out).all(), "traced prefill produced non-finite hidden states"
    logger.info(f"Traced single prefill complete (TP={tp}, seq_len={seq_len}); hidden shape {tuple(out.shape)}")


@parametrize_mesh_with_fabric()
def test_single_decode(mesh_device, reset_seeds, request):
    """Run exactly one decode step with the full model, for profiling.

    Isolates a single decode forward so the device-op breakdown can be
    captured with tracy and used to drive optimization. Embedding + per-layer
    inputs are computed on host (mirroring the demo / Generator decode path),
    so the profiled device work is: decoder layers (QKV proj → per-head norms
    → RoPE → KV-cache update → SDPA → out proj → MLP/MoE) + final norm +
    lm_head + softcapping + TP all-gather.

        HF_MODEL=google/gemma-4-31B-it python -m tracy -p -r -v -m \
            pytest models/demos/gemma4/tests/unit/test_model.py::test_single_decode

    A warmup decode compiles the kernels; the profiled decode then hits the
    cached programs, so the captured ops reflect steady-state decode rather
    than first-run compilation. No prefill is run — the KV cache is
    zero-initialized and the decode token's K/V is written at ``decode_pos``
    before SDPA attends over [0, decode_pos]. Output correctness is not
    checked here (see ``test_full_model`` for PCC); this test only exercises
    the decode kernels.

        pytest -k "1x4"   # blackhole 1x4 (TP=4) — the profiling target
    """
    import os

    import torch.nn.functional as F

    from models.demos.gemma4.tt.common import create_tt_model
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )
    _skip_if_config_only_model(model_path)

    # Skip combos where the model doesn't fit (same gating as test_full_model).
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    hf_config_check = TestFactory.create_hf_config()
    is_moe = getattr(hf_config_check, "enable_moe_block", False)
    if is_moe and tp < 8:
        pytest.skip(f"MoE model too large for TP={tp} (expert weights replicated)")
    if hf_config_check.hidden_size > 4096 and tp < 2:
        pytest.skip(f"Model too large for single device (hidden={hf_config_check.hidden_size})")

    # Position the single decode token attends from. The KV cache covers
    # [0, decode_pos]; a non-trivial position gives SDPA a realistic amount of
    # context to scan during the profile.
    decode_pos = 256
    max_seq_len = 8192
    page_block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=page_block_size,
        max_num_blocks=max_seq_len // page_block_size,
    )

    logger.info(f"Creating TT model with all layers (TP={tp}) for single-decode profiling...")
    model_args, model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # Identity page table for a single user.
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        1, paged_attention_config.max_num_blocks
    )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    def _make_decode_inputs(token_id, pos):
        """Host-side embedding + PLI + position tensors for one decode token."""
        embeds_torch, pli_torch = model.compute_host_embeddings(token_id)
        pos_padded = F.pad(torch.tensor([pos], dtype=torch.int32).reshape(1, 1), (0, 31), "constant", 0)
        inputs = {
            "embeds": ttnn.from_torch(
                embeds_torch,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            ),
            "position": ttnn.from_torch(
                pos_padded,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            ),
            "position_int32": ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            ),
        }
        if pli_torch is not None:
            inputs["pli"] = ttnn.from_torch(
                pli_torch,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
        return inputs

    def _decode(inputs):
        # Profile the model forward (incl. lm_head + all-gather), not the sampling generator.
        logits, _ = model.ttnn_decode_forward(
            x=inputs["embeds"],
            current_pos=inputs["position"],
            rot_mat_idxs=inputs["position_int32"],
            page_table=page_table_tt,
            kv_cache=tt_kv_cache,
            pli_combined=inputs.get("pli"),
        )
        return logits

    token_id = 1  # arbitrary non-pad token

    # ── Warmup decode (kernel compile — excluded from the meaningful profile) ──
    logger.info("Single-decode warmup (compiling kernels)...")
    warmup_logits = _decode(_make_decode_inputs(token_id, decode_pos))
    ttnn.synchronize_device(mesh_device)
    warmup_logits.deallocate(True)

    # ── Profiled single decode (cached programs — this is the profile sample) ──
    logger.info(f"Profiling single decode step at pos={decode_pos + 1}...")
    logits = _decode(_make_decode_inputs(token_id, decode_pos + 1))

    # On TP meshes where on-device sampling is initialized, the model leaves
    # decode logits TP-sharded (one vocab slice per device) for the sampling
    # module instead of all-gathering; gather the shards here so the smoke
    # check sees the full vocab. On a single device (or when the model already
    # all-gathered) tensor 0 holds the full row.
    if is_mesh and tp > 1:
        shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(logits)]
        out = shards[0] if shards[0].shape[-1] >= model_args.vocab_size else torch.cat(shards, dim=-1)
    else:
        out = ttnn.to_torch(logits).float()
    logits.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    out = out.reshape(-1)[: model_args.vocab_size]
    assert out.numel() == model_args.vocab_size, f"unexpected logits width {out.shape}"
    assert torch.isfinite(out).all(), "decode produced non-finite logits"
    next_token = int(out.argmax().item())
    logger.info(f"Single decode complete (TP={tp}, pos={decode_pos + 1}); next token id = {next_token}")
