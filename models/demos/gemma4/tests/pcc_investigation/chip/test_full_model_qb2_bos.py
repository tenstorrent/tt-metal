# Scratch copy of models/demos/gemma4/tests/unit/test_model.py::{test_full_model,test_full_model_decode},
# BOS VARIANT: also prepends the start-of-text token. Otherwise verbatim except the MoE "tp < 8" skip, so the tests run 26B-A4B on QuietBox 2 (1x4).
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
)


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
    # [scratch copy] MoE tp<8 skip removed: 26B-A4B fits and runs on the 1x4 QB2.
    if hf_config_check.hidden_size > 4096 and tp < 2:
        pytest.skip(f"Model too large for single device (hidden={hf_config_check.hidden_size})")

    # ── HF reference ─────────────────────────────────────────────────
    logger.info(f"Loading HF reference model from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # [1, seq_len]
    if tokenizer.bos_token_id is not None and int(input_ids[0, 0]) != tokenizer.bos_token_id:
        input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), input_ids], dim=1)  # [bos variant] start token
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
    # [scratch copy] MoE tp<8 skip removed: 26B-A4B fits and runs on the 1x4 QB2.
    if hf_config_check.hidden_size > 4096 and tp < 2:
        pytest.skip(f"Model too large for single device (hidden={hf_config_check.hidden_size})")

    # ── HF reference: prefill, then one decode step ──────────────────────
    logger.info(f"Loading HF reference from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if tokenizer.bos_token_id is not None and int(input_ids[0, 0]) != tokenizer.bos_token_id:
        input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), input_ids], dim=1)  # [bos variant] start token
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


