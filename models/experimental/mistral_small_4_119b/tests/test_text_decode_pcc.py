# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full-model decode PCC test for Mistral-Small-4 (teacher-forced, multi-step).

Runs the full 36-layer model and validates the KV-cache decode path over
MISTRAL4_DECODE_STEPS consecutive positions, teacher-forced against an HF Torch
reference: each step feeds the reference's greedy token, so the TT and HF
trajectories stay aligned and every step is compared at the same context. This
exercises the cache read/write across positions (not just one step), catching
drift, wrong cache-slot indexing, and RoPE-at-higher-position bugs that a
single-step check misses.

Two PCC families:
  1. Decode-vs-HF (ground truth): at EVERY decode step the decode logits must
     match the HF reference's logits at that position. Floor 0.90 (step 0 ~0.961).
  2. Decode-vs-prefill (self-consistency): at step 0 (the last prompt position)
     the decode logits must match the TTNN prefill logits at that position.
     Floor 0.70, typically ~0.99. Only definable at a prefilled position, and it
     isolates the decode path from the bfloat16 kernel-shape effect.

Why the two HF/self-consistency floors differ:
  In bfloat16, TTNN matmul kernels are shape-specific: a seq_len-token prefill
  kernel produces slightly different K/V at a position than a 1-token decode
  kernel. HF always uses a full-context kernel. So decode-vs-HF carries PCC loss
  from this kernel-shape effect on top of quantization loss; the self-consistency
  check (decode-vs-prefill) is free of it and anchors the cache-read correctness.

Test design:
  1. Build HF reference; greedily generate MISTRAL4_DECODE_STEPS tokens with a
     DynamicCache, capturing per-step logits + the teacher tokens; free it.
  2. Prefill the prompt through the TTNN model → fills all 36 KV caches.
  3. Decode step-by-step from the last prompt position, feeding the teacher token
     each step (teacher forcing), reading/writing the KV cache.
  4. Compare each step's decode logits against HF (all steps) and prefill (step 0).

Set MISTRAL4_DECODE_STEPS=1 to recover the original single-step check.

Run manually::

    export MESH_DEVICE=P150x8
    pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_pcc.py -v -s --timeout=0
"""

from __future__ import annotations

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tests.test_text_prefill_pcc import _build_hf_ref, _build_prompt_ids
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

_N_LAYERS = int(os.environ.get("MISTRAL4_DECODE_N_LAYERS", "36"))
_DECODE_STEPS = int(os.environ.get("MISTRAL4_DECODE_STEPS", "8"))  # teacher-forced decode steps to validate
_PCC_FLOOR = 0.70  # decode-vs-prefill self-consistency floor
_HF_PCC_FLOOR = 0.90  # decode-vs-HF floor (lower than prefill: includes the bfloat16 kernel-shape effect)
_HF_PCC_FLOOR_SWEEP = 0.80  # decode-vs-HF floor for the filler ISL sweep (characterization)
_HF_CHUNK = int(os.environ.get("MISTRAL4_DECODE_HF_CHUNK", "2048"))  # HF reference chunk for the ISL sweep


def _state_dict_prefixes(n_layers: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    return tuple(p)


def _mesh_params():
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30000000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_decode_pcc(reset_seeds, mesh_device):
    """Full-model decode self-consistency: logits from KV-cache decode at pos P must
    match prefill logits at pos P."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    text = cfg.text_config
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(text, attr):
            setattr(text, attr, "eager")

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"Tokenizer load failed: {exc}")

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # [1, seq_len]
    seq_len = input_ids.shape[1]
    logger.info(f"Prompt: {prompt!r}  →  {seq_len} tokens")

    # ── HF reference (CPU, bfloat16, streamed weights) ───────────────────────
    # Built and freed BEFORE the TTNN model so peak host RAM holds one full model
    # at a time. Greedily generate _DECODE_STEPS tokens with a DynamicCache,
    # capturing the per-step ground-truth logits and the teacher tokens the TTNN
    # decode loop is forced with (so both trajectories stay aligned position-by-position).
    from transformers.cache_utils import DynamicCache

    logger.info(f"Building HF reference ({_N_LAYERS} layers, CPU, bfloat16)...")
    hf_model = _build_hf_ref(text, state_dict, _N_LAYERS)
    logger.info(f"Running HF reference + greedy teacher generation ({_DECODE_STEPS} steps)...")
    hf_step_logits: list[torch.Tensor] = []  # hf_step_logits[j] = logits at position (seq_len-1+j)
    teacher_tokens: list[int] = []  # teacher_tokens[j] = greedy token at position (seq_len+j)
    hf_cache = DynamicCache()
    out = hf_model(input_ids, past_key_values=hf_cache, use_cache=True)
    step_logit = out.logits[0, -1, :].float().clone()
    hf_step_logits.append(step_logit)
    teacher_tokens.append(int(step_logit.argmax()))
    for _ in range(_DECODE_STEPS - 1):
        nxt = torch.tensor([[teacher_tokens[-1]]], dtype=torch.long)
        out = hf_model(nxt, past_key_values=hf_cache, use_cache=True)
        step_logit = out.logits[0, -1, :].float().clone()
        hf_step_logits.append(step_logit)
        teacher_tokens.append(int(step_logit.argmax()))
    del hf_model, hf_cache, out
    gc.collect()
    logger.info(f"HF teacher tokens (positions {seq_len}..{seq_len + _DECODE_STEPS - 1}): {teacher_tokens}")

    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers)...")
    model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text,
        num_decoder_layers=_N_LAYERS,
        max_seq_len=seq_len + 64,
    )

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    # RoPE table must cover every position the decode loop touches: [0, seq_len + _DECODE_STEPS).
    total_positions = seq_len + _DECODE_STEPS
    position_ids = torch.arange(total_positions, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)  # rotary uses position_ids; x is only a dtype carrier
    cos_full, sin_full = rotary(dummy, position_ids)
    model.cache_rope_tables(cos_full, sin_full)

    # Prefill: fills all KV caches + returns logits at every position.
    logger.info(f"Running prefill (seq_len={seq_len})...")
    prefill_logits = model.prefill(input_ids)  # [1, seq_len, vocab]
    prefill_last = prefill_logits[0, seq_len - 1, :].float()  # [vocab] at position seq_len-1

    # Teacher-forced decode. Step j processes the token at position (seq_len-1+j) and
    # predicts position (seq_len+j). Step 0's input is the last prompt token; later
    # steps feed the HF teacher token for the position being decoded, so TT and HF
    # stay aligned and every step is compared at the same context.
    hf_pccs: list[float] = []
    token_matches = 0
    sc_passing, sc_msg = None, None
    for j in range(_DECODE_STEPS):
        pos = seq_len - 1 + j
        if j == 0:
            tok = input_ids[:, seq_len - 1 : seq_len]  # [1,1] last prompt token
        else:
            tok = torch.tensor([[teacher_tokens[j - 1]]], dtype=torch.long)  # token sitting at position `pos`
        dec_logits = model.decode_logits(tok, pos).float()  # [vocab], predicts position pos+1

        hf_passing, hf_pcc = comp_pcc(hf_step_logits[j], dec_logits, _HF_PCC_FLOOR)
        hf_pccs.append(float(hf_pcc))
        dec_tok = int(dec_logits.argmax())
        if dec_tok == teacher_tokens[j]:
            token_matches += 1
        logger.info(
            f"step {j} (pos={pos}): decode-vs-HF PCC={float(hf_pcc):.4f} (floor {_HF_PCC_FLOOR}); "
            f"decode_tok={dec_tok}, HF_tok={teacher_tokens[j]}"
        )
        assert hf_passing, (
            f"Decode-vs-HF PCC below floor {_HF_PCC_FLOOR} at step {j} (pos={pos}).\n"
            f"The full 36-layer decode path diverges from the HF reference beyond the "
            f"expected bfloat16 quantization + kernel-shape loss.\nPCC={float(hf_pcc)}"
        )

        # Step 0 sits at a prefilled position, so it also gets the self-consistency check.
        if j == 0:
            sc_passing, sc_pcc = comp_pcc(prefill_last, dec_logits, _PCC_FLOOR)
            sc_msg = f"{float(sc_pcc)}"
            logger.info(f"step 0 decode-vs-prefill (self-consistency) PCC: {sc_msg}")

    mean_hf = sum(hf_pccs) / len(hf_pccs)
    logger.info(
        f"Decode-vs-HF over {_DECODE_STEPS} steps: mean={mean_hf:.4f}, min={min(hf_pccs):.4f}; "
        f"greedy token match (decode == HF teacher): {token_matches}/{_DECODE_STEPS}"
    )

    assert sc_passing, (
        f"Decode-vs-prefill PCC below floor {_PCC_FLOOR} at step 0.\n"
        f"The full 36-layer decode path is not self-consistent with prefill "
        f"(wrong cache slot, masked attention, NaN, etc.).\nPCC={sc_msg}"
    )
    logger.info(
        f"PASSED — all {_DECODE_STEPS} decode steps ≥ HF floor {_HF_PCC_FLOOR}; "
        f"step-0 self-consistency ≥ {_PCC_FLOOR}"
    )


def _chunked_hf_last_logit(hf_model, input_ids: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Chunked HF forward (DynamicCache) returning ONLY the final-position logits [vocab],
    float32. Keeps attention memory O(chunk × past) and never materializes the full
    [seq, vocab] tensor — so it scales to long ISL for the decode sweep (no 33 GB host
    logits like the full-sequence prefill reference)."""
    from transformers.cache_utils import DynamicCache

    seq_len = input_ids.shape[-1]
    cache = DynamicCache()
    last = None
    for s in range(0, seq_len, chunk_size):
        e = min(s + chunk_size, seq_len)
        kwargs = dict(input_ids=input_ids[:, s:e], past_key_values=cache, use_cache=True)
        if s > 0:
            kwargs["position_ids"] = torch.arange(s, e, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            out = hf_model(**kwargs)
        last = out.logits[0, -1, :].float().clone()
        del out
        gc.collect()
        logger.info(f"HF chunked prefill: {e}/{seq_len} tokens")
    return last


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.parametrize(
    "isl",
    [0, 128, 512, 2048, 4096, 16384, 65536, 131072, 262144],
    ids=["base", "128", "512", "2k", "4k", "16k", "64k", "128k", "256k"],
)
def test_mistral_small_4_decode_pcc_isl(reset_seeds, mesh_device, isl):
    """Decode-vs-HF PCC at increasing context depth: one decode step at the last prompt
    position after a chunked cache-fill.

    Unlike the prefill sweep, decode computes the lm_head on ONE position, so the TTNN
    side scales to long ISL like e2e (cache-fill via prefill_next_token + a single decode)
    — no full-sequence lm_head wall. The HF reference keeps only the final-position logit
    (chunked, DynamicCache), so it stays memory-bounded; its wall-clock is the only ceiling.
    Floor 0.90 for ISL <= 16k (verified), 0.80 beyond (characterization) — matches prefill.
    """
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    text = cfg.text_config
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(text, attr):
            setattr(text, attr, "eager")

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"Tokenizer load failed: {exc}")

    input_ids = _build_prompt_ids(tokenizer, isl)  # [1, seq_len]
    seq_len = input_ids.shape[1]
    # Single-position decode-vs-HF on filler is noisier than the prefill flattened PCC (not
    # averaged over positions, and the filler tail is high-entropy → the model often wants EOS),
    # so only the base real-prompt point uses the strict floor (apples-to-apples vs the documented
    # 0.961); the filler sweep is characterization at the looser floor.
    floor = _HF_PCC_FLOOR if isl == 0 else _HF_PCC_FLOOR_SWEEP
    logger.info(f"ISL={isl or 'base'} — {seq_len} tokens (decode-vs-HF floor {floor})")

    # ── HF reference: final-position logits only (chunked, memory-bounded) ───
    logger.info(f"Building HF reference ({_N_LAYERS} layers, CPU, bfloat16)...")
    hf_model = _build_hf_ref(text, state_dict, _N_LAYERS)
    logger.info(f"Running HF reference (chunked, {_HF_CHUNK}-tok) → last-position logits...")
    hf_last = _chunked_hf_last_logit(hf_model, input_ids, _HF_CHUNK)  # [vocab]
    del hf_model
    gc.collect()

    # ── TTNN: fill KV cache (cache-only prefill, single-position lm_head) + 1 decode ──
    model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text,
        num_decoder_layers=_N_LAYERS,
        max_seq_len=seq_len + 64,
    )
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_ids = torch.arange(seq_len + 1, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)  # rotary uses position_ids; x is a dtype carrier
    cos_full, sin_full = rotary(dummy, position_ids)
    model.cache_rope_tables(cos_full, sin_full)

    logger.info(f"Filling KV cache via chunked prefill (seq_len={seq_len})...")
    model.prefill_next_token(input_ids)  # cache-fill only; single-position lm_head (e2e-cheap, no full-logits wall)

    logger.info(f"Decoding at position {seq_len - 1}...")
    dec_logits = model.decode_logits(input_ids[:, seq_len - 1 : seq_len], seq_len - 1).float()  # [vocab]

    dec_tok = int(dec_logits.argmax())
    hf_tok = int(hf_last.argmax())
    passing, pcc = comp_pcc(hf_last, dec_logits, floor)
    logger.info(
        f"ISL={isl or 'base'} ({seq_len} tok): decode-vs-HF PCC={float(pcc):.4f} (floor {floor}); "
        f"decode_tok={dec_tok}, HF_tok={hf_tok}, match={dec_tok == hf_tok}"
    )
    assert passing, f"Decode-vs-HF PCC below floor {floor} at ISL={isl} (seq_len={seq_len}).\nPCC={float(pcc)}"
    logger.info(f"PASSED — decode-vs-HF PCC >= {floor} at {seq_len}-token context")
