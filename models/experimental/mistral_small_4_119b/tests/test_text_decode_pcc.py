# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full-model decode PCC test for Mistral-Small-4.

Runs the full 36-layer model and validates the KV-cache decode step at position
P with two independent PCC checks:

  1. Decode-vs-prefill (self-consistency): the decode logits at position P must
     match the TTNN prefill logits at position P.  Floor 0.70, typically ~0.99.
  2. Decode-vs-HF (ground truth): the decode logits must match the HF Torch
     reference's last-position logits.  Floor 0.90 (measured 0.967).

Test design:
  1. Build HF reference, run a forward pass, capture last-position logits, free it.
  2. Prefill the same prompt through the TTNN model → fills all 36 KV caches and
     returns reference logits at the last position.
  3. Decode at that same last position using the cached K/V → returns logits.
  4. Compare decode logits against both references via PCC.

Why two checks instead of just decode-vs-HF:
  In bfloat16, TTNN matmul kernels are shape-specific: a seq_len-token prefill
  kernel produces slightly different K/V at position P than a 1-token decode
  kernel.  HF always uses a full-context kernel.  So decode-vs-HF carries PCC
  loss from this kernel-shape effect on top of the quantization loss already
  present in prefill-vs-HF.  At position 4 of the 5-token prompt this measures
  0.967; the floor is 0.90.  The self-consistency check isolates the decode path
  itself (does it correctly read/use the cache prefill wrote?), free of the
  kernel-shape effect; the HF check anchors the absolute correctness.

Run manually::

    export MESH_DEVICE=P150x8
    pytest models/experimental/mistral_small_4_119b/tests/test_text_decode_pcc.py -v -s --timeout=0
"""

from __future__ import annotations

import gc
import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tests.test_text_prefill_pcc import _build_hf_ref
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

_N_LAYERS = int(os.environ.get("MISTRAL4_DECODE_N_LAYERS", "36"))
_PCC_FLOOR = 0.70  # decode-vs-prefill self-consistency floor
_HF_PCC_FLOOR = 0.90  # decode-vs-HF floor (lower than prefill: includes the bfloat16 kernel-shape effect)


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
    # at a time. HF's last-position logits are the ground-truth next-token
    # distribution for position seq_len-1 — the same prediction the decode step makes.
    logger.info(f"Building HF reference ({_N_LAYERS} layers, CPU, bfloat16)...")
    hf_model = _build_hf_ref(text, state_dict, _N_LAYERS)
    logger.info("Running HF reference forward pass...")
    hf_logits = hf_model(input_ids).logits[0, seq_len - 1, :].float()  # [vocab]
    del hf_model
    gc.collect()
    logger.info(f"HF reference logits shape: {tuple(hf_logits.shape)}")

    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers)...")
    model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text,
        num_decoder_layers=_N_LAYERS,
        max_seq_len=seq_len + 64,
    )

    embed_w = state_dict["language_model.model.embed_tokens.weight"].to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    hidden0 = F.embedding(input_ids, embed_w)
    cos_full, sin_full = rotary(hidden0, position_ids)
    model.cache_rope_tables(cos_full, sin_full)

    # Prefill: fills all KV caches + returns logits at every position.
    logger.info(f"Running prefill (seq_len={seq_len})...")
    prefill_logits = model.prefill(input_ids)  # [1, seq_len, vocab]
    ref_logits = prefill_logits[0, seq_len - 1, :].float()  # [vocab]
    logger.info(f"Reference logits shape: {tuple(ref_logits.shape)}")

    # Decode: re-run the last token through the full decode path using the KV
    # cache just filled by prefill, and collect the logit distribution.
    last_input_id = input_ids[:, seq_len - 1 : seq_len]  # [1, 1]
    logger.info(f"Running decode step at position {seq_len - 1}...")
    dec_logits = model.decode_logits(last_input_id, seq_len - 1)  # [vocab]

    ref_tok = int(ref_logits.argmax())
    dec_tok = int(dec_logits.argmax())
    hf_tok = int(hf_logits.argmax())
    logger.info(f"Greedy token: HF={hf_tok}, prefill={ref_tok}, decode={dec_tok}")

    # ── Check 1: decode-vs-prefill self-consistency (isolates the decode path) ──
    sc_passing, sc_msg = comp_pcc(ref_logits, dec_logits, _PCC_FLOOR)
    logger.info(f"Decode-vs-prefill (self-consistency) PCC (pos={seq_len - 1}): {sc_msg}")

    # ── Check 2: decode-vs-HF ground truth ─────────────────────────────────────
    hf_passing, hf_msg = comp_pcc(hf_logits, dec_logits, _HF_PCC_FLOOR)
    logger.info(f"Decode-vs-HF PCC (pos={seq_len - 1}): {hf_msg}")

    assert sc_passing, (
        f"Decode-vs-prefill PCC below floor {_PCC_FLOOR}.\n"
        f"The full 36-layer decode path is not self-consistent with prefill "
        f"(wrong cache slot, masked attention, NaN, etc.).\n{sc_msg}"
    )
    assert hf_passing, (
        f"Decode-vs-HF PCC below floor {_HF_PCC_FLOOR}.\n"
        f"The full 36-layer decode path does not match the HF reference beyond the "
        f"expected bfloat16 quantization + kernel-shape loss.\n{hf_msg}"
    )
    logger.info(f"PASSED — self-consistency PCC >= {_PCC_FLOOR}, HF PCC >= {_HF_PCC_FLOOR}")
