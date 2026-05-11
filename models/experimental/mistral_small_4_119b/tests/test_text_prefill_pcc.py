# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill logits PCC test for Mistral-Small-4 — TTNN vs HF reference.

Runs a real tokenized prompt through:
  - HF Mistral4ForCausalLM (CPU, float32, N layers)
  - TtMistral4TextModel     (device, bfloat16, same N layers)

and compares the prefill logits at every position.

PCC note:
  MoE top-k routing in float32 (HF) vs bfloat16 (TTNN) may select
  different experts when scores are close.  A routing disagreement at
  any layer propagates to downstream layers, so full-stack PCC is lower
  than per-layer attention PCC.  A healthy run typically gives 0.90+;
  values below 0.80 suggest a real bug (wrong weights, wrong norm,
  wrong attention, etc.).

Run manually::

    export MISTRAL4_PREFILL_PCC=1
    export MISTRAL4_PREFILL_N_LAYERS=2    # optional; default 2
    export MESH_DEVICE=P150x4             # optional
    pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_pcc.py -v -s --timeout=0
"""

from __future__ import annotations

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
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")

_N_LAYERS = int(os.environ.get("MISTRAL4_PREFILL_N_LAYERS", "2"))
_PCC_FLOOR = 0.90


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


def _dequantize_state_dict(state_dict: dict) -> dict:
    """
    Dequantize FP8 weights in place.

    Checkpoint uses static FP8 quantization:
      w_fp8 = round(w_fp32 / scale_factor)
      weight_scale_inv = 1 / scale_factor
    Dequantize: w_fp32 ≈ w_fp8 * (1 / weight_scale_inv) ... but since
    weight_scale_inv IS 1/scale_factor, dequant = w_fp8 * weight_scale_inv.
    (Verified numerically: w * scale_inv gives ~0.12 max, correct range.)
    """
    dq = {}
    for k, v in state_dict.items():
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Keys without .weight suffix (e.g. experts.gate_up_proj): append _scale_inv
            # Keys with .weight suffix (e.g. shared_experts.gate_proj.weight): replace .weight
            scale_key = k + "_scale_inv"
            if scale_key not in state_dict:
                scale_key = k.replace(".weight", ".weight_scale_inv")
            scale_inv = state_dict.get(scale_key)
            v_fp32 = v.to(torch.float32)
            if scale_inv is not None:
                s = scale_inv.to(torch.float32)
                while s.dim() < v_fp32.dim():
                    s = s.unsqueeze(-1)
                v_fp32 = v_fp32 * s
            dq[k] = v_fp32
        elif (
            not k.endswith("_scale_inv") and not k.endswith("_activation_scale") and not k.endswith("activation_scale")
        ):
            dq[k] = v.to(torch.float32)
    return dq


def _build_hf_ref(text_config, state_dict, n_layers: int) -> torch.nn.Module:
    """Instantiate N-layer HF model on CPU (float32) with properly dequantized weights."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    import copy

    cfg = copy.deepcopy(text_config)
    cfg.num_hidden_layers = n_layers

    model = Mistral4ForCausalLM(cfg).to(torch.float32).eval()

    # Dequantize FP8 weights, strip "language_model." prefix to match HF model param names.
    dq_sd = _dequantize_state_dict(state_dict)
    hf_sd = {k[len("language_model.") :]: v for k, v in dq_sd.items() if k.startswith("language_model.")}

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    if missing:
        logger.warning(f"HF model missing keys (first 5): {missing[:5]}")
    return model


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_PREFILL_PCC") != "1",
    reason="Set MISTRAL4_PREFILL_PCC=1 to run the prefill logits PCC test.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_prefill_pcc(reset_seeds, mesh_device):
    """Compare TTNN prefill logits to HF float32 reference for the same prompt."""
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

    # ── Tokenize a real prompt ────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"Tokenizer load failed: {exc}")

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # [1, seq_len]
    seq_len = input_ids.shape[1]
    logger.info(f"Prompt: {prompt!r}  →  {seq_len} tokens: {input_ids.tolist()}")

    # ── HF reference (CPU, float32) ───────────────────────────────────────
    logger.info(f"Building HF reference ({_N_LAYERS} layers, CPU, float32)...")
    hf_model = _build_hf_ref(text, state_dict, _N_LAYERS)
    ref_out = hf_model(input_ids)
    ref_logits = ref_out.logits[0].to(torch.float32)  # [seq_len, vocab]
    del hf_model
    logger.info(f"HF reference logits shape: {tuple(ref_logits.shape)}")

    # ── TTNN model ────────────────────────────────────────────────────────
    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers)...")
    model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text,
        num_decoder_layers=_N_LAYERS,
        max_seq_len=seq_len + 64,
    )

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    hidden0 = torch.nn.functional.embedding(
        input_ids,
        state_dict["language_model.model.embed_tokens.weight"].to(torch.bfloat16),
    )
    pos_emb = rotary(hidden0, position_ids)

    logger.info(f"Running TTNN prefill (seq_len={seq_len})...")
    ttnn_logits_host = model.prefill(input_ids, pos_emb)  # [1, seq_len, vocab]
    ttnn_logits = ttnn_logits_host[0].to(torch.float32)  # [seq_len, vocab]

    # ── PCC check at each position ─────────────────────────────────────────
    pccs = []
    for pos in range(seq_len):
        _, pcc_val = comp_pcc(ref_logits[pos], ttnn_logits[pos], _PCC_FLOOR)
        # comp_pcc returns (passing, message); extract the float
        pcc_float = float(pcc_val.split("=")[-1].strip() if "=" in str(pcc_val) else pcc_val)
        pccs.append(pcc_float)

    mean_pcc = sum(pccs) / len(pccs)
    min_pcc = min(pccs)
    logger.info(f"Logits PCC across {seq_len} positions: mean={mean_pcc:.4f}, min={min_pcc:.4f}")
    logger.info(f"Per-position PCCs: {[f'{p:.3f}' for p in pccs]}")

    # Also compare greedy predictions
    ref_tokens = ref_logits.argmax(dim=-1).tolist()
    ttnn_tokens = ttnn_logits.argmax(dim=-1).tolist()
    match = sum(r == t for r, t in zip(ref_tokens, ttnn_tokens))
    logger.info(f"Greedy token match: {match}/{seq_len}")
    logger.info(f"  HF   tokens: {ref_tokens}")
    logger.info(f"  TTNN tokens: {ttnn_tokens}")

    passing, pcc_msg = comp_pcc(ref_logits.flatten(), ttnn_logits.flatten(), _PCC_FLOOR)
    logger.info(f"Overall flattened logits PCC: {pcc_msg}")
    assert passing, (
        f"Prefill logits PCC below floor {_PCC_FLOOR}.\n"
        f"mean per-position PCC={mean_pcc:.4f}, greedy match={match}/{seq_len}\n"
        f"{pcc_msg}"
    )
    logger.info(f"PASSED — TTNN prefill logits PCC >= {_PCC_FLOOR}")
