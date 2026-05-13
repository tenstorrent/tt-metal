# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill logits PCC test for Mistral-Small-4 — TTNN vs HF reference.

Runs a real tokenized prompt through:
  - HF Mistral4ForCausalLM (CPU, bfloat16, N layers)
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
    export MESH_DEVICE=T3K                 # optional; T3K=1x8, P150x4=1x4, single=1x1
    pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_pcc.py -v -s --timeout=0
"""

from __future__ import annotations

import gc
import os

import psutil
import pytest
import torch
from loguru import logger


def _log_mem(tag: str) -> None:
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1e9
    sys = psutil.virtual_memory()
    used_gb = (sys.total - sys.available) / 1e9
    avail_gb = sys.available / 1e9
    logger.info(f"[MEM {tag}] process RSS={rss_gb:.1f} GB  sys used={used_gb:.1f} GB  sys avail={avail_gb:.1f} GB")


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


def _dequantize_state_dict(state_dict: dict, dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Dequantize FP8 weights to ``dtype`` (default bfloat16 to halve peak RSS vs float32).

    Checkpoint uses static FP8 quantization:
      w_fp8 = round(w_fp32 / scale_factor)
      weight_scale_inv = 1 / scale_factor
    Dequantize: w ≈ w_fp8 * weight_scale_inv  (verified numerically).
    """
    dq = {}
    for k, v in state_dict.items():
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale_key = k + "_scale_inv"
            if scale_key not in state_dict:
                scale_key = k.replace(".weight", ".weight_scale_inv")
            scale_inv = state_dict.get(scale_key)
            v_cast = v.to(torch.float32)
            if scale_inv is not None:
                s = scale_inv.to(torch.float32)
                while s.dim() < v_cast.dim():
                    s = s.unsqueeze(-1)
                v_cast = v_cast * s
            dq[k] = v_cast.to(dtype)
        elif (
            not k.endswith("_scale_inv") and not k.endswith("_activation_scale") and not k.endswith("activation_scale")
        ):
            dq[k] = v.to(dtype)
    return dq


def _build_hf_ref(text_config, state_dict, n_layers: int) -> torch.nn.Module:
    """Instantiate N-layer HF model on CPU (bfloat16) using streaming weight assignment.

    Uses accelerate.init_empty_weights so the model shell costs ~0 RAM, then assigns
    each parameter individually (FP8 → bfloat16) without ever holding two full copies
    of the weights in memory simultaneously.  Peak overhead above the FP8 state dict
    is only the size of one parameter tensor at a time (~MB), compared to +238 GB if
    load_state_dict were called on a fully-allocated bfloat16 model.
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

    import copy

    cfg = copy.deepcopy(text_config)
    cfg.num_hidden_layers = n_layers

    # Create model shell on meta device — zero RAM allocated for parameters.
    _log_mem("before HF model shell (meta device)")
    with init_empty_weights():
        model = Mistral4ForCausalLM(cfg)
    _log_mem("after HF model shell (meta device, ~0 GB extra)")

    # Stream weights in one parameter at a time: FP8 → bfloat16 → CPU.
    logger.info("Streaming FP8 → bfloat16 weights into HF model...")
    missing_keys = []
    for param_name, _ in model.named_parameters():
        sd_key = "language_model." + param_name
        if sd_key not in state_dict:
            missing_keys.append(param_name)
            continue
        v = state_dict[sd_key]
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale_key = sd_key + "_scale_inv"
            if scale_key not in state_dict:
                scale_key = sd_key.replace(".weight", ".weight_scale_inv")
            scale_inv = state_dict.get(scale_key)
            v_cast = v.to(torch.float32)
            if scale_inv is not None:
                s = scale_inv.to(torch.float32)
                while s.dim() < v_cast.dim():
                    s = s.unsqueeze(-1)
                v_cast = v_cast * s
            tensor = v_cast.to(torch.bfloat16)
            del v_cast
        else:
            tensor = v.to(torch.bfloat16)
        set_module_tensor_to_device(model, param_name, "cpu", value=tensor)
        del tensor

    if missing_keys:
        logger.warning(f"HF model missing keys (first 5): {missing_keys[:5]}")
    _log_mem("after streaming all weights into HF model")

    try:
        from optimum.quanto import quantize, qint4

        logger.info("Applying quanto int4 quantization to Linear weights (CPU)...")
        quantize(model, weights=qint4)
        _log_mem("after quanto int4 quantization (~60 GB vs 238 GB bfloat16)")
    except ImportError:
        logger.warning(
            "optimum-quanto not installed — keeping bfloat16 weights (~238 GB). "
            "Run `pip install optimum-quanto` to reduce to ~60 GB."
        )

    return model.eval()


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

    _log_mem("start — before state dict load")
    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_N_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")
    _log_mem("after state dict load")

    # ── Tokenize a real prompt ────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    except Exception as exc:
        pytest.skip(f"Tokenizer load failed: {exc}")

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # [1, seq_len]
    seq_len = input_ids.shape[1]
    logger.info(f"Prompt: {prompt!r}  →  {seq_len} tokens: {input_ids.tolist()}")

    # ── HF reference (CPU, bfloat16, streamed weights) ───────────────────
    logger.info(f"Building HF reference ({_N_LAYERS} layers, CPU, bfloat16)...")
    _log_mem("before _build_hf_ref")
    hf_model = _build_hf_ref(text, state_dict, _N_LAYERS)
    _log_mem("after _build_hf_ref — before forward pass")
    logger.info("Running HF reference forward pass...")
    ref_out = hf_model(input_ids)
    _log_mem("after HF forward pass")
    ref_logits = ref_out.logits[0].float()  # [seq_len, vocab] — upcast for PCC arithmetic only
    del hf_model, ref_out
    gc.collect()
    _log_mem("after del hf_model + gc")
    logger.info(f"HF reference logits shape: {tuple(ref_logits.shape)}")

    # ── TTNN model ────────────────────────────────────────────────────────
    logger.info(f"Building TtMistral4TextModel ({_N_LAYERS} layers)...")
    _log_mem("before TtMistral4TextModel construction")
    model = TtMistral4TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text,
        num_decoder_layers=_N_LAYERS,
        max_seq_len=seq_len + 64,
    )
    _log_mem("after TtMistral4TextModel construction")

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    hidden0 = torch.nn.functional.embedding(
        input_ids,
        state_dict["language_model.model.embed_tokens.weight"].to(torch.bfloat16),
    )
    pos_emb = rotary(hidden0, position_ids)

    logger.info(f"Running TTNN prefill (seq_len={seq_len})...")
    _log_mem("before TTNN prefill")
    ttnn_logits_host = model.prefill(input_ids, pos_emb)  # [1, seq_len, vocab]
    _log_mem("after TTNN prefill")
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
