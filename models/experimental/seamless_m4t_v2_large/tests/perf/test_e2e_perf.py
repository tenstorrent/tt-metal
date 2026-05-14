# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seamless M4T v2 Large — E2E performance (DINO-style checkpoints) for **all five** PCC tasks.

Each task logs macro wall-clock stages with ``time.perf_counter()`` and ``ttnn.synchronize_device``
after device work. Stage **names** differ by modality (text vs speech encoder; optional T2U stack).

Tasks (same coverage as ``tests/pcc/test_seamless_m4t_v2_model.py``):

  * **T2TT** — text → text
  * **S2TT** — speech → text
  * **T2ST** — text → speech (text-decoder path + T2U PCC path)
  * **S2ST** — speech → speech (same as T2ST but speech in)
  * **ASR** — speech → text (transcribe); same ``forward`` shape as S2TT in PCC tests

**Totals:** primary **TOTAL (incl. PCC)** / **Invocations/s (full)** include all validation. **Inference
only** sums stages whose names do not contain ``PCC`` (device + H2D + T2U host prep / H2D / forward).

Usage::

    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf.py -v
    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf.py -v -k t2tt
    pytest models/experimental/seamless_m4t_v2_large/tests/perf/test_e2e_perf.py -v -m models_performance_bare_metal
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    forward_text_modality_logits,
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_text_to_unit import (
    forward_t2u_logits_and_padding,
    hf_discrete_duration_counts_batch1,
    synthetic_t2u_inputs,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_seamless_m4t_v2_model import (
    PCC_THRESHOLD,
    _decoder_seed,
    _real_speech_features,
    _real_text_input_ids,
    _weights_dir_or_skip,
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)

# task_id -> (use_speech_input, tgt_lang, run_t2u_after_text)
_TASKS: Dict[str, Tuple[bool, str, bool]] = {
    "t2tt": (False, "eng", False),
    "s2tt": (True, "eng", False),
    "t2st": (False, "eng", True),
    "s2st": (True, "eng", True),
    "asr": (True, "eng", False),
}


def _log_timings_banner(
    title: str,
    timings: Dict[str, float],
    total_s: float,
) -> None:
    total_ms = total_s * 1000.0
    inference_only_s = sum(ms for name, ms in timings.items() if "pcc" not in name.lower()) / 1000.0
    logger.info("")
    logger.info("=" * 65)
    logger.info(title)
    logger.info("=" * 65)
    for name, ms in timings.items():
        pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
        logger.info(f"  {name:28s}  {ms:9.1f} ms  ({pct:5.1f}%)")
    logger.info(f"  {'─' * 45}")
    logger.info(f"  {'TOTAL (incl. PCC)':28s}  {total_ms:9.1f} ms  (100.0%)")
    logger.info(f"  {'Invocations/s (full)':28s}  {1.0 / total_s:9.2f}")
    if inference_only_s > 0:
        logger.info(f"  {'─' * 45}")
        logger.info(f"  {'Inference only (excl. PCC)':28s}  {inference_only_s * 1000.0:9.1f} ms")
        logger.info(f"  {'Invocations/s (inf only)':28s}  {1.0 / inference_only_s:9.2f}")
    logger.info("=" * 65)


def _assert_text_logits_pcc_local(
    ref_logits: torch.Tensor, logits_tt: ttnn.Tensor, *, ctx: str, pcc: float = PCC_THRESHOLD
) -> None:
    ref_f = ref_logits.detach().float().cpu()
    _, sd, v = ref_f.shape
    if logits_tt.storage_type() == ttnn.StorageType.DEVICE:
        flat = ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.bfloat16).contiguous().reshape(-1)
    else:
        flat = ttnn.to_torch(logits_tt).to(torch.bfloat16).contiguous().reshape(-1)
    sp = flat.numel() // v
    tt_f = flat.reshape(1, sp, v)[:, :sd, :v].contiguous().float().cpu()
    assert tt_f.shape == ref_f.shape, f"{ctx}: shape ref {tuple(ref_f.shape)} vs ttnn {tuple(tt_f.shape)}"
    ok, msg = check_with_pcc(ref_f, tt_f, pcc=pcc)
    logger.info(f"{ctx} text-decoder logits PCC: {msg}")
    assert ok, f"{ctx}: text-decoder logits PCC < {pcc}: {msg}"


def _t2u_timed_stages(model: Any, tt_model: Any, device: ttnn.Device, *, ctx: str) -> Dict[str, float]:
    """T2U PCC path (matches PCC test) with per-stage ms. Returns stage name -> ms."""
    t2u_cfg = model.t2u_model.config
    t_prep0 = time.perf_counter()
    inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
        t2u_cfg,
        batch=1,
        encoder_seq_len=32,
        seed=1,
        dtype=torch.bfloat16,
    )
    hf_dev = next(model.t2u_model.parameters()).device
    char_count_per_id_dev = char_count_per_id.to(hf_dev)

    ref_logits, _ = forward_t2u_logits_and_padding(
        model.t2u_model,
        inputs_embeds,
        attention_mask,
        char_input_ids,
        char_count_per_id_dev,
    )
    ref_logits_bf16 = ref_logits.to(torch.bfloat16).cpu()

    ref_durs = hf_discrete_duration_counts_batch1(
        model.t2u_model,
        inputs_embeds.to(hf_dev),
        attention_mask.to(hf_dev),
        char_input_ids.to(hf_dev),
        char_count_per_id_dev,
    )
    t_prep1 = time.perf_counter()

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
    inputs_embeds_tt = ttnn.from_torch(
        inputs_embeds.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_tt = ttnn.from_torch(
        mask_4d.cpu().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_ids_tt = ttnn.from_torch(
        char_input_ids.cpu().to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cc_list = [int(x) for x in char_count_per_id[0].cpu().tolist()]
    ttnn.synchronize_device(device)
    t_h2d = time.perf_counter()

    tt_logits_tt, _ = tt_model.t2u.forward(
        inputs_embeds_tt,
        attn_tt,
        char_ids_tt,
        cc_list,
        reference_discrete_durations=ref_durs,
    )
    ttnn.synchronize_device(device)
    t_fwd = time.perf_counter()

    ttnn.deallocate(inputs_embeds_tt)
    ttnn.deallocate(attn_tt)
    ttnn.deallocate(char_ids_tt)

    tt_logits = ttnn.to_torch(ttnn.from_device(tt_logits_tt)).to(torch.bfloat16).cpu()
    ttnn.deallocate(tt_logits_tt)

    v = int(ref_logits_bf16.shape[-1])
    flat = tt_logits.reshape(-1)
    sp = flat.numel() // v
    tt_logits_3d = flat.reshape(1, sp, v)[:, : ref_logits_bf16.shape[1], :].contiguous()
    assert (
        tt_logits_3d.shape == ref_logits_bf16.shape
    ), f"{ctx}: T2U logits shape ref={tuple(ref_logits_bf16.shape)} tt={tuple(tt_logits_3d.shape)}"
    ok, msg = check_with_pcc(ref_logits_bf16.float(), tt_logits_3d.float(), pcc=PCC_THRESHOLD)
    logger.info(f"{ctx} T2U logits PCC: {msg}")
    assert ok, f"{ctx}: T2U logits PCC < {PCC_THRESHOLD}: {msg}"
    t_end = time.perf_counter()

    return {
        "T2U host prep (HF+syn)": (t_prep1 - t_prep0) * 1000.0,
        "T2U Host→Device": (t_h2d - t_prep1) * 1000.0,
        "T2U forward": (t_fwd - t_h2d) * 1000.0,
        "T2U PCC validation": (t_end - t_fwd) * 1000.0,
    }


# Shared with ``test_e2e_perf_2cq.py`` (Pipeline host outputs may be TTNN on HOST).
SEAMLESS_E2E_TASKS = _TASKS
assert_text_logits_pcc_vs_ref = _assert_text_logits_pcc_local
t2u_timed_stages_for_e2e = _t2u_timed_stages


@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("task", list(_TASKS.keys()))
def test_e2e_perf(device, reset_seeds, task: str):
    """One timed E2E pass per canonical task (T2TT / S2TT / T2ST / S2ST / ASR)."""
    _ = reset_seeds
    use_speech, tgt_lang, needs_t2u = _TASKS[task]
    weights_dir = _weights_dir_or_skip()

    logger.info(f"[{task.upper()}] Loading HF checkpoint + building TT model...")
    t_load = time.perf_counter()
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device
    tt_model = make_tt_model(device, model, cfg, t2u_cfg)
    t_load_end = time.perf_counter()
    logger.info(f"[{task.upper()}] Weight loading + model init: {t_load_end - t_load:.2f}s")

    decoder_input_ids, decoder_attention_mask = _decoder_seed(cfg, dev, tgt_lang=tgt_lang)

    if use_speech:
        processor = AutoProcessor.from_pretrained(os.fspath(weights_dir), local_files_only=True)
        input_features, enc_attn = _real_speech_features(processor, dev)
        with torch.no_grad():
            ref_out = model(
                input_features=input_features,
                attention_mask=enc_attn,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
        ref_logits = ref_out.logits.to(torch.bfloat16).cpu().float()
    else:
        tokenizer = AutoTokenizer.from_pretrained(os.fspath(weights_dir), local_files_only=True)
        input_ids, enc_attn = _real_text_input_ids(tokenizer, dev)
        ref_logits = (
            forward_text_modality_logits(
                model,
                input_ids=input_ids,
                attention_mask=enc_attn,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            .to(torch.bfloat16)
            .cpu()
        )

    enc_label = "Speech encoder" if use_speech else "Text encoder"
    logger.info(f"[{task.upper()}] Starting E2E inference (1 iteration)...")
    t_start = time.perf_counter()

    input_ids_tt: ttnn.Tensor | None = None
    if use_speech:
        enc_in_tt = torch_feats_to_ttnn(device, input_features)
        enc_attn_tt = torch_ids_to_ttnn(device, enc_attn.cpu())
    else:
        input_ids_tt = torch_ids_to_ttnn(device, input_ids)
        enc_attn_tt = torch_ids_to_ttnn(device, enc_attn)

    dec_ids_tt = torch_ids_to_ttnn(device, decoder_input_ids)
    dec_mask_tt = torch_ids_to_ttnn(device, decoder_attention_mask)
    ttnn.synchronize_device(device)
    t_h2d = time.perf_counter()

    if use_speech:
        enc_tt, enc_attn_padded, enc_attn_owned = tt_model._encode_speech(enc_in_tt, enc_attn_tt)
    else:
        enc_tt, enc_attn_padded, enc_attn_owned = tt_model._encode_text(input_ids_tt, enc_attn_tt)
    ttnn.synchronize_device(device)
    t_encoder = time.perf_counter()

    logits = tt_model._decode_and_lm_head(enc_tt, enc_attn_padded, dec_ids_tt, dec_mask_tt)
    ttnn.synchronize_device(device)
    t_decoder = time.perf_counter()

    _assert_text_logits_pcc_local(ref_logits, logits, ctx=f"{task.upper()}_E2E_PERF")
    t_text_pcc = time.perf_counter()

    timings: Dict[str, float] = {
        "Host→Device": (t_h2d - t_start) * 1000.0,
        enc_label: (t_encoder - t_h2d) * 1000.0,
        "Decoder+lm_head": (t_decoder - t_encoder) * 1000.0,
        "Text PCC validation": (t_text_pcc - t_decoder) * 1000.0,
    }

    if enc_attn_owned:
        ttnn.deallocate(enc_attn_padded)
    ttnn.deallocate(enc_tt)
    ttnn.deallocate(logits)

    ttnn.deallocate(dec_ids_tt)
    ttnn.deallocate(dec_mask_tt)
    if use_speech:
        ttnn.deallocate(enc_in_tt)
        ttnn.deallocate(enc_attn_tt)
    else:
        ttnn.deallocate(input_ids_tt)
        ttnn.deallocate(enc_attn_tt)

    t_end = t_text_pcc
    if needs_t2u:
        t2u_timings = _t2u_timed_stages(model, tt_model, device, ctx=f"{task.upper()}_E2E_T2U")
        timings.update(t2u_timings)
        t_end = time.perf_counter()

    total_s = t_end - t_start
    title = f"Seamless M4T v2 Large  E2E ({task.upper()}, 1 iter)"
    _log_timings_banner(title, timings, total_s)
