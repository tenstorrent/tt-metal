# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E ISL sweep wall-clock perf for ``4p_climate_100min`` (demo-aligned).

Loads the demo script + voice clones once, then for each ISL:

  1. Crop the processor batch to the first ``ISL`` tokens (post-tokenization).
  2. Untimed warmup generate (warm program cache).
  3. Timed ``generate(max_new_tokens=None)`` — AR until EOS / ``max_length_times × ISL``.

Reports per ISL (same fields as ``demo.py`` meta):

  prefill_s, prefill_tok_s, ttft_s, decode_tok_s, ms_per_tok_steady, e2e_s, ar_tokens_generated

Env::

    VV_ISL_SWEEP=32,64,128          # override ISL list (default: 32…16384 + full prompt length)
    VV_ISL_SWEEP_MAX_ISL=2048       # drop checkpoints above this
    VV_ISL_WARMUP_TOKENS=4             # warmup AR steps (default 4)
    VV_ISL_MAX_LENGTH_TIMES=2.0        # AR budget multiplier (default 2)
    VV_TRACE_SEGMENT=0                 # disable fused-frame trace (on by default for max perf)

Run (from tt-metal root)::

    pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.vibevoice.common.config import TEXT_EXAMPLES_DIR
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.common.resource_utils import (
    DEMO_VOICE_CLONES,
    build_voice_samples,
    ensure_demo_resources,
    load_script,
    voice_preset_demo_id,
)
from models.experimental.vibevoice.demo.perf_metrics import (
    crop_processor_inputs_to_isl,
    default_isl_sweep,
    format_perf_line,
    summarize_generate_perf,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_DEMO_ID = "4p_climate_100min"
_CFG_SCALE = 1.3
_NUM_STEPS = 10
_SEED = 0


def _isl_list(full_len: int) -> list[int]:
    raw = os.environ.get("VV_ISL_SWEEP")
    if raw and raw.strip():
        isls = [int(x) for x in raw.split(",") if x.strip()]
    else:
        isls = default_isl_sweep(max_tokens=full_len)
    cap = os.environ.get("VV_ISL_SWEEP_MAX_ISL")
    if cap is not None and str(cap).strip():
        cap_v = int(cap)
        isls = [n for n in isls if n <= cap_v]
    # Never exceed the tokenized prompt.
    return [n for n in isls if n <= full_len]


def _warmup_tokens() -> int:
    return max(1, int(os.environ.get("VV_ISL_WARMUP_TOKENS", "4")))


def _max_length_times() -> float:
    return float(os.environ.get("VV_ISL_MAX_LENGTH_TIMES", "2.0"))


def _trace_enabled() -> bool:
    # Default ON (demo --trace) for peak decode; set VV_TRACE_SEGMENT=0 for eager.
    return os.environ.get("VV_TRACE_SEGMENT", "1") != "0"


def _device_params() -> dict:
    params = {"l1_small_size": 32768}
    if _trace_enabled():
        params.update(trace_region_size=1_400_000_000, num_command_queues=2)
    return params


def _build_full_processor_batch(model_path: str) -> tuple[object, dict, int]:
    from processor.vibevoice_processor import VibeVoiceProcessor

    ensure_demo_resources()
    text_path = TEXT_EXAMPLES_DIR / f"{_DEMO_ID}.txt"
    assert text_path.is_file(), f"Missing demo text: {text_path}"
    script = load_script(text_path)

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    kwargs = {
        "text": [script],
        "padding": True,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }
    if voice_preset_demo_id(_DEMO_ID) in DEMO_VOICE_CLONES:
        voices, _ = build_voice_samples(script, voice_preset_demo_id(_DEMO_ID))
        kwargs["voice_samples"] = [voices]

    inputs = processor(**kwargs)
    full_len = int(inputs["input_ids"].shape[1])
    return processor, inputs, full_len


def _generate_kwargs(processor, inputs: dict, *, max_new_tokens, max_length_times: float) -> dict:
    kw = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "speech_input_mask": inputs["speech_input_mask"],
        "tokenizer": processor.tokenizer,
        "cfg_scale": _CFG_SCALE,
        "num_diffusion_steps": _NUM_STEPS,
        "max_new_tokens": max_new_tokens,
        "max_length_times": max_length_times,
    }
    if inputs.get("speech_tensors") is not None:
        kw["speech_tensors"] = inputs["speech_tensors"]
        kw["speech_masks"] = inputs["speech_masks"]
    return kw


@pytest.mark.timeout(28800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_isl_sweep_4p_climate_100min(mesh_device, device_params):
    """Warm-cache ISL sweep on ``4p_climate_100min`` with demo-aligned metrics."""
    del device_params
    if _trace_enabled():
        os.environ["VV_TRACE_SEGMENT"] = "1"
        logger.info("VV_TRACE_SEGMENT=1 — fused-frame trace enabled for timed generates (default)")
    else:
        os.environ["VV_TRACE_SEGMENT"] = "0"
        logger.info("VV_TRACE_SEGMENT=0 — eager decode")

    try:
        model_path = str(ensure_model_weights(None))
    except Exception as exc:
        pytest.skip(f"VibeVoice weights unavailable: {exc}")

    processor, full_inputs, full_len = _build_full_processor_batch(model_path)
    isls = _isl_list(full_len)
    if not isls:
        pytest.skip(f"No ISL checkpoints ≤ tokenized length {full_len}")

    warmup_n = _warmup_tokens()
    max_length_times = _max_length_times()
    logger.info(
        f"ISL sweep demo={_DEMO_ID} full_tokens={full_len} isls={isls} "
        f"warmup_tokens={warmup_n} max_length_times={max_length_times} "
        f"max_new_tokens=None"
    )

    print("[isl_sweep] Loading TTVibeVoiceModel...", flush=True)
    t_load0 = time.perf_counter()
    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device,
        model_path,
        cfg_scale=_CFG_SCALE,
        num_diffusion_steps=_NUM_STEPS,
    )
    logger.info(f"model load: {time.perf_counter() - t_load0:.1f}s")

    rows: list[dict] = []
    for isl in isls:
        cropped = crop_processor_inputs_to_isl(full_inputs, isl)
        prefill_len = int(cropped["input_ids"].shape[1])
        assert prefill_len == isl

        # Untimed warmup at this ISL (program-cache warm).
        warm_kw = _generate_kwargs(processor, cropped, max_new_tokens=warmup_n, max_length_times=max_length_times)
        print(f"[isl_sweep] ISL={isl}: warmup max_new_tokens={warmup_n}...", flush=True)
        torch.manual_seed(_SEED)
        _ = tt_model.generate(**warm_kw)
        ttnn.synchronize_device(mesh_device)

        # Timed generate: max_new_tokens=None (EOS / max_length_times × ISL).
        timed_kw = _generate_kwargs(processor, cropped, max_new_tokens=None, max_length_times=max_length_times)
        print(f"[isl_sweep] ISL={isl}: timed generate (max_new_tokens=None)...", flush=True)
        torch.manual_seed(_SEED)
        t0 = time.perf_counter()
        tt_out = tt_model.generate(**timed_kw)
        ttnn.synchronize_device(mesh_device)
        e2e_s = time.perf_counter() - t0

        ar_tokens = int(tt_out.sequences.shape[1] - prefill_len)
        metrics = summarize_generate_perf(
            prefill_len=prefill_len,
            ar_tokens=ar_tokens,
            prefill_wall_s=tt_out.prefill_wall_s,
            decode_wall_s=tt_out.decode_wall_s,
            generate_wall_s=e2e_s,
            steady_decode_s=tt_out.steady_decode_s,
            steady_decode_frames=tt_out.steady_decode_frames,
        )
        metrics["isl"] = isl
        rows.append(metrics)
        print(f"[isl_sweep] {format_perf_line(metrics, prefix=f'ISL={isl}  ')}", flush=True)
        logger.info(metrics)

    # Summary table
    print("\n[isl_sweep] ===== summary =====", flush=True)
    hdr = (
        f"{'ISL':>6} {'prefill_s':>10} {'pref_tok/s':>10} {'TTFT_s':>8} "
        f"{'dec_tok/s':>10} {'ms/tok':>8} {'e2e_s':>10} {'ar_tok':>8}"
    )
    print(hdr, flush=True)
    for m in rows:
        print(
            f"{m['prefill_tokens']:6d} {m['prefill_s']:10.3f} {m['prefill_tok_s']:10.1f} "
            f"{m['ttft_s']:8.3f} {m['decode_tok_s']:10.2f} {m['ms_per_tok_steady']:8.2f} "
            f"{m['e2e_s']:10.3f} {m['ar_tokens_generated']:8d}",
            flush=True,
        )

    assert rows, "ISL sweep produced no rows"
    for m in rows:
        assert m["prefill_s"] > 0, f"empty prefill timing at ISL={m['prefill_tokens']}"
        assert m["ar_tokens_generated"] >= 0
