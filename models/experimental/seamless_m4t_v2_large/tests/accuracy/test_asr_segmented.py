# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VAD-segmentation fix for long-audio ASR.

SeamlessM4T is utterance-level: on a long clip it tends to translate (Hindi speech → English text)
instead of transcribing — the HF reference does this too, so it is a model property. The remedy is
to segment the long audio at silences into short chunks and transcribe each (resetting the generation
runtime between chunks), then join. This test demonstrates and gates that fix: on a long T2ST-chained
clip where unsegmented ASR flips to English, the **segmented** transcription stays in the target
script (Devanagari for hin).

See ``long_audio.segment_by_silence`` and [[seamless_m4t_v2_large_demo_asr_english]].
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.demo.demo import (  # noqa: E402
    _decode,
    _waveform_to_mono_fp32,
    make_tt_model,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.long_audio import segment_by_silence  # noqa: E402
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (  # noqa: E402
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import (  # noqa: E402
    ensure_seamless_m4t_v2_large_weights,
)
from models.experimental.seamless_m4t_v2_large.tests.accuracy.metrics import script_fraction  # noqa: E402
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (  # noqa: E402
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (  # noqa: E402
    TTSeamlessM4Tv2GenerationOutput,
)

_PROMPT = """Maya lived in a small coastal town where every morning began with the sound of fishing boats leaving the harbor. She worked at her grandfather's old bookstore, a narrow shop filled with dusty shelves, handwritten notes, and the smell of paper that had aged for decades. Most customers came looking for schoolbooks or travel guides, but Maya loved recommending forgotten stories hidden in the back corners of the store.

One rainy evening, while organizing a stack of returned books, she discovered a small blue journal tucked between two novels. The cover had no title, only a silver compass symbol that shimmered faintly under the light. Curious, she opened it and found detailed sketches of places around the town along with cryptic messages about a hidden lighthouse path that only appeared during storms.

At first, Maya thought someone was playing a prank. But the next night, as heavy clouds gathered over the sea, she noticed something unusual from the bookstore window."""

_TGT_HIN = "hin"
_MAX_NEW_TOKENS = 200
_MAX_SEC = 15.0
_MIN_DEVA_FRACTION = 0.5


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
        raise
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")
        raise


def _reset(tt_model, device) -> None:
    tt_model.clear_runtime_program_cache()
    tt_model.release_generation_runtime()
    ttnn.synchronize_device(device)


def _asr(tt_model, device, processor, tokenizer, wav, sr, common) -> str:
    ai = processor(audios=wav, sampling_rate=sr, return_tensors="pt")
    out = tt_model.generate(
        input_features=torch_feats_to_ttnn(device, ai["input_features"]),
        attention_mask=torch_ids_to_ttnn(device, ai["attention_mask"]),
        generate_speech=False,
        tgt_lang=_TGT_HIN,
        max_new_tokens=_MAX_NEW_TOKENS,
        repetition_penalty=1.0,
        **common,
    )
    return _decode(tokenizer, out.sequences)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE, indirect=["mesh_device", "device_params"])
def test_asr_long_audio_segmented_stays_in_target_language(mesh_device, device_params, reset_seeds):
    """Segmenting long audio at silences keeps ASR in the target script where unsegmented flips."""
    _ = reset_seeds
    _ = device_params

    weights_dir = _weights_dir_or_skip()
    path = os.fspath(weights_dir)
    torch.manual_seed(0)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    sr = int(getattr(cfg, "sampling_rate", 16000))

    text_enc = processor(text=_PROMPT, src_lang="eng", return_tensors="pt")
    common = dict(do_sample=False, num_beams=1, use_kv_cache=True, use_decode_trace=True, use_2cq=True)

    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, model, cfg, t2u_cfg)

        # Long Hindi speech from T2ST.
        t2st_out = tt_model.generate(
            input_ids=torch_ids_to_ttnn(mesh_device, text_enc["input_ids"]),
            attention_mask=torch_ids_to_ttnn(mesh_device, text_enc["attention_mask"]),
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang=_TGT_HIN,
            speaker_id=0,
            max_new_tokens=_MAX_NEW_TOKENS,
            repetition_penalty=1.1,
            **common,
        )
        assert isinstance(t2st_out, TTSeamlessM4Tv2GenerationOutput), type(t2st_out)
        hin_wav = _waveform_to_mono_fp32(t2st_out.waveform, t2st_out.waveform_lengths)

        # (a) Unsegmented ASR over the whole clip — the regime that flips to English (logged contrast).
        _reset(tt_model, mesh_device)
        hyp_full = _asr(tt_model, mesh_device, processor, tokenizer, hin_wav, sr, common)
        deva_full = script_fraction(hyp_full, "deva")

        # (b) Segmented ASR: split at silences, transcribe each chunk (reset between), join.
        segments = segment_by_silence(hin_wav, sr, max_sec=_MAX_SEC)
        parts = []
        for seg in segments:
            _reset(tt_model, mesh_device)
            parts.append(_asr(tt_model, mesh_device, processor, tokenizer, seg, sr, common))
        hyp_seg = " ".join(p.strip() for p in parts if p.strip())

    deva_seg = script_fraction(hyp_seg, "deva")
    dur = hin_wav.size / sr
    logger.info(f"[seg ASR] audio={hin_wav.size} ({dur:.1f}s) -> {len(segments)} segments (<= {_MAX_SEC:.0f}s)")
    logger.info(f"[seg ASR] UNSEGMENTED deva={deva_full:.3f}: {hyp_full[:140]}")
    logger.info(f"[seg ASR] SEGMENTED   deva={deva_seg:.3f}: {hyp_seg[:140]}")

    assert len(segments) >= 2, f"audio ({dur:.1f}s) not segmented (got {len(segments)}); raise prompt length"
    assert len(hyp_seg.strip()) > 0, "segmented ASR produced empty output"
    assert deva_seg >= _MIN_DEVA_FRACTION, (
        f"segmented ASR still not in target script: Devanagari {deva_seg:.3f} < {_MIN_DEVA_FRACTION} "
        f"(unsegmented was {deva_full:.3f}). hyp[:120]={hyp_seg[:120]!r}"
    )
    if deva_full < _MIN_DEVA_FRACTION:
        logger.info(f"[seg ASR] VAD restored target language: {deva_full:.3f} -> {deva_seg:.3f} Devanagari")
