# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E WER helpers for speech-output tasks (T2ST, S2ST).

Compares TT ``generate(generate_speech=True)`` intermediate translation text to offline HF
reference text (same inputs and generation kwargs). Uses ``jiwer.wer`` like the Whisper demo.

Includes sweep utilities (ISL ladder, thresholds, reference ensure), in-process result storage
for pytest summaries, and device run helpers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.demo_perf_sweep import (
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    sequence_lengths,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    SPEECH_OUTPUT_TASKS,
    TEXT_INPUT_TASKS,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_seamless_m4t_v2_model import (
    _make_tt_model,
    _torch_feats_to_ttnn,
    _torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    DEVICE_PARAMS_E2E_2CQ_GENERATE,
    DEVICE_PARAMS_E2E_2CQ_GENERATE_SINGLE,
    MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE_SINGLE_AND_1X4,
    MESH_SHAPE_SINGLE,
    _mesh_device_param,
    _mesh_device_param_for_shape,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import TTSeamlessM4Tv2GenerationOutput

# ---------------------------------------------------------------------------
# WER thresholds and sweep lengths
# ---------------------------------------------------------------------------

T2ST_WER_THRESHOLD = 0.35
S2ST_WER_THRESHOLD = 0.45
SANITY_SWEEP_LENGTHS = (32, 64, 128)

_REF_DIR = Path(__file__).resolve().parent.parent / "teacher_forced_sweep_outputs" / "wer_references"

_SWEEP_LEN_WER_OVERRIDES: dict[tuple[str, int], float] = {
    ("t2st", 256): 0.42,
    ("t2st", 512): 0.45,
    ("t2st", 1024): 0.48,
    ("t2st", 2048): 0.52,
    ("t2st", 4096): 0.55,
    ("s2st", 512): 0.50,
    ("s2st", 1024): 0.52,
    ("s2st", 2048): 0.55,
    ("s2st", 4096): 0.58,
}

_P150_WER_SCALE = 1.25
_P150_SWEEP_LEN_WER_OVERRIDES: dict[tuple[str, int], float] = {
    key: min(1.0, val * _P150_WER_SCALE) for key, val in _SWEEP_LEN_WER_OVERRIDES.items()
}

# ---------------------------------------------------------------------------
# In-process result store (pytest summary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WerResult:
    label: str
    seq_len: int
    wer: float
    wer_threshold: float
    reference_words: int
    tt_words: int
    passed: bool


_WER_RESULTS: list[WerResult] = []


def clear_wer_results() -> None:
    _WER_RESULTS.clear()


def get_wer_results() -> list[WerResult]:
    return list(_WER_RESULTS)


def record_wer_result(
    *,
    label: str,
    seq_len: int,
    wer: float,
    wer_threshold: float,
    reference_words: int,
    tt_words: int,
) -> None:
    _WER_RESULTS.append(
        WerResult(
            label=label,
            seq_len=seq_len,
            wer=wer,
            wer_threshold=wer_threshold,
            reference_words=reference_words,
            tt_words=tt_words,
            passed=wer <= wer_threshold,
        )
    )


def print_wer_summary() -> None:
    results = get_wer_results()
    if not results:
        return
    print("\n" + "=" * 72)
    print("Seamless M4T v2 E2E WER sweep summary")
    print("=" * 72)
    header = f"{'Label':<18} {'Len':>5}  {'WER':>7}  {'Thr':>7}  {'RefW':>5}  {'TTW':>5}  {'Status':>6}"
    print(header)
    print("-" * len(header))
    for row in results:
        status = "PASS" if row.passed else "FAIL"
        print(
            f"{row.label:<18} {row.seq_len:>5}  {row.wer:>6.3f}  {row.wer_threshold:>6.3f}  "
            f"{row.reference_words:>5}  {row.tt_words:>5}  {status:>6}"
        )
    passed = sum(1 for r in results if r.passed)
    print("-" * len(header))
    print(f"Total: {passed}/{len(results)} passed\n")


# ---------------------------------------------------------------------------
# Reference paths and text metrics
# ---------------------------------------------------------------------------


def wer_refpt_path(task: str, seq_len: int) -> Path:
    return _REF_DIR / f"seamless_m4t_v2_{task}_len{seq_len}.refpt"


def weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def normalize_text_for_wer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def word_count(text: str) -> int:
    norm = normalize_text_for_wer(text)
    return len(norm.split()) if norm else 0


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    ref = normalize_text_for_wer(reference_text)
    hyp = normalize_text_for_wer(hypothesis_text)
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))


@dataclass(frozen=True)
class WerSweepReference:
    task: str
    seq_len: int
    tgt_lang: str
    reference_text: str
    src_ids: torch.Tensor | None = None
    src_mask: torch.Tensor | None = None
    input_features: torch.Tensor | None = None
    mel_attention_mask: torch.Tensor | None = None


def load_wer_sweep_reference(path: Path) -> WerSweepReference:
    data = torch.load(path, map_location="cpu", weights_only=False)
    task = str(data["task"])
    return WerSweepReference(
        task=task,
        seq_len=int(data["seq_len"]),
        tgt_lang=str(data["tgt_lang"]),
        reference_text=str(data["reference_text"]),
        src_ids=data.get("src_ids"),
        src_mask=data.get("src_mask"),
        input_features=data.get("input_features"),
        mel_attention_mask=data.get("mel_attention_mask"),
    )


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def sweep_sequence_lengths() -> list[int]:
    return sequence_lengths(SEQ_LEN_MIN, SEQ_LEN_MAX)


def sweep_auto_ref_enabled() -> bool:
    return os.environ.get("SEAMLESS_SWEEP_AUTO_REF", "1") != "0"


def wer_sweep_mesh_parametrize():
    """Pytest mesh params for WER sweep.

    * ``MESH_DEVICE=P150`` → 1×1 only
    * ``MESH_DEVICE=BH-QB`` → 1×4 only
    * unset → both when hardware allows
    """
    mesh_env = os.environ.get("MESH_DEVICE", "").strip()
    if mesh_env == "P150":
        return (
            "mesh_device,device_params",
            [_mesh_device_param_for_shape(MESH_SHAPE_SINGLE, DEVICE_PARAMS_E2E_2CQ_GENERATE_SINGLE, id="1x1")],
        )
    if mesh_env == "BH-QB":
        return ("mesh_device,device_params", [_mesh_device_param(DEVICE_PARAMS_E2E_2CQ_GENERATE)])
    return MESH_DEVICE_PARAMETRIZE_E2E_2CQ_GENERATE_SINGLE_AND_1X4


def sweep_wer_threshold_for_task(task: str, seq_len: int, *, mesh_id: str = "") -> float:
    if task not in SPEECH_OUTPUT_TASKS:
        raise ValueError(f"WER sweep expects a speech-output task, got {task!r}")
    overrides = _P150_SWEEP_LEN_WER_OVERRIDES if mesh_id == "1x1" else _SWEEP_LEN_WER_OVERRIDES
    override = overrides.get((task, seq_len))
    if override is not None:
        return override
    if task == "t2st":
        base = T2ST_WER_THRESHOLD
    elif task == "s2st":
        base = S2ST_WER_THRESHOLD
    else:
        raise ValueError(f"unknown task {task!r}")
    if mesh_id == "1x1":
        return min(1.0, base * _P150_WER_SCALE)
    return base


def ensure_wer_sweep_reference(task: str, seq_len: int, weights_dir: str) -> Path:
    if task not in SPEECH_OUTPUT_TASKS:
        raise ValueError(f"WER sweep expects a speech-output task, got {task!r}")
    ref_path = wer_refpt_path(task, seq_len)
    if ref_path.is_file():
        return ref_path
    if not sweep_auto_ref_enabled():
        gen = "models/experimental/seamless_m4t_v2_large/scripts/generate_wer_sweep_reference.py"
        pytest.skip(
            f"Missing WER sweep reference {ref_path}. Run: "
            f"python {gen} --task {task} --seq_len {seq_len} "
            f"(or set SEAMLESS_SWEEP_AUTO_REF=1 to generate on first run)"
        )
    from models.experimental.seamless_m4t_v2_large.scripts.generate_wer_sweep_reference import (
        generate_speech_wer_sweep_reference,
        generate_text_wer_sweep_reference,
    )

    if task in TEXT_INPUT_TASKS:
        generate_text_wer_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            seq_len=seq_len,
        )
    else:
        generate_speech_wer_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            mel_frames=seq_len,
        )
    return ref_path


def maybe_skip_empty_wer_reference(ref_text: str, *, task: str, seq_len: int) -> None:
    if ref_text.strip():
        return
    pytest.skip(f"{task.upper()} WER sweep len={seq_len}: HF reference text is empty " f"(likely EOS on short input)")


def maybe_skip_short_s2st_wer(task: str, ref_text: str, seq_len: int) -> None:
    if task != "s2st" or seq_len > 64:
        return
    words = ref_text.split()
    if len(words) >= 2:
        return
    pytest.skip(
        f"S2ST WER sweep len={seq_len}: HF reference has {len(words)} word(s); "
        f"WER is unstable on ultra-short speech inputs"
    )


# ---------------------------------------------------------------------------
# TT device run
# ---------------------------------------------------------------------------


def decode_tt_sequences(tokenizer: Any, sequences_tt: ttnn.Tensor) -> str:
    ids = to_torch_replicated_first_shard(sequences_tt).to(torch.int64).cpu()
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def release_speech_generation_output(out: TTSeamlessM4Tv2GenerationOutput) -> None:
    ttnn.deallocate(out.sequences)
    ttnn.deallocate(out.waveform)
    ttnn.deallocate(out.waveform_lengths)
    if out.unit_sequences is not None:
        ttnn.deallocate(out.unit_sequences)


def run_speech_output_wer(
    mesh_device: ttnn.Device,
    hf_model: Any,
    tokenizer: Any,
    ref: WerSweepReference,
    *,
    wer_threshold: float,
    log_label: str,
) -> None:
    """Full TT ``generate(generate_speech=True)``; WER on intermediate translation text vs HF ref."""
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config
    gen_common = hf_aligned_generation_kwargs(hf_model.generation_config)
    tt_extra = dict(use_kv_cache=True, use_decode_trace=True, use_2cq=True)

    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        try:
            if ref.task == "t2st":
                assert ref.src_ids is not None and ref.src_mask is not None
                out = tt_model.generate(
                    input_ids=_torch_ids_to_ttnn(mesh_device, ref.src_ids),
                    attention_mask=_torch_ids_to_ttnn(mesh_device, ref.src_mask),
                    generate_speech=True,
                    return_intermediate_token_ids=True,
                    tgt_lang=ref.tgt_lang,
                    speaker_id=0,
                    **gen_common,
                    **tt_extra,
                )
            else:
                assert ref.task == "s2st"
                assert ref.input_features is not None and ref.mel_attention_mask is not None
                mel_frames = int(ref.mel_attention_mask.sum().item())
                tt_model.prewarm_speech_encoder([mel_frames])
                out = tt_model.generate(
                    input_features=_torch_feats_to_ttnn(mesh_device, ref.input_features),
                    attention_mask=_torch_ids_to_ttnn(mesh_device, ref.mel_attention_mask),
                    generate_speech=True,
                    return_intermediate_token_ids=True,
                    tgt_lang=ref.tgt_lang,
                    speaker_id=0,
                    **gen_common,
                    **tt_extra,
                )

            if not isinstance(out, TTSeamlessM4Tv2GenerationOutput):
                raise TypeError(f"expected TTSeamlessM4Tv2GenerationOutput, got {type(out)}")

            tt_text = decode_tt_sequences(tokenizer, out.sequences)
            release_speech_generation_output(out)
        finally:
            tt_model.release_generation_runtime()

    wer = compute_wer(ref.reference_text, tt_text)
    ref_words = word_count(ref.reference_text)
    tt_words = word_count(tt_text)
    logger.info(
        f"SeamlessM4Tv2 E2E WER ({log_label}) seq={ref.seq_len} "
        f"WER={wer:.4f} (threshold<={wer_threshold:.3f}, ref_words={ref_words}, tt_words={tt_words})"
    )
    logger.info(f"  HF reference: {ref.reference_text[:240]}{'…' if len(ref.reference_text) > 240 else ''}")
    logger.info(f"  TT output:    {tt_text[:240]}{'…' if len(tt_text) > 240 else ''}")

    record_wer_result(
        label=log_label,
        seq_len=ref.seq_len,
        wer=wer,
        wer_threshold=wer_threshold,
        reference_words=ref_words,
        tt_words=tt_words,
    )
    assert wer <= wer_threshold, (
        f"{log_label}: WER {wer:.4f} > threshold {wer_threshold:.3f} "
        f"(ref_words={ref_words}, tt_words={tt_words}). TT text: {tt_text!r}"
    )
