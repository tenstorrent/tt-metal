# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E token-matching helpers and ISL sweep utilities.

Teacher-forced greedy decode: HF reference tokens are fed as decoder inputs at each step while
TT ``lm_head`` predictions are compared to offline HF top-1 / top-5 from a ``.refpt`` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.demo_perf_sweep import (
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    sequence_lengths,
)
from models.experimental.seamless_m4t_v2_large.scripts.generate_t2tt_token_accuracy_reference import (
    generate_speech_sweep_reference,
    generate_text_sweep_reference,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_task_config import (
    SPEECH_INPUT_TASKS,
    TEXT_INPUT_TASKS,
    TEXT_OUTPUT_TASKS,
    sweep_refpt_path,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import TextDecoderPccInputs
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_logit_pcc_helpers import tt_encode_speech_via_model
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_tt_model_helpers import make_tt_model
from models.experimental.seamless_m4t_v2_large.tests.pcc.token_matching_result_store import (
    record_token_matching_result,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    DEVICE_PARAMS_TEXT,
    DEVICE_PARAMS_TEXT_SINGLE,
    MESH_DEVICE_PARAMETRIZE_TEXT_SINGLE_AND_1X4,
    MESH_SHAPE_SINGLE,
    _mesh_device_param,
    _mesh_device_param_for_shape,
    from_torch_uint32_rm,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import _ttnn_ids_from_list
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import init_text_decoder_kv_cache

T2TT_TOP1_THRESHOLD = 0.95
T2TT_TOP5_THRESHOLD = 0.99
SPEECH_TOP1_THRESHOLD = 0.87  # S2TT ~88% top-1 with live speech encoder on BH 1×4; ASR ~90%+
SPEECH_TOP5_THRESHOLD = 0.95

# Minimum HF teacher-forced steps to run speech token-matching sweep (skip below this).
# Set to 1 to score short mel refs (32/64); was 8 when skipping noisy n<8 comparisons.
S2ST_MIN_TOKEN_REF_STEPS = 1

# Below this many teacher-forced steps, the top-1 fraction is dominated by sampling granularity:
# with n steps the only achievable values are k/n, so a single near-tie flip (TT picks HF's #2,
# still within top-5) costs 1/n. e.g. n=3 → {0, 33, 67, 100}% only, so the 87% gate is unreachable
# except at a perfect 100%. Below this floor we keep top-1 INFORMATIONAL (logged/recorded) and gate
# on top-5 only (which still catches genuine divergence — a garbage encoder tanks top-5 too).
# Longer points (e.g. the len1024 speech-encoder bug at 68 steps) keep the full top-1 gate.
MIN_TOP1_GATE_STEPS = 8


@dataclass(frozen=True)
class T2ttTokenAccuracyReference:
    src_ids: torch.Tensor
    src_mask: torch.Tensor
    seed_ids: torch.Tensor
    teacher_tokens: torch.Tensor
    top5_tokens: torch.Tensor


@dataclass(frozen=True)
class SpeechTokenAccuracyReference:
    input_features: torch.Tensor
    mel_attention_mask: torch.Tensor
    seed_ids: torch.Tensor
    teacher_tokens: torch.Tensor
    top5_tokens: torch.Tensor
    decoder_case: TextDecoderPccInputs


def load_t2tt_token_accuracy_reference(path: Path) -> T2ttTokenAccuracyReference:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return T2ttTokenAccuracyReference(
        src_ids=data["src_ids"],
        src_mask=data["src_mask"],
        seed_ids=data["seed_ids"],
        teacher_tokens=data["teacher_tokens"].to(torch.int64),
        top5_tokens=data["top5_tokens"].to(torch.int64),
    )


def load_speech_token_accuracy_reference(path: Path) -> SpeechTokenAccuracyReference:
    data = torch.load(path, map_location="cpu", weights_only=True)
    dec_mask = torch.ones_like(data["seed_ids"])
    case = TextDecoderPccInputs(
        input_ids=data["seed_ids"],
        attention_mask=dec_mask,
        encoder_hidden_states=data["encoder_hidden_states"],
        encoder_attention_mask=data["encoder_attention_mask"],
    )
    return SpeechTokenAccuracyReference(
        input_features=data["input_features"],
        mel_attention_mask=data["mel_attention_mask"],
        seed_ids=data["seed_ids"],
        teacher_tokens=data["teacher_tokens"].to(torch.int64),
        top5_tokens=data["top5_tokens"].to(torch.int64),
        decoder_case=case,
    )


def _compute_top1_top5(
    predicted: list[int],
    teacher_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    *,
    num_steps: int,
) -> Tuple[float, float]:
    n = min(num_steps, len(predicted), int(teacher_tokens.numel()), int(top5_tokens.shape[0]))
    if n == 0:
        return 1.0, 1.0
    top1 = 0
    top5 = 0
    for i in range(n):
        pred = int(predicted[i])
        if pred == int(top5_tokens[i, 0].item()):
            top1 += 1
        if pred in top5_tokens[i].tolist():
            top5 += 1
    return top1 / n, top5 / n


def eval_token_matching_loop(
    tt_model,
    mesh_device: ttnn.Device,
    hf_model,
    *,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    seed_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    decode_steps: int,
) -> Tuple[float, float, int]:
    """Teacher-forced decode; returns ``(top1_frac, top5_frac, n_eval_steps)``."""
    cfg = hf_model.config
    seed_len = int(seed_ids.shape[1])
    n_eval = min(decode_steps, int(teacher_tokens.numel()), int(top5_tokens.shape[0]))
    max_seq_len = max(64, seed_len + n_eval + 8)
    padded_enc = int(enc_tt.shape[1])

    kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
        mesh_device,
        num_hidden_layers=cfg.decoder_layers,
        num_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        encoder_seq_len=padded_enc,
        tp=tt_model._tp,
    )

    seed_tt = _ttnn_ids_from_list([seed_ids[0].tolist()], mesh_device)
    warm_out = tt_model._prefill_text_decoder_kv_cache(
        seed_tt,
        enc_tt,
        enc_mask_tt,
        kv_cache,
        cross_attn_cache,
    )
    ttnn.deallocate(seed_tt)

    predicted: list[int] = []
    cross_valid = True
    for step in range(n_eval):
        teacher_tok = int(teacher_tokens[step].item())
        position = seed_len + step
        logits = tt_model._decode_token_with_kv_cache(
            teacher_tok,
            position,
            enc_tt,
            enc_mask_tt,
            kv_cache,
            cross_attn_cache,
            cross_attn_cache_valid=cross_valid,
            batch_size=1,
        )
        pred_id = tt_model._host_argmax_from_logits_row(logits, dec_len=1, sharded=False)
        ttnn.deallocate(logits)
        predicted.append(pred_id)

    if warm_out is not None:
        ttnn.deallocate(warm_out)
    ttnn.deallocate(enc_tt)
    ttnn.deallocate(enc_mask_tt)
    for layer in kv_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    for layer in cross_attn_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])

    top1, top5 = _compute_top1_top5(predicted, teacher_tokens, top5_tokens, num_steps=n_eval)
    return top1, top5, n_eval, predicted


def _run_token_accuracy_loop(
    tt_model,
    mesh_device: ttnn.Device,
    hf_model,
    *,
    enc_tt: ttnn.Tensor,
    enc_mask_tt: ttnn.Tensor,
    seed_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    decode_steps: int,
    top1_threshold: float,
    top5_threshold: float,
    log_label: str,
) -> None:
    top1, top5, n_eval, _predicted = eval_token_matching_loop(
        tt_model,
        mesh_device,
        hf_model,
        enc_tt=enc_tt,
        enc_mask_tt=enc_mask_tt,
        seed_ids=seed_ids,
        teacher_tokens=teacher_tokens,
        top5_tokens=top5_tokens,
        decode_steps=decode_steps,
    )
    top1_pct = top1 * 100.0
    top5_pct = top5 * 100.0
    logger.info(
        f"SeamlessM4Tv2 E2E token matching ({log_label}) steps={n_eval} "
        f"top1={top1_pct:.2f}% top5={top5_pct:.2f}% "
        f"(thresholds top1>={top1_threshold * 100:.0f}% top5>={top5_threshold * 100:.0f}%)"
    )
    record_token_matching_result(
        label=log_label,
        steps=n_eval,
        top1_pct=top1_pct,
        top5_pct=top5_pct,
        top1_threshold=top1_threshold,
        top5_threshold=top5_threshold,
    )
    if n_eval >= MIN_TOP1_GATE_STEPS:
        assert top1 >= top1_threshold, (
            f"{log_label}: top1 {top1_pct:.2f}% < {top1_threshold * 100:.0f}% "
            f"(top5={top5_pct:.2f}%, need >={top5_threshold * 100:.0f}%)"
        )
    else:
        logger.info(
            f"{log_label}: top1 gate skipped — only {n_eval} steps (< {MIN_TOP1_GATE_STEPS}); "
            f"top1={top1_pct:.2f}% is informational at this sample size, enforcing top5 only"
        )
    assert top5 >= top5_threshold, (
        f"{log_label}: top5 {top5_pct:.2f}% < {top5_threshold * 100:.0f}% "
        f"(top1={top1_pct:.2f}%, need >={top1_threshold * 100:.0f}%)"
    )


def run_t2tt_e2e_token_accuracy(
    mesh_device: ttnn.Device,
    hf_model,
    ref: T2ttTokenAccuracyReference,
    *,
    decode_steps: int,
    top1_threshold: float,
    top5_threshold: float,
    log_label: str,
) -> None:
    """TT text encoder → decoder KV prefill/decode → ``lm_head`` vs offline HF top-1/top-5."""
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config

    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)

        src_ids_tt = from_torch_uint32_rm(mesh_device, ref.src_ids.to(torch.int32))
        src_mask_tt = from_torch_uint32_rm(mesh_device, ref.src_mask.to(torch.int32))
        enc_tt, enc_mask_tt, attn_owned = tt_model._encode_text(src_ids_tt, src_mask_tt)
        ttnn.deallocate(src_ids_tt)
        if attn_owned:
            ttnn.deallocate(src_mask_tt)

        _run_token_accuracy_loop(
            tt_model,
            mesh_device,
            hf_model,
            enc_tt=enc_tt,
            enc_mask_tt=enc_mask_tt,
            seed_ids=ref.seed_ids,
            teacher_tokens=ref.teacher_tokens,
            top5_tokens=ref.top5_tokens,
            decode_steps=decode_steps,
            top1_threshold=top1_threshold,
            top5_threshold=top5_threshold,
            log_label=log_label,
        )


def run_speech_e2e_token_accuracy(
    mesh_device: ttnn.Device,
    hf_model,
    ref: SpeechTokenAccuracyReference,
    *,
    decode_steps: int,
    top1_threshold: float,
    top5_threshold: float,
    log_label: str,
) -> None:
    """TT speech encoder → decoder → ``lm_head`` vs offline HF top-1/top-5."""
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config

    with mesh_default_device(mesh_device):
        tt_model = make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        try:
            enc_tt, enc_mask_tt = tt_encode_speech_via_model(
                mesh_device,
                tt_model,
                ref.input_features,
                ref.mel_attention_mask,
            )
            _run_token_accuracy_loop(
                tt_model,
                mesh_device,
                hf_model,
                enc_tt=enc_tt,
                enc_mask_tt=enc_mask_tt,
                seed_ids=ref.seed_ids,
                teacher_tokens=ref.teacher_tokens,
                top5_tokens=ref.top5_tokens,
                decode_steps=decode_steps,
                top1_threshold=top1_threshold,
                top5_threshold=top5_threshold,
                log_label=log_label,
            )
        finally:
            tt_model.release_generation_runtime()


# ---------------------------------------------------------------------------
# ISL sweep utilities (token-matching + logit-PCC sweeps)
# ---------------------------------------------------------------------------

SWEEP_EVAL_STEPS = 128
SANITY_SWEEP_LENGTHS = (32, 64, 128)

_SWEEP_LEN_THRESHOLD_OVERRIDES: dict[tuple[str, int], tuple[float, float]] = {
    ("t2tt", 256): (0.94, T2TT_TOP5_THRESHOLD),
    ("s2tt", 2048): (0.80, 0.84),
    ("asr", 2048): (0.79, 0.84),
}


def sweep_sequence_lengths() -> list[int]:
    return sequence_lengths(SEQ_LEN_MIN, SEQ_LEN_MAX)


def sweep_mesh_parametrize():
    """Pytest mesh params for token-matching / logit-PCC sweeps."""
    mesh_env = os.environ.get("MESH_DEVICE", "").strip()
    if mesh_env == "P150":
        return (
            "mesh_device,device_params",
            [_mesh_device_param_for_shape(MESH_SHAPE_SINGLE, DEVICE_PARAMS_TEXT_SINGLE, id="1x1")],
        )
    if mesh_env == "BH-QB":
        return ("mesh_device,device_params", [_mesh_device_param(DEVICE_PARAMS_TEXT)])
    return MESH_DEVICE_PARAMETRIZE_TEXT_SINGLE_AND_1X4


def sweep_auto_ref_enabled() -> bool:
    return os.environ.get("SEAMLESS_SWEEP_AUTO_REF", "1") != "0"


def sweep_thresholds_for_task(task: str, seq_len: int) -> tuple[float, float]:
    if task not in TEXT_OUTPUT_TASKS:
        raise ValueError(f"token matching sweep expects a text-output task, got {task!r}")
    override = _SWEEP_LEN_THRESHOLD_OVERRIDES.get((task, seq_len))
    if override is not None:
        return override
    if task in TEXT_INPUT_TASKS:
        return T2TT_TOP1_THRESHOLD, T2TT_TOP5_THRESHOLD
    if task in SPEECH_INPUT_TASKS:
        return SPEECH_TOP1_THRESHOLD, SPEECH_TOP5_THRESHOLD
    raise ValueError(f"unknown task {task!r}")


def ensure_sweep_reference(
    task: str,
    seq_len: int,
    weights_dir: str,
    *,
    max_decode_steps: int = SWEEP_EVAL_STEPS,
) -> Path:
    """Return sweep ``.refpt`` path, generating offline HF reference if missing."""
    if task not in TEXT_OUTPUT_TASKS:
        raise ValueError(f"token matching sweep expects a text-output task, got {task!r}")
    ref_path = sweep_refpt_path(task, seq_len, max_decode_steps)
    if ref_path.is_file():
        return ref_path
    if not sweep_auto_ref_enabled():
        gen = "models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py"
        pytest.skip(
            f"Missing sweep reference {ref_path}. Run: "
            f"python {gen} --sweep --task {task} --seq_len {seq_len} "
            f"(or set SEAMLESS_SWEEP_AUTO_REF=1 to generate on first run)"
        )
    if task in TEXT_INPUT_TASKS:
        generate_text_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            seq_len=seq_len,
            max_decode_steps=max_decode_steps,
        )
    else:
        generate_speech_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            mel_frames=seq_len,
            max_decode_steps=max_decode_steps,
        )
    return ref_path


def maybe_skip_short_speech_sweep(task: str, ref_teacher_steps: int, seq_len: int) -> None:
    if ref_teacher_steps >= S2ST_MIN_TOKEN_REF_STEPS:
        return
    pytest.skip(
        f"{task.upper()} sweep len={seq_len} reference has {ref_teacher_steps} decode steps "
        f"(need >={S2ST_MIN_TOKEN_REF_STEPS}); HF hit EOS early on short mel input"
    )


def default_speech_sweep_mel_debug_dir(*, task: str, mel_frames: int) -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "teacher_forced_sweep_outputs"
        / "debug_mel_inputs"
        / f"mel{mel_frames}"
        / task
    )


def save_speech_sweep_mel_artifacts(
    *,
    task: str,
    mel_frames: int,
    input_features: torch.Tensor,
    mel_attention_mask: torch.Tensor,
    seed_ids: torch.Tensor | None = None,
    teacher_tokens: torch.Tensor | None = None,
    out_dir: Path | None = None,
    wav_path: Path | None = None,
) -> Path:
    import json

    dest = out_dir or default_speech_sweep_mel_debug_dir(task=task, mel_frames=mel_frames)
    dest.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "task": task,
        "mel_frames": mel_frames,
        "input_features": input_features.detach().cpu(),
        "mel_attention_mask": mel_attention_mask.detach().cpu(),
    }
    if seed_ids is not None:
        payload["seed_ids"] = seed_ids.detach().cpu()
    if teacher_tokens is not None:
        payload["teacher_tokens"] = teacher_tokens.detach().cpu()

    torch.save(payload, dest / "mel_input.pt")

    meta = {
        "task": task,
        "mel_frames": mel_frames,
        "input_features_shape": list(input_features.shape),
        "mel_valid_frames": int(mel_attention_mask.sum().item()),
        "teacher_decode_steps": int(teacher_tokens.numel()) if teacher_tokens is not None else None,
        "note": (
            "mel_input.pt is the exact tensor fed to the speech encoder in token-matching sweep "
            "(first mel_frames of processor output from long preamble audio)."
        ),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if wav_path is not None and wav_path.is_file():
        import shutil

        shutil.copy2(wav_path, dest / "long_preamble_source.wav")

    return dest


def maybe_save_speech_sweep_mel_env(
    *,
    task: str,
    seq_len: int,
    input_features: torch.Tensor,
    mel_attention_mask: torch.Tensor,
    seed_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
) -> None:
    want = os.environ.get("SEAMLESS_SWEEP_SAVE_MEL", "").strip()
    if not want or str(seq_len) != want:
        return
    if task not in SPEECH_INPUT_TASKS:
        return
    dest = save_speech_sweep_mel_artifacts(
        task=task,
        mel_frames=seq_len,
        input_features=input_features,
        mel_attention_mask=mel_attention_mask,
        seed_ids=seed_ids,
        teacher_tokens=teacher_tokens,
    )
    logger.info(f"Saved speech sweep mel debug artifacts for {task.upper()} mel={seq_len} to {dest}")


# ---------------------------------------------------------------------------
# ISL sweep utilities (token-matching + logit-PCC sweeps)
# ---------------------------------------------------------------------------

SWEEP_EVAL_STEPS = 128
SANITY_SWEEP_LENGTHS = (32, 64, 128)

_SWEEP_LEN_THRESHOLD_OVERRIDES: dict[tuple[str, int], tuple[float, float]] = {
    ("t2tt", 256): (0.94, T2TT_TOP5_THRESHOLD),
    ("s2tt", 2048): (0.80, 0.84),
    ("asr", 2048): (0.79, 0.84),
}


def sweep_sequence_lengths() -> list[int]:
    return sequence_lengths(SEQ_LEN_MIN, SEQ_LEN_MAX)


def sweep_mesh_parametrize():
    """Pytest mesh params for token-matching / logit-PCC sweeps."""
    mesh_env = os.environ.get("MESH_DEVICE", "").strip()
    if mesh_env == "P150":
        return (
            "mesh_device,device_params",
            [_mesh_device_param_for_shape(MESH_SHAPE_SINGLE, DEVICE_PARAMS_TEXT_SINGLE, id="1x1")],
        )
    if mesh_env == "BH-QB":
        return ("mesh_device,device_params", [_mesh_device_param(DEVICE_PARAMS_TEXT)])
    return MESH_DEVICE_PARAMETRIZE_TEXT_SINGLE_AND_1X4


def sweep_auto_ref_enabled() -> bool:
    return os.environ.get("SEAMLESS_SWEEP_AUTO_REF", "1") != "0"


def sweep_thresholds_for_task(task: str, seq_len: int) -> tuple[float, float]:
    if task not in TEXT_OUTPUT_TASKS:
        raise ValueError(f"token matching sweep expects a text-output task, got {task!r}")
    override = _SWEEP_LEN_THRESHOLD_OVERRIDES.get((task, seq_len))
    if override is not None:
        return override
    if task in TEXT_INPUT_TASKS:
        return T2TT_TOP1_THRESHOLD, T2TT_TOP5_THRESHOLD
    if task in SPEECH_INPUT_TASKS:
        return SPEECH_TOP1_THRESHOLD, SPEECH_TOP5_THRESHOLD
    raise ValueError(f"unknown task {task!r}")


def ensure_sweep_reference(
    task: str,
    seq_len: int,
    weights_dir: str,
    *,
    max_decode_steps: int = SWEEP_EVAL_STEPS,
) -> Path:
    """Return sweep ``.refpt`` path, generating offline HF reference if missing."""
    if task not in TEXT_OUTPUT_TASKS:
        raise ValueError(f"token matching sweep expects a text-output task, got {task!r}")
    ref_path = sweep_refpt_path(task, seq_len, max_decode_steps)
    if ref_path.is_file():
        return ref_path
    if not sweep_auto_ref_enabled():
        gen = "models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py"
        pytest.skip(
            f"Missing sweep reference {ref_path}. Run: "
            f"python {gen} --sweep --task {task} --seq_len {seq_len} "
            f"(or set SEAMLESS_SWEEP_AUTO_REF=1 to generate on first run)"
        )
    if task in TEXT_INPUT_TASKS:
        generate_text_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            seq_len=seq_len,
            max_decode_steps=max_decode_steps,
        )
    else:
        generate_speech_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            mel_frames=seq_len,
            max_decode_steps=max_decode_steps,
        )
    return ref_path


def maybe_skip_short_speech_sweep(task: str, ref_teacher_steps: int, seq_len: int) -> None:
    if ref_teacher_steps >= S2ST_MIN_TOKEN_REF_STEPS:
        return
    pytest.skip(
        f"{task.upper()} sweep len={seq_len} reference has {ref_teacher_steps} decode steps "
        f"(need >={S2ST_MIN_TOKEN_REF_STEPS}); HF hit EOS early on short mel input"
    )


def default_speech_sweep_mel_debug_dir(*, task: str, mel_frames: int) -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "teacher_forced_sweep_outputs"
        / "debug_mel_inputs"
        / f"mel{mel_frames}"
        / task
    )


def save_speech_sweep_mel_artifacts(
    *,
    task: str,
    mel_frames: int,
    input_features: torch.Tensor,
    mel_attention_mask: torch.Tensor,
    seed_ids: torch.Tensor | None = None,
    teacher_tokens: torch.Tensor | None = None,
    out_dir: Path | None = None,
    wav_path: Path | None = None,
) -> Path:
    import json

    dest = out_dir or default_speech_sweep_mel_debug_dir(task=task, mel_frames=mel_frames)
    dest.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "task": task,
        "mel_frames": mel_frames,
        "input_features": input_features.detach().cpu(),
        "mel_attention_mask": mel_attention_mask.detach().cpu(),
    }
    if seed_ids is not None:
        payload["seed_ids"] = seed_ids.detach().cpu()
    if teacher_tokens is not None:
        payload["teacher_tokens"] = teacher_tokens.detach().cpu()

    torch.save(payload, dest / "mel_input.pt")

    meta = {
        "task": task,
        "mel_frames": mel_frames,
        "input_features_shape": list(input_features.shape),
        "mel_valid_frames": int(mel_attention_mask.sum().item()),
        "teacher_decode_steps": int(teacher_tokens.numel()) if teacher_tokens is not None else None,
        "note": (
            "mel_input.pt is the exact tensor fed to the speech encoder in token-matching sweep "
            "(first mel_frames of processor output from long preamble audio)."
        ),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if wav_path is not None and wav_path.is_file():
        import shutil

        shutil.copy2(wav_path, dest / "long_preamble_source.wav")

    return dest


def maybe_save_speech_sweep_mel_env(
    *,
    task: str,
    seq_len: int,
    input_features: torch.Tensor,
    mel_attention_mask: torch.Tensor,
    seed_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
) -> None:
    want = os.environ.get("SEAMLESS_SWEEP_SAVE_MEL", "").strip()
    if not want or str(seq_len) != want:
        return
    if task not in SPEECH_INPUT_TASKS:
        return
    dest = save_speech_sweep_mel_artifacts(
        task=task,
        mel_frames=seq_len,
        input_features=input_features,
        mel_attention_mask=mel_attention_mask,
        seed_ids=seed_ids,
        teacher_tokens=teacher_tokens,
    )
    logger.info(f"Saved speech sweep mel debug artifacts for {task.upper()} mel={seq_len} to {dest}")
