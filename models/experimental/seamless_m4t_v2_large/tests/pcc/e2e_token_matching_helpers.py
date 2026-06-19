# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E token-matching helpers (``models/tt_transformers/demo/simple_text_demo.py`` pattern).

Teacher-forced greedy decode: HF reference tokens are fed as decoder inputs at each step while
TT ``lm_head`` predictions are compared to offline HF top-1 / top-5 from a ``.refpt`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import TextDecoderPccInputs
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_logit_pcc_helpers import (
    _align_tt_encoder_to_case,
    tt_encode_speech,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_seamless_m4t_v2_model import _make_tt_model
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import from_torch_uint32_rm, mesh_default_device
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import _ttnn_ids_from_list
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import init_text_decoder_kv_cache

# Same long English source as ``test_seamless_m4t_v2_model.py`` (yields ~100+ Hindi decode steps).
T2TT_REF_SOURCE_TEXT = """Maya lived in a small coastal town where every morning began with the sound of fishing boats leaving the harbor. She worked at her grandfather's old bookstore, a narrow shop filled with dusty shelves, handwritten notes, and the smell of paper that had aged for decades. Most customers came looking for schoolbooks or travel guides, but Maya loved recommending forgotten stories hidden in the back corners of the store.

One rainy evening, while organizing a stack of returned books, she discovered a small blue journal tucked between two novels. The cover had no title, only a silver compass symbol that shimmered faintly under the light. Curious, she opened it and found detailed sketches of places around the town along with cryptic messages about a hidden lighthouse path that only appeared during storms.

At first, Maya thought someone was playing a prank. But the next night, as heavy clouds gathered over the sea, she noticed something unusual from the bookstore window. A narrow trail of lantern lights stretched along the cliffs where no road existed before. Holding the journal tightly, she followed the glowing path through the rain until she reached an abandoned lighthouse overlooking the crashing waves."""

T2TT_REF_TGT_LANG = "hin"
T2TT_TOP1_THRESHOLD = 0.95
T2TT_TOP5_THRESHOLD = 0.99
SPEECH_TOP1_THRESHOLD = 0.87  # S2TT ~88% top-1 with live speech encoder on BH 1×4; ASR ~90%+
SPEECH_TOP5_THRESHOLD = 0.95

# S2ST reference must have enough teacher-forced steps (preamble WAV, not synthetic mel).
S2ST_MIN_TOKEN_REF_STEPS = 8

_REF_DIR = Path(__file__).resolve().parent.parent / "reference_outputs"
_REFPT_NAMES = {
    "t2tt": "seamless_m4t_v2_t2tt_eng_hin.refpt",
    "t2st": "seamless_m4t_v2_t2st.refpt",
    "s2tt": "seamless_m4t_v2_s2tt.refpt",
    "s2st": "seamless_m4t_v2_s2st.refpt",
    "asr": "seamless_m4t_v2_asr.refpt",
}

# E2E logit/token tests validate text-decoder intermediates for all five tasks.
# T2ST/S2ST share the same encoder+decoder stack as T2TT/S2TT (before T2U/vocoder).
ALL_E2E_TASKS = ("t2tt", "t2st", "s2tt", "s2st", "asr")
TASK_TGT_LANG = {
    "t2tt": "hin",
    "t2st": "hin",
    "s2tt": "eng",
    "s2st": "spa",
    "asr": "eng",
}
TEXT_INPUT_TASKS = frozenset({"t2tt", "t2st"})
SPEECH_INPUT_TASKS = frozenset({"s2tt", "s2st", "asr"})


def default_refpt_path(task: str) -> Path:
    return _REF_DIR / _REFPT_NAMES[task]


def default_t2tt_refpt_path() -> Path:
    return default_refpt_path("t2tt")


def weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def refpt_or_skip(task: str = "t2tt", path: Path | None = None) -> Path:
    ref_path = path or default_refpt_path(task)
    if not ref_path.is_file():
        gen = "models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py"
        pytest.skip(f"Missing token-matching reference {ref_path}. Run: python {gen} --task {task}")
    return ref_path


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
    return top1, top5, n_eval


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
    top1, top5, n_eval = eval_token_matching_loop(
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
    assert top1 >= top1_threshold, f"top1 {top1_pct:.2f}% < {top1_threshold * 100:.0f}%"
    assert top5 >= top5_threshold, f"top5 {top5_pct:.2f}% < {top5_threshold * 100:.0f}%"


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
        tt_model = _make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)

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
        tt_model = _make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        enc_tt, enc_mask_tt = tt_encode_speech(
            mesh_device,
            hf_model.speech_encoder,
            cfg,
            ref.input_features,
            ref.mel_attention_mask,
        )
        enc_tt, enc_mask_tt = _align_tt_encoder_to_case(
            mesh_device,
            enc_tt,
            enc_mask_tt,
            ref.decoder_case,
            pad_id=int(cfg.pad_token_id),
            hidden_size=int(cfg.hidden_size),
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
