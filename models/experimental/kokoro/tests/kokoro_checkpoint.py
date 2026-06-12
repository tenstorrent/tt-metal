# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Kokoro-82M reference weights and capture activations for unit PCC tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel

KOKORO_STYLE_DIM = 128
KOKORO_HIDDEN_DIM = 512

_VOICE = "af_heart"
_LANG_CODE = "a"

# Config E (recommended fallbacks): reference TorchSTFT + torch phase/SineGen on the
# generator harmonic path; device F0 + f0 upsample stay on-device. Matches
# ``test_tt_kmodel_stft_and_phase_fallback_pcc``.
STFT_PHASE_FALLBACK_KWARGS = dict(
    use_torch_stft_fallback=True,
    use_torch_phase_fallback=True,
)

_CKPT_CANDIDATES = (
    Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
    Path("/home/ubuntu/samyuktha/tt-metal/models/experimental/kokoro/checkpoints/kokoro-v1_0.pth"),
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
)


def find_checkpoint() -> Path | None:
    for p in _CKPT_CANDIDATES:
        if p.is_file():
            return p
        if p.is_dir():
            for child in p.rglob("kokoro-v1_0.pth"):
                return child
    return None


def load_kmodel(*, device: torch.device | str | None = None) -> KModel:
    """``KModel`` with Kokoro-82M weights (``reference/model.py`` + checkpoint)."""
    ckpt = find_checkpoint()
    if ckpt is None:
        raise FileNotFoundError("Kokoro-82M checkpoint not found locally.")
    model = KModel(repo_id="hexgrad/Kokoro-82M", model=str(ckpt), disable_complex=True)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def capture_predictor_lstm_input_nlc(
    ref: KModel,
    *,
    seq_len: int = 48,
    seed: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """``d`` tensor fed to ``ProsodyPredictor.lstm`` (NLC ``[B, T, 640]``) from a short synthetic pass."""
    input_ids, input_lengths, text_mask, ref_s, lengths = _synthetic_token_batch(ref, seq_len=seq_len, seed=seed)

    with torch.no_grad():
        bert_dur = ref.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = ref.bert_encoder(bert_dur).transpose(-1, -2)
        d = ref.predictor.text_encoder(d_en, ref_s[:, 128:], input_lengths, text_mask)
    return d.float().cpu(), lengths


def device_compute_config(device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )


def torch_bilstm_nlc(
    lstm: nn.LSTM,
    x_nlc: torch.Tensor,
    *,
    sequence_lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    """Reference BiLSTM matching ``tt_bilstm_nlc`` (full length, zero past valid timesteps)."""
    with torch.no_grad():
        if sequence_lengths is None:
            y, _ = lstm(x_nlc)
            return y
        seq_len = x_nlc.shape[1]
        lengths_cpu = torch.tensor(list(sequence_lengths), dtype=torch.long)
        packed = pack_padded_sequence(x_nlc, lengths_cpu, batch_first=True, enforce_sorted=False)
        lstm.flatten_parameters()
        y_packed, _ = lstm(packed)
        y, _ = pad_packed_sequence(y_packed, batch_first=True, total_length=seq_len)
        for bi, le in enumerate(sequence_lengths):
            le = max(0, min(int(le), seq_len))
            if le < seq_len:
                y[bi, le:, :] = 0.0
        return y


def _synthetic_token_batch(
    ref: KModel,
    *,
    seq_len: int,
    seed: int,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor, list[int]]:
    torch.manual_seed(seed)
    dev = next(ref.parameters()).device
    vocab_ids = [v for v in ref.vocab.values() if v is not None][: max(0, seq_len - 2)]
    ids = [0, *vocab_ids[: seq_len - 2], 0]
    t = len(ids)
    b = 1
    input_ids = torch.LongTensor([ids]).to(dev)
    input_lengths = torch.tensor([t], dtype=torch.long, device=dev)
    text_mask = torch.arange(t, device=dev).unsqueeze(0).expand(b, -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    ref_s = torch.randn(b, 256, device=dev)
    return input_ids, input_lengths, text_mask, ref_s, [t]


def capture_duration_encoder_lstm0_input_nlc(
    ref: KModel,
    *,
    seq_len: int = 48,
    seed: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """Input to ``predictor.text_encoder.lstms[0]`` (NLC ``[B, T, 640]``)."""
    input_ids, input_lengths, text_mask, ref_s, lengths = _synthetic_token_batch(ref, seq_len=seq_len, seed=seed)
    with torch.no_grad():
        bert_dur = ref.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = ref.bert_encoder(bert_dur).transpose(-1, -2)
        style = ref_s[:, 128:]
        x = d_en.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        x.masked_fill_(text_mask.unsqueeze(-1).transpose(0, 1), 0.0)
        # ``DurationEncoder`` LSTM path uses NLC ``[B, T, 640]`` (after ``transpose(0, 1)`` only).
        x = x.transpose(0, 1)
    return x.float().cpu(), lengths


def capture_text_encoder_lstm_input_nlc(
    ref: KModel,
    *,
    seq_len: int = 48,
    seed: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """CNN output fed to ``text_encoder.lstm`` (NLC ``[B, T, 512]``)."""
    input_ids, input_lengths, text_mask, _, lengths = _synthetic_token_batch(ref, seq_len=seq_len, seed=seed)
    m = text_mask.unsqueeze(1)
    with torch.no_grad():
        x = ref.text_encoder.embedding(input_ids)
        x = x.transpose(1, 2)
        x.masked_fill_(m, 0.0)
        for c in ref.text_encoder.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
    return x.float().cpu(), lengths


def assert_bilstm_config(
    lstm: nn.LSTM,
    *,
    name: str,
    input_size: int,
    hidden_size: int,
) -> None:
    assert lstm.input_size == input_size, f"{name}: input_size {lstm.input_size} != {input_size}"
    assert lstm.hidden_size == hidden_size, f"{name}: hidden_size {lstm.hidden_size} != {hidden_size}"
    assert lstm.batch_first and lstm.bidirectional and lstm.num_layers == 1


def get_module_attr(model: nn.Module, path: str) -> nn.Module:
    """Resolve ``'predictor.F0[0].norm1'`` style paths on ``KModel``."""
    obj: object = model
    for part in path.split("."):
        if "[" in part:
            name, idx = part[:-1].split("[")
            obj = getattr(obj, name)[int(idx)]
        else:
            obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Full-pipeline / stage helpers (shared by the kmodel + decoder PCC tests)
# ---------------------------------------------------------------------------


@contextmanager
def _zero_noise():
    """Force ``torch.rand``/``randn_like`` to zero so TT and reference share noise (=0)."""
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    torch.randn_like = lambda t, **kwargs: torch.zeros_like(t, **kwargs)
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _phonemize(text: str) -> tuple[str, torch.Tensor]:
    """Phonemes + matching ``ref_s`` voice vector for ``text`` (needs the ``kokoro`` package)."""
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=_LANG_CODE, model=False)
    results = list(pipe(text, voice=_VOICE))
    phonemes = results[0].phonemes
    pack = pipe.load_voice(_VOICE)
    ref_s = pack[len(phonemes) - 1]
    if not isinstance(ref_s, torch.Tensor):
        ref_s = torch.tensor(ref_s)
    ref_s = ref_s.float().cpu()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    return phonemes, ref_s


def _tokenize(
    vocab: dict, phonemes: str, context_length: int
) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor, list[int]]:
    """Phoneme string -> ``input_ids`` (BOS/EOS padded), ``text_mask``, lengths."""
    input_ids_list = list(filter(lambda i: i is not None, map(lambda p: vocab.get(p), phonemes)))
    assert len(input_ids_list) + 2 <= context_length
    input_ids = torch.LongTensor([[0, *input_ids_list, 0]])
    B, T = input_ids.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)
    text_mask = torch.arange(T).unsqueeze(0).expand(B, -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    return input_ids, text_mask, input_lengths, input_lengths.tolist()


def _pcc_row(name: str, ref: torch.Tensor, tt: torch.Tensor) -> tuple[str, float, str]:
    """``(name, pcc, note)`` row comparing a reference vs TT tensor (length-tolerant)."""
    ref_f = ref.detach().float().reshape(-1)
    tt_f = tt.detach().float().reshape(-1)
    note = ""
    if ref_f.numel() != tt_f.numel():
        n = min(ref_f.numel(), tt_f.numel())
        note = f"len ref={ref_f.numel()} tt={tt_f.numel()} (using first {n})"
        ref_f = ref_f[:n]
        tt_f = tt_f[:n]
    if ref_f.numel() == 0:
        return name, float("nan"), "empty"
    _, pcc = comp_pcc(ref_f.unsqueeze(0), tt_f.unsqueeze(0), pcc=0.0)
    if ref.shape != tt.shape and not note:
        note = f"shape ref={tuple(ref.shape)} tt={tuple(tt.shape)}"
    return name, float(pcc), note


@dataclass
class RefStages:
    d_en_bct: torch.Tensor
    d_nlc: torch.Tensor
    pred_dur: torch.LongTensor
    en_bct: torch.Tensor
    F0: torch.Tensor
    N: torch.Tensor
    asr_bct: torch.Tensor
    audio: torch.Tensor


def _run_ref_stages(ref: KModel, input_ids: torch.LongTensor, ref_s: torch.Tensor, speed: float = 1.0) -> RefStages:
    """Reference KModel forward broken into the per-stage tensors the kmodel PCC tests compare against."""
    input_ids = input_ids.to(ref.device)
    ref_s = ref_s.to(ref.device)
    input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(ref.device)

    with torch.no_grad(), _zero_noise():
        bert_dur = ref.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = ref.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = ref_s[:, 128:]
        d = ref.predictor.text_encoder(d_en, s_pred, input_lengths, text_mask)
        x, _ = ref.predictor.lstm(d)
        duration = ref.predictor.duration_proj(x)
        pred_dur = torch.round(torch.sigmoid(duration).sum(dim=-1) / speed).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=ref.device), pred_dur)
        aln = torch.zeros((input_ids.shape[1], indices.shape[0]), device=ref.device)
        aln[indices, torch.arange(indices.shape[0], device=ref.device)] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        F0_pred, N_pred = ref.predictor.F0Ntrain(en, s_pred)
        t_en = ref.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln
        s_style = ref_s[:, :128]
        audio = ref.decoder(asr, F0_pred, N_pred, s_style).squeeze()

    return RefStages(
        d_en_bct=d_en.cpu(),
        d_nlc=d.cpu(),
        pred_dur=pred_dur.cpu(),
        en_bct=en.cpu(),
        F0=F0_pred.cpu(),
        N=N_pred.cpu(),
        asr_bct=asr.cpu(),
        audio=audio.cpu().float(),
    )


def _ref_prosody(ref: KModel, phonemes: str, ref_s: torch.Tensor, speed: float = 1.0):
    """Reference ASR / F0 / N / style / predicted-duration for decoder-stack input."""
    vocab = ref.vocab
    input_ids_list = list(filter(lambda i: i is not None, map(lambda p: vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids_list, 0]]).to(ref.device)
    B, T = input_ids.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=ref.device)
    text_mask = torch.arange(T, device=ref.device).unsqueeze(0).expand(B, -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

    with torch.no_grad(), _zero_noise():
        bert_dur = ref.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = ref.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = ref_s[:, 128:]
        s_style = ref_s[:, :128]
        d = ref.predictor.text_encoder(d_en, s_pred, input_lengths, text_mask)
        x_lstm, _ = ref.predictor.lstm(d)
        duration = ref.predictor.duration_proj(x_lstm)
        dur_sum = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(dur_sum).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=ref.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=ref.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0, N = ref.predictor.F0Ntrain(en, s_pred)
        t_en = ref.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
    return asr.cpu(), F0.cpu(), N.cpu(), s_style.cpu(), pred_dur.cpu()
