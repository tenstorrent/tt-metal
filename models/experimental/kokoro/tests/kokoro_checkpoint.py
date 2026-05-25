# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load Kokoro-82M reference weights and capture activations for unit PCC tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import ttnn

from models.experimental.kokoro.reference.model import KModel

KOKORO_STYLE_DIM = 128
KOKORO_HIDDEN_DIM = 512

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
