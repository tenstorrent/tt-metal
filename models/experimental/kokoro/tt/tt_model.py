"""
TTNN port of KModel (reference/model.py).

Wires together:
  TTCustomAlbert (bert)         — torch fallback for full Albert transformer
  bert_encoder (Linear 768→512) — ttnn.linear
  TTProsodyPredictor            — TTNN linear/layernorm, LSTM torch fallback
  TTTextEncoder                 — TTNN embedding, torch CNN+LSTM
  TTDecoder                     — TTNN-hybrid decoder (see tt_istftnet.py)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


from .tt_utils import load_tt_linear, tt_linear
from .tt_modules import TTCustomAlbert, TTProsodyPredictor, TTTextEncoder
from .tt_istftnet import TTDecoder


class TTKModel(nn.Module):
    """
    TTNN port of KModel.

    Use from_kmodel() to construct from a loaded reference KModel.
    The TT device is stored and forwarded to all sub-modules.
    """

    def __init__(
        self,
        bert: TTCustomAlbert,
        bert_encoder_w,
        bert_encoder_b,
        bert_encoder_out: int,
        predictor: TTProsodyPredictor,
        text_encoder: TTTextEncoder,
        decoder: TTDecoder,
        vocab: dict,
        context_length: int,
        device,
    ):
        super().__init__()
        self.device = device
        self.vocab = vocab
        self.context_length = context_length

        self.bert = bert
        self._be_w = bert_encoder_w
        self._be_b = bert_encoder_b
        self._be_out = bert_encoder_out
        self.predictor = predictor
        self.text_encoder = text_encoder
        self.decoder = decoder

    @classmethod
    def from_kmodel(cls, kmodel, device):
        """Build TTKModel from a loaded reference KModel."""
        bert = TTCustomAlbert(kmodel.bert, device)

        be_w, be_b, be_out = load_tt_linear(kmodel.bert_encoder, device)

        predictor = TTProsodyPredictor(kmodel.predictor, device)
        text_encoder = TTTextEncoder(kmodel.text_encoder, device)
        decoder = TTDecoder(kmodel.decoder, device)

        return cls(
            bert=bert,
            bert_encoder_w=be_w,
            bert_encoder_b=be_b,
            bert_encoder_out=be_out,
            predictor=predictor,
            text_encoder=text_encoder,
            decoder=decoder,
            vocab=kmodel.vocab,
            context_length=kmodel.context_length,
            device=device,
        )

    def _bert_encode(self, bert_output: torch.Tensor) -> torch.Tensor:
        """bert_encoder Linear(768 → 512) via ttnn.linear."""
        return tt_linear(bert_output, self._be_w, self._be_b, self._be_out, self.device)

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
    ):
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )
        text_mask = (
            torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        # Albert runs on its own device (CPU/CUDA); result brought to torch
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self._bert_encode(bert_dur).transpose(-1, -2)  # (B, 512, T)

        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        # Prosody lstm (torch fallback)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        target_device = input_ids.device
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=target_device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=target_device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
    ):
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        assert len(input_ids) + 2 <= self.context_length
        # Move input_ids to same device as bert
        bert_device = next(self.bert.albert.parameters()).device
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(bert_device)
        ref_s = ref_s.to(bert_device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None

        @dataclass
        class Output:
            audio: torch.FloatTensor
            pred_dur: Optional[torch.LongTensor] = None

        return Output(audio=audio, pred_dur=pred_dur) if return_output else audio
