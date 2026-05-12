# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Full Kokoro-82M on TTNN: PL-BERT → predictor → :class:`KokoroIstftNetTt` vocoder.

Neural path stays on device (bf16 activations from PL-BERT/predictor, float32 vocoder
stack). ``pred_dur`` and the one-hot alignment are built on host (numpy + upload) from
TTNN duration logits; final waveform is read back once via ``ttnn.to_torch`` for the
``KokoroFullOutput`` contract.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import ttnn

from ..reference.kokoro_config import KokoroConfig
from ..reference.kokoro_full_model import KokoroFullOutput
from ..reference.kokoro_istftnet import load_decoder_from_huggingface
from ..reference.kokoro_plbert import load_plbert_from_huggingface
from ..reference.kokoro_predictor import load_predictor_from_huggingface
from .ttnn_kokoro_decoder import KokoroIstftNetTt, preprocess_kokoro_decoder_tt_parameters
from .ttnn_kokoro_plbert import TtKokoroPlBert
from .ttnn_kokoro_predictor import TtKokoroPredictor, preprocess_predictor_full


class KokoroFullTtnn(nn.Module):
    """
    End-to-end Kokoro on Tenstorrent: device PL-BERT, device predictor, device ISTFTNet.

    Vocoder uses the experimental :class:`KokoroIstftNetTt` (no host STFT fallback).

    Set ``use_torch_sinegen=True`` to run the reference PyTorch ``SineGen`` on CPU inside
    the TTNN generator (compare waveform PCC vs default device ``KokoroTtnnSineGen``).
    """

    def __init__(
        self,
        device: Any,
        *,
        repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
        disable_complex: bool = True,
        weights_dtype=ttnn.bfloat16,
        vocoder_cache_max: int = 16,
        use_torch_sinegen: bool = False,
    ):
        super().__init__()
        self.device = device
        self.repo_id = repo_id
        self.disable_complex = disable_complex
        self.weights_dtype = weights_dtype
        self.use_torch_sinegen = bool(use_torch_sinegen)
        self._vocoder_cache_max = int(vocoder_cache_max)
        self._vocoder_cache: OrderedDict[tuple[int, bool], KokoroIstftNetTt] = OrderedDict()

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.vocab: dict[str, int] = cfg["vocab"]
        self.context_length: int = int(cfg["plbert"]["max_position_embeddings"])

        plbert_cpu = load_plbert_from_huggingface(repo_id=repo_id, device="cpu")
        kokoro_pred_cpu = load_predictor_from_huggingface(repo_id=repo_id, device="cpu")
        istft_cpu = load_decoder_from_huggingface(repo_id=repo_id, device="cpu", disable_complex=disable_complex)
        self._decoder_cpu = istft_cpu.decoder

        self.tt_plbert = TtKokoroPlBert(device, plbert_cpu, activation_dtype=weights_dtype)
        pred_params = preprocess_predictor_full(kokoro_pred_cpu, device, weights_dtype=weights_dtype)
        self.tt_predictor = TtKokoroPredictor(device, pred_params)

    def _vocoder_for_tf(self, tf: int) -> KokoroIstftNetTt:
        key = (tf, self.use_torch_sinegen)
        if key in self._vocoder_cache:
            self._vocoder_cache.move_to_end(key)
            return self._vocoder_cache[key]
        params = preprocess_kokoro_decoder_tt_parameters(
            self._decoder_cpu,
            self.device,
            f0_coarse_time=tf,
            disable_complex=self.disable_complex,
            use_torch_sinegen=self.use_torch_sinegen,
        )
        voc = KokoroIstftNetTt(self.device, params)
        self._vocoder_cache[key] = voc
        self._vocoder_cache.move_to_end(key)
        while len(self._vocoder_cache) > self._vocoder_cache_max:
            self._vocoder_cache.popitem(last=False)
        return voc

    def phonemes_to_input_ids(self, phonemes: str) -> torch.LongTensor:
        ids = [self.vocab.get(p) for p in phonemes]
        ids = [i for i in ids if i is not None]
        if len(ids) + 2 > self.context_length:
            raise ValueError(f"Too many tokens: {len(ids)+2} > context_length={self.context_length}")
        return torch.tensor([[0, *ids, 0]], dtype=torch.long, device="cpu")

    @staticmethod
    def _bf16_to_f32_l1(x: ttnn.Tensor) -> ttnn.Tensor:
        l1 = ttnn.L1_MEMORY_CONFIG
        x = ttnn.to_memory_config(x, l1)
        return ttnn.typecast(x, ttnn.float32, memory_config=l1)

    @torch.no_grad()
    def forward(
        self,
        *,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_intermediates: bool = False,
        deterministic: bool = False,
    ) -> KokoroFullOutput:
        input_ids = self.phonemes_to_input_ids(phonemes)
        return self.forward_with_tokens(
            input_ids=input_ids,
            ref_s=ref_s,
            speed=speed,
            return_intermediates=return_intermediates,
            deterministic=deterministic,
        )

    @torch.no_grad()
    def forward_with_tokens(
        self,
        *,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_intermediates: bool = False,
        deterministic: bool = False,
    ) -> KokoroFullOutput:
        input_ids = input_ids.to(torch.device("cpu"))
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        d_en_tt, input_lengths, text_mask = self.tt_plbert.forward_d_en_ttnn(input_ids)
        try:
            pred = self.tt_predictor(
                d_en_bct=d_en_tt,
                ref_s=ref_s,
                input_ids=input_ids,
                input_lengths=input_lengths,
                text_mask=text_mask,
                speed=speed,
            )
        finally:
            ttnn.deallocate(d_en_tt)

        asr_bf = pred["asr"]
        f0_bf = pred["F0_pred"]
        n_bf = pred["N_pred"]
        pred_dur = pred["pred_dur"]

        tf = int(f0_bf.shape[2])
        voc = self._vocoder_for_tf(tf)

        asr_f32 = self._bf16_to_f32_l1(asr_bf)
        f0_f32 = self._bf16_to_f32_l1(f0_bf)
        n_f32 = self._bf16_to_f32_l1(n_bf)
        f0_curve = ttnn.permute(f0_f32, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        n_curve = ttnn.permute(n_f32, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

        ref_s_tt = ttnn.from_torch(
            ref_s,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        y_tt = voc(
            asr=asr_f32,
            f0_pred=f0_curve,
            n_pred=n_curve,
            ref_s=ref_s_tt,
            deterministic=deterministic,
        )
        audio = ttnn.to_torch(y_tt).to(torch.float32).squeeze().cpu()

        if return_intermediates:
            out = KokoroFullOutput(
                audio=audio,
                pred_dur=pred_dur.detach().cpu(),
                pred_aln_trg=ttnn.to_torch(pred["pred_aln_trg"]).cpu(),
                F0_pred=ttnn.to_torch(f0_bf).cpu(),
                N_pred=ttnn.to_torch(n_bf).cpu(),
                asr=ttnn.to_torch(asr_bf).cpu(),
            )
        else:
            out = KokoroFullOutput(audio=audio, pred_dur=pred_dur.detach().cpu())

        ttnn.deallocate(y_tt)
        ttnn.deallocate(asr_f32)
        ttnn.deallocate(f0_f32)
        ttnn.deallocate(n_f32)
        ttnn.deallocate(f0_curve)
        ttnn.deallocate(n_curve)
        ttnn.deallocate(ref_s_tt)
        return out
