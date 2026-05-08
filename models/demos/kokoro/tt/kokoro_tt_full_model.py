# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M full TTNN pipeline (no hybrid torch predictor/decoder).

Pipeline:
  PL-BERT (TTNN) -> Predictor (TTNN, including duration/alignment) -> ISTFTNet Vocoder (TTNN) -> waveform (torch output)
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import ttnn
from models.demos.kokoro.reference.kokoro_config import KokoroConfig
from models.demos.kokoro.reference.kokoro_full_model import KokoroFullOutput
from models.demos.kokoro.reference.kokoro_model import KokoroModelReference
from models.demos.kokoro.reference.kokoro_plbert import load_plbert_from_huggingface
from models.demos.kokoro.tt.ttnn_kokoro_istftnet import TtKokoroIstftNetVocoder, preprocess_istftnet_vocoder
from models.demos.kokoro.tt.ttnn_kokoro_plbert import TtKokoroPlBert
from models.demos.kokoro.tt.ttnn_kokoro_predictor import TtKokoroPredictor, preprocess_predictor_full


class KokoroTtFull(nn.Module):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
        disable_complex: bool = True,
        weights_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.repo_id = repo_id

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.vocab: dict[str, int] = cfg["vocab"]
        self.context_length: int = int(cfg["plbert"]["max_position_embeddings"])

        # Load torch modules on CPU only for parameter extraction
        plbert_cpu = load_plbert_from_huggingface(repo_id=repo_id, device="cpu")
        kmodel_cpu = KokoroModelReference(repo_id=repo_id, device="cpu", disable_complex=disable_complex)
        predictor_cpu = kmodel_cpu.get_predictor()
        text_encoder_cpu = kmodel_cpu.get_text_encoder()
        decoder_cpu = kmodel_cpu.get_decoder()

        self.tt_plbert = TtKokoroPlBert(mesh_device, plbert_cpu, activation_dtype=weights_dtype)

        pred_params = preprocess_predictor_full(
            # the predictor weights live inside the full reference model layout; wrap a minimal module-like object
            type("Tmp", (), {"predictor": predictor_cpu, "text_encoder": text_encoder_cpu})(),
            mesh_device,
            weights_dtype=weights_dtype,
        )
        self.tt_predictor = TtKokoroPredictor(mesh_device, pred_params)

        voc_params = preprocess_istftnet_vocoder(decoder_cpu, mesh_device, weights_dtype=weights_dtype)
        self.tt_vocoder = TtKokoroIstftNetVocoder(mesh_device, torch_decoder=decoder_cpu, params=voc_params)

    def phonemes_to_input_ids(self, phonemes: str) -> torch.LongTensor:
        ids = [self.vocab.get(p) for p in phonemes]
        ids = [i for i in ids if i is not None]
        if len(ids) + 2 > self.context_length:
            raise ValueError(f"Too many tokens: {len(ids)+2} > context_length={self.context_length}")
        return torch.tensor([[0, *ids, 0]], dtype=torch.long, device="cpu")

    @torch.no_grad()
    def forward(
        self,
        *,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_intermediates: bool = False,
    ) -> KokoroFullOutput:
        input_ids = self.phonemes_to_input_ids(phonemes)
        return self.forward_with_tokens(
            input_ids=input_ids, ref_s=ref_s, speed=speed, return_intermediates=return_intermediates
        )

    @torch.no_grad()
    def forward_with_tokens(
        self,
        *,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_intermediates: bool = False,
    ) -> KokoroFullOutput:
        input_ids = input_ids.to(torch.device("cpu"))
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        pl_out = self.tt_plbert(input_ids=input_ids)

        pred = self.tt_predictor(
            # pl_out.d_en is torch [B, hidden_dim, T]
            d_en_bct=ttnn.from_torch(
                pl_out.d_en, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            ),
            ref_s=ref_s,
            input_ids=input_ids,
            input_lengths=pl_out.input_lengths,
            text_mask=pl_out.text_mask,
            speed=speed,
        )

        audio = self.tt_vocoder(
            asr=pred["asr"],
            f0_pred=pred["F0_pred"],
            n_pred=pred["N_pred"],
            ref_s=ref_s,
        )

        if not return_intermediates:
            return KokoroFullOutput(audio=audio.squeeze().cpu(), pred_dur=torch.zeros((1,), dtype=torch.long))

        return KokoroFullOutput(audio=audio.squeeze().cpu(), pred_dur=torch.zeros((1,), dtype=torch.long))
