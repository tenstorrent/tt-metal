# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TtnnCosyVoice — E2E pipeline wiring LLM (N300) + flow (host) + vocoder (host).

Non-streaming Stage-1 orchestration:
    text → frontend (host) → speech_tokens (LLM on N300) → mel (flow, host) → waveform (vocoder, host)

Mirrors the CosyVoice2 public API: inference_sft, inference_zero_shot,
inference_cross_lingual, inference_instruct2.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import torch

_COSYVOICE_SRC = str(Path(__file__).resolve().parents[1] / "model_data" / "CosyVoice_src")
_MATCHA = str(Path(_COSYVOICE_SRC) / "third_party" / "Matcha-TTS")
if _COSYVOICE_SRC not in sys.path:
    sys.path.insert(0, _COSYVOICE_SRC)
if _MATCHA not in sys.path:
    sys.path.append(_MATCHA)

_CKPT_DIR = Path(__file__).resolve().parents[1] / "model_data" / "cosyvoice2-0.5B"
_FLOW_PT = _CKPT_DIR / "flow.pt"
_HIFT_PT = _CKPT_DIR / "hift.pt"

from models.demos.cosyvoice.tt.flow.cfm import CausalConditionalCFM
from models.demos.cosyvoice.tt.flow.estimator_ttnn import UNetEstimatorTtnn
from models.demos.cosyvoice.tt.flow.flow_matching import FlowEncoderModel
from models.demos.cosyvoice.tt.flow.weights import load_flow_weights
from models.demos.cosyvoice.tt.hifigan.generator import HiFTVocoder
from models.demos.cosyvoice.tt.llm.model import CosyVoiceLLM
from models.demos.cosyvoice.tt.model_config import SEED


def _stub_pyworld():
    if "pyworld" not in sys.modules:
        stub = types.ModuleType("pyworld")
        _noop = lambda *a, **k: None
        for n in (
            "wave_to_world",
            "world_to_wave",
            "pythonworld",
            "dio",
            "stft",
            "harvest",
            "cheaptrick",
            "d4c",
            "star",
            "vocoder",
        ):
            setattr(stub, n, _noop)
        sys.modules["pyworld"] = stub


def _patch_load_wav():
    import cosyvoice.utils.file_utils as fu
    import soundfile
    import torchaudio

    def load_wav(wav, target_sr, min_sr=16000):
        data, sample_rate = soundfile.read(str(wav), dtype="float32")
        t = torch.from_numpy(data)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        else:
            t = t.t().mean(dim=0, keepdim=True)
        speech = t
        if sample_rate != target_sr:
            assert sample_rate >= min_sr
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    fu.load_wav = load_wav


class TtnnCosyVoice:
    """Stage-1 E2E CosyVoice2 pipeline on N300.

    LLM runs on TT device; flow encoder + CFM + vocoder run on host (torch).
    Supports 2-chip pipeline parallelism: LLM on chip 0, CFM estimator on chip 1.
    """

    def __init__(self, mesh_device, model_dir: Optional[str] = None, mesh_device_flow=None):
        self.mesh_device = mesh_device
        self.mesh_device_flow = mesh_device_flow if mesh_device_flow is not None else mesh_device
        self.model_dir = Path(model_dir) if model_dir else _CKPT_DIR
        self.sample_rate = 24000

        _stub_pyworld()
        _patch_load_wav()

        self._init_frontend()
        self._init_llm()
        self._init_flow()
        self._init_vocoder()

    def _init_frontend(self):
        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        from hyperpyyaml import load_hyperpyyaml

        yaml_path = self.model_dir / "cosyvoice2.yaml"
        with open(yaml_path, "r") as f:
            configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": str(self.model_dir / "CosyVoice-BlankEN")})

        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            str(self.model_dir / "campplus.onnx"),
            str(self.model_dir / "speech_tokenizer_v2.onnx"),
            str(self.model_dir / "spk2info.pt"),
            configs["allowed_special"],
        )
        self._configs = configs

    def _init_llm(self):
        self.llm = CosyVoiceLLM(self.mesh_device)

    def _init_flow(self):
        flow_weights = load_flow_weights(_FLOW_PT)
        self.flow_encoder = FlowEncoderModel(flow_weights)
        self.flow_encoder.eval()

        estimator = UNetEstimatorTtnn(flow_weights["decoder"], self.mesh_device_flow)
        self.cfm = CausalConditionalCFM(estimator, n_timesteps=5)

    def _init_vocoder(self):
        self.vocoder = HiFTVocoder.from_checkpoint(_HIFT_PT)

    def add_zero_shot_spk(self, prompt_text: str, prompt_wav: str, spk_id: str) -> bool:
        assert spk_id != "", "do not use empty spk_id"
        model_input = self.frontend.frontend_zero_shot("", prompt_text, prompt_wav, self.sample_rate, "")
        del model_input["text"]
        del model_input["text_len"]
        self.frontend.spk2info[spk_id] = model_input
        self.frontend.spk2info[spk_id]["embedding"] = model_input["llm_embedding"]
        return True

    @torch.inference_mode()
    def _run_flow_and_vocoder(
        self,
        speech_tokens: List[int],
        flow_prompt_speech_token: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        flow_embedding: torch.Tensor,
    ) -> torch.Tensor:
        token = torch.tensor([speech_tokens], dtype=torch.int32)
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32)
        prompt_token = flow_prompt_speech_token
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32)
        prompt_feat = prompt_speech_feat
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32)
        embedding = flow_embedding

        mu, spks, conds = self.flow_encoder.forward(
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
        )

        mel_len1 = prompt_feat.shape[1]
        t_mel = mu.shape[2]
        mask = torch.ones(1, 1, t_mel, dtype=mu.dtype)

        mel = self.cfm.inference(mu, mask, spks, conds)
        mel = mel[:, :, mel_len1:]

        waveform, _ = self.vocoder.inference(mel)
        return waveform

    @torch.inference_mode()
    def inference_zero_shot(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav: str,
        seed: int = SEED,
    ) -> torch.Tensor:
        texts = self.frontend.text_normalize(tts_text, split=True)
        prompt_text_norm = self.frontend.text_normalize(prompt_text, split=False)

        all_waveforms = []
        for text_chunk in texts:
            model_input = self.frontend.frontend_zero_shot(
                text_chunk, prompt_text_norm, prompt_wav, self.sample_rate, ""
            )

            text_token_ids = torch.cat(
                [
                    model_input["prompt_text"].squeeze(0),
                    model_input["text"].squeeze(0),
                ]
            ).long()
            prompt_speech_token_ids = model_input["llm_prompt_speech_token"].squeeze(0).long()

            tts_text_len = model_input["text"].shape[1]
            min_len = int(tts_text_len * 2)
            max_len = int(tts_text_len * 20)

            speech_tokens = self.llm.generate(
                text_token_ids,
                prompt_speech_token_ids,
                min_len=min_len,
                max_len=max_len,
                seed=seed,
            )

            if not speech_tokens:
                continue

            waveform = self._run_flow_and_vocoder(
                speech_tokens,
                model_input["flow_prompt_speech_token"],
                model_input["prompt_speech_feat"],
                model_input["flow_embedding"],
            )
            all_waveforms.append(waveform)

        if not all_waveforms:
            return torch.zeros(1, 0)
        return torch.cat(all_waveforms, dim=1)

    @torch.inference_mode()
    def inference_cross_lingual(
        self,
        tts_text: str,
        prompt_wav: str,
        seed: int = SEED,
    ) -> torch.Tensor:
        texts = self.frontend.text_normalize(tts_text, split=True)

        all_waveforms = []
        for text_chunk in texts:
            model_input = self.frontend.frontend_cross_lingual(text_chunk, prompt_wav, self.sample_rate, "")

            text_token_ids = model_input["text"].squeeze(0).long()

            tts_text_len = model_input["text"].shape[1]
            min_len = int(tts_text_len * 2)
            max_len = int(tts_text_len * 20)

            speech_tokens = self.llm.generate(
                text_token_ids,
                None,
                min_len=min_len,
                max_len=max_len,
                seed=seed,
            )

            if not speech_tokens:
                continue

            waveform = self._run_flow_and_vocoder(
                speech_tokens,
                model_input["flow_prompt_speech_token"],
                model_input["prompt_speech_feat"],
                model_input["flow_embedding"],
            )
            all_waveforms.append(waveform)

        if not all_waveforms:
            return torch.zeros(1, 0)
        return torch.cat(all_waveforms, dim=1)

    @torch.inference_mode()
    def inference_instruct2(
        self,
        tts_text: str,
        instruct_text: str,
        prompt_wav: str,
        seed: int = SEED,
    ) -> torch.Tensor:
        texts = self.frontend.text_normalize(tts_text, split=True)

        all_waveforms = []
        for text_chunk in texts:
            model_input = self.frontend.frontend_instruct2(text_chunk, instruct_text, prompt_wav, self.sample_rate, "")

            text_token_ids = torch.cat(
                [
                    model_input["prompt_text"].squeeze(0),
                    model_input["text"].squeeze(0),
                ]
            ).long()

            tts_text_len = model_input["text"].shape[1]
            min_len = int(tts_text_len * 2)
            max_len = int(tts_text_len * 20)

            speech_tokens = self.llm.generate(
                text_token_ids,
                None,
                min_len=min_len,
                max_len=max_len,
                seed=seed,
            )

            if not speech_tokens:
                continue

            waveform = self._run_flow_and_vocoder(
                speech_tokens,
                model_input["flow_prompt_speech_token"],
                model_input["prompt_speech_feat"],
                model_input["flow_embedding"],
            )
            all_waveforms.append(waveform)

        if not all_waveforms:
            return torch.zeros(1, 0)
        return torch.cat(all_waveforms, dim=1)

    @torch.inference_mode()
    def inference_sft(
        self,
        tts_text: str,
        spk_id: str,
        seed: int = SEED,
    ) -> torch.Tensor:
        texts = self.frontend.text_normalize(tts_text, split=True)

        all_waveforms = []
        for text_chunk in texts:
            model_input = self.frontend.frontend_sft(text_chunk, spk_id)

            text_token_ids = model_input["text"].squeeze(0).long()

            tts_text_len = model_input["text"].shape[1]
            min_len = int(tts_text_len * 2)
            max_len = int(tts_text_len * 20)

            speech_tokens = self.llm.generate(
                text_token_ids,
                None,
                min_len=min_len,
                max_len=max_len,
                seed=seed,
            )

            if not speech_tokens:
                continue

            flow_embedding = model_input["flow_embedding"]
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
            prompt_speech_feat = torch.zeros(1, 0, 80)

            waveform = self._run_flow_and_vocoder(
                speech_tokens,
                flow_prompt_speech_token,
                prompt_speech_feat,
                flow_embedding,
            )
            all_waveforms.append(waveform)

        if not all_waveforms:
            return torch.zeros(1, 0)
        return torch.cat(all_waveforms, dim=1)

    TOKEN_HOP_LEN = 25
    TOKEN_MAX_HOP_LEN = 100
    STREAM_SCALE_FACTOR = 2
    PRE_LOOKAHEAD_LEN = 3
    TOKEN_MEL_RATIO = 2
    MEL_CACHE_LEN = 8
    SOURCE_CACHE_LEN = MEL_CACHE_LEN * 480

    def _token2wav_streaming(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        token_offset: int,
        prev_mel_end: int,
        hift_cache: Optional[dict],
        stream: bool = False,
        finalize: bool = False,
    ):
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], dtype=torch.int32)

        mu, spks, conds = self.flow_encoder.forward(
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            streaming=stream,
            finalize=finalize,
        )

        mel_len1 = prompt_feat.shape[1]
        t_mel = mu.shape[2]

        gen_mel = mu[:, :, mel_len1:]
        curr_mel_end = gen_mel.shape[2]

        delta_mel = gen_mel[:, :, prev_mel_end:curr_mel_end]
        delta_T = delta_mel.shape[2]

        if delta_T > 0:
            delta_mask = torch.ones(1, 1, delta_T, dtype=mu.dtype)
            delta_cond = torch.zeros(1, 80, delta_T, dtype=mu.dtype)
            tts_mel = self.cfm.inference(delta_mel, delta_mask, spks, delta_cond, streaming=True)
        else:
            tts_mel = torch.zeros(1, 80, 0, dtype=mu.dtype)

        if hift_cache is not None:
            hift_cache_mel, hift_cache_source = hift_cache["mel"], hift_cache["source"]
            tts_mel = torch.cat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)

        speech_window = np.hamming(2 * self.SOURCE_CACHE_LEN)

        if not finalize:
            tts_speech, tts_source = self.vocoder.inference(tts_mel, cache_source=hift_cache_source)
            if hift_cache is not None:
                tts_speech = self._fade_in_out(tts_speech, hift_cache["speech"], speech_window)
            hift_cache = {
                "mel": tts_mel[:, :, -self.MEL_CACHE_LEN :],
                "source": tts_source[:, :, -self.SOURCE_CACHE_LEN :],
                "speech": tts_speech[:, -self.SOURCE_CACHE_LEN :],
            }
            tts_speech = tts_speech[:, : -self.SOURCE_CACHE_LEN]
        else:
            tts_speech, tts_source = self.vocoder.inference(tts_mel, cache_source=hift_cache_source)
            if hift_cache is not None:
                tts_speech = self._fade_in_out(tts_speech, hift_cache["speech"], speech_window)
            hift_cache = None

        return tts_speech, hift_cache, curr_mel_end

    @staticmethod
    def _fade_in_out(fade_in: torch.Tensor, fade_out: torch.Tensor, window: np.ndarray) -> torch.Tensor:
        overlap_len = int(window.shape[0] / 2)
        w = torch.from_numpy(window).to(fade_in.dtype)
        fade_in = fade_in.clone()
        fade_in[..., :overlap_len] = (
            fade_in[..., :overlap_len] * w[:overlap_len] + fade_out[..., -overlap_len:] * w[overlap_len:]
        )
        return fade_in

    @torch.inference_mode()
    def inference_zero_shot_streaming(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav: str,
        seed: int = SEED,
    ) -> Generator[torch.Tensor, None, None]:
        texts = self.frontend.text_normalize(tts_text, split=True)
        prompt_text_norm = self.frontend.text_normalize(prompt_text, split=False)

        for text_chunk in texts:
            model_input = self.frontend.frontend_zero_shot(
                text_chunk, prompt_text_norm, prompt_wav, self.sample_rate, ""
            )

            text_token_ids = torch.cat([model_input["prompt_text"].squeeze(0), model_input["text"].squeeze(0)]).long()
            prompt_speech_token_ids = model_input["llm_prompt_speech_token"].squeeze(0).long()

            tts_text_len = model_input["text"].shape[1]
            min_len = int(tts_text_len * 2)
            max_len = int(tts_text_len * 20)

            flow_prompt_speech_token = model_input["flow_prompt_speech_token"]
            prompt_speech_feat = model_input["prompt_speech_feat"]
            flow_embedding = model_input["flow_embedding"]

            token_list: List[int] = []
            llm_done = threading.Event()
            llm_error: List[Exception] = []

            def _llm_worker():
                try:
                    for tok in self.llm.generate_streaming(
                        text_token_ids,
                        prompt_speech_token_ids,
                        min_len=min_len,
                        max_len=max_len,
                        seed=seed,
                    ):
                        token_list.append(tok)
                except Exception as e:
                    llm_error.append(e)
                finally:
                    llm_done.set()

            llm_thread = threading.Thread(target=_llm_worker)
            llm_thread.start()

            token_offset = 0
            token_hop_len = self.TOKEN_HOP_LEN
            prompt_token_pad = int(
                np.ceil(flow_prompt_speech_token.shape[1] / self.TOKEN_HOP_LEN) * self.TOKEN_HOP_LEN
                - flow_prompt_speech_token.shape[1]
            )
            hift_cache = None
            prev_mel_end = 0

            while True:
                time.sleep(0.005)
                this_hop = token_hop_len + prompt_token_pad if token_offset == 0 else token_hop_len
                available = len(token_list) - token_offset

                if available >= this_hop + self.PRE_LOOKAHEAD_LEN:
                    this_token = torch.tensor(
                        [token_list[: token_offset + this_hop + self.PRE_LOOKAHEAD_LEN]], dtype=torch.int32
                    )
                    tts_speech, hift_cache, prev_mel_end = self._token2wav_streaming(
                        token=this_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        token_offset=token_offset,
                        prev_mel_end=prev_mel_end,
                        hift_cache=hift_cache,
                        stream=True,
                        finalize=False,
                    )
                    token_offset += this_hop
                    token_hop_len = min(self.TOKEN_MAX_HOP_LEN, token_hop_len * self.STREAM_SCALE_FACTOR)
                    if tts_speech.shape[1] > 0:
                        yield tts_speech

                if llm_done.is_set() and len(token_list) - token_offset < this_hop + self.PRE_LOOKAHEAD_LEN:
                    break

            llm_thread.join()
            if llm_error:
                raise llm_error[0]

            if len(token_list) > token_offset:
                this_token = torch.tensor([token_list], dtype=torch.int32)
                tts_speech, hift_cache, _ = self._token2wav_streaming(
                    token=this_token,
                    prompt_token=flow_prompt_speech_token,
                    prompt_feat=prompt_speech_feat,
                    embedding=flow_embedding,
                    token_offset=token_offset,
                    prev_mel_end=prev_mel_end,
                    hift_cache=hift_cache,
                    stream=False,
                    finalize=True,
                )
                if tts_speech.shape[1] > 0:
                    yield tts_speech
