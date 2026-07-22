"""TtnnCosyVoice — E2E pipeline wiring LLM (N300) + flow (host) + vocoder (host).

Non-streaming Stage-1 orchestration:
    text → frontend (host) → speech_tokens (LLM on N300) → mel (flow, host) → waveform (vocoder, host)

Mirrors the CosyVoice2 public API: inference_sft, inference_zero_shot,
inference_cross_lingual, inference_instruct2.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import List, Optional

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
from models.demos.cosyvoice.tt.flow.flow_matching import FlowEncoderModel
from models.demos.cosyvoice.tt.flow.unet_estimator import UNetEstimator
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
    """

    def __init__(self, mesh_device, model_dir: Optional[str] = None):
        pass

        self.mesh_device = mesh_device
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

        estimator = UNetEstimator(flow_weights["decoder"])
        self.cfm = CausalConditionalCFM(estimator)

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
