from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio

from models.demos.audio.cosy_voice.demo.common import (
    CosyVoiceCase,
    configure_reference_imports,
    resolve_model_dir,
    resolve_prompt_audio,
    resolve_reference_repo,
    save_audio,
)
from models.demos.audio.cosy_voice.tt.llm import CosyVoiceSemanticInputs, build_semantic_inputs_from_model_input

LANGUAGE_TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")
QUALITY_WHISPER_MODEL = "small"


def normalize_transcript_text(text: str, language: str | None) -> str:
    text = LANGUAGE_TAG_PATTERN.sub(" ", text)
    text = text.strip()
    if language in {"zh", "ja", "ko", "yue"}:
        # Character-level normalization is more stable than whitespace tokenization for CJK scripts.
        return "".join(ch for ch in text if ch.isalnum())
    return " ".join(part for part in re.sub(r"[^0-9a-zA-Z]+", " ", text.lower()).split() if part)


def transcript_tokens(text: str, language: str | None) -> list[str]:
    normalized = normalize_transcript_text(text, language)
    if language in {"zh", "ja", "ko", "yue"}:
        return list(normalized)
    return normalized.split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    for idx in range(rows):
        distance[idx][0] = idx
    for idx in range(cols):
        distance[0][idx] = idx
    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                distance[row][col] = distance[row - 1][col - 1]
            else:
                distance[row][col] = min(
                    distance[row - 1][col] + 1,
                    distance[row][col - 1] + 1,
                    distance[row - 1][col - 1] + 1,
                )
    return distance[-1][-1]


def word_error_rate_percent(reference_text: str, hypothesis_text: str, language: str | None) -> float:
    reference = transcript_tokens(reference_text, language)
    hypothesis = transcript_tokens(hypothesis_text, language)
    if not reference:
        return 0.0
    return 100.0 * edit_distance(reference, hypothesis) / float(len(reference))


def cosine_similarity_percent(reference_embedding: torch.Tensor, hypothesis_embedding: torch.Tensor) -> float:
    reference_embedding = reference_embedding.reshape(-1).float()
    hypothesis_embedding = hypothesis_embedding.reshape(-1).float()
    similarity = torch.nn.functional.cosine_similarity(reference_embedding, hypothesis_embedding, dim=0)
    return float(similarity.item() * 100.0)


@dataclass
class GeneratedAudioResult:
    case: CosyVoiceCase
    model_dir: str
    sample_rate: int
    waveform: torch.Tensor
    wall_seconds: float

    @property
    def audio_seconds(self) -> float:
        return float(self.waveform.shape[-1]) / float(self.sample_rate) if self.sample_rate > 0 else 0.0

    @property
    def rtf(self) -> float | None:
        return self.wall_seconds / self.audio_seconds if self.audio_seconds > 0 else None


@dataclass
class SemanticTokenResult:
    case: CosyVoiceCase
    model_dir: str
    tokens: torch.Tensor
    wall_seconds: float

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[-1]) if self.tokens.ndim > 0 else 0

    @property
    def tokens_per_second(self) -> float | None:
        return self.token_count / self.wall_seconds if self.wall_seconds > 0 else None


class CosyVoiceReferenceSession:
    def __init__(self, reference_repo: str | None = None, model_root: str | None = None, text_frontend: bool = True):
        self.reference_repo = resolve_reference_repo(reference_repo)
        self.model_root = model_root
        self.text_frontend = text_frontend
        self._models: dict[str, Any] = {}
        self._whisper_model = None

    def _load_model(self, mode: str):
        configure_reference_imports(self.reference_repo)
        from cosyvoice.cli.cosyvoice import CosyVoice  # noqa: PLC0415

        model_dir = resolve_model_dir(mode, self.model_root)
        model = CosyVoice(str(model_dir))
        return model, model_dir

    def get_model(self, mode: str):
        if mode not in self._models:
            self._models[mode] = self._load_model(mode)
        return self._models[mode]

    def _get_whisper_model(self):
        if self._whisper_model is None:
            import whisper  # noqa: PLC0415

            self._whisper_model = whisper.load_model(QUALITY_WHISPER_MODEL, device="cpu")
        return self._whisper_model

    @staticmethod
    def _empty_text_tokens() -> torch.Tensor:
        return torch.zeros((1, 0), dtype=torch.int32)

    @staticmethod
    def _empty_token_lengths() -> torch.Tensor:
        return torch.zeros((1,), dtype=torch.int32)

    @staticmethod
    def _empty_prompt_features() -> torch.Tensor:
        return torch.zeros((1, 0, 80), dtype=torch.float32)

    @staticmethod
    def _empty_prompt_feature_lengths() -> torch.Tensor:
        return torch.zeros((1,), dtype=torch.int32)

    @staticmethod
    def _empty_embedding(model: Any) -> torch.Tensor:
        spk_embed_dim = int(model.model.llm.spk_embed_affine_layer.in_features)
        return torch.zeros((0, spk_embed_dim), dtype=torch.float32)

    def generate_semantic_tokens(
        self,
        case: CosyVoiceCase,
        model_input: dict[str, Any],
    ) -> SemanticTokenResult:
        model, model_dir = self.get_model(case.mode)
        llm = model.model.llm

        prompt_text = model_input.get("prompt_text", self._empty_text_tokens())
        prompt_text_len = model_input.get("prompt_text_len", self._empty_token_lengths())
        prompt_speech_token = model_input.get("llm_prompt_speech_token", self._empty_text_tokens())
        prompt_speech_token_len = model_input.get("llm_prompt_speech_token_len", self._empty_token_lengths())
        llm_embedding = model_input.get("llm_embedding", self._empty_embedding(model))

        start = time.perf_counter()
        tokens = list(
            llm.inference(
                text=model_input["text"],
                text_len=model_input["text_len"],
                prompt_text=prompt_text,
                prompt_text_len=prompt_text_len,
                prompt_speech_token=prompt_speech_token,
                prompt_speech_token_len=prompt_speech_token_len,
                embedding=llm_embedding,
            )
        )
        wall_seconds = time.perf_counter() - start
        token_tensor = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0) if tokens else self._empty_text_tokens()
        return SemanticTokenResult(
            case=case,
            model_dir=str(Path(model_dir).resolve()),
            tokens=token_tensor,
            wall_seconds=wall_seconds,
        )

    def generate_semantic_greedy_tokens(
        self,
        case: CosyVoiceCase,
        model_input: dict[str, Any],
    ) -> SemanticTokenResult:
        model, model_dir = self.get_model(case.mode)
        llm = model.model.llm
        semantic_inputs = build_semantic_inputs_from_model_input(
            llm_module=llm,
            model_input=model_input,
        )

        lm_input = semantic_inputs.lm_input
        att_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device)
        cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device)
        offset = 0
        tokens: list[int] = []
        start = time.perf_counter()
        for step_index in range(semantic_inputs.max_decode_length):
            att_mask = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(
                torch.bool
            )
            y_pred, att_cache, cnn_cache = llm.llm.forward_chunk(
                lm_input,
                offset=offset,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=att_mask,
            )
            logits = llm.llm_decoder(y_pred[:, -1]).squeeze(0)
            if step_index < semantic_inputs.min_decode_length:
                # Match the public semantic parity gate to the model's EOS suppression rule.
                logits = logits.clone()
                logits[llm.eos_token] = -float("inf")
            next_token = int(torch.argmax(logits).item())
            if next_token == llm.eos_token:
                break
            tokens.append(next_token)
            offset += lm_input.size(1)
            lm_input = llm.speech_embedding.weight[next_token].reshape(1, 1, -1)
        wall_seconds = time.perf_counter() - start
        token_tensor = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0) if tokens else self._empty_text_tokens()
        return SemanticTokenResult(
            case=case,
            model_dir=str(Path(model_dir).resolve()),
            tokens=token_tensor,
            wall_seconds=wall_seconds,
        )

    def prepare_semantic_inputs(
        self,
        case: CosyVoiceCase,
        model_input: dict[str, Any],
    ) -> CosyVoiceSemanticInputs:
        model, _ = self.get_model(case.mode)
        return build_semantic_inputs_from_model_input(
            llm_module=model.model.llm,
            model_input=model_input,
        )

    def transcribe_audio(self, audio_path: str | Path, language: str | None) -> str:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        whisper_language = {"yue": "zh"}.get(language or "", language)
        result = self._get_whisper_model().transcribe(
            waveform.squeeze(0).cpu().numpy(),
            language=whisper_language,
            task="transcribe",
            fp16=False,
            verbose=False,
        )
        return str(result["text"])

    def _extract_generated_audio_embedding(self, case: CosyVoiceCase, audio_path: str | Path) -> torch.Tensor:
        model, _ = self.get_model(case.mode)
        return model.frontend._extract_spk_embedding(str(audio_path)).detach().cpu().reshape(-1)

    def _target_speaker_embedding(self, case: CosyVoiceCase) -> torch.Tensor:
        model, _ = self.get_model(case.mode)
        if case.mode in {"sft", "instruct"}:
            # SFT and instruct use a fixed speaker embedding from the checked-in speaker table.
            target = model.frontend.spk2info[case.speaker_id]["embedding"]
            if not torch.is_tensor(target):
                target = torch.tensor(target)
            return target.detach().cpu().reshape(-1)

        # Zero-shot and cross-lingual quality must track the prompt speaker identity.
        prompt_audio = resolve_prompt_audio(self.reference_repo, case.prompt_audio)
        return model.frontend._extract_spk_embedding(prompt_audio).detach().cpu().reshape(-1)

    def evaluate_quality(self, case: CosyVoiceCase, audio_path: str | Path) -> dict[str, Any]:
        reference_text = normalize_transcript_text(case.text, case.language)
        transcription = self.transcribe_audio(audio_path, case.language)
        speaker_similarity_pct = cosine_similarity_percent(
            self._target_speaker_embedding(case),
            self._extract_generated_audio_embedding(case, audio_path),
        )
        return {
            "reference_text": reference_text,
            "transcribed_text": transcription,
            "wer_pct": word_error_rate_percent(reference_text, transcription, case.language),
            "speaker_similarity_pct": speaker_similarity_pct,
            "quality_asr_model": QUALITY_WHISPER_MODEL,
        }

    def synthesize_from_semantic_tokens(
        self,
        case: CosyVoiceCase,
        model_input: dict[str, Any],
        semantic_tokens: torch.Tensor,
        output_path: str | None = None,
    ) -> GeneratedAudioResult:
        model, model_dir = self.get_model(case.mode)
        source_tokens = semantic_tokens if semantic_tokens.ndim == 2 else semantic_tokens.reshape(1, -1)

        flow_prompt_speech_token = model_input.get("flow_prompt_speech_token", self._empty_text_tokens())
        prompt_speech_feat = model_input.get("prompt_speech_feat", self._empty_prompt_features())
        flow_embedding = model_input["flow_embedding"]

        start = time.perf_counter()
        chunks = list(
            model.model.tts(
                flow_embedding=flow_embedding,
                flow_prompt_speech_token=flow_prompt_speech_token,
                prompt_speech_feat=prompt_speech_feat,
                source_speech_token=source_tokens,
                stream=False,
            )
        )
        wall_seconds = time.perf_counter() - start
        if not chunks:
            raise RuntimeError(f"CosyVoice produced no audio chunks for case {case.name}")
        waveform = torch.cat([chunk["tts_speech"].cpu() for chunk in chunks], dim=1)
        result = GeneratedAudioResult(
            case=case,
            model_dir=str(Path(model_dir).resolve()),
            sample_rate=model.sample_rate,
            waveform=waveform,
            wall_seconds=wall_seconds,
        )
        if output_path is not None:
            save_audio(output_path, waveform, result.sample_rate)
        return result

    def generate(self, case: CosyVoiceCase, output_path: str | None = None) -> GeneratedAudioResult:
        model, model_dir = self.get_model(case.mode)
        prompt_audio = resolve_prompt_audio(self.reference_repo, case.prompt_audio)
        start = time.perf_counter()
        if case.mode == "sft":
            chunks = list(
                model.inference_sft(
                    case.text,
                    case.speaker_id,
                    stream=False,
                    text_frontend=self.text_frontend,
                )
            )
        elif case.mode == "zero_shot":
            chunks = list(
                model.inference_zero_shot(
                    case.text,
                    case.prompt_text,
                    prompt_audio,
                    stream=False,
                    text_frontend=self.text_frontend,
                )
            )
        elif case.mode == "cross_lingual":
            chunks = list(
                model.inference_cross_lingual(
                    case.text,
                    prompt_audio,
                    stream=False,
                    text_frontend=self.text_frontend,
                )
            )
        elif case.mode == "instruct":
            chunks = list(
                model.inference_instruct(
                    case.text,
                    case.speaker_id,
                    case.instruction,
                    stream=False,
                    text_frontend=self.text_frontend,
                )
            )
        else:  # pragma: no cover - protected by CosyVoiceCase validation
            raise ValueError(f"Unsupported mode: {case.mode}")
        wall_seconds = time.perf_counter() - start
        if not chunks:
            raise RuntimeError(f"CosyVoice produced no audio chunks for case {case.name}")
        waveform = torch.cat([chunk["tts_speech"].cpu() for chunk in chunks], dim=1)
        result = GeneratedAudioResult(
            case=case,
            model_dir=str(Path(model_dir).resolve()),
            sample_rate=model.sample_rate,
            waveform=waveform,
            wall_seconds=wall_seconds,
        )
        if output_path is not None:
            save_audio(output_path, waveform, result.sample_rate)
        return result
