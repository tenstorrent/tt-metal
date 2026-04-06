from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from models.demos.audio.cosy_voice.demo.common import CosyVoiceCase, save_audio
from models.demos.audio.cosy_voice.tt.flow import (
    CosyVoiceFlowInputs,
    CosyVoiceTTFlowBridge,
    CosyVoiceTorchFlowDecoder,
    CosyVoiceTTFlowEncoder,
    CosyVoiceTTFlowLengthRegulator,
    build_flow_inputs_from_model_input,
)
from models.demos.audio.cosy_voice.tt.frontend import CosyVoiceFrontendAdapter, PreparedFrontendInput
from models.demos.audio.cosy_voice.tt.llm import CosyVoiceSemanticInputs, CosyVoiceTTSemanticGenerator
from models.demos.audio.cosy_voice.tt.reference import (
    CosyVoiceReferenceSession,
    GeneratedAudioResult,
    SemanticTokenResult,
)


class CosyVoicePipeline:
    def __init__(self, reference_repo: str | None = None, model_root: str | None = None, text_frontend: bool = True):
        self.reference = CosyVoiceReferenceSession(
            reference_repo=reference_repo,
            model_root=model_root,
            text_frontend=text_frontend,
        )
        self.frontend = CosyVoiceFrontendAdapter(self.reference)
        self._ttnn = None
        self._device = None
        self._previous_fallback_setting: bool | None = None
        self._semantic_generators: dict[str, CosyVoiceTTSemanticGenerator] = {}
        self._flow_bridges: dict[str, CosyVoiceTTFlowBridge] = {}
        self._flow_encoders: dict[str, CosyVoiceTTFlowEncoder] = {}
        self._flow_length_regulators: dict[str, CosyVoiceTTFlowLengthRegulator] = {}
        self._flow_decoders: dict[str, CosyVoiceTorchFlowDecoder] = {}

    def _get_device(self):
        if self._device is None:
            import ttnn  # noqa: PLC0415

            self._ttnn = ttnn
            self._previous_fallback_setting = ttnn.CONFIG.throw_exception_on_fallback
            ttnn.CONFIG.throw_exception_on_fallback = True
            self._device = ttnn.open_device(device_id=0)
        return self._device

    def _get_semantic_generator(self, mode: str) -> CosyVoiceTTSemanticGenerator:
        if mode not in self._semantic_generators:
            model, _ = self.reference.get_model(mode)
            self._semantic_generators[mode] = CosyVoiceTTSemanticGenerator(
                model.model.llm,
                self._get_device(),
            )
        return self._semantic_generators[mode]

    def _get_flow_bridge(self, mode: str) -> CosyVoiceTTFlowBridge:
        if mode not in self._flow_bridges:
            model, _ = self.reference.get_model(mode)
            self._flow_bridges[mode] = CosyVoiceTTFlowBridge(
                model.model.flow,
                self._get_device(),
            )
        return self._flow_bridges[mode]

    def _get_flow_encoder(self, mode: str) -> CosyVoiceTTFlowEncoder:
        if mode not in self._flow_encoders:
            model, _ = self.reference.get_model(mode)
            self._flow_encoders[mode] = CosyVoiceTTFlowEncoder(
                model.model.flow,
                self._get_device(),
            )
        return self._flow_encoders[mode]

    def _get_flow_length_regulator(self, mode: str) -> CosyVoiceTTFlowLengthRegulator:
        if mode not in self._flow_length_regulators:
            model, _ = self.reference.get_model(mode)
            self._flow_length_regulators[mode] = CosyVoiceTTFlowLengthRegulator(
                model.model.flow,
                self._get_device(),
            )
        return self._flow_length_regulators[mode]

    def _get_flow_decoder(self, mode: str) -> CosyVoiceTorchFlowDecoder:
        if mode not in self._flow_decoders:
            model, _ = self.reference.get_model(mode)
            self._flow_decoders[mode] = CosyVoiceTorchFlowDecoder(model.model.flow)
        return self._flow_decoders[mode]

    def close(self) -> None:
        if self._device is not None and self._ttnn is not None:
            self._ttnn.close_device(self._device)
            self._device = None
            self._semantic_generators.clear()
            self._flow_bridges.clear()
            self._flow_encoders.clear()
            self._flow_length_regulators.clear()
            self._flow_decoders.clear()
            if self._previous_fallback_setting is not None:
                self._ttnn.CONFIG.throw_exception_on_fallback = self._previous_fallback_setting
                self._previous_fallback_setting = None

    def __del__(self):  # pragma: no cover - defensive cleanup for public CLI scripts
        try:
            self.close()
        except Exception:
            pass

    def prepare_case(self, case: CosyVoiceCase) -> PreparedFrontendInput:
        return self.frontend.prepare(case)

    def prepare_semantic_inputs(
        self,
        case: CosyVoiceCase,
        prepared: PreparedFrontendInput | None = None,
    ) -> CosyVoiceSemanticInputs:
        prepared = prepared or self.prepare_case(case)
        return self.reference.prepare_semantic_inputs(case, prepared.payload)

    def prepare_flow_inputs(
        self,
        case: CosyVoiceCase,
        semantic_tokens: SemanticTokenResult | torch.Tensor,
        prepared: PreparedFrontendInput | None = None,
    ) -> CosyVoiceFlowInputs:
        prepared = prepared or self.prepare_case(case)
        model, _ = self.reference.get_model(case.mode)
        token_tensor = semantic_tokens.tokens if isinstance(semantic_tokens, SemanticTokenResult) else semantic_tokens
        return build_flow_inputs_from_model_input(
            flow_module=model.model.flow,
            model_input=prepared.payload,
            source_speech_token=token_tensor,
        )

    def generate_semantic_tokens(
        self,
        case: CosyVoiceCase,
        prepared: PreparedFrontendInput | None = None,
    ) -> SemanticTokenResult:
        prepared = prepared or self.prepare_case(case)
        semantic_inputs = self.reference.prepare_semantic_inputs(case, prepared.payload)
        tokens, wall_seconds = self._get_semantic_generator(case.mode).generate(semantic_inputs)
        _, model_dir = self.reference.get_model(case.mode)
        return SemanticTokenResult(case=case, model_dir=str(model_dir), tokens=tokens, wall_seconds=wall_seconds)

    def evaluate_semantic_token_accuracy(
        self,
        case: CosyVoiceCase,
        prepared: PreparedFrontendInput | None = None,
    ) -> float:
        prepared = prepared or self.prepare_case(case)
        semantic_inputs = self.reference.prepare_semantic_inputs(case, prepared.payload)
        reference_tokens = self.reference.generate_semantic_greedy_tokens(case, prepared.payload)
        return self._get_semantic_generator(case.mode).teacher_forced_accuracy(semantic_inputs, reference_tokens.tokens)

    def evaluate_quality(self, case: CosyVoiceCase, output_path: str | Path) -> dict[str, Any]:
        return self.reference.evaluate_quality(case, output_path)

    def synthesize_semantic_tokens(
        self,
        case: CosyVoiceCase,
        semantic_tokens: SemanticTokenResult | object,
        prepared: PreparedFrontendInput | None = None,
        output_path: str | None = None,
    ) -> GeneratedAudioResult:
        prepared = prepared or self.prepare_case(case)
        token_tensor = semantic_tokens.tokens if isinstance(semantic_tokens, SemanticTokenResult) else semantic_tokens
        flow_inputs = self.prepare_flow_inputs(case, token_tensor, prepared=prepared)
        model, model_dir = self.reference.get_model(case.mode)
        bridge = self._get_flow_bridge(case.mode)
        encoder = self._get_flow_encoder(case.mode)
        length_regulator = self._get_flow_length_regulator(case.mode)
        decoder = self._get_flow_decoder(case.mode)

        start = time.perf_counter()
        speaker_projection = bridge.project_speaker_embedding(flow_inputs.flow_embedding)
        token_embedding = bridge.embed_tokens(flow_inputs.full_token, flow_inputs.full_token_len)
        encoded_hidden = encoder.encode(token_embedding, flow_inputs.full_token_len)
        projected_hidden = bridge.project_encoder_output(encoded_hidden)

        prompt_token_len = int(flow_inputs.prompt_speech_token.shape[1])
        acoustic_hidden, _ = length_regulator.inference(
            projected_hidden[:, :prompt_token_len],
            projected_hidden[:, prompt_token_len:],
            flow_inputs.prompt_mel_length,
            flow_inputs.decode_mel_length,
            model.model.flow.input_frame_rate,
        )
        total_mel_length = flow_inputs.prompt_mel_length + flow_inputs.decode_mel_length
        cond = flow_inputs.condition.transpose(1, 2).contiguous()
        mask = torch.ones((1, 1, total_mel_length), dtype=torch.bool, device=acoustic_hidden.device)
        feat, _ = decoder.inference(
            mu=acoustic_hidden.transpose(1, 2).contiguous(),
            mask=mask,
            spks=speaker_projection.to(dtype=acoustic_hidden.dtype, device=acoustic_hidden.device),
            cond=cond,
            n_timesteps=10,
            prompt_len=flow_inputs.prompt_mel_length,
            cache=torch.zeros((1, 80, 0, 2), dtype=acoustic_hidden.dtype, device=acoustic_hidden.device),
        )
        feat = feat[:, :, flow_inputs.prompt_mel_length :]
        waveform, _ = model.model.hift.inference(
            speech_feat=feat,
            cache_source=torch.zeros((1, 1, 0), dtype=feat.dtype, device=feat.device),
        )
        wall_seconds = time.perf_counter() - start
        result = GeneratedAudioResult(
            case=case,
            model_dir=str(Path(model_dir).resolve()),
            sample_rate=model.sample_rate,
            waveform=waveform.cpu(),
            wall_seconds=wall_seconds,
        )
        if output_path is not None:
            save_audio(output_path, result.waveform, result.sample_rate)
        return result

    def generate_case(self, case: CosyVoiceCase, output_path: str | None = None) -> GeneratedAudioResult:
        prepared = self.prepare_case(case)
        semantic_tokens = self.generate_semantic_tokens(case, prepared=prepared)
        return self.synthesize_semantic_tokens(case, semantic_tokens, prepared=prepared, output_path=output_path)

    def generate_mode(self, mode: str, text: str, output_path: str | None = None, **kwargs) -> GeneratedAudioResult:
        case = CosyVoiceCase(name=f"adhoc_{mode}", mode=mode, text=text, **kwargs)
        if output_path is None:
            output_path = str(Path("/tmp") / f"{case.name}.wav")
        return self.generate_case(case, output_path=output_path)
