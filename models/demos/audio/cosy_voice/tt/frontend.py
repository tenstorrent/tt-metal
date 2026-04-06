from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.demos.audio.cosy_voice.demo.common import CosyVoiceCase, resolve_prompt_audio
from models.demos.audio.cosy_voice.tt.reference import CosyVoiceReferenceSession


@dataclass
class PreparedFrontendInput:
    mode: str
    payload: dict[str, Any]


class CosyVoiceFrontendAdapter:
    def __init__(self, session: CosyVoiceReferenceSession):
        self.session = session

    def prepare(self, case: CosyVoiceCase) -> PreparedFrontendInput:
        model, _ = self.session.get_model(case.mode)
        frontend = model.frontend
        prompt_audio = resolve_prompt_audio(self.session.reference_repo, case.prompt_audio)
        if case.mode == "sft":
            payload = frontend.frontend_sft(case.text, case.speaker_id)
        elif case.mode == "zero_shot":
            payload = frontend.frontend_zero_shot(
                case.text,
                case.prompt_text,
                prompt_audio,
                model.sample_rate,
                "",
            )
        elif case.mode == "cross_lingual":
            payload = frontend.frontend_cross_lingual(case.text, prompt_audio, model.sample_rate, "")
        elif case.mode == "instruct":
            payload = frontend.frontend_instruct(case.text, case.speaker_id, case.instruction)
        else:  # pragma: no cover - protected by CosyVoiceCase validation
            raise ValueError(f"Unsupported mode: {case.mode}")
        return PreparedFrontendInput(mode=case.mode, payload=payload)
