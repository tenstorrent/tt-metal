# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""What a synthesis call is conditioned on (`PromptContext`) and what it draws at random (`RandomSources`).

The prompt side of CosyVoice2 comes from upstream's frontend -- an ONNX speech tokenizer, an ONNX CAM++ speaker
encoder and a mel filterbank -- which runs once, in the reference venv, in `scripts/prepare_inputs.py`. That
script writes one flat `.npz` per case; `PromptContext.from_npz` reads it here without importing onnxruntime,
whisper or the upstream `cosyvoice` package.

**Modes are prompt construction, not networks.** Every mode runs the same three stages with the same weights;
they differ in which parts of the LLM's prefix are filled (upstream's `frontend_*` in cosyvoice/cli/frontend.py):

| mode | LLM prefix: prompt text / prompt speech | flow prompt (tokens + mel) | speaker embedding |
|---|---|---|---|
| `zero_shot` | the prompt transcript / yes | yes | from the prompt audio |
| `cross_lingual` | none / none (the target text carries a `<|en|>`-style tag) | yes | from the prompt audio |
| `instruct2` | the instruction / none | yes | from the prompt audio |

CosyVoice2-0.5B ships no `spk2info.pt`, so there is no `sft` mode. Only `zero_shot` is wired end to end so far.
CosyVoice2's LLM takes no speaker embedding (`Qwen2LM.inference` accepts one and ignores it), so there is one
`embedding`, the flow's.

**Randomness is injected, never drawn inside a forward pass.** CosyVoice2 draws in two places at inference:
- the LLM's sampling, RAS on the host (`tt/llm/sampling.py`), seeded by `RandomSources.llm_seed`;
- SineGen2's per-call noise in the vocoder's NSF source, `[1, T_audio, harmonic_num + 1]`.

The CFM's initial noise is not a draw: upstream's `CausalConditionalCFM` slices a fixed `rand_noise` buffer, which
the port carries. SineGen2's `rand_ini` phase offset is drawn upstream too, but never reaches the output through
the downsample (tests/pcc/test_sine_gen2.py), so it is not modelled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

MODES = ("zero_shot", "cross_lingual", "instruct2")

# upstream frontend: the prompt mel runs at twice the speech-token rate (24 kHz / 480 hop = 50 Hz vs 25 Hz), and
# `frontend_zero_shot` trims both so that feat == 2 x tokens exactly
TOKEN_MEL_RATIO = 2


def describe_mode(mode: str) -> dict:
    """Which prompt fields `mode` fills (the module docstring's table), for the demo and the mode tests."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    return {
        "zero_shot": {"llm_prompt_text": True, "llm_prompt_speech": True, "flow_prompt": True},
        "cross_lingual": {"llm_prompt_text": False, "llm_prompt_speech": False, "flow_prompt": True},
        "instruct2": {"llm_prompt_text": True, "llm_prompt_speech": False, "flow_prompt": True},
    }[mode]


@dataclass
class RandomSources:
    """The draws of one synthesis call. Leave a field None to draw fresh; set it to reproduce a run.

    `sine_noise` is a captured `[1, T_audio, harmonic_num + 1]` array; a fresh draw is a standard normal of that
    shape (SineGen2 scales it by its voiced/unvoiced amplitude itself). `llm_seed` seeds torch's global RNG
    before generation, which is what host-side RAS draws from. The torch version sets that stream, so a seed
    reproduces tokens only against the same torch (see scripts/run_reference.py).

    `noise_seed` draws the vocoder's noise from a generator of its own instead (one per call, drawn from in order),
    so the tokens stay those of `llm_seed` whatever the noise: a noise draw over fixed tokens (scripts/noise_draws.py).
    Streaming uses it too, in place of its default `llm_seed + 1` (`CosyVoice2TTNN.synthesize_stream`).
    """

    sine_noise: torch.Tensor | None = None
    llm_seed: int | None = None
    noise_seed: int | None = None
    _noise_gen: torch.Generator | None = field(default=None, repr=False, compare=False)

    def noise_generator(self) -> torch.Generator | None:
        """The generator `noise_seed` draws from, made on first use; None without a `noise_seed`."""
        if self.noise_seed is not None and self._noise_gen is None:
            self._noise_gen = torch.Generator().manual_seed(self.noise_seed)
        return self._noise_gen

    def sine_noise_for(self, audio_len: int, harmonics: int) -> torch.Tensor:
        if self.sine_noise is None:
            return torch.randn(1, audio_len, harmonics, generator=self.noise_generator())
        if tuple(self.sine_noise.shape) != (1, audio_len, harmonics):
            raise ValueError(f"captured sine_noise is {tuple(self.sine_noise.shape)}; need {(1, audio_len, harmonics)}")
        return self.sine_noise


@dataclass
class PromptContext:
    """One case from `scripts/prepare_inputs.py`: the prompt, as host tensors (the pipeline uploads them).

    `meta` is for reporting and parity checks only, never for the model: the corpus entry (`case`), upstream's
    normalized segments of the case's text and their token ids, the normalized prompt transcript, and the
    versions the file was made with.
    """

    mode: str
    lang: str
    prompt_text_ids: torch.Tensor | None  # [1, P] int32; zero_shot: the transcript, instruct2: the instruction
    llm_prompt_speech_tokens: torch.Tensor | None  # [1, S] int32; zero_shot only
    flow_prompt_speech_tokens: torch.Tensor  # [1, S] int32; every mode
    prompt_feat: torch.Tensor  # [1, 2S, 80] float32, the 24 kHz prompt mel
    embedding: torch.Tensor  # [1, 192] float32, CAM++ x-vector of the prompt audio (the flow's)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        expect = describe_mode(self.mode)
        if (self.prompt_text_ids is not None) != expect["llm_prompt_text"]:
            raise ValueError(f"{self.mode}: prompt_text_ids must be {'set' if expect['llm_prompt_text'] else 'None'}")
        if (self.llm_prompt_speech_tokens is not None) != expect["llm_prompt_speech"]:
            raise ValueError(
                f"{self.mode}: llm_prompt_speech_tokens must be {'set' if expect['llm_prompt_speech'] else 'None'}"
            )
        s = self.flow_prompt_speech_tokens.shape[1]
        if tuple(self.prompt_feat.shape) != (1, TOKEN_MEL_RATIO * s, 80):
            raise ValueError(f"prompt_feat {tuple(self.prompt_feat.shape)} is not [1, 2 x {s} tokens, 80]")
        if tuple(self.embedding.shape) != (1, 192):
            raise ValueError(f"embedding {tuple(self.embedding.shape)} is not [1, 192]")

    @property
    def n_prompt_tokens(self) -> int:
        return int(self.flow_prompt_speech_tokens.shape[1])

    @property
    def prompt_mel_frames(self) -> int:
        return int(self.prompt_feat.shape[1])

    @classmethod
    def from_npz(cls, path: str) -> "PromptContext":
        """Read one `.npz` written by `scripts/prepare_inputs.py` (keys listed in that script's docstring)."""
        import numpy as np

        with np.load(path) as d:
            case = json.loads(str(d["case_json"]))
            mode = case["mode"]
            fills = describe_mode(mode)

            def ids(key, wanted):
                return torch.from_numpy(d[key].astype(np.int32)) if wanted and key in d.files else None

            return cls(
                mode=mode,
                lang=case["lang"],
                prompt_text_ids=ids("prompt_text_ids", fills["llm_prompt_text"]),
                llm_prompt_speech_tokens=ids("llm_prompt_speech_tokens", fills["llm_prompt_speech"]),
                flow_prompt_speech_tokens=torch.from_numpy(d["flow_prompt_speech_tokens"].astype(np.int32)),
                prompt_feat=torch.from_numpy(d["prompt_feat"].astype(np.float32)),
                embedding=torch.from_numpy(d["flow_embedding"].astype(np.float32)).reshape(1, -1),
                meta={
                    "case": case,
                    "segments": json.loads(str(d["segments_json"])),
                    "segment_text_ids": json.loads(str(d["segment_text_ids_json"])),
                    "prompt_text_normalized": str(d["prompt_text_normalized"]),
                    "versions": json.loads(str(d["meta_json"])) if "meta_json" in d.files else {},
                    "path": path,
                },
            )
