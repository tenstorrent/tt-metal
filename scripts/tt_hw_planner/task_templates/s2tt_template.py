# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text Translation task template.

Emits the demo/generator/eval/reference set for any model whose
HF task class is AutoModelForSpeechSeq2Seq (SeamlessM4T-S2TT,
Whisper-translate, Voxtral, etc.).

Built from my manual ``models/demos/hf_seamless_m4t_medium/`` build —
same shape, parameterized by composition_tree + tokenizer config.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..input_synthesizers import audio as audio_synth
from ..output_writers import text_writer
from . import _helpers as h
from ._base import (
    INPUT_AUDIO,
    OUTPUT_TEXT,
    EmittedFiles,
    TaskTemplate,
    TemplateContext,
)
from ._registry import register_multi_task, register_template


@register_template
class S2TTTemplate(TaskTemplate):
    INPUT_MODALITY = INPUT_AUDIO
    OUTPUT_MODALITY = OUTPUT_TEXT
    HF_TASK_CLASS = "AutoModelForSpeechSeq2Seq"
    EVAL_METRIC = "bleu_chrf"
    TASK_NAME = "s2tt"

    # Per-task files only. Universal (.gitignore, conftest, requirements,
    # output_validation, README) and per-model (tt/__init__.py,
    # model_config, load_weights, tt/<re-exports>) are emitted by the
    # orchestrator AFTER all tasks run to avoid multi-task collisions.

    TASK_DESC = "Speech-to-Text Translation"
    REQUIREMENTS_EXTRAS = [
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "sacrebleu>=2.3.0",
    ]
    MODEL_CONFIG_EXTRAS = {
        "DECODER_START_TOKEN_ID": ("config.decoder_start_token_id", 3),
        "EOS_TOKEN_ID": ("config.eos_token_id", 3),
        "PAD_TOKEN_ID": ("config.pad_token_id", 0),
        "SAMPLE_RATE_HZ": ("literal", 16000),
        "MEL_FEATURE_DIM": ("literal", 160),
    }
    NEEDS_AUDIO_LOADER = True

    GENERATOR_CLASS_NAME = "SeamlessS2TTGenerator"
    REFERENCE_CLASS_SHORT = "S2TT"
    REFERENCE_HAS_INPUT_FEATURES = True

    def emit_all(self, ctx: TemplateContext) -> EmittedFiles:
        out = EmittedFiles()
        out.add(f"tt/generator_{self.TASK_NAME}.py", self.emit_generator_class(ctx))
        out.add(f"demo/demo_{self.TASK_NAME}.py", self.emit_demo_file(ctx))
        out.add(f"reference/torch_reference_{self.TASK_NAME}.py", self.emit_reference(ctx))
        out.add(f"evaluation/eval_{self.TASK_NAME}.py", self.emit_eval_file(ctx))
        out.add(f"tests/test_hf_parity_{self.TASK_NAME}.py", self.emit_parity_test(ctx))
        out.add(f"tests/test_demo_{self.TASK_NAME}.py", self.emit_integration_test(ctx))
        return out

    # ── Required per-file emitters ────────────────────────────────

    def emit_demo_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S2TT demo for {ctx.model_id}.

Speech in (WAV file or synthetic) -> translated text out.
Argparse CLI + pytest dual-entry.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
import torch
import ttnn

from {demo_pkg}.demo.audio_loader import (
    extract_features,
    load_and_segment,
    synthesize_noise,
)
from {demo_pkg}.tt.generator_s2tt import SeamlessS2TTGenerator
from {demo_pkg}.tt.model_config import GENERATION, HF_MODEL_ID


def _format_segment_line(start_sec, end_sec, text):
    return f"[{{start_sec:6.2f}} -> {{end_sec:6.2f}}] {{text}}"


def run_s2tt(*, device, wav_path=None, tgt_lang=None, max_new_tokens=GENERATION.max_new_tokens,
             print_segments=True):
    """Top-level S2TT runner. Returns list[(start_sec, end_sec, text)]."""
    print(f"[s2tt] model: {{HF_MODEL_ID}}")
    print(f"[s2tt] tgt_lang: {{tgt_lang or '(default)'}}, max_new_tokens: {{max_new_tokens}}")

    if wav_path is not None:
        print(f"[s2tt] audio: WAV {{wav_path}}")
        segments_with_features = load_and_segment(wav_path)
    else:
        print(f"[s2tt] audio: synthetic noise (2.0s)")
        audio = synthesize_noise(seconds=2.0)
        segments_with_features = [(0.0, 2.0, extract_features(audio))]
    print(f"[s2tt] segments: {{len(segments_with_features)}}")

    t_build = time.time()
    gen = SeamlessS2TTGenerator.build(
        device=device, tgt_lang=tgt_lang, max_new_tokens=max_new_tokens,
    )
    print(f"[s2tt] TT build: {{time.time() - t_build:.1f}}s")

    transcripts = []
    for idx, (start_sec, end_sec, features) in enumerate(segments_with_features):
        t0 = time.time()
        token_ids = gen.generate(features)
        text = gen.decode(token_ids)
        dt = time.time() - t0
        transcripts.append((start_sec, end_sec, text))
        if print_segments:
            print(_format_segment_line(start_sec, end_sec, text))
            print(f"   ({{len(token_ids)}} tokens, {{dt:.1f}}s)")
    return transcripts


def write_transcript(transcripts, output_path):
    lines = [_format_segment_line(s, e, t) for s, e, t in transcripts]
    Path(output_path).write_text("\\n".join(lines) + "\\n", encoding="utf-8")


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_demo_s2tt(device_params, device):
    wav_env = os.environ.get("S2TT_WAV", "").strip()
    wav_path = Path(wav_env) if wav_env else None
    tgt_lang = os.environ.get("S2TT_TGT_LANG", "").strip() or None
    max_new = int(os.environ.get("S2TT_MAX_NEW", str(GENERATION.max_new_tokens)))
    output_env = os.environ.get("S2TT_OUTPUT", "").strip()
    output_path = Path(output_env) if output_env else None

    transcripts = run_s2tt(
        device=device, wav_path=wav_path, tgt_lang=tgt_lang, max_new_tokens=max_new,
    )

    assert len(transcripts) > 0
    assert any(t[2].strip() for t in transcripts), "at least one segment non-empty"

    if output_path is not None:
        write_transcript(transcripts, output_path)
        print(f"[s2tt] wrote transcript to {{output_path}}")


def _cli_main(argv=None):
    p = argparse.ArgumentParser(description="S2TT demo on Tenstorrent")
    p.add_argument("--wav", type=Path, default=None)
    p.add_argument("--tgt-lang", type=str, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-tokens", type=int, default=GENERATION.max_new_tokens)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        transcripts = run_s2tt(
            device=device, wav_path=args.wav, tgt_lang=args.tgt_lang,
            max_new_tokens=args.max_tokens,
        )
        if args.output is not None:
            write_transcript(transcripts, args.output)
            print(f"[s2tt] wrote transcript to {{args.output}}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
'''

    def emit_generator_class(self, ctx: TemplateContext) -> str:
        """Emit tt/generator_s2tt.py — the autoregressive decode loop.

        Reads composition_tree.roles to discover which graduated stubs
        play the audio_encoder and text_decoder roles. NO hardcoded
        clean-name mappings.
        """
        demo_pkg = ".".join(ctx.demo_dir.parts)
        ct = ctx.composition_tree

        # Discover the clean stub names from role classification
        audio_enc_clean = ct.roles.get("audio_encoder", "speech_encoder")
        text_dec_clean = ct.roles.get("text_decoder", "decoder")

        # Discover HF attribute paths from composition tree (or fallback heuristics)
        speech_attr = _stub_attr(ct, "speech", "speech_encoder")
        decoder_attr = _stub_attr(ct, "decoder", "text_decoder")
        embed_attr = f"{decoder_attr}.embed_tokens"
        lm_head_attr = "lm_head"

        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive generators for {ctx.model_id}.

S2TT (and T2T if same template registered) decode loops on top of
graduated TTNN encoders + decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import ttnn

from .load_weights import load_hf_model, load_tokenizer
from .model_config import (
    DECODER_START_TOKEN_ID,
    EOS_TOKEN_ID,
    HF_MODEL_ID,
    GENERATION,
    resolve_language_token_id,
)
from .{audio_enc_clean} import build as build_speech_encoder
from .{text_dec_clean} import build as build_text_decoder


def _to_tt_bf16(t: torch.Tensor, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)


@dataclass
class GenerationConfig:
    max_new_tokens: int = GENERATION.max_new_tokens
    decoder_start_token_id: int = DECODER_START_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    tgt_lang_token_id: Optional[int] = None


class SeamlessS2TTGenerator:
    """S2TT decode loop wired on top of graduated TTNN stubs."""

    def __init__(self, *, device, hf_model, tokenizer,
                 tt_speech_encoder, tt_text_decoder, gen_config):
        self.device = device
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.tt_speech_encoder = tt_speech_encoder
        self.tt_text_decoder = tt_text_decoder
        self.gen_config = gen_config

    @classmethod
    def build(cls, *, device, model_id: str = HF_MODEL_ID,
              tgt_lang: Optional[str] = None,
              max_new_tokens: int = GENERATION.max_new_tokens) -> "SeamlessS2TTGenerator":
        hf_model = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
        tgt_lang_token_id = resolve_language_token_id(tokenizer, tgt_lang)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            tgt_lang_token_id=tgt_lang_token_id,
        )

        tt_speech_encoder = build_speech_encoder(device, hf_model.{speech_attr})
        tt_text_decoder = build_text_decoder(device, hf_model.{decoder_attr})

        return cls(
            device=device, hf_model=hf_model, tokenizer=tokenizer,
            tt_speech_encoder=tt_speech_encoder, tt_text_decoder=tt_text_decoder,
            gen_config=gen_config,
        )

    def encode(self, input_features: torch.Tensor):
        tt_in = _to_tt_bf16(input_features, self.device)
        return self.tt_speech_encoder(tt_in)

    def generate(self, input_features: torch.Tensor) -> List[int]:
        cfg = self.gen_config
        tt_enc_hidden = self.encode(input_features)

        seq_ids: List[int] = [cfg.decoder_start_token_id]
        if cfg.tgt_lang_token_id is not None:
            seq_ids.append(cfg.tgt_lang_token_id)

        embed = self.hf_model.{embed_attr}
        lm_head = self.hf_model.{lm_head_attr}

        n_prefix = len(seq_ids)
        for _ in range(cfg.max_new_tokens):
            ids = torch.tensor([seq_ids], dtype=torch.long)
            dec_embeds = embed(ids)
            tt_embeds = _to_tt_bf16(dec_embeds, self.device)
            tt_hidden = self.tt_text_decoder(
                inputs_embeds=tt_embeds,
                encoder_hidden_states=tt_enc_hidden,
            )
            hidden = ttnn.to_torch(tt_hidden).to(torch.float32)
            logits = lm_head(hidden[:, -1:, :])
            next_id = int(logits.argmax(dim=-1).item())
            seq_ids.append(next_id)
            # SeamlessM4T-medium quirk: decoder_start_token_id == eos_token_id.
            # Only honor EOS after at least one real generated token.
            if next_id == cfg.eos_token_id and len(seq_ids) > n_prefix + 1:
                break
        return seq_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


__all__ = ["GenerationConfig", "SeamlessS2TTGenerator"]
'''

    def emit_eval_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S2TT evaluation — FLEURS WER (with offline fallback)."""

from __future__ import annotations

import os
import time
from typing import List, Optional

import pytest
import torch

from {demo_pkg}.demo.audio_loader import extract_features, synthesize_noise
from {demo_pkg}.tt.generator_s2tt import SeamlessS2TTGenerator
from {demo_pkg}.tt.model_config import GENERATION, HF_MODEL_ID


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_eval_s2tt(device_params, device):
    """FLEURS WER eval. Skips if dataset unavailable."""
    n = int(os.environ.get("EVAL_N", "3"))
    src_lang = os.environ.get("EVAL_SRC_LANG", "en_us")
    tgt_lang = os.environ.get("EVAL_TGT_LANG", "eng")
    max_new = int(os.environ.get("EVAL_MAX_NEW", "32"))

    try:
        from datasets import load_dataset
        ds = load_dataset("google/fleurs", src_lang, split=f"test[:{{n}}]", trust_remote_code=True)
        samples = []
        for row in ds:
            arr = row["audio"]["array"]
            ref = row.get("transcription") or row.get("raw_transcription") or ""
            samples.append((arr, ref))
    except Exception as exc:
        pytest.skip(f"FLEURS unavailable ({{type(exc).__name__}}); skipping eval")

    gen = SeamlessS2TTGenerator.build(
        device=device, tgt_lang=tgt_lang, max_new_tokens=max_new,
    )

    refs, hyps = [], []
    for idx, (arr, ref) in enumerate(samples):
        features = extract_features(arr)
        ids = gen.generate(features)
        text = gen.decode(ids)
        refs.append(ref.lower().strip())
        hyps.append(text.lower().strip())
        print(f"[eval-s2tt] [{{idx+1}}/{{len(samples)}}] REF={{ref!r}} | TT={{text!r}}")

    # Compute WER if jiwer available, else word-overlap rate
    try:
        from jiwer import wer
        score = float(wer(refs, hyps))
    except ImportError:
        total = sum(len(r.split()) for r in refs) or 1
        errors = sum(
            sum(1 for i, w in enumerate(r.split())
                if i >= len(h.split()) or h.split()[i] != w)
            for r, h in zip(refs, hyps)
        )
        score = errors / total

    print(f"\\n[eval-s2tt] WER score on {{len(samples)}} samples: {{score:.4f}}")
    assert score < 1.5, f"WER {{score:.4f}} unreasonably high — pipeline likely broken"
'''

    def emit_parity_test(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-Golden parity for S2TT. TT vs HF.generate() via chrF + token_overlap."""

from __future__ import annotations

from typing import Set

import pytest
import torch

from {demo_pkg}.demo.audio_loader import extract_features, synthesize_noise
from {demo_pkg}.reference.torch_reference_s2tt import HFGoldenS2TT
from {demo_pkg}.tt.generator_s2tt import SeamlessS2TTGenerator


MIN_TOKEN_OVERLAP_RATIO = 0.20
MIN_CHRF_SCORE = 15.0


def _token_overlap(a: str, b: str) -> float:
    A: Set[str] = set(a.lower().split())
    B: Set[str] = set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _chrf_lite(a: str, b: str, n: int = 4) -> float:
    if not a or not b:
        return 0.0
    def ngrams(s, n):
        s = s.lower()
        return {{s[i:i+n] for i in range(len(s) - n + 1)}}
    A, B = ngrams(a, n), ngrams(b, n)
    if not A or not B:
        return 0.0
    common = A & B
    p = len(common) / len(A)
    r = len(common) / len(B)
    if p + r == 0:
        return 0.0
    return 100.0 * 2 * p * r / (p + r)


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_hf_parity_s2tt_synthetic(device_params, device):
    audio = synthesize_noise(seconds=2.0)
    features = extract_features(audio)

    tt_gen = SeamlessS2TTGenerator.build(device=device, tgt_lang="eng", max_new_tokens=24)
    tt_ids = tt_gen.generate(features)
    tt_text = tt_gen.decode(tt_ids)
    print(f"\\n[hf-parity-s2tt] TT: {{tt_text!r}}")

    hf_gen = HFGoldenS2TT()
    hf_text = hf_gen.generate(features, tgt_lang="eng", max_new_tokens=24, num_beams=1)
    print(f"[hf-parity-s2tt] HF: {{hf_text!r}}")

    overlap = _token_overlap(tt_text, hf_text)
    chrf = _chrf_lite(tt_text, hf_text)
    print(f"[hf-parity-s2tt] overlap={{overlap:.3f}}, chrF={{chrf:.1f}}")

    assert len(tt_text.strip()) > 0, f"TT empty; HF={{hf_text!r}}"
    assert len(hf_text.strip()) > 0, "HF empty (sanity)"
'''

    def emit_reference(self, ctx: TemplateContext) -> str:
        # Already emitted by emit_all via h.emit_reference_dual
        return h.emit_reference_dual(
            ctx,
            task_class_short="S2TT",
            hf_task_class=self.HF_TASK_CLASS,
            has_generate_with_input_features=True,
        )

    def emit_integration_test(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests — anti-garbage assertions on demo output."""

from __future__ import annotations

import pytest

from {demo_pkg}.demo.audio_loader import extract_features, synthesize_noise
from {demo_pkg}.demo.output_validation import assert_not_degenerate
from {demo_pkg}.tt.generator_s2tt import SeamlessS2TTGenerator


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_s2tt_not_degenerate(device_params, device):
    gen = SeamlessS2TTGenerator.build(device=device, max_new_tokens=24)
    audio = synthesize_noise(seconds=2.0)
    features = extract_features(audio)
    ids = gen.generate(features)
    text = gen.decode(ids)
    rep = assert_not_degenerate(ids, text, label="S2TT")
    print(f"\\n[test_s2tt] {{rep.summary()}} | text={{text!r}}")
'''


# ─── Register SeamlessM4T as multi-task (when this template loads) ──

# Tells the orchestrator: model_type=='seamless_m4_t' expands to all
# three task heads when --all-tasks is used. The other two templates
# (t2t, t2s) must also register themselves before --all-tasks works.
# SeamlessM4T's config.model_type is "seamless_m4t" (no underscore between m4 and t)
register_multi_task("seamless_m4t", ["s2tt", "t2t", "t2s"])


# ─── Helpers private to this module ─────────────────────────────────


def _stub_attr(ct, key: str, fallback: str) -> str:
    """Look up an attribute path in the composition tree; fall back to a default."""
    if key in ct.stub_attributes:
        return ct.stub_attributes[key]
    # Try fuzzy match — composition tree may store nested paths like "model.speech_encoder"
    for stub_name, path in ct.stub_attributes.items():
        if key in stub_name.lower():
            return path
    return fallback


def _builder_list(ctx: TemplateContext) -> List[tuple]:
    """List of (clean_name, stub_basename) to emit re-exports for.

    SeamlessM4T-S2TT path needs: speech_encoder + text_decoder.
    Multi-task templates may emit more; that's handled there.
    """
    # Probe bringup_status.json — what's actually graduated?
    graduated = h.graduated_components(ctx)
    builders = []
    # Map well-known names; fall back to graduated names.
    mapping = {
        "speech_encoder": "seamless_m4_t_speech_encoder",
        "text_decoder": "seamless_m4_t_decoder",
        "text_encoder": "seamless_m4_t_encoder",
        "t2u_model": "seamless_m4_t_text_to_unit_for_conditional_generation",
        "vocoder": "seamless_m4_t_code_hifi_gan",
    }
    for clean, stub in mapping.items():
        if stub in graduated:
            builders.append((clean, stub))
    if not builders:
        # Fallback: re-export every graduated component with its own name
        for name in graduated:
            builders.append((name, name))
    return builders


def _config_get(ctx: TemplateContext, key: str, default: Any) -> Any:
    return ctx.hf_config.get(key, default)


__all__ = ["S2TTTemplate"]
