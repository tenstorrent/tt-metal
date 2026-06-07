# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text-to-Text Translation task template.

Emits demo/generator/eval/reference for AutoModelForSeq2SeqLM models
(NLLB, M2M100, mBART, SeamlessM4T-T2T, etc.).
"""

from __future__ import annotations

from typing import Any, List, Tuple

from . import _helpers as h
from ._base import (
    EmittedFiles,
    INPUT_TEXT,
    OUTPUT_TEXT,
    TaskTemplate,
    TemplateContext,
)
from ._registry import register_template


@register_template
class T2TTemplate(TaskTemplate):
    INPUT_MODALITY = INPUT_TEXT
    OUTPUT_MODALITY = OUTPUT_TEXT
    HF_TASK_CLASS = "AutoModelForSeq2SeqLM"
    EVAL_METRIC = "bleu"
    TASK_NAME = "t2t"

    TASK_DESC = "Text-to-Text Translation"
    REQUIREMENTS_EXTRAS = ["sacrebleu>=2.3.0"]
    MODEL_CONFIG_EXTRAS = {
        "DECODER_START_TOKEN_ID": ("config.decoder_start_token_id", 3),
        "EOS_TOKEN_ID": ("config.eos_token_id", 3),
        "PAD_TOKEN_ID": ("config.pad_token_id", 0),
    }
    NEEDS_AUDIO_LOADER = False
    GENERATOR_CLASS_NAME = "SeamlessT2TGenerator"
    REFERENCE_CLASS_SHORT = "T2T"
    REFERENCE_HAS_INPUT_FEATURES = False

    def emit_all(self, ctx: TemplateContext) -> EmittedFiles:
        out = EmittedFiles()
        out.add(f"tt/generator_{self.TASK_NAME}.py", self.emit_generator_class(ctx))
        out.add(f"demo/demo_{self.TASK_NAME}.py", self.emit_demo_file(ctx))
        out.add(f"reference/torch_reference_{self.TASK_NAME}.py", self.emit_reference(ctx))
        out.add(f"evaluation/eval_{self.TASK_NAME}.py", self.emit_eval_file(ctx))
        out.add(f"tests/test_hf_parity_{self.TASK_NAME}.py", self.emit_parity_test(ctx))
        out.add(f"tests/test_demo_{self.TASK_NAME}.py", self.emit_integration_test(ctx))
        return out

    def emit_demo_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T2T demo for {ctx.model_id}.

Text in -> translated text out. Argparse CLI + pytest dual-entry.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import ttnn

from {demo_pkg}.tt.generator_t2t import SeamlessT2TGenerator
from {demo_pkg}.tt.model_config import GENERATION, HF_MODEL_ID


DEFAULT_SOURCE_TEXT = "The quick brown fox jumps over the lazy dog."


def run_t2t(*, device, text, tgt_lang=None, max_new_tokens=GENERATION.max_new_tokens):
    print(f"[t2t] model: {{HF_MODEL_ID}}")
    print(f"[t2t] source: {{text!r}}")
    print(f"[t2t] tgt_lang: {{tgt_lang or '(default)'}}, max_new_tokens: {{max_new_tokens}}")
    t_build = time.time()
    gen = SeamlessT2TGenerator.build(
        device=device, tgt_lang=tgt_lang, max_new_tokens=max_new_tokens,
    )
    print(f"[t2t] TT build: {{time.time() - t_build:.1f}}s")
    t0 = time.time()
    token_ids = gen.generate(text)
    translation = gen.decode(token_ids)
    dt = time.time() - t0
    print(f"[t2t] generated {{len(token_ids)}} tokens in {{dt:.1f}}s")
    print(f"[t2t] translation: {{translation!r}}")
    return translation


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_demo_t2t(device_params, device):
    text = os.environ.get("T2T_TEXT", DEFAULT_SOURCE_TEXT)
    tgt_lang = os.environ.get("T2T_TGT_LANG", "").strip() or None
    max_new = int(os.environ.get("T2T_MAX_NEW", str(GENERATION.max_new_tokens)))
    output_env = os.environ.get("T2T_OUTPUT", "").strip()

    translation = run_t2t(device=device, text=text, tgt_lang=tgt_lang, max_new_tokens=max_new)
    assert isinstance(translation, str)
    assert len(translation.strip()) > 0

    if output_env:
        Path(output_env).write_text(translation + "\\n", encoding="utf-8")


def _cli_main(argv=None):
    p = argparse.ArgumentParser(description="T2T demo on Tenstorrent")
    p.add_argument("--text", type=str, default=DEFAULT_SOURCE_TEXT)
    p.add_argument("--tgt-lang", type=str, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-tokens", type=int, default=GENERATION.max_new_tokens)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        translation = run_t2t(
            device=device, text=args.text, tgt_lang=args.tgt_lang,
            max_new_tokens=args.max_tokens,
        )
        if args.output is not None:
            args.output.write_text(translation + "\\n", encoding="utf-8")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
'''

    def emit_generator_class(self, ctx: TemplateContext) -> str:
        ct = ctx.composition_tree
        text_enc_clean = ct.roles.get("text_encoder", "encoder")
        text_dec_clean = ct.roles.get("text_decoder", "decoder")
        text_enc_attr = _stub_attr(ct, "text_encoder", "text_encoder")
        decoder_attr = _stub_attr(ct, "decoder", "text_decoder")
        embed_attr = f"{decoder_attr}.embed_tokens"
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T2T generator for {ctx.model_id}."""

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
from .{text_enc_clean} import build as build_text_encoder
from .{text_dec_clean} import build as build_text_decoder


def _to_tt_bf16(t, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)


def _to_tt_ids(ids, device):
    return ttnn.from_torch(
        ids.to(torch.int32),
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )


@dataclass
class T2TConfig:
    max_new_tokens: int = GENERATION.max_new_tokens
    decoder_start_token_id: int = DECODER_START_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    tgt_lang_token_id: Optional[int] = None


class SeamlessT2TGenerator:
    """T2T decode loop wired on top of graduated TTNN stubs."""

    def __init__(self, *, device, hf_model, tokenizer,
                 tt_text_encoder, tt_text_decoder, gen_config):
        self.device = device
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.tt_text_encoder = tt_text_encoder
        self.tt_text_decoder = tt_text_decoder
        self.gen_config = gen_config

    @classmethod
    def build(cls, *, device, model_id=HF_MODEL_ID, tgt_lang=None,
              max_new_tokens=GENERATION.max_new_tokens):
        hf_model = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
        tgt_lang_token_id = resolve_language_token_id(tokenizer, tgt_lang)
        gen_config = T2TConfig(
            max_new_tokens=max_new_tokens, tgt_lang_token_id=tgt_lang_token_id,
        )
        tt_text_encoder = build_text_encoder(device, hf_model.{text_enc_attr})
        tt_text_decoder = build_text_decoder(device, hf_model.{decoder_attr})
        return cls(
            device=device, hf_model=hf_model, tokenizer=tokenizer,
            tt_text_encoder=tt_text_encoder, tt_text_decoder=tt_text_decoder,
            gen_config=gen_config,
        )

    def encode_text(self, text):
        encoded = self.tokenizer(text, return_tensors="pt")
        tt_ids = _to_tt_ids(encoded["input_ids"], self.device)
        return self.tt_text_encoder(input_ids=tt_ids)

    def generate(self, text):
        cfg = self.gen_config
        tt_enc_hidden = self.encode_text(text)

        seq_ids = [cfg.decoder_start_token_id]
        if cfg.tgt_lang_token_id is not None:
            seq_ids.append(cfg.tgt_lang_token_id)

        embed = self.hf_model.{embed_attr}
        lm_head = self.hf_model.lm_head

        n_prefix = len(seq_ids)
        for _ in range(cfg.max_new_tokens):
            ids = torch.tensor([seq_ids], dtype=torch.long)
            dec_embeds = embed(ids)
            tt_embeds = _to_tt_bf16(dec_embeds, self.device)
            tt_hidden = self.tt_text_decoder(
                inputs_embeds=tt_embeds, encoder_hidden_states=tt_enc_hidden,
            )
            hidden = ttnn.to_torch(tt_hidden).to(torch.float32)
            logits = lm_head(hidden[:, -1:, :])
            next_id = int(logits.argmax(dim=-1).item())
            seq_ids.append(next_id)
            if next_id == cfg.eos_token_id and len(seq_ids) > n_prefix + 1:
                break
        return seq_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


__all__ = ["T2TConfig", "SeamlessT2TGenerator"]
'''

    def emit_eval_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T2T evaluation — FLORES BLEU (with offline fallback)."""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import pytest

from {demo_pkg}.tt.generator_t2t import SeamlessT2TGenerator
from {demo_pkg}.tt.model_config import HF_MODEL_ID


_FALLBACK_PAIRS: List[Tuple[str, str]] = [
    ("The quick brown fox jumps over the lazy dog.",
     "Le rapide renard brun saute par-dessus le chien paresseux."),
    ("Hello, how are you today?",
     "Bonjour, comment allez-vous aujourd'hui?"),
    ("I love programming.",
     "J'aime programmer."),
    ("The weather is nice today.",
     "Il fait beau aujourd'hui."),
]


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_eval_t2t(device_params, device):
    """FLORES BLEU (auto-falls-back to hand-curated pairs offline)."""
    n = int(os.environ.get("EVAL_N", "3"))
    src_lang_flores = os.environ.get("EVAL_SRC_LANG", "eng_Latn")
    tgt_lang_flores = os.environ.get("EVAL_TGT_LANG_FLORES", "fra_Latn")
    tgt_lang_tok = os.environ.get("EVAL_TGT_LANG", "fra")
    max_new = int(os.environ.get("EVAL_MAX_NEW", "32"))

    try:
        from datasets import load_dataset
        src = load_dataset("facebook/flores", src_lang_flores, split=f"devtest[:{{n}}]", trust_remote_code=True)
        tgt = load_dataset("facebook/flores", tgt_lang_flores, split=f"devtest[:{{n}}]", trust_remote_code=True)
        pairs = [(s["sentence"], t["sentence"]) for s, t in zip(src, tgt)]
    except Exception as exc:
        print(f"[eval-t2t] FLORES unavailable ({{type(exc).__name__}}); using fallback pairs")
        pairs = _FALLBACK_PAIRS[:n]

    gen = SeamlessT2TGenerator.build(
        device=device, tgt_lang=tgt_lang_tok, max_new_tokens=max_new,
    )

    refs, hyps = [], []
    for idx, (src_text, ref) in enumerate(pairs):
        ids = gen.generate(src_text)
        translation = gen.decode(ids)
        print(f"[eval-t2t] [{{idx+1}}/{{len(pairs)}}] SRC={{src_text!r}}")
        print(f"           REF={{ref!r}}")
        print(f"           TT={{translation!r}}")
        refs.append(ref)
        hyps.append(translation)

    # BLEU if sacrebleu available, else word-overlap rate
    try:
        import sacrebleu
        bleu = float(sacrebleu.corpus_bleu(hyps, [refs]).score)
    except ImportError:
        overlap = 0
        total = 0
        for r, h in zip(refs, hyps):
            R = set(r.lower().split())
            H = set(h.lower().split())
            overlap += len(R & H)
            total += len(R)
        bleu = 100.0 * overlap / max(total, 1)

    print(f"\\n[eval-t2t] BLEU on {{len(pairs)}} samples: {{bleu:.2f}}")
    assert bleu > 0.5, f"BLEU {{bleu:.2f}} too low — pipeline likely broken"
'''

    def emit_parity_test(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-Golden parity for T2T."""

from __future__ import annotations

from typing import Set

import pytest

from {demo_pkg}.reference.torch_reference_t2t import HFGoldenT2T
from {demo_pkg}.tt.generator_t2t import SeamlessT2TGenerator


MIN_TOKEN_OVERLAP_RATIO = 0.20
MIN_CHRF_SCORE = 15.0


def _token_overlap(a, b):
    A: Set[str] = set(a.lower().split())
    B: Set[str] = set(b.lower().split())
    return 0.0 if not A or not B else len(A & B) / len(A | B)


def _chrf_lite(a, b, n=4):
    if not a or not b:
        return 0.0
    def ng(s, n):
        s = s.lower()
        return {{s[i:i+n] for i in range(len(s)-n+1)}}
    A, B = ng(a, n), ng(b, n)
    if not A or not B:
        return 0.0
    c = A & B
    p, r = len(c) / len(A), len(c) / len(B)
    return 0.0 if p + r == 0 else 100.0 * 2 * p * r / (p + r)


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_hf_parity_t2t(device_params, device):
    src = "The quick brown fox jumps over the lazy dog."
    tgt = "fra"
    max_new = 32

    tt_gen = SeamlessT2TGenerator.build(device=device, tgt_lang=tgt, max_new_tokens=max_new)
    tt_ids = tt_gen.generate(src)
    tt_text = tt_gen.decode(tt_ids)
    print(f"\\n[hf-parity-t2t] TT: {{tt_text!r}}")

    hf_gen = HFGoldenT2T()
    hf_text = hf_gen.generate(src, tgt_lang=tgt, max_new_tokens=max_new, num_beams=1)
    print(f"[hf-parity-t2t] HF: {{hf_text!r}}")

    overlap = _token_overlap(tt_text, hf_text)
    chrf = _chrf_lite(tt_text, hf_text)
    print(f"[hf-parity-t2t] overlap={{overlap:.3f}}, chrF={{chrf:.1f}}")

    assert overlap >= MIN_TOKEN_OVERLAP_RATIO, (
        f"TT/HF disagree: overlap={{overlap:.3f}}\\n  TT: {{tt_text!r}}\\n  HF: {{hf_text!r}}"
    )
    assert chrf >= MIN_CHRF_SCORE, (
        f"TT/HF chrF too low: {{chrf:.1f}}\\n  TT: {{tt_text!r}}\\n  HF: {{hf_text!r}}"
    )
'''

    def emit_reference(self, ctx: TemplateContext) -> str:
        return h.emit_reference_dual(
            ctx,
            task_class_short="T2T",
            hf_task_class=self.HF_TASK_CLASS,
            has_generate_with_input_features=False,
        )

    def emit_integration_test(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration test — T2T non-degenerate output."""

from __future__ import annotations

import pytest

from {demo_pkg}.demo.output_validation import assert_not_degenerate
from {demo_pkg}.tt.generator_t2t import SeamlessT2TGenerator


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_t2t_not_degenerate(device_params, device):
    gen = SeamlessT2TGenerator.build(device=device, max_new_tokens=24)
    ids = gen.generate("The weather is nice today.")
    text = gen.decode(ids)
    rep = assert_not_degenerate(ids, text, label="T2T")
    print(f"\\n[test_t2t] {{rep.summary()}} | text={{text!r}}")
'''


def _stub_attr(ct, key, fallback):
    if key in ct.stub_attributes:
        return ct.stub_attributes[key]
    for stub_name, path in ct.stub_attributes.items():
        if key in stub_name.lower():
            return path
    return fallback


__all__ = ["T2TTemplate"]
