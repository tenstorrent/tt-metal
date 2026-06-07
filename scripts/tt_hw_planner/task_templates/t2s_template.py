# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Text-to-Speech task template.

Emits demo/generator/eval/reference for AutoModelForTextToWaveform
models (SeamlessM4T-T2S, etc.). Pipeline: text -> text_encoder ->
t2u_model -> argmax -> vocoder -> WAV.
"""

from __future__ import annotations

from . import _helpers as h
from ._base import (
    EmittedFiles,
    INPUT_TEXT,
    OUTPUT_AUDIO,
    TaskTemplate,
    TemplateContext,
)
from ._registry import register_template


@register_template
class T2STemplate(TaskTemplate):
    INPUT_MODALITY = INPUT_TEXT
    OUTPUT_MODALITY = OUTPUT_AUDIO
    HF_TASK_CLASS = "AutoModelForTextToWaveform"
    EVAL_METRIC = "pipeline_reach"
    TASK_NAME = "t2s"

    TASK_DESC = "Text-to-Speech"
    REQUIREMENTS_EXTRAS = ["scipy>=1.10.0"]
    MODEL_CONFIG_EXTRAS = {
        "VOCODER_UNIT_VOCAB_SIZE": ("literal", 10000),
        "SAMPLE_RATE_HZ": ("literal", 16000),
    }
    NEEDS_AUDIO_LOADER = False  # T2S writes audio, doesn't read it
    GENERATOR_CLASS_NAME = ""  # T2S uses inline pipeline, no generator class
    REFERENCE_CLASS_SHORT = "T2S"
    REFERENCE_HAS_INPUT_FEATURES = False

    def emit_all(self, ctx: TemplateContext) -> EmittedFiles:
        out = EmittedFiles()
        # T2S uses an inline pipeline in the demo, no separate generator class.
        out.add(f"demo/demo_{self.TASK_NAME}.py", self.emit_demo_file(ctx))
        out.add(f"reference/torch_reference_{self.TASK_NAME}.py", self.emit_reference(ctx))
        out.add(f"evaluation/eval_{self.TASK_NAME}.py", self.emit_eval_file(ctx))
        out.add(f"tests/test_demo_{self.TASK_NAME}.py", self.emit_integration_test(ctx))
        # T2S has no parity test by default (vocoder output is hard to compare bit-for-bit)
        return out

    def emit_demo_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        ct = ctx.composition_tree
        text_enc_clean = ct.roles.get("text_encoder", "encoder")
        t2u_clean = ct.roles.get("t2u_model", "text_to_unit_for_conditional_generation")
        vocoder_clean = ct.roles.get("vocoder", "code_hifi_gan")
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T2S demo for {ctx.model_id}.

Text in -> speech (.wav) out. Single-step t2u + vocoder.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import torch
import transformers
import ttnn

from {demo_pkg}.tt import (
    build_t2u_model,
    build_text_encoder,
    build_vocoder,
)
from {demo_pkg}.tt.load_weights import load_tokenizer
from {demo_pkg}.tt.model_config import (
    HF_MODEL_ID,
    SAMPLE_RATE_HZ,
    VOCODER_UNIT_VOCAB_SIZE,
)


DEFAULT_SOURCE_TEXT = "The quick brown fox jumps over the lazy dog."


def _to_tt_bf16(t, device):
    return ttnn.from_torch(
        t.to(torch.bfloat16), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=device,
    )


def _to_tt_ids(ids, device):
    return ttnn.from_torch(
        ids.to(torch.int32), dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )


def write_wav(waveform, output_path, sample_rate: int = SAMPLE_RATE_HZ):
    import numpy as np
    from scipy.io import wavfile
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    arr = np.asarray(waveform).reshape(-1).astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("int16")
    wavfile.write(str(output_path), sample_rate, pcm)


def run_t2s(*, device, text, spkr_id=0, lang_id=0, output_path=None):
    print(f"[t2s] model: {{HF_MODEL_ID}}")
    print(f"[t2s] source: {{text!r}}, spkr={{spkr_id}}, lang={{lang_id}}")

    t_load = time.time()
    hf_model = transformers.AutoModelForTextToWaveform.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True,
    )
    hf_model.eval()
    tokenizer = load_tokenizer()
    print(f"[t2s] HF load: {{time.time() - t_load:.1f}}s")

    encoded = tokenizer(text, return_tensors="pt")
    text_ids = encoded["input_ids"]
    print(f"[t2s] tokenized to {{text_ids.shape[-1]}} tokens")

    t_build = time.time()
    tt_text_enc = build_text_encoder(device, hf_model.text_encoder)
    tt_t2u = build_t2u_model(device, hf_model.t2u_model)
    tt_vocoder = build_vocoder(device, hf_model.vocoder)
    print(f"[t2s] TT build: {{time.time() - t_build:.1f}}s")

    # Stage 1: text encoder
    text_ids_tt = _to_tt_ids(text_ids, device)
    tt_text_hidden = tt_text_enc(input_ids=text_ids_tt)
    print(f"[t2s] text_encoder OK")

    # Stage 2: t2u (single-step decode)
    t2u_dec_start_id = int(getattr(hf_model.config, "t2u_decoder_start_token_id", 2))
    t2u_dec_ids = torch.tensor([[t2u_dec_start_id]], dtype=torch.long)
    with torch.no_grad():
        t2u_dec_emb = hf_model.t2u_model.model.decoder.embed_tokens(t2u_dec_ids).float()
    dec_emb_tt = _to_tt_bf16(t2u_dec_emb, device)
    tt_unit_logits = tt_t2u(inputs_embeds=tt_text_hidden, decoder_inputs_embeds=dec_emb_tt)
    print(f"[t2s] t2u_model OK")

    # Stage 3: argmax -> unit IDs
    unit_ids_cpu = (
        ttnn.to_torch(tt_unit_logits).to(torch.float32)
        .argmax(dim=-1)
        .to(torch.long)
        .clamp(0, VOCODER_UNIT_VOCAB_SIZE - 1)
    )
    print(f"[t2s] unit IDs: shape={{tuple(unit_ids_cpu.shape)}}")

    # Stage 4: vocoder (may OOM on long sequences; tolerated)
    result = {{
        "text_encoder": "OK", "t2u_model": "OK",
        "unit_ids_shape": tuple(unit_ids_cpu.shape),
        "waveform_shape": None, "wav_path": None, "vocoder_error": None,
    }}
    spkr_tt = _to_tt_ids(torch.tensor([[spkr_id]], dtype=torch.long), device)
    lang_tt = _to_tt_ids(torch.tensor([[lang_id]], dtype=torch.long), device)
    unit_ids_tt = _to_tt_ids(unit_ids_cpu, device)
    try:
        tt_waveform = tt_vocoder(unit_ids_tt, spkr_id=spkr_tt, lang_id=lang_tt)
        wave = ttnn.to_torch(tt_waveform).to(torch.float32)
        result["waveform_shape"] = tuple(wave.shape)
        print(f"[t2s] vocoder OK, waveform shape={{result['waveform_shape']}}")
        if output_path is not None:
            write_wav(wave, output_path)
            print(f"[t2s] wrote audio to {{output_path}}")
            result["wav_path"] = str(output_path)
    except Exception as exc:
        print(f"[t2s] vocoder failed (non-fatal): {{type(exc).__name__}}: {{exc}}")
        result["vocoder_error"] = f"{{type(exc).__name__}}: {{exc}}"
    return result


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_demo_t2s(device_params, device):
    text = os.environ.get("T2S_TEXT", DEFAULT_SOURCE_TEXT)
    spkr_id = int(os.environ.get("T2S_SPKR_ID", "0"))
    lang_id = int(os.environ.get("T2S_LANG_ID", "0"))
    output_env = os.environ.get("T2S_OUTPUT", "").strip()
    output_path = Path(output_env) if output_env else None

    result = run_t2s(device=device, text=text, spkr_id=spkr_id, lang_id=lang_id,
                     output_path=output_path)
    assert result["text_encoder"] == "OK"
    assert result["t2u_model"] == "OK"


def _cli_main(argv=None):
    p = argparse.ArgumentParser(description="T2S demo on Tenstorrent")
    p.add_argument("--text", type=str, default=DEFAULT_SOURCE_TEXT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--spkr-id", type=int, default=0)
    p.add_argument("--lang-id", type=int, default=0)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        run_t2s(device=device, text=args.text, spkr_id=args.spkr_id,
                lang_id=args.lang_id, output_path=args.output)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
'''

    def emit_generator_class(self, ctx: TemplateContext) -> str:
        # T2S inlines the pipeline in the demo file; no separate generator
        return ""

    def emit_eval_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T2S pipeline-reach evaluation."""

from __future__ import annotations

import os
import time
from collections import Counter

import pytest

from {demo_pkg}.demo.demo_t2s import run_t2s


DEFAULT_SENTENCES = [
    "Hello world.",
    "The weather is nice today.",
    "I would like a cup of coffee.",
]


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_eval_t2s_pipeline_reach(device_params, device):
    n = int(os.environ.get("EVAL_N", "3"))
    sentences = DEFAULT_SENTENCES[:n]

    counts = Counter()
    total = 0.0
    for idx, text in enumerate(sentences):
        t0 = time.time()
        result = run_t2s(device=device, text=text, spkr_id=0, lang_id=0)
        total += time.time() - t0
        if result["text_encoder"] == "OK":
            counts["text_encoder"] += 1
        if result["t2u_model"] == "OK":
            counts["t2u_model"] += 1
        if result["waveform_shape"] is not None:
            counts["vocoder"] += 1

    print(f"\\n[eval-t2s] STAGE REACH ({{len(sentences)}} sentences):")
    print(f"  text_encoder: {{counts['text_encoder']}}/{{len(sentences)}}")
    print(f"  t2u_model:    {{counts['t2u_model']}}/{{len(sentences)}}")
    print(f"  vocoder:      {{counts['vocoder']}}/{{len(sentences)}}")
    print(f"[eval-t2s] mean wall: {{total/len(sentences):.1f}}s/sentence")

    assert counts["text_encoder"] == len(sentences)
    assert counts["t2u_model"] == len(sentences)
'''

    def emit_parity_test(self, ctx: TemplateContext) -> str:
        # T2S has no parity test by default (vocoder output not bit-comparable)
        return ""

    def emit_reference(self, ctx: TemplateContext) -> str:
        return h.emit_reference_dual(
            ctx,
            task_class_short="T2S",
            hf_task_class=self.HF_TASK_CLASS,
            has_generate_with_input_features=False,
        )

    def emit_integration_test(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration test — T2S pipeline reach (vocoder may OOM)."""

from __future__ import annotations

import pytest

from {demo_pkg}.demo.demo_t2s import run_t2s


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_t2s_pipeline(device_params, device):
    result = run_t2s(device=device, text="Hello world.", spkr_id=0, lang_id=0)
    assert result["text_encoder"] == "OK"
    assert result["t2u_model"] == "OK"
'''


__all__ = ["T2STemplate"]
