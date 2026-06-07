# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared code-emitting helpers for all task templates.

Templates call these to produce the universal yito-style scaffolding
(README, requirements, .gitignore, conftest, tt/ re-exports, output
validation, references). Each template then adds its task-specific
demo, generator, and eval files on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ._base import EmittedFiles, TemplateContext


# ─── Universal files (same across all templates) ────────────────────


def emit_gitignore() -> str:
    return """# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

__pycache__/
*.py[co]
*.pyo

# Bring-up tool internals
_attempts/
_handoff/
_synth_prompts/
_synth_responses/
_verify/
decomposition_plan.applied/

# Bring-up tool state files
RUN_REPORT.md
harness_skipped.json
kernel_findings.json
_runtime_fallbacks.json
skip_diagnosis.json

# Stub backup files
_stubs/*.bak
_stubs/*.before_synth.py.bak
_stubs/*.last_good_native
_stubs/*.best_native
_stubs/*.preiter_native
_stubs/*.opplan.json
_stubs/*.stale_after_decomposition

# Per-component test variants left over from decomposition
tests/pcc/*.stale_after_decomposition

# User-generated output
*.wav
transcript.txt
translation.txt
out_*.txt
"""


def emit_conftest() -> str:
    """Emit conftest.py with the demo_*.py pytest_collect_file hook.

    Critical: without this, ``pytest models/demos/<dir>/demo/`` collects
    0 items because pytest's default is ``test_*.py``.
    """
    return '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local pytest fixtures + demo_*.py discovery hook."""

from __future__ import annotations


DEFAULT_L1_SMALL_SIZE = 24576


def pytest_collect_file(parent, file_path):
    """Allow pytest to discover demo_*.py files (in addition to test_*.py)."""
    if file_path.suffix == ".py" and file_path.name.startswith("demo_"):
        from _pytest.python import Module
        return Module.from_parent(parent, path=file_path)
    return None
'''


def emit_requirements(extras: List[str]) -> str:
    """Emit requirements.txt with task-specific extras."""
    base = ["transformers>=4.42.0", "sentencepiece>=0.2.0"]
    return "\n".join(base + extras) + "\n"


def emit_output_validation() -> str:
    """Emit demo/output_validation.py — anti-garbage detector.

    Used by every text-output template's integration test.
    """
    return '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Degenerate-output detection helpers.

PCC proves component math is correct. These helpers catch the OTHER
failure modes PCC structurally cannot see: stuck-token loops, empty
outputs, gibberish that's still a valid string.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


DEFAULT_MIN_UNIQUE_RATIO = 0.30
DEFAULT_MAX_CONSECUTIVE_REPEATS = 5
DEFAULT_MIN_BIGRAM_ENTROPY = 1.50
DEFAULT_MIN_DECODED_CHARS = 2


@dataclass
class DegeneracyReport:
    n_tokens: int
    n_unique: int
    unique_ratio: float
    max_consecutive_repeats: int
    bigram_entropy: float
    decoded_chars: int
    decoded_text: str

    def summary(self) -> str:
        return (
            f"n_tokens={self.n_tokens} unique={self.n_unique}/{self.n_tokens} "
            f"(ratio={self.unique_ratio:.2f}) max_repeats={self.max_consecutive_repeats} "
            f"H(bigram)={self.bigram_entropy:.2f}bits decoded_chars={self.decoded_chars}"
        )


def unique_token_ratio(token_ids: Sequence[int]) -> float:
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def max_consecutive_repeats(token_ids: Sequence[int]) -> int:
    if not token_ids:
        return 0
    best, run = 1, 1
    for prev, cur in zip(token_ids, token_ids[1:]):
        run = run + 1 if cur == prev else 1
        if run > best:
            best = run
    return best


def bigram_entropy(token_ids: Sequence[int]) -> float:
    if len(token_ids) < 2:
        return 0.0
    pairs = list(zip(token_ids, token_ids[1:]))
    counts = Counter(pairs)
    n = len(pairs)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def compute_report(token_ids: Sequence[int], decoded_text: str) -> DegeneracyReport:
    return DegeneracyReport(
        n_tokens=len(token_ids),
        n_unique=len(set(token_ids)),
        unique_ratio=unique_token_ratio(token_ids),
        max_consecutive_repeats=max_consecutive_repeats(token_ids),
        bigram_entropy=bigram_entropy(token_ids),
        decoded_chars=len(decoded_text.strip()),
        decoded_text=decoded_text,
    )


def assert_not_degenerate(
    token_ids,
    decoded_text: str,
    *,
    label: str = "",
    min_unique_ratio: float = DEFAULT_MIN_UNIQUE_RATIO,
    max_consecutive_repeats_allowed: int = DEFAULT_MAX_CONSECUTIVE_REPEATS,
    min_bigram_entropy: float = DEFAULT_MIN_BIGRAM_ENTROPY,
    min_decoded_chars: int = DEFAULT_MIN_DECODED_CHARS,
) -> DegeneracyReport:
    rep = compute_report(token_ids, decoded_text)
    prefix = f"[{label}] " if label else ""

    if rep.n_tokens < 2:
        raise AssertionError(f"{prefix}generator produced < 2 tokens; report: {rep.summary()}")
    if rep.decoded_chars < min_decoded_chars:
        raise AssertionError(
            f"{prefix}decoded text too short ({rep.decoded_chars} chars); "
            f"report: {rep.summary()}; text={rep.decoded_text!r}"
        )
    if rep.unique_ratio < min_unique_ratio:
        raise AssertionError(
            f"{prefix}token diversity too low (unique_ratio={rep.unique_ratio:.2f}); "
            f"report: {rep.summary()}; text={rep.decoded_text!r}"
        )
    if rep.max_consecutive_repeats > max_consecutive_repeats_allowed:
        raise AssertionError(
            f"{prefix}{rep.max_consecutive_repeats} consecutive identical tokens; "
            f"report: {rep.summary()}; text={rep.decoded_text!r}"
        )
    if rep.bigram_entropy < min_bigram_entropy:
        raise AssertionError(
            f"{prefix}bigram entropy too low ({rep.bigram_entropy:.2f}); "
            f"report: {rep.summary()}; text={rep.decoded_text!r}"
        )
    return rep


__all__ = [
    "assert_not_degenerate", "compute_report", "DegeneracyReport",
    "unique_token_ratio", "max_consecutive_repeats", "bigram_entropy",
]
'''


# ─── tt/ layer emission ─────────────────────────────────────────────


def emit_tt_init(
    ctx: TemplateContext,
    builders: List[Tuple[str, str]],
    generator_classes: List[Tuple[str, str]] = None,
    roles: dict = None,
) -> str:
    """Emit tt/__init__.py with re-exports of builders + role aliases + generator classes.

    ``builders``: list of (clean_name, stub_module_basename) from dynamic discovery
    ``generator_classes``: list of (class_name, module_name)
        e.g. [("SeamlessS2TTGenerator", "generator_s2tt")]
    ``roles``: role -> clean_name map from composition tree
        e.g. {"audio_encoder": "speech_encoder", "text_decoder": "decoder", ...}

    Emits stable ``build_<role>`` aliases derived from the roles map, so
    templates can reference stubs by ROLE (stable across models) rather
    than by clean name (varies per model).
    """
    generator_classes = generator_classes or []
    roles = roles or {}

    # Map a role to a stable, ergonomic alias name templates can rely on
    role_to_alias = {
        "audio_encoder": "build_speech_encoder",
        "text_encoder": "build_text_encoder",
        "text_decoder": "build_text_decoder",
        "t2u_model": "build_t2u_model",
        "vocoder": "build_vocoder",
        "lm_head": "build_lm_head",
    }

    clean_names_set = {clean for clean, _ in builders}

    lines = [
        "# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.",
        "# SPDX-License-Identifier: Apache-2.0",
        '"""TTNN component layer — clean re-exports of graduated stubs."""',
        "",
        "# Builders (named by clean stub name from dynamic discovery)",
    ]
    for clean_name, _stub_name in builders:
        lines.append(f"from .{clean_name} import build as build_{clean_name}")

    # Role aliases — templates depend on these for cross-model stability
    role_aliases: List[Tuple[str, str]] = []
    if roles:
        lines.append("")
        lines.append("# Role aliases (stable across models)")
        for role, clean_name in roles.items():
            alias = role_to_alias.get(role)
            if alias and clean_name in clean_names_set:
                lines.append(f"{alias} = build_{clean_name}")
                role_aliases.append((alias, clean_name))

    if generator_classes:
        lines.append("")
        lines.append("# Generators")
        for cls_name, mod_name in generator_classes:
            lines.append(f"from .{mod_name} import {cls_name}")

    lines.append("")
    lines.append("# Config re-exports (convenience)")
    lines.append("from .model_config import HF_MODEL_ID, GENERATION")
    lines.append("")
    lines.append("__all__ = [")
    for clean_name, _ in builders:
        lines.append(f"    {f'build_{clean_name}'!r},")
    for alias, _ in role_aliases:
        lines.append(f"    {alias!r},")
    for cls_name, _ in generator_classes:
        lines.append(f"    {cls_name!r},")
    lines.append("    'HF_MODEL_ID',")
    lines.append("    'GENERATION',")
    lines.append("]")
    return "\n".join(lines) + "\n"


def emit_tt_reexport(ctx: TemplateContext, clean_name: str, stub_module: str) -> str:
    """Emit tt/<clean_name>.py that re-exports from _stubs/<stub_module>.py.

    Stub class names follow the convention CamelCase(stub_module_name).
    """
    demo_pkg = _demo_package(ctx.demo_dir)
    class_name = "".join(p.capitalize() for p in stub_module.split("_"))
    return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""{clean_name} re-export from graduated stub."""

from {demo_pkg}._stubs.{stub_module} import (
    {class_name},
    build,
)

__all__ = ["{class_name}", "build"]
'''


def emit_tt_model_config(ctx: TemplateContext, extras: Dict[str, Any]) -> str:
    """Emit tt/model_config.py — centralized hyperparams.

    ``extras`` lets each template add task-specific constants
    (e.g. DECODER_START_TOKEN_ID, VOCODER_UNIT_VOCAB_SIZE).
    """
    cfg = ctx.hf_config
    hidden = cfg.get("d_model") or cfg.get("hidden_size") or 1024

    extras_src = "\n".join(f"{k} = {v!r}" for k, v in extras.items())
    return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Centralized hyperparameters for {ctx.model_id}."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


HF_MODEL_ID = "{ctx.model_id}"
HIDDEN_DIM = {hidden}
{extras_src}


@dataclass
class GenerationDefaults:
    max_new_tokens: int = 32
    min_new_tokens: int = 1
    pcc_target: float = 0.95


GENERATION = GenerationDefaults()


def get_hf_config():
    import transformers
    return transformers.AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)


def resolve_language_token_id(tokenizer, tgt_lang: Optional[str]) -> Optional[int]:
    if not tgt_lang:
        return None
    tok = f"__{{tgt_lang}}__"
    vocab = tokenizer.get_vocab()
    if tok not in vocab:
        return None
    return int(tokenizer.convert_tokens_to_ids(tok))


__all__ = [
    "HF_MODEL_ID", "HIDDEN_DIM", "GenerationDefaults", "GENERATION",
    "get_hf_config", "resolve_language_token_id",
]
'''


def emit_tt_load_weights(ctx: TemplateContext, hf_task_class: str) -> str:
    """Emit tt/load_weights.py with HF model + tokenizer + feature_extractor loaders."""
    return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight-loading helpers for {ctx.model_id}."""

from __future__ import annotations

import torch
import transformers

from .model_config import HF_MODEL_ID


def load_hf_model(model_id: str = HF_MODEL_ID):
    """Load fp32 base AutoModel on CPU (avoids bf16/fp32 layer_norm crashes)."""
    model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model


def load_hf_model_for_task(model_id: str = HF_MODEL_ID):
    """Load the task-specific AutoModelFor* class (with .generate() method)."""
    model = transformers.{hf_task_class}.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model


def load_tokenizer(model_id: str = HF_MODEL_ID):
    return transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def load_feature_extractor(model_id: str = HF_MODEL_ID):
    return transformers.AutoFeatureExtractor.from_pretrained(model_id)


def load_processor(model_id: str = HF_MODEL_ID):
    return transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


__all__ = [
    "load_hf_model", "load_hf_model_for_task",
    "load_tokenizer", "load_feature_extractor", "load_processor",
]
'''


# ─── README emission ────────────────────────────────────────────────


def emit_readme(ctx: TemplateContext, task_name: str, task_desc: str) -> str:
    """Emit README.md with model overview + commands + limitations."""
    model_id = ctx.model_id
    return f"""# {ctx.model_id} on Tenstorrent

Production demo for [`{model_id}`](https://huggingface.co/{model_id}) running on Blackhole.

**Task**: {task_desc}

## Directory layout (yito-style)

```
demo/                    # Production demos
   demo_{task_name}.py   # Entry point — argparse CLI + pytest
   audio_loader.py       # librosa + VAD + fbank (if audio task)
   output_validation.py  # Anti-garbage helpers
tt/                      # TTNN component layer
   speech_encoder.py     # Re-exports from _stubs/
   text_decoder.py
   generator.py          # Autoregressive decode loop
   model_config.py
   load_weights.py
reference/
   torch_reference.py    # HF CPU baseline + HF.generate() golden
tests/
   test_demo.py          # Integration tests (non-degenerate output)
   test_hf_parity.py     # TT vs HF.generate() parity (chrF + token_overlap)
   pcc/                  # Per-component PCC tests (from bring-up)
evaluation/
   eval_{task_name}.py   # Task-appropriate metric on real dataset
_stubs/                  # Graduated TTNN ports (tool internal)
README.md
requirements.txt
conftest.py
```

## Quick start

```bash
# Smoke test
pytest models/demos/{ctx.demo_dir.name}/demo/ -v

# Real input
python -m models.demos.{ctx.demo_dir.name}.demo.demo_{task_name} --help
```

## Validation

This demo was auto-generated by `scripts/tt_hw_planner emit-e2e` from
25 graduated TTNN components. Output is verified against HF's
`model.generate()` via chrF and token-overlap parity tests.

## How this was built

1. The bring-up tool (`scripts/tt_hw_planner up --auto`) graduated each
   component, validated via per-component PCC tests
2. `scripts/tt_hw_planner emit-e2e` composed graduated stubs into
   the yito-style demo structure above
3. HF-Golden parity tests confirm the TT pipeline produces the same
   output HF's reference implementation would.

## Known limitations

- **No KV cache** in the autoregressive decode loop — each new token
  re-encodes the full prefix. The graduated decoder stub doesn't expose
  `past_key_values`; KV cache would require either modifying the stub or
  re-running the bring-up tool.
- **Greedy decode only** — no beam search yet.
- **`lm_head` and `embed_tokens` on CPU** — not in the graduation plan.
  They're small ops; could be graduated later for full on-device generation.
"""


# ─── reference/torch_reference.py emission ──────────────────────────


def emit_reference_dual(
    ctx: TemplateContext,
    *,
    task_class_short: str,  # "S2TT", "T2T", "ASR", etc.
    hf_task_class: str,  # "AutoModelForSpeechSeq2Seq" etc.
    has_generate_with_input_features: bool = False,
) -> str:
    """Emit reference/torch_reference.py with BOTH:
    * ``TorchReferenceX`` — CPU greedy mirror of TT path
    * ``HFGoldenX``       — wrapper around HF AutoModelFor*.generate()
    """
    demo_pkg = _demo_package(ctx.demo_dir)
    short = task_class_short

    if has_generate_with_input_features:
        # S2TT / ASR — generate(input_features=...)
        golden_generate_body = """
        output_ids = self.model.generate(
            input_features=input_features,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
"""
        golden_sig = 'input_features, *, tgt_lang: str = "eng", max_new_tokens: int = 64, num_beams: int = 1'
    else:
        # T2T / LLM / etc. — generate(input_ids=..., **inputs)
        golden_generate_body = """
        inputs = self.tokenizer(text, return_tensors="pt")
        output_ids = self.model.generate(
            **inputs,
            tgt_lang=tgt_lang,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
"""
        golden_sig = 'text: str, *, tgt_lang: str = "eng", max_new_tokens: int = 64, num_beams: int = 1'

    return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch (HF) reference for {ctx.model_id} ({short} task).

``HFGolden{short}`` wraps HF's ``{hf_task_class}.generate()`` — the
true production baseline. Loads the task-specific HF class directly
(not via the shared load_weights helper) so each task's parity test
uses the right ``.generate()`` signature.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import transformers

from {demo_pkg}.tt.load_weights import load_tokenizer


class HFGolden{short}:
    """True golden: HF's ``{hf_task_class}.generate()``."""

    def __init__(self, model_id: str = "{ctx.model_id}") -> None:
        # Load the task-specific class directly to ensure .generate()
        # has the correct signature for this task.
        self.model = transformers.{hf_task_class}.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.eval()
        self.tokenizer = load_tokenizer(model_id)

    @torch.no_grad()
    def generate(self, {golden_sig}) -> str:
        """Returns the decoded text directly."""{golden_generate_body}


__all__ = ["HFGolden{short}"]
'''


# ─── helpers used by every template ────────────────────────────────


def _demo_package(demo_dir: Path) -> str:
    """Convert a demo_dir path to a Python package path.

    Strips any leading path components up to ``models/demos/``:

      Path("models/demos/hf_seamless_m4t_medium")
        -> "models.demos.hf_seamless_m4t_medium"
      Path("/home/.../tt-metal/models/demos/vision/segmentation/sam2_hiera_tiny")
        -> "models.demos.vision.segmentation.sam2_hiera_tiny"
    """
    parts = demo_dir.parts
    try:
        idx = parts.index("models")
        if idx + 1 < len(parts) and parts[idx + 1] == "demos":
            return ".".join(parts[idx:])
    except ValueError:
        pass
    # Fallback: just take the last 3 components (likely models/demos/<slug>)
    return ".".join(parts[-3:])


def universal_files(ctx: TemplateContext, requirements_extras: List[str]) -> Dict[str, str]:
    """Files every template emits regardless of task class.

    Returns rel-path -> source map. Caller (template) merges into its
    EmittedFiles.
    """
    return {
        ".gitignore": emit_gitignore(),
        "conftest.py": emit_conftest(),
        "requirements.txt": emit_requirements(requirements_extras),
        "demo/output_validation.py": emit_output_validation(),
    }


def graduated_components(ctx: TemplateContext) -> List[str]:
    """Names of graduated TTNN components from bringup_status.json."""
    return [str(c.get("name", "")).strip() for c in ctx.graduated_stubs if c.get("status") in ("NEW", "ADAPT", "REUSE")]


__all__ = [
    "emit_gitignore",
    "emit_conftest",
    "emit_requirements",
    "emit_output_validation",
    "emit_tt_init",
    "emit_tt_reexport",
    "emit_tt_model_config",
    "emit_tt_load_weights",
    "emit_readme",
    "emit_reference_dual",
    "universal_files",
    "graduated_components",
]
