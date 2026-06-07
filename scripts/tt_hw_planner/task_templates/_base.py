# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TaskTemplate abstract base class for the emit-e2e demo emitter.

Every task category (LLM, ASR, S2TT, T2T, T2S, VLM, diffusion,
classification, segmentation, time-series) implements this contract.
The orchestrator dispatches to a concrete TaskTemplate based on the
HF AutoModel class detected from the model's config.

Each ``emit_*`` method returns Python source code as a STRING. The
orchestrator writes that string to disk. No file I/O happens inside
template implementations -- pure functions that take a context and
produce source. This keeps templates testable and deterministic.

The LLM step does NOT fire inside templates. It fires only in the
post-emit validator (``_cli_helpers/demo_validator.py``) when output
diverges from HF golden -- the iter-fix loop then patches the
already-emitted file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── INPUT/OUTPUT MODALITY ENUMS (string literals — keep simple) ────

INPUT_AUDIO = "audio"
INPUT_IMAGE = "image"
INPUT_TEXT = "text"
INPUT_VIDEO = "video"
INPUT_CSV = "csv"

OUTPUT_TEXT = "text"
OUTPUT_AUDIO = "audio"
OUTPUT_IMAGE = "image"
OUTPUT_SCALAR = "scalar"


# ─── COMPOSITION TREE (built by composition_tree.py, consumed by templates) ──


@dataclass(frozen=True)
class CompositionTree:
    """Map of graduated TTNN stubs to their HF model attribute paths
    AND to abstract task roles.

    Built by ``_cli_helpers/composition_tree.py`` for each model.
    Templates read:
      * ``stub_attributes`` to know how to wire ``hf_model.X.Y``
        into generated source code.
      * ``roles`` to look up which graduated stub plays which abstract
        role (e.g. "audio_encoder", "text_decoder") — keys are role
        names, values are the clean stub name from dynamic discovery.

    Both maps are populated by composition-tree extraction. Templates
    NEVER hardcode role-to-name mappings — they always lookup via
    ``ctx.composition_tree.roles[ROLE]``.
    """

    model_id: str
    task_class: str  # e.g. "AutoModelForSpeechSeq2Seq"
    stub_attributes: Dict[str, str]  # {"speech_encoder": "model.speech_encoder", ...}
    roles: Dict[str, str] = field(default_factory=dict)
    """Abstract role -> clean stub name. e.g.:
       "audio_encoder" -> "speech_encoder"
       "text_decoder"  -> "decoder"
       "text_encoder"  -> "encoder"
       "t2u_model"     -> "text_to_unit_for_conditional_generation"
       "vocoder"       -> "code_hifi_gan"
    """
    cpu_bridges: List[str] = field(default_factory=list)
    multi_task_heads: List[str] = field(default_factory=list)


# ─── QUIRK (one entry from quirk_database.yaml) ──────────────────────


@dataclass(frozen=True)
class Quirk:
    """A model-family-specific bug + workaround.

    Used by templates to inject defensive code (dtype casts,
    EOS-skip guards, language-token resolution, etc.).
    """

    name: str  # short identifier, e.g. "skip_first_eos"
    rationale: str  # human description for the comment
    applies_to_template: List[str]  # task template names that should honor it


# ─── TEMPLATE CONTEXT (passed to every emit_*) ───────────────────────


@dataclass
class TemplateContext:
    """All inputs a template needs to emit a complete demo.

    Built by the orchestrator from probe + composition_tree + quirk DB
    + bringup_status.json + HF config + tokenizer.
    """

    model_id: str  # "facebook/hf-seamless-m4t-medium"
    composition_tree: CompositionTree
    graduated_stubs: List[Dict[str, Any]]  # parsed bringup_status.json entries
    quirks: List[Quirk]  # quirks detected for this model
    hf_config: Dict[str, Any]  # AutoConfig.from_pretrained dict
    demo_dir: Path  # output target, e.g. models/demos/<slug>/

    # Optional — populated when known
    tokenizer_name: Optional[str] = None
    feature_extractor_name: Optional[str] = None
    processor_name: Optional[str] = None


# ─── EMITTED FILES (return type of TaskTemplate.emit_all) ────────────


@dataclass
class EmittedFiles:
    """Source code for every file the template wants written.

    Keys are paths relative to ``demo_dir``. Values are Python source
    (or other text). The orchestrator writes these to disk and applies
    smart-merge overwrite policy.
    """

    files: Dict[Path, str] = field(default_factory=dict)

    def add(self, rel_path: str, source: str) -> None:
        """Helper: add a file with a string-path key."""
        self.files[Path(rel_path)] = source


# ─── THE CONTRACT ────────────────────────────────────────────────────


class TaskTemplate(ABC):
    """Abstract base for all task-class templates.

    Concrete subclasses live in ``s2tt_template.py``, ``llm_template.py``,
    etc. Each declares its modalities + eval metric + the HF AutoModel
    class it serves, then implements the ``emit_*`` methods to produce
    deterministic source code from a ``TemplateContext``.
    """

    INPUT_MODALITY: str = ""  # e.g. INPUT_AUDIO
    OUTPUT_MODALITY: str = ""  # e.g. OUTPUT_TEXT
    HF_TASK_CLASS: str = ""  # e.g. "AutoModelForSpeechSeq2Seq"
    EVAL_METRIC: str = ""  # e.g. "wer" | "bleu" | "fid" | ...
    TASK_NAME: str = ""  # short slug, e.g. "s2tt"; used in filenames

    @abstractmethod
    def emit_demo_file(self, ctx: TemplateContext) -> str:
        """Source for ``demo/demo_<task>.py``.

        Must produce a file that supports BOTH:
          * ``pytest demo_<task>.py`` (a ``test_demo_<task>`` function)
          * ``python -m models.demos.<dir>.demo.demo_<task> --<args>``
            (an argparse ``_cli_main()`` entry point)

        Includes a smoke fallback (synthetic input when no real input
        is provided) so CI can run without committed test data.
        """

    @abstractmethod
    def emit_generator_class(self, ctx: TemplateContext) -> str:
        """Source for ``tt/generator.py``.

        Encapsulates the task-specific inference pattern:
          * autoregressive decode loop for seq2seq tasks
          * scheduler loop for diffusion
          * single forward for classification
          * encoder + mask decoder for segmentation
          * encoder + decoder forecast for time-series

        Materializes language tokens, EOS guards, dtype casts at
        ``build()`` time (not per-step) for performance.
        """

    @abstractmethod
    def emit_eval_file(self, ctx: TemplateContext) -> str:
        """Source for ``evaluation/eval_<task>.py``.

        Runs the task-appropriate metric (WER/BLEU/FID/Top-K/etc.)
        against a real dataset (FLEURS/FLORES/LibriSpeech/...) with
        an offline fallback so CI can pass without network.
        """

    @abstractmethod
    def emit_parity_test(self, ctx: TemplateContext) -> str:
        """Source for ``tests/test_hf_parity.py``.

        Compares TT generator output to HF's golden ``model.generate()``
        using BOTH ``chrF`` and word-set Jaccard ``token_overlap`` as
        complementary metrics. Catches argmax/decode/EOS divergence
        that PCC cannot see.
        """

    @abstractmethod
    def emit_reference(self, ctx: TemplateContext) -> str:
        """Source for ``reference/torch_reference.py``.

        Emits BOTH classes:
          * ``TorchReferenceX`` — CPU greedy mirror of the TT decode
            (token-by-token a/b comparison for debugging drift)
          * ``HFGoldenX`` — wrapper around HF's task-class
            ``AutoModelFor*.generate()`` (the true production baseline)
        """

    @abstractmethod
    def emit_integration_test(self, ctx: TemplateContext) -> str:
        """Source for ``tests/test_demo.py``.

        Calls ``assert_not_degenerate`` (unique_ratio / repeats /
        bigram_entropy) on demo output. Catches stuck-token loops,
        empty outputs, gibberish — failures PCC cannot see.
        """

    # ─── Default emit-all (composes all the above) ──────────────────

    def emit_all(self, ctx: TemplateContext) -> EmittedFiles:
        """Compose every emit_* into the full file tree for this task.

        Subclasses can override to add extra files (e.g. helpers
        that don't fit any of the standard slots), but the default
        covers the yito-style layout.
        """
        out = EmittedFiles()
        task = self.TASK_NAME
        out.add(f"demo/demo_{task}.py", self.emit_demo_file(ctx))
        out.add(f"tt/generator.py", self.emit_generator_class(ctx))
        out.add(f"evaluation/eval_{task}.py", self.emit_eval_file(ctx))
        out.add(f"tests/test_hf_parity.py", self.emit_parity_test(ctx))
        out.add(f"reference/torch_reference.py", self.emit_reference(ctx))
        out.add(f"tests/test_demo.py", self.emit_integration_test(ctx))
        return out


__all__ = [
    "INPUT_AUDIO",
    "INPUT_IMAGE",
    "INPUT_TEXT",
    "INPUT_VIDEO",
    "INPUT_CSV",
    "OUTPUT_TEXT",
    "OUTPUT_AUDIO",
    "OUTPUT_IMAGE",
    "OUTPUT_SCALAR",
    "CompositionTree",
    "Quirk",
    "TemplateContext",
    "EmittedFiles",
    "TaskTemplate",
]
