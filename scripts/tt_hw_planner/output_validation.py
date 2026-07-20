"""Real PCC gate for ``tt_hw_planner up --auto``.

Background — the "false green" problem
======================================
The runtime-repair loop (:mod:`runtime_repair`) closes the gap where a
fast-path Python error needs an iterative LLM patch. But there is a
*second* failure mode it does not see: the demo's pytest may exit ``0``
with a model that loaded and produced tokens, yet the tokens are
gibberish (e.g. the actual ``medgemma-4b-it`` run that produced
"posit posit posit ... ದ ದablabl ... Kern kern Kern Kern ..." after the
RoPE patches compiled). pytest says SUCCESS, the planner banner says
``OUTCOME: SUCCESS rc=0``, and the user is misled into thinking the
model works.

This module closes that gap by giving the planner a way to compare the
TT demo's actual decoded output against an HF CPU reference for the
same prompt, and reject success if the two diverge beyond a configurable
tolerance.

Design
======
The gate is intentionally split into pure functions (which the
invariant tests pin) plus one impure entry point that loads HF and
generates tokens:

  * :func:`extract_demo_user_output` — pull the
    "``==USER 0 - OUTPUT``" block out of the demo's captured stdout.
    No I/O, regex-driven. Stable across loguru's prefix formatting.
  * :func:`load_demo_first_prompt` — read the first prompt out of
    ``input_data_questions_prefill_128.json`` (the file the default
    ``test_demo_text`` parametrization actually feeds to the demo).
    The captured log only shows the head + tail of the prompt with
    ``<long prompt not printed in full>`` between them, so we cannot
    reconstruct it from the log alone.
  * :func:`compare_token_sequences` — pure token-overlap +
    repetition + non-alpha heuristic. Returns a
    :class:`ValidationResult` with ``ok``, mismatch ratio, the two
    sequences side-by-side, and a human-readable reason.
  * :func:`generate_hf_reference` — IMPURE; calls
    ``AutoModelForCausalLM`` (falling through to
    ``AutoModelForImageTextToText`` / ``AutoModel`` for VLMs), tokenizes
    the prompt, and greedy-decodes ``max_new_tokens``. Wrapped so the
    planner can call it once and reuse the result across multiple
    PCC-repair iterations.
  * :func:`build_pcc_repair_prompt` — render the focused prompt the
    LLM repair agent receives when the gate fires. Pure (kept here
    instead of in :mod:`runtime_repair` to keep the PCC vocabulary
    co-located with the comparison code).

The actual loop driver (which runs HF, calls
:func:`compare_token_sequences`, and feeds the mismatch back into
``_runtime_repair_loop``) lives in ``cli.py`` because it shares the
existing agent / heartbeat infrastructure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .discovery import safe_relative_to_root


_LOG = logging.getLogger(__name__)


DEFAULT_PROMPTS_FILE = "models/tt_transformers/demo/sample_prompts/" "input_data_questions_prefill_128.json"


DEFAULT_COMPARE_TOKENS = 32


DEFAULT_MISMATCH_TOLERANCE = 0.7


DEFAULT_MAX_REPEAT_RATIO = 0.5


DEFAULT_MAX_NON_ASCII_RATIO = 0.25


DEFAULT_HF_TOKEN_TIMEOUT_S = 60


_LOGURU_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[\.,]\d+)?\s+\|\s+" r"[A-Z]+\s+\|\s+")


_USER_OUTPUT_RE = re.compile(
    r"^==USER\s+(?P<idx>\d+)\s+-\s+OUTPUT\s*$",
    re.MULTILINE,
)


@dataclass
class ValidationResult:
    """Outcome of a PCC comparison.

    Mirrors :class:`runtime_repair.TracebackInfo` in spirit: a single
    dataclass the planner can pass to the agent if the gate fires.
    """

    ok: bool
    reason: str
    tt_text: str = ""
    hf_text: str = ""
    tt_token_ids: List[int] = field(default_factory=list)
    hf_token_ids: List[int] = field(default_factory=list)
    mismatch_count: int = 0
    mismatch_ratio: float = 0.0
    max_repeat_ratio: float = 0.0
    non_ascii_ratio: float = 0.0
    compared_tokens: int = 0

    def summary(self) -> str:
        return (
            f"PCC gate: ok={self.ok} "
            f"reason={self.reason!r} "
            f"mismatch={self.mismatch_count}/{self.compared_tokens} "
            f"({self.mismatch_ratio:.0%}) "
            f"repeat={self.max_repeat_ratio:.0%} "
            f"non_ascii={self.non_ascii_ratio:.0%}"
        )


def extract_demo_user_output(
    captured_output: str,
    user_idx: int = 0,
) -> Optional[str]:
    """Pull the "``==USER {user_idx} - OUTPUT``" block out of the
    demo's captured stdout.

    Returns the joined text (without the loguru prefixes) or ``None``
    if the marker is not found. Empty string is a legitimate result
    (means the demo produced 0 decoded tokens, which is itself a
    fail), so callers should distinguish ``None`` from ``""``.

    The demo emits the output block as a single ``logger.info`` call
    whose payload is ``\\n==REPEAT BATCH N\\n==USER i - PROMPT\\n<short
    prompt>\\n==USER i - OUTPUT\\n<text_after_prompt.strip()>\\n``.
    loguru wraps that into a series of stdout lines:

        2026-05-23 20:52:11.450 | INFO | ... -
        ==REPEAT BATCH 0
        ==USER 0 - PROMPT
        <prompt head>
        <long prompt not printed in full>
        <prompt tail>
        ==USER 0 - OUTPUT
        <decoded text on one line>

    so we anchor on ``==USER {idx} - OUTPUT`` and read forward until
    we hit either (a) the next ``==USER ... -`` marker, (b) a line
    with a fresh loguru prefix, or (c) end-of-blob.
    """
    if not captured_output:
        return None

    matches = list(_USER_OUTPUT_RE.finditer(captured_output))
    target = None
    for m in matches:
        if int(m.group("idx")) == user_idx:
            target = m
            break
    if target is None:
        return None

    start = target.end()

    if start < len(captured_output) and captured_output[start] == "\n":
        start += 1

    out_lines: List[str] = []
    for line in captured_output[start:].splitlines():
        if line.startswith("==USER ") and " - " in line:
            break
        if line.startswith("==REPEAT BATCH "):
            break
        if _LOGURU_PREFIX_RE.match(line):
            break
        out_lines.append(line)

    text = "\n".join(out_lines).rstrip()
    return text


def load_demo_first_prompt(
    repo_root: Optional[Path] = None,
    prompts_file: str = DEFAULT_PROMPTS_FILE,
) -> Optional[str]:
    """Load the first prompt out of the demo's JSON prompts file.

    The captured log only shows the first ~100 chars + last ~100 chars
    of the prompt with ``<long prompt not printed in full>`` between
    them, so the prompt cannot be reconstructed from the log alone.
    Reading it from the source-of-truth file avoids that issue and
    keeps the HF reference faithful to what the TT demo actually saw.

    Returns ``None`` if the file is missing or the JSON is malformed.
    """
    if repo_root is None:
        repo_root = _detect_repo_root()
    if repo_root is None:
        return None

    path = repo_root / prompts_file
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, list) or not data:
        return None

    first = data[0]
    if isinstance(first, dict) and "prompt" in first:
        return str(first["prompt"])
    if isinstance(first, str):
        return first
    return None


def _detect_repo_root() -> Optional[Path]:
    """Best-effort detection of the tt-metal repo root.

    When the planner runs inside an isolated worktree (env var
    TT_HW_PLANNER_BRINGUP_CWD set), return that. Otherwise walk up
    from this module's location to find ``models/tt_transformers/demo``.
    """
    from .discovery import BRINGUP_ROOT, REPO_ROOT as _CANONICAL_REPO_ROOT

    root = BRINGUP_ROOT()
    if root != _CANONICAL_REPO_ROOT and (root / "models" / "tt_transformers" / "demo").is_dir():
        return root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "models" / "tt_transformers" / "demo").is_dir():
            return parent
    return None


def _count_non_ascii_ratio(text: str) -> float:
    """Fraction of characters in ``text`` that are NOT printable ASCII
    (and not whitespace). Used as a gibberish heuristic for
    multilingual-noise cases (Kannada/Chinese/Devanagari fragments in
    the medgemma garbage)."""
    if not text:
        return 0.0
    printable = set(string.printable)
    non_ascii = sum(1 for ch in text if ch not in printable)
    return non_ascii / len(text)


def _max_repeat_ratio(tokens: Sequence[int]) -> float:
    """How often the most-common token appears in ``tokens``. A value
    near 1.0 means the model is stuck in a degenerate one-token loop;
    near 1/len(tokens) means each token is unique. HF references on
    standard prompts almost never exceed ~0.35 on the first 32 tokens.
    """
    if not tokens:
        return 0.0
    cnt = Counter(tokens)
    most = cnt.most_common(1)[0][1]
    return most / len(tokens)


def compare_token_sequences(
    tt_token_ids: Sequence[int],
    hf_token_ids: Sequence[int],
    tt_text: str,
    hf_text: str,
    *,
    compare_tokens: int = DEFAULT_COMPARE_TOKENS,
    mismatch_tolerance: float = DEFAULT_MISMATCH_TOLERANCE,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
    max_non_ascii_ratio: float = DEFAULT_MAX_NON_ASCII_RATIO,
) -> ValidationResult:
    """Pure comparison of two token sequences + their decoded text.

    Returns a :class:`ValidationResult` with ``ok=True`` iff *all* of
    the following hold for the first ``compare_tokens`` tokens:

      1. The TT output is not stuck in a single-token loop
         (most-common-token frequency below ``max_repeat_ratio``).
      2. The TT-decoded text does not contain an absurdly high
         fraction of non-printable-ASCII characters (below
         ``max_non_ascii_ratio``).
      3. The mismatch ratio against the HF reference is below
         ``mismatch_tolerance``.

    Each individual check produces a descriptive ``reason`` string so
    the repair loop can surface exactly why the gate fired.

    All thresholds are tuned against the medgemma-4b-it garbage as the
    canonical failure case; legitimate small-model outputs (e.g.
    Llama-3.2-1B answering "What is your favorite condiment?") sit
    comfortably on the passing side of every threshold.
    """

    n = min(compare_tokens, len(tt_token_ids), len(hf_token_ids))
    tt_prefix = list(tt_token_ids[:n])
    hf_prefix = list(hf_token_ids[:n])

    repeat_ratio = _max_repeat_ratio(tt_prefix)
    non_ascii_ratio = _count_non_ascii_ratio(tt_text)

    if n == 0:
        return ValidationResult(
            ok=False,
            reason=(
                "no tokens to compare: TT side produced "
                f"{len(tt_token_ids)} tokens, HF side produced "
                f"{len(hf_token_ids)} tokens"
            ),
            tt_text=tt_text,
            hf_text=hf_text,
            tt_token_ids=tt_prefix,
            hf_token_ids=hf_prefix,
            compared_tokens=0,
            max_repeat_ratio=repeat_ratio,
            non_ascii_ratio=non_ascii_ratio,
        )

    mismatch_count = sum(1 for a, b in zip(tt_prefix, hf_prefix) if a != b)
    mismatch_ratio = mismatch_count / n

    if repeat_ratio >= max_repeat_ratio:
        reason = (
            f"TT output is stuck on one token "
            f"({repeat_ratio:.0%} of the first {n} tokens are "
            f"identical); expected < {max_repeat_ratio:.0%}"
        )
        ok = False
    elif non_ascii_ratio >= max_non_ascii_ratio:
        reason = (
            f"TT output is mostly non-ASCII gibberish "
            f"({non_ascii_ratio:.0%} of characters); expected < "
            f"{max_non_ascii_ratio:.0%}"
        )
        ok = False
    elif mismatch_ratio >= mismatch_tolerance:
        reason = (
            f"TT and HF outputs disagree on "
            f"{mismatch_count}/{n} tokens "
            f"({mismatch_ratio:.0%}); expected < "
            f"{mismatch_tolerance:.0%}"
        )
        ok = False
    else:
        reason = (
            f"OK: {n - mismatch_count}/{n} tokens match "
            f"(mismatch {mismatch_ratio:.0%} < tolerance "
            f"{mismatch_tolerance:.0%}, repeat {repeat_ratio:.0%}, "
            f"non_ascii {non_ascii_ratio:.0%})"
        )
        ok = True

    return ValidationResult(
        ok=ok,
        reason=reason,
        tt_text=tt_text,
        hf_text=hf_text,
        tt_token_ids=tt_prefix,
        hf_token_ids=hf_prefix,
        mismatch_count=mismatch_count,
        mismatch_ratio=mismatch_ratio,
        max_repeat_ratio=repeat_ratio,
        non_ascii_ratio=non_ascii_ratio,
        compared_tokens=n,
    )


def gather_model_architecture_context(
    model_id: str,
    *,
    trust_remote_code: Optional[bool] = None,
) -> str:
    """Read the model's HF config and surface the architecture fields
    most relevant to the "TT decoded the wrong tokens" failure class.

    The LLM-repair agent has historically wasted iterations guessing
    at config shapes (e.g. is rope_scaling flat or nested? does this
    model use sliding window? what's the softcap?). Surfacing the raw
    config in the prompt eliminates that whole class of dead end.

    Returns a multi-line ``key: value`` block. If anything fails
    (network, gated repo, missing cache), returns a short note so the
    prompt still renders.
    """
    import os as _os

    if trust_remote_code is None:
        env = _os.environ.get("HF_TRUST_REMOTE_CODE", "").lower()
        trust_remote_code = env in ("1", "true", "yes")

    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        return f"  (could not load HF config for {model_id}: " f"{type(exc).__name__}: {exc})"

    interesting_fields = (
        "model_type",
        "architectures",
        "torch_dtype",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "rope_local_base_freq",
        "rope_traditional",
        "layer_types",
        "sliding_window",
        "sliding_window_pattern",
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "query_pre_attn_scalar",
        "attention_bias",
        "use_qk_norm",
        "rms_norm_eps",
        "tie_word_embeddings",
    )

    def _dump(cfg_obj, label: str) -> List[str]:
        out: List[str] = []
        out.append(f"  ----- {label} -----")
        present = 0
        for f in interesting_fields:
            if hasattr(cfg_obj, f):
                val = getattr(cfg_obj, f)
                if val is None:
                    continue
                present += 1
                rendered = repr(val)

                if len(rendered) > 240:
                    rendered = rendered[:237] + "..."
                out.append(f"    {f}: {rendered}")
        if present == 0:
            out.append("    (no recognised architecture fields present)")
        return out

    lines: List[str] = []

    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        lines.extend(_dump(cfg.text_config, "text_config (language model)"))
        lines.append("")
        lines.extend(_dump(cfg, "top-level config (multimodal wrapper)"))
    else:
        lines.extend(_dump(cfg, "config"))
    return "\n".join(lines)


def gather_tt_weight_cache_summary(
    model_id: str,
    repo_root: Optional[Path] = None,
) -> str:
    """Probe whether a TT-native weight cache exists for ``model_id``
    and return a short, human-readable summary the LLM-repair agent
    can use to decide *which* layer to patch.

    The cache lives at ``model_cache/<org>/<name>/<topology>/`` and is
    populated the first time the demo runs. Once present, the demo
    *bypasses* ``load_checkpoints.py`` and the state-dict-conversion
    code: it just ``torch.load(...)``s the pre-converted ``.bin``
    files. That means edits to load-time code (key renaming, dtype
    casts, embedding-scale wiring) are INVISIBLE to the next pytest
    run unless the cache is invalidated first.

    Surfacing this to the agent BEFORE iter 1 avoids a whole class of
    wasted "I edited load_checkpoints.py and nothing changed" cycles.
    """
    if repo_root is None:
        repo_root = _detect_repo_root()
    if repo_root is None:
        return "  (could not detect repo root; cache state unknown)"

    target = repo_root / "model_cache"
    for part in model_id.split("/"):
        target = target / part
    if not target.exists():
        return (
            f"  cache_path: {target}\n"
            f"  status:     NOT PRESENT -- load_checkpoints/state-dict\n"
            f"              conversion code WILL run on the next pytest\n"
            f"              invocation, so edits there ARE on the hot path."
        )

    try:
        topo_dirs = sorted(p.name for p in target.iterdir() if p.is_dir())
    except OSError:
        topo_dirs = []
    try:
        total_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    except OSError:
        total_size = 0
    size_gb = total_size / (1024**3)
    topo_str = ", ".join(topo_dirs[:6]) if topo_dirs else "(none)"
    if len(topo_dirs) > 6:
        topo_str += f", ... (+{len(topo_dirs) - 6} more)"
    return (
        f"  cache_path: {target}\n"
        f"  status:     PRESENT ({size_gb:.1f} GB across topologies: "
        f"{topo_str})\n"
        f"  ===> KEY IMPLICATION: the demo loads these .bin files\n"
        f"       directly and BYPASSES load_checkpoints.py /\n"
        f"       state-dict renaming / weight-conversion code. If you\n"
        f"       edit a weight-loading helper expecting it to change\n"
        f"       the output, IT WILL NOT (the cache shadows it). Focus\n"
        f"       on RUNTIME code (attention/RoPE/softcap/lm_head) or\n"
        f"       request a cache invalidation by editing a load-time\n"
        f"       file AND noting it in your reply -- the PCC loop\n"
        f"       will invalidate the cache automatically when the\n"
        f"       verdict doesn't move between iterations."
    )


def gather_backend_file_paths(
    repo_root: Optional[Path] = None,
    *,
    backend_dir: str = "models/tt_transformers/tt",
    max_files: int = 60,
) -> str:
    """Produce a compact list of the backend's source files so the
    LLM-repair agent doesn't have to ``find . -name '*.py'`` itself.

    Helps with the "agent edited the wrong file" failure mode by
    putting the actual file surface (and a one-line description per
    file based on its top-of-file docstring or class names) in front
    of the agent.
    """
    if repo_root is None:
        repo_root = _detect_repo_root()
    if repo_root is None:
        return "  (could not detect repo root)"

    base = repo_root / backend_dir
    if not base.is_dir():
        return f"  (backend directory not found: {backend_dir})"

    files = sorted(p for p in base.glob("*.py") if not p.name.startswith("_"))
    if not files:
        return f"  (no .py files in {backend_dir})"

    lines: List[str] = [f"  ----- {backend_dir}/ files -----"]
    for f in files[:max_files]:
        rel = safe_relative_to_root(f)

        desc = ""
        try:
            with f.open("r", encoding="utf-8") as fh:
                head = "".join(fh.readline() for _ in range(60))
        except OSError:
            head = ""
        if '"""' in head:
            try:
                docstring = head.split('"""', 2)[1].strip()
                if docstring:
                    desc = docstring.splitlines()[0][:80]
            except IndexError:
                pass
        if not desc:
            for line in head.splitlines():
                stripped = line.strip()
                if stripped.startswith("class ") and ":" in stripped:
                    desc = stripped.split(":", 1)[0]
                    break
        if desc:
            lines.append(f"    {rel}  -  {desc}")
        else:
            lines.append(f"    {rel}")
    if len(files) > max_files:
        lines.append(f"    ... ({len(files) - max_files} more files truncated)")
    return "\n".join(lines)


def build_pcc_repair_prompt(
    *,
    model_id: str,
    result: ValidationResult,
    prompt: str,
    iter_idx: int,
    max_iters: int,
    previous_attempts: Optional[List[str]] = None,
    model_config_block: Optional[str] = None,
    backend_files_block: Optional[str] = None,
    weight_cache_block: Optional[str] = None,
) -> str:
    """Render the focused prompt sent to the LLM repair agent when
    the PCC gate fires (i.e. ``cmd_prepare`` exited 0 but the output
    is garbage).

    Sibling of :func:`runtime_repair.build_repair_prompt`. The
    semantic difference is critical: there is no Python traceback to
    point at, so the prompt instead tells the agent that the model
    *runs* but produces wrong tokens, and lists the suspects (RoPE,
    scale factors, KV-cache layout, attention mask shape).

    Kept pure (string in, string out) so the template can be diffed
    in tests.
    """
    previous_attempts = previous_attempts or []
    prev_block = ""
    if previous_attempts:
        prev_block = (
            "\nPREVIOUS REPAIR ATTEMPTS (each tried in an earlier "
            "iteration of this loop and did NOT close the PCC gap; "
            "do not repeat the same fix):\n"
        )
        for i, p in enumerate(previous_attempts, 1):
            prev_block += f"  attempt {i}: {p}\n"

    tt_preview = (result.tt_text or "")[:200].replace("\n", " ")
    hf_preview = (result.hf_text or "")[:200].replace("\n", " ")

    config_block = (
        model_config_block if model_config_block is not None else "  (HF config not provided to prompt builder)"
    )
    files_block = (
        backend_files_block
        if backend_files_block is not None
        else "  (backend file list not provided to prompt builder)"
    )
    cache_block = (
        weight_cache_block if weight_cache_block is not None else "  (cache state not provided to prompt builder)"
    )

    return (
        f"You are an automated PCC-repair agent for the `tt_hw_planner`\n"
        f"model bring-up tool. The user ran:\n"
        f"    python -m scripts.tt_hw_planner up {model_id} --auto ...\n"
        f"\n"
        f"The demo pytest exited 0 (no Python error), but the model's\n"
        f"actual decoded output is wrong: it does not match the HF\n"
        f"reference for the same prompt with the same decoding params\n"
        f"(greedy, temperature=0). This is iteration {iter_idx}/{max_iters}\n"
        f"of the PCC-repair loop.\n"
        f"\n"
        f"PROMPT (first 200 chars)\n"
        f"------------------------\n"
        f"{prompt[:200]}\n"
        f"\n"
        f"TT-DEMO OUTPUT (first 200 chars)\n"
        f"--------------------------------\n"
        f"{tt_preview}\n"
        f"\n"
        f"HF-REFERENCE OUTPUT (first 200 chars; greedy CPU)\n"
        f"-------------------------------------------------\n"
        f"{hf_preview}\n"
        f"\n"
        f"GATE VERDICT\n"
        f"------------\n"
        f"  {result.reason}\n"
        f"  compared {result.compared_tokens} tokens; mismatch "
        f"{result.mismatch_count} ({result.mismatch_ratio:.0%})\n"
        f"  repeat ratio {result.max_repeat_ratio:.0%}, non_ascii "
        f"{result.non_ascii_ratio:.0%}\n"
        f"\n"
        f"MODEL CONFIG (read directly from HF AutoConfig)\n"
        f"-----------------------------------------------\n"
        f"{config_block}\n"
        f"\n"
        f"BACKEND FILE SURFACE (where the bug almost certainly lives)\n"
        f"-----------------------------------------------------------\n"
        f"{files_block}\n"
        f"\n"
        f"TT-NATIVE WEIGHT CACHE STATE (read this before editing\n"
        f"load_checkpoints / state-dict-conversion code)\n"
        f"-----------------------------------------------------------\n"
        f"{cache_block}\n"
        f"{prev_block}"
        f"\n"
        f"YOUR JOB\n"
        f"--------\n"
        f"The model loaded, ran, and decoded tokens, but the tokens are\n"
        f"semantically wrong. The Python-level error was already fixed\n"
        f"in a previous repair pass (otherwise we wouldn't be here);\n"
        f"the remaining bug is *numerical* or *layout*. CROSS-REFERENCE\n"
        f"the MODEL CONFIG block above with the BACKEND FILE SURFACE\n"
        f"block to find which file(s) implement each feature, then\n"
        f"patch the first suspect that explains the divergence:\n"
        f"\n"
        f"  1. Per-layer attention-type dispatch (HIGHEST PRIORITY).\n"
        f"     Check the config for `layer_types`, `sliding_window`,\n"
        f"     or `sliding_window_pattern`. If present, this model\n"
        f"     alternates between full-attention and sliding-window\n"
        f"     layers per the schedule in the config. The backend was\n"
        f"     likely written for a uniform-attention sibling. Look at\n"
        f"     `model.py` (the transformer-block construction loop) and\n"
        f"     `attention.py` (whether the per-layer attention kernel\n"
        f"     branches on layer type). If it doesn't, the KV cache\n"
        f"     layout is subtly wrong from layer 1 onward.\n"
        f"  2. RoPE freq dispatch. Check the config for `rope_scaling`\n"
        f"     and `rope_local_base_freq`. Gemma-3-family models use\n"
        f"     DIFFERENT rope_theta for full-attention vs sliding-window\n"
        f"     layers. Look at `rope.py` and `common.py::rope_scaling_*`\n"
        f"     to confirm both freq tables are precomputed AND that the\n"
        f"     correct one is selected per layer at runtime.\n"
        f"  3. Q/K-norm. Check the config for `use_qk_norm` or weights\n"
        f"     named `*.attention.q_norm.weight` / `k_norm.weight` in\n"
        f"     the tensor cache list. If those weights exist but the\n"
        f"     attention kernel doesn't apply them BEFORE the QK^T\n"
        f"     matmul, attention scores are noise.\n"
        f"  4. Final logit softcap. Check `final_logit_softcapping`\n"
        f"     and `attn_logit_softcapping` in the config. If they're\n"
        f"     non-null and the backend doesn't apply\n"
        f"     `tanh(x / cap) * cap` at the LM-head / attention-score\n"
        f"     step, the argmax will prefer high-frequency tokens (the\n"
        f"     'posit posit posit' / 'kern kern kern' signature is\n"
        f"     exactly what unsoftcapped logits look like).\n"
        f"  5. Attention mask / sliding window size. An off-by-one in\n"
        f"     window construction or a missing causal mask will\n"
        f"     produce output that looks fine for the first 1-2 tokens\n"
        f"     then diverges.\n"
        f"\n"
        f"DIAGNOSTIC HINTS FROM THE GATE VERDICT\n"
        f"--------------------------------------\n"
        f"  * If `repeat ratio` is very high ({result.max_repeat_ratio:.0%} here),\n"
        f"    one token dominates -- this is the LOGIT SOFTCAP / Q-K-NORM\n"
        f"    signature. Argmax is degenerate.\n"
        f"  * If `non_ascii` is high, the embedding/decoder is producing\n"
        f"    out-of-distribution token ids -- probably ATTENTION TYPE\n"
        f"    DISPATCH (wrong KV cache layout per layer).\n"
        f"  * If `repeat ratio` is moderate and `mismatch` is high (no\n"
        f"    one-token dominance, just wrong tokens), the most common\n"
        f"    cause is ROPE WRONG-FREQ-PER-LAYER.\n"
        f"\n"
        f"BEFORE you exit, run a sanity grep to verify your edit is on\n"
        f'the hot path: `grep -rn "def forward" {{file_you_edited}}`\n'
        f"and confirm the function is called from the demo's prefill /\n"
        f"decode flow. If you can't find the call site, you edited the\n"
        f"wrong file -- the demo will then produce byte-identical\n"
        f"output and the next iteration will catch you.\n"
        f"\n"
        f"BUDGET + COMMIT RULE (READ THIS FIRST)\n"
        f"--------------------------------------\n"
        f"You have a hard wall-clock budget of ~25 minutes per\n"
        f"iteration. The previous PCC-repair attempt in this loop\n"
        f"frequently consumed the entire budget on Read/Grep tool\n"
        f"calls without making a single edit, then was wall-clock\n"
        f"killed. That is a TOOL FAILURE, not a clean exit. To\n"
        f"prevent it:\n"
        f"\n"
        f"  * Spend AT MOST the first 30% of your budget reading\n"
        f"    files. After that point you must START EDITING even\n"
        f"    if you are not 100% confident about the root cause.\n"
        f"  * A partial / wrong fix is STILL better than 0 edits.\n"
        f"    The next iteration sees your diff and learns from it.\n"
        f"    Zero edits is the worst possible outcome -- the next\n"
        f"    iteration's prompt will explicitly call you out for it.\n"
        f"  * Before exiting, you MUST have made at least one\n"
        f"    `Edit` / `Write` / `MultiEdit` tool call against a\n"
        f"    file under `models/`. If you cannot identify which\n"
        f"    file to edit, edit the most-likely suspect from the\n"
        f"    checklist above with an instrumented `print()` so the\n"
        f"    next iteration has DATA to work with rather than\n"
        f"    speculation.\n"
        f"\n"
        f"HARD CONSTRAINTS\n"
        f"----------------\n"
        f"  1. Do NOT edit any file under:\n"
        f"        scripts/tt_hw_planner/    (the planner tool)\n"
        f"        ttnn/ttnn/                (generated bindings)\n"
        f"        python_env/               (venv site-packages)\n"
        f"        tests/, conftest.py       (test infrastructure)\n"
        f"        pytest.ini, pyproject.toml\n"
        f"  2. Do NOT add or expand CPU-only fallbacks. The model is\n"
        f"     supposed to run on TT. A CPU fallback would close this\n"
        f"     gate without actually fixing the problem.\n"
        f"  3. Do NOT regress already-supported models. Any helper you\n"
        f"     change must produce the same numbers for the previous\n"
        f"     inputs.\n"
        f"  4. Prefer minimal, surgical patches. If you can express the\n"
        f"     fix as a 10-line change in one file, do that. A large\n"
        f"     refactor is almost certainly the wrong tool here.\n"
        f"  5. Reply with the SHORTEST sentence describing what you\n"
        f"     changed; the loop driver echoes that into a one-line\n"
        f"     audit log. The actual diff goes via your file-edit tool\n"
        f"     calls.\n"
    )


@dataclass
class _HFRefOutput:
    """Container for the result of :func:`generate_hf_reference`."""

    token_ids: List[int]
    text: str
    truncated: bool = False

    step0_logits: Optional[Any] = None


def generate_hf_reference(
    model_id: str,
    prompt: str,
    *,
    max_new_tokens: int = DEFAULT_COMPARE_TOKENS,
    instruct: bool = True,
    per_token_timeout_s: int = DEFAULT_HF_TOKEN_TIMEOUT_S,
    trust_remote_code: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    return_logits: bool = False,
) -> _HFRefOutput:
    """Run the same prompt through HF on CPU greedy and return the
    generated token ids + decoded text.

    This is the only function in this module that touches HF; tests
    monkey-patch it. The function tries the most common model class
    cascade:

      1. ``AutoModelForCausalLM`` — pure text LMs.
      2. ``AutoModelForImageTextToText`` — most modern multimodal
         LMs (Gemma-3 family, Llava-Next, Idefics, etc.) that accept
         text-only input by passing only ``input_ids``.
      3. ``AutoModel`` — last-resort generic load.

    For VLMs we generate with text-only input (no images). The
    underlying transformer handles missing image inputs gracefully on
    every model we tested in 2025-2026.

    ``per_token_timeout_s`` is enforced via a ``StoppingCriteria`` that
    aborts ``generate()`` when the wall-clock per-token rate falls
    behind. On a 70 B model on CPU this prevents the gate from running
    forever; on a 4 B model it is never triggered.
    """

    import torch
    from transformers import (
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    if trust_remote_code is None:
        env = os.environ.get("HF_TRUST_REMOTE_CODE", "").lower()
        trust_remote_code = env in ("1", "true", "yes")

    load_kwargs = dict(
        trust_remote_code=trust_remote_code,
    )
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)

    model = _load_hf_for_text_generation(model_id, **load_kwargs)
    model.eval()

    if instruct and getattr(tokenizer, "chat_template", None):
        try:
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            _LOG.warning(
                "PCC gate: chat-template formatting failed (%s); " "falling back to raw prompt",
                e,
            )
            input_text = prompt
    else:
        input_text = prompt

    enc = tokenizer(input_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[-1]

    class _PerTokenTimeout(StoppingCriteria):
        def __init__(self, budget_s: int) -> None:
            self.budget_s = budget_s
            self.last_ts = time.monotonic()
            self.triggered = False

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            now = time.monotonic()
            if now - self.last_ts > self.budget_s:
                self.triggered = True
                return True
            self.last_ts = now
            return False

    stop = _PerTokenTimeout(per_token_timeout_s)
    stopping = StoppingCriteriaList([stop])

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stopping_criteria=stopping,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    )
    if return_logits:
        generate_kwargs["output_scores"] = True
        generate_kwargs["return_dict_in_generate"] = True
    with torch.inference_mode():
        out = model.generate(**generate_kwargs)

    if return_logits:
        sequences = out.sequences
        step0_scores = out.scores[0] if out.scores else None
    else:
        sequences = out
        step0_scores = None

    new_tokens = sequences[0, prompt_len:].tolist()
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    step0_logits = None
    if step0_scores is not None:
        try:
            step0_logits = step0_scores[0].detach().to(dtype=torch.float32).cpu().numpy()
        except Exception:
            step0_logits = None

    return _HFRefOutput(
        token_ids=new_tokens,
        text=decoded,
        truncated=stop.triggered,
        step0_logits=step0_logits,
    )


def _load_hf_for_text_generation(model_id: str, **load_kwargs):
    """Cascade through HF model classes until one loads.

    See :func:`generate_hf_reference` docstring for the ordering and
    rationale.
    """
    import torch
    from transformers import AutoConfig

    load_kwargs = {
        **load_kwargs,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }

    AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=load_kwargs.get("trust_remote_code", False),
    )

    last_err: Optional[Exception] = None
    for loader_name in (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModel",
    ):
        try:
            import transformers

            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue
            return loader.from_pretrained(model_id, **load_kwargs)
        except (KeyError, ValueError, OSError, AttributeError) as e:
            last_err = e
            continue

    raise RuntimeError(f"could not load {model_id!r} for HF reference generation; " f"last error: {last_err!r}")


def tokenize_text_for_compare(
    model_id: str,
    text: str,
    *,
    trust_remote_code: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> List[int]:
    """Tokenize TT-decoded text back into ids using the *same*
    tokenizer the HF reference used.

    This is the right way to compare: token-level mismatch on the same
    vocabulary is meaningful, whereas character-level mismatch is too
    sensitive to spacing/normalization. Lazily imports HF so callers
    that don't reach the PCC gate don't pay the cost.
    """
    from transformers import AutoTokenizer

    if trust_remote_code is None:
        env = os.environ.get("HF_TRUST_REMOTE_CODE", "").lower()
        trust_remote_code = env in ("1", "true", "yes")

    load_kwargs = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    tok = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    return tok(text, add_special_tokens=False)["input_ids"]
