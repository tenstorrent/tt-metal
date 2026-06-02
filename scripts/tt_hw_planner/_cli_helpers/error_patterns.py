"""Shared regex patterns + extractors for Python error tracebacks the
bring-up loop classifies.

Single source of truth for the patterns that previously lived in TWO
sibling modules with near-identical regex strings:

  - ``failure_classifier._TOOL_BUG_PATTERNS`` (used to route a failure
    to the TOOL_BUG bucket -> persist a skip rather than waste LLM iters)
  - ``cli._classify_failure`` (used to bucket pytest failures into a
    ``failure_class`` string that drives prompt routing for the next
    iteration)

Both consumers act on the same underlying error shapes (kwarg-shape
mismatch, missing state_dict key), but take different downstream actions.
Sharing the PATTERNS — not the consumer logic — preserves modularity
while removing the duplication caught during the Qwen2.5-14B bring-up
audit (2026-06-01).

Structure
---------
Two pattern groups + three extractors:

  * ``KWARG_SHAPE_REGEXES`` — patterns for "the canonical/test harness
    received the wrong constructor args" (unexpected keyword,
    missing positional, arity mismatch). Used by both consumers.
  * ``STATE_DICT_KEY_REGEX`` — pattern for KeyError on a state_dict
    lookup inside a canonical class. LLM-fixable via key remap;
    distinct from TOOL_BUG (which is unfixable harness garbage).
  * ``extract_missing_state_dict_key`` — pulls the missing key string
    out of ``KeyError: '<key>'``
  * ``extract_unexpected_kwarg`` — pulls the unexpected kwarg name out
    of ``got an unexpected keyword argument '<name>'``
  * ``extract_missing_args_description`` — pulls the comma-separated
    arg list out of ``missing N required positional arguments: '<...>'``

Generic across models — these are Python-language error shapes, not
model-specific.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Patterns for "the call site passed the wrong constructor args". These
# can fire from two causes:
#   - The test harness / canonical-wrapper template builds a call with
#     wrong args (TOOL_BUG case — unfixable by LLM iteration).
#   - The LLM wrote a wrapper that misuses the canonical's signature
#     (LLM-fixable — the next iter prompt should surface the exact
#     missing/extra kwarg and the canonical's actual signature).
KWARG_SHAPE_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"\bmissing (?:\d+ )?(?:required )?positional arg", re.IGNORECASE),
    re.compile(r"\bunexpected keyword argument", re.IGNORECASE),
    re.compile(r"\btakes \d+ positional arguments? but \d+ (?:were|was) given", re.IGNORECASE),
]

# Pattern for "stub passed state_dict to the canonical but it can't find
# a key it's looking up". Common ADAPT failure mode where HF naming
# (`q_proj.weight`, `gate_proj.weight`) doesn't match the canonical's
# Llama/Meta-style naming (`layers.N.attention.wq.weight`,
# `layers.N.feed_forward.w1.weight`). LLM-fixable in the wrapper's
# `build()` by remapping keys via `convert_hf_to_meta` (from
# `models.tt_transformers.tt.load_checkpoints`) plus a `layers.N.<module>.`
# prefix.
STATE_DICT_KEY_REGEX: re.Pattern[str] = re.compile(r"KeyError:\s*['\"]([^'\"]+)['\"]")

# Capture-only variants for extracting structured info from the same
# error shapes. Kept separate from the matching patterns above so the
# matching side stays cheap (boolean) and the extraction side can fail
# gracefully when the format is slightly different.
_UNEXPECTED_KWARG_CAPTURE = re.compile(r"got an unexpected keyword argument\s+['\"]([^'\"]+)['\"]")
_MISSING_ARGS_CAPTURE = re.compile(r"missing\s+\d+\s+required\s+(?:positional\s+)?arguments?:\s*(.+?)(?:\n|$)")

# Pattern for `TT_FATAL @ <repo>/<op>/<...>.cpp:<line>: <predicate>` style
# device-side assertion failures. These carry MORE info than generic
# "TT_FATAL_OPAQUE" — the predicate often names two tensors and an
# operator (e.g. `a.logical_shape()[-1] == gamma.value().logical_shape()[-1]`
# in layernorm). Extracting the op name + predicate lets the diagnosis
# point the LLM at the exact mismatch instead of generic
# "check reshape/transpose."
_TT_FATAL_REGEX: re.Pattern[str] = re.compile(
    r"TT_FATAL\s+@\s+(?P<path>[^:]+?\.cpp):(?P<line>\d+):\s+(?P<predicate>[^\n]+)"
)


def matches_kwarg_shape_error(text: str) -> bool:
    """Return True if the traceback contains any kwarg-shape mismatch
    signal (unexpected keyword, missing positional, arity mismatch).

    Useful for both the TOOL_BUG router (failure_classifier) and the
    prompt-routing classifier (cli._classify_failure). The semantic
    distinction (TOOL_BUG vs LLM-fixable) belongs at the call site,
    not in the pattern matcher.
    """
    if not text:
        return False
    return any(p.search(text) for p in KWARG_SHAPE_REGEXES)


def matches_state_dict_key_error(text: str) -> bool:
    """Return True if the traceback contains ``KeyError: '<key>'`` which
    indicates the canonical class is looking up a state_dict key the
    wrapper didn't provide (commonly an HF vs Meta naming mismatch)."""
    if not text:
        return False
    return STATE_DICT_KEY_REGEX.search(text) is not None


def extract_missing_state_dict_key(text: str) -> Optional[str]:
    """Pull the missing key string out of ``KeyError: '<key>'``.

    Returns ``None`` if no match (caller should fall back to a generic
    diagnosis without the specific key)."""
    if not text:
        return None
    m = STATE_DICT_KEY_REGEX.search(text)
    return m.group(1) if m else None


def extract_unexpected_kwarg(text: str) -> Optional[str]:
    """Pull the unexpected kwarg name out of
    ``got an unexpected keyword argument '<name>'``."""
    if not text:
        return None
    m = _UNEXPECTED_KWARG_CAPTURE.search(text)
    return m.group(1) if m else None


def extract_missing_args_description(text: str) -> Optional[str]:
    """Pull the comma-separated arg list out of
    ``missing N required positional arguments: '<a>', '<b>', and '<c>'``.

    Returns the raw substring (caller can split/clean as needed)."""
    if not text:
        return None
    m = _MISSING_ARGS_CAPTURE.search(text)
    return m.group(1).strip() if m else None


def matches_tt_fatal_with_predicate(text: str) -> bool:
    """Return True if the traceback contains a TT_FATAL with a parseable
    cpp-file path and an assertion predicate (not just ``TT_FATAL: false``).
    Distinct from the generic TT_FATAL_OPAQUE case."""
    if not text:
        return False
    m = _TT_FATAL_REGEX.search(text)
    if not m:
        return False
    pred = (m.group("predicate") or "").strip()
    # Reject bare "false" / "0" — those genuinely carry no info.
    return bool(pred) and pred not in ("false", "0", "true", "1")


def extract_tt_fatal_op_and_predicate(text: str) -> Optional[Dict[str, str]]:
    """Pull the operator name (from the cpp file path) and the assertion
    predicate from a ``TT_FATAL @ <path>.cpp:<line>: <predicate>`` message.

    Example input:
      ``TT_FATAL @ /home/x/ttnn/cpp/ttnn/operations/normalization/layernorm/
        device/layernorm_device_operation.cpp:80:
        a.logical_shape()[-1] == gamma.value().logical_shape()[-1]``

    Returns dict with:
      - ``op``: best-effort operator name (``"layernorm"`` from the path)
      - ``predicate``: the raw assertion text the LLM should make true
      - ``cpp_file``: full path to the failing cpp file
      - ``line``: line number in the cpp file

    Returns ``None`` if the format doesn't match."""
    if not text:
        return None
    m = _TT_FATAL_REGEX.search(text)
    if not m:
        return None
    path = m.group("path")
    line = m.group("line")
    predicate = (m.group("predicate") or "").strip()
    # Operator name heuristic: the directory two levels above the .cpp
    # file. Examples:
    #   .../operations/normalization/layernorm/device/layernorm_device_operation.cpp
    #     -> "layernorm"
    #   .../operations/matmul/device/matmul_op.cpp -> "matmul"
    #   .../operations/eltwise/binary/device/binary_device_operation.cpp -> "binary"
    op = ""
    try:
        parts = path.replace("\\", "/").split("/")
        if "operations" in parts:
            idx = parts.index("operations")
            # Walk inward until we land on a non-"device" directory name
            # right before the .cpp file.
            inner = [p for p in parts[idx + 1 :] if p and not p.endswith(".cpp")]
            if inner:
                op = inner[-1] if inner[-1] != "device" else (inner[-2] if len(inner) >= 2 else inner[-1])
    except Exception:
        op = ""
    return {
        "op": op,
        "predicate": predicate,
        "cpp_file": path,
        "line": line,
    }


# ─── HuggingFace weight download/load failures ──────────────────────
#
# When the demo can't load model weights from HF (network down, gated
# repo, corrupt cache, wrong model_id), iterating in the LLM loop will
# never help — the failure is environmental, not in the TT code. The
# tool should detect this and exit with a friendly "please download
# the weights locally" message instead of burning iterations.
#
# Categories — informational only, the user-facing exit message is
# the same: "download the weights locally and re-run":
#   - "gated"     — auth/license required (HfHubHTTPError 401/403,
#                   GatedRepoError, "gated repo", "restricted access")
#   - "not_found" — repo or file doesn't exist (RepositoryNotFoundError,
#                   404, LocalEntryNotFoundError)
#   - "network"   — connection issue (ConnectionError, ProxyError,
#                   "couldn't connect to", 5xx HTTP errors)
#   - "corrupt"   — partial / corrupted file (SafetensorError,
#                   PytorchStreamReader, EOFError, BadZipFile)
#   - "load"      — generic "OSError: Can't load …" (last-resort
#                   bucket for HF errors we didn't recognize specifically)


@dataclass(frozen=True)
class HFWeightFailure:
    """Categorized HF weight-download/load failure detected in
    captured pytest stdout. ``category`` is the bucket label,
    ``detail`` is a short human-readable description, ``excerpt`` is
    the matched line for inclusion in the user-facing message."""

    category: str
    detail: str
    excerpt: str


# Order matters: more specific patterns first. The first match wins,
# so put "GatedRepoError" before generic "Can't load weights" so a
# gated-repo failure is correctly identified.
_HF_WEIGHT_FAILURE_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    # Gated / auth failures
    ("gated", re.compile(r"\bGatedRepoError\b", re.IGNORECASE), "gated repo (license / login required)"),
    ("gated", re.compile(r"gated repo|restricted access", re.IGNORECASE), "gated repo (license / login required)"),
    (
        "gated",
        re.compile(r"401\s+Client Error|401\s+Unauthorized", re.IGNORECASE),
        "401 Unauthorized (HF token missing or invalid)",
    ),
    ("gated", re.compile(r"403\s+Client Error|403\s+Forbidden", re.IGNORECASE), "403 Forbidden (license not accepted)"),
    # Not-found / wrong-id
    ("not_found", re.compile(r"\bRepositoryNotFoundError\b", re.IGNORECASE), "repository not found on HuggingFace"),
    ("not_found", re.compile(r"\bLocalEntryNotFoundError\b", re.IGNORECASE), "local cache entry missing"),
    (
        "not_found",
        re.compile(r"404\s+Client Error|404\s+Not Found", re.IGNORECASE),
        "404 Not Found (model id or file path wrong)",
    ),
    # Corrupt / partial weights
    (
        "corrupt",
        re.compile(r"safetensors_rust\.SafetensorError|SafetensorError:\s*Error while deserializing", re.IGNORECASE),
        "corrupted .safetensors file (partial / truncated download)",
    ),
    (
        "corrupt",
        re.compile(r"PytorchStreamReader failed locating file|PytorchStreamReader failed reading", re.IGNORECASE),
        "corrupted PyTorch checkpoint (partial / truncated download)",
    ),
    (
        "corrupt",
        re.compile(r"BadZipFile|zipfile\.BadZipFile|not a zip file", re.IGNORECASE),
        "corrupted model archive (partial download)",
    ),
    # Network / connection
    (
        "network",
        re.compile(r"HfHubHTTPError|huggingface_hub\.utils\._errors\.HfHubHTTPError", re.IGNORECASE),
        "HuggingFace Hub HTTP error",
    ),
    (
        "network",
        re.compile(r"requests\.exceptions\.(?:ConnectionError|ProxyError|Timeout)", re.IGNORECASE),
        "network connection error reaching HuggingFace",
    ),
    (
        "network",
        # Match transformers' offline error: "OSError: We couldn't
        # connect to 'https://...' to load this model, ..." — the
        # URL may be quoted with ' or " or appear bare, so don't
        # require a specific delimiter after the phrase.
        re.compile(r"(?:OSError|RuntimeError):\s*(?:We\s+couldn't connect to|Couldn't reach)\b", re.IGNORECASE),
        "could not reach huggingface.co",
    ),
    (
        "network",
        re.compile(
            r"\b(?:503|504|429)\s+(?:Server\s+Error|Service\s+Unavailable|Too\s+Many\s+Requests)", re.IGNORECASE
        ),
        "HuggingFace service unavailable / rate-limited",
    ),
    # Generic load failure (LAST — fallback for HF-related load errors
    # we didn't classify more precisely above).
    (
        "load",
        re.compile(r"OSError:\s*Can't load (?:weights|tokenizer|the model|config) (?:for|from)", re.IGNORECASE),
        "model weights / tokenizer could not be loaded",
    ),
    (
        "load",
        re.compile(r"OSError:\s*Unable to load (?:weights|model)", re.IGNORECASE),
        "model weights could not be loaded",
    ),
]


def detect_hf_weight_failure(text: str) -> Optional[HFWeightFailure]:
    """Pattern-match captured pytest output for HF weight download /
    load failures. Returns the first match (specific patterns first)
    or ``None`` if no HF-failure signature is present.

    Generic across models — these are HuggingFace-library error
    shapes, not model-specific. Used by the cli helper that
    short-circuits the LLM loop with a "please download the weights
    locally" message when the failure is environmental.
    """
    if not text:
        return None
    for category, pattern, detail in _HF_WEIGHT_FAILURE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        # Find the line containing the match for a useful excerpt.
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        excerpt = text[line_start:line_end].strip()
        # Bound excerpt length so a massive line doesn't dominate output.
        if len(excerpt) > 240:
            excerpt = excerpt[:240] + "…"
        return HFWeightFailure(category=category, detail=detail, excerpt=excerpt)
    return None


def format_hf_weight_failure_message(model_id: str, failure: HFWeightFailure) -> str:
    """Render a uniform user-facing message for an HF weight failure.

    Same headline + remediation across categories — the user always
    needs to make the weights available locally — with a per-category
    addendum so the user knows whether to ``huggingface-cli login``,
    delete a corrupt cache entry, retry the network, etc.

    Every category also surfaces the two ways to point the tool at
    locally-available weights: ``HF_HOME`` (standard HF env var) or
    ``--local-dir`` (tt_hw_planner CLI flag). On rerun, HuggingFace
    will load from the local cache automatically once the weights
    are there — no extra config needed for the default cache path.
    """
    remediation_by_category = {
        "gated": (
            "  • This is a gated repo. Run `huggingface-cli login` with an HF token,\n"
            "    accept the model license at https://huggingface.co/{model_id} ,\n"
            "    then re-run this command.\n"
        ),
        "not_found": (
            "  • Check the model id spelling. If correct, the repo or a required\n"
            "    file may be missing — try downloading from the official source\n"
            "    or a known-good mirror.\n"
        ),
        "network": (
            "  • Network reach to huggingface.co failed. Pre-download the weights\n"
            "    on a connected machine and copy them into the HF cache.\n"
        ),
        "corrupt": (
            "  • The cached weights are partial or corrupted. Delete the cache\n"
            "    entry under ~/.cache/huggingface/hub/models--<org>--<name>/\n"
            "    and re-download (or copy known-good weights into place).\n"
        ),
        "load": (
            "  • HuggingFace could not load the weights. The cache may be partial,\n"
            "    the format may be unsupported, or a required file is missing.\n"
        ),
    }
    addendum = remediation_by_category.get(failure.category, "")
    # Universal local-weights hint — same wording for every category
    # so the user always knows their two options for pointing at
    # locally-available weights on rerun. Phrased after the per-category
    # remediation so category-specific guidance comes first.
    local_weights_hint = (
        "\n"
        "  On rerun, the tool will use locally-available weights automatically:\n"
        "    • Default cache (~/.cache/huggingface/hub/) is checked by HuggingFace\n"
        "      itself — no extra config needed; just re-run the same command.\n"
        "    • For a custom directory, either:\n"
        f"        export HF_HOME=/path/to/dir && python -m scripts.tt_hw_planner up {model_id} ...\n"
        f"      or pass the flag:\n"
        f"        python -m scripts.tt_hw_planner up {model_id} --local-dir /path/to/dir ...\n"
        "    • To force offline mode (skip all network checks) on rerun:\n"
        "        export HF_HUB_OFFLINE=1\n"
    )
    sep = "=" * 72
    return (
        f"\n{sep}\n"
        f"  TT_HW_PLANNER: HF weight download/load failed for {model_id}\n"
        f"{sep}\n"
        f"  Category : {failure.category}\n"
        f"  Detail   : {failure.detail}\n"
        f"  Excerpt  : {failure.excerpt}\n"
        f"\n"
        f"  Please download the weights locally and re-run this command:\n"
        f"\n"
        f"    huggingface-cli download {model_id}\n"
        f"\n"
        f"{addendum}"
        f"{local_weights_hint}"
        f"\n"
        f"  No further iteration will help until the weights load cleanly.\n"
        f"{sep}\n"
    )


# ─── Pytest "no tests collected / all deselected" detection ───────────


@dataclass(frozen=True)
class ZeroTestsCollected:
    """Pytest ran to completion but collected/selected ZERO tests.

    Two flavors:
      * ``"none_collected"`` — pytest exited with code 5 and
        ``no tests ran`` / ``no tests collected`` in output. The path
        or test id didn't resolve to anything.
      * ``"all_deselected"`` — pytest collected tests but a ``-k`` /
        ``-m`` filter (or parametrize-name mismatch) deselected all
        of them. Output reports ``N deselected, 0 selected`` /
        ``X deselected in …``.

    Both are CONFIGURATION bugs (test selector, parametrize name,
    or wrong path), not PCC failures. Surfacing them as a distinct
    outcome prevents the runtime-repair / escalation loop from
    burning iter budget trying to "fix" a model that never ran.
    """

    flavor: str  # "none_collected" or "all_deselected"
    excerpt: str  # the matched line, for the user-facing message


# Pytest's documented exit codes:
#   0 = all passed
#   1 = some failed
#   2 = interrupted (Ctrl-C / SIGINT)
#   3 = internal error
#   4 = usage error
#   5 = no tests collected
# rc=5 is the strongest signal but we also check stdout markers so a
# wrapper that rewrites the rc still triggers the short-circuit.
_PYTEST_NO_TESTS_RC = 5

_PYTEST_NO_TESTS_PATTERNS: List[Tuple[str, re.Pattern[str], str]] = [
    # "no tests ran" / "no tests collected" — rc=5 path.
    (
        "none_collected",
        re.compile(r"no tests (?:ran|collected)", re.IGNORECASE),
        "no tests collected (path or test-id resolved to nothing)",
    ),
    (
        "none_collected",
        re.compile(r"ERROR:\s*not found:", re.IGNORECASE),
        "pytest could not resolve the requested test id",
    ),
    # "N deselected, 0 selected" — typical -k/-m filter mismatch.
    # Pytest's summary line looks like "1 deselected, 1 warning in
    # 0.01s" (no "selected" word when zero matched). Catch both forms.
    (
        "all_deselected",
        re.compile(r"\b\d+\s+deselected,\s+0\s+selected\b", re.IGNORECASE),
        "every collected test was filtered out by -k / -m",
    ),
    (
        "all_deselected",
        # "1 deselected" in summary with NO matching "passed"/"failed"
        # /"error" — pytest's terse "all deselected" summary.
        re.compile(r"=+\s*\d+\s+deselected(?:,\s+\d+\s+warnings?)?\s+in\s+[\d.]+s\s*=+", re.IGNORECASE),
        "every collected test was filtered out by -k / -m",
    ),
    (
        "all_deselected",
        re.compile(r"collected \d+ items? /\s*\d+ deselected /\s*0 selected", re.IGNORECASE),
        "every collected test was filtered out by -k / -m",
    ),
]


def detect_zero_tests_collected(rc: int, text: str) -> Optional[ZeroTestsCollected]:
    """Return a :class:`ZeroTestsCollected` if pytest ran but collected
    zero tests (rc=5 or matching stdout markers), else ``None``.

    Strong signal when rc==5; otherwise we still scan stdout because
    wrapper scripts sometimes rewrite the rc but the pytest summary
    line survives.
    """
    if rc != _PYTEST_NO_TESTS_RC and not text:
        return None
    for flavor, pattern, detail in _PYTEST_NO_TESTS_PATTERNS:
        m = pattern.search(text or "")
        if not m:
            continue
        line_start = (text or "").rfind("\n", 0, m.start()) + 1
        line_end = (text or "").find("\n", m.end())
        if line_end == -1:
            line_end = len(text or "")
        excerpt = (text or "")[line_start:line_end].strip()
        if len(excerpt) > 240:
            excerpt = excerpt[:240] + "…"
        return ZeroTestsCollected(flavor=flavor, excerpt=excerpt or detail)
    # rc=5 with no matching stdout marker still counts — surface as
    # the generic flavor so the caller can short-circuit.
    if rc == _PYTEST_NO_TESTS_RC:
        return ZeroTestsCollected(
            flavor="none_collected",
            excerpt="pytest exited with rc=5 (no tests collected)",
        )
    return None


def format_zero_tests_message(model_id: str, info: ZeroTestsCollected) -> str:
    """Render the user-facing diagnostic for a zero-tests-collected outcome."""
    sep = "=" * 72
    if info.flavor == "all_deselected":
        cause = (
            "  Every collected test was filtered out by a -k / -m selector\n"
            "  (or a parametrize-name mismatch). This is a TEST-SELECTOR bug,\n"
            "  not a model bring-up failure.\n"
        )
        remedy = (
            "  Fix the -k filter or add an entry for this model to the demo's\n"
            "  parametrize list. Possible causes:\n"
            "    • the model isn't in the demo's parametrize table yet\n"
            "    • the parametrize id uses a different naming convention\n"
            "      than the -k filter (e.g. 'performance-batch-1' vs 'perf')\n"
            "    • a fixture marker (e.g. @pytest.mark.skipif) excluded it\n"
        )
    else:
        cause = (
            "  Pytest collected zero tests — the path or test id did not\n"
            "  resolve to anything. This is a TEST-DISCOVERY bug, not a\n"
            "  model bring-up failure.\n"
        )
        remedy = (
            "  Check that the demo file path exists in this worktree and\n"
            "  that the test function name (if specified) matches.\n"
        )
    return (
        f"\n{sep}\n"
        f"  TT_HW_PLANNER: zero tests selected for {model_id}\n"
        f"{sep}\n"
        f"  Flavor   : {info.flavor}\n"
        f"  Excerpt  : {info.excerpt}\n"
        f"\n"
        f"{cause}"
        f"\n"
        f"{remedy}"
        f"\n"
        f"  No iteration will help until the test actually runs.\n"
        f"{sep}\n"
    )


__all__ = [
    "KWARG_SHAPE_REGEXES",
    "STATE_DICT_KEY_REGEX",
    "matches_kwarg_shape_error",
    "matches_state_dict_key_error",
    "matches_tt_fatal_with_predicate",
    "extract_missing_state_dict_key",
    "extract_unexpected_kwarg",
    "extract_missing_args_description",
    "extract_tt_fatal_op_and_predicate",
    "HFWeightFailure",
    "detect_hf_weight_failure",
    "format_hf_weight_failure_message",
    "ZeroTestsCollected",
    "detect_zero_tests_collected",
    "format_zero_tests_message",
]
