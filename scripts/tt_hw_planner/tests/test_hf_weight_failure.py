"""Unit tests for HF weight failure detection + the cli short-circuit.

The 2026-06-02 audit added a hard short-circuit for HuggingFace weight
download / load failures: when captured pytest output contains any
recognizable HF-error signature (gated repo, network failure, corrupted
.safetensors, etc.), the tool exits with a clear "please download the
weights locally and re-run" message instead of burning the LLM iter
budget trying to fix something the code can't fix.

These tests pin:

  * Detection — pure regex matching across the five categories
    (gated / not_found / network / corrupt / load).
  * Specificity ordering — when a line could match multiple patterns,
    the more specific category wins (GatedRepoError beats the generic
    "Can't load weights" bucket).
  * Excerpt extraction — the matched line is preserved as a short
    quote in the user-facing message.
  * The cli wrapper ``_exit_if_hf_weight_failure`` returns silently on
    no-match and ``sys.exit(2)`` on match.
"""

from __future__ import annotations

import pytest

from scripts.tt_hw_planner._cli_helpers.error_patterns import (
    HFWeightFailure,
    detect_hf_weight_failure,
    format_hf_weight_failure_message,
)
from scripts.tt_hw_planner.cli import _exit_if_hf_weight_failure


# ─── Detector: empty / no-match cases ────────────────────────────────


def test_detect_returns_none_for_empty_string() -> None:
    assert detect_hf_weight_failure("") is None


def test_detect_returns_none_for_unrelated_traceback() -> None:
    """A normal Python error that isn't HF-related must NOT trip the
    detector — otherwise the tool would bail on every PCC failure."""
    text = """\
Traceback (most recent call last):
  File "test_attention.py", line 12, in test_attention
    out = tt_attention(x)
RuntimeError: shape mismatch [1, 32, 4096] vs [1, 32, 2048]
"""
    assert detect_hf_weight_failure(text) is None


def test_detect_returns_none_for_pcc_failure() -> None:
    """PCC < 0.99 is a normal failure mode handled by the iterate loop;
    must NOT trigger the HF-weight short-circuit."""
    assert detect_hf_weight_failure("PCC 0.85 below target 0.99") is None


# ─── Detector: each category fires for representative inputs ─────────


@pytest.mark.parametrize(
    "text",
    [
        "huggingface_hub.errors.GatedRepoError: Access to model X is restricted.",
        "OSError: You are trying to access a gated repo. Please log in.",
        "Cannot access gated repo for url https://huggingface.co/google/medgemma-27b",
        "requests.HTTPError: 401 Client Error: Unauthorized for url ...",
        "HfHubHTTPError: 403 Client Error: Forbidden for url ...",
    ],
)
def test_detect_gated_category(text: str) -> None:
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert f.category == "gated"


@pytest.mark.parametrize(
    "text",
    [
        "huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error",
        "LocalEntryNotFoundError: Cannot find the requested files in the disk cache",
        "404 Not Found: model/path/config.json",
    ],
)
def test_detect_not_found_category(text: str) -> None:
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert f.category == "not_found"


@pytest.mark.parametrize(
    "text",
    [
        "huggingface_hub.utils._errors.HfHubHTTPError: connection broke",
        "requests.exceptions.ConnectionError: HTTPSConnectionPool",
        "requests.exceptions.ProxyError: Cannot connect to proxy",
        "OSError: We couldn't connect to 'https://huggingface.co' to load",
        "503 Service Unavailable for url https://huggingface.co/api/models/X",
        "429 Too Many Requests for url ...",
    ],
)
def test_detect_network_category(text: str) -> None:
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert f.category == "network"


@pytest.mark.parametrize(
    "text",
    [
        "safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge",
        "SafetensorError: Error while deserializing header: invalid",
        "RuntimeError: PytorchStreamReader failed locating file data.pkl: file not found",
        "zipfile.BadZipFile: File is not a zip file",
    ],
)
def test_detect_corrupt_category(text: str) -> None:
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert f.category == "corrupt"


@pytest.mark.parametrize(
    "text",
    [
        "OSError: Can't load weights from pytorch_model.bin",
        "OSError: Can't load tokenizer for 'org/model'",
        "OSError: Unable to load weights for the model",
    ],
)
def test_detect_load_category(text: str) -> None:
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert f.category == "load"


# ─── Specificity ordering: specific categories beat generic ─────────


def test_gated_beats_generic_load_when_both_present() -> None:
    """If a traceback contains both a specific GatedRepoError and the
    generic "Can't load weights" fallback message (HuggingFace
    transformers wraps the gated error in a Can't-load OSError), the
    specific category must win — that's what tells the user to
    huggingface-cli login, not delete the cache."""
    text = """\
OSError: Can't load weights for 'org/model'. If you're trying to access a
private or gated repository, make sure you have access (GatedRepoError).
"""
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert f.category == "gated"


def test_corrupt_beats_load_when_safetensors_present() -> None:
    """SafetensorError is more actionable than the wrapped 'Can't load'
    — the user needs to delete the corrupt cache, not re-login."""
    text = """\
safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge
OSError: Can't load weights for 'org/model'.
"""
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert f.category == "corrupt"


# ─── Excerpt extraction ──────────────────────────────────────────────


def test_excerpt_captures_matched_line() -> None:
    """The excerpt should be the actual line that matched, so the user
    can copy-paste it into a search engine or bug report."""
    text = (
        "Some preamble text.\n"
        "huggingface_hub.errors.GatedRepoError: Access denied for org/model\n"
        "More trailing context.\n"
    )
    f = detect_hf_weight_failure(text)
    assert f is not None
    assert "GatedRepoError" in f.excerpt
    assert "Access denied" in f.excerpt
    # Excerpt should NOT include the preamble or trailing context.
    assert "preamble" not in f.excerpt
    assert "trailing" not in f.excerpt


def test_long_excerpt_is_bounded() -> None:
    """Defensive: a single 5000-char line shouldn't flood the message."""
    long_line = "huggingface_hub.errors.GatedRepoError: " + "x" * 5000
    f = detect_hf_weight_failure(long_line)
    assert f is not None
    assert len(f.excerpt) <= 241  # 240 chars + ellipsis


# ─── format_hf_weight_failure_message ────────────────────────────────


def test_message_includes_model_id_and_category() -> None:
    failure = HFWeightFailure(category="gated", detail="gated repo", excerpt="x")
    msg = format_hf_weight_failure_message("org/some-model", failure)
    assert "org/some-model" in msg
    assert "gated" in msg
    assert "huggingface-cli" in msg  # the remediation hint


def test_message_remediation_differs_by_category() -> None:
    """Each category should produce different remediation text so the
    user gets actionable guidance — login for gated, delete cache for
    corrupt, retry network for network."""
    for cat in ("gated", "not_found", "network", "corrupt", "load"):
        failure = HFWeightFailure(category=cat, detail="x", excerpt="x")
        msg = format_hf_weight_failure_message("org/m", failure)
        assert "Please download the weights locally" in msg
        # category-specific addendum is non-empty for the known categories
        if cat == "gated":
            assert "huggingface-cli login" in msg
        elif cat == "network":
            assert "Pre-download" in msg or "HF_HOME" in msg
        elif cat == "corrupt":
            assert "Delete" in msg or "cache" in msg


# ─── cli wrapper: _exit_if_hf_weight_failure ─────────────────────────


def test_cli_wrapper_returns_silently_on_no_match(capsys) -> None:
    """No HF failure pattern → caller proceeds with normal flow.
    Must NOT print anything (otherwise the noise would confuse users
    who hit a non-HF failure)."""
    _exit_if_hf_weight_failure("org/m", "PCC 0.85 below target 0.99")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_wrapper_exits_on_match(capsys) -> None:
    """HF failure pattern → sys.exit(2). The friendly message goes to
    stderr (so CI logs catch it) and the rc is 2 (distinguishes
    environmental bail from rc=1 PCC fail and rc=0 success)."""
    text = "huggingface_hub.errors.GatedRepoError: Access denied"
    with pytest.raises(SystemExit) as exc_info:
        _exit_if_hf_weight_failure("org/m", text)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "TT_HW_PLANNER: HF weight download/load failed" in captured.err
    assert "org/m" in captured.err
    assert "huggingface-cli" in captured.err  # remediation hint visible


def test_cli_wrapper_exits_for_each_category(capsys) -> None:
    """Smoke test across categories: exit-on-match contract holds for
    every category the detector recognizes."""
    samples = {
        "gated": "GatedRepoError: gated",
        "not_found": "RepositoryNotFoundError: 404",
        "network": "requests.exceptions.ConnectionError: network down",
        "corrupt": "safetensors_rust.SafetensorError: Error while deserializing header",
        "load": "OSError: Can't load weights from path",
    }
    for cat, text in samples.items():
        with pytest.raises(SystemExit) as exc_info:
            _exit_if_hf_weight_failure("org/m", text)
        assert exc_info.value.code == 2, f"category {cat} should exit with rc=2"
        capsys.readouterr()  # drain so next iteration's check is clean
