# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the MiniMax-H3 weights resolver: precedence, cache selection, download gating."""

"""
Written by human: this test was written to freeze behavior of the weights resolver when it was first
implementend; but the behavior/implementation was not well thought out or designed.

This test file deserves as much thought as the thought that went into the actual weight resolver,
aka NONE! Feel free to modify the weight resolution behavior and nuke this file in the future.
"""
from pathlib import Path

import pytest

from models.tt_dit.pipelines.minimax_h3 import weights_minimax_h3 as weights

REPO_DIR = "models--MiniMaxAI--MiniMax-H3"


def _make_snapshot(root: Path, *partitions: str, always: bool = True) -> Path:
    """A diffusers-style snapshot directory holding `partitions` (and, by default, the always-fetched assets)."""
    root.mkdir(parents=True, exist_ok=True)
    for name in partitions:
        (root / name).mkdir()
    if always:
        (root / "model_index.json").write_text("{}")
        for name in ("scheduler", "audio_scheduler", "tokenizer", "processor"):
            (root / name).mkdir()
    return root


def _cache_snapshot(hf_home: Path, revision: str, *partitions: str, always: bool = True, main: bool = False) -> Path:
    """A snapshot inside a HuggingFace cache rooted at `hf_home`; `main=True` also points `refs/main` at it."""
    repo = hf_home / "hub" / REPO_DIR
    snapshot = _make_snapshot(repo / "snapshots" / revision, *partitions, always=always)
    if main:
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(revision)
    return snapshot


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """No explicit path, downloads off, and the HF cache pointed at an empty directory under tmp_path."""
    monkeypatch.delenv(weights.MODEL_PATH_ENV, raising=False)
    monkeypatch.delenv(weights.ALLOW_DOWNLOAD_ENV, raising=False)
    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    return hf_home


@pytest.fixture
def fake_download(monkeypatch, tmp_path):
    """Replace `huggingface_hub.snapshot_download` with one that records its call and materializes a snapshot."""
    calls = []

    def snapshot_download(*, repo_id, allow_patterns):
        calls.append({"repo_id": repo_id, "allow_patterns": list(allow_patterns)})
        partitions = [p[: -len("/*")] for p in allow_patterns if p.endswith("/*")]
        return str(_make_snapshot(tmp_path / "downloaded", *(p for p in partitions if p not in weights._ALWAYS_DIRS)))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    return calls


def _no_download(monkeypatch):
    def snapshot_download(**kwargs):
        raise AssertionError(f"snapshot_download must not be called, got {kwargs}")

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)


# --- explicit path -------------------------------------------------------------------------------------


def test_explicit_path_wins_over_cache_and_download(isolated_env, monkeypatch, tmp_path):
    _cache_snapshot(isolated_env, "cached", "vae", main=True)
    _no_download(monkeypatch)
    explicit = _make_snapshot(tmp_path / "explicit", "vae", always=False)

    assert weights.resolve_weights_dir("vae", weights_dir=explicit, allow_download=True) == explicit


def test_env_path_is_the_explicit_path(isolated_env, monkeypatch, tmp_path):
    explicit = _make_snapshot(tmp_path / "explicit", "vae", always=False)
    monkeypatch.setenv(weights.MODEL_PATH_ENV, str(explicit))

    assert weights.resolve_weights_dir("vae") == explicit


def test_explicit_path_needs_only_the_required_partitions(isolated_env, tmp_path):
    # A partition-only directory is the caller's own layout; the always-fetched assets are not demanded of it.
    explicit = _make_snapshot(tmp_path / "vae_only", "vae", always=False)

    assert weights.resolve_weights_dir("vae", weights_dir=explicit) == explicit


def test_explicit_path_missing_partition_fails_without_downloading(isolated_env, monkeypatch, tmp_path, expect_error):
    _no_download(monkeypatch)
    explicit = _make_snapshot(tmp_path / "explicit", "vae")

    with expect_error(weights.WeightsNotFoundError, r"missing \['audio_vae'\]"):
        weights.resolve_weights_dir("vae", "audio_vae", weights_dir=explicit, allow_download=True)


def test_explicit_path_that_is_not_a_directory_fails(isolated_env, tmp_path, expect_error):
    with expect_error(weights.WeightsNotFoundError, "not a directory"):
        weights.resolve_weights_dir("vae", weights_dir=tmp_path / "absent")


# --- HuggingFace cache ----------------------------------------------------------------------------------


def test_complete_cached_snapshot_is_used_without_downloading(isolated_env, monkeypatch):
    snapshot = _cache_snapshot(isolated_env, "rev1", "vae", "audio_vae")
    _no_download(monkeypatch)

    assert weights.resolve_weights_dir("vae", "audio_vae", allow_download=True) == snapshot


def test_cached_snapshot_lacking_a_required_partition_is_skipped(isolated_env, expect_error):
    _cache_snapshot(isolated_env, "rev1", "vae")

    with expect_error(weights.WeightsNotFoundError, "weights not found"):
        weights.resolve_weights_dir("vae", "transformer")


def test_partial_cached_snapshot_without_always_assets_is_completed_by_download(isolated_env, fake_download):
    # Every weight partition is present but the tokenizer and friends are not (another tool's allow_patterns).
    _cache_snapshot(isolated_env, "rev1", "vae", "audio_vae", always=False)

    resolved = weights.resolve_weights_dir("vae", "audio_vae", allow_download=True)

    assert len(fake_download) == 1
    assert resolved.name == "downloaded"
    assert weights._missing_always_assets(resolved) == []


def test_partial_cached_snapshot_without_always_assets_fails_when_downloads_are_off(isolated_env, expect_error):
    _cache_snapshot(isolated_env, "rev1", "vae", "audio_vae", always=False)

    with expect_error(weights.WeightsNotFoundError, weights.ALLOW_DOWNLOAD_ENV):
        weights.resolve_weights_dir("vae", "audio_vae")


def test_refs_main_revision_is_preferred(isolated_env, monkeypatch):
    import os

    older = _cache_snapshot(isolated_env, "aaaa", "vae", main=True)
    newer = _cache_snapshot(isolated_env, "zzzz", "vae")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    _no_download(monkeypatch)

    assert weights.resolve_weights_dir("vae") == older


def test_refs_main_pointing_at_an_incomplete_snapshot_falls_back_to_the_newest_complete_one(isolated_env):
    import os

    _cache_snapshot(isolated_env, "head", "vae", always=False, main=True)
    old = _cache_snapshot(isolated_env, "old", "vae")
    new = _cache_snapshot(isolated_env, "new", "vae")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    assert weights.resolve_weights_dir("vae") == new


# --- download gating ------------------------------------------------------------------------------------


def test_no_cache_and_downloads_off_fails_with_actionable_message(isolated_env, monkeypatch, expect_error):
    _no_download(monkeypatch)

    with expect_error(weights.WeightsNotFoundError, f"{weights.MODEL_PATH_ENV}.*{weights.ALLOW_DOWNLOAD_ENV}=1"):
        weights.resolve_weights_dir("transformer", "vae")


def test_download_opt_in_env_var_must_be_exactly_1(isolated_env, monkeypatch, expect_error):
    _no_download(monkeypatch)
    monkeypatch.setenv(weights.ALLOW_DOWNLOAD_ENV, "true")

    with expect_error(weights.WeightsNotFoundError, "weights not found"):
        weights.resolve_weights_dir("vae")


def test_download_fetches_only_required_partitions_plus_always_assets(isolated_env, monkeypatch, fake_download):
    monkeypatch.setenv(weights.ALLOW_DOWNLOAD_ENV, "1")

    resolved = weights.resolve_weights_dir("transformer", "vae")

    (call,) = fake_download
    assert call["repo_id"] == weights.MINIMAX_H3_REPO_ID
    assert call["allow_patterns"] == [*weights._ALWAYS_PATTERNS, "transformer/*", "vae/*"]
    assert "text_encoder/*" not in call["allow_patterns"]
    assert resolved.name == "downloaded"


def test_download_that_comes_back_incomplete_fails(isolated_env, monkeypatch, tmp_path, expect_error):
    def snapshot_download(**kwargs):
        return str(_make_snapshot(tmp_path / "downloaded", "vae"))  # asked for transformer too

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    with expect_error(weights.WeightsNotFoundError, r"downloaded .* missing \['transformer'\]"):
        weights.resolve_weights_dir("transformer", "vae", allow_download=True)
