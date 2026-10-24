# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for populating the HF cache from the local weights mirror."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from models.tt_dit.utils.checkpoint import WEIGHTS_ROOT_ENV, CheckpointCacheError, ensure_cached, hub_cache_dir

REPO_ID = "Org/Model-Diffusers"
COMMIT = "a" * 40
FILES = {"model_index.json": "{}", "vae/config.json": '{"vae": 1}', "vae/weights.safetensors": "weights"}


def _etag(rel: str) -> str:
    return f"etag-{rel.replace('/', '-')}"


def make_mirror(root: Path, repo_id: str = REPO_ID, files: dict = FILES, commit: str = COMMIT) -> Path:
    """A mirror laid out as `hf download --local-dir` leaves it: the files plus per-file download metadata."""
    mirror = root / repo_id
    for rel, content in files.items():
        (mirror / rel).parent.mkdir(parents=True, exist_ok=True)
        (mirror / rel).write_text(content)
        metadata = mirror / ".cache" / "huggingface" / "download" / f"{rel}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"{commit}\n{_etag(rel)}\n1790000000.0\n")
    return mirror


def repo_cache(cache: Path, repo_id: str = REPO_ID) -> Path:
    return cache / f"models--{repo_id.replace('/', '--')}"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Mirror root and hub cache in tmp_path; returns (mirror root, hub cache)."""
    root, cache = tmp_path / "mirror", tmp_path / "hub"
    monkeypatch.setenv(WEIGHTS_ROOT_ENV, str(root))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return root, cache


# --- populating ---------------------------------------------------------------------------------------


def test_empty_cache_is_populated_with_links_into_the_mirror(env):
    root, cache = env
    mirror = make_mirror(root)

    assert ensure_cached(REPO_ID) == REPO_ID

    entry = repo_cache(cache)
    assert (entry / "refs" / "main").read_text() == COMMIT
    for rel, content in FILES.items():
        pointer = entry / "snapshots" / COMMIT / rel
        assert pointer.is_symlink() and not os.path.isabs(os.readlink(pointer))  # relative, as HF writes them
        assert pointer.read_text() == content
        blob = entry / "blobs" / _etag(rel)
        assert Path(os.readlink(blob)) == (mirror / rel).resolve()


def test_huggingface_hub_resolves_the_populated_cache_offline(env):
    root, cache = env
    make_mirror(root)
    ensure_cached(REPO_ID)

    path = hf_hub_download(REPO_ID, "vae/config.json", cache_dir=cache, local_files_only=True)

    assert Path(path) == repo_cache(cache) / "snapshots" / COMMIT / "vae" / "config.json"
    assert Path(path).read_text() == FILES["vae/config.json"]


def test_complete_cache_is_left_alone(env):
    root, cache = env
    make_mirror(root)
    ensure_cached(REPO_ID)
    before = {p: p.lstat().st_mtime_ns for p in repo_cache(cache).rglob("*")}

    ensure_cached(REPO_ID)

    assert {p: p.lstat().st_mtime_ns for p in repo_cache(cache).rglob("*")} == before


def test_partial_cache_at_the_same_commit_is_completed_and_existing_blobs_kept(env):
    root, cache = env
    make_mirror(root)
    entry = repo_cache(cache)
    downloaded = entry / "blobs" / _etag("vae/config.json")  # a real blob from an earlier download
    downloaded.parent.mkdir(parents=True)
    downloaded.write_text(FILES["vae/config.json"])
    (entry / "snapshots" / COMMIT / "vae").mkdir(parents=True)
    os.symlink(
        Path("..", "..", "..", "blobs", _etag("vae/config.json")), entry / "snapshots" / COMMIT / "vae" / "config.json"
    )
    (entry / "refs").mkdir()
    (entry / "refs" / "main").write_text(COMMIT)

    ensure_cached(REPO_ID)

    assert not downloaded.is_symlink()
    assert all((entry / "snapshots" / COMMIT / rel).read_text() == content for rel, content in FILES.items())


def test_complete_snapshot_without_a_ref_gets_one(env):
    root, cache = env
    make_mirror(root)
    ensure_cached(REPO_ID)
    (repo_cache(cache) / "refs" / "main").unlink()

    ensure_cached(REPO_ID)

    assert (repo_cache(cache) / "refs" / "main").read_text() == COMMIT


def test_metadata_for_files_no_longer_in_the_mirror_is_ignored(env):
    root, cache = env
    mirror = make_mirror(root)
    stale = mirror / ".cache" / "huggingface" / "download" / "Ref2VA" / "big.safetensors.metadata"
    stale.parent.mkdir(parents=True)
    stale.write_text(f"{COMMIT}\netag-stale\n")

    ensure_cached(REPO_ID)

    assert not (repo_cache(cache) / "snapshots" / COMMIT / "Ref2VA").exists()


def test_concurrent_ranks_populate_consistently(env):
    root, cache = env
    make_mirror(root)

    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(lambda _: ensure_cached(REPO_ID), range(8))) == [REPO_ID] * 8

    entry = repo_cache(cache)
    assert all((entry / "snapshots" / COMMIT / rel).read_text() == content for rel, content in FILES.items())
    assert not [p for p in entry.rglob("*.tmp")]


# --- no-ops -------------------------------------------------------------------------------------------


def test_unset_root_does_nothing(monkeypatch, tmp_path):
    make_mirror(tmp_path / "mirror")
    monkeypatch.delenv(WEIGHTS_ROOT_ENV, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    assert ensure_cached(REPO_ID) == REPO_ID
    assert not (tmp_path / "hub").exists()


def test_repo_absent_from_the_mirror_does_nothing(env):
    _, cache = env
    assert ensure_cached(REPO_ID) == REPO_ID
    assert not cache.exists()


@pytest.mark.parametrize("name", ["model-only", "a/b/c", "../escape", "/abs/path", "org/"])
def test_non_repo_ids_are_returned_unchanged(env, name):
    assert ensure_cached(name) == name


# --- errors -------------------------------------------------------------------------------------------


def test_cache_at_a_different_commit_errors(env, expect_error):
    root, cache = env
    make_mirror(root)
    (repo_cache(cache) / "refs").mkdir(parents=True)
    (repo_cache(cache) / "refs" / "main").write_text("b" * 40)

    with expect_error(CheckpointCacheError, r"is at commit b+, but local mirror .* is at a+"):
        ensure_cached(REPO_ID)


def test_mirror_without_download_metadata_errors(env, expect_error):
    root, _ = env
    mirror = make_mirror(root)
    (mirror / "vae" / "extra.json").write_text("{}")  # copied in by hand, so no metadata

    with expect_error(CheckpointCacheError, r"no download metadata for \['vae/extra.json'\]"):
        ensure_cached(REPO_ID)


def test_mirror_mixing_commits_errors(env, expect_error):
    root, _ = env
    mirror = make_mirror(root)
    metadata = mirror / ".cache" / "huggingface" / "download" / "model_index.json.metadata"
    metadata.write_text(f"{'c' * 40}\n{_etag('model_index.json')}\n")

    with expect_error(CheckpointCacheError, "mixes commits"):
        ensure_cached(REPO_ID)


# --- hub cache location -------------------------------------------------------------------------------


def test_hub_cache_prefers_hf_hub_cache_over_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    assert hub_cache_dir() == tmp_path / "cache"


def test_hub_cache_defaults_to_hf_home_hub(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert hub_cache_dir() == tmp_path / "home" / "hub"
