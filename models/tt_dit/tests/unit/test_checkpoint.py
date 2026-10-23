# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the local weights mirror resolver."""

from pathlib import Path

import pytest

from models.tt_dit.utils.checkpoint import WEIGHTS_ROOT_ENV, local_checkpoint_dir, local_revision, resolve_checkpoint

REPO_ID = "Org/Model-Diffusers"


@pytest.fixture
def mirror(monkeypatch, tmp_path) -> Path:
    root = tmp_path / "mirror"
    monkeypatch.setenv(WEIGHTS_ROOT_ENV, str(root))
    return root


def _mirror_repo(mirror: Path, *entries: str) -> Path:
    directory = mirror / REPO_ID
    directory.mkdir(parents=True)
    for name in entries:
        (directory / name).mkdir()
    return directory


def test_repo_in_mirror_resolves_to_its_directory(mirror):
    directory = _mirror_repo(mirror, "vae")
    assert resolve_checkpoint(REPO_ID) == str(directory)
    assert Path(resolve_checkpoint(REPO_ID)).name == "Model-Diffusers"  # tt_dit cache namespace is the basename


def test_repo_absent_from_mirror_resolves_to_repo_id(mirror):
    assert resolve_checkpoint(REPO_ID) == REPO_ID


def test_mirror_missing_a_required_entry_resolves_to_repo_id(mirror):
    _mirror_repo(mirror, "vae")
    assert local_checkpoint_dir(REPO_ID, "vae", "transformer") is None
    assert resolve_checkpoint(REPO_ID, "vae", "transformer") == REPO_ID


def test_empty_root_disables_the_mirror(monkeypatch, tmp_path):
    _mirror_repo(tmp_path, "vae")
    monkeypatch.setenv(WEIGHTS_ROOT_ENV, "")
    assert resolve_checkpoint(REPO_ID) == REPO_ID


def test_unset_root_disables_the_mirror(monkeypatch, tmp_path):
    _mirror_repo(tmp_path, "vae")
    monkeypatch.chdir(tmp_path)  # a relative default root would find the repo here
    monkeypatch.delenv(WEIGHTS_ROOT_ENV, raising=False)
    assert local_checkpoint_dir(REPO_ID) is None


def test_existing_directory_is_returned_unchanged(mirror, tmp_path):
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path)


@pytest.mark.parametrize("name", ["model-only", "a/b/c", "../escape", "/abs/path", "org/"])
def test_non_repo_ids_are_not_looked_up(mirror, name):
    assert local_checkpoint_dir(name) is None


def test_local_revision_reads_hf_download_metadata(tmp_path):
    download = tmp_path / ".cache" / "huggingface" / "download"
    download.mkdir(parents=True)
    (download / "model_index.json.metadata").write_text("abc123\netag\n1790000000.0\n")
    assert local_revision(str(tmp_path)) == "abc123"


def test_local_revision_is_none_without_metadata(tmp_path):
    assert local_revision(str(tmp_path)) is None
    assert local_revision(REPO_ID) is None
