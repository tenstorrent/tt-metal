# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The download of the matmul config model into ttnn's cache (ttnn/ttnn/matmul_model/__init__.py), against a fake
registry: no device, no network."""

import hashlib
import io
import json
import re
import stat
import urllib.error

import pytest
import ttnn

from ttnn import matmul_model as dl

BLOB = b"matmul config model" * 1000
SHA256 = hashlib.sha256(BLOB).hexdigest()


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """Pins BLOB, caches under tmp_path, and serves `served` as ghcr would (or raises it); records the requested URLs."""
    state = {"served": BLOB, "urls": []}

    def urlopen(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        state["urls"].append(url)
        if isinstance(state["served"], Exception):
            raise state["served"]
        if url == "https://ghcr.io/token?scope=repository:org/repo/model:pull":
            return io.BytesIO(json.dumps({"token": "t"}).encode())
        assert url == f"https://ghcr.io/v2/org/repo/model/blobs/sha256:{SHA256}"
        assert request.get_header("Authorization") == "Bearer t"
        return io.BytesIO(state["served"])

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl, "_pin", lambda: {"repository": "org/repo/model", "sha256": SHA256})
    monkeypatch.setattr(dl, "_failed", set())
    with ttnn.manage_config("cache_path", str(tmp_path / "cache")):
        yield state


def test_the_pin_is_a_sha256():
    pin = dl._pin()
    assert set(pin) == {"repository", "sha256"}
    assert re.fullmatch("[0-9a-f]{64}", pin["sha256"])


def test_the_cache_path_is_under_ttnn_cache_path(registry, tmp_path):
    assert dl.cache_path() == tmp_path / "cache" / "matmul_model" / f"{SHA256}.blob"


def test_downloads_the_pinned_blob(registry, tmp_path):
    path = tmp_path / "models" / "model.blob"
    assert dl.download(path) == path
    assert path.read_bytes() == BLOB
    assert list(path.parent.iterdir()) == [path]
    (tmp_path / "plain").touch()
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE((tmp_path / "plain").stat().st_mode)
    # Once there, it isn't downloaded again; a damaged file is
    registry["urls"].clear()
    dl.download(path)
    assert registry["urls"] == []
    path.write_bytes(b"damaged")
    dl.download(path)
    assert path.read_bytes() == BLOB


def test_rejects_a_blob_with_another_sha256(registry, tmp_path, expect_error):
    registry["served"] = BLOB[:-1]
    path = tmp_path / "models" / "model.blob"
    with expect_error(ValueError, "expected sha256"):
        dl.download(path)
    assert list(path.parent.iterdir()) == []


def test_downloads_to_the_cache_path_when_matmul_model_path_is_unset(registry):
    with ttnn.manage_config("matmul_model_path", ""):
        path = dl.download()
        assert path == dl.cache_path() and path.read_bytes() == BLOB
        assert str(ttnn.CONFIG.matmul_model_path) == str(path)


def test_does_not_fill_a_matmul_model_path_that_does_not_exist(registry, tmp_path, expect_error):
    with ttnn.manage_config("matmul_model_path", str(tmp_path / "typo.blob")):
        with expect_error(FileNotFoundError, "typo.blob"):
            dl.download()
    assert registry["urls"] == [] and not (tmp_path / "typo.blob").exists()


def test_download_if_enabled(registry, tmp_path):
    with ttnn.manage_config("matmul_model_path", str(dl.cache_path())):
        # Flag off: no network
        with ttnn.manage_config("matmul_model_config", False):
            dl.download_if_enabled()
        assert registry["urls"] == [] and not dl.cache_path().exists()
        with ttnn.manage_config("matmul_model_config", True):
            # A failed download doesn't raise, and isn't tried again in this process
            registry["served"] = urllib.error.URLError("unreachable")
            dl.download_if_enabled()
            assert len(registry["urls"]) == 1
            dl.download_if_enabled()
            assert len(registry["urls"]) == 1 and not dl.cache_path().exists()
            dl._failed.clear()
            registry["served"] = BLOB
            dl.download_if_enabled()
            assert dl.cache_path().read_bytes() == BLOB
    # A matmul_model_path set to another file is left alone
    registry["urls"].clear()
    with ttnn.manage_config("matmul_model_path", str(tmp_path / "mine.blob")), ttnn.manage_config(
        "matmul_model_config", True
    ):
        dl.download_if_enabled()
    assert registry["urls"] == []


def test_on_import_sets_the_default_path(registry, tmp_path):
    mine = str(tmp_path / "mine.blob")
    with ttnn.manage_config("matmul_model_config", False):
        with ttnn.manage_config("matmul_model_path", ""):
            dl.on_import(tmp_path)
            assert str(ttnn.CONFIG.matmul_model_path) == str(dl.cache_path())
        build_blob = tmp_path / "runtime" / "matmul_model" / "matmul_model.blob"
        build_blob.parent.mkdir(parents=True)
        build_blob.write_bytes(BLOB)
        with ttnn.manage_config("matmul_model_path", ""):
            dl.on_import(tmp_path)
            assert str(ttnn.CONFIG.matmul_model_path) == str(build_blob)
        with ttnn.manage_config("matmul_model_path", mine):
            dl.on_import(tmp_path)
            assert str(ttnn.CONFIG.matmul_model_path) == mine
    assert registry["urls"] == []
