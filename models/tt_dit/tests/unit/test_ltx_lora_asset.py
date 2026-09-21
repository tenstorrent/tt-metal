# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from huggingface_hub.errors import LocalEntryNotFoundError

from models.tt_dit.utils import ltx_lora_asset as asset


def test_missing_offline_adapter_has_actionable_failure(monkeypatch, tmp_path, expect_error):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.delenv("LORA_PATH", raising=False)
    monkeypatch.setattr(asset.Path, "home", lambda: tmp_path)

    def missing(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", missing)
    with expect_error(RuntimeError, "Stage it"):
        asset.resolve_lora()


def test_preflight_fetches_pinned_adapter_into_writable_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("LORA_PATH", raising=False)
    monkeypatch.setattr(asset.Path, "home", lambda: tmp_path)

    def download(repo, filename, **kwargs):
        assert (repo, filename, kwargs["revision"]) == (asset.REPO_ID, asset.FILENAME, asset.REVISION)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        assert kwargs["local_dir"] == str(tmp_path / "writable")
        return str(tmp_path / "writable" / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert asset.resolve_lora(download_dir=str(tmp_path / "writable")).endswith(asset.FILENAME)


def test_explicit_missing_file_fails_before_loading_pipeline(monkeypatch, tmp_path, expect_error):
    monkeypatch.setenv("LORA_PATH", str(tmp_path / "missing"))
    with expect_error(FileNotFoundError, "LORA_PATH"):
        asset.resolve_lora()


def test_preflight_copies_cached_file_to_shared_worker_storage(monkeypatch, tmp_path):
    cached = tmp_path / "driver-only.safetensors"
    cached.write_bytes(b"adapter")
    monkeypatch.setenv("LORA_PATH", str(cached))
    destination = tmp_path / "shared"
    staged = asset.resolve_lora(download_dir=destination)
    assert asset.Path(staged) == destination / asset.FILENAME
    assert asset.Path(staged).read_bytes() == b"adapter"


def test_online_local_run_still_downloads_uncached_adapter(monkeypatch, tmp_path):
    monkeypatch.delenv("LORA_PATH", raising=False)
    monkeypatch.setattr(asset.Path, "home", lambda: tmp_path)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)

    def download(repo, filename, **kwargs):
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        assert kwargs == {"revision": asset.REVISION}
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert asset.resolve_lora() == str(tmp_path / asset.FILENAME)
