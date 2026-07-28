# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A transient checkpoint-download failure must not kill the server.

``resolve_checkpoint_dir`` is the first thing that runs when a server starts on a host with a cold
weight cache, and it raises straight out of ``initialize_vllm_model`` -> ``load_model`` ->
``_init_executor``, where any exception is fatal to the vLLM EngineCore. On 2026-07-28 one
``HTTP 500`` from HuggingFace's CDN partway through a ~50 GB fetch killed a CI eval before any
DiffusionGemma code ran, and the harness then held a QB2 runner for the rest of its 3600 s
health-check timeout.
"""
import time

from models.experimental.diffusion_gemma import checkpoint as ckpt


def test_transient_download_error_is_retried(monkeypatch, tmp_path):
    """A 5xx-style failure is retried, and the eventual success is returned."""
    calls = []

    def flaky(repo_id, **kwargs):
        calls.append(repo_id)
        if len(calls) < 3:
            raise ConnectionError("Network error: HTTP status server error (500 Internal Server Error)")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", flaky)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    out = ckpt._snapshot_download_with_retry("google/diffusiongemma-26B-A4B-it", attempts=5)

    assert out == str(tmp_path)
    assert len(calls) == 3, "should have retried twice before succeeding"


def test_exhausted_retries_raise_with_the_last_error(monkeypatch, expect_error):
    """After the last attempt it fails loudly, naming what actually went wrong."""

    def always_500(repo_id, **kwargs):
        raise ConnectionError("HTTP status server error (500 Internal Server Error)")

    monkeypatch.setattr("huggingface_hub.snapshot_download", always_500)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with expect_error(RuntimeError, match="failed after 3 attempts"):
        ckpt._snapshot_download_with_retry("google/diffusiongemma-26B-A4B-it", attempts=3)


def test_missing_repo_fails_immediately(monkeypatch, expect_error):
    """A wrong or gated repo id is not transient -- retrying it just wastes the same hour slowly."""
    from huggingface_hub.utils import RepositoryNotFoundError

    calls = []
    # Built with __new__: HfHubHTTPError requires a `response` kwarg on this huggingface_hub, and a
    # TypeError from constructing the exception would itself look transient and be retried -- which
    # is exactly what happened the first time this test was written.
    missing_error = RepositoryNotFoundError.__new__(RepositoryNotFoundError)

    def missing(repo_id, **kwargs):
        calls.append(repo_id)
        raise missing_error

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with expect_error(RepositoryNotFoundError, match=""):
        ckpt._snapshot_download_with_retry("google/does-not-exist", attempts=5)
    assert len(calls) == 1, "a missing repo must not be retried"
