# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: GdnPrefixStateCache (the model-owned GDN prefix snapshots) takes only a paged KV block size that divides
128. A prefill resumes from a snapshot at a multiple of 128 tokens, and the resume position must start a KV block.

    PYTHONPATH=<worktree>:<tools>/ttnn-host-stub pytest --noconftest \
        models/demos/blackhole/qwen36/tests/test_prefix_cache_block_size_host.py
"""

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen36.tt import prefix_cache as pc


class _ReachedAllocation(Exception):
    """GdnPrefixStateCache.__init__ accepted the block size and went on to the scratch allocation."""


def _model(block=None):
    """A model with a paged cache of ``block``-token blocks, or with no cache yet (QWEN36_KV_BLOCK_SIZE decides)."""

    def scratch():
        raise _ReachedAllocation

    kv = None if block is None else [(SimpleNamespace(shape=(8, 1, block, 256)), None)]
    return SimpleNamespace(_paged_kv_caches=kv, _ensure_gdn_prefill_scratch=scratch)


@pytest.mark.parametrize("block", [32, 64, 128])
def test_a_block_size_that_divides_128_is_accepted(monkeypatch, block):
    monkeypatch.setenv("QWEN36_KV_BLOCK_SIZE", str(block))
    for model in (_model(block), _model()):
        with pytest.raises(_ReachedAllocation):  # allow-pytest.raises: host test, runs with --noconftest
            pc.GdnPrefixStateCache(model, num_slots=1)


@pytest.mark.parametrize("block", [96, 160, 256, 1024])
def test_a_block_size_that_does_not_divide_128_is_refused(monkeypatch, block):
    # 96: the snapshot at 128 lies inside block 1 (96 .. 191), so the resumed prefill would start mid-block.
    monkeypatch.setenv("QWEN36_KV_BLOCK_SIZE", str(block))
    for model in (_model(block), _model()):
        with pytest.raises(ValueError, match="divides 128"):  # allow-pytest.raises: host test, runs with --noconftest
            pc.GdnPrefixStateCache(model, num_slots=1)
