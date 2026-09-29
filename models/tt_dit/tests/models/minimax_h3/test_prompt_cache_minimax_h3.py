# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks on the MiniMax-H3 conditioner-tap cache.

A prompt cache is only safe if a miss is impossible to mistake for a hit. These tests are mostly
about what has to make the key *differ*: a serving endpoint hands the same prompt string to
different tasks, pad lengths and keyframes all day, and returning one request's tap for another's
presentation would be silent -- the DiT would happily denoise against the wrong conditioning.
"""

import torch

from models.tt_dit.pipelines.minimax_h3.prompt_cache import PromptEmbedCache, prompt_cache_key

HIDDEN = 5376


def _ids(values):
    return torch.tensor([values], dtype=torch.int64)


def _tap(seq_len, fill=0.0):
    return torch.full((1, seq_len, HIDDEN), fill, dtype=torch.float32)


def test_the_same_presentation_hashes_the_same():
    ids, types = _ids([1, 2, 3, 4]), _ids([0, 0, 0, 0])
    assert prompt_cache_key(ids, types) == prompt_cache_key(ids.clone(), types.clone())


def test_every_part_of_the_presentation_changes_the_key():
    ids, types = _ids([1, 2, 3, 4]), _ids([0, 0, 0, 0])
    base = prompt_cache_key(ids, types)

    # Different tokens.
    assert prompt_cache_key(_ids([1, 2, 3, 5]), types) != base
    # Same tokens, different pad length -- the conditioner is run over the padded ids, so the two
    # presentations are genuinely different inputs.
    assert prompt_cache_key(_ids([1, 2, 3, 4, 0, 0]), _ids([0] * 6)) != base
    # Same tokens, different modality tags: this is the fl2va case where a vision block is
    # video-tagged, and it changes what the DiT's AdaLN keys off.
    assert prompt_cache_key(ids, _ids([0, 1, 1, 0])) != base
    # Same ids and tags, different keyframe pixels.
    with_pixels = prompt_cache_key(ids, types, torch.zeros(4, 8))
    assert with_pixels != base
    assert prompt_cache_key(ids, types, torch.ones(4, 8)) != with_pixels


def test_the_key_ignores_only_dtype_and_shape_spelling():
    """A presentation must not miss just because it arrived as a different dtype."""
    ids, types = _ids([1, 2, 3, 4]), _ids([0, 0, 0, 0])
    assert prompt_cache_key(ids.to(torch.int32), types.to(torch.int32)) == prompt_cache_key(ids, types)
    pixels = torch.rand(4, 8)
    assert prompt_cache_key(ids, types, pixels.to(torch.bfloat16).to(torch.float32)) == prompt_cache_key(
        ids, types, pixels.to(torch.bfloat16).to(torch.float32)
    )
    # But a flattened id vector is a different shape and must not collide with the batched one.
    assert prompt_cache_key(ids.reshape(-1), types.reshape(-1)) != prompt_cache_key(ids, types)


def test_memory_lru_evicts_oldest_and_counts():
    cache = PromptEmbedCache(capacity=2)
    cache.put("a", _tap(8, 1.0))
    cache.put("b", _tap(8, 2.0))
    assert cache.get("a") is not None  # promotes "a"
    cache.put("c", _tap(8, 3.0))  # evicts "b", the least recently used
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.stats()["resident"] == 2
    assert cache.stats()["hits"] == 3
    assert cache.stats()["misses"] == 1


def test_stored_taps_are_bf16_and_round_trip_through_disk(tmp_path):
    values = torch.randn(1, 32, HIDDEN)
    writer = PromptEmbedCache(capacity=1, disk_dir=tmp_path)
    stored = writer.put("k", values)
    assert stored.dtype == torch.bfloat16
    torch.testing.assert_close(stored.to(torch.float32), values.to(torch.bfloat16).to(torch.float32))

    # A fresh cache -- a restarted process -- finds it on disk and promotes it into memory.
    reader = PromptEmbedCache(capacity=1, disk_dir=tmp_path)
    got = reader.get("k")
    assert got is not None
    torch.testing.assert_close(got.to(torch.float32), stored.to(torch.float32))
    assert reader.stats()["disk_hits"] == 1
    assert reader.get("k") is not None
    assert reader.stats()["hits"] == 1  # the second read came from memory, not disk


def test_a_truncated_disk_entry_is_a_miss_and_not_a_crash(tmp_path):
    cache = PromptEmbedCache(capacity=1, disk_dir=tmp_path)
    cache.put("k", _tap(8))
    path = tmp_path / "k.pt"
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])
    cache._memory.clear()  # noqa: SLF001 -- force the read to reach disk
    assert cache.get("k") is None
    assert not path.exists()  # the bad entry is dropped so the next put rewrites it


def test_capacity_zero_disables_the_memory_tier(tmp_path):
    cache = PromptEmbedCache(capacity=0, disk_dir=tmp_path)
    cache.put("k", _tap(8))
    assert cache.stats()["resident"] == 0
    # ... but the disk tier still works, so a zero-capacity cache is a disk-only cache.
    assert cache.get("k") is not None
