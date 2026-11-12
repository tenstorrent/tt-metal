# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Traced KV-cache writes stay inside their caches.

Every talker and code-predictor step writes its cache with `paged_fused_update_cache` at a
moving position. A write that runs past the end corrupts whatever was allocated next, with
no error and audio that is only slightly wrong. So each cache gets a sentinel buffer
allocated right after it, two traced utterances run, and every sentinel must read back
exactly as written.

The sentinels are allocated with the caches, before any capture. A buffer allocated after
capture is no use as a sentinel: a trace replays into addresses that were free when it was
captured, so that buffer can be overwritten by design (`allocator.cpp` warns about it).

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/pcc/test_kv_cache_guard.py
"""

import pytest
import torch

import ttnn
from models.demos.audio.qwen3_tts.tests.reference_helpers import synthetic_voiced_clip
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor_decode import TtCodePredictorCachedDecoder
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline, build_clone_reference
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker_decode import TtTalkerCachedDecoder

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

TEXT = "Good morning. Today is a beautiful day for a walk in the park, with bright sun and a gentle breeze."
REFERENCE_TEXT = "Good morning. This sentence is being spoken by a language model."
LANGUAGE = "English"
CLIP_SECONDS = 2.0
MAX_FRAMES = 128
UTTERANCES = 2
SEED = 0

# 6 MiB of bfloat16 per sentinel: room for a write that runs well past the end of a cache.
SENTINEL_SHAPE = (1, 1, 1024, 3072)
SENTINEL_VALUE = 3.0

# Enough traced steps that a write at a moving position walks well past the prompt.
MIN_FRAMES = 32


def _sentinel(device):
    return ttnn.from_torch(
        torch.full(SENTINEL_SHAPE, SENTINEL_VALUE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_kv_cache_writes_stay_inside_their_caches(device, monkeypatch):
    """A sentinel after every talker and code-predictor cache survives two utterances."""
    after = {}  # id(cache tensor) -> the sentinel allocated right after it

    talker_cache_tensor = TtTalkerCachedDecoder._cache_tensor

    def talker_cache_then_sentinel(self, empty):
        cache = talker_cache_tensor(self, empty)
        after[id(cache)] = _sentinel(self.device)
        return cache

    monkeypatch.setattr(TtTalkerCachedDecoder, "_cache_tensor", talker_cache_then_sentinel)

    # The predictor builds its caches inline, so follow the uploads of exactly its cache shape.
    predictor_init = TtCodePredictorCachedDecoder.__init__
    upload = ttnn.from_torch

    def predictor_init_with_sentinels(self, device, *args, **kwargs):
        def from_torch(tensor, *upload_args, **upload_kwargs):
            out = upload(tensor, *upload_args, **upload_kwargs)
            if tuple(tensor.shape) == (1, self.kv_heads, self.max_seq, self.head_dim):
                after[id(out)] = _sentinel(device)
            return out

        monkeypatch.setattr(ttnn, "from_torch", from_torch)
        try:
            predictor_init(self, device, *args, **kwargs)
        finally:
            monkeypatch.setattr(ttnn, "from_torch", upload)

    monkeypatch.setattr(TtCodePredictorCachedDecoder, "__init__", predictor_init_with_sentinels)

    reference = build_clone_reference(device, synthetic_voiced_clip(seconds=CLIP_SECONDS), REFERENCE_TEXT)
    pipeline = Qwen3TTSPipeline(device, max_frames=MAX_FRAMES, seed=SEED)

    caches = {
        "talker": pipeline.talker.k_cache + pipeline.talker.v_cache,
        "predictor": pipeline.predictor.k_cache + pipeline.predictor.v_cache,
    }
    unguarded = [name for name, tensors in caches.items() for cache in tensors if id(cache) not in after]
    assert not unguarded, f"caches with no sentinel after them: {unguarded}"

    frames = []
    for _ in range(UTTERANCES):
        pipeline.reseed(SEED)
        _, codes = pipeline.generate_clone(TEXT, reference, language=LANGUAGE)
        frames.append(int(codes.shape[0]))
    pipeline.release()
    assert min(frames) >= MIN_FRAMES, f"too few traced steps to mean anything: {frames} frames"

    corrupt = {}
    for name, tensors in caches.items():
        for index, cache in enumerate(tensors):
            changed = int((ttnn.to_torch(after[id(cache)]).float() != SENTINEL_VALUE).sum())
            if changed:
                corrupt[f"{name} cache {index}"] = changed

    guarded = sum(len(tensors) for tensors in caches.values())
    print(f"{frames} frames; {guarded} cache sentinels checked")
    assert not corrupt, f"elements changed after a cache (of {torch.Size(SENTINEL_SHAPE).numel()}): {corrupt}"
