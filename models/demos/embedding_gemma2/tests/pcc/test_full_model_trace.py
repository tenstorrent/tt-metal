# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced full multimodal forward (towers + on-device merge + text encoder + pooling in one trace).

All ShapeKeys the module uses are captured up front by `EmbeddingGemma2.capture_all`; eager runs
afterwards use only captured (hence compiled) shapes.

    test_trace_matches_eager   replay == eager, bit-exact, and cosine vs HF fp32 >= 0.99
    test_trace_reuse           two different items with the same ShapeKey replay the same trace
    test_determinism           5 replays + 2 eager runs bit-identical
"""

import pytest
import torch
from loguru import logger

from models.demos.embedding_gemma2.tests.common import COSINE_EMBEDDING, log_result
from models.demos.embedding_gemma2.tests.pcc.test_full_model import (  # noqa: F401  (fixtures)
    cos,
    full_items,
    full_model,
    hf_embedding,
    hf_full,
    hf_processor,
)

TRACE_ITEMS = (
    "text_query",
    "text_long",
    "image_dog_caption",
    "image_house_caption",
    "video_text",
    "audio_speech10s_text",
    "mixed_image_audio",
)
BATCH_ITEMS = ("image_house_caption", "image_pajamas_caption")  # one forward, batch 2 / 2 images
MRL_ITEM = "image_dog_caption"
N_REPEAT = 5


@pytest.fixture(scope="module")
def traced(full_model, full_items):
    m = full_model
    keys = set()
    for n in TRACE_ITEMS:
        keys.update(m.shape_keys([full_items[n]]))
    keys.update(m.shape_keys([full_items[n] for n in BATCH_ITEMS]))
    keys.update(m.shape_keys([full_items[MRL_ITEM]], truncate_dim=256))
    logger.info(f"capturing {len(keys)} traces: {sorted(str(k) for k in keys)}")
    m.capture_all(keys)
    yield m
    m.release_traces()


@pytest.mark.parametrize("name", TRACE_ITEMS)
def test_trace_matches_eager(name, traced, full_items, hf_embedding):
    item = [full_items[name]]
    eager = traced.encode(item)
    replay = traced.encode(item, traced=True)
    d = float((eager - replay).abs().max())
    log_result("full_trace_vs_eager_max_abs", d, 0.0, item=name)
    assert torch.equal(eager, replay), f"{name}: replay differs from eager (max {d})"
    assert log_result("full_trace_cosine", cos(replay[0], hf_embedding(name)), COSINE_EMBEDDING, item=name)


def test_trace_batched_and_mrl(traced, full_items, hf_embedding):
    items = [full_items[n] for n in BATCH_ITEMS]
    eager, replay = traced.encode(items), traced.encode(items, traced=True)
    assert torch.equal(eager, replay)
    for i, n in enumerate(BATCH_ITEMS):
        assert log_result("full_trace_batched_cosine", cos(replay[i], hf_embedding(n)), COSINE_EMBEDDING, item=n)
    e256 = traced.encode([full_items[MRL_ITEM]], truncate_dim=256)
    r256 = traced.encode([full_items[MRL_ITEM]], truncate_dim=256, traced=True)
    assert torch.equal(e256, r256)
    assert log_result(
        "full_trace_cosine_mrl256", cos(r256[0], hf_embedding(MRL_ITEM, 256)), COSINE_EMBEDDING, item=MRL_ITEM
    )


def test_trace_reuse(traced, full_items):
    """image_dog_caption and image_house_caption share b1_s512_image1x2560: one trace, two inputs."""
    m = traced
    k1 = m.shape_keys([full_items["image_dog_caption"]])
    k2 = m.shape_keys([full_items["image_house_caption"]])
    assert k1 == k2
    a = m.encode([full_items["image_dog_caption"]], traced=True)
    b = m.encode([full_items["image_house_caption"]], traced=True)
    a2 = m.encode([full_items["image_dog_caption"]], traced=True)
    assert not torch.equal(a, b) and torch.equal(a, a2)


@pytest.mark.parametrize("name", ["mixed_image_audio", "video_text"])
def test_determinism(name, traced, full_items):
    item = [full_items[name]]
    base = traced.encode(item, traced=True)
    for _ in range(N_REPEAT - 1):
        assert torch.equal(base, traced.encode(item, traced=True)), "traced replay is not deterministic"
    for _ in range(2):
        assert torch.equal(base, traced.encode(item)), "eager run differs from replay"
    logger.info(f"full-model determinism {name}: {N_REPEAT} replays + 2 eager runs bit-identical")
