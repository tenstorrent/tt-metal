# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of tt/multimodal.py: item parsing, the embedding cache, and the scatter of item embeddings into a
prefill chunk's rows (chunks that start inside an item, items that end before the chunk, video's is_embed mask,
APC-stripped items served from the cache), and the image-only fallback that finds items by their placeholder runs."""

import torch

from models.experimental.mimo_v2_6_flash.tt.multimodal import (
    EmbeddingCache,
    ItemEncoder,
    MMInputError,
    MMItem,
    chunk_embeds,
    content_id,
    items_from_image_lists,
    items_overlapping,
    parse_items,
    placeholder_rows,
)

HIDDEN = 8
IMAGE = 151655


def rows_of(tag, n):
    """Item ``tag``'s embedding rows: row k is tag + k / 64 in every column (bf16)."""
    return (tag + torch.arange(n, dtype=torch.float32)[:, None] / 64).expand(n, HIDDEN).to(torch.bfloat16)


class FakeBackend:
    """Embeddings tagged by the item: image rows from pixel_values[0, 0], video likewise, audio from mel[0, 0]."""

    def __init__(self):
        self.calls = 0

    def embed_image(self, pixel_values, grid_thw):
        self.calls += 1
        n = int(torch.as_tensor(grid_thw).prod(-1).sum()) // 4
        return rows_of(float(pixel_values[0, 0]), n)

    embed_video = embed_image

    def embed_audio(self, mel):
        self.calls += 1
        return rows_of(float(mel[0, 0]), int(mel.shape[0]) // 4)


def image_item(tag, offset, h=4, w=4, identifier=None, stripped=False):
    grid = torch.tensor([1, h, w])
    data = None if stripped else {"pixel_values": torch.full((h * w, 6), float(tag)), "image_grid_thw": grid}
    return {
        "modality": "image",
        "identifier": identifier or f"img{tag}",
        "offset": offset,
        "length": h * w // 4,
        "is_embed": None,
        "data": data,
    }


def scatter(items, start, end, encoder):
    return chunk_embeds(parse_items(items), start, end, encoder, HIDDEN)


def test_parse_validates(expect_error):
    with expect_error(MMInputError, "not served"):
        parse_items([{"modality": "audio", "identifier": "a", "offset": 0, "length": 4}], modalities=("image",))
    with expect_error(MMInputError, "is_embed"):
        parse_items([{"modality": "video", "identifier": "v", "offset": 0, "length": 4, "is_embed": [True]}])
    with expect_error(MMInputError, "missing processor fields"):
        parse_items([{"modality": "image", "identifier": "i", "offset": 0, "length": 4, "data": {"pixel_values": 1}}])
    with expect_error(MMInputError, "overlap"):
        parse_items([image_item(1, 0), image_item(2, 3)])
    with expect_error(MMInputError, "malformed"):
        parse_items([{"modality": "image"}])
    # An all-True mask is the same as no mask; items come back in prompt order.
    items = parse_items([image_item(2, 10), {**image_item(1, 0), "is_embed": torch.ones(4, dtype=torch.bool)}])
    assert [it.offset for it in items] == [0, 10] and items[0].is_embed is None


def test_scatter_whole_and_partial_items():
    enc = ItemEncoder(FakeBackend(), EmbeddingCache())
    # Image 1 at [4, 8), image 2 at [20, 36) (an 8 x 8 grid: 16 rows).
    items = [image_item(1, 4), image_item(2, 20, 8, 8)]
    embeds, rows = scatter(items, 0, 32, enc)
    assert rows.nonzero().flatten().tolist() == list(range(4, 8)) + list(range(20, 32))
    assert torch.equal(embeds[4:8], rows_of(1, 4))
    assert torch.equal(embeds[20:32], rows_of(2, 16)[:12])  # image 2 continues past the chunk
    assert not embeds[~rows].any()
    # The next chunk starts inside image 2; image 1 ended before it. Image 2 is not encoded again.
    calls = enc.backend.calls
    embeds, rows = scatter(items, 32, 64, enc)
    assert rows.nonzero().flatten().tolist() == list(range(0, 4))
    assert torch.equal(embeds[:4], rows_of(2, 16)[12:])
    assert enc.backend.calls == calls
    # A chunk with no item rows: nothing to merge.
    assert scatter(items, 8, 20, enc) == (None, None)
    assert scatter(items, 40, 64, enc) == (None, None)


def test_scatter_video_is_embed():
    """A video's range interleaves timestamp / frame-marker tokens (False) with frame rows (True)."""
    mask = torch.tensor([False, False, True, True, True, True, False, False, True, True, True, True, False])
    item = {
        "modality": "video",
        "identifier": "vid",
        "offset": 5,
        "length": mask.numel(),
        "is_embed": mask,
        "data": {"pixel_values_videos": torch.full((32, 6), 3.0), "video_grid_thw": torch.tensor([2, 4, 4])},
    }
    enc = ItemEncoder(FakeBackend(), EmbeddingCache())
    embeds, rows = scatter([item], 0, 32, enc)
    want = torch.zeros(32, dtype=torch.bool)
    want[5 : 5 + mask.numel()] = mask
    assert torch.equal(rows, want)
    assert torch.equal(embeds[rows], rows_of(3, 8))
    # A chunk boundary inside the video, between its two frames: each chunk gets its frame's rows.
    first, r1 = scatter([item], 0, 12, enc)
    second, r2 = scatter([item], 12, 32, enc)
    assert torch.equal(first[r1], rows_of(3, 8)[:4]) and torch.equal(second[r2], rows_of(3, 8)[4:])
    # Only the masked rows count: a chunk covering just the separators has nothing to merge.
    assert scatter([item], 11, 13, enc) == (None, None)
    assert [it.identifier for it in items_overlapping(parse_items([item]), 11, 14)] == ["vid"]


def test_stripped_item_from_cache():
    """APC strips the data of an item inside the cached prefix: a chunk start moved back into it uses the cache."""
    cache = EmbeddingCache()
    enc = ItemEncoder(FakeBackend(), cache)
    scatter([image_item(7, 0, 8, 8)], 0, 16, enc)  # the request (or another one) that encoded it
    calls = enc.backend.calls
    embeds, rows = scatter([image_item(7, 0, 8, 8, stripped=True)], 8, 32, enc)
    assert rows.nonzero().flatten().tolist() == list(range(0, 8))
    assert torch.equal(embeds[:8], rows_of(7, 16)[8:])
    assert enc.backend.calls == calls and cache.hits >= 1


def test_uncached_stripped_item_falls_back_to_tokens():
    """No data and no cache entry: the item keeps its placeholder rows (logged), nothing raises."""
    enc = ItemEncoder(FakeBackend(), EmbeddingCache())
    items = [image_item(1, 0, stripped=True, identifier="gone"), image_item(2, 8)]
    embeds, rows = scatter(items, 0, 16, enc)
    assert rows.nonzero().flatten().tolist() == list(range(8, 12))
    assert "gone" in enc.failed


def test_encoder_mismatch_raises():
    class Short(FakeBackend):
        def embed_image(self, pixel_values, grid_thw):
            return rows_of(1, 3)

    enc = ItemEncoder(Short(), EmbeddingCache())
    # The encoder's row count must match the prompt's placeholder rows: logged and left to the tokens.
    embeds, rows = scatter([image_item(1, 0)], 0, 8, enc)
    assert embeds is None and "img1" in enc.failed


def test_cache_lru_budget():
    row = 2 * HIDDEN  # bytes of one bf16 row
    cache = EmbeddingCache(budget_bytes=10 * row)
    for k in range(4):
        cache.put(f"x{k}", rows_of(k, 3))  # 3 rows each: 3 fit
    assert "x0" not in cache and all(f"x{k}" in cache for k in (1, 2, 3))
    cache.get("x1")  # x1 is now the most recent: x2 goes next
    cache.put("x4", rows_of(4, 3))
    assert "x2" not in cache and "x1" in cache
    cache.put("big", rows_of(9, 11))  # larger than the budget: not kept
    assert "big" not in cache and cache.used <= cache.budget
    cache.put(None, rows_of(1, 1))  # no identifier: nothing to key on
    assert len(cache) == 3


def test_image_lists_fallback(expect_error):
    """The plugin's image-only kwargs: the k-th placeholder run is image k."""
    tokens = torch.tensor([1, 2, IMAGE, IMAGE, IMAGE, IMAGE, 3, 4, IMAGE, IMAGE, IMAGE, IMAGE, 5, 6])
    grid = torch.tensor([1, 4, 4])
    pv = [torch.full((16, 6), 1.0), torch.full((16, 6), 2.0)]
    items = items_from_image_lists(tokens, tokens.numel(), pv, [grid, grid], IMAGE)
    assert [(it.offset, it.length) for it in items] == [(2, 4), (8, 4)]
    assert items[0].identifier == content_id(pv[0], grid) != items[1].identifier
    # A chunk that ends inside image 2: its run is cut at the end; images past the end are left out.
    items = items_from_image_lists(tokens, 10, pv, [grid, grid], IMAGE)
    assert [(it.offset, it.length) for it in items] == [(2, 4), (8, 4)]
    items = items_from_image_lists(tokens, 7, pv, [grid, grid], IMAGE)
    assert [(it.offset, it.length) for it in items] == [(2, 4)]
    # A stripped image (inside the cached prefix) has no data and no identifier.
    items = items_from_image_lists(tokens, 14, [None, pv[1]], [grid, grid], IMAGE)
    assert items[0].data is None and items[0].identifier is None and items[1].data is not None
    with expect_error(MMInputError, "placeholder tokens"):
        items_from_image_lists(tokens, 14, pv, [torch.tensor([1, 4, 8]), grid], IMAGE)
    assert placeholder_rows(tokens, IMAGE) == [(2, 4), (8, 4)]


def test_item_positions():
    item = MMItem("video", "v", 10, 5, torch.tensor([False, True, True, False, True]), None)
    assert item.num_embeds == 3 and item.end == 15
    assert item.embed_positions().tolist() == [11, 12, 14]
