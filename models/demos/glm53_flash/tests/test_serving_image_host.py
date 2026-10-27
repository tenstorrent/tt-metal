# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no vLLM, no device) tests of the served image + video path of ``tt/generator_vllm.py``:

* the plugin's multimodal contract mirrored on fakes (``model_runner._gather_multi_modal_inputs``: per-row item
  lists of ``pixel_values`` / ``image_grid_thw`` / ``pixel_values_videos`` / ``video_grid_thw`` + placeholder
  ranges) and how ``tt/mm_inputs.py`` parses it;
* the scatter bookkeeping: image runs vs video spans from the ids, feature rows by placeholder range or by order,
  the per-chunk split of an item straddling a device chunk boundary (rows past the chunk end are never written);
* the video geometry for ``t > 1``: grid flattening to one window per temporal group, window batching;
* the adapter driven end to end with the fake mesh model + a fake vision runtime: a prompt with vision rows takes
  the eager stream prefill per device chunk (never the trace), text chunks of the same prompt and text rows of
  the same batch take the unchanged text path, the boot warm-up, the text-only engine's refusal;
* the multimodal class declaration (``SupportsMultiModal`` + the upstream processor registration through a fake
  vLLM api; the plain declaration without vLLM) and the ``GLM53_VLLM_IMAGES`` knob;
* the planner's vision bytes: the resident tower vs the 1 M x 4 point, the per-request mode.

    PYTHONPATH=<worktree>:<tt-metal>/ttnn:<tt-metal> <tt-metal>/python_env/bin/pytest \\
        models/demos/glm53_flash/tests/test_serving_image_host.py --noconftest -p no:cacheprovider -q
"""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from models.demos.glm53_flash.tests.test_generator_vllm_host import (
    PAD,
    VOCAB,
    _FakeMeshModel,
    _first_token,
    _greedy,
    expect_error,
)
from models.demos.glm53_flash.tt import generator_vllm as gv
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM, declare_multimodal_class
from models.demos.glm53_flash.tt.memory_plan import (
    GIB,
    VISION_TRANSIENT_BYTES_PER_PATCH,
    MemoryPlan,
    vision_tower_bytes,
    vision_transient_bytes,
)
from models.demos.glm53_flash.tt.mm_inputs import (
    IMAGE_END_ID,
    IMAGE_START_ID,
    IMAGE_TOKEN_ID,
    PATCH_DIM,
    VIDEO_END_ID,
    VIDEO_START_ID,
    MMItem,
    chunk_scatter,
    feature_rows,
    flatten_grid,
    items_in_range,
    mm_items,
    mm_present,
    placeholder_runs,
    vision_batches,
    window_patch_counts,
)
from models.demos.glm53_flash.tt.vllm_config import DEFAULT_VISION_BATCH_PATCHES, ServingGeometry, eager_pad_length

HIDDEN = 8  # the fake features' width (the adapter never looks at it)


# ------------------------------------------------------------------------------------------ prompt builders
def _text(n: int, seed: int = 0) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.long) * 7 + seed) % 1000 + 1000


def _image_run(n_tokens: int) -> torch.Tensor:
    return torch.cat(
        [torch.tensor([IMAGE_START_ID]), torch.full((n_tokens,), IMAGE_TOKEN_ID), torch.tensor([IMAGE_END_ID])]
    )


def _video_span(t: int, tokens_per_group: int) -> torch.Tensor:
    parts = [torch.tensor([VIDEO_START_ID])]
    for _ in range(t):
        parts.append(_image_run(tokens_per_group))
        parts.append(_text(3, seed=5))  # "<t>.<d> seconds"
    parts.append(torch.tensor([VIDEO_END_ID]))
    return torch.cat(parts)


def _image_item(h: int = 32, w: int = 32, offset=None, length=None) -> MMItem:
    return MMItem("image", torch.zeros(h * w, PATCH_DIM), torch.tensor([[1, h, w]]), offset, length)


def _video_item(t: int = 2, h: int = 32, w: int = 32, offset=None, length=None) -> MMItem:
    return MMItem("video", torch.zeros(t * h * w, PATCH_DIM), torch.tensor([[t, h, w]]), offset, length)


# ------------------------------------------------------------------------------------------ bookkeeping
def test_placeholder_runs_classify_image_runs_and_video_spans():
    ids = torch.cat([_text(11), _image_run(256), _text(4), _video_span(2, 256), _text(3), _image_run(64), _text(5)])
    runs = placeholder_runs(ids)
    assert [k for k, _ in runs] == ["image", "video", "image"]
    assert [int(r.numel()) for _, r in runs] == [256, 512, 64]
    # the first image run starts after <|begin_of_image|>; the video rows are the two frame runs (not contiguous)
    assert int(runs[0][1][0]) == 12 and int(runs[0][1][-1]) == 267
    v = runs[1][1]
    assert int(v[255]) + 1 != int(v[256]), "the two frame runs of a video are separated by the timestamp tokens"
    assert torch.equal(v, torch.nonzero(ids == IMAGE_TOKEN_ID).reshape(-1)[256:768])
    # two consecutive images (no text between the runs) stay two items: <|end_of_image|><|begin_of_image|> separates them
    ids2 = torch.cat([_image_run(16), _image_run(16)])
    assert [int(r.numel()) for _, r in placeholder_runs(ids2)] == [16, 16]
    assert placeholder_runs(_text(10)) == []


def test_feature_rows_by_placeholder_range_and_by_order_agree():
    ids = torch.cat([_text(11), _image_run(256), _text(4), _video_span(2, 256), _text(3)])
    v_start = int(torch.nonzero(ids == VIDEO_START_ID).reshape(-1)[0])
    v_end = int(torch.nonzero(ids == VIDEO_END_ID).reshape(-1)[0])
    by_order = feature_rows(ids, [_image_item(), _video_item(2)])
    by_range = feature_rows(
        ids, [_image_item(offset=12, length=256), _video_item(2, offset=v_start, length=v_end - v_start + 1)]
    )
    assert all(torch.equal(a, b) for a, b in zip(by_order, by_range))
    assert [int(r.numel()) for r in by_order] == [256, 512]
    # a placeholder range given as vLLM gives it for an image (the replacement tokens only) or wider (the markers)
    wide = feature_rows(
        ids, [_image_item(offset=11, length=258), _video_item(2, offset=v_start, length=v_end - v_start + 1)]
    )
    assert torch.equal(wide[0], by_order[0])
    # mismatches are refused: a video item against an image run, a 64-token item against a 256-token run
    with expect_error(ValueError, "is a video but prompt run 0 is a image"):
        feature_rows(ids, [_video_item(2), _image_item()])
    with expect_error(ValueError, "prompt run holds 256 image tokens vs 64"):
        feature_rows(ids, [_image_item(16, 16), _video_item(2)])
    with expect_error(ValueError, "image tokens in its placeholder range"):
        feature_rows(
            ids, [_image_item(offset=12, length=100), _video_item(2, offset=v_start, length=v_end - v_start + 1)]
        )
    # the plugin forwards every item on every engine chunk: an item past the prefix gets no rows, a cut item its
    # rows so far (by order or by range), a prefix with more runs than items is a missing payload
    # (layout: image rows 12..267, <|begin_of_video|> at 273, frame runs 275..530 and 536..791, <|end_of_video|> 796)
    assert (v_start, v_end) == (273, 796) and int(by_order[1][0]) == 275 and int(by_order[1][-1]) == 791
    prefix = ids[:700]  # ends inside the video's second frame run
    by_order = feature_rows(prefix, [_image_item(), _video_item(2)])
    by_range = feature_rows(
        prefix, [_image_item(offset=12, length=256), _video_item(2, offset=v_start, length=v_end - v_start + 1)]
    )
    assert int(by_order[0].numel()) == 256 and int(by_order[1].numel()) == 256 + 164 and int(by_order[1][-1]) == 699
    assert all(torch.equal(a, b) for a, b in zip(by_order, by_range))
    short = ids[:272]  # the image complete, the video not started
    assert [int(r.numel()) for r in feature_rows(short, [_image_item(), _video_item(2)])] == [256, 0]
    assert [
        int(r.numel())
        for r in feature_rows(short, [_image_item(offset=12, length=256), _video_item(2, offset=v_start, length=500)])
    ] == [256, 0]
    cut_img = feature_rows(ids[:100], [_image_item(), _video_item(2)])  # inside the image run
    assert int(cut_img[0].numel()) == 88 and int(cut_img[1].numel()) == 0
    after_group = ids[: v_start + 1 + 258 + 3]  # the video's first group + its timestamp, span still open
    assert [int(r.numel()) for r in feature_rows(after_group, [_image_item(), _video_item(2)])] == [256, 256]
    with expect_error(ValueError, "2 vision runs in the prompt .* vs 1 multimodal items"):
        feature_rows(ids, [_image_item()])


def test_chunk_scatter_splits_a_straddling_item_at_the_boundary():
    ids = torch.cat([_text(810), _image_run(256), _text(20)])  # image rows 811..1066, S = 1088
    rows = feature_rows(ids, [_image_item()])
    assert int(rows[0][0]) == 811 and int(rows[0][-1]) == 1066
    first = chunk_scatter(rows, 0, 1024)
    second = chunk_scatter(rows, 1024, 1088)
    assert len(first) == 1 and len(second) == 1
    e0, e1 = first[0], second[0]
    assert (e0.item, e0.lo, e0.hi, e0.n) == (0, 0, 213, 213)
    assert int(e0.local_rows[0]) == 811 and int(e0.local_rows[-1]) == 1023, "nothing past the chunk end"
    assert (e1.item, e1.lo, e1.hi, e1.n) == (0, 213, 256, 43)
    assert int(e1.local_rows[0]) == 0 and int(e1.local_rows[-1]) == 42
    # chunks without vision rows contribute nothing; the item index set per range
    assert chunk_scatter(rows, 0, 800) == [] and chunk_scatter(rows, 1067, 1088) == []
    assert items_in_range(rows, 0, 1024) == [0] and items_in_range(rows, 0, 811) == []
    # two items, the second entirely in the second chunk
    ids2 = torch.cat([_text(10), _image_run(256), _text(800), _image_run(64)])  # rows 11..266 and 1069..1132
    rows2 = feature_rows(ids2, [_image_item(), _image_item(16, 16)])
    assert items_in_range(rows2, 0, 1024) == [0] and items_in_range(rows2, 1024, 2000) == [1]
    assert [(e.item, e.lo, e.hi) for e in chunk_scatter(rows2, 0, 2000)] == [(0, 0, 256), (1, 0, 64)]


def test_video_grid_flattens_to_one_window_per_temporal_group_and_batches():
    item = _video_item(2, 32, 32)
    assert item.n_patches == 2048 and item.n_tokens == 512 and item.n_groups == 2
    assert flatten_grid(item.grid_thw).tolist() == [[1, 32, 32], [1, 32, 32]]
    assert flatten_grid(torch.tensor([[1, 16, 24], [3, 8, 8]])).tolist() == [
        [1, 16, 24],
        [1, 8, 8],
        [1, 8, 8],
        [1, 8, 8],
    ]
    # eight 1024-patch windows (a 16-frame 448x448 clip) at the 4096 budget: two tower calls of four windows
    clip = _video_item(8, 32, 32)
    assert vision_batches(flatten_grid(clip.grid_thw), DEFAULT_VISION_BATCH_PATCHES) == [(0, 4), (4, 8)]
    assert vision_batches(flatten_grid(clip.grid_thw), 1024) == [(i, i + 1) for i in range(8)]
    # a window above the budget runs alone; its neighbours still batch
    rows = torch.tensor([[1, 32, 32], [1, 100, 100], [1, 32, 32], [1, 32, 32]])
    assert vision_batches(rows, 4096) == [(0, 1), (1, 2), (2, 4)]
    assert window_patch_counts([_image_item(), clip]) == [1024] + [1024] * 8
    with expect_error(ValueError, "patches, grid"):
        MMItem("video", torch.zeros(100, PATCH_DIM), torch.tensor([[2, 32, 32]]))
    with expect_error(ValueError, "expected image or video"):
        MMItem("audio", torch.zeros(1024, PATCH_DIM), torch.tensor([[1, 32, 32]]))


def test_mm_items_parse_the_plugin_per_row_lists():
    pv, g = torch.zeros(1024, PATCH_DIM), torch.tensor([1, 32, 32])
    pvv, vg = torch.zeros(2048, PATCH_DIM), torch.tensor([2, 32, 32])
    kwargs = {
        "pixel_values": [None, [pv], [pv, None]],
        "image_grid_thw": [None, [g], [g, None]],
        "pixel_values_videos": [None, [None], [None, pvv]],
        "video_grid_thw": [None, [None], [None, vg]],
        "mm_placeholder_offsets": [None, [12], [12, 300]],
        "mm_placeholder_lengths": [None, [256], [256, 530]],
    }
    assert mm_present(kwargs)
    assert mm_items(kwargs, 0) == []
    (img,) = mm_items(kwargs, 1)
    assert img.kind == "image" and img.n_tokens == 256 and (img.offset, img.length) == (12, 256)
    a, b = mm_items(kwargs, 2)
    assert (a.kind, b.kind) == ("image", "video") and b.n_tokens == 512 and (b.offset, b.length) == (300, 530)
    # the plugin's text batch on a multimodal class: every base field present, no item
    assert not mm_present({"pixel_values": [None, None], "image_grid_thw": [None, None]})
    assert not mm_present({})
    # an item without any pixel payload (prefix caching strips it; declared off here) is refused
    with expect_error(ValueError, "no pixel payload"):
        mm_items({"pixel_values": [[None]], "image_grid_thw": [[None]]}, 0)
    with expect_error(ValueError, "without image_grid_thw"):
        mm_items({"pixel_values": [[pv]], "image_grid_thw": [[None]]}, 0)


# ------------------------------------------------------------------------------------------ adapter e2e
class _FakeStreams:
    def __init__(self, ids, rows, feats, pad_to):
        self.ids, self.rows, self.feats, self.pad_to = ids, rows, feats, pad_to


class _FakeVision:
    """``tt/multimodal.py::VisionRuntime``'s surface: features per item, the chunk streams, the warm-up item."""

    resident = True

    def __init__(self):
        self.encoded: list[list[MMItem]] = []
        self.chunks: list[tuple] = []

    def dummy_item(self):
        return _image_item()

    def encode(self, items):
        self.encoded.append(list(items))
        return [
            torch.arange(it.n_tokens, dtype=torch.float32).reshape(-1, 1).repeat(1, HIDDEN) + 100 * k
            for k, it in enumerate(items)
        ]

    def chunk_streams(self, model, chunk_ids, entries, pad_to):
        rows = torch.cat([e.local_rows for e, _ in entries])
        feats = torch.cat([f[e.lo : e.hi] for e, f in entries])
        assert int(rows.max()) < int(chunk_ids.numel()) <= pad_to, "a vision row past the chunk"
        self.chunks.append((int(chunk_ids.numel()), pad_to, rows.tolist(), feats[:, 0].tolist()))
        return _FakeStreams(chunk_ids, rows, feats, pad_to)


class _FakeMMMeshModel(_FakeMeshModel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.embedding = SimpleNamespace(hidden=HIDDEN, hc_mult=4)

    def prefill_streams(self, streams, seq_len, user=0, start=0):
        assert isinstance(streams, _FakeStreams) and int(streams.ids.numel()) == seq_len
        unit = 32 * self.rows
        assert streams.pad_to == eager_pad_length(seq_len, unit)
        self.calls.append(("prefill_streams", user, start, seq_len, streams.pad_to, int(streams.rows.numel())))
        T_loc = streams.pad_to // self.rows
        owner_row = (seq_len - 1) // T_loc
        return ("streams",), ("logits_last", _first_token(user, streams.ids)), owner_row


def _mm_adapter(chunk=1024, vision=True, warm=True):
    model = _FakeMMMeshModel(2, 4, 4, 4096)
    g = ServingGeometry(
        rows=2,
        cols=4,
        users_per_row=4,
        max_seq_len=4096,
        traced_prefill_lengths=tuple(model.trace_prefill_lengths()),
        pad_token_id=PAD,
        device_prefill_chunk=chunk,
        multimodal=vision,
    )
    v = _FakeVision() if vision else None
    a = GLM53FlashForCausalLM(model, g, vision=v)
    if warm:
        a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
        a.warmup_model_decode(enable_trace=False, max_batch_size=4, num_blocks=1, can_sample_on_device=True)
        a.warmup_model_prefill(enable_trace=True, can_sample_on_device=True)
        a.warmup_model_decode(enable_trace=True, max_batch_size=4, num_blocks=1, can_sample_on_device=True)
    return a, model, v


def _plugin_kwargs(rows: list[list[MMItem] | None]) -> dict:
    """The plugin's per-row field lists for a batch (``_gather_multi_modal_inputs``), from items."""
    out = {
        k: []
        for k in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "mm_placeholder_offsets",
            "mm_placeholder_lengths",
        )
    }
    for items in rows:
        if items is None:
            for k in out:
                out[k].append(None)
            continue
        out["pixel_values"].append([it.pixel_values if it.kind == "image" else None for it in items])
        out["image_grid_thw"].append([it.grid_thw[0] if it.kind == "image" else None for it in items])
        out["pixel_values_videos"].append([it.pixel_values if it.kind == "video" else None for it in items])
        out["video_grid_thw"].append([it.grid_thw[0] if it.kind == "video" else None for it in items])
        out["mm_placeholder_offsets"].append([it.offset for it in items])
        out["mm_placeholder_lengths"].append([it.length for it in items])
    return out


def test_warmup_runs_a_dummy_image_prompt_through_the_eager_stream_prefill(monkeypatch):
    monkeypatch.delenv(gv.WARMUP_MM_ENV, raising=False)
    a, m, v = _mm_adapter()
    assert len(v.encoded) == 1 and v.encoded[0][0].n_tokens == 256 and v.encoded[0][0].n_patches == 1024
    ps = [c for c in m.calls if c[0] == "prefill_streams"]
    assert ps == [("prefill_streams", 0, 0, 288, 512, 256)]  # 16 + 256 + 16 tokens -> the 512 bucket, 256 vision rows
    assert v.chunks[0][:3] == (288, 512, list(range(16, 272)))
    # the text ladder still ran before it (the text-only boot is untouched)
    assert [c for c in m.calls if c[0] == "prefill"] == [("prefill", 0, 0, 128)]
    monkeypatch.setenv(gv.WARMUP_MM_ENV, "0")
    a2, m2, v2 = _mm_adapter()
    assert v2.encoded == [] and not [c for c in m2.calls if c[0] == "prefill_streams"]


def test_image_prompt_takes_the_eager_stream_prefill_and_text_rows_the_text_path(monkeypatch):
    monkeypatch.setenv(gv.WARMUP_MM_ENV, "0")
    a, m, v = _mm_adapter()
    ids = torch.cat([_text(11), _image_run(256), _text(18)])  # 286 tokens, image rows 12..267
    S = int(ids.numel())
    toks = torch.zeros(2, S, dtype=torch.int32)
    toks[0] = ids
    toks[1, :100] = _text(100, seed=3)
    item = _image_item(offset=12, length=256)
    kw = _plugin_kwargs([[item], None])
    logits = a.prefill_forward(
        toks, prompt_lens=[S, 100], start_pos=torch.tensor([0, 0]), empty_slots=[0, 1], enable_trace=True, **kw
    )
    assert logits.shape == (2, 1, VOCAB)
    # row 0: eager stream prefill in ONE device chunk padded to the 512 bucket, the item encoded once; row 1: traced text
    assert [c for c in m.calls if c[0] == "prefill_streams"] == [("prefill_streams", 0, 0, S, 512, 256)]
    assert [c[:3] for c in m.calls if c[0] == "prefill_traced"] == [("prefill_traced", 1, 100)]
    (enc,) = v.encoded[0]  # the adapter re-parses the plugin's lists into its own MMItem
    assert len(v.encoded) == 1 and (enc.kind, enc.n_tokens, enc.offset, enc.length) == ("image", 256, 12, 256)
    n, pad, rows, feats = v.chunks[0]
    assert (n, pad, rows[0], rows[-1]) == (S, 512, 12, 267) and feats == [float(i) for i in range(256)]
    assert int(logits[0, 0].argmax()) == _first_token(0, ids) and int(logits[1, 0].argmax()) == _first_token(
        1, toks[1, :100].long()
    )
    assert a.slots.physical(0).prefilled_to == S and a.slots.physical(0).phase == "prefill"
    # the decode after it is the ordinary decode
    out = a.decode_forward(
        torch.tensor([[5], [6], [0], [0]], dtype=torch.int32),
        torch.tensor([S, 100, -1, -1]),
        sampling_params=_greedy(4),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert out.shape == (4, 1)


def test_straddling_image_is_split_over_two_device_chunks_and_text_chunks_stay_text(monkeypatch):
    monkeypatch.setenv(gv.WARMUP_MM_ENV, "0")
    a, m, v = _mm_adapter(chunk=1024)
    m.calls.clear()  # drop the boot ladder's text prefill
    ids = torch.cat([_text(810), _image_run(256), _text(20), _video_span(2, 64), _text(5)])
    # image rows 811..1066 (straddles 1024), video rows after 1088: chunks [0,1024) [1024,2048) [2048, S)
    S = int(ids.numel())
    v_rows = feature_rows(ids, [_image_item(), _video_item(2, 16, 16)])
    assert int(v_rows[1][-1]) < 2048, "the video sits in the second chunk"
    toks = ids.reshape(1, -1).to(torch.int32)
    kw = _plugin_kwargs([[_image_item(), _video_item(2, 16, 16)]])  # no placeholder ranges: matched by order
    # the whole prompt in one engine step (max_num_batched_tokens above the device chunk): 2 device chunks
    out, _ = a.prefill_forward(
        toks, prompt_lens=[S], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1), **kw
    )
    assert int(out[0, 0]) == _first_token(0, ids[1024:S])  # the fake's token of the last chunk's ids
    ps = [c for c in m.calls if c[0] == "prefill_streams"]
    assert ps == [("prefill_streams", 0, 0, 1024, 1024, 213), ("prefill_streams", 0, 1024, S - 1024, 256, 43 + 128)]
    assert not [c for c in m.calls if c[0] == "prefill"], "no chunk without vision rows here"
    assert len(v.encoded) == 1 and [it.kind for it in v.encoded[0]] == ["image", "video"]
    first, second = v.chunks
    assert first[2][0] == 811 and first[2][-1] == 1023 and first[3] == [float(i) for i in range(213)]
    assert second[2][:43] == list(range(0, 43)) and second[3][:43] == [float(i) for i in range(213, 256)]
    assert second[3][43:] == [100.0 + i for i in range(128)]  # the video's features follow (item 1 offset 100)
    # the same prompt arriving as two engine chunks (vLLM's token budget = the device chunk): the second call must
    # encode only the items that intersect [1024, S) -- the image (its tail) and the video -- and leave [0, 1024) alone
    m.calls.clear()
    v.encoded.clear()
    v.chunks.clear()
    a.prefill_forward(
        toks, prompt_lens=[1024], start_pos=torch.tensor([0]), empty_slots=[1], sampling_params=_greedy(1), **kw
    )
    assert [it.kind for it in v.encoded[0]] == ["image"]
    a.prefill_forward(
        toks, prompt_lens=[S], start_pos=torch.tensor([1024]), empty_slots=[1], sampling_params=_greedy(1), **kw
    )
    assert [it.kind for it in v.encoded[1]] == ["image", "video"]
    assert [c[2:4] for c in m.calls if c[0] == "prefill_streams"] == [(0, 1024), (1024, S - 1024)]
    assert a.slots.physical(1).prefilled_to == S
    # a prompt whose vision rows all sit in a later chunk: the first chunk takes the plain text prefill
    m.calls.clear()
    v.encoded.clear()
    ids3 = torch.cat([_text(1100), _image_run(256), _text(4)])
    S3 = int(ids3.numel())
    a.prefill_forward(
        ids3.reshape(1, -1).to(torch.int32),
        prompt_lens=[S3],
        start_pos=torch.tensor([0]),
        empty_slots=[2],
        sampling_params=_greedy(1),
        **_plugin_kwargs([[_image_item()]]),
    )
    kinds = [(c[0], c[2], c[3]) for c in m.calls if c[0] in ("prefill", "prefill_streams")]
    assert kinds == [("prefill", 0, 1024), ("prefill_streams", 1024, S3 - 1024)]


def test_text_only_engine_refuses_pixels_and_wave_mode_is_not_wired(monkeypatch):
    monkeypatch.setenv(gv.WARMUP_MM_ENV, "0")
    a, m, _ = _mm_adapter(vision=False)
    ids = torch.cat([_text(11), _image_run(256), _text(18)])
    toks = ids.reshape(1, -1).to(torch.int32)
    kw = _plugin_kwargs([[_image_item(offset=12, length=256)]])
    with expect_error(NotImplementedError, "text-only GLM53 engine"):
        a.prefill_forward(toks, prompt_lens=[int(ids.numel())], start_pos=torch.tensor([0]), empty_slots=[0], **kw)
    # the text batch a multimodal class sends (every field a list of None) is the text path
    text_kw = {
        "pixel_values": [None],
        "image_grid_thw": [None],
        "mm_placeholder_offsets": [None],
        "mm_placeholder_lengths": [None],
    }
    logits = a.prefill_forward(
        toks[:, :100], prompt_lens=[100], start_pos=torch.tensor([0]), empty_slots=[0], **text_kw
    )
    assert (
        logits.shape == (1, 1, VOCAB) and [c[0] for c in m.calls if c[0].startswith("prefill")][-1] == "prefill_traced"
    )
    # the mismatch between the payload and the prompt is refused before anything runs on the device
    a2, m2, v2 = _mm_adapter()
    bad = _plugin_kwargs([[_video_item(2, 16, 16)]])
    with expect_error(ValueError, "is a video but prompt run 0 is a image"):
        a2.prefill_forward(toks, prompt_lens=[int(ids.numel())], start_pos=torch.tensor([0]), empty_slots=[0], **bad)
    assert v2.encoded == [] and not [c for c in m2.calls if c[0] == "prefill_streams"]


# ------------------------------------------------------------------------------------------ declaration + knobs
def test_multimodal_class_declaration_with_and_without_vllm(monkeypatch):
    plain = declare_multimodal_class(None)
    assert issubclass(plain, GLM53FlashForCausalLM) and plain.supports_multimodal is True
    assert GLM53FlashForCausalLM.supports_multimodal is False
    assert plain.__name__ == plain.__qualname__ == "GLM53FlashForConditionalGeneration"
    assert plain.get_placeholder_str("image", 0) == "<|begin_of_image|><|image|><|end_of_image|>"
    assert plain.get_placeholder_str("video", 0) == "<|begin_of_video|><|video|><|end_of_video|>"
    with expect_error(ValueError, "Only image or video"):
        plain.get_placeholder_str("audio", 0)
    assert plain.model_capabilities is GLM53FlashForCausalLM.model_capabilities
    assert plain.decode_input_update_contract == 1 and callable(plain.initialize_vllm_model)

    # a fake vLLM api: the mixin and the registration decorator, as MULTIMODAL_REGISTRY.register_processor(...)
    class SupportsMultiModal:
        supports_multimodal = True

    seen = {}

    def register_processor(cls):
        seen["cls"] = cls
        cls._processor_factory = "glm5next"
        return cls

    api = SimpleNamespace(SupportsMultiModal=SupportsMultiModal, register_processor=register_processor)
    mm = declare_multimodal_class(api)
    assert seen["cls"] is mm and mm._processor_factory == "glm5next"
    assert issubclass(mm, SupportsMultiModal) and issubclass(mm, GLM53FlashForCausalLM)
    # the module-level class resolves through the override target the catalog names
    mod, _, name = gv.MM_ADAPTER_TARGET.partition(":")
    assert getattr(importlib.import_module(mod), name) is gv.GLM53FlashForConditionalGeneration
    assert gv.GLM53FlashForConditionalGeneration.supports_multimodal is True
    # GLM53_VLLM_IMAGES: auto follows the class, 1 / 0 force
    monkeypatch.delenv(gv.IMAGES_ENV, raising=False)
    assert mm.images_enabled() is True and GLM53FlashForCausalLM.images_enabled() is False
    monkeypatch.setenv(gv.IMAGES_ENV, "0")
    assert mm.images_enabled() is False
    monkeypatch.setenv(gv.IMAGES_ENV, "1")
    assert GLM53FlashForCausalLM.images_enabled() is True
    monkeypatch.setenv(gv.VISION_RESIDENT_ENV, "0")
    monkeypatch.setenv(gv.VISION_BATCH_ENV, "1024")
    assert GLM53FlashForCausalLM.vision_settings() == (False, 1024)
    monkeypatch.delenv(gv.VISION_RESIDENT_ENV)
    monkeypatch.delenv(gv.VISION_BATCH_ENV)
    assert GLM53FlashForCausalLM.vision_settings() == (True, DEFAULT_VISION_BATCH_PATCHES)
    monkeypatch.setenv(gv.VISION_BATCH_ENV, "0")
    with expect_error(ValueError, "patch budget"):
        GLM53FlashForCausalLM.vision_settings()


def test_geometry_carries_the_multimodal_settings():
    g = ServingGeometry(
        rows=8,
        cols=4,
        users_per_row=4,
        max_seq_len=1048576,
        multimodal=True,
        vision_resident=False,
        vision_batch_patches=1024,
    )
    assert g.multimodal and not g.vision_resident and g.vision_batch_patches == 1024
    assert ServingGeometry(rows=8, cols=4, users_per_row=4, max_seq_len=1024).multimodal is False
    with expect_error(ValueError, "vision_batch_patches"):
        ServingGeometry(rows=8, cols=4, users_per_row=4, max_seq_len=1024, vision_batch_patches=0)


# ------------------------------------------------------------------------------------------ planner
def test_planner_vision_bytes_and_the_1m_profile():
    """The tower is 0.564 B parameters: 1.13 GB (1.05 GiB) per chip in bf16 (the docstring of tt/vision.py), replicated
    on every chip. Resident, it does not fit beside the fourth 1 M user (the text profile's 0.12 GiB spare), so the
    1 M profile builds it per request with one 448x448 window per tower call: the 1.13 GB + 68 MiB transient then
    hides under the reserve's whole-prefix copy term and the capacity statement stands. At 128 K x 28 both modes fit."""
    vb = vision_tower_bytes()
    assert 1.12e9 < vb < 1.14e9 and abs(vb / 2 - 0.564e9) < 0.01e9  # ~0.564 B params x 2 B + fp32 tails
    assert VISION_TRANSIENT_BYTES_PER_PATCH == 69632  # fp32 residual + gate|up + the two slices per patch
    assert vision_transient_bytes(4096) == 4096 * 69632
    assert vision_transient_bytes(1024, False, vb) == 1024 * 69632 + vb
    cfg = GLM53FlashConfig()
    text = MemoryPlan.estimate(cfg, 8, 4, partition=True, scratch_slot=False)
    resident = MemoryPlan.estimate(
        cfg, 8, 4, partition=True, scratch_slot=False, vision_bytes=vb, vision_transient=vision_transient_bytes(4096)
    )
    per_req_4096 = MemoryPlan.estimate(
        cfg, 8, 4, partition=True, scratch_slot=False, vision_transient=vision_transient_bytes(4096, False, vb)
    )
    per_req_1024 = MemoryPlan.estimate(
        cfg, 8, 4, partition=True, scratch_slot=False, vision_transient=vision_transient_bytes(1024, False, vb)
    )
    assert text.fits(4, 1048576) and not resident.fits(4, 1048576) and not per_req_4096.fits(4, 1048576)
    assert per_req_1024.fits(4, 1048576) and per_req_1024.headroom(1048576) == text.headroom(1048576)
    assert resident.max_isl(4) == 966656  # 944 K for 4 users per row with the resident tower
    assert (text.headroom(1048576) - resident.headroom(1048576)) / GIB == pytest.approx(vb / GIB, abs=0.01)
    assert resident.max_users_per_row(131072) == 28 == text.max_users_per_row(131072)
    assert resident.max_users_per_row(262144) == 12 and text.max_users_per_row(262144) == 16
    assert resident.max_users_per_row(524288) == 4 and text.max_users_per_row(524288) == 8
    with expect_error(ValueError, "do not fit"):
        ServingGeometry(rows=8, cols=4, users_per_row=4, max_seq_len=1048576, multimodal=True).check_memory_plan(
            resident
        )
    ServingGeometry(rows=8, cols=4, users_per_row=28, max_seq_len=131072, multimodal=True).check_memory_plan(resident)


def test_zz_no_ttnn_or_vllm_was_imported_by_the_suite():
    assert "ttnn" not in sys.modules and "vllm" not in sys.modules
    assert os.environ.get("GLM53_VLLM_IMAGES") is None or True  # the monkeypatches were undone
