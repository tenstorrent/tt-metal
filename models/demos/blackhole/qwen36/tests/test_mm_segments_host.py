# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): a multimodal prefill of Qwen36Model in segments (tt/model.py).

vllm-tt-plugin prefills a multimodal prompt of a model with MambaSpec layers from position 0 in one step, and the
adapter splits it at a prefill checkpoint, which can fall inside an image; a request replayed after preemption
continues after its last image. Each segment must take its M-RoPE positions from the whole row and splice exactly the
vision rows of its own image tokens. These tests run prefill_traced_chunked's short path and _set_vision_merge with
the device calls replaced by recorders.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import model as qwen36_model

IMAGE = 9
DIM = 16


def _row(length, image_start, image_len):
    ids = torch.full((1, length), 5, dtype=torch.int64)
    ids[0, image_start : image_start + image_len] = IMAGE
    return ids


@pytest.fixture
def model(monkeypatch):
    """A Qwen36Model at TP=8 with 1024-token KV blocks whose RoPE staging and masked-bucket forward record their
    arguments."""
    monkeypatch.setattr(qwen36_model, "get_block_size", lambda caches: 1024)
    m = qwen36_model.Qwen36Model.__new__(qwen36_model.Qwen36Model)
    m._chunked_chunk_size = 2048
    m._chunked_trace_id = None
    m._paged_kv_caches = None
    m.num_devices = 8
    m._req_image_grid_thw = torch.tensor([[1, 32, 32]])
    m._req_video_grid_thw = None
    m.args = SimpleNamespace(hf_config=SimpleNamespace(image_token_id=IMAGE, video_token_id=10), dim=DIM)
    calls = []
    m._build_request_rope = lambda ids, vision: calls.append(("rope", ids.shape[1], vision is not None))

    def bucket(ids, page_table, **kw):
        calls.append(("bucket", ids.shape[1], kw["chunk_start"], kw["vis_row_offset"], kw["stage_rope"]))
        return "logits"

    m.prefill_masked_bucket = bucket
    return m, calls


def test_segments_take_the_whole_row_rope_and_their_own_vision_rows(model):
    """A 2000-token prompt with an image at [900, 1156) and a checkpoint after 1024: the first segment ends inside
    the image, the second starts 124 image tokens into it. Both stage the RoPE of all 2000 tokens."""
    m, calls = model
    row = _row(2000, 900, 256)
    vision = torch.zeros(256, DIM)
    m.prefill_traced_chunked(row[:, :1024], None, 1024, vision_tokens=vision, start_pos=0, rope_token_ids=row)
    m.prefill_traced_chunked(row, None, 2000, vision_tokens=vision, start_pos=1024, rope_token_ids=row)
    assert calls == [
        ("rope", 2000, True),
        ("bucket", 1024, 0, 0, False),
        ("rope", 2000, True),
        ("bucket", 976, 1024, 124, False),
    ]


def test_a_text_prefill_stages_its_own_rope(model):
    m, calls = model
    row = torch.full((1, 1500), 5, dtype=torch.int64)
    m.prefill_traced_chunked(row, None, 1500, start_pos=1024)
    assert calls == [("rope", 1500, False), ("bucket", 476, 1024, 0, False)]


def test_a_segment_splices_the_vision_rows_of_its_own_image_tokens(monkeypatch):
    """_set_vision_merge places rows [offset, offset + n) of the host copy at the segment's n image positions."""
    m = qwen36_model.Qwen36Model.__new__(qwen36_model.Qwen36Model)
    m.num_devices = 1
    m.args = SimpleNamespace(hf_config=SimpleNamespace(image_token_id=IMAGE, video_token_id=10), dim=DIM)
    m._req_image_grid_thw = torch.tensor([[1, 8, 10]])
    m._req_video_grid_thw = None
    m._vis_buf = SimpleNamespace(shape=(1, 64, DIM))
    m._vis_mask_buf = object()
    staged = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: staged.append(t) or t)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda src, dst: None)
    rows = torch.arange(40 * DIM, dtype=torch.float32).reshape(40, DIM).to(torch.bfloat16)
    segment = _row(64, 10, 20)
    m._set_vision_merge(segment, rows, vis_row_offset=15)
    vis_full, mask = staged
    assert torch.equal(vis_full[0, 10:30], rows[15:35])
    assert int(mask.sum()) == 20 and torch.count_nonzero(vis_full[0, :10]) == 0


def test_vision_rows_are_read_to_host_once(monkeypatch):
    m = qwen36_model.Qwen36Model.__new__(qwen36_model.Qwen36Model)
    m.num_devices = 1
    reads = []
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: reads.append(t) or torch.ones(3, DIM))
    host = m.vision_rows_host("device rows")
    assert reads == ["device rows"] and host.dtype == torch.bfloat16
    assert m.vision_rows_host(host) is host


# ------------------------------------------------------------------ an image and a video in one prompt

VIDEO = 10


def _mm_model(image_grid=None, video_grid=None):
    m = qwen36_model.Qwen36Model.__new__(qwen36_model.Qwen36Model)
    m.num_devices = 1
    m.args = SimpleNamespace(hf_config=SimpleNamespace(image_token_id=IMAGE, video_token_id=VIDEO), dim=DIM)
    m._req_image_grid_thw, m._req_video_grid_thw = image_grid, video_grid
    return m


def test_placeholder_tokens_follow_the_stashed_grids():
    """A request's vision rows splice into the placeholder tokens of the modalities it carries; with no grid the mask
    is HF's image mask, as before."""
    ids = torch.tensor([5, IMAGE, IMAGE, 5, VIDEO, VIDEO, VIDEO, 5])
    grid = torch.ones(1, 3)
    assert _mm_model(grid, None)._vision_placeholder_mask(ids).nonzero().flatten().tolist() == [1, 2]
    assert _mm_model(None, grid)._vision_placeholder_mask(ids).nonzero().flatten().tolist() == [4, 5, 6]
    assert _mm_model(grid, grid)._vision_placeholder_mask(ids).nonzero().flatten().tolist() == [1, 2, 4, 5, 6]
    assert _mm_model()._vision_placeholder_mask(ids).nonzero().flatten().tolist() == [1, 2]
    assert _mm_model(grid, grid)._vis_row_offset_for(ids[None], 5) == 3


@pytest.mark.parametrize("video_first", [False, True])
def test_vision_rows_pack_an_image_and_a_video_in_prompt_order(video_first):
    """vision_rows runs the tower once per modality, stashes both grids and packs the rows in the order of their
    placeholder tokens, so a segment splices the right modality's rows at each position."""
    m = _mm_model()
    image_rows = torch.arange(3 * DIM, dtype=torch.float32).reshape(3, DIM).to(torch.bfloat16)
    video_rows = -torch.arange(1, 1 + 4 * DIM, dtype=torch.float32).reshape(4, DIM).to(torch.bfloat16)
    calls = []
    m.get_image_features = lambda pixels, grid: calls.append(("image", grid)) or image_rows
    m.get_video_features = lambda pixels, grid: calls.append(("video", grid)) or video_rows
    img, vid = [IMAGE] * 3, [VIDEO] * 2 + [5] + [VIDEO] * 2  # the video's two frames are separated by a timestamp
    ids = torch.tensor([5] + (vid + [5] + img if video_first else img + [5] + vid) + [5])
    grid_i, grid_v = torch.tensor([[1, 2, 6]]), torch.tensor([[2, 2, 4]])
    rows = m.vision_rows(ids, image=("pixels", grid_i), video=("pixels", grid_v))
    assert [c[0] for c in calls] == ["image", "video"]
    assert m._req_image_grid_thw is grid_i and m._req_video_grid_thw is grid_v
    want = torch.cat([video_rows, image_rows] if video_first else [image_rows, video_rows])
    assert torch.equal(rows, want)
    m._vis_buf = SimpleNamespace(shape=(1, 16, DIM))
    m._vis_mask_buf = object()
    staged = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ttnn, "from_torch", lambda t, **kw: staged.append(t) or t)
        mp.setattr(ttnn, "copy_host_to_device_tensor", lambda src, dst: None)
        m._set_vision_merge(ids[None], rows)
    vis_full, mask = staged
    positions = torch.nonzero((ids == IMAGE) | (ids == VIDEO)).flatten()
    assert torch.equal(vis_full[0, positions], want) and int(mask.sum()) == 7


def test_vision_rows_of_one_modality_and_of_a_text_prompt():
    m = _mm_model()
    rows = torch.ones(2, DIM, dtype=torch.bfloat16)
    m.get_video_features = lambda pixels, grid: rows
    grid = torch.tensor([[1, 2, 4]])
    assert m.vision_rows(torch.tensor([5, VIDEO, VIDEO, 5]), video=("pixels", grid)) is rows
    assert m._req_video_grid_thw is grid and m._req_image_grid_thw is None
    assert m.vision_rows(torch.tensor([5, 5])) is None
    assert m._req_video_grid_thw is None and m._req_image_grid_thw is None


def test_vision_rows_refuse_rows_that_do_not_match_the_placeholders(expect_error):
    """The prompt must hold every placeholder token of its items: a row cut before its last image tokens raises."""
    m = _mm_model()
    m.get_image_features = lambda pixels, grid: torch.ones(4, DIM, dtype=torch.bfloat16)
    with expect_error(ValueError, "3 placeholder tokens 9 for 4 vision rows"):
        m.vision_rows(torch.tensor([5, IMAGE, IMAGE, IMAGE]), image=("pixels", torch.tensor([[1, 4, 4]])))


# --------------------------------------------------------------------------------- M-RoPE delta per row


def _rope_model():
    """A Qwen36Model with the Qwen3.8 RoPE setup (64 rotary dims, sections 11/11/10) on a stand-in mesh."""
    from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

    args = SimpleNamespace(
        rope_head_dim=64,
        max_seq_len=4096,
        rope_theta=10_000_000,
        mrope_section=[11, 11, 10],
        rope_attention_scaling=1.0,
        spatial_merge_size=2,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
    )
    m = qwen36_model.Qwen36Model.__new__(qwen36_model.Qwen36Model)
    m.args = args
    m.rope = Qwen36RoPESetup(None, args)
    return m


def _image_prompt(n_text_before, n_text_after, grid=(1, 8, 8)):
    """Token ids of text, one image of ``grid`` (t * h * w / 4 placeholder tokens), text."""
    n_image = grid[0] * grid[1] * grid[2] // 4
    return torch.tensor([[5] * n_text_before + [IMAGE] * n_image + [5] * n_text_after]), torch.tensor([list(grid)])


def test_decode_rope_at_the_row_delta_continues_the_prompts_mrope_table():
    """The verify and drafter rows of a multimodal request take decode RoPE at KV position + its M-RoPE delta. That is
    the M-RoPE table's own rotation for the prompt's trailing text, so it continues the table the prefill used."""
    m = _rope_model()
    ids, grid = _image_prompt(7, 9)
    delta = m.rope.build_request_rope(ids, image_grid_thw=grid)
    assert delta == -12  # 16 image tokens take 4 positions
    n = ids.shape[1]
    tail = torch.arange(n - 9, n)  # the text after the image
    cos, sin = m._rope_tp_cos_sin_decode_torch(tail, [delta] * len(tail))
    assert torch.equal(cos[0, :, 0], m.rope._req_cos[n - 9 : n]) and torch.equal(sin[0, :, 0], m.rope._req_sin[n - 9 :])
    plain, _ = m._rope_tp_cos_sin_decode_torch(tail)  # rope.rope_delta, the last prefill's
    assert torch.equal(plain, cos)


def test_each_row_takes_its_own_delta(expect_error):
    """Rows of different requests in one verify or drafter call: each row's rotation is that of its own delta."""
    m = _rope_model()
    pos = torch.tensor([30, 31, 500, 501])
    deltas = [0, 0, -12, -12]
    cos, sin = m._rope_tp_cos_sin_decode_torch(pos, deltas)
    for r in range(4):
        m.rope.rope_delta = deltas[r]
        c, s = m._rope_tp_cos_sin_decode_torch(pos[r : r + 1])
        assert torch.equal(cos[0, r], c[0, 0]) and torch.equal(sin[0, r], s[0, 0])
    with expect_error(ValueError, "3 M-RoPE deltas for 4 positions"):
        m._rope_tp_cos_sin_decode_torch(pos, deltas[:3])


def test_verify_takes_each_users_delta_for_its_rows(monkeypatch):
    """verify_traced gives user u's delta to all T of its rows; without deltas every row takes rope.rope_delta."""
    from unittest.mock import MagicMock

    B, T = 2, 3
    m = qwen36_model.Qwen36Model.__new__(qwen36_model.Qwen36Model)
    m._vfy_trace_id, m._vfy_T, m._vfy_B, m._vfy_kv_grouped = "trace", T, B, False
    m._vfy_pt_host, m.gdn_blocks, m.device, m.vocab_size = torch.zeros(B, 4, dtype=torch.int32), object(), None, 8
    for name in ("_vfy_token_buf", "_vfy_kvpos_buf", "_vfy_cos_buf", "_vfy_sin_buf", "_vfy_ids_out", "_vfy_rows_out"):
        setattr(m, name, MagicMock())
    seen = []
    m._rope_tp_cos_sin_decode_torch = lambda pos, deltas=None: seen.append((pos.tolist(), deltas)) or (None, None)
    monkeypatch.setattr(ttnn, "to_torch", lambda t, **kw: torch.zeros(B * T))
    m.verify_traced([[1] * T] * B, [10, 40], [0] * B, rope_deltas=[0, -12])
    m.verify_traced([[1] * T] * B, [10, 40], [0] * B)
    assert seen == [
        ([10, 11, 12, 40, 41, 42], [0, 0, 0, -12, -12, -12]),
        ([10, 11, 12, 40, 41, 42], None),
    ]


def test_drafter_step_takes_each_rows_delta(monkeypatch, expect_error):
    """ttnn_mtp_decode_forward rotates row i at position i + its own delta; without deltas, rope.rope_delta."""
    from models.demos.blackhole.qwen36.tt.attention import rope_tp

    m = qwen36_model.Qwen36Model.__new__(qwen36_model.Qwen36Model)
    m.args = SimpleNamespace(rope_head_dim=64, max_seq_len=4096, rope_theta=10_000_000)
    m.rope = SimpleNamespace(rope_delta=-3)
    m.device = None
    m.mtp = SimpleNamespace(forward_decode=lambda *a, **kw: ("logits", "hidden"))
    rotated = []
    monkeypatch.setattr(
        rope_tp, "rot_mats_decode", lambda dev, rd, msl, th, pos: rotated.append(pos.tolist()) or (0, 0)
    )
    monkeypatch.setattr(ttnn, "Tensor", type("Tensor", (), {}))
    m.ttnn_mtp_decode_forward("h", [1, 2], [100, 200], "pt", rope_deltas=[0, -12])
    m.ttnn_mtp_decode_forward("h", [1, 2], [100, 200], "pt")
    assert rotated == [[100, 188], [97, 197]]
    with expect_error(ValueError, "1 M-RoPE deltas for 2 positions"):
        m.ttnn_mtp_decode_forward("h", [1, 2], [100, 200], "pt", rope_deltas=[0])


def test_drafter_prompt_kv_takes_the_prompts_mrope_table(monkeypatch):
    """The drafter's prompt KV (ttnn_mtp_prefill_forward, run from the prefill's on_chunk) rotates each position with
    the M-RoPE table the prefill staged for the whole row, image positions included."""
    m = _rope_model()
    ids, grid = _image_prompt(100, 40)
    m.rope.build_request_rope(ids, image_grid_thw=grid)
    m.device, m._paged_kv_caches = None, None
    rotations = []

    def forward_prefill(hidden, tok, cos, sin, pt_full, **kw):
        rotations.append((cos, sin))

    m.mtp = SimpleNamespace(forward_prefill=forward_prefill, attention=SimpleNamespace(paged_k=torch.zeros(9)))
    monkeypatch.setattr(qwen36_model, "get_block_size", lambda caches: 64)
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    S = 256
    m.ttnn_mtp_prefill_forward("hidden", torch.zeros(1, S, dtype=torch.int32), 0, torch.arange(8)[None], valid_len=S)
    cos, sin = rotations[0]
    n = ids.shape[1]
    assert torch.equal(cos[0, 0, :n], m.rope._req_cos[:n]) and torch.equal(sin[0, 0, :n], m.rope._req_sin[:n])
    # past the prompt (bucket padding rows, which write the drafter's scratch block): text continuation positions
    assert torch.equal(
        cos[0, 0, n:], m._rope_tp_cos_sin_decode_torch(torch.arange(n, S), [m.rope.rope_delta] * (S - n))[0][0, :, 0]
    )


def test_a_checkpoint_inside_an_image_needs_no_second_segment(model):
    """A 2000-token prompt with an image at [900, 1156) and a checkpoint after 1024, inside the image: one eager
    forward over the whole row splices all 256 vision rows and computes the checkpoint 1024 tokens in."""
    m, calls = model
    m.prefill_masked_bucket = lambda ids, page_table, **kw: calls.append(
        ("bucket", ids.shape[1], kw["chunk_start"], kw["vis_row_offset"], kw["checkpoint_len"])
    )
    row = _row(2000, 900, 256)
    m.prefill_traced_chunked(
        row, None, 2000, vision_tokens=torch.zeros(256, DIM), start_pos=0, rope_token_ids=row, checkpoint_pos=1024
    )
    assert calls == [("rope", 2000, True), ("bucket", 2000, 0, 0, 1024)]
