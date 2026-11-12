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
