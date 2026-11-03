# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host tests of the SDPA program config (sdpa_program_config in tt/vision/common.py) and of M3VLAttention's SDPA call.
The configs are built with the real ttnn classes, and the attention runs on stand-ins for its device tensors and ops.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.common import sdpa_program_config
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig

from .vision_helpers import grid_device

CONFIG = MiniMaxM3VisionConfig()
NUM_HEADS, PADDED_HEAD_DIM = CONFIG.num_attention_heads, nearest_32(CONFIG.head_dim)
# The q chunk per core grid and patch count. Each comment gives the slowest core's query rows for q chunks of 128, 160,
# 192 and 256: the fewest rows win, the larger chunk on a tie.
Q_CHUNKS = {
    (11, 10): {
        16: 128,  # 128, 160, 192, 256
        144: 128,  # 128, 160, 192, 256
        256: 128,  # 128, 160, 192, 256
        324: 128,  # 128, 160, 192, 256
        768: 128,  # 128, 160, 192, 256
        1024: 192,  # 256, 320, 192, 256
        1296: 256,  # 256, 320, 384, 256
        1564: 128,  # 256, 320, 384, 512
        2160: 192,  # 384, 480, 384, 512
        2268: 192,  # 384, 480, 384, 512
        2304: 192,  # 384, 480, 384, 512
    },
    (12, 10): {
        16: 128,  # 128, 160, 192, 256
        144: 128,  # 128, 160, 192, 256
        256: 128,  # 128, 160, 192, 256
        324: 128,  # 128, 160, 192, 256
        768: 128,  # 128, 160, 192, 256
        1024: 160,  # 256, 160, 192, 256
        1296: 192,  # 256, 320, 192, 256
        1564: 256,  # 256, 320, 384, 256
        2160: 160,  # 384, 320, 384, 512
        2268: 160,  # 384, 320, 384, 512
        2304: 160,  # 384, 320, 384, 512
    },
    (13, 10): {
        16: 128,  # 128, 160, 192, 256
        144: 128,  # 128, 160, 192, 256
        256: 128,  # 128, 160, 192, 256
        324: 128,  # 128, 160, 192, 256
        768: 128,  # 128, 160, 192, 256
        1024: 128,  # 128, 160, 192, 256
        1296: 192,  # 256, 320, 192, 256
        1564: 256,  # 256, 320, 384, 256
        2160: 160,  # 384, 320, 384, 512
        2268: 160,  # 384, 320, 384, 512
        2304: 160,  # 384, 320, 384, 512
    },
}


def test_vision_sdpa_program_config_q_chunks():
    """Each grid and patch count gets the q chunk of Q_CHUNKS, k chunks of 128 and the whole grid.

    Changing a config leaves the next call's config as it was.
    """
    for (x, y), q_chunks in Q_CHUNKS.items():
        for patches, q_chunk in q_chunks.items():
            config = sdpa_program_config(grid_device(x, y), patches, NUM_HEADS)
            grid = config.compute_with_storage_grid_size
            assert (config.q_chunk_size, config.k_chunk_size, grid.x, grid.y) == (q_chunk, 128, x, y), (x, y, patches)
            config.q_chunk_size = 32
            assert sdpa_program_config(grid_device(x, y), patches, NUM_HEADS).q_chunk_size == q_chunk, (x, y, patches)


@pytest.mark.parametrize("patches", [1564, 2304])
def test_vision_attention_sdpa_call(monkeypatch, patches):
    """On a 12x10 grid the attention's SDPA gets its patch count's config and output_concat_heads=True, and its output
    goes straight to the output projection."""
    device, events = grid_device(12, 10), []  # events: (op, inputs, keyword arguments, output)

    def tensor(*shape):
        """A device tensor stand-in: its shape, padded to whole tiles, its device and its memory config, DRAM."""
        padded_shape = shape[:-2] + tuple(nearest_32(d) for d in shape[-2:])
        return SimpleNamespace(
            shape=shape, padded_shape=padded_shape, device=lambda: device, memory_config=lambda: ttnn.DRAM_MEMORY_CONFIG
        )

    def op(name, output):
        def run(*inputs, **kwargs):
            events.append((name, inputs, kwargs, output(*inputs, **kwargs)))
            return events[-1][3]

        return run

    def heads(x, num_heads, **kwargs):
        return tuple(tensor(1, num_heads, x.shape[-2], x.shape[-1] // (3 * num_heads)) for _ in range(3))

    def rope(x, *args, **kwargs):
        return tensor(*x.shape)

    def sdpa(q, k, v, **kwargs):
        _, num_heads, rows, head_dim = q.shape
        return tensor(1, 1, rows, num_heads * head_dim)  # with output_concat_heads=True

    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kwargs: tensor(*t.shape))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
    hidden = 64
    attention = M3VLAttention(
        device,
        NUM_HEADS,
        CONFIG.head_dim,
        wqkv_weight=torch.zeros(hidden, 3 * NUM_HEADS * PADDED_HEAD_DIM),
        wo_weight=torch.zeros(NUM_HEADS * PADDED_HEAD_DIM, hidden),
    )
    monkeypatch.setattr(ttnn, "linear", op("linear", lambda x, w, **kwargs: tensor(*x.shape[:-1], w.shape[-1])))
    monkeypatch.setattr(ttnn, "deallocate", op("free", lambda t: None))
    monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads", op("heads", heads))
    monkeypatch.setattr(ttnn.experimental, "rotary_embedding_llama", op("rope", rope))
    monkeypatch.setattr(ttnn.experimental, "nlp_concat_heads", op("nlp_concat_heads", lambda x, **kwargs: x))
    monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", op("sdpa", sdpa))

    cos_sin = tensor(1, 1, patches, PADDED_HEAD_DIM)
    out = attention(tensor(1, 1, patches, hidden), cos_sin, cos_sin)

    ops = [name for name, *_ in events]
    assert ops == ["linear", "heads", "free", "rope", "rope", "sdpa", "free", "free", "free", "linear", "free"], ops
    q_heads, k_heads, _ = events[1][3]
    for (_, (x, cos, sin, trans_mat), kwargs, _), heads in zip(events[3:5], (q_heads, k_heads)):
        assert x is heads and cos is cos_sin and sin is cos_sin and trans_mat is attention.trans_mat
        assert kwargs == dict(is_decode_mode=False, compute_kernel_config=attention.rope_compute_kernel_config), kwargs
    _, _, kwargs, attn = events[5]
    config = kwargs.pop("program_config")
    grid = config.compute_with_storage_grid_size
    assert (config.q_chunk_size, config.k_chunk_size, grid.x, grid.y) == (Q_CHUNKS[12, 10][patches], 128, 12, 10)
    assert kwargs == dict(
        is_causal=False,
        scale=attention.scale,
        compute_kernel_config=attention.sdpa_compute_kernel_config,
        output_concat_heads=True,
    )
    assert events[9][1][0] is attn and events[10][1][0] is attn and events[9][3] is out
