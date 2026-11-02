# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-only checks of the vision encoder's SDPA program config (tt/vision/common.py, sdpa_program_config) and of
M3VLAttention's SDPA call, which runs with it. No device: the configs are built with the real ttnn classes, and the
attention's forward runs on stand-ins for its device tensors and ops.

- On the 11x10, 12x10 and 13x10 core grids that Blackhole chips have depending on harvesting, h16w16, h32w32, h34w46,
  h40w54 and h48w48 get the q chunks pinned in Q_CHUNKS, with k chunks of 128 on the whole grid, and a new config at
  each call.
- The attention passes SDPA the config of its image's patch count and output_concat_heads=True, and gives SDPA's
  output, the heads side by side, to the output projection without nlp_concat_heads.

The 1-card leg selects these tests by name (test_vision_), as they have no mesh id.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_sdpa_program_config.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.common import sdpa_program_config

CONFIG = MiniMaxM3VisionConfig()
NUM_HEADS, PADDED_HEAD_DIM = CONFIG.num_attention_heads, nearest_32(CONFIG.head_dim)
# The q chunk per core grid and patch count (h16w16, h32w32, h34w46, h40w54 and h48w48). In the comments, the query
# rows of the slowest core for q chunks of 128, 160, 192 and 256: ceil(16 heads x ceil(patches / q) / cores) chunks of
# q rows each. The fewest rows win, the larger q on a tie.
Q_CHUNKS = {
    (11, 10): {
        256: 128,  # 128, 160, 192, 256
        1024: 192,  # 256, 320, 192, 256
        1564: 128,  # 256, 320, 384, 512
        2160: 192,  # 384, 480, 384, 512
        2304: 192,  # 384, 480, 384, 512
    },
    (12, 10): {
        256: 128,  # 128, 160, 192, 256
        1024: 160,  # 256, 160, 192, 256
        1564: 256,  # 256, 320, 384, 256
        2160: 160,  # 384, 320, 384, 512
        2304: 160,  # 384, 320, 384, 512
    },
    (13, 10): {
        256: 128,  # 128, 160, 192, 256
        1024: 128,  # 128, 160, 192, 256
        1564: 256,  # 256, 320, 384, 256
        2160: 160,  # 384, 320, 384, 512
        2304: 160,  # 384, 320, 384, 512
    },
}


def _device(x: int, y: int):
    """Stands in for the mesh device: the program configs only ask it for its compute grid."""
    return SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(x, y))


def test_vision_sdpa_program_config_q_chunks():
    """The q chunks of Q_CHUNKS with k chunks of 128 on the whole grid; changing a config leaves the next call's."""
    for (x, y), q_chunks in Q_CHUNKS.items():
        for patches, q_chunk in q_chunks.items():
            config = sdpa_program_config(_device(x, y), patches, NUM_HEADS)
            grid = config.compute_with_storage_grid_size
            assert (config.q_chunk_size, config.k_chunk_size, grid.x, grid.y) == (q_chunk, 128, x, y), (x, y, patches)
            config.q_chunk_size = 32
            assert sdpa_program_config(_device(x, y), patches, NUM_HEADS).q_chunk_size == q_chunk, (x, y, patches)


@pytest.mark.parametrize("patches", [1564, 2304])
def test_vision_attention_sdpa_call(monkeypatch, patches):
    """On a 12x10 grid the attention's SDPA gets the config of its patch count (q chunks of 256 at 1564 patches, 160
    at 2304) and output_concat_heads=True. Its output goes to the output projection as it is and is freed after it;
    nlp_concat_heads never runs."""
    device, events = _device(12, 10), []  # events: (op, inputs, keyword arguments, output)

    def tensor(*shape):
        """A device tensor stand-in: its shape, padded to whole tiles, and its device."""
        padded_shape = shape[:-2] + tuple(nearest_32(d) for d in shape[-2:])
        return SimpleNamespace(shape=shape, padded_shape=padded_shape, device=lambda: device)

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
