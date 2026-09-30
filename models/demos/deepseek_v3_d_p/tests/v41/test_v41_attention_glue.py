# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""V4.1 attention glue around ``sparse_sdpa`` (bead 8y7.9.7): contracts of the reshaped data paths.

* the fused head layout through the TP reshards (bead 8y7.9.9) equals the previous path bit for bit: q as
  ``wq_b`` emits it, resharded head->sequence (``in_dim`` 3: the chips' heads side by side) and split into
  sparse_sdpa's row-major heads with the RoPE tail by ``q_heads``, against nlp_create_qkv_heads + reshard (``in_dim``
  1) + the RoPE tail glue (slice / rotary / untilize / slice_write); the sparse_sdpa output taken to the tiled head
  groups with the inverse RoPE by ``o_heads`` and resharded sequence->head, against the inverse RoPE tail glue +
  tilize + reshard + nlp_concat_heads of each group.
* ``_index_rows`` returns, per query, its valid KV rows in [window, top-k] order followed by a sentinel tail, on the
  first chunk (queries with missing window rows: compacted) and on a later chunk (no compaction). The expected rows
  are built in torch from the same window table and top-k.
* the grouped output projection's ``nlp_concat_heads`` of [1, groups, S, rank] equals the per-group concatenation.

Shapes are the per-chip call-site shapes at chunk 5120 (production) and the small config.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41 import rope as v41_rope
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41ChunkTables
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS

MESH = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]
# chunk and config per size: per chip q [1, heads/tp, chunk/sp, head_dim]
SIZES = {"small": (SmallV41Config, 512), "production": (C, 5120)}
SENTINEL = -1


def _glue(mesh_device, cfg):
    """The attributes the glue methods read, without building the attention weights."""
    trans = ttnn.from_torch(
        get_rot_transformation_mat(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return SimpleNamespace(rope_dim=cfg.QK_ROPE_HEAD_DIM, trans_mat=trans, window=cfg.SLIDING_WINDOW)


def _mesh_tensor(mesh_device, t, dims, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    shape = tuple(mesh_device.shape)
    return ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=layout, mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims)
    )


def _down(mesh_device, t):
    shape = tuple(mesh_device.shape)
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(0, 1)))


def _cos_sin(mesh_device, cfg, chunk):
    """Per-chip cos / sin of the chip's SP rows (replicated over TP), as ``V41ChunkTables.rope``."""
    sp = mesh_device.shape[0]
    cos, sin = v41_rope.cos_sin(cfg, True, torch.arange(chunk))
    return tuple(_mesh_tensor(mesh_device, t.reshape(sp, 1, chunk // sp, -1), (0, None)) for t in (cos, sin))


def _old_rope_tail(g, t, cos, sin):
    """``TtV41Attention._rope_tail`` before 8y7.9.9: tiled [b, h, s, d] -> row-major, RoPE on the tail."""
    b, h, s, d = t.shape
    tail = ttnn.slice(t, [0, 0, 0, d - g.rope_dim], [b, h, s, d])
    tail = ttnn.experimental.rotary_embedding_llama(tail, cos, sin, g.trans_mat, is_decode_mode=False)
    out = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.experimental.slice_write(
        ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT), out, [0, 0, 0, d - g.rope_dim], [b, h, s, d], [1, 1, 1, 1]
    )
    return out


def _old_inverse_rope_tail(g, t, cos, sin):
    """``TtV41Attention._inverse_rope_tail`` before 8y7.9.9: row-major [b, h, s, d] (in place) -> tiled."""
    b, h, s, d = t.shape
    tail = ttnn.to_layout(ttnn.slice(t, [0, 0, 0, d - g.rope_dim], [b, h, s, d]), ttnn.TILE_LAYOUT)
    tail = ttnn.experimental.rotary_embedding_llama(tail, cos, ttnn.neg(sin), g.trans_mat, is_decode_mode=False)
    ttnn.experimental.slice_write(
        ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT), t, [0, 0, 0, d - g.rope_dim], [b, h, s, d], [1, 1, 1, 1]
    )
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_head_layout_matches_glue(mesh_device, device_params, size):
    cfg, chunk = SIZES[size]
    sp, tp = tuple(mesh_device.shape)
    g, ccl = _glue(mesh_device, cfg), V41Collectives(mesh_device)
    heads, d, groups = cfg.NUM_ATTENTION_HEADS, cfg.HEAD_DIM, cfg.O_GROUPS
    cos, sin = _cos_sin(mesh_device, cfg, chunk)
    cos_s, sin_s = (ttnn.mesh_partition(t, dim=2, cluster_axis=TP_AXIS) for t in (cos, sin))
    torch.manual_seed(0)
    # q per chip as wq_b emits it: [1, 1, S/sp, H/tp * d]
    q_host = torch.randn(sp, 1, chunk // sp, heads * d).to(torch.bfloat16)
    q = _mesh_tensor(mesh_device, q_host, (0, 3))
    old_q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
        q, num_heads=heads // tp, num_kv_heads=0, transpose_k_heads=False
    )
    old = _old_rope_tail(g, ccl.tp_all_to_all(old_q, in_dim=1, out_dim=2), cos_s, sin_s)
    new = q_heads(ccl.tp_all_to_all(q, in_dim=3, out_dim=2), cos_s, sin_s, g.trans_mat, heads, g.rope_dim)
    assert new.layout == ttnn.ROW_MAJOR_LAYOUT and tuple(new.shape) == tuple(old.shape), (new.shape, old.shape)
    assert torch.equal(_down(mesh_device, old), _down(mesh_device, new))

    # sparse_sdpa output: per chip [1, H, S/(sp*tp), d] row-major -> per chip [1, G/tp, S/sp, (H/G) * d] tiled
    o_host = torch.randn(sp, tp * heads, chunk // (sp * tp), d).to(torch.bfloat16)
    o_old = _mesh_tensor(mesh_device, o_host, (0, 1), layout=ttnn.ROW_MAJOR_LAYOUT)
    o_new = _mesh_tensor(mesh_device, o_host, (0, 1), layout=ttnn.ROW_MAJOR_LAYOUT)
    old = ccl.tp_all_to_all(_old_inverse_rope_tail(g, o_old, cos_s, sin_s), in_dim=2, out_dim=1)
    groups_local, rows = groups // tp, chunk // sp
    old = ttnn.reshape(old, [groups_local, heads // groups, rows, d])
    old = ttnn.reshape(ttnn.experimental.nlp_concat_heads(old), [1, groups_local, rows, heads // groups * d])
    new = o_heads(o_new, cos_s, ttnn.neg(sin_s), g.trans_mat, groups, g.rope_dim)
    new = ccl.tp_all_to_all(new, in_dim=2, out_dim=1)
    assert tuple(new.shape) == tuple(old.shape), (new.shape, old.shape)
    assert torch.equal(_down(mesh_device, old), _down(mesh_device, new))


def _expected_rows(window: torch.Tensor, comp: torch.Tensor | None, base: int, k: int) -> torch.Tensor:
    """[q, W] window rows (-1: none) and [q, t] top-k (-1 tail) -> [q, k]: valid window rows, valid top-k + base, -1."""
    out = torch.full((window.shape[0], k), SENTINEL, dtype=torch.int64)
    for i in range(window.shape[0]):
        valid = [int(v) for v in window[i] if v != SENTINEL]
        if comp is not None:
            valid += [int(v) + base for v in comp[i] if v != SENTINEL]
        out[i, : len(valid)] = torch.tensor(valid, dtype=torch.int64)
    return out


@pytest.mark.parametrize("start", [0, 5120], ids=["first_chunk", "later_chunk"])
@pytest.mark.parametrize("compressed", [False, True], ids=["window_only", "window_topk"])
@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_index_rows_valid_first(mesh_device, device_params, size, compressed, start):
    cfg, chunk = SIZES[size]
    sp, tp = tuple(mesh_device.shape)
    q_rows = chunk // (sp * tp)
    start = start * chunk // 5120  # a later chunk: the second one
    tables = V41ChunkTables(mesh_device, cfg, 2 * chunk, chunk, [2])
    window_width = cfg.SLIDING_WINDOW
    width = -(-(window_width + (cfg.INDEX_TOPK if compressed else 0)) // TOPK_ALIGN) * TOPK_ALIGN
    g = _glue(mesh_device, cfg)
    g.compact_rows = g.window - 1
    window = tables.window_rows(start)
    window_host = _down(mesh_device, window).reshape(-1, window_width)  # chip order = query order
    comp, comp_host, base = None, None, 128 + chunk
    if compressed:
        # top-k per query: distinct rows of the visible ones, sentinel tail (a varying number of valid slots)
        torch.manual_seed(1)
        topk = cfg.INDEX_TOPK
        comp_host = torch.full((sp * tp * q_rows, topk), SENTINEL, dtype=torch.int64)
        for i in range(comp_host.shape[0]):
            visible = (start + i + 1) // 2
            n = min(topk, visible)
            comp_host[i, :n] = torch.randperm(max(visible, 1))[:n]
        comp = _mesh_tensor(
            mesh_device, comp_host.reshape(sp, tp, q_rows, topk).to(torch.int32), (0, 1), dtype=ttnn.int32
        )
        comp = ttnn.to_layout(ttnn.typecast(comp, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)  # as topk_large_indices
    rows = TtV41Attention._index_rows(g, window, base, comp, start)
    assert rows.layout == ttnn.ROW_MAJOR_LAYOUT and rows.dtype == ttnn.uint32
    got = _down(mesh_device, ttnn.typecast(ttnn.to_layout(rows, ttnn.TILE_LAYOUT), ttnn.int32))
    got = got.reshape(-1, width).to(torch.int64)
    expected = _expected_rows(window_host, comp_host, base, width)
    assert torch.equal(got, expected), (got[:4, :8], expected[:4, :8])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_grouped_concat_heads(mesh_device, device_params):
    """wo_a output per chip [1, groups, S, rank] -> [1, 1, S, groups * rank] (groups side by side)."""
    sp, tp = tuple(mesh_device.shape)
    groups, rows, rank = C.O_GROUPS // tp, 5120 // sp, C.O_LORA_RANK
    torch.manual_seed(2)
    host = torch.randn(sp, tp, groups, rows, rank).to(torch.bfloat16)
    t = _mesh_tensor(mesh_device, host.reshape(sp, tp * groups, rows, rank), (0, 1))
    out = _down(mesh_device, ttnn.experimental.nlp_concat_heads(t))
    expected = host.permute(0, 1, 3, 2, 4).reshape(sp, tp, rows, groups * rank)
    assert torch.equal(out.reshape(sp, tp, rows, groups * rank), expected)
