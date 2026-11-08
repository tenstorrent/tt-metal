# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE op checks for QWEN36_SPEC_HOSTSET (work/notes/39_hostset_impl.md): the exact ttnn op sequences the verify and
drafter traces run on the packed input buffer, on ONE chip, against the host values. Seconds of device time, no model.

* uint32 ROW_MAJOR slice [row, 0:n] of the packed buffer == the host segment;
* typecast uint32 -> int32 on that slice (positions) is exact;
* ttnn.embedding(idx, RoPE table) -> cos / sin [1,1,n,rope_dim] TILE equals the host's _rope_tp_cos_sin_decode_torch rows;
* the anchor one-hot from an index into an identity table equals anchor_selector (bf16, TILE, padded rows);
* the narrow-read slice (reshape [1,1,B*T,V] -> [B,T,V], slice T axis) equals column 0 of each user;
* slices stay valid after the packed buffer is re-staged (a trace replays the slices every time).

Run (one chip):
    MESH_DEVICE=P150 pytest models/demos/blackhole/qwen36/tests/test_spec_hostset_ops.py -sv
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.rope import compute_rope_freqs
from models.demos.blackhole.qwen36.tt.spec_engine import anchor_selector
from models.demos.blackhole.qwen36.tt.tp_common import pack_u32_rows, rope_row_index

RD, THETA, S = 64, 10000000.0, 4096


def _up(t, dtype, layout, device):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _down(t):
    return ttnn.to_torch(t)


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("B,T", [(8, 4), (1, 4), (4, 8), (2, 12)])
def test_packed_slices_typecast_embedding_match_host(device, B, T):
    R = B * T
    g = torch.Generator().manual_seed(B * 10 + T)
    cos, sin = compute_rope_freqs(RD, S, THETA)
    tab_c = _up(cos.to(torch.bfloat16).contiguous(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    tab_s = _up(sin.to(torch.bfloat16).contiguous(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    eye = _up(torch.eye(R, dtype=torch.bfloat16).reshape(1, 1, R, R), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    pk = None
    for step in range(3):  # re-stage the same buffer: the derived tensors must follow it
        pos = torch.randint(0, S - 1, (R,), generator=g)
        pos[:3] = torch.tensor([0, 63, 64])[: min(3, R)]
        m = torch.randint(0, T, (B,), generator=g)
        anchor = torch.arange(B) * T + m
        host = pack_u32_rows([pos, rope_row_index(pos, 0, S), anchor])
        if pk is None:
            pk = _up(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
        else:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), pk)
        # positions: slice + typecast (+ reshape to the 1-D buffer shape the SDPA reads)
        p_u = ttnn.slice(pk, [0, 0], [1, R])
        p_i = ttnn.typecast(p_u, ttnn.int32)
        kvpos = ttnn.reshape(p_i, (R,))
        assert _down(kvpos).reshape(-1)[:R].tolist() == pos.to(torch.int32).tolist()
        # cos / sin: the lookup the batched decode loop runs
        ridx = ttnn.slice(pk, [1, 0], [2, R])
        c = ttnn.reshape(
            ttnn.embedding(ridx, tab_c, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG), (1, 1, R, RD)
        )
        s = ttnn.reshape(
            ttnn.embedding(ridx, tab_s, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG), (1, 1, R, RD)
        )
        idx = rope_row_index(pos, 0, S).to(torch.int64)
        assert torch.equal(_down(c).reshape(R, RD)[:R], cos.to(torch.bfloat16)[idx])
        assert torch.equal(_down(s).reshape(R, RD)[:R], sin.to(torch.bfloat16)[idx])
        # anchor one-hot
        aidx = ttnn.slice(pk, [2, 0], [3, B])
        sel = ttnn.reshape(
            ttnn.embedding(aidx, eye, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG), (1, 1, B, R)
        )
        assert torch.equal(_down(sel)[0, 0, :B, :R], anchor_selector(m.tolist(), B, T)[0, 0])
        for t in (p_u, kvpos, ridx, c, s, aidx, sel):
            ttnn.deallocate(t)


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ctrl_page_slice_is_the_whole_page(device):
    """A [1, N] uint32 slice (N a multiple of 16 words, as the gdn_spec_step ctrl page) returns the page unchanged."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_page, spec_ctrl_words

    B, Nv, T = 8, 6, 4
    N = spec_ctrl_words(B, Nv)
    mi = [0, 3, 1, 2, 0, 0, 3, 1]
    page = spec_ctrl_page(mi, B, Nv, parity=1).reshape(-1)
    host = pack_u32_rows([torch.arange(B * T), torch.arange(B * T), page])
    pk = _up(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    ctrl = ttnn.slice(pk, [2, 0], [3, N])
    assert tuple(ctrl.shape) == (1, N) and ctrl.dtype == ttnn.uint32 and ctrl.layout == ttnn.ROW_MAJOR_LAYOUT
    assert torch.equal(_down(ctrl).reshape(-1).to(torch.int32), page)


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("B,T", [(8, 4), (2, 12)])
def test_narrow_logits_slice_equals_column_zero(device, B, T):
    V = 248320
    g = torch.Generator().manual_seed(B)
    full = torch.randn(1, 1, B * T, V, generator=g).to(torch.bfloat16)
    x = _up(full, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    v = ttnn.reshape(x, (B, T, V))
    out = ttnn.slice(v, [0, 0, 0], [B, 1, V])
    got = _down(out).reshape(B, V)
    assert torch.equal(got, full.reshape(B * T, V)[::T])
