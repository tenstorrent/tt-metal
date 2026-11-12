# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device checks of tt/stream_relayout.py on Blackhole 1x4: weights rebuilt from decode stream tensors equal the
stored bf8 copies bit for bit (QKV in K / V-first order, o_proj, router in bf16, LM head in bank groups with the
power-of-two padding), and resplit_vocab moves decode-split logits to the prefill split exactly.

Run on a QuietBox 2 (Blackhole 1x4):
    pytest models/demos/gpt_oss/tests/unit/test_stream_relayout_device.py
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gpt_oss.tt.fused_decode.stream import (
    as_stream_tensor,
    linear_stream_rows,
    qkv_column_order,
    stream_linear_layout,
)
from models.demos.gpt_oss.tt.stream_relayout import LinearStreamPlan, rebuild_linear, resplit_vocab

from ..test_factory import parametrize_mesh_with_fabric

TILE = 32

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="QuietBox 2 only (Blackhole 1x4); run locally.")


def _stored_bf8(w):
    """The bf8 values ttnn stores for w [K, N] (host tile conversion)."""
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def _per_device(t):
    return [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)]


@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize(
    "k, n, kind, groups, pad_n",
    [
        (2880, 1280, "qkv", 1, None),  # QKV at TP 4, K / V-first slot order
        (1024, 2880, "plain", 1, None),  # o_proj at TP 4
        (2880, 32, "router", 1, None),  # router (bf16 out)
        (128, 50272, "plain", 4, 65536),  # LM head columns (K reduced), bank groups + power-of-two padding
    ],
)
def test_rebuild_linear_matches_stored_copy(mesh_device, device_params, k, n, kind, groups, pad_n):
    if not is_blackhole():
        pytest.skip("The fused decode streams are Blackhole only")
    banks = mesh_device.dram_grid_size().x
    tp = mesh_device.shape[1]
    gen = torch.Generator().manual_seed(k + n)
    ws = [torch.randn(k, n, generator=gen) for _ in range(tp)]
    order = qkv_column_order(n // TILE, 16 * 64 // TILE, banks) if kind == "qkv" else None
    layouts = [stream_linear_layout(w, torch.randn(n, generator=gen), banks, order) for w in ws]
    rows = linear_stream_rows(mesh_device, k, n)
    stream = as_stream_tensor(mesh_device, layouts, rows, ttnn.bfloat8_b, None)
    plan = LinearStreamPlan(k, n, banks, order)
    dtype = ttnn.bfloat16 if kind == "router" else ttnn.bfloat8_b
    out = rebuild_linear(stream, plan, dtype, bank_groups=groups, pad_n=pad_n)
    assert out.dtype == dtype and out.memory_config().buffer_type == ttnn.BufferType.DRAM
    for d, got in enumerate(_per_device(out)):
        got = got.reshape(k, -1)
        assert got.shape[-1] == (pad_n or n)
        assert torch.equal(got[:, :n], _stored_bf8(ws[d])), f"device {d}: rebuilt weight differs from the bf8 copy"
        assert not got[:, n:].any(), f"device {d}: padding columns not zero"
    ttnn.deallocate(out)
    # the stream is untouched
    for d, got in enumerate(_per_device(stream)):
        assert torch.equal(got.reshape(rows, -1), _stored_bf8(layouts[d])), f"device {d}: stream changed"


@parametrize_mesh_with_fabric([(1, 4)])
def test_resplit_vocab(mesh_device, device_params):
    if not is_blackhole():
        pytest.skip("Blackhole 1x4 only")
    tp, vocab = mesh_device.shape[1], 201088
    vd = -(-vocab // tp // TILE) * TILE
    per_device = 1 << (vd - 1).bit_length()
    values = (torch.arange(tp * vd) % 251 - 125).float()  # bf8-exact, distinct per tile
    dec = torch.zeros(tp, 1, TILE, per_device)
    for d in range(tp):
        dec[d, 0, :, :vd] = values[d * vd : (d + 1) * vd]
    logits = ttnn.from_torch(
        dec,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    out = resplit_vocab(logits, tp, vd, per_device, cluster_axis=1)
    want = torch.zeros(tp * per_device)
    want[: tp * vd] = values
    for d, got in enumerate(_per_device(out)):
        got = got.reshape(TILE, per_device)
        assert torch.equal(got, want[d * per_device : (d + 1) * per_device].expand(TILE, -1)), f"device {d}"
