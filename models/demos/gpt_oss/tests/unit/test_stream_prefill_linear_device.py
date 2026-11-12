# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device checks of tt/stream_prefill_linear.py on Blackhole 1x4: the prefill matmul on decode stream tensors (QKV in
K / V-first slot order, o_proj with a bf8 input, router, LM head with the power-of-two zero columns) against torch on
the stream's own stored values (x @ W + the bias tiles' NBIAS rows): PCC >= 0.999 per device, every output column
tile in its logical place, zero columns exactly zero; resplit_vocab moves decode-split logits to the prefill split
exactly. The LM head streams its weights in groups (not resident); a small L1 budget gives several M blocks per core.

Run on a QuietBox 2 (Blackhole 1x4):
    pytest models/demos/gpt_oss/tests/unit/test_stream_prefill_linear_device.py
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gpt_oss.tt import stream_prefill_linear as spl
from models.demos.gpt_oss.tt.fused_decode.stream import (
    NBIAS,
    as_stream_tensor,
    linear_stream_rows,
    qkv_column_order,
    stream_linear_layout,
)
from models.demos.gpt_oss.tt.stream_prefill_linear import (
    StreamPrefillLinear,
    StreamSpec,
    resplit_vocab,
    unpack_stream_weight,
)

from ..test_factory import parametrize_mesh_with_fabric

TILE = 32

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="QuietBox 2 only (Blackhole 1x4); run locally.")


def _per_device(t):
    return [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)]


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _stored(layout_dev, spec):
    """(W [K, N], bias [N]) a device's stream holds: its weight tiles and the sum of its bias tiles' NBIAS rows."""
    w = unpack_stream_weight(layout_dev, spec).double()
    b = torch.zeros(spec.n, dtype=torch.float64)
    for bank in range(spec.banks):
        shard = layout_dev[:, bank * TILE : (bank + 1) * TILE].double()
        for j, col in enumerate(spec.columns(bank, 0, spec.real[bank])):
            t = (j * spec.kt + spec.kx) * TILE
            b[col * TILE : (col + 1) * TILE] = shard[t : t + NBIAS].sum(0)
    return w, b


def _column_order_ok(got, want, n_tiles):
    """Every output column tile correlates best with the reference column tile of the same index."""
    g = got.double().reshape(-1, n_tiles, TILE).transpose(0, 1).reshape(n_tiles, -1)
    r = want.double().reshape(-1, n_tiles, TILE).transpose(0, 1).reshape(n_tiles, -1)
    g = (g - g.mean(1, keepdim=True)) / g.std(1, keepdim=True).clamp_min(1e-30)
    r = (r - r.mean(1, keepdim=True)) / r.std(1, keepdim=True).clamp_min(1e-30)
    return torch.equal((g @ r.T).argmax(1), torch.arange(n_tiles))


@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize(
    "k, n, kind, rows, x_dtype, out_dtype, out_width, budget",
    [
        (2880, 1280, "qkv", 128, ttnn.bfloat16, ttnn.bfloat16, None, None),  # QKV at TP 4, K / V-first slot order
        (2880, 1280, "qkv", 1056, ttnn.bfloat16, ttnn.bfloat16, None, None),  # 33 row tiles: DST block tails
        (2880, 1280, "qkv", 4096, ttnn.bfloat16, ttnn.bfloat16, None, None),
        (1024, 2880, "plain", 128, ttnn.bfloat8_b, ttnn.bfloat8_b, None, None),  # o_proj at TP 4 (bf8 in / out)
        (1024, 2880, "plain", 2048, ttnn.bfloat8_b, ttnn.bfloat8_b, None, None),
        (2880, 32, "plain", 8, ttnn.bfloat16, ttnn.bfloat16, None, None),  # router, 8 tokens (one padded tile)
        (2880, 32, "plain", 1024, ttnn.bfloat16, ttnn.bfloat16, None, None),
        (2880, 50272, "plain", 32, ttnn.bfloat16, ttnn.bfloat8_b, 65536, None),  # LM head + power-of-two zero columns
        (1024, 2880, "plain", 640, ttnn.bfloat16, ttnn.bfloat16, None, 240 * 1024),  # small L1: several M blocks
    ],
    ids=["qkv128", "qkv1056", "qkv4096", "o128", "o2048", "router8", "router1024", "lm_head", "small_l1"],
)
def test_stream_prefill_linear(mesh_device, device_params, k, n, kind, rows, x_dtype, out_dtype, out_width, budget):
    if not is_blackhole():
        pytest.skip("The fused decode streams are Blackhole only")
    banks = mesh_device.dram_grid_size().x
    tp = mesh_device.shape[1]
    gen = torch.Generator().manual_seed(k + n + rows)
    order = qkv_column_order(n // TILE, 16 * 64 // TILE, banks) if kind == "qkv" else None
    layouts = [
        stream_linear_layout(torch.randn(k, n, generator=gen) * 0.05, torch.randn(n, generator=gen), banks, order)
        for _ in range(tp)
    ]
    srows = linear_stream_rows(mesh_device, k, n)
    stream = as_stream_tensor(mesh_device, layouts, srows, ttnn.bfloat8_b, None)
    x = ttnn.from_torch(
        torch.randn(1, 1, rows, k, generator=gen),
        dtype=x_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    op = StreamPrefillLinear(mesh_device, k, n, order)
    saved_cap = spl.L1_CAP
    if budget is not None:
        spl.L1_CAP = budget
    try:
        out = op(x, stream, out_dtype, out_width=out_width)
    finally:
        spl.L1_CAP = saved_cap
    width = out_width or n
    assert out.dtype == out_dtype and list(out.shape) == [1, 1, rows, width]
    spec = StreamSpec(k, n, banks, order)
    xs = _per_device(x)
    for d, (got, layout_dev) in enumerate(zip(_per_device(out), _per_device(stream))):
        w, b = _stored(layout_dev.reshape(srows, -1), spec)
        want = xs[d].reshape(rows, k).double() @ w + b
        got = got.reshape(rows, width)
        pcc = _pcc(got[:, :n], want)
        assert pcc >= 0.999, f"device {d}: PCC {pcc}"
        if n <= 2880:
            assert _column_order_ok(got[:, :n], want, n // TILE), f"device {d}: output columns out of order"
        assert not got[:, n:].any(), f"device {d}: zero columns not zero"
    # the stream is untouched
    for d, got in enumerate(_per_device(stream)):
        ref = ttnn.to_torch(ttnn.from_torch(layouts[d], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        assert torch.equal(got.reshape(srows, -1), ref), f"device {d}: stream changed"


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
