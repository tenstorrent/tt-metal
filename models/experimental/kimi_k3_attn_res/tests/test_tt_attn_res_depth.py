# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rung 5: the 93-layer depth harness.

AttnRes sits on the residual highway for the whole stack — 186 chained softmax
mixtures. bf16 compounds through that on its own, so the gate is **relative**:
no worse than a bf16 torch implementation of the same math. An absolute PCC
number at this depth is either vacuous or fails for reasons that have nothing to
do with our kernels.

Both backends are driven through one shared `_walk` calling each one's own
shipped `attn_res_layer`, so the seal schedule and the read count cannot diverge
between them, and what this harness gates is the code a model calls rather than a
copy of it kept in a test.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_k3_attn_res.torch_functional.attn_res import (
    BLOCK_SIZE,
    EPS,
    NUM_LAYERS,
    AttnResStream,
    attn_res_layer,
)
from models.experimental.kimi_k3_attn_res.tt.attn_res import TtAttnRes
from models.experimental.kimi_k3_attn_res.tt.attn_res_stream import (
    TtAttnResStream,
    attn_res_layer as tt_attn_res_layer,
)

PROJ_STD = 0.02

# How much worse than torch-bf16 the device is allowed to be, in PCC.
DEPTH_PCC_SLACK = 1e-3


def _pcc(a, b):
    stacked = torch.stack((a.double().reshape(-1), b.double().reshape(-1)))
    return torch.corrcoef(stacked)[0, 1].item()


def _make_stack(num_tokens, hidden_size, num_layers, seed=0):
    """Module weights scale as `1/sqrt(num_layers)` so the residual stream's
    variance grows linearly with depth, as in a real stack. At unit scale each
    layer would roughly double the stream and 186 accumulations would overflow
    bf16 long before the harness said anything about AttnRes."""
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    module_scale = num_layers**-0.5

    hidden_states = randn(num_tokens, hidden_size)
    queries = [(1.0 + 0.1 * randn(hidden_size)) * (PROJ_STD * randn(hidden_size)) for _ in range(2 * num_layers + 1)]
    weights = [(module_scale * randn(hidden_size), module_scale * randn(hidden_size)) for _ in range(num_layers)]
    return hidden_states, queries[:num_layers], queries[num_layers:-1], queries[-1], weights


def _walk(stream, layer_fn, weights, q_pre, q_post, q_out, apply_module, record):
    """A whole stack, one layer at a time, through the backend's own `layer_fn`.

    A stand-in module is a single `[d]` multiply, so `apply_module` closes over
    that layer's weight to reach the `attn_fn(h)` shape the drivers take."""
    for layer_idx, (attn_weight, mlp_weight) in enumerate(weights):
        layer_fn(
            stream,
            layer_idx,
            q_pre[layer_idx],
            q_post[layer_idx],
            lambda h, w=attn_weight: apply_module(h, w),
            lambda h, w=mlp_weight: apply_module(h, w),
        )
        record(layer_idx, stream)

    return stream.read(q_out)


def _walk_torch(hidden_states, weights, q_pre, q_post, q_out, dtype):
    cast = lambda t: t.to(dtype)
    stream = AttnResStream(cast(hidden_states), block_size=BLOCK_SIZE, eps=EPS)
    curve = []
    out = _walk(
        stream,
        attn_res_layer,
        [(cast(a), cast(m)) for a, m in weights],
        [cast(q) for q in q_pre],
        [cast(q) for q in q_post],
        cast(q_out),
        lambda h, w: h * w,
        lambda _, s: curve.append(s.prefix_sum.float()),
    )
    return out.float(), curve


def _walk_device(mesh_device, hidden_states, weights, q_pre, q_post, q_out, hidden_size, record=None, op=None):
    """`record` defaults to collecting the whole per-layer curve. At production `T`
    that is 93 x [T, d] on the host, so the Phase-7 harness passes its own.

    `op` lets a caller supply a distributed `TtAttnRes`; its mappers then place
    both the stream and the per-layer weight vectors, so the Phase-8 harness reuses
    this walk instead of forking it."""
    op = op or TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)
    place = lambda t, mapper: ttnn.from_torch(
        t.reshape(1, 1, -1, hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mapper,
    )
    # The stream splits on tokens and hidden; a `[d]` module weight has no token
    # axis to split, so it takes the same placement as a folded query.
    to_tt = lambda t: place(t, op.stream_mapper)
    to_vector = lambda t: place(t, op.vector_mapper)
    from_tt = lambda t: ttnn.to_torch(t, mesh_composer=op.stream_composer).reshape(-1, hidden_size).float()
    stream = TtAttnResStream(op, to_tt(hidden_states), block_size=BLOCK_SIZE)
    curve = []
    out = _walk(
        stream,
        tt_attn_res_layer,
        [(to_vector(a), to_vector(m)) for a, m in weights],
        [op.to_query(q) for q in q_pre],
        [op.to_query(q) for q in q_post],
        op.to_query(q_out),
        ttnn.mul,
        record or (lambda _, s: curve.append(from_tt(s.prefix_sum))),
    )
    result = from_tt(out)
    ttnn.deallocate(out)
    stream.deallocate()
    return result, curve


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("hidden_size", [256, 7168])
def test_depth_fidelity(mesh_device, hidden_size):
    """93 layers, 8 seals, 187 reads. Gate the device against torch-bf16, not
    against an absolute number."""
    num_tokens = 64
    hidden_states, q_pre, q_post, q_out, weights = _make_stack(num_tokens, hidden_size, NUM_LAYERS)

    reference, reference_curve = _walk_torch(hidden_states, weights, q_pre, q_post, q_out, torch.float32)
    analog, analog_curve = _walk_torch(hidden_states, weights, q_pre, q_post, q_out, torch.bfloat16)
    device, device_curve = _walk_device(mesh_device, hidden_states, weights, q_pre, q_post, q_out, hidden_size)

    # A stream that has decayed to noise or overflowed would make every PCC below
    # meaningless, so state the regime before trusting the gate.
    norms = [c.norm().item() for c in reference_curve]
    logger.info(f"d={hidden_size}: stream norm {norms[0]:.1f} -> {norms[-1]:.1f} over {NUM_LAYERS} layers")
    assert torch.isfinite(reference).all(), "fp32 reference stream diverged"
    assert torch.isfinite(device).all(), "device stream diverged"

    device_pcc = _pcc(device, reference)
    analog_pcc = _pcc(analog, reference)
    logger.info(f"d={hidden_size}: final PCC device {device_pcc:.7f}, torch-bf16 {analog_pcc:.7f}")

    # Depth *dilutes* a per-read scale defect rather than compounding it: injecting
    # `ttnn.softmax(dim=1)`, which loses ~4 % of the mass per read, moves this ratio
    # only to 0.995 and leaves every PCC here passing. So this gate catches a gross
    # scale error and nothing finer — the op-level magnitude gate (D13) is the
    # primary detector, and it does catch that one. Measured as-shipped: 1.004.
    norm_ratio = (device.double().norm() / reference.double().norm()).item()
    logger.info(f"d={hidden_size}: output norm ratio device/fp32 {norm_ratio:.6f}")
    assert abs(norm_ratio - 1.0) <= 2e-2, f"d={hidden_size}: output norm ratio {norm_ratio:.6f} — gross scale error"

    per_layer = [
        (layer_idx, _pcc(d, r), _pcc(a, r))
        for layer_idx, (d, a, r) in enumerate(zip(device_curve, analog_curve, reference_curve))
    ]
    worst = min(per_layer, key=lambda row: row[1] - row[2])
    logger.info(f"d={hidden_size}: widest gap at layer {worst[0]} — device {worst[1]:.7f} vs bf16 {worst[2]:.7f}")
    for layer_idx, layer_device_pcc, layer_analog_pcc in per_layer:
        logger.info(f"  layer {layer_idx:2d}  device {layer_device_pcc:.7f}  torch-bf16 {layer_analog_pcc:.7f}")

    assert device_pcc >= analog_pcc - DEPTH_PCC_SLACK, (
        f"d={hidden_size}: device PCC {device_pcc:.7f} trails torch-bf16 {analog_pcc:.7f} "
        f"by more than {DEPTH_PCC_SLACK}"
    )
    assert worst[1] >= worst[2] - DEPTH_PCC_SLACK, (
        f"d={hidden_size}: layer {worst[0]} device PCC {worst[1]:.7f} trails torch-bf16 {worst[2]:.7f} "
        f"by more than {DEPTH_PCC_SLACK}"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_device_lifecycle_matches_torch(mesh_device):
    """The seal schedule and snapshot growth, on device. Seals fire at
    `{0, 12, ..., 84}`, `S` ramps 0 -> 8, and the walk performs 185 in-layer reads
    plus the model-level read."""
    executed_reads, parameter_sets = 186, 2 * NUM_LAYERS + 1
    # 187 folded queries but only 186 reads: `q_pre[0]` is loaded and never used,
    # because the layer-0 pre-attention read is skipped at `S == 0`.
    assert parameter_sets == executed_reads + 1
    num_tokens, hidden_size = 64, 256
    hidden_states, q_pre, q_post, q_out, weights = _make_stack(num_tokens, hidden_size, NUM_LAYERS)

    op = TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)
    stream = TtAttnResStream(
        op,
        ttnn.from_torch(
            hidden_states.reshape(1, 1, num_tokens, hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        ),
        block_size=BLOCK_SIZE,
    )

    reads = 0
    original_read = stream.read

    def counting_read(q):
        nonlocal reads
        reads += 1
        return original_read(q)

    stream.read = counting_read
    sealed_after = []
    _walk(
        stream,
        tt_attn_res_layer,
        [
            (
                ttnn.from_torch(
                    a.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
                ),
                ttnn.from_torch(
                    m.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
                ),
            )
            for a, m in weights
        ],
        [op.to_query(q) for q in q_pre],
        [op.to_query(q) for q in q_post],
        op.to_query(q_out),
        ttnn.mul,
        lambda _, s: sealed_after.append(s.num_sealed),
    )

    assert reads == executed_reads, f"{reads} reads, expected {executed_reads}"
    seal_layers = [layer_idx for layer_idx in range(NUM_LAYERS) if layer_idx % BLOCK_SIZE == 0]
    assert seal_layers == [0, 12, 24, 36, 48, 60, 72, 84]
    assert sealed_after[0] == 1 and sealed_after[-1] == 8
    assert sealed_after == sorted(sealed_after), "snapshot count must never shrink"
