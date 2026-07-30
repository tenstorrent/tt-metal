# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rung 5: the 93-layer depth harness.

AttnRes sits on the residual highway for the whole stack — 186 chained softmax
mixtures. bf16 compounds through that on its own, so the gate is **relative**:
no worse than a bf16 torch implementation of the same math. An absolute PCC
number at this depth is either vacuous or fails for reasons that have nothing to
do with our kernels.

The two backends are driven through one shared `_walk`, so the read/seal/write
order provably cannot diverge between them. That is the whole point of making
`TtAttnResStream` interface-compatible with the torch one.
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
)
from models.experimental.kimi_k3_attn_res.tt.attn_res import TtAttnRes
from models.experimental.kimi_k3_attn_res.tt.attn_res_stream import TtAttnResStream

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


def _walk(stream, weights, q_pre, q_post, q_out, apply_module, free, record):
    """One layer of residual bookkeeping, repeated. Reference order.

    The pre-attention read is skipped at `S == 0` — only layer 0 — so `h` there
    aliases the stream's own `prefix_sum`, which `seal` then takes ownership of.
    Freeing it would free `block_residual`; hence `borrowed`."""
    for layer_idx, (attn_weight, mlp_weight) in enumerate(weights):
        h, borrowed = stream.prefix_sum, True
        if stream.num_sealed > 0:
            h, borrowed = stream.read(q_pre[layer_idx]), False

        if layer_idx % stream.block_size == 0:
            stream.seal()

        stream.accumulate(apply_module(h, attn_weight))
        if not borrowed:
            free(h)

        h = stream.read(q_post[layer_idx])
        stream.accumulate(apply_module(h, mlp_weight))
        free(h)
        record(layer_idx, stream)

    return stream.read(q_out)


def _walk_torch(hidden_states, weights, q_pre, q_post, q_out, dtype):
    cast = lambda t: t.to(dtype)
    stream = AttnResStream(cast(hidden_states), block_size=BLOCK_SIZE, eps=EPS)
    curve = []
    out = _walk(
        stream,
        [(cast(a), cast(m)) for a, m in weights],
        [cast(q) for q in q_pre],
        [cast(q) for q in q_post],
        cast(q_out),
        lambda h, w: h * w,
        lambda _: None,
        lambda _, s: curve.append(s.prefix_sum.float()),
    )
    return out.float(), curve


def _walk_device(mesh_device, hidden_states, weights, q_pre, q_post, q_out, hidden_size, record=None):
    """`record` defaults to collecting the whole per-layer curve. At production `T`
    that is 93 x [T, d] on the host, so the Phase-7 harness passes its own."""
    to_tt = lambda t: ttnn.from_torch(
        t.reshape(1, 1, -1, hidden_size), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    op = TtAttnRes(mesh_device, hidden_size=hidden_size, eps=EPS)
    stream = TtAttnResStream(op, to_tt(hidden_states), block_size=BLOCK_SIZE)
    curve = []
    out = _walk(
        stream,
        [(to_tt(a), to_tt(m)) for a, m in weights],
        [op.to_query(q) for q in q_pre],
        [op.to_query(q) for q in q_post],
        op.to_query(q_out),
        ttnn.mul,
        ttnn.deallocate,
        record or (lambda _, s: curve.append(ttnn.to_torch(s.prefix_sum).reshape(-1, hidden_size).float())),
    )
    result = ttnn.to_torch(out).reshape(-1, hidden_size).float()
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
        ttnn.deallocate,
        lambda _, s: sealed_after.append(s.num_sealed),
    )

    assert reads == executed_reads, f"{reads} reads, expected {executed_reads}"
    seal_layers = [layer_idx for layer_idx in range(NUM_LAYERS) if layer_idx % BLOCK_SIZE == 0]
    assert seal_layers == [0, 12, 24, 36, 48, 60, 72, 84]
    assert sealed_after[0] == 1 and sealed_after[-1] == 8
    assert sealed_after == sorted(sealed_after), "snapshot count must never shrink"
