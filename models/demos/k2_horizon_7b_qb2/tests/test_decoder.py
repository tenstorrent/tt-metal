# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-weight HF comparisons for the serving decoder layers: prefill, then a traced decode step.

Both layer classes the model mixes per layer are built exactly as `tt/model.py` builds them, with the
selected precision policy. Lengths cover an aligned chunk and a chunk with a partial tail.
"""

import pytest
import torch
from transformers import DynamicCache

import ttnn

from ..tt.collective_buffers import DecodeCollectiveBuffers
from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_full_model_policy import OptimizedFullModelDecoder
from ..tt.precision_config import layer_policy, load_precision_config
from .reference import load_reference, pcc, read, real_activations, to_device

LAYER = 0
THRESHOLD = 0.995


def hf_forward(hf, rope, x, start, cache):
    """The HF layer on positions start..start+n with a causal mask over the cache plus these rows."""
    n = x.shape[1]
    positions = torch.arange(start, start + n).unsqueeze(0)
    embeddings = rope(x, positions)
    keys = torch.arange(start + n)[None, :]
    mask = torch.where(keys <= positions[0][:, None], 0.0, torch.finfo(torch.bfloat16).min).bfloat16()[None, None]
    with torch.no_grad():
        out = hf(x, position_embeddings=embeddings, attention_mask=mask, past_key_values=cache)
    return out, embeddings


def build_layer(kind, state, config, mesh):
    precision = load_precision_config()
    cls = MultichipDecoder if kind == "multichip" else OptimizedFullModelDecoder
    return cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER,
        mesh_device=mesh,
        collective_buffers=DecodeCollectiveBuffers(mesh),
        policy=layer_policy(precision, LAYER),
        ccl_dtype=precision["dtypes"]["ccl"],
        residual_dtype=precision["dtypes"]["residual"],
        norm_fidelity=precision["accumulation"]["norm_fidelity"],
        matmul_output_dtype=precision["dtypes"]["matmul_output"],
        math_approx_mode=precision["accumulation"]["math_approx_mode"],
        packer_l1_acc=precision["accumulation"]["packer_l1_acc"],
    )


@pytest.mark.parametrize("kind", ["multichip", "optimized_full_model"])
@pytest.mark.parametrize("length", [256, 1025])
def test_prefill_and_traced_decode(qb2_mesh, kind, length):
    torch.set_num_threads(8)
    config, state, hf, rope = load_reference(LAYER)
    layer = build_layer(kind, state, config, qb2_mesh)
    del state
    inputs = real_activations(length + 1)[None]  # (1, length + 1, hidden)

    capacity = ((length + 1 + 31) // 32) * 32 + 64
    pages = capacity // 32
    torch.manual_seed(2026 + length)
    table = torch.randperm(pages).reshape(1, -1).int()
    tt_table = to_device(table, qb2_mesh, True)
    caches = tuple(
        ttnn.zeros(
            (pages, 2, 32, 128),
            dtype=layer.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=qb2_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    )

    # Prefill from position zero; every output row is scored against HF.
    x = inputs[:, :length]
    cache = DynamicCache(config=config)
    expected, embeddings = hf_forward(hf, rope, x, 0, cache)
    tx = to_device(x.unsqueeze(0), qb2_mesh)
    tr = tuple(to_device(r.unsqueeze(1), qb2_mesh) for r in embeddings)
    plan = layer.prepare_prefill(seq_len=length, start_pos=0)
    out = layer.prefill_forward(tx, rope=tr, kv_cache=caches, page_table=tt_table, plan=plan)
    actual = read(out, qb2_mesh, -1).reshape_as(expected)
    out.deallocate(True)
    score = pcc(actual, expected)
    assert score >= THRESHOLD, f"prefill pcc {score}"

    # One decode step at position `length`, eager then traced; replay must be deterministic.
    dx = inputs[:, length : length + 1]
    expected, embeddings = hf_forward(hf, rope, dx, length, cache)
    td = to_device(dx.unsqueeze(0), qb2_mesh)
    tr = tuple(to_device(r.unsqueeze(0).repeat(1, 1, 32, 1), qb2_mesh) for r in embeddings)
    tp = to_device(torch.tensor([length], dtype=torch.int32), qb2_mesh, True)
    kwargs = dict(rope=tr, kv_cache=caches, page_table=tt_table, current_pos=tp)
    eager = layer.decode_forward(td, **kwargs)
    score = pcc(read(eager, qb2_mesh, -1).reshape_as(expected), expected)
    eager.deallocate(True)
    assert score >= THRESHOLD, f"eager decode pcc {score}"
    ttnn.synchronize_device(qb2_mesh)
    trace = ttnn.begin_trace_capture(qb2_mesh, cq_id=0)
    out = layer.decode_forward(td, **kwargs)
    ttnn.end_trace_capture(qb2_mesh, trace, cq_id=0)
    try:
        ttnn.execute_trace(qb2_mesh, trace, cq_id=0, blocking=True)
        traced = read(out, qb2_mesh, -1).reshape_as(expected)
        score = pcc(traced, expected)
        assert score >= THRESHOLD, f"traced decode pcc {score}"
        ttnn.execute_trace(qb2_mesh, trace, cq_id=0, blocking=True)
        assert torch.equal(traced, read(out, qb2_mesh, -1).reshape_as(expected))
    finally:
        ttnn.release_trace(qb2_mesh, trace)
