# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only coordinate and metric helpers for native-input prefill tests."""

import math

import torch


def assemble_head(shards, plane, head, *, stripe_tokens=256):
    """Join four SP shards in natural token order for one plane and one TP head."""
    if len(shards) != 32 or type(stripe_tokens) is not int or stripe_tokens <= 0:
        raise ValueError("Expected 32 shards and a positive stripe length")
    shape = tuple(shards[0].shape)
    if len(shape) != 4 or shape[1] != 1 or not shape[2] or shape[2] % stripe_tokens:
        raise ValueError("Each shard must contain complete SP stripes and one local KV head")
    if not 0 <= plane < shape[0] or not 0 <= head < 8:
        raise ValueError("Plane or TP head is out of range")
    if any(tuple(s.shape) != shape or s.dtype != shards[0].dtype or s.device.type != "cpu" for s in shards):
        raise ValueError("All shards must have matching CPU shape and dtype")
    # Local stripe 0 from SP0..3 precedes local stripe 1 from SP0..3.
    pieces = [shards[sp * 8 + head][plane, 0].reshape(-1, stripe_tokens, shape[-1]) for sp in range(4)]
    return torch.stack(pieces, dim=1).reshape(-1, shape[-1])


CHUNKS = ((0, 0, 1024), (1, 0, 1024), (0, 1024, 2048), (1, 1024, 2048))
LAYER_NAMES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def join_hidden_tp_replicas(shards):
    """Validate all 32 chip replicas before retaining one 256-token stripe from each SP."""
    if len(shards) != 32:
        raise ValueError("Expected 32 hidden shards")
    shape = tuple(shards[0].shape)
    if len(shape) != 4 or shape[:3] != (1, 1, 256):
        raise ValueError("Expected hidden shard [1,1,256,features]")
    if any(tuple(x.shape) != shape or x.dtype != shards[0].dtype or x.device.type != "cpu" for x in shards):
        raise ValueError("Hidden shards must share CPU shape and dtype")
    pieces = []
    for sp in range(4):
        first = shards[sp * 8]
        for tp in range(8):
            if not torch.equal(first, shards[sp * 8 + tp]):
                raise AssertionError(f"TP replica mismatch at SP={sp}, TP={tp}")
        pieces.append(first[0, 0])
    return torch.cat(pieces, dim=0).contiguous()


def full_hidden(chunks):
    """Only two complete, ordered chunks can form one native 2K layer output."""
    if set(chunks) != {0, 1024}:
        raise ValueError("Expected chunks beginning at 0 and 1024")
    left, right = chunks[0], chunks[1024]
    if left.ndim != 2 or left.shape[0] != 1024 or right.shape != left.shape or right.dtype != left.dtype:
        raise ValueError("Expected two equal [1024,features] hidden chunks")
    return torch.cat((left, right), dim=0)


def layer_input(embedding_chunks, hidden_layers, layer_idx):
    """Layer zero consumes native embedding; later layers consume the previous native output."""
    if type(layer_idx) is not int or not 0 <= layer_idx < 32:
        raise ValueError("Layer index outside [0,32)")
    return full_hidden(embedding_chunks if layer_idx == 0 else hidden_layers[layer_idx - 1])


def local_limits(cache_dtype, kind):
    """Use the published decoder bounds without an accumulated-error allowance."""
    if cache_dtype not in ("bfloat16", "bfloat8_b") or kind not in ("hidden", "k", "v"):
        raise ValueError("Unknown dtype or comparison kind")
    if kind == "hidden":
        return (0.999, 0.025) if cache_dtype == "bfloat16" else (0.999, 0.05)
    return (0.9999, 0.01) if cache_dtype == "bfloat16" else (0.999, 0.02)


def windows():
    """Each interval is one natural 256-token SP stripe inside one of the two chunks."""
    return [(chunk, sp, chunk + sp * 256, chunk + (sp + 1) * 256) for chunk in (0, 1024) for sp in range(4)]


def validate_observer_order(observed):
    if observed != list(range(32)):
        raise AssertionError("Every real layer must complete once in checkpoint order")


def score_row(metrics, expected, actual, limits, **coordinates):
    assert expected.shape == actual.shape and expected.numel()
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    pcc, nl2 = metrics(expected, actual)
    assert math.isfinite(pcc) and math.isfinite(nl2)
    return dict(coordinates, pcc=pcc, nl2=nl2, limits=limits, within_limits=pcc >= limits[0] and nl2 <= limits[1])
