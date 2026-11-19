# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: device attention sublayer (tt/attention.py) vs the pure-torch ground truth
(reference/attention.py), tenstorrent/tt-metal#65885.

Every stage is compared, so a failure names the projection it is in:
    latent (q_a, kv_a, both norms, RoPE) | q_b | gate | kv_b | the core with its sink | o_proj,
then the sublayer end to end. On one chip first (the arithmetic), then on the 8x4 Galaxy with
every projection sharded once across the 32 chips, at a small width and at Hy4's own, with BF16
and BFP8 weights, and last with layer 0-5 weights from the checkpoint.

Each stage is fed the reference's input for that stage, not the previous device stage's output,
so the per-stage numbers do not compound; the end-to-end check is the one that does.
"""

import pytest
import torch

import ttnn
from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, causal_mask, rope_cos_sin
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
)
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.common import MeshPlan

HY4 = AttentionConfig()
# Hy4's structure at about 1/8 of every width; still splits 4 ways (hidden) and 8 ways (heads), with
# both low ranks a whole number of tiles per hidden slice.
SMALL = AttentionConfig(
    hidden_size=768,
    num_heads=8,
    q_lora_rank=256,
    kv_lora_rank=128,
    qk_nope_head_dim=96,
    qk_rope_head_dim=64,
    v_head_dim=128,
)
CONFIGS = {"small": SMALL, "hy4": HY4}

# (weight dtype, per-stage threshold, end-to-end threshold)
DTYPES = {"bf16": (ttnn.bfloat16, 0.999, 0.998), "bfp8": (ttnn.bfloat8_b, 0.998, 0.995)}


def _reference(cfg, seed):
    """A reference layer with random parameters at the scale a trained layer has: linear outputs
    of order one, norm weights and sinks away from their trivial init."""
    ref = Hy4Attention(cfg).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in Hy4Attention.PARAMETER_NAMES:
            p = getattr(ref, name)
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5)
            elif name == "learnable_sink_param":
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=g))
    return ref


def _up(plan, t, mesh_mapper=None):
    return ttnn.from_torch(
        t.float().contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )


def _up_rows(plan, t, mesh_mapper=None):
    """[T, width] -> [1, 1, T, width]."""
    return _up(plan, t.reshape(1, 1, *t.shape), mesh_mapper)


def _heads_down(plan, t):
    """Device [1, heads/ctp, T, width] per chip -> reference order [H, T, width]."""
    return plan.to_torch_head_concat(t, dim=1)[0]


def _check_sublayer(plan, cfg, ref, weight_dtype, pcc_stage, pcc_e2e, seq_len, seed, label):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(seq_len, cfg.hidden_size, generator=g)
    positions = torch.arange(seq_len) + 3
    cos, sin = rope_cos_sin(positions, cfg)
    mask = causal_mask(seq_len)
    H, htp = cfg.num_heads, plan.hidden_tp

    with torch.no_grad():
        r_q_resid, r_k_pass, r_k_rope = ref.latent(x, cos, sin)
        r_q = ref.queries(r_q_resid, cos, sin)  # [H, S, qk]
        r_keys, r_values = ref.keys_values(r_k_pass, r_k_rope)
        r_gate = torch.nn.functional.linear(x, ref.linear_gate)  # logits, [S, H*v]
        r_attn = ref.core(r_q, r_keys, r_values, mask)  # [H, S, v]
        r_out = ref(x, cos, sin, mask)

    tt = TtHy4Attention(plan, cfg, ref.weights(), weight_dtype=weight_dtype)
    d_x = _up_rows(plan, x, plan.shard_hidden(3))
    # The device works in the cache's frame (tt/frames.py): interleaved RoPE tables, and the rope dims of
    # k_rope, the queries and the keys in interleaved order. Everything else equals the reference as is.
    d_cos, d_sin = (_up_rows(plan, t) for t in frames.rope_tables(cos, sin))
    rope = cfg.qk_rope_head_dim
    d_mask = _up_rows(plan, mask)

    # latent: q_a, kv_a, both norms, RoPE on k_rope. Whole on every chip.
    d_q_resid, d_k_pass, d_k_rope = tt.latent(d_x, d_cos, d_sin)
    for name, r, d in (
        ("q_resid", r_q_resid, d_q_resid),
        ("k_pass", r_k_pass, d_k_pass),
        ("k_rope", frames.rope_to_device(r_k_rope), d_k_rope),
    ):
        shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(d)]
        check_pcc(f"{label} {name}", r, shards[0], pcc_stage)
        check_pcc(f"{label} {name} last chip vs chip 0", shards[0], shards[-1], 0.99999)

    # q_b, gate, kv_b: per head, this chip's heads.
    d_q, d_gate, d_kv = tt.head_projections(d_x, _up_rows(plan, r_q_resid), _up_rows(plan, r_k_pass))
    check_pcc(f"{label} gate", r_gate, plan.to_torch_head_concat(d_gate, dim=-1), pcc_stage)
    d_qh = tt.queries(d_q, d_cos, d_sin)
    check_pcc(f"{label} q", frames.tail_to_device(r_q, rope), _heads_down(plan, d_qh), pcc_stage)
    d_keys, d_values = tt.keys_values(d_kv, _up_rows(plan, frames.rope_to_device(r_k_rope)))
    check_pcc(f"{label} keys", frames.tail_to_device(r_keys, rope), _heads_down(plan, d_keys), pcc_stage)
    check_pcc(f"{label} values", r_values, _heads_down(plan, d_values), pcc_stage)

    # core with the sink, on the reference's q / keys / values (both in the reference's frame: q . k is the same)
    head_mapper = plan.shard(head_dim=1)
    d_attn = tt.core(
        _up(plan, r_q[None], head_mapper),
        _up(plan, r_keys[None], head_mapper),
        _up(plan, r_values[None], head_mapper),
        d_mask,
    )
    check_pcc(f"{label} core", r_attn, _heads_down(plan, d_attn), pcc_stage)

    # o_proj on the reference's core output and gate logits
    d_out = tt.output(
        _up(plan, r_attn[None], head_mapper),
        _up_rows(plan, r_gate, plan.shard(head_dim=3)),
    )
    check_pcc(f"{label} o_proj", r_out, plan.to_torch_hidden_concat(d_out, dim=-1)[0, 0], pcc_stage)

    # end to end
    d_out = tt(d_x, d_cos, d_sin, d_mask)
    assert tuple(d_out.shape) == (1, 1, seq_len, cfg.hidden_size // htp)
    check_pcc(f"{label} end to end", r_out, plan.to_torch_hidden_concat(d_out, dim=-1)[0, 0], pcc_e2e)
    assert H % plan.head_tp == 0


# ---------------------------------------------------------------------------------------------
# One chip: the arithmetic, no collectives
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [32, 96], ids=["S32", "S96"])
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_attention_one_chip(device, seq_len, dtype):
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    plan = MeshPlan(device, hidden_axis=None)
    _check_sublayer(plan, SMALL, _reference(SMALL, seed=1), weight_dtype, pcc_stage, pcc_e2e, seq_len, 2, "one chip")


def test_sink_takes_probability_mass(device):
    """A large sink logit must shrink the core's output towards zero: the sink column is in the softmax."""
    cfg = SMALL
    plan = MeshPlan(device, hidden_axis=None)
    ref = _reference(cfg, seed=5)
    S = 32
    g = torch.Generator().manual_seed(6)
    q = torch.randn(1, cfg.num_heads, S, cfg.qk_head_dim, generator=g)
    keys = torch.randn(1, cfg.num_heads, S, cfg.qk_head_dim, generator=g)
    values = torch.randn(1, cfg.num_heads, S, cfg.v_head_dim, generator=g)
    mask = _up_rows(plan, causal_mask(S))

    def run(sink):
        weights = ref.weights()
        weights["learnable_sink_param"] = torch.full((cfg.num_heads,), sink)
        tt = TtHy4Attention(plan, cfg, weights, weight_dtype=ttnn.bfloat16)
        return ttnn.to_torch(tt.core(_up(plan, q), _up(plan, keys), _up(plan, values), mask)).float()

    base, sunk = run(0.0), run(30.0)
    assert sunk.abs().max() < 1e-3 * base.abs().max(), (sunk.abs().max().item(), base.abs().max().item())


# ---------------------------------------------------------------------------------------------
# 8x4 Galaxy: every projection sharded once, hidden over the 4-chip axis and heads over the 8-chip axis
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("seq_len", [32, 96], ids=["S32", "S96"])
def test_attention_sc1(mesh_device, config, dtype, seq_len):
    cfg = CONFIGS[config]
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8)
    _check_sublayer(plan, cfg, _reference(cfg, seed=1), weight_dtype, pcc_stage, pcc_e2e, seq_len, 2, f"sc1 {config}")


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("layer", range(6), ids=lambda layer: f"layer{layer}")
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_attention_sc1_real_weights(mesh_device, layer, dtype):
    """Layers 0-5 with the checkpoint's attention weights (random hidden states: the layer's real
    input needs the layers below it, which #65893 brings)."""
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    cfg = HY4
    prefix = f"model.layers.{layer}.self_attn."
    names = {n: prefix + n + ("" if n == "learnable_sink_param" else ".weight") for n in Hy4Attention.PARAMETER_NAMES}
    tensors = hy4_tensors(list(names.values()))
    ref = Hy4Attention(cfg).eval()
    with torch.no_grad():
        for name, value in zip(names, tensors):
            getattr(ref, name).copy_(value.reshape(getattr(ref, name).shape))
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    _check_sublayer(plan, cfg, ref, weight_dtype, pcc_stage, pcc_e2e, 96, 2, f"sc1 real L{layer}")
