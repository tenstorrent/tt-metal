# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pick the mesh shape a run uses. sharding.py shards weights on a mesh; this picks the mesh.

Must run before any device is opened, since opening one freezes the descriptor path, the fixture's
shape param, and the fabric config. Constants here come from COST_MODEL_DATA.md.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from toy_problem.utils import _CABLE_BW

from models.tt_transformers.tt.auto_shard.cost_model.cost_models import placement_cost
from models.tt_transformers.tt.params import BETA, DECODE_STEPS, PREFILL_LEN
from models.tt_transformers.tt.auto_shard.cost_model.sharding import (
    AttentionShapes,
    MLPShapes,
    _drop_all_replicated,
    legal_correct_placements,
    legal_mlp_placements,
)

_DESCRIPTORS = Path(__file__).resolve().parents[5] / "tt_metal" / "fabric" / "mesh_graph_descriptors"

# The 0<->1 edge has 6 eth channels, the other three have 2, and a descriptor's count applies to the
# whole mesh. So runs confined to 0<->1 can claim 6; anything wider must use 2 or routing setup
# fails on the thin edges (control_plane.cpp:1194).
_COUNT6 = str(_DESCRIPTORS / "n300_2x2_mesh_graph_descriptor.textproto")
_COUNT2 = str(_DESCRIPTORS / "n300_2x2_count2_mesh_graph_descriptor.textproto")


@dataclass(frozen=True)
class MeshPlan:
    """How to bring the machine up for one candidate shape."""

    shape: tuple  # shape the model runs on
    open_shape: Union[tuple, int]  # fixture param; an int opens a line
    descriptor: str  # TT_MESH_GRAPH_DESC_PATH
    submesh: Optional[tuple]  # carve out of the opened mesh, or None for the whole thing


# Fabric must own every device, so sub-machine shapes open the 2x2 and carve. The 1x4 opens as a
# line and comes up as a ring; the 2x2 opens 2D and gets FABRIC_1D (Linear).
PLANS = {
    (1, 1): MeshPlan((1, 1), (2, 2), _COUNT6, (1, 1)),
    (1, 2): MeshPlan((1, 2), (2, 2), _COUNT6, (1, 2)),  # == MESH_DEVICE=N300x1x2
    (2, 2): MeshPlan((2, 2), (2, 2), _COUNT2, None),  # == MESH_DEVICE=N300x2x2
    (1, 4): MeshPlan((1, 4), 4, _COUNT2, None),  # == no MESH_DEVICE
}


@dataclass(frozen=True)
class ModelShapes:
    attn: AttentionShapes
    mlp: MLPShapes
    n_layers: int
    vocab_size: int  # picks the sampling path, see fixed_decode_cost


def model_shapes():
    """Weight shapes for $HF_MODEL, without opening a device.

    ModelArgs needs a mesh_device, and its _set_hf_params just calls AutoConfig on the same env
    var, so read that directly.
    """
    from transformers import AutoConfig

    name = os.getenv("HF_MODEL")
    if not name:
        raise ValueError("MESH_DEVICE=AUTO needs HF_MODEL set to pick a mesh (e.g. meta-llama/Llama-3.1-8B)")
    c = AutoConfig.from_pretrained(name).to_dict()
    c = {**c.get("text_config", {}), **{k: v for k, v in c.items() if k != "text_config"}}

    dim, n_heads = c["hidden_size"], c["num_attention_heads"]
    n_kv_heads = c.get("num_key_value_heads", n_heads)
    head_dim = c.get("head_dim") or dim // n_heads
    return ModelShapes(
        attn=AttentionShapes(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dim=dim,
            qkv_size=head_dim * (n_heads + 2 * n_kv_heads),
        ),
        mlp=MLPShapes(dim=dim, hidden_dim=c["intermediate_size"]),
        n_layers=c["num_hidden_layers"],
        vocab_size=c["vocab_size"],
    )


# Wires per mesh axis, as cable classes; stands in for fabric_link_report before open. Axis 1
# (columns) crosses cards over QSFP-DD/WARP100, axis 0 (rows) is the intra-card TRACE link, None is
# a size-1 axis that never splits.
#
#   (1,2)  on the fat 0<->1 edge, count-6 descriptor claims all 6 wires
#   (2,2)  one TRACE wire is dispatch-only; columns capped at the thin 3<->2 edge's 2 wires
#   (1,4)  a path across the 2x2 can't use column edges alone, so it crosses a TRACE hop
_LINKS = {
    (1, 1): (None, None),
    (1, 2): (None, ("QSFP_DD",) * 4 + ("WARP100",) * 2),
    (2, 2): (("TRACE",), ("QSFP_DD",) * 2),
    (1, 4): (None, ("TRACE",)),
}


def _beta(shape):
    """Per-axis all-reduce bandwidth (bytes/s), absolute.

    cost_model_analytical normalizes within one mesh, which makes every mesh's fastest axis BETA
    and is useless for comparing meshes. BETA was measured on the 1x4, whose only live axis is one
    TRACE hop, so treat it as the speed of a TRACE unit and scale with _CABLE_BW.
    """
    unit = BETA / _CABLE_BW["TRACE"]
    return [sum(_CABLE_BW[c] for c in wires) * unit if wires else BETA for wires in _LINKS[shape]]


# Measured per-decode-token seconds for everything after the last block: final norm, lm_head,
# sampling. Largest mesh-dependent term, and discontinuous, because the mesh shape decides which
# sampling path runs. model.py takes the on-device path only when vocab_size // num_devices <= 64K;
# model_auto_shard forces host on any 2D mesh.
#
# On-device cost is set by the padded TopK width, where doubling costs ~6.5x (issue #40399) and the
# lm_head matmul is noise. Host has no TopK, so lm_head dominates and cost tracks 2*dim*vocab flops.
_SAMPLING_ON_DEVICE_S = {32768: 3.4e-3, 65536: 22.0e-3}
_ON_DEVICE_VOCAB_LIMIT = 64 * 1024

# Host path, three additive terms:
#
#   C = base + PER_FLOP * (2 * dim * vocab)     lm_head matvec + host argmax, every mesh
#     + GATHER                                  logits gather, only when devices > 1
#     + GATHER_2D                               second gather over axis 0, only on a 2D mesh
#
# Least squares over six host measurements (Qwen 1x1/1x2/2x2/1x4, Llama 1x4, Gemma 1x4), worst
# residual 2.1%. Implied lm_head rate is 2.1e11 flop/s, about half TFLOPS_DECODE.
#
# A flat 7.0e-3 fit Llama and Qwen but only by coincidence -- Qwen's smaller dim cancels its larger
# vocab, so their lm_heads are within 4% in flops. Gemma has 2.7x the flops at 15.5 ms, where the
# flat constant was 2.2x low. Vocab alone doesn't fit either (off 10-13% on the 7-8B models).
_SAMPLING_HOST_BASE_S = 2.542e-4
_SAMPLING_HOST_PER_FLOP_S = 4.868e-12
_SAMPLING_HOST_GATHER_S = 1.518e-3
# Only one sample constrains this (Qwen 2x2), so it's unknown whether the second gather is constant
# or scales with payload; additive is the conservative read. A Gemma 2x2 sweep would settle it.
_SAMPLING_HOST_GATHER_2D_S = 1.782e-3


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def fixed_decode_cost(shape, model):
    """Seconds of per-token work outside the block stack."""
    devices = shape[0] * shape[1]
    # model.py hardcodes a split of 2 on a [1,1] mesh instead of using num_devices.
    sampling_splits = devices if devices > 1 else 2
    per_device_vocab = model.vocab_size // sampling_splits

    # Mirrors model.py's gate and model_auto_shard's 2D override. If either moves, this silently
    # ranks meshes on a path they no longer take.
    is_2d = shape[0] > 1 and shape[1] > 1
    if per_device_vocab > _ON_DEVICE_VOCAB_LIMIT or is_2d:
        # lm_head is unsplit here, so it costs the same everywhere; only the gathers vary by shape.
        host = _SAMPLING_HOST_BASE_S + _SAMPLING_HOST_PER_FLOP_S * 2 * model.attn.dim * model.vocab_size
        if devices > 1:
            host += _SAMPLING_HOST_GATHER_S
        if is_2d:
            host += _SAMPLING_HOST_GATHER_2D_S
        return host

    width = _next_pow2(per_device_vocab)
    if width in _SAMPLING_ON_DEVICE_S:
        return _SAMPLING_ON_DEVICE_S[width]
    # The table covers the only two widths our models reach on 4 chips. Falling back to the nearest
    # is a guess, and a bad one given the 6.5x step between entries.
    nearest = min(_SAMPLING_ON_DEVICE_S, key=lambda w: abs(w - width))
    logger.warning(f"auto-mesh: no sampling measurement for width {width}; using the {nearest} entry")
    return _SAMPLING_ON_DEVICE_S[nearest]


def _best_placement(shapes, shape, beta, prefill_len, decode_steps):
    """Sharding this mesh would actually run for one layer kind, and its cost.

    Same enumerators and cost function as select_sharding, so dry run and live run agree.
    """
    attention = isinstance(shapes, AttentionShapes)
    placements = legal_correct_placements(shapes, shape) if attention else legal_mlp_placements(shapes, shape)
    placements = _drop_all_replicated(placements, shape)
    if not placements:
        raise ValueError(f"no legal sharding for {shapes} on candidate mesh {shape}")
    cost = lambda p: placement_cost(p, shapes, shape, beta, prefill_len, decode_steps)  # noqa: E731
    best = min(placements, key=cost)
    return best, cost(best)


def mesh_cost(shape, model, prefill_len=PREFILL_LEN, decode_steps=DECODE_STEPS):
    """Cost of running `model` on `shape`, as t = C + n_layers * P. Host-only.

    Each mesh is scored at its own best sharding, so the comparison is best-vs-best. C is charged
    per decode step, not per layer: per-layer cost barely varies with mesh shape (under 4% across
    these two models) while C alone swings 3.4 -> 22.8 ms.

    Not modeled:
      * Prefill's fixed cost, so prefill-heavy workloads rank on per-layer terms alone.
        COST_MODEL_DATA.md has the TTFT numbers to fit it.
      * Batch. Every constant is batch 1 and tt_sampling pads to 32, so the two sampling paths
        converge as batch rises and may swap order.
      * Hop count. ALPHA is flat per all-reduce, so a 4-chip reduce costs the same as a 2-chip one.
        Measured comm doesn't grow with mesh size either, so the error is small.
      * DRAM feasibility. No memory_params here (batch and max_seq_len are pytest params, unknown
        at import), so a mesh too small for the weights gets costed instead of rejected.
    """
    beta = _beta(shape)
    chosen, costs = {}, []
    for kind, shapes in (("attn", model.attn), ("mlp", model.mlp)):
        chosen[kind], cost = _best_placement(shapes, shape, beta, prefill_len, decode_steps)
        costs.append(cost)
    total = sum(costs) * model.n_layers + decode_steps * fixed_decode_cost(shape, model)
    return total, chosen


def select_mesh_plan(workload=None):
    """Cheapest candidate by mesh_cost, ties broken at random.

    `workload` is the (prefill_len, decode_steps) the run will actually do, read off the case the
    demo is about to run. Ranking depends on that mix, so the params.py defaults are a fallback.

    Assigns TT_MESH_GRAPH_DESC_PATH outright rather than setdefault-ing, since toy_problem.utils
    setdefaults it at import and would pin every shape to the count-6 descriptor. Must run before
    the mesh_device fixture opens anything.
    """
    model = model_shapes()
    prefill_len, decode_steps = workload or (PREFILL_LEN, DECODE_STEPS)
    logger.info(f"auto-mesh: ranking for prefill_len={prefill_len}, decode_steps={decode_steps}")
    scored = {shape: mesh_cost(shape, model, prefill_len, decode_steps) for shape in PLANS}

    cheapest = min(cost for cost, _ in scored.values())
    dearest = max(cost for cost, _ in scored.values())
    plan = PLANS[random.choice([s for s, (cost, _) in scored.items() if cost == cheapest])]
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = plan.descriptor

    axes = lambda p: f"i{p.intermediate_axis}/m{p.model_axis}"  # noqa: E731
    for shape, (cost, chosen) in sorted(scored.items(), key=lambda kv: kv[1][0]):
        picked = " ".join(f"{kind}={axes(p)}" for kind, p in chosen.items())
        logger.info(f"auto-mesh: {str(shape):>6} {cost * 1e3:9.1f} ms  best sharding {picked}")
    logger.info(
        f"auto-mesh: spread {(dearest / cheapest - 1) * 100:.2f}% -> picked {plan.shape} "
        f"(opens {plan.open_shape}, submesh {plan.submesh}, {os.path.basename(plan.descriptor)})"
    )
    return plan
