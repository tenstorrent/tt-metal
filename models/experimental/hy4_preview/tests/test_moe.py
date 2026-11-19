# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: device feed-forward sublayers (tt/moe.py) vs the pure-torch ground truth (reference/moe.py),
tenstorrent/tt-metal#65890.

The dense FFN (layer 0 and the shared expert) and the MoE are compared stage by stage, so a failure
names where it is:
    FFN:  activation | down | end to end
    MoE:  router scores | router picks and weights | routed experts | shared expert | end to end
On one chip first (the arithmetic), then on the 8x4 Galaxy with the FFN sharded once across the 32
chips and the routed experts 8 to a chip, and last with layer 0-5 weights from the checkpoint.

Each stage is fed the reference's input for that stage, so the per-stage numbers do not compound.
The routed experts are checked with the reference's routing weights.

Expert picks. The top-k is a discrete choice: where a token's last pick and its next score are
closer than the error of the device's router (its input is BF16 and its matmul is not bit-exact) the
choice goes the other way, the token runs a different expert, and that costs PCC like an arithmetic
error without being one (one token of 32 on checkpoint layers 3 and 5 with random hidden states:
on layer 5 the BFP8 sublayer drops from 0.9999 to 0.9966, the BFP4 one from 0.9854 to 0.9822). So the router and
the arithmetic are judged apart. The router: every expert the device picks must score within
PICK_TOLERANCE of the reference's cut on the input the device saw, carry the reference's weight for
that pick within WEIGHT_TOLERANCE, and MIN_PICK_AGREEMENT of the tokens must have exactly the
reference's picks. The sublayer end to end: against the reference run on the device's picks with the
reference's own weights for them, so only the discrete choice is taken from the device; the PCC
against the reference's own picks is logged next to it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference.moe import Hy4FFN, Hy4MoE, MoEConfig
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
    pcc,
)
from models.experimental.hy4_preview.tt.common import MeshPlan, compute_config
from models.experimental.hy4_preview.tt.moe import TtHy4FFN, TtHy4MoE

HY4 = MoEConfig()
# Hy4's structure at 1/8 of every width, with all 256 experts: 8 to a chip on the 8x4 mesh.
SMALL = MoEConfig(hidden_size=768, intermediate_size=256, dense_intermediate_size=2304)
# One chip holds every expert, so fewer of them.
ONE_CHIP = MoEConfig(hidden_size=768, num_experts=32, top_k=4, intermediate_size=256, dense_intermediate_size=2304)
# Hy4's widths with 64 experts (2 to a chip): random weights for all 256 would be 39 GB on the host.
HY4_WIDE = MoEConfig(num_experts=64)
CONFIGS = {"small": SMALL, "hy4-wide": HY4_WIDE}

# weight dtype -> (weight dtype, per-stage threshold, end-to-end threshold)
DTYPES = {"bf16": (ttnn.bfloat16, 0.999, 0.998), "bfp8": (ttnn.bfloat8_b, 0.998, 0.995)}
# MoE precision -> (shared-expert dtype, routed-expert dtype, per-stage threshold, end-to-end threshold).
# "bfp4" is the module's default policy (routed experts BFP4, shared expert BFP8): expert_dtype=None.
# Measured on the 8x4 Galaxy, 2026-11-19, random hidden states. BFP4 routed experts: 0.9792 to 0.9813 with
# random weights; 0.9887, 0.9877, 0.9850, 0.9825, 0.9821 with the checkpoint's layers 1 to 5. The MoE sublayer
# end to end: 0.9863 (random weights, one chip) to 0.9934, lowest checkpoint layer 0.9854 (layer 5). BFP8
# experts on the same inputs: 0.9999.
BFP4_PCC_STAGE, BFP4_PCC_E2E = 0.975, 0.983
MOE_DTYPES = {
    "bf16": (ttnn.bfloat16, None, 0.999, 0.998),
    "bfp8": (ttnn.bfloat8_b, ttnn.bfloat8_b, 0.998, 0.995),
    "bfp4": (ttnn.bfloat8_b, None, BFP4_PCC_STAGE, BFP4_PCC_E2E),
}
# Tokens whose 8 picks must be exactly the reference's. A near tie between the 8th and 9th score can
# go either way on the device, whose matmul is not bit-exact with torch.
MIN_PICK_AGREEMENT = 0.95
# How far below the reference router's cut (its top_k-th biased score, on the BF16 input the device
# router saw) a device pick may score, and how far a device routing weight may be from the reference's
# weight for the same pick. Measured on the 8x4 Galaxy: no device pick below the cut at all (the picks
# that differ from the reference's differ because the input was rounded to BF16), and weights within
# 4.3e-5 on checkpoint layers 1-5 and 6.1e-4 with random weights.
PICK_TOLERANCE = 2e-3
WEIGHT_TOLERANCE = 2e-3
# An all-reduce on its own (test_all_reduce_gain): (|gain - 1|, worst |error| / sum of |parts|) for BF16,
# for fp32 where a chip's share of the width is whole tiles, and for fp32 where it is not (exact). The
# inexact ones are twice what the 8x4 Galaxy measures (3.0e-3 and 1.3e-2; 4.6e-4 and 1.7e-3).
ALL_REDUCE_BOUNDS = {"bf16": (6e-3, 3e-2), ("fp32", True): (1e-3, 4e-3), ("fp32", False): (1e-5, 1e-5)}
# |gain - 1| of a dense FFN on one chip (test_ffn_gain_by_fidelity).
FFN_GAIN = 2e-2


def _randomize(ref, seed, scale=1.0):
    """Random parameters at the scale a trained layer has: linear outputs of order ``scale``, and a
    correction bias large enough to change which experts win."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ref.parameters():
            if p.dim() == 1:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(scale * torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
    return ref


def _up(plan, t, mesh_mapper=None, dtype=ttnn.bfloat16):
    """[T, width] -> device [1, 1, T, width]."""
    return ttnn.from_torch(
        t.float().reshape(1, 1, *t.shape).contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=dtype,
        mesh_mapper=mesh_mapper,
    )


def _hidden_down(plan, t):
    """Device [1, 1, T, hidden/tp] per chip -> [T, hidden]."""
    return plan.to_torch_hidden_concat(t, dim=-1)[0, 0]


def _tokens(cfg, tokens, seed, scale=1.0):
    return scale * torch.randn(tokens, cfg.hidden_size, generator=torch.Generator().manual_seed(seed))


def _check_ffn(plan, ref, weight_dtype, pcc_stage, pcc_e2e, x, label):
    with torch.no_grad():
        r_act = ref.activation(x)
        r_out = ref(x)
    tt = TtHy4FFN(plan, ref.hidden_size, ref.intermediate_size, ref.weights(), weight_dtype=weight_dtype)
    d_x = _up(plan, x, plan.shard_hidden(3))

    d_act = tt.activation(d_x)
    check_pcc(f"{label} activation", r_act, plan.to_torch_head_concat(d_act, dim=-1)[0, 0], pcc_stage)
    d_out = tt.down(_up(plan, r_act, plan.shard(head_dim=3)))
    check_pcc(f"{label} down", r_out, _hidden_down(plan, d_out), pcc_stage)

    d_out = tt(d_x)
    assert tuple(d_out.shape) == (1, 1, x.shape[0], ref.hidden_size // plan.hidden_tp)
    check_pcc(f"{label} end to end", r_out, _hidden_down(plan, d_out), pcc_e2e)


def _weights_for(cfg, scores, picks):
    """The router's weights for a given choice of experts: ``scores`` [T, E] unbiased, ``picks`` [T, E]
    bool -> [T, E], the picked scores normalised and scaled, zero elsewhere."""
    chosen = scores * picks
    return chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor


def _check_moe(plan, cfg, ref, weight_dtype, pcc_stage, pcc_e2e, x, label, expert_dtype=None):
    T = x.shape[0]
    with torch.no_grad():
        r_scores = ref.scores(x)
        r_weights = ref.dense_weights(*ref.route(x))
        r_routed = ref.routed(x, r_weights)
        r_shared = ref.shared(x)
        r_out = ref(x)

    tt = TtHy4MoE(plan, cfg, ref.weights(), weight_dtype=weight_dtype, expert_dtype=expert_dtype)
    d_x = _up(plan, x, plan.shard_hidden(3))

    # the token whole on every chip
    d_whole = tt.whole(d_x)
    shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(d_whole)]
    check_pcc(f"{label} gathered token", x, shards[0], 0.9999)
    check_pcc(f"{label} gathered token, last chip vs chip 0", shards[0], shards[-1], 0.99999)

    # router: scores, then picks and weights
    check_pcc(f"{label} router scores", r_scores, plan.to_torch_replicated(tt.scores(d_whole)), 0.9999)
    d_weights = tt.route(d_whole)
    shards = [ttnn.to_torch(s)[0, 0].float() for s in ttnn.get_device_tensors(d_weights)]
    assert torch.equal(shards[0], shards[-1]), f"{label}: chips disagree on the routing"
    picks = shards[0] != 0
    assert (picks.sum(dim=-1) == cfg.top_k).all(), f"{label}: picks per token {picks.sum(dim=-1).unique().tolist()}"
    agreement = (picks == (r_weights != 0)).all(dim=-1).float().mean().item()
    logger.info(f"{label} router picks: {agreement:.4f} of {T} tokens match the reference exactly")
    assert agreement >= MIN_PICK_AGREEMENT, f"{label}: only {agreement:.4f} of tokens have the reference's picks"
    total = shards[0].sum(dim=-1)
    assert torch.allclose(total, torch.full_like(total, cfg.routed_scaling_factor), rtol=1e-3), total
    check_pcc(f"{label} router weights", r_weights, shards[0], 0.99 if agreement < 1 else 0.9999)
    # a pick that is not the reference's must be a near tie, and every pick must carry the reference's weight
    with torch.no_grad():
        seen = ref.scores(x.bfloat16().float())  # the device router's input is BF16
        biased = seen + ref.router_bias
    cut = biased.topk(cfg.top_k, dim=-1).values[:, -1]
    below = (cut - biased.masked_fill(~picks, float("inf")).min(dim=-1).values).max().item()
    off = (shards[0] - _weights_for(cfg, seen, picks)).abs().max().item()
    logger.info(
        f"{label} router: the worst device pick is {below:.2e} below the reference's cut; "
        f"the worst device weight is {off:.2e} from the reference's weight for that pick"
    )
    assert below <= PICK_TOLERANCE, f"{label}: a device pick scores {below:.2e} below the reference's cut"
    assert off <= WEIGHT_TOLERANCE, f"{label}: a device routing weight is {off:.2e} from the reference's"

    # routed experts on the reference's weights
    d_routed = tt.routed(d_whole, _up(plan, r_weights, dtype=ttnn.float32))
    check_pcc(f"{label} routed experts", r_routed, _hidden_down(plan, d_routed), pcc_stage)

    # the shared expert is never BFP4: it keeps the BFP8 / BF16 threshold
    shared_pcc = max(pcc_stage, DTYPES["bfp8"][1]) if tt.expert_dtype == ttnn.bfloat4_b else pcc_stage
    check_pcc(f"{label} shared expert", r_shared, _hidden_down(plan, tt.shared(d_x)), shared_pcc)

    # end to end: the reference follows the device's picks (its own weights for them); see the module docstring
    d_out = tt(d_x)
    assert tuple(d_out.shape) == (1, 1, T, cfg.hidden_size // plan.hidden_tp)
    out = _hidden_down(plan, d_out)
    logger.info(f"{label} end to end against the reference's own picks: pcc={pcc(r_out, out):.6f}")
    with torch.no_grad():
        r_followed = ref.routed(x, _weights_for(cfg, r_scores, picks)) + r_shared
    check_pcc(f"{label} end to end", r_followed, out, pcc_e2e)


# ---------------------------------------------------------------------------------------------
# One chip: the arithmetic, no collectives
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tokens", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_ffn_one_chip(device, tokens, dtype):
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    cfg = ONE_CHIP
    plan = MeshPlan(device, hidden_axis=None)
    ref = _randomize(Hy4FFN(cfg.hidden_size, cfg.dense_intermediate_size).eval(), seed=1)
    _check_ffn(plan, ref, weight_dtype, pcc_stage, pcc_e2e, _tokens(cfg, tokens, 2), "one chip ffn")


@pytest.mark.parametrize("tokens", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("dtype", list(MOE_DTYPES), ids=list(MOE_DTYPES))
def test_moe_one_chip(device, tokens, dtype):
    weight_dtype, expert_dtype, pcc_stage, pcc_e2e = MOE_DTYPES[dtype]
    cfg = ONE_CHIP
    plan = MeshPlan(device, hidden_axis=None)
    ref = _randomize(Hy4MoE(cfg).eval(), seed=1)
    x = _tokens(cfg, tokens, 2)
    _check_moe(plan, cfg, ref, weight_dtype, pcc_stage, pcc_e2e, x, f"one chip moe {dtype}", expert_dtype)


@pytest.mark.parametrize("fidelity", ["HiFi2", "HiFi4"])
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_ffn_gain_by_fidelity(device, dtype, fidelity):
    """Where the FFN's scale goes: the same weights and tokens at each weight dtype and each matmul
    fidelity. PCC is blind to an output that is uniformly too small, so the gain (the least-squares
    factor between device and reference) is logged and bounded next to it.

    Measured on one chip (2026-11-19): HiFi4 gives 0.9999 with BF16 weights and 0.9996 with BFP8;
    HiFi2 gives 0.9910 and 0.9908. So it is HiFi2, which tt/moe.py and tt/attention.py use with
    block-float weights, and not the weights' own rounding, that makes a BFP8 FFN 0.9% too small.
    PCC hardly sees it: 0.999913 (HiFi4) against 0.999901 (HiFi2) with BFP8 weights."""
    weight_dtype, _, pcc_e2e = DTYPES[dtype]
    cfg = ONE_CHIP
    plan = MeshPlan(device, hidden_axis=None)
    ref = _randomize(Hy4FFN(cfg.hidden_size, cfg.dense_intermediate_size).eval(), seed=1)
    x = _tokens(cfg, 32, 2).bfloat16().float()
    with torch.no_grad():
        r_out = ref(x)
    tt = TtHy4FFN(plan, ref.hidden_size, ref.intermediate_size, ref.weights(), weight_dtype=weight_dtype)
    tt.ckc = compute_config(getattr(ttnn.MathFidelity, fidelity))
    d_out = _hidden_down(plan, tt(_up(plan, x))).float()
    label = f"one chip ffn {dtype} weights {fidelity}"
    check_pcc(label, r_out, d_out, pcc_e2e)
    r, d = r_out.flatten().double(), d_out.flatten().double()
    gain = (torch.dot(r, d) / torch.dot(r, r)).item()
    logger.info(f"{label}: gain={gain:.5f}")
    assert abs(gain - 1) <= FFN_GAIN, f"{label}: gain={gain:.5f}"


def test_expert_clamp_bites(device):
    """Activations well past swiglu_limit: the device must clamp the routed experts as the reference does."""
    cfg = ONE_CHIP
    plan = MeshPlan(device, hidden_axis=None)
    ref = _randomize(Hy4MoE(cfg).eval(), seed=3, scale=6.0)
    x = _tokens(cfg, 32, 4)
    with torch.no_grad():
        weights = ref.dense_weights(*ref.route(x))
        clamped = ref.routed(x, weights)
        ref.cfg = MoEConfig(**{**cfg.__dict__, "swiglu_limit": 1e9})
        unclamped = ref.routed(x, weights)
        ref.cfg = cfg
    assert (clamped - unclamped).abs().max() > 0.1 * clamped.abs().max(), "the clamp does not bite in this case"
    tt = TtHy4MoE(plan, cfg, ref.weights(), weight_dtype=ttnn.bfloat16)
    d_routed = tt.routed(tt.whole(_up(plan, x)), _up(plan, weights, dtype=ttnn.float32))
    check_pcc("clamped routed experts", clamped, _hidden_down(plan, d_routed), 0.999)


# ---------------------------------------------------------------------------------------------
# 8x4 Galaxy: FFN sharded once (hidden over the 4-chip axis, intermediate over the 8-chip axis), experts EP32
# ---------------------------------------------------------------------------------------------
def _galaxy(fn):
    fn = pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)(fn)
    return pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)(fn)


def _sc1_plan(mesh_device):
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8)
    return plan


@_galaxy
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("which", ["dense", "shared"])
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("tokens", [1, 32], ids=["T1", "T32"])
def test_ffn_sc1(mesh_device, config, which, dtype, tokens):
    cfg = CONFIGS[config]
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    intermediate = cfg.dense_intermediate_size if which == "dense" else cfg.shared_intermediate_size
    ref = _randomize(Hy4FFN(cfg.hidden_size, intermediate).eval(), seed=1)
    x = _tokens(cfg, tokens, 2)
    _check_ffn(_sc1_plan(mesh_device), ref, weight_dtype, pcc_stage, pcc_e2e, x, f"sc1 {config} {which} ffn")


@_galaxy
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("dtype", list(MOE_DTYPES), ids=list(MOE_DTYPES))
@pytest.mark.parametrize("tokens", [1, 32], ids=["T1", "T32"])
def test_moe_sc1(mesh_device, config, dtype, tokens):
    cfg = CONFIGS[config]
    weight_dtype, expert_dtype, pcc_stage, pcc_e2e = MOE_DTYPES[dtype]
    ref = _randomize(Hy4MoE(cfg).eval(), seed=1)
    x = _tokens(cfg, tokens, 2)
    label = f"sc1 {config} moe {dtype}"
    _check_moe(_sc1_plan(mesh_device), cfg, ref, weight_dtype, pcc_stage, pcc_e2e, x, label, expert_dtype)


_ALL_REDUCE_CASES = [(1, "mixed"), (1, "positive"), (32, "mixed"), (64, "mixed"), (512, "mixed"), (512, "positive")]


@_galaxy
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("axis", ["hidden", "head"])
@pytest.mark.parametrize("width, values", _ALL_REDUCE_CASES, ids=[f"W{w}-{v}" for w, v in _ALL_REDUCE_CASES])
def test_all_reduce_gain(mesh_device, axis, dtype, width, values):
    """The collective every partial sum here travels through, on its own: one tensor per chip along
    ``axis``, summed by MeshPlan.all_reduce, against the same sum on the host.

    PCC cannot see a sum that comes out uniformly too large or too small, so what is measured is the
    gain (the least-squares factor between device and host), the mean signed error, and the worst
    error against the sum of the parts' magnitudes. The inputs are rounded to the tensor's dtype
    first: what is left is the collective's own arithmetic. ``values`` are partial sums of both signs
    (a projection's) or all positive (a sum of squares'); ``width`` 1 is the [T, 1] tensor a norm
    reduces.

    What the 8x4 Galaxy does (2026-11-19), and the bounds hold it to:
      * BF16: the sum comes out too large, by 0.13% to 0.20% over the 4-chip axis and 0.18% to 0.30%
        over the 8-chip axis, at every width and with either signs. A sublayer whose partial sums cross
        two or three BF16 all-reduces is 0.4% to 0.7% too large for that reason alone (with BF16 weights the
        decoder layer's sublayer outputs measure 1.0025 to 1.0075 against HF).
      * fp32 at widths 1, 32 and 64: exact (2e-7). ttnn.all_reduce takes its gather-and-local-sum path
        there, because a chip's share of the width is not a whole number of tiles.
      * fp32 at width 512: NOT exact. A chip's share is whole tiles, ttnn takes its reduce-scatter
        path, and the sum is 0.04% to 0.05% too small with single values off by up to 1.7e-3 of the
        parts' magnitude: about 10 bits. The norm and iHC reduce at most 9 fp32 columns and are on the
        exact path; a wide fp32 tensor must not be all-reduced where its precision is the point.
    """
    plan = _sc1_plan(mesh_device)
    mesh_axis = plan.hidden_axis if axis == "hidden" else plan.head_axis
    tp = plan.mesh_shape[mesh_axis]
    g = torch.Generator().manual_seed(11)
    parts = torch.randn(tp, 1, 32, width, generator=g)
    if values == "positive":
        parts = parts.abs() + 0.5
    if dtype == "bf16":
        parts = parts.bfloat16().float()
    mapper = plan.shard(hidden_dim=0) if axis == "hidden" else plan.shard(head_dim=0)
    d_parts = ttnn.from_torch(
        parts,
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=ttnn.bfloat16 if dtype == "bf16" else ttnn.float32,
        mesh_mapper=mapper,
    )
    shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(plan.all_reduce(d_parts, mesh_axis))]
    for index, shard in enumerate(shards[1:], start=1):
        assert torch.equal(shard, shards[0]), f"chip {index} holds a different sum than chip 0"
    ref = parts.sum(dim=0, keepdim=True)
    dev = shards[0]
    label = f"all_reduce {axis} axis ({tp} chips) {dtype} {values} W{width}"
    r, d = ref.flatten().double(), dev.flatten().double()
    gain = (torch.dot(r, d) / torch.dot(r, r)).item()
    signed = ((d - r) * r.sign()).mean().item() / r.abs().mean().item()
    worst = ((d - r).abs() / parts.abs().sum(dim=0).flatten().double()).max().item()
    logger.info(
        f"{label}: pcc={pcc(ref, dev):.6f} | gain={gain:.6f} | mean signed error / mean |sum| = {signed:+.2e} | "
        f"worst |error| / sum of |parts| = {worst:.2e}"
    )
    whole_tiles = width % (tp * ttnn.TILE_SIZE) == 0  # a chip's share of the width: the reduce-scatter path
    gain_bound, worst_bound = ALL_REDUCE_BOUNDS["bf16" if dtype == "bf16" else ("fp32", whole_tiles)]
    assert abs(gain - 1) <= gain_bound, f"{label}: gain={gain:.6f} (bound {gain_bound})"
    assert worst <= worst_bound, f"{label}: worst error {worst:.2e} of the parts' magnitude (bound {worst_bound})"


# ---------------------------------------------------------------------------------------------
# Checkpoint weights (random hidden states: a layer's real input needs the layers below it, #65893)
# ---------------------------------------------------------------------------------------------
@_galaxy
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_dense_layer0_sc1_real_weights(mesh_device, dtype):
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    cfg = HY4
    ref = Hy4FFN(cfg.hidden_size, cfg.dense_intermediate_size).eval()
    tensors = hy4_tensors([f"model.layers.0.mlp.{name}" for name in Hy4FFN.PARAMETER_NAMES])
    with torch.no_grad():
        for name, value in zip(Hy4FFN.PARAMETER_NAMES, tensors):
            getattr(ref, name.removesuffix(".weight")).copy_(value)
    x = _tokens(cfg, 32, 2)
    _check_ffn(_sc1_plan(mesh_device), ref, weight_dtype, pcc_stage, pcc_e2e, x, "sc1 real L0 dense ffn")


@_galaxy
@pytest.mark.parametrize("dtype", ["bfp4", "bfp8"])
@pytest.mark.parametrize("layer", range(1, 6), ids=lambda layer: f"layer{layer}")
def test_moe_sc1_real_weights(mesh_device, layer, dtype):
    """All 256 experts of one layer, 8 to a chip, at both expert precisions (a layer's experts are 19 GB to read)."""
    weight_dtype, expert_dtype, pcc_stage, pcc_e2e = MOE_DTYPES[dtype]
    cfg = HY4
    ref = Hy4MoE(cfg, init=False).eval()
    names = list(Hy4MoE.PARAMETER_NAMES)
    tensors = hy4_tensors([f"model.layers.{layer}.mlp.{name}" for name in names])
    with torch.no_grad():
        for name, value in zip(names, tensors):
            ref.parameter(name).copy_(value)
    del tensors
    x = _tokens(cfg, 32, 2)
    label = f"sc1 real L{layer} moe {dtype}"
    _check_moe(_sc1_plan(mesh_device), cfg, ref, weight_dtype, pcc_stage, pcc_e2e, x, label, expert_dtype)
