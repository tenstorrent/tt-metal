# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: device feed-forward sublayers (tt/moe.py) vs the pure-torch ground truth (reference/moe.py),
tenstorrent/tt-metal#65890.

The dense FFN (layer 0 and the shared expert) and the MoE are compared stage by stage, so a failure
names where it is:
    FFN:  activation | down | end to end
    MoE:  router scores | router picks and weights | routed experts | shared expert | end to end
On one chip first (the arithmetic), then on the 4x8 Galaxy with the FFN sharded once across the 32
chips and the routed experts 8 to a chip, and last with layer 0-5 weights from the checkpoint.

Each stage is fed the reference's input for that stage, so the per-stage numbers do not compound.
The routed experts are checked with the reference's routing weights; the end-to-end check uses the
device's own picks.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference.moe import Hy4FFN, Hy4MoE, MoEConfig
from models.experimental.hy4_preview.tests.common import GALAXY_DEVICE_PARAMS, GALAXY_MESH, check_pcc, hy4_tensors
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.moe import TtHy4FFN, TtHy4MoE

HY4 = MoEConfig()
# Hy4's structure at 1/8 of every width, with all 256 experts: 8 to a chip on the 4x8 mesh.
SMALL = MoEConfig(hidden_size=768, intermediate_size=256, dense_intermediate_size=2304)
# One chip holds every expert, so fewer of them.
ONE_CHIP = MoEConfig(hidden_size=768, num_experts=32, top_k=4, intermediate_size=256, dense_intermediate_size=2304)
# Hy4's widths with 64 experts (2 to a chip): random weights for all 256 would be 39 GB on the host.
HY4_WIDE = MoEConfig(num_experts=64)
CONFIGS = {"small": SMALL, "hy4-wide": HY4_WIDE}

# weight dtype -> (weight dtype, per-stage threshold, end-to-end threshold)
DTYPES = {"bf16": (ttnn.bfloat16, 0.999, 0.998), "bfp8": (ttnn.bfloat8_b, 0.998, 0.995)}
# Tokens whose 8 picks must be exactly the reference's. A near tie between the 8th and 9th score can
# go either way on the device, whose matmul is not bit-exact with torch.
MIN_PICK_AGREEMENT = 0.95


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


def _check_moe(plan, cfg, ref, weight_dtype, pcc_stage, pcc_e2e, x, label):
    T = x.shape[0]
    with torch.no_grad():
        r_scores = ref.scores(x)
        r_weights = ref.dense_weights(*ref.route(x))
        r_routed = ref.routed(x, r_weights)
        r_shared = ref.shared(x)
        r_out = ref(x)

    tt = TtHy4MoE(plan, cfg, ref.weights(), weight_dtype=weight_dtype)
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

    # routed experts on the reference's weights
    d_routed = tt.routed(d_whole, _up(plan, r_weights, dtype=ttnn.float32))
    check_pcc(f"{label} routed experts", r_routed, _hidden_down(plan, d_routed), pcc_stage)

    check_pcc(f"{label} shared expert", r_shared, _hidden_down(plan, tt.shared(d_x)), pcc_stage)

    # end to end, on the device's own picks
    d_out = tt(d_x)
    assert tuple(d_out.shape) == (1, 1, T, cfg.hidden_size // plan.hidden_tp)
    check_pcc(f"{label} end to end", r_out, _hidden_down(plan, d_out), pcc_e2e)


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
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_moe_one_chip(device, tokens, dtype):
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    cfg = ONE_CHIP
    plan = MeshPlan(device, hidden_axis=None)
    ref = _randomize(Hy4MoE(cfg).eval(), seed=1)
    _check_moe(plan, cfg, ref, weight_dtype, pcc_stage, pcc_e2e, _tokens(cfg, tokens, 2), "one chip moe")


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
# 4x8 Galaxy: FFN sharded once (hidden over the 4 rows, intermediate over the 8 columns), experts EP32
# ---------------------------------------------------------------------------------------------
def _galaxy(fn):
    fn = pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)(fn)
    return pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)(fn)


def _sc1_plan(mesh_device):
    plan = MeshPlan(mesh_device, hidden_axis=0)
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
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("tokens", [1, 32], ids=["T1", "T32"])
def test_moe_sc1(mesh_device, config, dtype, tokens):
    cfg = CONFIGS[config]
    weight_dtype, pcc_stage, pcc_e2e = DTYPES[dtype]
    ref = _randomize(Hy4MoE(cfg).eval(), seed=1)
    x = _tokens(cfg, tokens, 2)
    _check_moe(_sc1_plan(mesh_device), cfg, ref, weight_dtype, pcc_stage, pcc_e2e, x, f"sc1 {config} moe")


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
@pytest.mark.parametrize("layer", range(1, 6), ids=lambda layer: f"layer{layer}")
def test_moe_sc1_real_weights(mesh_device, layer):
    """All 256 experts of one layer, 8 to a chip, BFP8. One dtype: a layer's experts are 19 GB to read."""
    weight_dtype, pcc_stage, pcc_e2e = DTYPES["bfp8"]
    cfg = HY4
    ref = Hy4MoE(cfg, init=False).eval()
    names = list(Hy4MoE.PARAMETER_NAMES)
    tensors = hy4_tensors([f"model.layers.{layer}.mlp.{name}" for name in names])
    with torch.no_grad():
        for name, value in zip(names, tensors):
            ref.parameter(name).copy_(value)
    del tensors
    x = _tokens(cfg, 32, 2)
    _check_moe(_sc1_plan(mesh_device), cfg, ref, weight_dtype, pcc_stage, pcc_e2e, x, f"sc1 real L{layer} moe")
