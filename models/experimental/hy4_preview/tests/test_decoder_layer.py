# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the device decoder layer (tt/decoder_layer.py) vs the pure-torch ground truth
(reference/decoder.py) and vs the real HF model's goldens (reference/hf_harness.py),
tenstorrent/tt-metal#65893.

Random weights, small config, on one chip and on the 8x4 Galaxy:
  * a dense layer and a MoE layer stage by stage, each stage fed the reference's input for it so the
    numbers do not compound:
        hc_pre | pre-norm | attention | norm + attention | hc_post, then the same at the FFN site,
    then the layer end to end, where every stage is compared again on the device's own inputs;
  * a stack of layers (dense, MoE, MoE) free-running from the embedding so the compounded error
    shows, and each layer fed the reference's input.

Expert picks. The router's top-k is a discrete choice, and where the last pick and the next score
are closer than the error on the router's input it goes the other way; that token then runs a
different expert, which is a legitimate outcome and not an arithmetic error, but it costs PCC like
one (rounding the MoE input to BF16 on the host flips 1 token of 32 in the one-chip case here, and
that alone takes the MoE output to PCC 0.9944). So with random weights the MoE is judged in two
parts. The router: on exactly the input the device router saw, every device pick must score within
PICK_TOLERANCE of the reference's cut, and the weights the device gives its picks must be the
reference's weights for those picks within WEIGHT_TOLERANCE. The arithmetic: the device is compared
against the reference run on the device's picks with the reference's own weights for them
(reference.decoder's ``expert_weights``), so only the discrete choice is taken from the device,
never a number; how many tokens got the reference's own picks, and the PCC against the reference's
own picks, are logged next to it. The checkpoint tests have no such allowance: the HF golden is
what it is, and the pick agreement is only reported.

Lengths. A layer is run at 1 token, at 50 (not a whole number of tiles) and at 32 / 96; the
checkpoint layers at the golden's 128 tokens and again on its first 1 and 47, which prefill being
causal makes the first rows of the same golden.

Checkpoint weights, layers 0-5, on the 8x4 Galaxy, against the HF goldens of a 128-token prompt:
each layer fed the golden input, and free-running from the golden embedding, with BFP8 weights
throughout, with BF16 weights, and with the deployment policy (BFP4 routed experts, BFP8 elsewhere).

What is reported for a layer. The streams carry the residual, so a PCC on them flatters the layer:
it stays high even if a sublayer is wrong. Next to it stands the PCC of the layer's update, the
streams it returns minus the streams it was given, which is the layer's own work. Free-running,
the update is taken from the embedding: everything the stack has added so far. PCC is blind to
scale, so the gain (the least-squares factor between device and reference, one when they agree) of
the update and of each sublayer's output is bounded too, and each pre-norm is also run on rows
whose mean of squares is below eps, where a wrong eps is a wrong scale.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rms_norm
from models.experimental.hy4_preview.reference.decoder import (
    LAYER_PARAMETERS,
    STAGES,
    DecoderConfig,
    Hy4DecoderLayer,
    Hy4DecoderStack,
    checkpoint_names,
)
from models.experimental.hy4_preview.reference.decoder import dense_prefill_inputs as reference_prefill_inputs
from models.experimental.hy4_preview.reference.ihc import IHCConfig, ihc_expand
from models.experimental.hy4_preview.reference.moe import MoEConfig
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
    pcc,
)
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import (
    TtHy4DecoderLayer,
    dense_prefill_inputs,
    hidden_to_device,
    hidden_to_torch,
    streams_to_device,
    streams_to_torch,
)
from models.experimental.hy4_preview.tt.ihc import ihc_expand as tt_ihc_expand

HY4 = DecoderConfig()


def _small(num_experts=256, top_k=8) -> DecoderConfig:
    """Hy4's structure at 1/8 of every width: splits 4 ways (hidden) and 8 ways (heads, experts)."""
    return DecoderConfig(
        attention=AttentionConfig(
            hidden_size=768,
            num_heads=8,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=96,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        moe=MoEConfig(
            hidden_size=768, num_experts=num_experts, top_k=top_k, intermediate_size=256, dense_intermediate_size=2304
        ),
        ihc=IHCConfig(dim=768),
    )


SMALL = _small()
# One chip holds every expert, so fewer of them.
ONE_CHIP = _small(num_experts=32, top_k=4)

# weight dtype -> (weight dtype, per-stage threshold, sublayer threshold, layer-update threshold)
# "bfp8" is BFP8 throughout. "bfp4" is the deployment policy, TtHy4MoE's default beside BFP8 weights:
# routed experts BFP4, everything else BFP8 (tenstorrent/tt-metal#65884). Its thresholds are set from the
# device (2026-11-18, random weights, one chip and the 8x4 Galaxy; BFP8 on the same inputs in brackets):
# MoE sublayer 0.9861 (0.9998), layer update 0.9868 (0.9998), and after a 3-layer stack running free
# the streams 0.9856 (0.9997) and the accumulated update 0.9824 (0.9996).
DTYPES = {
    "bf16": (ttnn.bfloat16, 0.999, 0.998, 0.998),
    "bfp8": (ttnn.bfloat8_b, 0.998, 0.995, 0.995),
    "bfp4": (ttnn.bfloat8_b, 0.998, 0.982, 0.978),
}
EXPERT_DTYPE = {"bf16": None, "bfp8": ttnn.bfloat8_b, "bfp4": None}
PCC_IHC = 0.999  # iHC pieces and the streams, which are fp32 whatever the weights are
# The streams after a layer: fp32 arithmetic, but they carry the MoE's error, which BFP4 experts raise.
PCC_STREAMS = {"bf16": PCC_IHC, "bfp8": PCC_IHC, "bfp4": 0.982}
PCC_NORM = 0.9999
# How far below the reference router's cut (its top_k-th biased score, on the device's own input) a
# device pick may score. The device's router matmul is HiFi4 with an fp32 result, about 3e-4 off on
# an order-one logit, a quarter of that on the sigmoid score.
PICK_TOLERANCE = 2e-3
# How far a device routing weight may be from the reference's weight for the same pick on the same
# input. The weights are order 2.827 / top_k; test_moe.py measures 2e-4 on the device.
WEIGHT_TOLERANCE = 2e-3
# |gain - 1| of a sublayer's output and of a layer's update: the device's is the reference's times
# the gain, plus noise.
GAIN_TOLERANCE = 0.03
# Checkpoint weights against the BF16 goldens, per weight dtype. Set from the 8x4 Galaxy (2026-11-19,
# layers 0-5 on the 128-token golden), each about twice as far from one as the lowest of the six layers,
# which is in the comment (bfp8 / bf16 / bfp4). What sets the floor with BFP8 and BF16 weights is not
# arithmetic but the router: on the golden MoE input the device makes HF's picks on 128 of 128 tokens
# in every layer, on its own MoE input (PCC 0.99995 with the golden's) 5% to 13% of the tokens get at
# least one other pick, and mlp_out stands at 0.998 to 0.9999. BFP4 experts add their own error.
#   "mlp"          mlp_out, fed the golden input                         0.998362 / 0.998101 / 0.991671
#   "streams"      streams_out, fed the golden input                     0.999892 / 0.999910 / 0.997669
#   "update"       the layer's update, fed the golden input              0.999091 / 0.999123 / 0.992477
#   "prefix"       the update on the first 1 or 47 tokens alone, vs HF   0.997751 / 0.999202 / 0.991246
#   "free_streams" the streams, free-running from the embedding          0.999734 / 0.999717 / 0.997640
#   "free_update"  the accumulated update, free-running                  0.999767 / 0.999756 / 0.997961
#   "gain"         |gain - 1| of attn_out, mlp_out and every update      0.0108   / 0.0075   / 0.0187
REAL = {
    "bfp8": dict(mlp=0.996, streams=0.9995, update=0.998, prefix=0.995, free_streams=0.9994, free_update=0.9995),
    "bf16": dict(mlp=0.996, streams=0.9995, update=0.998, prefix=0.995, free_streams=0.9994, free_update=0.9995),
    "bfp4": dict(mlp=0.983, streams=0.995, update=0.985, prefix=0.982, free_streams=0.995, free_update=0.995),
}
REAL_GAIN_TOLERANCE = {"bfp8": 0.025, "bf16": 0.025, "bfp4": 0.04}
# The stages no expert has touched, the same at every weight dtype: the attention output (lowest
# 0.999690, layer 5 with BFP8 weights) and the iHC / pre-norm stages on either side of it (0.999945).
PCC_REAL_ATTN = 0.999
PCC_REAL_STAGE = 0.9998
# The update on a prefix against the same rows of the device's own full run (lowest 0.998254, layer 3
# on 47 tokens; 0.99996 on one token).
PCC_REAL_PREFIX_VS_FULL = 0.996
# Prefixes of the golden prompt a checkpoint layer is run on again: one token, and a length that is
# not a whole number of tiles.
REAL_PREFIXES = (1, 47)
# One token; a whole tile; a length that is not a whole number of tiles; three tiles.
SEQ_LENS = (1, 32, 50, 96)
STAGE_KIND = {"attn_post": "post", "mlp_post": "post", "streams_in": "streams", "streams_mid": "streams"}
STAGE_KIND["streams_out"] = "streams"


def _randomize(layer: Hy4DecoderLayer, seed: int) -> Hy4DecoderLayer:
    """Random parameters at the scale a trained layer has, every one away from its trivial init:
    linear outputs of order one, norm weights and sinks spread around their init, iHC logits of order
    one with distinct pre / post scales, a router bias large enough to change which experts win."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, path in layer.parameter_names.items():
            p = layer.get_parameter(path)
            if name.endswith("hc_fn"):
                value = 0.02 * torch.randn(p.shape, generator=g)
            elif name.endswith("hc_scale"):
                value = torch.tensor([1.0, 0.5])
            elif name.endswith(("hc_base", "learnable_sink_param")):
                value = torch.randn(p.shape, generator=g)
            elif name.endswith("e_score_correction_bias"):
                value = 0.05 * torch.randn(p.shape, generator=g)
            elif p.dim() == 1:  # norm weights
                value = 1.0 + 0.25 * torch.randn(p.shape, generator=g)
            else:
                value = torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5
            if name == "mlp.gate.weight":
                # The device holds the router in BF16, which is exact for the checkpoint. Random values
                # are made exact too, so the device and the reference route with the same weights.
                value = value.bfloat16().float()
            p.copy_(value)
    return layer.eval()


def _streams(cfg, seq_len, seed):
    return torch.randn(seq_len, cfg.ihc.n, cfg.hidden_size, generator=torch.Generator().manual_seed(seed))


def _down(plan, cfg, key, tensor):
    """A device stage back on the host in the reference's shape."""
    kind = STAGE_KIND.get(key, "hidden")
    if kind == "streams":
        return streams_to_torch(plan, tensor, cfg.ihc.n)
    if kind == "post":
        return plan.to_torch_replicated(tensor).float()[0, 0]
    return hidden_to_torch(plan, tensor)


def _up_post(plan, post):
    """Reference post [T, n] -> device [1, 1, T, n], fp32, the same on every chip."""
    return ttnn.from_torch(
        post.float().reshape(1, 1, *post.shape).contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=ttnn.float32,
    )


def _replicas_agree(plan, tensor, label):
    """A tensor split over the hidden axis must be identical on every chip that holds the same slice."""
    if plan.hidden_axis is None:
        return
    shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(tensor)]
    cols = plan.mesh_shape[1]
    for index, shard in enumerate(shards):
        first = shards[index // cols * cols] if plan.hidden_axis == 0 else shards[index % cols]
        assert torch.equal(shard, first), f"{label}: chip {index} differs from the first replica of its hidden slice"


def _update_pcc(ref_out, ref_in, dev_out, dev_in):
    return pcc(ref_out.float() - ref_in.float(), dev_out.float() - dev_in.float())


def _gain(ref, dev):
    """The least-squares factor g in device = g * reference: the scale, which PCC cannot see."""
    r, d = ref.flatten().double(), dev.flatten().double()
    return (torch.dot(r, d) / torch.dot(r, r)).item()


def _update_gain(ref_out, ref_in, dev_out, dev_in):
    return _gain(ref_out.float() - ref_in.float(), dev_out.float() - dev_in.float())


def _check_stage(label, ref, dev, threshold, gain_tolerance=GAIN_TOLERANCE):
    """PCC and gain of one stage."""
    check_pcc(label, ref, dev, threshold)
    gain = _gain(ref.float(), dev.float())
    logger.info(f"{label}: gain={gain:.4f}")
    assert abs(gain - 1) <= gain_tolerance, f"{label}: gain={gain:.4f} (tolerance {gain_tolerance})"


def _faint(h, eps):
    """h scaled so that its rows' mean of squares is about eps / 10: the regime where eps sets the
    norm's scale."""
    return h * (0.1 * eps / h.pow(2).mean()).sqrt()


def _check_update(label, ref_out, ref_in, dev_out, dev_in, threshold, gain_tolerance=GAIN_TOLERANCE):
    """The layer's own work: PCC and gain of (streams out - streams in). Returns the PCC."""
    update = _update_pcc(ref_out, ref_in, dev_out, dev_in)
    gain = _update_gain(ref_out, ref_in, dev_out, dev_in)
    logger.info(f"{label}: pcc={update:.6f} | gain={gain:.4f}")
    assert update >= threshold, f"{label}: pcc={update:.6f} (threshold {threshold})"
    assert abs(gain - 1) <= gain_tolerance, f"{label}: gain={gain:.4f} (tolerance {gain_tolerance})"
    return update


def _device_routing(plan, tt, d_mlp_in):
    """The device MoE's routing of a BF16 [1, 1, T, hidden/tp] input, on the host: [T, E], a token's
    weight for every expert, zero where not picked."""
    return plan.to_torch_replicated(tt.mlp.route(tt.mlp.whole(d_mlp_in))).float()[0, 0]


def _pick_agreement(weights, other):
    """The fraction of tokens whose set of picked experts is the same in two [T, E] routings."""
    return ((weights != 0) == (other != 0)).all(dim=-1).float().mean().item()


def _reference_weights(ref, x, picked):
    """The reference router's weights on x [T, hidden] for a given choice of experts ``picked``
    [T, E] bool: the unbiased scores of the picks, normalised and scaled, zero elsewhere."""
    with torch.no_grad():
        chosen = ref.mlp.scores(x) * picked
    return chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20) * ref.cfg.moe.routed_scaling_factor


def _routing_check(ref, x_device, d_weights, label, x_reference=None):
    """The device router against the reference router. Returns the device's picks, [T, E] bool.

    ``x_device`` [T, hidden] is the host copy of exactly what the device router was given, so the
    two differ by the device's matmul error alone: every expert the device picked must score within
    PICK_TOLERANCE of the reference's cut (its top_k-th biased score) on that input, and carry the
    weight the reference gives that pick within WEIGHT_TOLERANCE. How many tokens end up with exactly
    the picks the reference makes on ``x_reference``, its own input, is logged.
    """
    k = ref.cfg.moe.top_k
    picked = d_weights != 0
    assert (picked.sum(dim=-1) == k).all(), f"{label}: picks per token {picked.sum(dim=-1).unique().tolist()}"
    with torch.no_grad():
        biased = ref.mlp.scores(x_device) + ref.mlp.router_bias
        own = ref.mlp.dense_weights(*ref.mlp.route(x_device if x_reference is None else x_reference))
    cut = biased.topk(k, dim=-1).values[:, -1]
    lowest = biased.masked_fill(~picked, float("inf")).min(dim=-1).values
    below = (cut - lowest).max().item()
    same = _pick_agreement(d_weights, own)
    off = (d_weights - _reference_weights(ref, x_device, picked)).abs().max().item()
    logger.info(
        f"{label} router picks: {same:.4f} of {picked.shape[0]} tokens match the reference exactly; "
        f"the worst device pick is {below:.2e} below the reference's cut; "
        f"the worst device weight is {off:.2e} from the reference's weight for that pick"
    )
    assert below <= PICK_TOLERANCE, f"{label}: a device pick scores {below:.2e} below the reference's cut"
    assert off <= WEIGHT_TOLERANCE, f"{label}: a device routing weight is {off:.2e} from the reference's"
    return picked


def _moe_reference(ref, x, picked):
    """The reference MoE on x with the device's picks and its own weights for them."""
    with torch.no_grad():
        return ref.mlp.routed(x, _reference_weights(ref, x, picked)) + ref.mlp.shared(x)


def _check_layer(plan, cfg, ref, dtype, seq_len, seed, label):
    """One layer, stage by stage on the reference's inputs, then end to end."""
    weight_dtype, pcc_stage, pcc_sublayer, pcc_update = DTYPES[dtype]
    streams = _streams(cfg, seq_len, seed)
    ref_inputs = reference_prefill_inputs(cfg, seq_len)
    rec = {}
    with torch.no_grad():
        ref(streams, *ref_inputs, rec)

    tt = TtHy4DecoderLayer(
        plan, cfg, ref.weights(), dense=ref.dense, weight_dtype=weight_dtype, expert_dtype=EXPERT_DTYPE[dtype]
    )
    pcc_attn = DTYPES["bfp8"][2] if dtype == "bfp4" else pcc_sublayer  # the attention weights are BFP8 either way
    inputs = dense_prefill_inputs(plan, cfg, seq_len)
    d_streams = streams_to_device(plan, streams)
    check_pcc(f"{label} streams round trip", streams, streams_to_torch(plan, d_streams, cfg.ihc.n), 0.99999)

    # --- the attention site, each stage on the reference's input for it
    h, post = tt.hc_attn.hc_pre(d_streams)
    check_pcc(f"{label} attn hc_pre", rec["attn_hc"], hidden_to_torch(plan, h), PCC_IHC)
    check_pcc(f"{label} attn post", rec["attn_post"], _down(plan, cfg, "attn_post", post), PCC_IHC)
    x = tt.input_layernorm(hidden_to_device(plan, rec["attn_hc"]))
    assert x.dtype == ttnn.bfloat16
    _check_stage(f"{label} input_layernorm", rec["attn_in"], hidden_to_torch(plan, x), PCC_NORM)
    faint = _faint(rec["attn_hc"], cfg.norm_eps)
    with torch.no_grad():
        r_faint = rms_norm(faint, ref.input_layernorm, cfg.norm_eps)
    x = tt.input_layernorm(hidden_to_device(plan, faint))
    _check_stage(f"{label} input_layernorm below eps", r_faint, hidden_to_torch(plan, x), PCC_NORM)
    out = tt.attention(hidden_to_device(plan, rec["attn_in"], ttnn.bfloat16), **inputs)
    _check_stage(f"{label} attention", rec["attn_out"], hidden_to_torch(plan, out), pcc_attn)
    out = tt.attention_sublayer(hidden_to_device(plan, rec["attn_hc"]), **inputs)
    assert out.dtype == ttnn.float32
    _check_stage(f"{label} norm + attention", rec["attn_out"], hidden_to_torch(plan, out), pcc_attn)
    mid = tt.hc_attn.hc_post(hidden_to_device(plan, rec["attn_out"]), d_streams, _up_post(plan, rec["attn_post"]))
    check_pcc(f"{label} attn hc_post", rec["streams_mid"], streams_to_torch(plan, mid, cfg.ihc.n), PCC_IHC)

    # --- the FFN site
    d_mid = streams_to_device(plan, rec["streams_mid"])
    h, post = tt.hc_mlp.hc_pre(d_mid)
    check_pcc(f"{label} mlp hc_pre", rec["mlp_hc"], hidden_to_torch(plan, h), PCC_IHC)
    check_pcc(f"{label} mlp post", rec["mlp_post"], _down(plan, cfg, "mlp_post", post), PCC_IHC)
    x = tt.post_attention_layernorm(hidden_to_device(plan, rec["mlp_hc"]))
    _check_stage(f"{label} post_attention_layernorm", rec["mlp_in"], hidden_to_torch(plan, x), PCC_NORM)
    faint = _faint(rec["mlp_hc"], cfg.norm_eps)
    with torch.no_grad():
        r_faint = rms_norm(faint, ref.post_attention_layernorm, cfg.norm_eps)
    x = tt.post_attention_layernorm(hidden_to_device(plan, faint))
    _check_stage(f"{label} post_attention_layernorm below eps", r_faint, hidden_to_torch(plan, x), PCC_NORM)

    x_mlp = rec["mlp_in"].bfloat16().float()  # the sublayer input is BF16 on the device
    d_x = hidden_to_device(plan, x_mlp, ttnn.bfloat16)
    out = hidden_to_torch(plan, tt.mlp(d_x))
    r_out = rec["mlp_out"]
    if not ref.dense:  # the arithmetic is judged on the device's own picks, the picks separately
        picked = _routing_check(ref, x_mlp, _device_routing(plan, tt, d_x), f"{label} mlp", rec["mlp_in"])
        logger.info(f"{label} mlp against the reference's own picks: pcc={pcc(rec['mlp_out'], out):.6f}")
        r_out = _moe_reference(ref, rec["mlp_in"], picked)
    _check_stage(f"{label} mlp", r_out, out, pcc_sublayer)

    d_sub = {}
    out = hidden_to_torch(plan, tt.mlp_sublayer(hidden_to_device(plan, rec["mlp_hc"]), d_sub))
    r_out = rec["mlp_out"]
    if not ref.dense:
        d_weights = _device_routing(plan, tt, d_sub["mlp_in"])
        x_device = hidden_to_torch(plan, d_sub["mlp_in"])
        picked = _routing_check(ref, x_device, d_weights, f"{label} norm + mlp", rec["mlp_in"])
        r_out = _moe_reference(ref, rec["mlp_in"], picked)
    _check_stage(f"{label} norm + mlp", r_out, out, pcc_sublayer)
    out = tt.hc_mlp.hc_post(hidden_to_device(plan, rec["mlp_out"]), d_mid, _up_post(plan, rec["mlp_post"]))
    check_pcc(f"{label} mlp hc_post", rec["streams_out"], streams_to_torch(plan, out, cfg.ihc.n), PCC_IHC)

    # --- end to end: every stage again, now on the device's own inputs
    d_rec = {}
    d_out = tt(d_streams, d_rec, **inputs)
    assert d_out.dtype == ttnn.float32
    assert tuple(d_out.shape) == (1, 1, seq_len, cfg.ihc.n * cfg.hidden_size // plan.hidden_tp)
    _replicas_agree(plan, d_out, label)
    down = {key: _down(plan, cfg, key, d_rec[key]) for key in STAGES}
    if not ref.dense:  # follow the device's picks, as above
        d_weights = _device_routing(plan, tt, d_rec["mlp_in"])
        picked = _routing_check(ref, down["mlp_in"], d_weights, f"{label} e2e", rec["mlp_in"])
        own = _update_pcc(rec["streams_out"], streams, down["streams_out"], streams)
        logger.info(f"{label} layer update against the reference's own picks: pcc={own:.6f}")
        # the reference's MoE input does not depend on the routing, so its weights for the device's
        # picks can be taken from the run above
        r_weights = _reference_weights(ref, rec["mlp_in"], picked)
        rec = {}
        with torch.no_grad():
            ref(streams, *ref_inputs, rec, expert_weights=r_weights)
    for key in STAGES[1:]:
        # up to the attention input nothing but iHC and the norm has run; from its output on, every
        # stage carries a sublayer's error
        threshold = {"attn_hc": PCC_IHC, "attn_post": PCC_IHC, "attn_in": PCC_NORM}.get(key, pcc_sublayer)
        _check_stage(f"{label} e2e {key}", rec[key], down[key], threshold)
    _check_update(f"{label} layer update", rec["streams_out"], streams, down["streams_out"], streams, pcc_update)


def _check_stack(plan, cfg, dtype, num_layers, seq_len, label):
    """Layers 0 .. num_layers-1 with random weights, free-running from the embedding, and each layer
    fed the reference's input.

    The reference runs in step with the device and takes each MoE layer's routing from it, so the
    two stay on the same experts all the way down; how often the reference would have picked
    differently is logged per layer.
    """
    weight_dtype, _, _, pcc_update = DTYPES[dtype]
    n = cfg.ihc.n
    stack = Hy4DecoderStack(cfg, num_layers)
    for index, layer in enumerate(stack.layers):
        _randomize(layer, seed=20 + index)
    embeds = torch.randn(seq_len, cfg.hidden_size, generator=torch.Generator().manual_seed(7))
    ref_inputs = reference_prefill_inputs(cfg, seq_len)
    first = ihc_expand(embeds[None], n)[0]

    inputs = dense_prefill_inputs(plan, cfg, seq_len)
    d_free = tt_ihc_expand(hidden_to_device(plan, embeds), n)
    check_pcc(f"{label} expand", first, streams_to_torch(plan, d_free, n), 0.99999)
    r_free = first
    for index, layer in enumerate(stack.layers):
        tt = TtHy4DecoderLayer(
            plan, cfg, layer.weights(), dense=layer.dense, weight_dtype=weight_dtype, expert_dtype=EXPERT_DTYPE[dtype]
        )

        def reference(streams, d_rec, what):
            """The reference layer on ``streams`` with the experts the device picked in ``d_rec``,
            weighted by the reference's own router."""
            rec = {}
            with torch.no_grad():
                layer(streams, *ref_inputs, rec)
            if layer.dense:
                return rec
            d_weights = _device_routing(plan, tt, d_rec["mlp_in"])
            x_device = hidden_to_torch(plan, d_rec["mlp_in"])
            picked = _routing_check(layer, x_device, d_weights, f"{label} layer {index} {what}", rec["mlp_in"])
            r_weights = _reference_weights(layer, rec["mlp_in"], picked)
            rec = {}
            with torch.no_grad():
                layer(streams, *ref_inputs, rec, expert_weights=r_weights)
            return rec

        # free-running
        d_rec = {}
        d_free = tt(d_free, d_rec, **inputs)
        rec = reference(r_free, d_rec, "free-running")
        r_free = rec["streams_out"]
        free = streams_to_torch(plan, d_free, n)
        check_pcc(f"{label} layer {index} free-running, streams", r_free, free, PCC_STREAMS[dtype])
        _check_update(f"{label} layer {index} free-running, accumulated update", r_free, first, free, first, pcc_update)

        # fed the reference's input
        streams_in = rec["streams_in"]
        d_rec = {}
        fed = streams_to_torch(plan, tt(streams_to_device(plan, streams_in), d_rec, **inputs), n)
        rec = reference(streams_in, d_rec, "fed the reference input")
        check_pcc(
            f"{label} layer {index} fed the reference input, streams", rec["streams_out"], fed, PCC_STREAMS[dtype]
        )
        what = f"{label} layer {index} fed the reference input, update"
        _check_update(what, rec["streams_out"], streams_in, fed, streams_in, pcc_update)


# ---------------------------------------------------------------------------------------------
# One chip: the composition, no collectives
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_layer_one_chip(device, kind, dtype, seq_len):
    cfg = ONE_CHIP
    plan = MeshPlan(device, hidden_axis=None)
    ref = _randomize(Hy4DecoderLayer(cfg, dense=kind == "dense"), seed=1)
    _check_layer(plan, cfg, ref, dtype, seq_len, 2, f"one chip {kind}")


@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_stack_one_chip(device, dtype):
    plan = MeshPlan(device, hidden_axis=None)
    _check_stack(plan, ONE_CHIP, dtype, num_layers=3, seq_len=32, label="one chip stack")


def test_attention_seam(device):
    """The layer hands the attention callable the normed BF16 hidden state and the caller's keyword
    arguments untouched, and uses what it returns: swapping the callable swaps the sublayer."""
    cfg = ONE_CHIP
    plan = MeshPlan(device, hidden_axis=None)
    ref = _randomize(Hy4DecoderLayer(cfg, dense=True), seed=3)
    seq_len = 32
    streams = _streams(cfg, seq_len, 4)
    seen = {}

    def halve(x, **inputs):
        seen.update(inputs, dtype=x.dtype, shape=tuple(x.shape))
        return ttnn.multiply(x, 0.5)

    weights = {name: value for name, value in ref.weights().items() if not name.startswith("self_attn.")}
    tt = TtHy4DecoderLayer(plan, cfg, weights, dense=True, weight_dtype=ttnn.bfloat16, attention=halve)
    d_rec = {}
    tt(streams_to_device(plan, streams), d_rec, positions="p", cache="c")
    assert seen == {"positions": "p", "cache": "c", "dtype": ttnn.bfloat16, "shape": (1, 1, seq_len, cfg.hidden_size)}

    with torch.no_grad():
        h, post = ref.hc_attn.hc_pre(streams[None])
        normed = rms_norm(h[0], ref.input_layernorm, cfg.norm_eps)
        mid = ref.hc_attn.hc_post(0.5 * normed[None], streams[None], post)[0]
    check_pcc("seam attn_out", 0.5 * normed, hidden_to_torch(plan, d_rec["attn_out"]), PCC_NORM)
    check_pcc("seam streams_mid", mid, streams_to_torch(plan, d_rec["streams_mid"], cfg.ihc.n), PCC_IHC)


# ---------------------------------------------------------------------------------------------
# 8x4 Galaxy: the SC1 layout
# ---------------------------------------------------------------------------------------------
def _galaxy(fn):
    fn = pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)(fn)
    return pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)(fn)


def _sc1_plan(mesh_device):
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8)
    return plan


@_galaxy
@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_layer_sc1(mesh_device, kind, dtype, seq_len):
    cfg = SMALL
    ref = _randomize(Hy4DecoderLayer(cfg, dense=kind == "dense"), seed=1)
    _check_layer(_sc1_plan(mesh_device), cfg, ref, dtype, seq_len, 2, f"sc1 {kind}")


@_galaxy
@pytest.mark.parametrize("dtype", list(DTYPES), ids=list(DTYPES))
def test_stack_sc1(mesh_device, dtype):
    _check_stack(_sc1_plan(mesh_device), SMALL, dtype, num_layers=3, seq_len=32, label="sc1 stack")


# ---------------------------------------------------------------------------------------------
# Checkpoint weights, layers 0-5, against the HF goldens
# ---------------------------------------------------------------------------------------------
# The free-running streams, on the host, keyed by (weight dtype, layer): layer L continues from what
# layer L-1 left here, so the free-running half needs the layers run in order in one session.
_FREE_RUNNING = {}


@_galaxy
@pytest.mark.parametrize("dtype", ["bfp8", "bf16", "bfp4"])
@pytest.mark.parametrize("layer", range(hf_harness.DEFAULT_NUM_LAYERS), ids=lambda layer: f"layer{layer}")
def test_layer_sc1_checkpoint(mesh_device, layer, dtype):
    """One real layer on the 8x4 mesh against the BF16 HF golden of a 128-token prompt.

    Fed the golden input: every stage and the layer's update, at the full length and on the first
    REAL_PREFIXES tokens alone. Free-running from the golden embedding: the streams and the
    accumulated update after this layer; that half is skipped, and says so, when the layer below
    has not run in this session.
    """
    golden = hf_harness.find_golden()
    if golden is None:
        pytest.skip(f"no golden under {hf_harness.GOLDEN_ROOT}: record one with reference/hf_harness.py")
    cfg = HY4
    weight_dtype = DTYPES[dtype][0]
    dense = cfg.is_dense(layer)
    hf = golden["layers"][layer]
    seq_len, n = golden["meta"]["seq_len"], cfg.ihc.n
    assert hf["dsa_dropped"] == 0, "the golden layer's DSA picks dropped visible keys: dense attention is not HF here"
    plan = _sc1_plan(mesh_device)
    label = f"real {dtype} L{layer}"

    weights = dict(zip(LAYER_PARAMETERS[dense], hy4_tensors(checkpoint_names(layer, dense))))
    router = (
        None if dense else (weights["mlp.gate.weight"].clone(), weights["mlp.gate.e_score_correction_bias"].clone())
    )
    tt = TtHy4DecoderLayer(plan, cfg, weights, dense=dense, weight_dtype=weight_dtype, expert_dtype=EXPERT_DTYPE[dtype])
    del weights
    inputs = dense_prefill_inputs(plan, cfg, seq_len, golden["positions"])

    def report_picks(d_mlp_in, what):
        """The device's picks on a device MoE input against HF's picks for this layer, and against a
        host router on that same input (how far below the cut the worst device pick scores)."""
        d_weights = _device_routing(plan, tt, d_mlp_in)
        picked = d_weights != 0
        hf_picked = torch.zeros_like(picked).scatter(1, hf["router_indices"].long(), True)
        same = (picked == hf_picked).all(dim=-1).float().mean().item()
        shared = (picked & hf_picked).sum().item() / hf_picked.sum().item()
        biased = torch.sigmoid(torch.nn.functional.linear(hidden_to_torch(plan, d_mlp_in), router[0])) + router[1]
        cut = biased.topk(cfg.moe.top_k, dim=-1).values[:, -1]
        below = (cut - biased.masked_fill(~picked, float("inf")).min(dim=-1).values).max().item()
        logger.info(
            f"{label} {what} router picks: {same:.4f} of {seq_len} tokens match HF exactly, {shared:.4f} of all "
            f"picks are HF's; the worst device pick is {below:.2e} below a host router's cut on the same input"
        )

    # --- fed the golden input
    streams_in = hf["streams_in"].float()
    d_rec = {}
    tt(streams_to_device(plan, streams_in), d_rec, **inputs)
    down = {key: _down(plan, cfg, key, d_rec[key]) for key in STAGES}
    scores = {key: pcc(hf[key].float(), down[key]) for key in STAGES[1:]}
    gains = {key: _gain(hf[key].float(), down[key]) for key in ("attn_out", "mlp_out")}
    for key, score in scores.items():
        logger.info(
            f"{label} golden input, {key}: pcc={score:.6f}" + (f" | gain={gains[key]:.4f}" if key in gains else "")
        )
    update = _update_pcc(hf["streams_out"], streams_in, down["streams_out"], streams_in)
    gain = _update_gain(hf["streams_out"], streams_in, down["streams_out"], streams_in)
    logger.info(f"{label} golden input, layer update: pcc={update:.6f} | gain={gain:.4f}")
    if not dense:
        report_picks(hidden_to_device(plan, hf["mlp_in"].float(), ttnn.bfloat16), "on the golden mlp_in,")
        report_picks(d_rec["mlp_in"], "golden input, on the device's own mlp_in,")

    # --- the first tokens of the same prompt alone. Prefill is causal, so their golden is the first
    # rows of the full one (HF in BF16 run on the prefix reproduces those rows to a layer-update PCC
    # of 0.9997 or better, not bit for bit), and the device must also agree with its own rows of
    # the full run.
    prefix = {}
    for length in REAL_PREFIXES:
        short = dense_prefill_inputs(plan, cfg, length, golden["positions"][:length])
        out = streams_to_torch(plan, tt(streams_to_device(plan, streams_in[:length]), **short), n)
        hf_out, hf_in = hf["streams_out"][:length], streams_in[:length]
        prefix[length] = (
            _update_pcc(hf_out, hf_in, out, hf_in),
            _update_gain(hf_out, hf_in, out, hf_in),
            _update_pcc(down["streams_out"][:length], hf_in, out, hf_in),
        )
        logger.info(
            f"{label} golden input, first {length} tokens alone, layer update: pcc={prefix[length][0]:.6f} | "
            f"gain={prefix[length][1]:.4f}"
        )
        logger.info(
            f"{label} golden input, first {length} tokens alone vs the same rows of the full run, "
            f"layer update: pcc={prefix[length][2]:.6f}"
        )

    # --- free-running from the golden embedding
    free_update = None
    embedding = golden["embedding"].float()
    first = ihc_expand(embedding[None], n)[0]
    if layer == 0:
        d_free = tt_ihc_expand(hidden_to_device(plan, embedding), n)
    elif (dtype, layer - 1) in _FREE_RUNNING:
        d_free = streams_to_device(plan, _FREE_RUNNING[(dtype, layer - 1)])
    else:
        d_free = None
        logger.info(f"{label} free-running: not run, layer {layer - 1} has not run in this session")
    if d_free is not None:
        free = streams_to_torch(plan, tt(d_free, **inputs), n)
        _FREE_RUNNING[(dtype, layer)] = free
        free_streams = pcc(hf["streams_out"].float(), free)
        free_update = _update_pcc(hf["streams_out"], first, free, first)
        free_gain = _update_gain(hf["streams_out"], first, free, first)
        logger.info(f"{label} free-running, streams: pcc={free_streams:.6f}")
        logger.info(f"{label} free-running, accumulated update: pcc={free_update:.6f} | gain={free_gain:.4f}")

    real, gain_tolerance = REAL[dtype], REAL_GAIN_TOLERANCE[dtype]
    floor = {key: PCC_REAL_STAGE for key in scores}
    floor.update(attn_out=PCC_REAL_ATTN, mlp_out=real["mlp"], streams_out=real["streams"])
    low = {key: (round(score, 6), floor[key]) for key, score in scores.items() if score < floor[key]}
    assert not low, f"{label} golden input, stages under their threshold (pcc, threshold): {low}"
    assert update >= real["update"], f"{label} layer update pcc={update:.6f} (threshold {real['update']})"
    assert abs(gain - 1) <= gain_tolerance, f"{label} layer update gain={gain:.4f}"
    off = {key: round(g, 4) for key, g in gains.items() if abs(g - 1) > gain_tolerance}
    assert not off, f"{label} golden input, sublayer output gains off by more than {gain_tolerance}: {off}"
    for length, (short_update, short_gain, against_full) in prefix.items():
        what = f"{label} first {length} tokens alone"
        assert short_update >= real["prefix"], f"{what}: layer update pcc={short_update:.6f}"
        assert abs(short_gain - 1) <= gain_tolerance, f"{what}: layer update gain={short_gain:.4f}"
        assert against_full >= PCC_REAL_PREFIX_VS_FULL, f"{what} vs the full run: layer update pcc={against_full:.6f}"
    if free_update is not None:
        assert free_streams >= real["free_streams"], f"{label} free-running streams pcc={free_streams:.6f}"
        assert free_update >= real["free_update"], f"{label} free-running update pcc={free_update:.6f}"
        assert abs(free_gain - 1) <= gain_tolerance, f"{label} free-running update gain={free_gain:.4f}"
