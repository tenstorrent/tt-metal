# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's first N layers against the model's own per-layer outputs.

The depth ladder. `test_block_layer0*.py` gate one layer in isolation; this runs the stack the way
prefill does — embedding, then N layers threading one AttnRes walk — and scores **every** layer
against `decoder_output_layer_i` from the vLLM trace. The per-layer curve is what says whether error
accumulates or stays flat, which a single end-of-stack number cannot.

The curve comes from a tap, not a second forward: `TtKimiK3Transformer.forward(layer_tap=...)` fires
after each layer with the live residual, and under AttnRes the live residual **is** what the trace
records (pinned host-side in `test_golden_contract.py`).

**Cost is why the ladder starts at 1 and 2.** Layer 0 is dense; every layer after it is LatentMoE
with 896 routed experts at 33 M parameters each — 59 GB of bf16 per layer, read from the 5.5 TB
dequantized store. One MoE layer is affordable; the 5/12/24-layer rungs want a built TTNN cache
rather than a fresh conversion each run.

**Fabric2D**, which is the fabric Kimi-K3 runs on. This used to be forced:
`attn_res_gather_softmax` hung on the wrapped fabrics. #53318 fixed that at source (the routing
plane now picks its direction from the forwarding table rather than from rank order), so the torus
arms are green upstream and this is a choice rather than a constraint.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# Aliased: the local `attn_res` in this module is the TtAttnRes instance, and the collision is
# silent until the reference is called and LightweightModule.__call__ looks for a `forward`.
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import attn_res as attn_res_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import fold_query
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import (
    TRACE_1M,
    TRACE_100K,
    load_checkpoint_tensors,
    resolve_checkpoint,
    resolve_trace,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, gather_cache_tp0, unrotate_cache_layer

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120

# Per-layer, against the model itself. The package's chunked per-layer bar is 0.88 at depth 61-78;
# at depths 1-2 the accumulated error should be far smaller, so this starts strict and the ladder's
# deeper rungs will say where it has to relax.
# 0.99 holds with margin while the only sealed candidate is the embedding, which is exact. Past the
# second seal it does not, and the reason is structural rather than a defect: immediately after a
# seal the live stream carries one layer of signal while the sealed candidate dominates the softmax
# mixture and contributes the error accumulated across the block it summarises. As the live sum
# grows it dominates again and the error washes out. Measured at depth 24, that is a step at layer
# 12, a minimum of 0.9864 at layer 19, and recovery to 0.9980 by layer 23 — a defect in the second
# sealed block would keep falling instead of climbing back.
#
# So the bar follows the regime. Depths through 12 hold the shallow 0.99; depth 24 uses the
# package's own per-layer depth threshold, which exists for exactly this accumulation
# (`LAYER_PCC_THRESHOLD = 0.88` in test_prefill_transformer_chunked.py). 0.98 is set here instead:
# still well inside what was measured, but tight enough that a real regression past the second seal
# shows up rather than hiding under a floor with 10 points of slack.
SHALLOW_LAYER_PCC = 0.99
DEEP_LAYER_PCC = 0.98

# 1 and 2 are cheap-ish; 5 is the first rung with a full-attention layer (layer 3) and so the first
# that needs a KV cache at all. 12 and 24 follow the same shape and are gated on a built TTNN weight
# cache rather than a per-run conversion.
DEPTHS = [1, 2, 5, 12, 24]

# The 100k trace instruments the inside of a layer — kda_*, moe_io, mla_io — but only records
# decoder_output for layers 0..4, so it can only score depths up to 5. The 1M trace records
# decoder_output for layers 0..24 and the 24 MLA layers' KV, and nothing else; it is the only oracle
# for the deeper rungs, and the inner taps below fall silent there because `trace.has` says so.
DEEP_TRACE_FROM = 12

# The package's shallow-layer KV bar. Depth 24 is still shallow by its standards (the 0.85 floor
# exists for full-depth bf8_b drift), so hold the tighter one and let a real regression show.
KV_CACHE_PCC = 0.96

PLACEMENTS = [
    pytest.param(
        (8, 4),
        # 4096, between the AttnRes suite's 1152 and the 24576 the rest of the package uses. Both ends
        # are wrong here. At depth 24 the sealed set has two blocks for the first time and
        # `inter_block`'s statistics collective needs 16 B per bank more than 1152 provides, so 1152
        # fails outright. But L1_SMALL comes out of the same L1 the circular buffers use, and at
        # 24576 MLA's chunked attention then fails to place its CBs ("statically allocated circular
        # buffers ... clash with L1 buffers") once there is a second chunk to attend over. 4096
        # clears AttnRes with margin while leaving MLA its working space.
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 4096},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


def _model_state_dict(checkpoint: Path, num_layers: int, root: str, cache: Path | None = None) -> dict:
    """The transformer's state dict: embedding, final norm, and one entry per layer.

    Routed experts are fetched per layer rather than up front — 59 GB each — so a caller that only
    wants layer 0 never pays for them, and a layer whose TTNN cache is already complete is not read
    from the checkpoint at all.
    """
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict_cached, load_tensors

    model = load_tensors(
        checkpoint, {"embed_weight": f"{root}embed_tokens.weight", "norm_weight": f"{root}norm.weight"}
    )
    layers = [load_layer_state_dict_cached(checkpoint, idx, cache) for idx in range(num_layers)]
    return {"embed_weight": model["embed_weight"].float(), "norm_weight": model["norm_weight"], "layers": layers}


def _compose(mesh_device, tensor):
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    ).reshape(-1, KimiK3Config.EMB_SIZE)[:SEQ_LEN]


# Almost all of this is reading routed experts out of the TTNN cache, which is a fixed cost per
# depth and already puts L1 at 254s against pytest.ini's global 300s cap. The rungs that do real
# work have no headroom left, so give the ladder its own budget rather than let L12 and L24 die
# on a timeout that has nothing to do with what they measure.
@pytest.mark.timeout(4800)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", DEPTHS, ids=[f"L{n}" for n in DEPTHS])
def test_depth_ladder_matches_golden(mesh_device, device_params, num_layers):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_1M if num_layers >= DEEP_TRACE_FROM else TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    # `$TT_KIMI_K3_PREFILL_TTNN_CACHE` turns the checkpoint read into a one-time cost per depth.
    # Every component already writes its tensorbins when handed a `weight_cache_path`; the markers
    # below record which layers finished, so a later run skips the 59 GB per MoE layer entirely.
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)
    state_dict = _model_state_dict(checkpoint, num_layers, root, cache)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        # Same pools the KDA layers cycle. AttnRes's private fallback set is only safe while nothing
        # else has a collective in flight on this axis, which is not true inside a K3 layer stack.
        tt_ccl=get_tt_ccl(mesh_device),
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden, block_residual=None):
        # Single-rank tests: nothing is inherited, so the second argument is always None.
        walk = TtAttnResWalk(
            attn_res,
            hidden,
            list(attn_res.weights.pre),
            list(attn_res.weights.post),
            attn_res.weights.output,
            num_layers,
        )
        return TtAttnResResidual(walk)

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        state_dict,
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=cache,
    )

    # One KV slot per FULL-ATTENTION layer, not per layer. Depths 1 and 2 hold none — layers 0 and 1
    # are both KDA — so there is nothing to allocate and `kvpe_cache=None` is the honest argument.
    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=SEQ_LEN,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
        )

    # The repo's own placement, not a hand-rolled mapper: tokens shard on the SEQUENCE axis, and
    # `prepare_prefill_input_tensor` is what produces the [sp_factor, 1, isl_per_chip] uint32
    # ROW_MAJOR layout the embedding reads. `is_balanced=False` is chunked prefill's block-cyclic
    # order, which is the order the golden trace's tokens are in too.
    tokens_tt = prepare_prefill_input_tensor(
        trace.token_ids(SEQ_LEN)[0].tolist(),
        mesh_device,
        tuple(mesh_device.shape)[SP_AXIS],
        False,
        tuple(mesh_device.shape),
        SP_AXIS,
    )

    per_layer = {}
    inner = {}

    def tap(local_idx, hidden):
        per_layer[local_idx] = _compose(mesh_device, hidden)

    # Wrap each layer's two norms so their outputs are recorded without adding debug plumbing to the
    # block. The FFN norm's output is the model's own `moe_input_layer_i`, and the attention norm's
    # is `kda_input_layer_i` where the trace records it — so a divergence can be placed on one side
    # of the layer or the other instead of only at its output.
    def _record(layer, name, fn):
        def wrapped(x):
            out = fn(x)
            inner[(layer, name)] = _compose(mesh_device, out)
            return out

        return wrapped

    for local_idx, layer in enumerate(model.layers):
        layer.attn_norm = _record(local_idx, "attn_norm", layer.attn_norm)
        if not layer.kv_only:
            layer.ffn_norm = _record(local_idx, "ffn_norm", layer.ffn_norm)

        # And the attention output itself. The trace records it directly only for layer 0
        # (`kda_output_layer_0`), but for any later layer it is derivable from the schedule:
        # `out_i = out_{i-1} + attn_i + mlp_i` with no seal, so
        # `attn_i = decoder_output_i - decoder_output_{i-1} - moe_output_i`.
        attention = layer.attention
        inner_fn = attention.forward

        def _attn(normed, ctx, _idx=local_idx, _fn=inner_fn):
            out = _fn(normed, ctx)
            inner[(_idx, "attn_out")] = _compose(mesh_device, out)
            return out

        attention.forward = _attn

    try:
        # No rope tensors: K3 is NoPE, so `ttMLA` binds `_apply_rope_none` at construction and the
        # rope dict is only ever indexed inside the two rotating paths. Passing None is correct
        # rather than lazy.
        model.forward(tokens_tt, kvpe_cache=kvpe, layer_tap=tap)
    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    # Report the inner taps first: when a layer's output is wrong, these say which half.
    for local_idx in range(num_layers):
        got = inner.get((local_idx, "attn_norm"))
        if trace.has("kda", f"kda_input_layer_{local_idx}"):
            want = trace.rows("kda", f"kda_input_layer_{local_idx}", 0, SEQ_LEN)
            logger.info(f"  L{num_layers} layer {local_idx} attn_norm vs kda_input: {comp_pcc(want, got, 0.99)[1]}")
        elif got is not None and local_idx == 1:
            # The trace records kda_input only for layer 0, but layer 1's is derivable and is the
            # single most diagnostic number in this test: it is the first PRE-ATTENTION AttnRes read
            # the walk issues (layer 0's is skipped, nothing being sealed yet), and the first read
            # where the sealed candidate carries real weight — 27% of the softmax mass against the
            # 4% layer 0's post-read gave it. So it separates "the read is wrong" from "the
            # recurrence is wrong" in one comparison.
            #     read_1   = attn_res(running_sum=out_0, block_residual=[embed], q_pre[1])
            #     kda_in_1 = input_layernorm_1(read_1)
            names = [
                f"{root}layers.1.{k}"
                for k in ("self_attention_res_norm.weight", "self_attention_res_proj.weight", "input_layernorm.weight")
            ]
            w = {k: v.float() for k, v in load_checkpoint_tensors(checkpoint, names).items()}
            read1 = attn_res_reference(
                trace.decoder_output(0, 0, SEQ_LEN),
                trace.decoder_input(0, SEQ_LEN).unsqueeze(1),
                fold_query(w[names[0]], w[names[1]]),
                eps=KimiK3Config.RMS_NORM_EPS,
            )
            want = read1 * torch.rsqrt(read1.pow(2).mean(-1, keepdim=True) + KimiK3Config.RMS_NORM_EPS) * w[names[2]]
            logger.info(f"  L{num_layers} layer 1 attn_norm vs DERIVED kda_input: {comp_pcc(want, got, 0.99)[1]}")
        got = inner.get((local_idx, "attn_out"))
        if got is not None and local_idx == 0 and trace.has("kda", "kda_output_layer_0"):
            want = trace.rows("kda", "kda_output_layer_0", 0, SEQ_LEN)
            logger.info(f"  L{num_layers} layer 0 attn_out vs kda_output: {comp_pcc(want, got, 0.99)[1]}")
        elif (
            got is not None
            and local_idx > 0
            and local_idx % KimiK3Config.ATTN_RES_BLOCK_SIZE
            and trace.has("moe_io", f"moe_output_layer_{local_idx}")
        ):
            # `attn_i = out_i - out_{i-1} - moe_i` holds only while the running sum is continuous
            # across the boundary. At a seal layer (`i % 12 == 0`) the stream restarts, so out_i
            # carries none of out_{i-1} and the subtraction is meaningless. Skip those rather than
            # print a number that looks like a failure.
            want = (
                trace.decoder_output(local_idx, 0, SEQ_LEN)
                - trace.decoder_output(local_idx - 1, 0, SEQ_LEN)
                - trace.rows("moe_io", f"moe_output_layer_{local_idx}", 0, SEQ_LEN)
            )
            logger.info(f"  L{num_layers} layer {local_idx} attn_out vs derived attn: {comp_pcc(want, got, 0.99)[1]}")
        if trace.has("moe_io", f"moe_input_layer_{local_idx}"):
            got = inner.get((local_idx, "ffn_norm"))
            want = trace.rows("moe_io", f"moe_input_layer_{local_idx}", 0, SEQ_LEN)
            logger.info(f"  L{num_layers} layer {local_idx} ffn_norm vs moe_input: {comp_pcc(want, got, 0.99)[1]}")

    # The KV cache is scored separately because the residual stream does not imply it. MLA writes
    # its slab and reads it back for causal attention within the chunk, so a wrong write can be
    # partly self-consistent here and only surface at chunk 2, when attention reads the previous
    # chunk's cached KV. The golden ships one file per MLA layer, so there is no reason not to check.
    # Initialised outside the guard: depths 1 and 2 hold no MLA layer, so there is no cache to
    # score and the assert below still has to have something to assert on.
    kv_failures: list[str] = []
    if kvpe is not None and trace.has_kv_cache(model.schedule.mla_layer_ids[0]):
        # The device cache is indexed by rank-local MLA SLOT; the golden by MODEL layer. The
        # schedule is the only thing that knows the mapping, which is exactly why the block never
        # calls `KimiK3Config.mla_kv_slot`.
        cache = gather_cache_tp0(kvpe.storage, mesh_device)
        positions = blockcyclic_positions(tuple(mesh_device.shape)[SP_AXIS], SEQ_LEN, SEQ_LEN)
        for slot, model_layer in enumerate(model.schedule.mla_layer_ids[: model.schedule.num_mla_layers]):
            device_rows = unrotate_cache_layer(cache[slot], positions, SEQ_LEN)
            golden_rows = trace.kv_cache(model_layer, 0, SEQ_LEN)
            # Kimi-K3 is NoPE, so the second half carries no rotation to re-base — the rule at
            # test_mla.py:608 in reverse.
            pcc_nope, pcc_pe = cache_half_pccs(golden_rows, device_rows, KimiK3Config.KV_LORA_RANK, pe_interleave=False)
            logger.info(
                f"L{num_layers} KV slot {slot} (model layer {model_layer}): " f"lora={pcc_nope:.6f} rope={pcc_pe:.6f}"
            )
            # Score every slot before failing. Which slots diverge is the whole diagnosis here -- slot 0
            # alone says the write path works, slots 1..n-1 say the per-slot stride does not -- and
            # asserting inside the loop throws that away at the first bad one.
            if min(pcc_nope, pcc_pe) < KV_CACHE_PCC:
                kv_failures.append(f"slot {slot} (model layer {model_layer}): lora={pcc_nope:.6f} rope={pcc_pe:.6f}")

    # Score every layer before asserting on any of them. Asserting inside the loop stops at the
    # first shortfall and hides the shape of the curve after it — and the shape is the diagnosis:
    # a step that then holds flat is accumulation, a step that keeps falling is a real defect in
    # whatever changed at that layer. At depth 24 that distinction lands exactly on the second seal.
    scores = {}
    for local_idx in range(num_layers):
        want = trace.decoder_output(local_idx, 0, SEQ_LEN)
        _, message = comp_pcc(want, per_layer[local_idx], SHALLOW_LAYER_PCC)
        scores[local_idx] = float(str(message).split()[-1])
        seal = " <- seal" if local_idx % KimiK3Config.ATTN_RES_BLOCK_SIZE == 0 else ""
        logger.info(
            f"L{num_layers} layer {local_idx} vs decoder_output_layer_{local_idx}: {scores[local_idx]:.7f}{seal}"
        )

    layer_pcc = DEEP_LAYER_PCC if num_layers > KimiK3Config.ATTN_RES_BLOCK_SIZE else SHALLOW_LAYER_PCC
    worst_idx = min(scores, key=scores.get)
    logger.info(f"L{num_layers} worst layer {worst_idx}: {scores[worst_idx]:.7f} (bar {layer_pcc})")
    assert scores[worst_idx] >= layer_pcc, (
        f"worst layer {worst_idx} diverged from the model: {scores[worst_idx]:.7f} < {layer_pcc}; "
        f"full curve {[f'{i}:{v:.5f}' for i, v in scores.items()]}"
    )
    # Asserted last, after the residual curve, for the same reason the KV loop scores every slot: a
    # bad KV slot and a bad residual layer look identical from one assertion, and which of the two
    # moved is the whole diagnosis. The residual stream is the model; the cache is what it wrote.
    assert not kv_failures, "KV cache diverged: " + "; ".join(kv_failures)
