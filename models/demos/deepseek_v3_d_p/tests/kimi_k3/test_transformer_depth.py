# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's first N layers against the model's own per-layer outputs.

The depth ladder. This runs the stack the way
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
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID

# Aliased: the local `attn_res` in this module is the TtAttnRes instance, and the collision is
# silent until the reference is called and LightweightModule.__call__ looks for a `forward`.
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import attn_res as attn_res_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import fold_query
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import (
    TRACE_1M,
    TRACE_100K,
    TRACE_K31,
    TRACE_K31_128K,
    load_checkpoint_tensors,
    resolve_checkpoint,
    resolve_trace,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_slabs import attach_kda_states
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import CACHE_LEN, cache_rows, compose
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
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
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": KimiK3Config.L1_SMALL_SIZE},
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

    slabs = attach_kda_states(mesh_device, model, sp_axis=SP_AXIS, tp_axis=TP_AXIS)
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
        if slabs is not None:
            slabs.deallocate()

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


# --- Kimi-K3.1 on real weights against its vLLM trace -------------------------------------------------------
#
# K3's ladder above, scored the same way (every layer's live residual against `decoder_output_layer_i`, the
# same depth-aware bars, every full-attention layer's kvpe against `kv_post_transform_layer_i` by halves at
# KV_CACHE_PCC) against `TRACE_K31`, which records all 93 layers. Two differences, both from what the K3.1
# trace and weights are:
#
# * The stack is the runner's PP4 rank 0 (`k31_stage.first_stage_params`: K3.1 TTNN weight cache, scaled-FP8
#   kvpe, bf8 indexer key cache, the K3 manifest's knobs), run eager so the tap can read every layer. The model
#   is too large for one galaxy, so the ladder covers PP4 stage 0, layers 0-23.
# * The prompt is the trace's 8192 prompt ids then its 2048 decoded ids, teacher-forced:
#   two chunks, so chunk 1's DSA layers read chunk 0's KV and its KDA layers continue chunk 0's carry. The bar
#   is held per chunk on prompt rows (chunk 0 [0, 5120), chunk 1 [5120, 8192)), as the chunked K3 test holds it
#   per chunk; rows past the prompt (vLLM's decode steps) feed the model but are not scored.
#
# The bar is K3's, per layer by depth: SHALLOW_LAYER_PCC through layer 11, DEEP_LAYER_PCC through layer 23 (K3's
# depth-24 bar, set for the second seal). KV: KV_CACHE_PCC at every DSA layer. Measured at 10K:
# worst 0.9966 (layers 0-11), 0.9869 (12-23).
#
# The DSA indexer, per DSA layer, as GLM-5.3's runner gate and the package's indexer tests measure it:
#
# * Key cache: the trace's `indexer/indexer_k_layer_i` is the post-LayerNorm key in
#   the natural basis (its own fp8 cache dequantizes to it at PCC 0.9996). K3.1's decode reads its cache in that
#   basis, so the device key is compared as stored (`index_hadamard=False`): a
#   Hadamard-rotated cache reads PCC -0.22..0.17 and fails. PCC on the prompt rows, as the producer's index check does.
#   Bar KV_CACHE_PCC, the KV cache's (the producer holds the index cache to the same threshold as the kvpe).
# * Top-k: each DSA layer's device top-2048 against `dsa/dsa_topk_indices_layer_i`,
#   scored as set recall |device ∩ trace| / |trace| per query row (test_ttnn_csa*.py's measure), averaged
#   over the prompt rows at positions >= INDEX_TOPK. Rows before that select every causal key on both sides
#   (recall 1 by construction, checked). Bar K31_TOPK_RECALL, per golden (below).
#
#     scripts/run_safe_pytest.sh \
#         "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_transformer_depth.py::test_k31_depth_ladder_matches_golden[blackhole-L24-torus-xy-8x4]"
def k31_layer_pcc(layer: int) -> float:
    """K3's per-layer bar for a layer of PP4 stage 0 (see above)."""
    assert layer < 2 * KimiK3Config.ATTN_RES_BLOCK_SIZE, f"layer {layer}: K3 sets no bar past depth 24"
    return SHALLOW_LAYER_PCC if layer < KimiK3Config.ATTN_RES_BLOCK_SIZE else DEEP_LAYER_PCC


# The package's indexer overlap floor (test_ttnn_csa_indexer._assert_topk_overlap). Measured at 10K:
# mean recall 0.913 (layer 51) .. 0.971. The trace's cut is crowded (~150 of 2048
# selected keys within 1% of the row's score spread above it), and through layer 47 >= 98% of the trace keys the
# device misses rank >= 1500 of 2048 by trace score: near-ties a bf8 key/query moves across the cut. From layer 55
# on, 6-11% of misses come from the trace's top half, in rows owned mostly by newline/','/'.' tokens: the
# frequent-token hidden-state error that sets in at layers 48-59, not the indexer
# (its key cache stays >= 0.992 there). 0.95, the CSA chunked floor, would fail layers 3, 7, 51 and 63-92 on that.
#
# Near-tie recall falls with the number of candidates, so one floor cannot serve both goldens: at 128K the cut holds
# ~300-380 keys within 1% of the spread (twice 10K's) and recall falls smoothly with position, at every layer. Layers
# 11-23 keep the 10K signature (>= 94% of misses rank >= 1500, median margin 0.3-0.6% of the spread): mean 0.928-0.941 at LoFi.
# Layers 3 and 7 carry ~3x the later layers' per-key score error at both lengths (miss margin p50 ~1.2%, p90 ~3.4% of
# the spread, unchanged from 10K), so the crowding costs them more. The first layers' larger score error and the
# trend toward 1M are not fully explained; the largest device step in that error, the score matmul's LoFi, is now
# HiFi2 (``index_score_math_fidelity``): layers 3 / 7 0.867 / 0.875 at 128K (0.834 / 0.837 at
# LoFi), 0.944 / 0.952 at 10K (0.929 / 0.936). The 128K floor is 0.85, ~1.7 points under them, as 0.9 sits ~1.3
# under the 10K minimum (layer 51, LoFi-measured; HiFi2 raised every measured layer); a fallback to LoFi fails it.
# Unlike 0.9, 0.85 comes from this measurement alone, not from an external reference (the LoFi-era floor was 0.8).
K31_TOPK_RECALL = {TRACE_K31: 0.9, TRACE_K31_128K: 0.85}


def k31_topk_rows(mesh_device, indices, rows: int) -> torch.Tensor:
    """A DSA layer's top-k `[1, 1, rows/sp(/tp), k]` (one chunk's query rows over SP, TP-split or TP-replicated)
    as host int64 `[rows, k]` in token order; the uint32 no-key sentinel reads 0xFFFFFFFF."""
    sp, tp = mesh_device.shape[SP_AXIS], mesh_device.shape[TP_AXIS]
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 1
    host = (
        ttnn.to_torch(
            indices,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
        ).to(torch.int64)
        & 0xFFFFFFFF
    )  # [1, tp, sp * local, k]
    local = host.shape[2] // sp
    if local * sp == rows:
        return host[0, 0]
    assert local * sp * tp == rows, f"top-k rows {host.shape} do not tile a {rows}-row chunk on {sp}x{tp}"
    return torch.cat([host[0, t, s * local : (s + 1) * local] for s in range(sp) for t in range(tp)])


def topk_recall(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per row the fraction of `want`'s valid (>= 0) keys that occur in `got`, i.e. |got ∩ want| / |want| as sets
    (a key `got` holds twice counts once). Both `[rows, k]` int64; `got`'s 0xFFFFFFFF sentinel matches nothing.
    Rows with no valid `want` entry read NaN."""
    rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
    valid = want >= 0
    return (torch.isin(rows + want, rows + got) & valid).sum(dim=1) / valid.sum(dim=1)


def test_topk_recall_counts_each_trace_key_once():
    """Hand-checked: a duplicated device index does not stand in for a missed key, keys never match across rows,
    and the no-key sentinel matches nothing."""
    none = 0xFFFFFFFF
    got = torch.tensor([[0, 0, 2], [0, 1, 2], [3, 4, 5], [0, none, none]])
    want = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, -1, -1]])
    assert topk_recall(got, want).tolist() == pytest.approx([2 / 3, 1.0, 0.0, 1.0])


# rung -> golden; both run all of PP4 stage 0. "L24-128k" is K3's long-context check (test_chunked_prefill.py's L24:
# layers 0-23 over every chunk) on the 128K trace.
K31_RUNGS = {"L24": TRACE_K31, "L24-128k": TRACE_K31_128K}


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", k31_stage.PLACEMENT, indirect=True)
@pytest.mark.parametrize("rung", list(K31_RUNGS))
def test_k31_depth_ladder_matches_golden(rung, mesh_device, device_params):
    golden_path = K31_RUNGS[rung]
    trace = resolve_trace(golden_path)
    assert trace is not None, f"no K3.1 golden at {golden_path}"
    chunk = k31_stage.CHUNK
    # The trace's whole teacher-forced sequence (prompt, then the ids vLLM decoded), padded to whole chunks.
    ids = trace.metadata["token_ids"] + trace.metadata["completion_token_ids"]
    prompt_len, real_len = len(trace.metadata["token_ids"]), len(ids)
    num_chunks = -(-real_len // chunk)
    prompt = ids + [MTP_PAD_TOKEN_ID] * (num_chunks * chunk - real_len)
    hf_config, params = k31_stage.first_stage_params(max_seq_len=-(-real_len // CACHE_LEN) * CACHE_LEN)
    first, num_layers = params.first_layer_idx, params.num_layers
    runtime = k31_stage.ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    kv_caches = k31_stage.ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(kv_caches)

    def pcc(want, got):
        return float(str(comp_pcc(want, got)[1]).split()[-1])

    # Scored chunk by chunk on the prompt rows, so host memory holds one chunk's taps.
    scores = {}  # (global layer, chunk) -> PCC

    def score(c, taps):
        lo = c * chunk
        hi = min(lo + chunk, prompt_len)
        if hi <= lo:
            return
        for layer, got in sorted(taps.items()):
            golden = trace.decoder_output(layer, lo, hi)
            scores[(layer, c)] = pcc(golden, got[: hi - lo].float())

    # Per DSA layer, the top-k recall of every prompt row, filled chunk by chunk (the indices are not kept).
    recall = {}  # global layer -> [prompt rows] float
    topk_miscounted = []

    def score_topk(c, topk):
        lo = c * chunk
        if lo >= prompt_len:
            return
        for layer, got in sorted(topk.items()):
            want = trace.rows("dsa", f"dsa_topk_indices_layer_{layer}", lo, min(lo + chunk, prompt_len)).long()
            n = want.shape[0]
            got = got[:n]
            # Every device row selects min(position + 1, top-k) keys, as every trace row does.
            want_count = torch.clamp(torch.arange(lo, lo + n) + 1, max=want.shape[1])
            miscounted = int(((got < 2**31).sum(dim=1) != want_count).sum())
            if miscounted:
                topk_miscounted.append(f"layer {layer} chunk {c}: {miscounted} rows")
            recall.setdefault(layer, torch.full((prompt_len,), float("nan")))[lo : lo + n] = topk_recall(got, want)

    topk = {}
    dsa_mlas = {
        first + local: layer.attention.mla
        for local, layer in enumerate(runtime.model.layers)
        if runtime.model.schedule.is_mla(first + local)
    }

    def with_topk(layer, mla_forward):
        def forward_returning_topk(*args, **kwargs):
            out, indices = mla_forward(*args, return_indexer_indices=True, **kwargs)
            topk[layer] = k31_topk_rows(mesh_device, indices, chunk)
            return out

        return forward_returning_topk

    for layer, mla in dsa_mlas.items():
        mla.forward = with_topk(layer, mla.forward)

    taps = {}
    forward = runtime.model.forward

    def tapped(*args, **kwargs):
        tap = lambda idx, h: taps.__setitem__(first + idx, compose(mesh_device, h, chunk))
        return forward(*args, layer_tap=tap, **kwargs)

    runtime.model.forward = tapped
    try:
        for c in range(num_chunks):
            chunk_input = runtime.make_chunk_input(prompt[c * chunk : (c + 1) * chunk])
            out = runtime.prefill_chunk(
                chunk_input, kv_caches, slot_id=0, actual_start=c * chunk, actual_end=min((c + 1) * chunk, real_len)
            )
            ttnn.synchronize_device(mesh_device)
            logger.info(f"K31 {rung} chunk {c}/{num_chunks} done: layers [{first}, {first + num_layers})")
            if out is not None:
                ttnn.deallocate(out)
            score(c, taps)
            taps.clear()
            score_topk(c, topk)
            topk.clear()
        slots = {
            slot: runtime.model.schedule.global_index(local)
            for local, slot in enumerate(runtime.model.schedule.kv_slot_of_local)
            if slot is not None
        }
        kv = {
            name: cache_rows(mesh_device, cache, chunk, prompt_len)
            for name, cache in (("kvpe", kv_caches.kvpe), ("index", kv_caches.index))
            if cache is not None
        }
    finally:
        runtime.model.forward = forward
        for mla in dsa_mlas.values():
            del mla.forward  # the instance attribute; the class's forward shows through again
        if kv_caches.kda_states is not None:
            kv_caches.kda_states.deallocate()
        runtime.release_trace()  # the sub-device managers, before the fixture closes the mesh

    kv_failures = []
    for slot, layer in sorted(slots.items()):
        lora, rope = cache_half_pccs(
            trace.kv_cache(layer, 0, prompt_len), kv["kvpe"][slot], KimiK3Config.KV_LORA_RANK, pe_interleave=False
        )
        index = pcc(trace.rows("indexer", f"indexer_k_layer_{layer}", 0, prompt_len), kv["index"][slot])
        if min(lora, rope) < KV_CACHE_PCC:
            kv_failures.append(f"slot {slot} (layer {layer}): lora={lora:.6f} rope={rope:.6f}")
        if index < KV_CACHE_PCC:
            kv_failures.append(f"slot {slot} (layer {layer}): index key={index:.6f}")

    # Top-k recall per DSA layer: gated on the prompt rows past the first top-k; the rows before it are checked
    # for completeness.
    topk_k = KimiK31Config.INDEX_TOPK
    topk_failures = []
    for layer, per_row in sorted(recall.items()):
        full, gated = per_row[:topk_k], per_row[topk_k:]
        if gated.mean() < K31_TOPK_RECALL[golden_path] or full.min() < 1:
            topk_failures.append(f"layer {layer}: {gated.mean():.6f} (rows < {topk_k}: min {full.min():.4f})")

    worst = min(scores, key=lambda k: scores[k] - k31_layer_pcc(k[0]))
    logger.info(
        f"K31 {rung} worst layer vs its bar: {worst[0]} chunk {worst[1]}: {scores[worst]:.7f} (bar {k31_layer_pcc(worst[0])})"
    )
    below = {k: v for k, v in scores.items() if v < k31_layer_pcc(k[0])}
    assert not below, "layers below their bar: " + ", ".join(
        f"{l}/chunk{c}:{v:.6f}<{k31_layer_pcc(l)}" for (l, c), v in sorted(below.items())
    )
    assert not kv_failures, "KV cache diverged: " + "; ".join(kv_failures)
    assert not topk_miscounted, "device top-k selects a different number of keys: " + "; ".join(topk_miscounted)
    assert not topk_failures, f"top-k recall below {K31_TOPK_RECALL[golden_path]}: " + "; ".join(topk_failures)
