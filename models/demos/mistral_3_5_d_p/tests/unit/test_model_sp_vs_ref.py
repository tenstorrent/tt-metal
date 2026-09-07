# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The WHOLE model at target SP x TP vs a composed torch reference: sequence sharded across the SP
rows, residual stream carried through every layer.
Pattern: ``minimax_m3/tests/unit/test_model_sp_vs_ref.py``.

This is the test that catches what single-layer tests structurally cannot:

  * **per-layer weight slicing** — layer ``i`` must get layer ``i``'s weights. The single-layer test
    only ever builds one layer, so a ``model.layers.{i}`` substate that always returns layer 0 (or is
    off by ``first_layer_idx``) passes everything below this file and produces a model that is
    self-consistent and wrong.
  * **the KV cache sized for N layers** — every layer writes its own ``(user, layer)`` slot, so a
    slot formula that ignores ``layer_idx`` makes layer 1 onward read layer 0's keys.
  * **the tail** — embedding in, final norm and LM head out, with the vocab shard reassembled.

Run at REAL width (hidden 12288, intermediate 28672, 96/8 heads, head_dim 128) and reduced DEPTH.
The depth reduction is a host limit, not a device one: the full 88-layer model is ~121 B parameters,
which no host can materialise as random weights to feed both sides. Every layer is identical here
(dense GQA + dense SwiGLU, full-causal), so N layers exercise the same code path as 88; what depth
buys is accumulation, which the P1/P2 golden-trace runs measure at full depth instead.

**Two different bars, for two different quantities.**

  * **Per-layer K/V** is compared against the fp32 oracle at the spec's 0.99. This is the recipe's
    actual correctness gate (§8: "per-layer KV PCC recorded in README"), and it passes with room —
    measured 0.99994 (layer 0) and 0.99712 (layer 1) at 2 layers / 512 tokens.
  * **The residual stream** (hidden states, and the logits derived from them) is compared against the
    fp32 oracle AND calibrated against a bf16 CPU forward of the same weights, because it is the most
    fp32-sensitive quantity in the model: a sum of many contributions with cancellation, which the
    final norm then amplifies. Measured at 2 layers, per SP row (row 0 holds the earliest positions,
    row 3 the latest):

        row      0        1        2        3
        device   0.9936   0.9900   0.9880   0.9865
        bf16 ref 0.9988   0.9955   0.9897   0.9801     (chunk 0)
        device   0.9859   0.9851   0.9848   0.9834
        bf16 ref 0.9673   0.9545   0.9397   0.9202     (chunk 1)

    Two things fall out. Early positions favour the bf16 reference, which has no bfloat8_b weights.
    LATER positions favour the DEVICE, by a widening margin — the ring SDPA's online softmax
    accumulates more accurately than torch's bf16 eager attention does. So an absolute 0.99 bar on
    the residual stream would fail a pure bf16 torch implementation of this model long before it
    failed this one. See ``reference.model.bf16_reference_forward``.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import allocate_kv_cache
from models.demos.mistral_3_5_d_p.tt.model import Model
from models.demos.mistral_3_5_d_p.tt.rope import build_indexed_rope, hf_to_meta_head_permutation

from ..test_factory import build_mesh_and_ccl, meta_swizzle, parametrize_mesh

YARN = dict(
    rope_theta=C.ROPE_THETA,
    yarn_factor=C.YARN_FACTOR,
    yarn_orig_max_pos=C.YARN_ORIG_MAX_POS,
    yarn_beta_fast=C.YARN_BETA_FAST,
    yarn_beta_slow=C.YARN_BETA_SLOW,
    truncate=C.YARN_TRUNCATE,
)
# Vocab reduced so the host can hold the embedding table and the LM head at real hidden width
# (the real 131072 x 12288 table is 1.6 B parameters on each side of the comparison).
VOCAB = 2048


def build_reference(num_layers, seq_len, *, seed=17):
    """The torch reference: config, weights, tokens, the fp32 outputs and the bf16 calibration run."""
    hf_config = reduced_text_config(num_hidden_layers=num_layers, vocab_size=VOCAB)
    model = reference.build_reference_model(hf_config, seed=seed)
    state_dict = model.state_dict()
    torch.manual_seed(seed + 1)
    token_ids = torch.randint(0, VOCAB, (1, seq_len))
    fp32 = reference.model_reference_forward(model, token_ids)
    bf16 = reference.bf16_reference_forward(hf_config, state_dict, token_ids)
    return hf_config, state_dict, token_ids, fp32, bf16


# The residual stream's bar. NOT the spec's 0.99, and the reason is measured rather than assumed
# (see the table in the module docstring): a bf16 CPU forward of the same weights scores 0.920-0.999
# per SP row against fp32, and the device scores 0.983-0.994 — better than bf16 arithmetic at later
# positions, slightly worse at early ones where the spec's bfloat8_b weights cost more than bf16
# storage does. Holding the residual stream to 0.99 would hold it above a pure bf16 implementation.
#
# The recipe's graded quantity is per-layer K/V (§8: "per-layer KV PCC recorded in README"), and that
# IS held to the spec's 0.99 below — it passes at 0.997 and better.
RESIDUAL_PCC = 0.98
# How far below a bf16 CPU forward the device may fall before it counts as a regression rather than
# a dataformat cost.
BF16_TRACKING_TOLERANCE = 0.01


def assert_residual_tracks_bf16(want_fp32, got_device, want_bf16, *, label):
    """Two assertions on a residual-stream quantity, neither of which alone would be enough.

    1. an ABSOLUTE floor (:data:`RESIDUAL_PCC`) against the fp32 oracle, so a catastrophic error
       fails even if the bf16 reference happens to be bad too;
    2. the device must TRACK a bf16 CPU forward to within :data:`BF16_TRACKING_TOLERANCE`, so a
       regression that is small in absolute PCC but real still fails.

    Returns ``(device_pcc, bf16_pcc)``.
    """
    device_pcc = float(comp_pcc(want_fp32, got_device, 0.0)[1])
    bf16_pcc = float(comp_pcc(want_fp32, want_bf16, 0.0)[1])
    logger.info(f"{label}: device-vs-fp32={device_pcc:.6f}  bf16ref-vs-fp32={bf16_pcc:.6f}")
    assert device_pcc >= RESIDUAL_PCC, f"{label}: device-vs-fp32 {device_pcc:.6f} below the floor {RESIDUAL_PCC}"
    assert device_pcc >= bf16_pcc - BF16_TRACKING_TOLERANCE, (
        f"{label}: the device ({device_pcc:.6f}) trails a bf16 CPU forward ({bf16_pcc:.6f}) by more "
        f"than {BF16_TRACKING_TOLERANCE}; that is an accuracy regression, not a dataformat limit"
    )
    return device_pcc, bf16_pcc


def build_tt_model(mesh_device, hf_config, state_dict, *, capacity, ccl, mesh_config, num_layers):
    """The production Model, fed the reference's own weights with q/k Meta-swizzled."""
    return Model(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=meta_swizzle(state_dict, hf_config.head_dim),
        ccl_manager=ccl,
        mesh_config=mesh_config,
        tensor_cache_path=None,
        max_seq_len=capacity,
        sequence_parallel=True,
        num_layers=num_layers,
    )


@parametrize_mesh()
@pytest.mark.parametrize("num_layers", [2], ids=["l2"])
@pytest.mark.parametrize("chunk, n_chunks", [(512, 1), (256, 2)], ids=["one_shot", "chunked2"])
def test_model_sp_vs_ref(mesh_device, device_params, num_layers, chunk, n_chunks, reset_seeds):
    """The whole model, SP-sharded, against the composed torch reference — hidden states and KV.

    Both one-shot and chunked, because the two take different attention cores (the all-gather
    bootstrap vs the cache-backed ring) and the per-layer weight/slot wiring has to hold for both.
    """
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    assert (rows, cols) == SPEC.mesh_shape
    total = chunk * n_chunks
    chunk_local = chunk // sp

    hf_config, state_dict, token_ids, ref, ref_bf16 = build_reference(num_layers, total)

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    model = build_tt_model(
        mesh_device, hf_config, state_dict, capacity=total, ccl=ccl, mesh_config=mesh_config, num_layers=num_layers
    )
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=total,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=hf_config.head_dim,
    )
    rope_mats = build_indexed_rope(
        mesh_device, head_dim=hf_config.head_dim, max_seq_len=total, chunk_size=chunk, sp_axis=SPEC.sp_axis, **YARN
    )

    tdims = [None, None]
    tdims[SPEC.sp_axis] = 3  # tokens sharded on the seq dim across the SP rows

    hidden_worst = 1.0
    for c in range(n_chunks):
        lo = c * chunk
        tokens_chunk = token_ids[:, lo : lo + chunk].reshape(1, 1, 1, chunk)
        tt_tokens = ttnn.from_torch(
            tokens_chunk,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(tdims)),
        )
        x = model.embed(ttnn.reshape(tt_tokens, [1, 1, tt_tokens.shape[-1]]))
        tt_tokens.deallocate(True)
        out = model.prefill_forward(
            x,
            rot_mats_global=rope_mats,
            kv_cache=kv_cache,
            cached_len=lo,
            user_id=0,
            skip_lm_head=True,  # compare the residual stream; the head has its own test + the run below
            indexed_rope=True,
        )
        ttnn.synchronize_device(mesh_device)

        # The final norm is applied inside prefill_forward only when the head runs, so with
        # skip_lm_head the output is the PRE-final-norm residual. Norm it on the host to compare.
        shards = ttnn.get_device_tensors(out)
        for r in range(rows):
            got = ttnn.to_torch(shards[r * cols]).float().reshape(1, chunk_local, hf_config.hidden_size)
            got_normed = reference.golden_rms_norm(got, state_dict["model.norm.weight"], hf_config.rms_norm_eps)
            start = lo + r * chunk_local
            pcc, _ = assert_residual_tracks_bf16(
                ref.hidden_states[:, start : start + chunk_local, :],
                got_normed,
                ref_bf16.hidden_states[:, start : start + chunk_local, :],
                label=f"hidden chunk {c} SP row {r}",
            )
            hidden_worst = min(hidden_worst, pcc)
        out.deallocate(True)

    logger.info(f"whole model ({num_layers}L, {n_chunks} x {chunk}): worst hidden-state pcc={hidden_worst}")

    # --- per-layer KV: this is what catches per-layer weight slicing and slot indexing ---
    from .test_kv_cache_write_vs_ref import read_cache_natural

    perm = hf_to_meta_head_permutation(hf_config.head_dim)
    kv_worst = 1.0
    for layer_idx in range(num_layers):
        got_k = read_cache_natural(
            kv_cache.k,
            mesh_device,
            slot=layer_idx,
            chunk_size=chunk,
            capacity=total,
            n_kv=hf_config.num_key_value_heads,
            n_tokens=total,
        )
        got_v = read_cache_natural(
            kv_cache.v,
            mesh_device,
            slot=layer_idx,
            chunk_size=chunk,
            capacity=total,
            n_kv=hf_config.num_key_value_heads,
            n_tokens=total,
        )
        want_k = ref.kv[layer_idx][0][0][..., perm]  # golden K, HF -> Meta
        want_v = ref.kv[layer_idx][1][0]
        ok_k, pcc_k = comp_pcc(want_k, got_k, SPEC.pcc)
        ok_v, pcc_v = comp_pcc(want_v, got_v, SPEC.pcc)
        kv_worst = min(kv_worst, float(pcc_k), float(pcc_v))
        logger.info(f"  layer {layer_idx}: K pcc={pcc_k} V pcc={pcc_v}")
        assert ok_k, f"layer {layer_idx} K disagrees with the golden: {pcc_k}"
        assert ok_v, f"layer {layer_idx} V disagrees with the golden: {pcc_v}"
    logger.info(f"whole model per-layer KV: worst pcc={kv_worst}")


@parametrize_mesh()
@pytest.mark.parametrize("num_layers", [2], ids=["l2"])
def test_model_logits_and_first_token(mesh_device, device_params, num_layers, reset_seeds):
    """The full tail: final norm + column-parallel LM head, reassembled, vs the reference logits.

    Also checks the argmax token per position, because a vocab shard reassembled in the wrong order
    permutes the logits in a way PCC over the flattened tensor tolerates.
    """
    chunk = 256
    rows, cols = tuple(mesh_device.shape)
    sp = rows
    hf_config, state_dict, token_ids, ref, ref_bf16 = build_reference(num_layers, chunk)

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    model = build_tt_model(
        mesh_device,
        hf_config,
        state_dict,
        capacity=chunk * 2,
        ccl=ccl,
        mesh_config=mesh_config,
        num_layers=num_layers,
    )
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=chunk * 2,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=hf_config.head_dim,
    )
    rope_mats = build_indexed_rope(
        mesh_device, head_dim=hf_config.head_dim, max_seq_len=chunk * 2, chunk_size=chunk, sp_axis=SPEC.sp_axis, **YARN
    )
    tdims = [None, None]
    tdims[SPEC.sp_axis] = 3
    tt_tokens = ttnn.from_torch(
        token_ids.reshape(1, 1, 1, chunk),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(tdims)),
    )
    x = model.embed(ttnn.reshape(tt_tokens, [1, 1, tt_tokens.shape[-1]]))
    logits = model.prefill_forward(
        x,
        rot_mats_global=rope_mats,
        kv_cache=kv_cache,
        cached_len=0,
        user_id=0,
        skip_lm_head=False,
        indexed_rope=True,
    )
    ttnn.synchronize_device(mesh_device)

    # logits per chip: [1, 1, chunk_local, padded_vocab/tp], SP-sharded on seq and TP-sharded on vocab.
    chunk_local = chunk // sp
    shards = ttnn.get_device_tensors(logits)
    per_row = [
        torch.cat(
            [ttnn.to_torch(shards[r * cols + c]).float().reshape(1, chunk_local, -1) for c in range(cols)], dim=-1
        )
        for r in range(rows)
    ]
    full = torch.cat(per_row, dim=1)[:, :, : hf_config.vocab_size]

    assert_residual_tracks_bf16(ref.logits, full, ref_bf16.logits, label=f"logits ({num_layers}L, {chunk} tokens)")

    # Top-1 against the fp32 oracle, and against the bf16 reference's own top-1 as the calibration:
    # the device must not disagree with truth more often than bf16 arithmetic does.
    device_agree = (ref.logits[0].argmax(-1) == full[0].argmax(-1)).float().mean().item()
    bf16_agree = (ref.logits[0].argmax(-1) == ref_bf16.logits[0].argmax(-1)).float().mean().item()
    logger.info(f"top-1 agreement with fp32: device={device_agree:.4f} bf16ref={bf16_agree:.4f}")
    assert device_agree >= bf16_agree - 0.05, (
        f"the device's top-1 agrees with fp32 on {device_agree:.3f} of positions against the bf16 "
        f"reference's {bf16_agree:.3f} — a real regression, not a dataformat limit"
    )
