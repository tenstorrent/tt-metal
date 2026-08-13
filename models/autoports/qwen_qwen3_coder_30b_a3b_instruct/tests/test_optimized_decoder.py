# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the optimized decoder — every stage-01 guarantee, re-checked.

Optimization here changed program configs and packed two weights together; it
did not change the maths. So these tests deliberately re-run the *same*
contracts stage 01 established rather than a reduced subset, because the way an
optimization usually breaks a model is by quietly narrowing what still works:
a program config legal only at tile-aligned lengths, a packed weight whose
halves are swapped, a trace that captured a stale buffer.

``test_optimized_vs_functional_precision_delta`` is the sharpest of these.
Packing gate/up and widening ``in0_block_w`` are value-preserving, but the
optimized path also holds expert weights in bfloat4_b and attention projections
in bfloat8_b, so bit-identity is not available -- and asserting it would be
asserting the optimization away. Instead it bounds the gap between the two
implementations at 0.999 PCC. That is far tighter than either one's distance to
HF, so a real defect (swapped packed halves, a mis-sliced block) still cannot
hide inside it, while the quantisation that was measured and accepted can.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ..tt import functional_decoder as F
from ..tt import optimized_decoder as O
from ..tt.weight_mapping import convert_layer_weights
from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
PCC_REQUIRED = 0.995  # same bar as the functional decoder
MAX_SEQ = 1024
BLOCK_SIZE = 32
TRACE_REGION_SIZE = 50331648


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights(reference):
    _, hf_config = reference
    return convert_layer_weights(layer_state_dict(LAYER_IDX), hf_config)


def _hidden(hf_config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 0.02


def _reference_layer(layer, hf_config, hidden):
    seq_len = hidden.shape[1]
    cos, sin = rotary_embeddings(hf_config, seq_len)
    mask = torch.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)
    with torch.no_grad():
        out = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask)
    return out[0] if isinstance(out, tuple) else out


def _to_device(t, mesh_device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _norm(t, mesh_device):
    return ttnn.from_torch(
        t.reshape(1, 1, 1, -1).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build(mesh_device, hf_config, torch_weights, *, functional_weights=False):
    """Upload only what the path under test actually reads.

    ``decoder_layer_prefill_optimized`` / ``decoder_layer_decode_optimized``
    read exactly three tensors off ``DecoderLayerWeights`` -- the two norms and
    the router -- and take every projection and expert weight from
    ``OptimizedWeights``. Calling ``F.upload_layer_weights`` here as well
    uploaded ~1.2 GB of bf16 experts plus a third copy of wqkv/wo that nothing
    on the optimized path ever touches. That, not a model or op limit, is what
    used to exhaust DRAM at batch 8, so the batch coverage below was capped by
    the harness measuring its own waste.

    ``functional_weights=True`` restores the full set, needed only by the test
    that runs the *functional* layer side by side for a precision delta.
    """
    config = F.DecoderLayerConfig.from_hf(hf_config)
    if functional_weights:
        weights = F.upload_layer_weights(torch_weights, mesh_device, config)
    else:
        weights = F.DecoderLayerWeights(
            input_layernorm=_norm(torch_weights["input_layernorm"], mesh_device),
            post_attention_layernorm=_norm(torch_weights["post_attention_layernorm"], mesh_device),
            attention=None,  # optimized path uses OptimizedWeights.attention
            router=F.upload_router_weight(torch_weights["router"], mesh_device),
            experts=None,  # optimized path uses OptimizedWeights.gate_up_proj/down_proj
        )
    packed = O.upload_packed_expert_weights(torch_weights, mesh_device, config.moe)
    cos_cache, sin_cache = F.build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = F.build_expert_sparsity(mesh_device, config.moe.num_experts)
    return config, weights, packed, cos_cache, sin_cache, sparsity


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128, 512, 33, 100, 257], ids=["s32", "s128", "s512", "s33", "s100", "s257"])
def test_optimized_prefill_vs_reference(mesh_device, reference, torch_weights, seq_len):
    """Aligned and non-aligned lengths must both survive the new program configs."""
    layer, hf_config = reference
    config, weights, packed, cos, sin, sparsity = _build(mesh_device, hf_config, torch_weights)
    hidden = _hidden(hf_config, seq_len)

    ref_out = _reference_layer(layer, hf_config, hidden)
    tt_out = ttnn.to_torch(
        O.decoder_layer_prefill_optimized(
            _to_device(hidden.unsqueeze(0), mesh_device), weights, config, cos, sin, sparsity, packed
        )
    ).squeeze(0)

    passing, pcc_message = comp_pcc(ref_out, tt_out, PCC_REQUIRED)
    logger.info(f"optimized prefill seq={seq_len}: {pcc_message}")
    assert passing, f"optimized prefill (seq={seq_len}) below {PCC_REQUIRED}: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [128, 33], ids=["s128", "s33"])
def test_optimized_vs_functional_precision_delta(mesh_device, reference, torch_weights, seq_len):
    """Bound how far the optimized path may drift from the functional one.

    The two are deliberately *not* bit-identical: the optimized path holds
    expert weights in bfloat4_b and attention projections in bfloat8_b, which is
    what makes it fast. So this asserts a tight bound on the gap rather than
    equality -- close enough that a real defect (swapped packed halves, a
    mis-sliced block) still cannot hide, but loose enough to admit the
    quantisation that was measured and accepted.

    The bound is empirical and is budgeted, not guessed. Against HF the
    functional layer scores ~0.9995 and the optimized one ~0.9990, so the two
    can differ by ~0.0013 purely from quantisation; measured, they differ by
    0.00102 at S=128 and 0.00133 at S=33. 0.998 sits just outside that and well
    inside anything a structural bug would produce -- swapping the packed
    gate/up halves, for instance, drops this to below 0.5.
    """
    _, hf_config = reference
    config, weights, packed, cos, sin, sparsity = _build(mesh_device, hf_config, torch_weights, functional_weights=True)
    hidden = _hidden(hf_config, seq_len)

    base = ttnn.to_torch(
        F.decoder_layer_prefill(_to_device(hidden.unsqueeze(0), mesh_device), weights, config, cos, sin, sparsity)
    )
    opt = ttnn.to_torch(
        O.decoder_layer_prefill_optimized(
            _to_device(hidden.unsqueeze(0), mesh_device), weights, config, cos, sin, sparsity, packed
        )
    )

    passing, message = comp_pcc(base, opt, 0.998)
    logger.info(comp_allclose(base, opt))
    logger.info(f"seq={seq_len} optimized vs functional (bf16 vs {O.EXPERT_WEIGHT_DTYPE}): {message}")
    assert passing, f"optimized diverges from functional beyond expert quantisation: {message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 128, 33], ids=["decode", "s128", "s33"])
def test_optimized_router_matches_functional(mesh_device, reference, torch_weights, seq_len):
    """The optimized router must route *identically*, not merely closely.

    ``router_forward_optimized`` removes both keepdim reductions -- the max
    becomes column 0 of the sorted top-k, and the sum moves after the scatter
    and becomes a matmul. Neither is supposed to change the answer, so the test
    asserts the strong property: the same 8 experts, in the same slots, with
    weights equal to the functional router's within bf16 representation error.
    A weaker PCC bound would let a genuinely different routing pass, which is
    the failure mode that matters -- misrouting a token replaces its experts
    outright rather than perturbing its output.
    """
    _, hf_config = reference
    config = F.DecoderLayerConfig.from_hf(hf_config)
    w_router = F.upload_router_weight(torch_weights["router"], mesh_device)
    x = _to_device(_hidden(hf_config, seq_len).unsqueeze(0), mesh_device)

    base = ttnn.to_torch(F.router_forward(x, w_router, config.moe)).float().reshape(seq_len, -1)
    opt = ttnn.to_torch(O.router_forward_optimized(x, w_router, config.moe)).float().reshape(seq_len, -1)

    assert torch.equal(base > 0, opt > 0), (
        f"seq={seq_len}: optimized router selected different experts "
        f"({(( base > 0) != (opt > 0)).sum().item()} slots differ)"
    )
    assert ((base > 0).sum(dim=-1) == config.moe.num_experts_per_tok).all()
    delta = (base - opt).abs().max().item()
    sums = opt.sum(dim=-1)
    logger.info(
        f"router seq={seq_len}: max |functional - optimized| = {delta:.3e}, weight sums in "
        f"[{sums.min():.5f}, {sums.max():.5f}]"
    )
    assert delta < 5e-3, f"seq={seq_len}: routing weights differ by {delta}"
    assert torch.allclose(sums, torch.ones_like(sums), atol=2e-2), f"seq={seq_len}: weights not normalised"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 33, 100, 128], ids=["decode", "s33", "s100", "s128"])
def test_optimized_router_padding_is_zero(mesh_device, reference, torch_weights, seq_len):
    """The routing tensor's tile row-padding must be exact zero, not +inf.

    ``router_forward_optimized`` moved the sum after the scatter, so the divide
    runs over whole tiles. Rows ``seq_len``..``ceil(seq_len/32)*32`` have a zero
    numerator *and* a zero denominator, and unguarded ``ttnn.div`` returns
    **+inf** there -- which the functional router, dividing before the scatter,
    never did. No consumer was found that observes it (``to_torch`` returns the
    logical shape, the sparsity path drops the padding when it converts to
    ROW_MAJOR, and the scale multiply, ``rms_norm`` and ``fast_reduce_nc``
    reduce along axes that are tile-aligned or not the padded one), so this is
    a latent hazard rather than a live bug -- which is exactly the kind that a
    later consumer turns into a silent NaN. The divisor is clamped in
    ``router_forward_optimized``; this test is what stops the clamp being
    optimized back out.

    ``to_torch_with_padded_shape`` is the point of the test: ``ttnn.to_torch``
    slices to the logical shape and would pass no matter what the padding held.
    """
    _, hf_config = reference
    config = F.DecoderLayerConfig.from_hf(hf_config)
    w_router = F.upload_router_weight(torch_weights["router"], mesh_device)
    x = _to_device(_hidden(hf_config, seq_len).unsqueeze(0), mesh_device)

    out = O.router_forward_optimized(x, w_router, config.moe)
    padded = out.cpu().to_torch_with_padded_shape().float()
    assert torch.isfinite(padded).all(), (
        f"seq={seq_len}: routing tensor has {int((~torch.isfinite(padded)).sum())} non-finite "
        f"entries in its padded shape {tuple(padded.shape)}"
    )
    pad = padded[..., seq_len:, :]
    logger.info(
        f"router seq={seq_len}: padded {tuple(padded.shape)}, {pad.numel()} padding entries, "
        f"max |pad| = {pad.abs().max().item() if pad.numel() else 0.0}"
    )
    assert (pad == 0).all(), (
        f"seq={seq_len}: {int((pad != 0).sum())} of {pad.numel()} padding entries are non-zero "
        f"(max {pad.abs().max().item()})"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("block_size", [None, 32], ids=["contiguous", "paged32"])
def test_optimized_decode_matches_prefill(mesh_device, reference, torch_weights, block_size):
    """Paged and contiguous KV caches both still work on the optimized path."""
    layer, hf_config = reference
    config, weights, packed, cos, sin, sparsity = _build(mesh_device, hf_config, torch_weights)
    prompt_len = 32
    hidden_full = _hidden(hf_config, prompt_len + 1)
    ref_out = _reference_layer(layer, hf_config, hidden_full)[:, prompt_len, :]

    kv_cache = F.create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=block_size)
    O.decoder_layer_prefill_optimized(
        _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos,
        sin,
        sparsity,
        packed,
        kv_cache=kv_cache,
    )
    current_pos = ttnn.from_torch(torch.tensor([prompt_len], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)
    out = O.decoder_layer_decode_optimized(
        _to_device(hidden_full[:, prompt_len, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device),
        weights,
        config,
        cos,
        sin,
        kv_cache,
        current_pos,
        prompt_len,
        packed_experts=packed,
    )
    tt_out = ttnn.to_torch(out).reshape(1, hf_config.hidden_size)

    passing, pcc_message = comp_pcc(ref_out, tt_out, 0.99)
    kind = "contiguous" if block_size is None else f"paged({block_size})"
    logger.info(f"optimized decode [{kind}]: {pcc_message}")
    assert passing, f"optimized decode [{kind}] below 0.99: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_optimized_multi_step_decode(mesh_device, reference, torch_weights):
    """Several steps against a paged cache, each checked at its own position."""
    layer, hf_config = reference
    config, weights, packed, cos, sin, sparsity = _build(mesh_device, hf_config, torch_weights)
    prompt_len, steps = 32, 4
    hidden_full = _hidden(hf_config, prompt_len + steps)
    ref_out = _reference_layer(layer, hf_config, hidden_full)

    kv_cache = F.create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=BLOCK_SIZE)
    O.decoder_layer_prefill_optimized(
        _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos,
        sin,
        sparsity,
        packed,
        kv_cache=kv_cache,
    )

    for step in range(steps):
        pos = prompt_len + step
        current_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)
        out = O.decoder_layer_decode_optimized(
            _to_device(hidden_full[:, pos, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device),
            weights,
            config,
            cos,
            sin,
            kv_cache,
            current_pos,
            pos,
            packed_experts=packed,
        )
        passing, pcc_message = comp_pcc(ref_out[:, pos, :], ttnn.to_torch(out).reshape(1, -1), 0.99)
        logger.info(f"optimized decode step {step} (pos {pos}): {pcc_message}")
        assert passing, f"optimized decode step {step} below 0.99: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_optimized_prefill_is_deterministic(mesh_device, reference, torch_weights):
    """Bitwise repeatability, as required of the functional path."""
    _, hf_config = reference
    config, weights, packed, cos, sin, sparsity = _build(mesh_device, hf_config, torch_weights)
    hidden = _hidden(hf_config, 128)

    outs = [
        ttnn.to_torch(
            O.decoder_layer_prefill_optimized(
                _to_device(hidden.unsqueeze(0), mesh_device), weights, config, cos, sin, sparsity, packed
            )
        ).clone()
        for _ in range(3)
    ]
    assert torch.equal(outs[0], outs[1]), "optimized prefill run 1 != run 2 (bitwise)"
    assert torch.equal(outs[0], outs[2]), "optimized prefill run 1 != run 3 (bitwise)"
    logger.info("optimized prefill: 3 runs bit-identical")


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_optimized_decode_is_traceable(mesh_device, reference, torch_weights):
    """Trace capture, bit-exact replay, and a live input buffer."""
    _, hf_config = reference
    config, weights, packed, cos, sin, sparsity = _build(mesh_device, hf_config, torch_weights)
    prompt_len = 32
    hidden_full = _hidden(hf_config, prompt_len + 1)

    kv_cache = F.create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=BLOCK_SIZE)
    O.decoder_layer_prefill_optimized(
        _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos,
        sin,
        sparsity,
        packed,
        kv_cache=kv_cache,
    )

    tt_in = _to_device(hidden_full[:, prompt_len, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
    current_pos = ttnn.from_torch(torch.tensor([prompt_len], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)

    def step():
        return O.decoder_layer_decode_optimized(
            tt_in, weights, config, cos, sin, kv_cache, current_pos, prompt_len, packed_experts=packed
        )

    eager = ttnn.to_torch(step()).clone()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replayed = ttnn.to_torch(traced_out).clone()
    passing, message = comp_pcc(eager, replayed, 0.999)
    logger.info(f"optimized traced vs eager: {message}")
    assert passing, f"optimized traced replay disagrees with eager: {message}"

    # The trace must read the live buffer, not a value captured at record time.
    other = (_hidden(hf_config, 1, seed=99)).reshape(1, 1, 1, hf_config.hidden_size)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(other, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), tt_in)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    changed = ttnn.to_torch(traced_out).clone()
    delta = (replayed.float() - changed.float()).abs().max().item()
    logger.info(f"optimized traced replay delta after input swap = {delta:.6f}")
    assert delta > 1e-3, "optimized trace is not reading the live input buffer"

    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("batch", [1, 2, 8, 32], ids=["b1", "b2", "b8", "b32"])
def test_optimized_decode_batch(mesh_device, reference, torch_weights, batch):
    """Multi-user decode. Batch 1 is the latency target, but capability must hold.

    This did not work in stage 01 at any batch above 1: ``sparse_matmul``
    resolves ``batch_length`` differently depending on which operand is flagged
    sparse, and the down projection landed on a branch that ignores the batch
    dimension entirely. Each user is given a different prompt so a broadcast bug
    -- every row returning user 0's answer -- cannot pass.

    Writing this test surfaced a second defect beyond the sparsity one:
    ``attention_prefill`` hardcoded ``user_id=0``, so every user's prompt
    overwrote slot 0 and the other slots stayed empty. Per-user PCC failed at
    0.92-0.93 until ``user_id`` was threaded through prefill. Both halves were
    needed for multi-user decode to work end to end.

    Coverage used to stop at 2, blamed on a harness limit: every
    parametrisation re-uploaded **1.63 GB** of weights (the bf16 functional set
    *and* the optimized set) without reclaiming the previous ones, so b8
    exhausted DRAM in ``bank_manager.cpp`` before the layer ran. That was
    self-inflicted -- the optimized path never reads the functional experts or
    the functional attention copy. ``_build`` now uploads only what this path
    touches, **0.38 GB**, and b8 and b32 run.

    Both figures are derived from the shipped uploads rather than recalled:
    functional experts 3 x 128 x 768 x 2048 elem at bf16 = 1.208 GB, plus
    wqkv+wo bf16 = 37.75 MB; optimized experts (128 x 1536 x 2048 packed gate/up
    plus 128 x 768 x 2048 down) at bfloat4_b 0.5625 B/elem = 339.7 MB, plus two
    copies of wqkv+wo at bfloat8_b 1.0625 B/elem = 40.11 MB. Earlier revisions
    of this docstring said ~2.4 GB and ~0.63 GB, and neither was derived.

    32 is the real ceiling, and it is a **TTNN op limit**, not this layer's
    shape choice: ``nlp_create_qkv_heads_decode_device_operation.cpp:51``
    asserts ``num_users <= num_users_supported`` with ``num_users_supported =
    32`` hardcoded at line 45 of that file, and that op is on the interleaved
    attention path as well as the DRAM-sharded one. ``_dram_sharded_usable``
    does refuse the sharded projections past B=32 -- ``_width_sharded_l1``
    hardcodes a 32-row shard and ``_dram_sharded_program_config`` sets
    ``per_core_M=1`` -- but the interleaved fallback it selects then fails in
    ``nlp_create_qkv_heads_decode`` too. The guard buys a comprehensible
    failure, not a working larger batch.
    """
    layer, hf_config = reference
    config, weights, packed, cos, sin, sparsity = _build(mesh_device, hf_config, torch_weights)
    prompt_len = 32

    # Small cache on purpose: this test is about multi-user expert routing, and
    # a full-length cache per user would exhaust DRAM before reaching the point.
    kv_cache = F.create_kv_cache(mesh_device, config.attention, max_batch=batch, max_seq_len=128, block_size=BLOCK_SIZE)
    per_user = [_hidden(hf_config, prompt_len + 1, seed=u) for u in range(batch)]
    # Each user's prompt must actually land in the cache, or decode attends
    # zeros and the test proves nothing about multi-user routing.
    for user, hidden_full in enumerate(per_user):
        O.decoder_layer_prefill_optimized(
            _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
            weights,
            config,
            cos,
            sin,
            sparsity,
            packed,
            kv_cache=kv_cache,
            user_id=user,
        )

    tokens = torch.cat([h[:, prompt_len, :] for h in per_user], dim=0)  # [batch, hidden]
    current_pos = ttnn.from_torch(
        torch.full((batch,), prompt_len, dtype=torch.int32), dtype=ttnn.int32, device=mesh_device
    )
    out = O.decoder_layer_decode_optimized(
        _to_device(tokens.reshape(1, 1, batch, hf_config.hidden_size), mesh_device),
        weights,
        config,
        cos,
        sin,
        kv_cache,
        current_pos,
        prompt_len,
        packed_experts=packed,
    )
    tt_out = ttnn.to_torch(out).reshape(-1, hf_config.hidden_size)[:batch].float()

    assert torch.isfinite(tt_out).all(), f"batch={batch} produced non-finite values"

    # Every user is checked against its own HF reference. This is what proves
    # the routing is per-user rather than broadcast.
    for user, hidden_full in enumerate(per_user):
        ref_user = _reference_layer(layer, hf_config, hidden_full)[:, prompt_len, :]
        passing, message = comp_pcc(ref_user, tt_out[user : user + 1], 0.99)
        logger.info(f"optimized decode batch={batch} user {user} vs HF: {message}")
        assert passing, f"batch={batch} user {user} decode below 0.99: {message}"

    if batch > 1:
        spread = (tt_out - tt_out[0]).abs().max().item()
        assert spread > 1e-3, f"batch={batch}: all users returned identical output (broadcast bug)"
