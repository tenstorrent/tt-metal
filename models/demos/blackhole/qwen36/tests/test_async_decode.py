# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for Qwen3.6 DEVICE DECODE CONTINUITY (the prerequisite for async scheduling).

vLLM async scheduling submits decode step N+1 before step N's sampled token has been applied to
host state, so the host's token AND position are one step stale. A decode that re-stages its
inputs from host every step therefore cannot be overlapped. Continuity moves the three stale
things onto the device:

  * the on-device sampler writes the sampled id back into the decode trace's token buffer,
  * the traced graph advances ``current_pos`` itself (``ttnn.plus_one``),
  * RoPE is gathered on device from a position index the graph advances the same way.

A mistake in any of the three does NOT crash -- it silently shifts the rotation by a position or
replays a token, which shows up only as degraded output. So the gate here is exact token equality
against the known-good host-staged path, over several steps, not a single-step PCC.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

N_LAYERS = 8  # layer_types[:8] of the real checkpoint = 6 GDN + 2 full attention (both paths covered)
BLOCK = 64
CTX = 1024
BPU = CTX // BLOCK
START_POS = 16
# Enough steps that a drifting rope or position would separate the two paths. A one-position
# error is usually invisible on step 1 and only compounds, so a short run can pass by luck.
STEPS = 24


def _parametrize_traced(max_tp=4, trace_bytes=805306368):
    """parametrize_mesh_tp plus a trace region, so a test can capture decode traces."""
    shape = _resolve_mesh_shape(max_tp)

    def decorator(fn):
        fn = pytest.mark.parametrize(
            "device_params",
            [
                {
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                    "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
                    "trace_region_size": trace_bytes,
                }
            ],
            indirect=True,
        )(fn)
        fn = pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(
            fn
        )
        return fn

    return decorator


def _greedy_params(slots):
    return SamplingParams(temperature=[0.0] * slots, top_k=[1] * slots, top_p=[1.0] * slots)


def _build(mesh_device, bmax):
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=bmax, max_seq_len=CTX, n_layers=N_LAYERS, hf_model="Qwen/Qwen3.6-27B"
    )
    if model.sampling is None:
        pytest.skip("on-device sampling unsupported on this mesh; device continuity is a no-op")
    args = model.args
    model.allocate_kv_caches((bmax * BPU, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat16, batch_size=bmax)
    page_table = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(bmax)])
    return model, page_table


def _sampled_ids(tok_tensor, mesh_device, B):
    return ttnn.to_torch(ttnn.get_device_tensors(tok_tensor)[0]).reshape(-1)[:B].to(torch.int64).tolist()


def _mark_trace_buffers_corruptible(value):
    mark_corruptible = getattr(ttnn, "mark_corruptible", None)
    if mark_corruptible is None or value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _mark_trace_buffers_corruptible(item)
        return
    mark_corruptible(value)


def _gdn_snapshot(model):
    def first_device(tensor):
        tensors = ttnn.get_device_tensors(tensor)
        return ttnn.to_torch(tensors[0] if tensors else tensor)

    saved = []
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        recurrent = getattr(dn, "rec_state", None)
        if recurrent is None:
            recurrent = dn.recurrent_state
        conv_states = getattr(dn, "conv_states", None)
        if conv_states is None:
            fused = getattr(dn, "fused_conv_state", None)
            conv_states = [] if fused is None else [fused]
        saved.append(
            {
                "recurrent": first_device(recurrent),
                "conv": [first_device(conv) for conv in conv_states],
            }
        )
    return saved


@torch.no_grad()
@parametrize_mesh_tp()
def test_device_rope_gather_matches_host_rope(mesh_device, reset_seeds, ensure_gc):
    """The gathered cos/sin must be the same numbers the host path packs, at the same shape.

    Cheap and specific: if the gather layout is off by a transpose the decode still runs, and the
    only symptom is wrong attention. Checked over a per-slot position vector (not one shared
    position) because that is what batched decode feeds.
    """
    B = 8
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=B, max_seq_len=CTX, n_layers=1, hf_model="Qwen/Qwen3.6-27B"
    )
    rd = model.args.rope_head_dim
    # Stay a step below the last table row: the plus_one check below reads position+1, and the
    # gather table has exactly max_seq_len rows (same bound as the host cos_cpu it replaces).
    positions = torch.tensor([13, 400, 7, CTX - 2, 0, 99, 256, 512], dtype=torch.int32)

    # host path (the reference this replaces)
    inv_freq = 1.0 / (model.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    host_cos = emb.cos().reshape(1, B, 1, rd).to(torch.bfloat16).float()
    host_sin = emb.sin().reshape(1, B, 1, rd).to(torch.bfloat16).float()

    model.rope._ensure_decode_gather_tables()
    idx = model.rope.get_decode_rot_idxs(positions, on_host=False)
    cos, sin = model.rope.get_rot_mats_from_idxs(idx)
    assert list(cos.shape) == [1, B, 1, rd], f"cos shape {list(cos.shape)} != [1,{B},1,{rd}]"
    assert list(sin.shape) == [1, B, 1, rd], f"sin shape {list(sin.shape)} != [1,{B},1,{rd}]"
    dev_cos = ttnn.to_torch(ttnn.get_device_tensors(cos)[0]).float().reshape(B, rd)
    dev_sin = ttnn.to_torch(ttnn.get_device_tensors(sin)[0]).float().reshape(B, rd)
    cos_err = float((dev_cos - host_cos.reshape(B, rd)).abs().max())
    sin_err = float((dev_sin - host_sin.reshape(B, rd)).abs().max())
    logger.info(f"device-gathered rope vs host-packed rope: max abs err cos={cos_err:.3e} sin={sin_err:.3e}")
    assert cos_err == 0.0 and sin_err == 0.0

    # the graph advances this index itself; one plus_one must equal position+1
    ttnn.plus_one(idx)
    cos1, _ = model.rope.get_rot_mats_from_idxs(idx)
    got = ttnn.to_torch(ttnn.get_device_tensors(cos1)[0]).float().reshape(B, rd)
    want = model.rope.cos_cpu[(positions + 1).long()].to(torch.bfloat16).float()
    assert float((got - want).abs().max()) == 0.0, "plus_one on the rope index did not advance the rotation"

    # Position continuity is independent from direct sampler token feedback.
    model.device_token_feedback = False
    host_inputs = model.prepare_decode_inputs_host(
        torch.arange(B, dtype=torch.int32).reshape(B, 1),
        positions,
    )
    assert list(host_inputs[0].shape) == [B, 1]
    assert host_inputs[2].dtype == ttnn.uint32
    logger.info("PASSED: device rope gather is bit-identical to the host path, and advances correctly")


@torch.no_grad()
@_parametrize_traced()
@pytest.mark.parametrize("B", [1, 8], ids=["width1", "width8"])
def test_continuity_matches_host_staged_decode(mesh_device, B, reset_seeds, ensure_gc):
    """THE gate: N traced decode steps with NO host input staging must produce the exact same
    tokens as N host-staged steps.

    The reference runs first and eagerly, because it allocates device buffers every step and doing
    that while a trace is live corrupts the trace (tt_metal/impl/allocator.cpp:123). The continuity
    run then captures its traces and allocates nothing inside the loop.
    """
    BMAX = 8
    model, page_table = _build(mesh_device, BMAX)
    slots = model._decode_token_slots
    pt = page_table[:B]
    tok0 = torch.tensor([[100 + u] for u in range(B)], dtype=torch.int32)
    pos0 = torch.full((B,), START_POS, dtype=torch.int32)
    model.sampling.apply_decode_state([_greedy_params(slots)], reset_batch=True)
    assert model.sampling.tt_sampling.force_argmax_sampling, "expected the greedy (argmax) sampler path"

    # ---- reference: host-staged, eager, tokens fed back through the host every step -----------
    model.device_position_continuity = False
    model.device_token_feedback = False
    model.reset_tp()
    ref, tok, pos = [], tok0.clone(), pos0.clone()
    for _ in range(STEPS):
        dev = model.prepare_inputs_decode(tok, pos, pt)
        lg = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)
        out = model.sampling.sample(lg, enable_trace=False)
        ids = _sampled_ids(out[0] if isinstance(out, tuple) else out, mesh_device, B)
        ref.append(ids)
        tok = torch.tensor(ids, dtype=torch.int32).reshape(B, 1)
        pos = pos + 1
    logger.info(f"reference (host-staged, eager) tokens: {ref}")

    # ---- device continuity: capture, then replay with nothing coming from host ----------------
    model.device_position_continuity = True
    model.device_token_feedback = True
    model.reset_tp()
    dev = model.prepare_inputs_decode(tok0, pos0, pt)
    assert list(dev[0].shape) == [1, 1, 1, slots], f"token buffer {list(dev[0].shape)} is not sampler-shaped"
    assert dev[2].dtype == ttnn.uint32, "rope input should be a position index, not packed cos/sin"
    model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)
    ttnn.synchronize_device(mesh_device)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    logits = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    model.sampling.set_trace_bucket(B)
    # tt_out_tok=dev[0] is the whole point: the sampler writes into the trace's own token input.
    model.sampling.sample(logits, tt_out_tok=dev[0], enable_trace=True)
    ttnn.synchronize_device(mesh_device)

    # Compile + both captures each ran a decode step, so rewind: GDN in place (preserves the
    # trace-baked addresses) and the decode inputs back to step 0 via the same buffers.
    model._reset_gdn_state_for_new_sequence()
    host0 = model.prepare_decode_inputs_host(tok0, pos0, page_table=pt)
    for h, d in zip(host0, dev):
        if h is not None:
            ttnn.copy_host_to_device_tensor(h, d)
    ttnn.synchronize_device(mesh_device)

    got = []
    for step in range(STEPS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        model.sampling.sample(logits, tt_out_tok=dev[0], enable_trace=True)
        ttnn.synchronize_device(mesh_device)
        got.append(_sampled_ids(dev[0], mesh_device, B))
        dev_pos = ttnn.to_torch(ttnn.get_device_tensors(dev[1])[0]).reshape(-1)[:B].tolist()
        want_pos = [START_POS + step + 1] * B
        assert dev_pos == want_pos, f"step {step}: device positions {dev_pos} != {want_pos}"
    logger.info(f"device-continuity (traced, no host staging) tokens: {got}")

    ttnn.release_trace(mesh_device, tid)
    assert got == ref, (
        f"device continuity diverged from the host-staged reference at width {B}\n"
        f"  host-staged: {ref}\n  continuity : {got}"
    )
    logger.info(f"PASSED width={B}: {STEPS} traced steps with no host staging match the host-staged reference")


@torch.no_grad()
@_parametrize_traced()
def test_canonical_bucket_inputs_share_state(mesh_device, reset_seeds, ensure_gc):
    """All bucket widths must bind and update one canonical resident input set."""
    BMAX = 8
    model, page_table = _build(mesh_device, BMAX)
    model.initialize_decode_trace_inputs(BPU)

    host1 = model.prepare_decode_trace_inputs_host(
        torch.tensor([[100]], dtype=torch.int32),
        torch.tensor([START_POS], dtype=torch.int32),
        page_table[:1],
    )
    dev1 = model.prepare_decode_trace_inputs(host1)
    host8 = model.prepare_decode_trace_inputs_host(
        torch.tensor([[100 + u] for u in range(BMAX)], dtype=torch.int32),
        torch.full((BMAX,), START_POS, dtype=torch.int32),
        page_table,
    )
    dev8 = model.prepare_decode_trace_inputs(host8)

    assert dev1 is dev8
    assert list(dev8[1].shape) == [BMAX]
    assert list(dev8[2].shape) == [1, BMAX]
    assert list(dev8[3].shape) == [BMAX, BPU]

    for _ in range(5):
        model.ttnn_decode_forward(dev8[0], dev8[1], rot_mat_idxs=dev8[2], page_table=dev8[3])

    # Selecting width 1 changes only the captured prefix; no host/device copy occurs.
    model.prepare_decode_trace_inputs_host(
        torch.tensor([[999]], dtype=torch.int32),
        torch.tensor([999], dtype=torch.int32),
        page_table[:1],
    )
    model.ttnn_decode_forward(dev1[0], dev1[1], rot_mat_idxs=dev1[2], page_table=dev1[3])
    ttnn.synchronize_device(mesh_device)

    positions = ttnn.to_torch(ttnn.get_device_tensors(dev8[1])[0]).reshape(-1)[:BMAX].tolist()
    rope = ttnn.to_torch(ttnn.get_device_tensors(dev8[2])[0]).reshape(-1)[:BMAX].tolist()
    expected = [START_POS + 6] + [START_POS + 5] * (BMAX - 1)
    assert positions == expected
    assert rope == expected
    logger.info("PASSED: width 8 -> 1 preserved and advanced one canonical decode state")


@torch.no_grad()
@_parametrize_traced()
def test_bucket_trace_stores_share_canonical_inputs(mesh_device, reset_seeds, ensure_gc):
    """The served Generator path must bind every bucket trace to the same input objects."""
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    BMAX = 8
    model, _ = _build(mesh_device, BMAX)
    generator = Qwen36ForCausalLM([model], [model.args], mesh_device)
    warmup = dict(
        kv_cache=None,
        max_batch_size=BMAX,
        num_blocks=BPU,
        can_sample_on_device=True,
        greedy_only=True,
    )
    generator.warmup_model_decode(enable_trace=False, **warmup)
    generator.warmup_model_decode(enable_trace=True, **warmup)

    canonical = model._canonical_decode_inputs
    assert canonical is not None
    assert set(generator._bucket_trace_store) == {1, 2, 4, BMAX}
    for B, (_, trace_inputs, _) in generator._bucket_trace_store.items():
        assert trace_inputs[True][0] is canonical, f"width {B} on-device trace did not bind canonical inputs"
        assert trace_inputs[False][0] is canonical, f"width {B} host trace did not bind canonical inputs"

    generator.__del__()
    generator._bucket_trace_store = {}
    generator.trace_ids_decode = {}
    generator.model = []
    logger.info("PASSED: all served decode bucket traces share canonical input buffers")


@torch.no_grad()
@_parametrize_traced()
@pytest.mark.parametrize(
    "schedule",
    [
        pytest.param((1, 1, 1, 8, 8, 8, 1, 1, 1), id="1-8-1"),
        pytest.param((8, 8, 8, 2, 2, 2, 8, 8, 8), id="8-2-8"),
    ],
)
def test_bucket_switch_continuity_matches_host_reference(mesh_device, schedule, reset_seeds, ensure_gc):
    """Bucket switches preserve exact token, position, RoPE, and GDN state."""
    from models.tt_transformers.tt.common import copy_host_to_device

    BMAX = 8
    model, page_table = _build(mesh_device, BMAX)
    slots = model._decode_token_slots
    widths = tuple(sorted(set(schedule)))
    tok0 = torch.tensor([[100 + u] for u in range(BMAX)], dtype=torch.int32)
    pos0 = torch.full((BMAX,), START_POS, dtype=torch.int32)
    model.sampling.apply_decode_state([_greedy_params(slots)], reset_batch=True)

    # Reference: eager decode with host-authoritative inputs every step.
    model.device_position_continuity = False
    model.device_token_feedback = False
    model.reset_tp()
    ref_tokens, ref_positions = [], []
    tok, pos = tok0.clone(), pos0.clone()
    for B in schedule:
        dev = model.prepare_inputs_decode(tok[:B], pos[:B], page_table[:B])
        logits = model.ttnn_decode_forward(
            dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True
        )
        sampled = model.sampling.sample(logits, enable_trace=False)
        ids = _sampled_ids(sampled[0] if isinstance(sampled, tuple) else sampled, mesh_device, B)
        tok[:B] = torch.tensor(ids, dtype=torch.int32).reshape(B, 1)
        pos[:B] += 1
        ref_tokens.append(tok.reshape(-1).tolist())
        ref_positions.append(pos.tolist())
    ttnn.synchronize_device(mesh_device)
    ref_gdn = _gdn_snapshot(model)

    # Subject: all traces share one canonical input set. Compile every width first.
    model.device_position_continuity = True
    model.device_token_feedback = True
    model.initialize_decode_trace_inputs(BPU)
    canonical = model._canonical_decode_inputs
    for B in widths:
        model._reset_gdn_state_for_new_sequence()
        host = model.prepare_decode_trace_inputs_host(tok0[:B], pos0[:B], page_table[:B])
        dev = model.prepare_decode_trace_inputs(host)
        logits = model.ttnn_decode_forward(
            dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True
        )
        model.sampling.set_trace_bucket(B)
        model.sampling.sample(logits, tt_out_tok=dev[0], enable_trace=False)
    ttnn.synchronize_device(mesh_device)

    tids, logits_of = {}, {}
    for B in widths:
        model._reset_gdn_state_for_new_sequence()
        host = model.prepare_decode_trace_inputs_host(tok0[:B], pos0[:B], page_table[:B])
        dev = model.prepare_decode_trace_inputs(host)
        assert dev is canonical
        _mark_trace_buffers_corruptible(dev)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        logits = model.ttnn_decode_forward(
            dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True
        )
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        _mark_trace_buffers_corruptible(logits)
        model.sampling.set_trace_bucket(B)
        model.sampling.sample(logits, tt_out_tok=dev[0], enable_trace=True, skip_precompile=True)
        ttnn.synchronize_device(mesh_device)
        tids[B], logits_of[B] = tid, logits

    # Rewind once. No host input is staged during the switch sequence.
    model._reset_gdn_state_for_new_sequence()
    initial_host = model.prepare_decode_trace_inputs_host(tok0, pos0, page_table)
    copy_host_to_device(host_tensors=initial_host, device_tensors=canonical)
    ttnn.synchronize_device(mesh_device)

    for step, B in enumerate(schedule):
        model.sampling.set_trace_bucket(B)
        ttnn.execute_trace(mesh_device, tids[B], cq_id=0, blocking=False)
        model.sampling.sample(logits_of[B], tt_out_tok=canonical[0], enable_trace=True)
        ttnn.synchronize_device(mesh_device)

        got_tokens = _sampled_ids(canonical[0], mesh_device, BMAX)
        got_positions = (
            ttnn.to_torch(ttnn.get_device_tensors(canonical[1])[0]).reshape(-1)[:BMAX].to(torch.int64).tolist()
        )
        got_rope = ttnn.to_torch(ttnn.get_device_tensors(canonical[2])[0]).reshape(-1)[:BMAX].to(torch.int64).tolist()
        assert got_tokens == ref_tokens[step], f"step {step}, width {B}: token state diverged"
        assert got_positions == ref_positions[step], f"step {step}, width {B}: position state diverged"
        assert got_rope == ref_positions[step], f"step {step}, width {B}: RoPE state diverged"

    got_gdn = _gdn_snapshot(model)
    assert len(got_gdn) == len(ref_gdn)
    for layer, (got_state, ref_state) in enumerate(zip(got_gdn, ref_gdn)):
        assert torch.equal(got_state["recurrent"], ref_state["recurrent"]), f"GDN recurrent layer {layer} diverged"
        assert len(got_state["conv"]) == len(ref_state["conv"])
        for tap, (got_conv, ref_conv) in enumerate(zip(got_state["conv"], ref_state["conv"])):
            assert torch.equal(got_conv, ref_conv), f"GDN conv layer {layer}, tap {tap} diverged"

    model.sampling.reset_trace()
    for tid in tids.values():
        ttnn.release_trace(mesh_device, tid)
    logger.info(f"PASSED: schedule {schedule} matches the always-host-reloaded reference")


@pytest.mark.skip(
    reason="HANGS THE BOARD (needs tt-smi -r). Capturing/replaying a sampling trace from this bare "
    "harness deadlocks: the generator drives the sampler through sample_decode_on_device, which "
    "also calls seed_manager.get_new_values() and apply_decode_state() around every step, and this "
    "test does not. The server itself runs the same non-greedy sampling trace without hanging, so "
    "the fault is in the harness, not the model. Reproduce the sampler cost with a server A/B "
    "instead: rerun the benchmark with --extra-body '{\"temperature\": 0}' and compare TPOT. Only "
    "the decode-trace-alone part of this test (which does run) was used for a reported number."
)
@torch.no_grad()
@_parametrize_traced()
@pytest.mark.parametrize("mode", ["served", "greedy"])
@pytest.mark.parametrize("B", [1, 8], ids=["width1", "width8"])
def test_decode_plus_sampling_device_floor(mesh_device, B, mode, reset_seeds, ensure_gc):
    """Device floor INCLUDING the on-device sampler, at the served sampling params.

    The 42.4 / 49.1 ms floors quoted so far measured the decode trace ALONE. The server also runs
    the sampler on device every step, and Qwen3.6's generation_config.json makes every request
    non-greedy (temperature 1.0, top_k 20, top_p 0.95), which takes the topk + all-gather +
    ttnn.sampling path instead of the cheap argmax. That cost belongs in the floor.

    ONE sampling trace per test run, hence the ``mode`` parameter rather than a loop. Capturing a
    second sampling trace while the decode trace and the first sampling trace are already live
    allocates device buffers under a live trace ("Allocating device buffers is unsafe due to the
    existence of an active trace", tt_metal/impl/allocator.cpp:123) and hangs the board.
    """
    ITERS = 50
    ctx = int(os.environ.get("QWEN36_ASYNC_FLOOR_CTX", "4096"))
    bpu = ctx // BLOCK
    BMAX = 8
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=BMAX, max_seq_len=ctx, n_layers=None, hf_model="Qwen/Qwen3.6-27B"
    )
    if model.sampling is None:
        pytest.skip("no on-device sampling on this mesh")
    args = model.args
    slots = model._decode_token_slots
    model.allocate_kv_caches((BMAX * bpu, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat16, batch_size=BMAX)
    pt = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(BMAX)])[:B]
    model.reset_tp()

    dev = model.prepare_inputs_decode(
        torch.tensor([[100 + u] for u in range(B)], dtype=torch.int32),
        torch.full((B,), ctx - 64, dtype=torch.int32),
        pt,
    )
    model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    logits = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    model.sampling.set_trace_bucket(B)

    def _time(fn):
        fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) * 1000.0 / ITERS

    decode_ms = _time(lambda: ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False))
    logger.info(f"FLOOR width={B} ctx={ctx}: decode trace alone      = {decode_ms:.3f} ms/step")

    if mode == "greedy":
        params = SamplingParams(temperature=[0.0] * slots, top_k=[1] * slots, top_p=[1.0] * slots)
    else:  # the params generation_config.json applies to every served request
        params = SamplingParams(temperature=[1.0] * slots, top_k=[20] * slots, top_p=[0.95] * slots)
    model.sampling.apply_decode_state([params], reset_batch=True)
    model.sampling.sample(logits, tt_out_tok=dev[0], enable_trace=True)  # capture (the only one)
    ttnn.synchronize_device(mesh_device)

    def _step():
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        model.sampling.sample(logits, tt_out_tok=dev[0], enable_trace=True)

    total = _time(_step)
    logger.info(
        f"FLOOR width={B} ctx={ctx} {mode:6s} (argmax={model.sampling.tt_sampling.force_argmax_sampling}): "
        f"decode+sampling = {total:.3f} ms/step, sampler = {total - decode_ms:.3f} ms, "
        f"ceiling {1000.0 / total:.2f} tok/s"
    )
    ttnn.release_trace(mesh_device, tid)
    assert decode_ms > 0 and total >= decode_ms
