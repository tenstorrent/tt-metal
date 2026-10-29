# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The MULTI-SLOT serving decoder (tt/dflash2_serving.py) must be lossless per request while requests
join and leave a live batch on ONE set of captures, the way vLLM drives it:

  slot 0 joins (prefill -> ingest -> begin), speculates alone for a few steps;
  slot 1 joins WHILE slot 0 is mid-generation (its seed is a hold-replay for slot 0);
  both speculate together; each leaves when it has MAX_NEW tokens;
  a THIRD request joins the freed slot 0 while slot 1 may still be live, and runs to MAX_NEW.

Every request's tokens are then checked against a teacher-forced single-user plain greedy reference
(test_spec_lossless._reference_greedy on a separate max_batch_size=1 model), with the near-tie gate
(test_spec_batched._assert_lossless). A hold that is not a bit-exact no-op, a per-slot seed that
touches another slot, or a stale page-table row shows up here as a confident mismatch.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_dflash2_serving.py -v -s
Needs the DFlash2 drafter matched to the served weights (DFLASH_WEIGHTS) and the full 64-layer model.
"""

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_spec_batched import (
    _assert_lossless,
    _batch_prompts,
    _blocks_per_user,
    _reference_model,
    _release,
)
from models.demos.blackhole.qwen36.tests.test_spec_lossless import MAX_NEW, NUM_BLOCKS, _reference_greedy
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# Slots allocated (the verify bucket is (B, K+1)); the scenario itself uses slots 0 and 1, so B > 2 runs the same
# joins / leaves with the extra slots HELD -- QWEN36_DFLASH_SERVING_TEST_B=4 exercises the served 4x8 bucket (TP=2's
# only bucket at Nv=24) with two held rows.
B = int(os.environ.get("QWEN36_DFLASH_SERVING_TEST_B", "2"))
K = 7


def _prefill(model, dec, u, prompt_ids, page_tables):
    """The serving prefill of one request into slot u: eager, taps armed, each chunk's taps ingested into
    the drafter's ring for that slot. Returns the greedy first token."""
    T = len(prompt_ids)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
    pt_u = page_tables[u : u + 1].contiguous()

    def on_chunk(hidden, chunk_start, valid_len):
        taps = model.take_dflash_eager_taps()
        assert taps is not None, "eager prefill captured no drafter taps"
        dec.ingest_prompt(u, taps, chunk_start + valid_len, chunk_start=chunk_start)

    model._dflash_tap = True
    try:
        logits = model.prefill_for_spec(prompt, pt_u, T, on_chunk, slot=u)
    finally:
        model._dflash_tap = False
    lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    ttnn.deallocate(logits)
    first = int(lt.reshape(-1)[: model.vocab_size].float().argmax())
    assert dec.ctx_len[u] == T, f"slot {u}: ingested {dec.ctx_len[u]} of {T} positions"
    return first


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash2_serving_slots_are_lossless(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_serving import DFlash2ServingDecoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=B, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    assert len(model.layers) >= 62, "the DFlash2 taps live at layers 5..61: run the full model"
    model.set_gdn_fused_decode(True)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(3, tokenizer)  # three distinct, ragged prompts (130 / 147 / 165 tokens)
    bpu = _blocks_per_user(max(len(p) for p in prompts), K, MAX_NEW + 8)
    page_tables = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)

    dec = DFlash2ServingDecoder(model, num_blocks=bpu)
    outs = {}
    try:
        # ---- server warm-up: allocate, compile every program eagerly, capture, dummy session -------- #
        dec.alloc()
        dec.warm()
        first_w = _prefill(model, dec, 0, prompts[0], page_tables)  # compiles the prefill bucket + ingest
        dec.capture(warm_position=len(prompts[0]) + 1)
        dec.begin(0, first_w, len(prompts[0]), page_tables[0])
        for _ in range(3):
            dec.step()  # captures the drafter's draft/extend traces
        dec.end(0)

        # ---- scenario --------------------------------------------------------------------------- #
        # request A -> slot 0, alone for 3 steps
        firstA = _prefill(model, dec, 0, prompts[0], page_tables)
        dec.begin(0, firstA, len(prompts[0]), page_tables[0])
        outA = [firstA]
        for _ in range(3):
            outA.extend(dec.step()[0])
        logger.info(f"[serving] A alone: {len(outA)} tokens after 3 steps")
        # request B joins slot 1 while A is mid-generation (its seed HOLDS A's rows). A hold must be a
        # bit-exact no-op for A's durable spec state: its GDN ring slot mi, its conv window row and its
        # attention KV blocks are read back before and after and compared exactly.
        gdn0 = dec._gdn[0]
        att0 = next(layer.attention for layer in model.layers if layer.is_full_attention)

        def _snap():
            Nv, Dk, Dv, K_ = gdn0.Nv, gdn0.Dk, gdn0.Dv, gdn0.K
            blk = (dec.mi[0] * B + 0) * Nv
            ring = ttnn.to_torch(ttnn.get_device_tensors(gdn0._spec_ring)[0])[blk : blk + Nv].clone()
            win = ttnn.to_torch(ttnn.get_device_tensors(gdn0.verify_win_cur())[0])[0].clone()
            blocks = page_tables[0].tolist()
            kc = ttnn.to_torch(ttnn.get_device_tensors(att0.paged_k)[0])[blocks].clone()
            return ring, win, kc

        def _assert_hold_noop(before, after, what):
            for name, x, y in zip(("GDN ring slot", "conv window row", "attention K blocks"), before, after):
                assert torch.equal(
                    x, y
                ), f"{what}: HOLD is not a no-op for slot 0's {name} (max |delta| {float((x.float() - y.float()).abs().max()):.3e})"
            logger.info(f"[serving] {what} left slot 0's ring slot, conv window and KV blocks bit-identical")

        before = _snap()
        firstB = _prefill(model, dec, 1, prompts[1], page_tables)
        dec.begin(1, firstB, len(prompts[1]), page_tables[1])
        _assert_hold_noop(before, _snap(), "seed of slot 1 (begin)")
        outB = [firstB]
        # ...and a speculative step in which slot 1 steps while slot 0 HOLDS (step(only=)): the hold the
        # serving loop relies on every time slots run at different cadences. With QWEN36_DFLASH_FOLD_SEED=1
        # begin() replays nothing, so this is also where slot 1's folded seed row runs next to A's hold.
        before = _snap()
        com = dec.step(only=[1])
        assert set(com) == {1}, f"step(only=[1]) committed for slots {sorted(com)}"
        outB.extend(com[1])
        _assert_hold_noop(before, _snap(), "step(only=[1]) with slot 0 held")
        outC = None
        while dec.active[0] or dec.active[1]:
            com = dec.step()
            if 0 in com:
                cur = outA if outC is None else outC
                cur.extend(com[0])
                if len(cur) >= MAX_NEW:
                    dec.end(0)
            if 1 in com:
                outB.extend(com[1])
                if len(outB) >= MAX_NEW:
                    dec.end(1)
            if outC is None and not dec.active[0]:
                # A is done: request C joins the freed slot 0 (while B is live, if B is still running)
                logger.info(f"[serving] C joins slot 0 (B live: {dec.active[1]})")
                firstC = _prefill(model, dec, 0, prompts[2], page_tables)
                dec.begin(0, firstC, len(prompts[2]), page_tables[0])
                outC = [firstC]
        outs = {"A": outA, "B": outB, "C": outC}
        for name in ("A", "B", "C"):
            outs[name] = outs[name][:MAX_NEW]
            logger.info(f"[serving] {name}: {outs[name]}")
            logger.info(f"[serving] {name} text: {tokenizer.decode(outs[name])!r}")
            assert len(outs[name]) == MAX_NEW, f"{name}: {len(outs[name])} tokens"
    finally:
        dec.release()
        _release(model)

    # NOTE: C joined slot 0 while B was live only if A finished first; either way every request ran
    # at least part of its life next to another slot's holds or speculation.
    del dec, model
    gc.collect()

    ref_model, pt1, kv1 = _reference_model(device)
    try:
        for name, prompt in (("A", prompts[0]), ("B", prompts[1]), ("C", prompts[2])):
            ref, gaps = _reference_greedy(ref_model, prompt, pt1, kv1, outs[name])
            _assert_lossless(outs[name], ref, gaps, tokenizer, f"serving request {name}")
    finally:
        _release(ref_model)
    logger.info("[serving] all three requests lossless across joins and leaves")
