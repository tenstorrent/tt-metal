# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving stress: alternate eager masked-bucket prefill with traced batched decode.

WHY THIS EXISTS
---------------
A vLLM benchmark sweep on T3K hung the device during sweep point 8 (ISL 1024, OSL 128,
concurrency 8, 32 requests). The host was stuck in
``gdn/tp.py::forward_prefill -> _causal_conv1d_fir -> ttnn.concat``
(ttnn_gated_deltanet.py) and the device reported::

    TT_THROW: TIMEOUT: device timeout in fetch queue wait, potential hang detected

which killed the vLLM EngineCore (EngineDeadError -> HTTP 503). A separate run instead hung
in decode immediately after a 4K prefill. The common factor in both is that a captured decode
trace and a NON-traced prefill take turns on the same (1, 8) mesh, which is exactly what
continuous batching does every time a slot recycles.

Nothing in the existing suite drives that alternation repeatedly:
``test_model_tp_prefill_paged_slots`` does one prefill then a few decode steps, and
``test_model_tp_decode_batched`` never prefills mid-flight. This test closes that gap.

WHAT THIS TEST ACTUALLY ESTABLISHED (2026-09-23, T3K, 8 layers)
---------------------------------------------------------------
Both arms hang, which REFUTES the trace/eager-alternation hypothesis this test was written to
check:

    arm                     hang site                          when
    traced decode           _project_qkvzab (gdn/tp.py:469)    cycle 5, ~23 s in
    eager decode (control)  _causal_conv1d_fir (:163)          the INITIAL admission

The control arm captures no trace at all, so a parked decode trace is NOT required. And the two
arms run identical code up to the initial admission yet diverged there, so the hang is
NON-DETERMINISTIC -- a race, not a deterministic trace interaction.

What both share is ``prefill_paged_slots`` driving the eager masked-bucket GDN prefill for
several requests. The control's site is exactly the one the vLLM benchmark reported
(``_causal_conv1d_fir`` -> ``ttnn.concat``); the traced arm blocked one op earlier, in the
in-projection, which fits the device stalling on something earlier and the host merely blocking
at its next enqueue. So the specific op is an observation point, not the cause.

REPRODUCTION IS FLAKY -- READ THIS BEFORE TRUSTING A PASS
---------------------------------------------------------
It hung on the two runs above, then two consecutive 20-cycle runs passed cleanly -- including a
deliberate N>1 positive control that was supposed to hang. So a PASS here does NOT mean the bug
is absent, and this test cannot yet validate a fix or compare configurations.

That already invalidated one experiment: ``QWEN36_STRESS_PREFILL=one`` (one request per
prefill_paged_slots call, 20 cycles) passed, but so did the N>1 control in the same session, so
nothing can be concluded about batched-vs-single prefill from it.

MEASURED HIT RATE: 1 hang in 8 runs (12.5%) at ``QWEN36_STRESS_CYCLES=8``, ~20 s per run. The
hang landed at cycle 7/8 there, at cycle 5/6 in one earlier run, and at the initial admission in
another, so the hazard looks roughly per-cycle (~1.6%/cycle) rather than tied to one spot.

HOW TO USE IT ANYWAY
--------------------
At 12.5%/run a single pass is meaningless, but repetition is cheap: P(no hang | 12.5%) is
0.875^N, so ~30 consecutive clean runs (~10 min) is strong evidence (p ~ 0.02) that a fix works.
Raising ``QWEN36_STRESS_CYCLES`` should raise the per-run rate roughly proportionally if the
hazard really is per-cycle -- 40 cycles would be a much sharper single run. Always pair a
candidate fix with an unfixed positive control in the same session; the reason this test's first
narrowing attempt was void is that the control silently failed to reproduce.

WHY ISL < 2048 MATTERS
----------------------
``prefill_masked_bucket`` is the path every sub-2048 request takes, and it passes a real
``valid_len`` down to the GDN, so these prompts keep the test on the EAGER masked-bucket
prefill rather than a replayed trace. That is the scheduling shape that hung under serving.

The conv arm it reaches is the NATIVE depthwise ``ttnn.conv1d``: ``gdn/tp.py`` admits a
``valid_len`` whenever ``K-1 <= valid_len <= T`` on a single sequence, which every entry in
``PROMPT_LENS`` satisfies. The MAC FIR is now reachable only with ``valid_len < K-1`` or a
per-row ``valid_len`` list, so this test does NOT cover it; forcing the FIR needs a prompt
shorter than the conv kernel or a batched prefill.

FAILURE MODE
------------
A regression here is a HANG or a device TT_FATAL, not a failed assertion -- so this test is
only meaningful with a pytest timeout (``--timeout`` locally; no CI leg runs this file). The
assertions on finiteness exist to catch state corruption that stops short of a hang, and they
are weak: corrupted GDN state in bf16 is still finite. A pass is close to "did not hang".

Run::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.8-27B \
      pytest models/demos/wormhole/qwen38/tests/test_serving_stress.py -svq --timeout 2400

Knobs: ``QWEN36_STRESS_CYCLES`` (default 6), ``QWEN36_STRESS_LAYERS`` (default 8; set 64 to
run the full model if the reduced depth does not reproduce).
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen36.tt.model import Qwen36Model
from models.demos.wormhole.qwen38.tests.test_factory import model_path, parametrize_mesh_tp
from models.tt_transformers.tt.common import copy_host_to_device

# Sub-2048 so every prefill takes prefill_masked_bucket -> valid_len set -> eager, native conv1d.
# Varied so slots do not all sit at the same position (mirrors real serving).
PROMPT_LENS = [896, 960, 1024, 1088]
DECODES_PER_CYCLE = 4  # traced decode steps between re-admissions
SLOTS_PER_CYCLE = 2  # how many slots recycle each cycle


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [8], ids=["B8"])
def test_trace_eager_alternation_stress(mesh_device, B, reset_seeds, ensure_gc):
    """Repeatedly alternate a captured batched decode trace with eager per-slot prefill.

    Each cycle: replay the decode trace DECODES_PER_CYCLE times, then re-admit SLOTS_PER_CYCLE
    slots through prefill_paged_slots (eager, masked-bucket, native conv1d) while the other slots
    stay live. That is the continuous-batching pattern that hung the benchmark.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) serving path"

    cycles = int(os.environ.get("QWEN36_STRESS_CYCLES", "6"))
    n_layers = int(os.environ.get("QWEN36_STRESS_LAYERS", "8"))

    # The masked prefill writes K/V across the FULL rounded-up bucket, not just the real prompt
    # length (the contract is stated on prefill_paged_slots in generator_interface.py), so the page
    # table has to span the bucket: 1088 rounds up to 2048, i.e. 32 blocks, not the 17 the prompt
    # alone would need. Budgeting from the prompt lets the masked write run past the mapped blocks.
    block_size = 64
    bucket = Qwen36Model._mask_bucket_for(max(PROMPT_LENS))
    bpu = -(-(bucket + cycles * DECODES_PER_CYCLE) // block_size)
    max_seq_len = block_size * bpu
    assert max(PROMPT_LENS) < 2048, "prompts must stay under the 2048 chunk so the masked path is used"

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=max_seq_len, n_layers=n_layers)
    args = model.args
    vocab = args.vocab_size

    num_blocks = B * bpu
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    model.allocate_kv_caches(
        (num_blocks, args.n_local_kv_heads, block_size, args.head_dim), ttnn.bfloat16, batch_size=B
    )

    # Batched prefill warmup exactly as qwen36_vllm.warmup_model_prefill: bind a B=1 GDN scratch,
    # warm the masked buckets (capture_chunk_trace=False -> no chunk trace parked), restore the
    # batched decode buffers. Request-time prefills then only REPLAY pre-warmed programs; a
    # compile at request time is what clobbers parked traces.
    warmup_pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    prev = model._alloc_gdn_scratch_b1()
    try:
        model.capture_prefill_trace_chunked(mesh_device, warmup_pt, chunk_size=2048, capture_chunk_trace=False)
    finally:
        model._restore_gdn_batched(prev)

    torch.manual_seed(0)
    prompt_lens = [PROMPT_LENS[u % len(PROMPT_LENS)] for u in range(B)]
    prompts = [torch.randint(0, vocab, (prompt_lens[u],)).tolist() for u in range(B)]

    # NARROWING knob: "one" calls prefill_paged_slots once per request (N=1) instead of once
    # with N requests. Separates "batched per-slot prefill" from "this prefill path at all".
    prefill_mode = os.environ.get("QWEN36_STRESS_PREFILL", "batch").lower()

    def admit(tok_list, slots, lens):
        """Prefill requests into slots. page_table is indexed by REQUEST, not slot."""
        if prefill_mode == "one":
            out = []
            for k, u in enumerate(slots):
                pt1 = page_table[torch.tensor([u], dtype=torch.long)]
                out.extend(model.prefill_paged_slots([tok_list[k]], pt1, [u], valid_lens=[lens[k]]))
            return out
        pt = page_table[torch.tensor(slots, dtype=torch.long)]
        return model.prefill_paged_slots(tok_list, pt, slots, valid_lens=lens)

    # ---- initial admission of all B slots (eager masked-bucket prefill) ----
    token_list = [torch.tensor([prompts[u]], dtype=torch.long) for u in range(B)]
    pf_host = admit(token_list, list(range(B)), prompt_lens)
    toks = [int(torch.argmax(pf_host[u].reshape(-1, vocab)[0])) for u in range(B)]
    pos = list(prompt_lens)

    # ---- capture the batched decode trace ----
    # Capture runs a forward that advances GDN state, so snapshot/restore around it (ttnn.copy
    # preserves the trace buffers' addresses, which execute_trace has baked in).
    gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)

    def snapshot():
        return [
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
            )
            for dn in gdn
        ]

    def restore(snap):
        for dn, (rec, convs) in zip(gdn, snap):
            ttnn.copy(
                ttnn.from_torch(
                    rec, dtype=dn.rec_state.dtype, layout=dn.rec_state.layout, mesh_mapper=mapper, device=mesh_device
                ),
                dn.rec_state,
            )
            for c_dev, c_host in zip(dn.conv_states, convs):
                ttnn.copy(
                    ttnn.from_torch(
                        c_host, dtype=c_dev.dtype, layout=c_dev.layout, mesh_mapper=mapper, device=mesh_device
                    ),
                    c_dev,
                )

    tokens_step = torch.tensor([[toks[u]] for u in range(B)], dtype=torch.int32)
    pos_t = torch.tensor(pos, dtype=torch.int32)
    dev = model.prepare_inputs_decode(tokens_step, pos_t, page_table)

    # QWEN36_STRESS_EAGER_DECODE=1 is the CONTROL arm: decode runs eagerly with no captured
    # trace, so the only thing left is repeated eager prefill. If the traced arm hangs and this
    # one does not, the trace/eager hand-off is implicated rather than the prefill itself.
    eager_decode = os.environ.get("QWEN36_STRESS_EAGER_DECODE") == "1"
    trace_id, tt_logits = None, None
    if not eager_decode:
        snap = snapshot()
        model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])  # warm
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_logits = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])[0]
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        restore(snap)
    logger.info(
        f"decode {'EAGER (control arm)' if eager_decode else 'trace captured'} "
        f"(B={B}, layers={n_layers}, prefill={prefill_mode}); starting {cycles} alternation cycles"
    )

    def decode_step():
        host = model.prepare_decode_inputs_host(
            torch.tensor([[toks[u]] for u in range(B)], dtype=torch.int32),
            torch.tensor(pos, dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(host[:3], device_tensors=dev[:3])
        if eager_decode:
            out = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])[0]
        else:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            out = tt_logits
        ttnn.synchronize_device(mesh_device)
        lg = model.process_output_decode(out, B)  # [B, 1, vocab]
        assert torch.isfinite(lg).all(), "decode logits went non-finite"
        for u in range(B):
            toks[u] = int(torch.argmax(lg[u, 0, :vocab]))
        for u in range(B):
            pos[u] += 1

    try:
        for cycle in range(cycles):
            for _ in range(DECODES_PER_CYCLE):
                decode_step()

            # Recycle SLOTS_PER_CYCLE slots: a fresh request is prefilled eagerly into each while
            # the remaining slots stay mid-generation. This is the trace/eager hand-off.
            slots = [(cycle * SLOTS_PER_CYCLE + k) % B for k in range(SLOTS_PER_CYCLE)]
            new_lens = [PROMPT_LENS[(cycle + k) % len(PROMPT_LENS)] for k in range(len(slots))]
            new_toks = [torch.randint(0, vocab, (1, new_lens[k])).long() for k in range(len(slots))]
            pf = admit(new_toks, slots, new_lens)
            for k, u in enumerate(slots):
                row = pf[k].reshape(-1, vocab)[0].float()
                assert torch.isfinite(row).all(), f"cycle {cycle}: prefill logits non-finite for slot {u}"
                toks[u] = int(torch.argmax(row))
                pos[u] = new_lens[k]
            logger.info(f"cycle {cycle + 1}/{cycles}: re-admitted slots {slots} lens {new_lens}; positions {pos}")
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)

    logger.info(
        f"PASSED: {cycles} cycles x ({DECODES_PER_CYCLE} traced decodes + {SLOTS_PER_CYCLE} eager prefills) "
        f"with no device hang"
    )
