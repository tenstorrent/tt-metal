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

Next narrowing step: run ``prefill_paged_slots`` with ONE request per call instead of N, to
separate "batched per-slot prefill" from "this prefill path at all".

WHY ISL < 2048 MATTERS
----------------------
``prefill_masked_bucket`` is the path every sub-2048 request takes, and it passes a real
``valid_len`` down to the GDN. ``gdn/tp.py`` then selects the conv implementation::

    if self._gdn_conv1d and valid_len is None:   # traced chunk path
        conv, conv_new_state = self._conv1d_prefill(...)      # native depthwise conv1d
    else:                                        # masked bucket path
        conv, conv_new_state = _causal_conv1d_fir(...)        # MAC FIR -- the hang site

So a masked-bucket prefill is both eager AND on the FIR conv. Keep the prompts under 2048 or
this test stops exercising the code that hung.

FAILURE MODE
------------
A regression here is a HANG or a device TT_FATAL, not a failed assertion -- so this test is
only meaningful with a pytest timeout (the CI leg supplies one; ``--timeout`` locally). The
assertions on finiteness exist to catch state corruption that stops short of a hang.

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

# Sub-2048 so every prefill takes prefill_masked_bucket -> valid_len set -> FIR conv.
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
    slots through prefill_paged_slots (eager, masked-bucket, FIR conv) while the other slots
    stay live. That is the continuous-batching pattern that hung the benchmark.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) serving path"

    cycles = int(os.environ.get("QWEN36_STRESS_CYCLES", "6"))
    n_layers = int(os.environ.get("QWEN36_STRESS_LAYERS", "8"))

    block_size, bpu = 64, 24  # 24 * 64 = 1536 tokens per user, covers the longest prompt + decode
    max_seq_len = block_size * bpu
    assert max(PROMPT_LENS) < 2048, "prompts must stay under the 2048 chunk so the masked path is used"
    assert max(PROMPT_LENS) + cycles * DECODES_PER_CYCLE < max_seq_len, "prompt + decode must fit bpu blocks"

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

    # ---- initial admission of all B slots (eager masked-bucket prefill) ----
    token_list = [torch.tensor([prompts[u]], dtype=torch.long) for u in range(B)]
    pf_host = model.prefill_paged_slots(token_list, page_table, list(range(B)), valid_lens=prompt_lens)
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
        f"(B={B}, layers={n_layers}); starting {cycles} alternation cycles"
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
            # page_table is indexed by REQUEST, not slot: one row per prefilled request, with
            # empty_slots[u] naming its destination slot (see prefill_paged_slots' assert).
            slot_pt = page_table[torch.tensor(slots, dtype=torch.long)]
            pf = model.prefill_paged_slots(new_toks, slot_pt, slots, valid_lens=new_lens)
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
