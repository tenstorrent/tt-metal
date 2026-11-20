# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ends of a decode step around the fused layers, through the existing ttnn modules (tt/embedding.py, tt/ihc.py
TtIHCHead, tt/lm_head.py), and what the hand-over between their tensors and the root chip's streams costs.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x

    tokens -> TtHy4Embedding (device) -> host -> the 4 streams (4 copies of the embedding row, reference/ihc.py
              ihc_expand) -> the root's site streams (Stack.set_streams)
           -> the 19 programs of layers 0..5 (fused/model, launched one by one)
           -> the root's streams -> host -> device in the ttnn modules' layout (hidden split over the mesh columns)
           -> TtIHCHead, final norm + LM head, greedy token (device) -> host
Both hand-overs go through the host in this build. Logged (lines HY4_MK11): each stage's wall time (best of 5, with
its synchronize), the embedding against the golden's streams, the logits and the greedy token against HF's and against
the reference tail on the same streams.
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.model.tests.test_chain import (
    Chain,
    MOE_FIDELITY,
    PROMPT,
    galaxy,
    head_reference,
    host_logits,
)
from models.experimental.hy4_preview.reference.ihc import ihc_expand
from models.experimental.hy4_preview.tests.common import HY4_HF_MODEL_ENV, pcc
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import hidden_to_torch, streams_to_device
from models.experimental.hy4_preview.tt.model import (
    HEAD_GROUP,
    WEIGHT_CACHE_ENV,
    Checkpoint,
    ModelConfig,
    WeightCache,
    build_head,
)


def best(fn, mesh, repeats=5):
    """(what fn returned last, the best wall time in ms of fn followed by a synchronize)."""
    times, out = [], None
    for _ in range(repeats):
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh)
        times.append(1e3 * (time.perf_counter() - start))
    return out, min(times)


@galaxy
def test_ends(mesh_device):
    chain = Chain(mesh_device)
    st, mesh = chain.st, mesh_device
    try:
        plan = MeshPlan(mesh)
        cfg = ModelConfig(num_layers=chain.n)
        ckpt = Checkpoint(os.environ[HY4_HF_MODEL_ENV])
        weights = None
        if os.environ.get(WEIGHT_CACHE_ENV):
            weights = WeightCache(os.environ[WEIGHT_CACHE_ENV], plan)
            if not weights.complete(HEAD_GROUP):
                weights = None  # (never write another piece's cache from here)
        t0 = time.perf_counter()
        embedding, hc_head, lm_head = build_head(plan, cfg, ckpt, weights)
        ttnn.synchronize_device(mesh)
        logger.info(
            f"HY4_MK11 ends: embedding, head readout and LM head built in {time.perf_counter() - t0:.0f} s beside the six fused "
            f"layers (L1 tensors {st.l1_bytes()} B per core)"
        )
        chain.set_batch(0)
        tokens = chain.g["input_ids"][chain.positions]
        programs = st.step_programs()
        label = f"ends P{PROMPT} [{MOE_FIDELITY}]"

        # in: tokens -> embedding on the device -> host -> the root's streams
        emb, ms_embed = best(lambda: embedding(embedding.host_table_rows(tokens)), mesh)
        if is_mock():
            Stack.launch(programs)
            return
        host_emb, ms_read = best(lambda: hidden_to_torch(plan, emb), mesh)
        streams_in = ihc_expand(host_emb.unsqueeze(0), 4)[0].float()
        gold_in = chain.gold(0, "streams_in")
        _, ms_write = best(lambda: st.set_streams(streams_in), mesh)
        logger.info(
            f"HY4_MK11 {label} IN: ttnn embedding of 8 tokens (host ids up, lookup, spread over the hidden axis) {ms_embed:.2f} ms; "
            f"embedding to the host {ms_read:.2f} ms; host to the root's streams {ms_write:.2f} ms; streams against the golden's "
            f"max_abs={(gold_in - streams_in).abs().max():.3e} (equal: {torch.equal(gold_in, streams_in)})"
        )

        # the layers
        Stack.launch(programs)
        ttnn.synchronize_device(mesh)
        status = st.status()
        assert not status, f"words not at rest after the step: {status}"

        # out: the root's streams -> host -> the ttnn layout -> head readout, norm, LM head, greedy
        streams, ms_out_read = best(st.read_streams, mesh)
        tt_streams, ms_out_write = best(lambda: streams_to_device(plan, streams), mesh)

        def tail():
            logits = lm_head(hc_head(tt_streams))
            return logits, lm_head.greedy(logits)

        (logits, ids), ms_tail = best(tail, mesh)
        got, ms_ids = best(lambda: ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).flatten()[:8].long(), mesh)
        dev_logits = lm_head.to_torch(logits)
        g, J = chain.g, chain.J
        gl, want = g["logits"][J].float(), g["cached_logits_argmax"][J]
        head, lm = head_reference()
        _, ref_logits = host_logits(head, lm, streams)
        logger.info(
            f"HY4_MK11 {label} OUT: the root's streams to the host {ms_out_read:.2f} ms; host to the ttnn layout {ms_out_write:.2f} ms; "
            f"ttnn head readout + final norm + LM head + greedy {ms_tail:.2f} ms; token ids to the host {ms_ids:.2f} ms"
        )
        logger.info(
            f"HY4_MK11 {label} ttnn tail on the fused layers' streams: logits vs HF pcc={pcc(gl, dev_logits):.6f} (per row min "
            f"{min(pcc(gl[r], dev_logits[r]) for r in range(8)):.6f}), vs the reference tail on the same streams pcc="
            f"{pcc(ref_logits, dev_logits):.6f}; greedy token equal to HF's on {int((got == want).sum())} of 8 rows, to the reference "
            f"tail's on {int((got == ref_logits.argmax(-1)).sum())} of 8 (device {got.tolist()}, HF {want.tolist()})"
        )
        logger.info(
            f"HY4_MK11 {label} hand-over cost per step through the host: in {ms_read + ms_write:.2f} ms, out {ms_out_read + ms_out_write:.2f} ms"
            f"; the ttnn modules themselves: embedding {ms_embed:.2f} ms, tail {ms_tail:.2f} ms"
        )
    finally:
        chain.free()
