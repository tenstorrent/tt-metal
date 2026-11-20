# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The step's ends on the device (fused/model/ends.py): first the two layout conversions alone against the host
(test_layout: only ttnn ops and one L1 tensor in the site's layout; no fused program runs), then a whole step as one
trace (test_step): token ids -> embedding -> the root's streams -> the fused layers -> head -> greedy ids.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x -k layout
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.ihc_site import layout as ihc_layout
from models.experimental.hy4_preview.fused.ihc_site.op import IhcSite
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.model import ends
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.model.tests.test_chain import (
    MOE_FIDELITY,
    PROMPT,
    Chain,
    full_status,
    galaxy,
    head_reference,
    host_logits,
)
from models.experimental.hy4_preview.reference.ihc import IHCConfig, ihc_expand
from models.experimental.hy4_preview.tests.common import HY4_HF_MODEL_ENV, pcc
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import streams_to_torch
from models.experimental.hy4_preview.tt.model import (
    HEAD_GROUP,
    WEIGHT_CACHE_ENV,
    Checkpoint,
    ModelConfig,
    WeightCache,
    build_head,
)

C = 6144
# A queued job of this file runs only if the job queued before it passed (HY4_GUARD_FILE: that queue's output; a line
# "DID NOT PASS" in it means a device job failed or timed out, after which nothing may open the board).
_guard = os.environ.get("HY4_GUARD_FILE")
if _guard and os.path.exists(_guard) and "DID NOT PASS" in open(_guard).read():
    pytest.skip("the job queued before this one did not pass: no device work", allow_module_level=True)


def build_ends(mesh_device, layers):
    plan = MeshPlan(mesh_device)
    cfg = ModelConfig(num_layers=layers)
    ckpt = Checkpoint(os.environ[HY4_HF_MODEL_ENV])
    weights = None
    if os.environ.get(WEIGHT_CACHE_ENV):
        weights = WeightCache(os.environ[WEIGHT_CACHE_ENV], plan)
        if not weights.complete(HEAD_GROUP):
            weights = None
    return (plan,) + tuple(build_head(plan, cfg, ckpt, weights))


@galaxy
def test_layout(mesh_device):
    """IN: the ttnn embedding of 8 tokens as streams pages on every chip against the host's pages of the reference
    embedding. OUT: known streams in the root's pages (other data on the other 31 chips) through the mask, the
    broadcast and the layout back, against the host."""
    plan, embedding, _, _ = build_ends(mesh_device, 6)
    site = IhcSite(mesh_device, IHCConfig(dim=C), f_dtype=ttnn.bfloat16, **A.SITE)
    config = site._memory_config(C)
    g = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, 120832, (8,), generator=g)
    ids = embedding.host_token_ids(tokens)
    pages = ends.pages_from_embedding(ends.embedding_whole(plan, embedding, ids), config)
    mask = ends.root_mask(mesh_device)
    streams = torch.randn(8, 4, C, generator=g)
    shards = []
    for chip in range(mesh_device.get_num_devices()):
        data = streams if chip == 0 else torch.randn(8, 4, C, generator=g)
        m = ihc_layout.streams_to_pages(data).reshape(1, 1, 32, C)
        shards.append(ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT))
    held = site.streams_to_device(torch.zeros(8, 4, C))
    ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, mesh_device.shape), held)
    out = ends.streams_from_pages(plan, held, mask)
    if is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    from models.experimental.hy4_preview.tests.common import hy4_tensors

    (table,) = hy4_tensors(["model.embed_tokens.weight"])
    want = ihc_layout.streams_to_pages(ihc_expand(table[tokens].unsqueeze(0), 4)[0].float())
    got = [ttnn.to_torch(t).float()[0, 0] for t in ttnn.get_device_tensors(pages)]
    equal = sum(torch.equal(v, want) for v in got)
    back = streams_to_torch(plan, out, 4)
    logger.info(
        f"HY4_MK11 ends layout IN: the embedding as streams pages equal to the host's on {equal} of {len(got)} chips (max abs "
        f"{max(float((v - want).abs().max()) for v in got):.3e}); memory config of the result {pages.memory_config().memory_layout}; "
        f"OUT: the root's streams back in the modules' layout: equal to the host's {torch.equal(back, streams)} (max abs "
        f"{float((back - streams).abs().max()):.3e}, pcc {pcc(streams, back):.8f}, largest relative error "
        f"{float(((back - streams).abs() / streams.abs().clamp_min(1e-3)).max()):.3e}; the BF16 rounding HF itself applies "
        f"to the streams before the head: {float((streams - streams.to(torch.bfloat16).float()).abs().max()):.3e})"
    )
    # IN is exact. OUT is the layout exactly, but the fp32 values pass ttnn.multiply and ttnn.all_reduce, which carry
    # an fp32 operand at the FPU's reduced mantissa: below the BF16 rounding HF applies to the streams before the head
    assert equal == len(got)
    assert pcc(streams, back) > 0.999999 and float((back - streams).abs().max()) < 2.0**-7 * float(
        streams.abs().max()
    )


def use_streams(st, tensor):
    """The fused programs' streams tensor from now on (the site's CBs are bound to it when descriptors are built)."""
    st.a.streams = tensor
    st.a._programs = {}
    for m in st.mlps.values():
        m.streams = tensor
        m._programs = {}


@galaxy
def test_step(mesh_device):
    """Two decode steps as ONE trace with nothing from the host between embedding, layers, head and the next step:
    ids -> embedding -> root streams -> the fused layers -> head -> greedy ids -> (step 2) embedding -> ...
    The cache rows of both steps are HF's (the append is not built), each step has its selection table and RoPE
    tensor (host-written before the trace). Checked: every status, the canary, the tokens and logits of both steps
    against HF's; then the time of the trace against the same layers without the ends."""
    chain = Chain(mesh_device, wait_limit=30_000_000)
    st, mesh = chain.st, mesh_device
    plan, embedding, hc_head, lm_head = build_ends(mesh, chain.n)
    try:
        chain.set_batch(0, steps=2)
        mask = ends.root_mask(mesh)
        config = st.site._memory_config(C)
        ttnn.deallocate(st.a.streams)  # (the streams tensor is the embedding's output from here on)
        g, J = chain.g, chain.J
        tokens = g["input_ids"][chain.positions]
        ids_in = embedding.host_token_ids(tokens)
        layer_programs = {}

        def step(ids, k):
            pages = ends.pages_from_embedding(ends.embedding_whole(plan, embedding, ids), config)
            use_streams(st, pages)
            chain.use_step(k)
            layer_programs[k] = st.step_programs()
            Stack.launch(layer_programs[k])
            logits = lm_head(hc_head(ends.streams_from_pages(plan, pages, mask)))
            return pages, logits, lm_head.greedy(logits)

        def two_steps():
            first = step(ids_in, 0)
            second = step(first[2], 1)
            return first, second

        label = f"ends in the trace, {chain.n} layers P{PROMPT} [{MOE_FIDELITY}]"
        t0 = time.perf_counter()
        results = two_steps()
        if is_mock():
            return
        ttnn.synchronize_device(mesh)
        untraced_s = time.perf_counter() - t0
        status = st.status()
        assert not status, f"words not at rest after the untraced steps: {status}"
        head, lm = head_reference()

        def check(results, what):
            out = []
            for k, (pages, logits, ids) in enumerate(results):
                streams = ihc_layout.pages_to_streams(ttnn.to_torch(ttnn.get_device_tensors(pages)[0]).float()[0, 0])
                got = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).flatten()[:8].long()
                dev_logits = lm_head.to_torch(logits)
                jk = [j + k for j in J]
                gl, want = g["logits"][jk].float(), g["cached_logits_argmax"][jk]
                gold = g["layers"][chain.n - 1]["streams_out"][jk].float()
                _, ref_logits = host_logits(head, lm, streams)
                off = ends.off_shards(dev_logits, ref_logits)
                if off:  # (PLAN.md section 42)
                    logger.warning(
                        f"HY4_MK11 {label} {what} step {k + 1}: ENDS OFF: chips' shards of the vocabulary that leave the reference tail on the step's own streams {off}"
                    )
                fed = (
                    "HF's tokens"
                    if k == 0
                    else f"the device's tokens of step 1 ({int((out[0] == g['cached_logits_argmax'][J]).sum())} of 8 are HF's)"
                )
                logger.info(
                    f"HY4_MK11 {label} {what} step {k + 1} (fed {fed}): streams after the last layer vs HF pcc={pcc(gold, streams):.6f}; "
                    f"device logits vs HF pcc={pcc(gl, dev_logits):.6f}, vs the reference tail on the same streams {pcc(ref_logits, dev_logits):.6f}; "
                    f"greedy token = HF's on {int((got == want).sum())} of 8 rows, = the reference tail's on {int((got == ref_logits.argmax(-1)).sum())} of 8"
                )
                assert bool(torch.isfinite(streams).all())
                out.append(got)
            return out

        check(results, "untraced")
        for pages, logits, ids in results:
            ttnn.deallocate(pages)
        results = None
        trace_id, results = tracing.capture(mesh, two_steps)
        ttnn.synchronize_device(mesh)
        try:
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
            logger.info(
                f"HY4_MK11 {label}: trace buffer {view.total_bytes_allocated_per_bank} B per bank; untraced the two steps took {1e3 * untraced_s:.0f} ms"
            )
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            status = full_status(chain)
            assert not status, f"a status is left after the traced steps: {status}"
            tokens_out = check(results, "ONE trace")
            canary = st.canary_summary()
            logger.info(f"HY4_MK11 {label} canary: {canary}")
            for kind, v in canary.items():
                assert v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"], f"canary {kind}: {v}"
            repeats = int(os.environ.get("HY4_ENDS_REPEATS", "0"))
            if repeats:
                # PLAN.md section 42: replays whose logits are off while the streams are right. Each replay's device
                # logits against the reference tail on that replay's OWN streams, per chip's shard of the vocabulary
                off, worst = [], 1.0
                for i in range(repeats):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    for k, (pages, logits, ids) in enumerate(results):
                        streams = ihc_layout.pages_to_streams(
                            ttnn.to_torch(ttnn.get_device_tensors(pages)[0]).float()[0, 0]
                        )
                        dev = lm_head.to_torch(logits)
                        _, ref = host_logits(head, lm, streams)
                        shard = dev.shape[1] // 32
                        per = [
                            min(
                                pcc(dev[u, c * shard : (c + 1) * shard], ref[u, c * shard : (c + 1) * shard])
                                for u in range(8)
                            )
                            for c in range(32)
                        ]
                        worst = min(worst, min(per))
                        if min(per) < 0.999:
                            off.append(
                                (
                                    i,
                                    k,
                                    [c for c in range(32) if per[c] < 0.999],
                                    round(min(per), 4),
                                    [round(pcc(dev[u], ref[u]), 5) for u in range(8)],
                                )
                            )
                status = full_status(chain)
                logger.info(
                    f"HY4_MK11 {label} ENDS over {repeats} single replays of the two-step trace ({2 * repeats} steps): steps whose device "
                    f"logits leave the reference tail on their own streams in some chip's shard of the vocabulary (below 0.999): {len(off)}: "
                    f"(replay, step, chips, the shard's min pcc, logits pcc per row) {off}; the lowest shard pcc of all steps {worst:.5f}; status {status}"
                )
                assert not status
            if os.environ.get("HY4_CHAIN_TRACE_JOB", "A") == "A":
                logger.info(f"HY4_MK11 {label} job A done: two steps with their ends as one trace, every status empty")
                return
            rounds = [tracing.replay_ms(mesh, trace_id, 20) / 2 for _ in range(3)]
            ttnn.synchronize_device(mesh)
            status = full_status(chain)
            assert not status, f"a status is left after the replays: {status}"
            check(results, "after 61 replays")
        finally:
            ttnn.release_trace(mesh, trace_id)
        # the same two steps' layers alone, on the streams tensor the trace left
        both = layer_programs[0] + layer_programs[1]
        r = tracing.traced_ms(mesh, lambda: Stack.launch(both))
        layers_ms = r["replay_ms"] / 2
        best = min(rounds)
        logger.info(
            f"HY4_MK11 {label} LATENCY per step, 8 rows, two steps per trace, 20 replays back to back: WITH the ends {best:.3f} ms "
            f"(rounds {[round(v, 3) for v in rounds]}) = {8e3 / best:.1f} tokens/s; the layers alone {layers_ms:.3f} ms = "
            f"{8e3 / layers_ms:.1f} tokens/s; the ends {1e3 * (best - layers_ms):.0f} us per step; at 78 layers by these rates "
            f"{best * 78 / chain.n:.1f} ms with a {chain.n}-layer step's ends scaled (not meaningful unless this IS 78 layers); "
            f"host per step: none inside the trace (the ids of every step are on the device for a read-back)"
        )
        status = st.status()
        assert not status, f"words not at rest after the layers-only trace: {status}"
    finally:
        chain.free()
