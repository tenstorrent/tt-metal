# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: Qwen36ForCausalLM in mamba_cache_mode "align": prefix restore, prefill checkpoints, chunked prefill and
decode block crossings, with the GDN state in vLLM KV cache blocks (tt/gdn_block_state.py).

The adapter is driven as vllm-tt-plugin drives it, with the block ids that vLLM would hand out chosen by the test, on a
truncated model: 8 layers, attention at 3 and 7, so 2 layer slots and 3 GDN groups of 2 layers. Every restored state is
compared with a cold prefill of the same tokens. ``kv_bf8`` runs bfloat8_b K and V (QWEN_SDPA_BF8=1): at TP=8 the
blocks then hold 1024 tokens and even and odd blocks start their float32 state pages at different offsets, so the
tests place start, checkpoint and destination blocks on both parities.

    MESH_DEVICE=P150x8 HF_MODEL=<Qwen3.8-27B dir> pytest models/demos/blackhole/qwen36/tests/test_gdn_prefix_restore.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.tests.vllm_stubs import stub_vllm_if_missing
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc
from models.demos.blackhole.qwen36.tests.test_gdn_block_state import (
    BMAX,
    _build,
    _compare_state,
    _decode_flags,
    _gdn_layers,
    _parametrize,
    _prompts,
    read_block_state,
)

NUM_BLOCKS = 64
PAGE_COLUMNS = 8  # page-table width: covers 4096 tokens at the 512-token bfloat16 blocks


@pytest.fixture(params=[False, True], ids=["bf16", "bf8"])
def kv_bf8(request, monkeypatch):
    if request.param:
        monkeypatch.setenv("QWEN_SDPA_BF8", "1")
    else:
        monkeypatch.delenv("QWEN_SDPA_BF8", raising=False)
    return request.param


class Run:
    """The adapter over NUM_BLOCKS blocks, with helpers that build vLLM's page tables and mamba_state."""

    def __init__(self, mesh_device, monkeypatch, kv_bf8):
        stub_vllm_if_missing(monkeypatch)
        from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

        self.model = _build(mesh_device)
        self.gen = Qwen36ForCausalLM([self.model], [self.model.args], mesh_device)
        self.layout = PluginLayout.for_model(self.model, num_blocks=NUM_BLOCKS, kv_bf8=kv_bf8)
        self.kv = self.gen.allocate_kv_cache_per_layer(self.layout.per_layer_specs)
        self.bs = self.layout.block_size
        self.vocab = self.model.args.vocab_size
        self.used = 2 * (2 * self.model.args.gdn_dk + self.model.args.gdn_dv) // 32
        self.group_of = {i: g for g, layers in enumerate(self.layout.gdn_groups) for i in layers}
        self.next_block = 1

    def blocks(self, n, parity=None):
        """n fresh block ids; with ``parity`` 0 or 1, all even or all odd."""
        out = []
        while len(out) < n:
            b = self.next_block
            self.next_block += 1
            if parity is None or b % 2 == parity:
                out.append(b)
        assert self.next_block <= NUM_BLOCKS
        return tuple(out)

    def page_row(self, columns):
        row = torch.zeros(PAGE_COLUMNS, dtype=torch.int32)
        row[: len(columns)] = torch.tensor(columns, dtype=torch.int32)
        return row

    def prefill(self, rows):
        """rows: dicts with tokens [1, T], start, end, pages (attention blocks per column), dest (block per group) and
        optionally start_blocks and checkpoint=(p, blocks). Returns host logits [N, vocab]."""
        n = len(rows)
        width = max(r["end"] for r in rows)
        tokens = torch.zeros(n, width, dtype=torch.int64)
        for i, r in enumerate(rows):
            tokens[i, : r["end"]] = r["tokens"][0, : r["end"]]
        state = self.layout.step_state(
            [r["dest"] for r in rows],
            BMAX,
            start_blocks=[r.get("start_blocks") for r in rows],
            checkpoints=[r.get("checkpoint") for r in rows],
        )
        logits, _ = self.gen.prefill_forward(
            tokens,
            torch.stack([self.page_row(r["pages"]) for r in rows]),
            self.kv,
            [r["end"] for r in rows],
            start_pos=[r["start"] for r in rows],
            mamba_state=state,
        )
        return logits[:, 0, : self.vocab].float()

    def state(self, blocks):
        """{GDN layer: per-device state} of each layer's group block in ``blocks`` (one block per group)."""
        return {i: read_block_state(self.model, i, blocks[g]) for i, g in self.group_of.items()}

    def assert_same_state(self, a, b, label, exact=True):
        for i in _gdn_layers(self.model):
            _compare_state(a[i], b[i], self.used, f"{label}: layer {i}", exact=exact)


def _row(tokens, start, end, pages, dest, **kw):
    return dict(tokens=tokens, start=start, end=end, pages=pages, dest=dest, **kw)


@torch.no_grad()
@_parametrize
def test_checkpoint_and_prefix_restore_match_cold_prefill(mesh_device, kv_bf8, monkeypatch, reset_seeds, ensure_gc):
    """(1) An unchunked 2500-token prefill that writes a checkpoint after 2048 tokens: the checkpoint equals a cold
    2048-token prefill's state, and the end state and logits equal a 2500-token prefill without a checkpoint. (2) A
    prefix-cache hit on the checkpoint resumes at 2048 with 500 other tokens: logits and end state equal a cold prefill
    of the same 2548 tokens, and the checkpoint blocks stay unchanged. Both prefills split at 2048, a multiple of the
    2048-token prefill chunk, so the computations are the same and the comparisons are exact."""
    run = Run(mesh_device, monkeypatch, kv_bf8)
    bs = run.bs
    prompt = _prompts(1, 2500, seed=11)
    other = torch.cat([prompt[:, :2048], _prompts(1, 500, seed=12)], dim=1)
    n_cols = lambda T: -(-T // bs)

    # A: one prefill of 2500 tokens with a checkpoint after 2048 (odd checkpoint blocks, even destination blocks).
    pages_a = run.blocks(n_cols(2500))
    ckpt, dest_a = run.blocks(3, parity=1), run.blocks(3, parity=0)
    logits_a = run.prefill([_row(prompt, 0, 2500, pages_a, dest_a, checkpoint=(2048, ckpt))])
    state_ckpt = run.state(ckpt)

    cold_2048 = run.blocks(3)
    run.prefill([_row(prompt, 0, 2048, run.blocks(n_cols(2048)), cold_2048)])
    run.assert_same_state(state_ckpt, run.state(cold_2048), "checkpoint vs cold 2048-token prefill")

    cold_2500 = run.blocks(3)
    logits_cold = run.prefill([_row(prompt, 0, 2500, run.blocks(n_cols(2500)), cold_2500)])
    assert torch.equal(logits_a, logits_cold), "the checkpoint changed the 2500-token prefill's logits"
    run.assert_same_state(run.state(dest_a), run.state(cold_2500), "end state with vs without a checkpoint")

    # B: prefix hit on A's checkpoint; B shares A's attention blocks of the first 2048 tokens (even destination blocks).
    shared = pages_a[: 2048 // bs]
    pages_b = shared + run.blocks(n_cols(2548) - len(shared))
    dest_b = run.blocks(3, parity=0)
    logits_b = run.prefill([_row(other, 2048, 2548, pages_b, dest_b, start_blocks=ckpt)])
    cold_b = run.blocks(3)
    logits_cold_b = run.prefill([_row(other, 0, 2548, run.blocks(n_cols(2548)), cold_b)])
    assert torch.equal(logits_b, logits_cold_b), "prefix restore: logits differ from a cold prefill"
    run.assert_same_state(run.state(dest_b), run.state(cold_b), "prefix restore: end state vs cold prefill")
    run.assert_same_state(run.state(ckpt), state_ckpt, "the checkpoint blocks changed")


@torch.no_grad()
@_parametrize
def test_chunked_prefill_matches_unchunked(mesh_device, kv_bf8, monkeypatch, reset_seeds, ensure_gc):
    """Chunked equivalents of a 2500-token prefill, each chunk resuming from the block the previous chunk wrote:
    2048 + 452 is the same computation as one prefill (exact); a chunk end on every KV block boundary (512, 1024 or
    2048 tokens) splits the first 2048-token prefill chunk unless the block is 2048 tokens, so it is compared by PCC
    then. A chunk never changes its start blocks."""
    run = Run(mesh_device, monkeypatch, kv_bf8)
    bs = run.bs
    prompt = _prompts(1, 2500, seed=21)
    cols = -(-2500 // bs)

    whole = run.blocks(3)
    logits_whole = run.prefill([_row(prompt, 0, 2500, run.blocks(cols), whole)])
    state_whole = run.state(whole)

    for ends, exact in (([2048, 2500], True), ([*range(bs, 2500, bs), 2500], bs >= 2048)):
        pages = run.blocks(cols)
        start, prev = 0, None
        for end in ends:
            dest = run.blocks(3, parity=(end // 1024) % 2)
            before = run.state(prev) if prev else None
            logits = run.prefill([_row(prompt, start, end, pages, dest, start_blocks=prev)])
            if prev:
                run.assert_same_state(run.state(prev), before, f"chunks {ends}: start blocks of chunk at {start}")
            start, prev = end, dest
        label = f"chunks {ends} vs one prefill"
        run.assert_same_state(run.state(prev), state_whole, label, exact=exact)
        if exact:
            assert torch.equal(logits, logits_whole), label
        else:
            pcc = compute_pcc(logits, logits_whole)
            logger.info(f"{label}: logits PCC {pcc:.6f}")
            assert pcc >= 0.99 and torch.equal(logits.argmax(-1), logits_whole.argmax(-1)), f"{label}: PCC {pcc}"


@torch.no_grad()
@_parametrize
def test_diverging_continuations_read_one_kept_prefix(mesh_device, kv_bf8, monkeypatch, reset_seeds, ensure_gc):
    """Three requests continue one kept 2048-token prefix with different tokens in one prefill call (all three read the
    same start blocks): each equals its own cold prefill, and the kept blocks stay unchanged."""
    run = Run(mesh_device, monkeypatch, kv_bf8)
    bs = run.bs
    prefix = _prompts(1, 2048, seed=31)
    kept, prefix_pages = run.blocks(3, parity=1), run.blocks(2048 // bs)
    run.prefill([_row(prefix, 0, 2048, prefix_pages, kept)])
    state_kept = run.state(kept)
    lens = [2048 + 300, 2048 + 700, 2048 + 1100]
    prompts = [torch.cat([prefix, _prompts(1, L - 2048, seed=32 + i)], dim=1) for i, L in enumerate(lens)]
    rows, dests = [], []
    for p, L in zip(prompts, lens):
        dests.append(run.blocks(3, parity=len(dests) % 2))
        pages = prefix_pages + run.blocks(-(-L // bs) - len(prefix_pages))
        rows.append(_row(p, 2048, L, pages, dests[-1], start_blocks=kept))
    logits = run.prefill(rows)
    run.assert_same_state(run.state(kept), state_kept, "the kept prefix blocks changed")
    for i, (p, L) in enumerate(zip(prompts, lens)):
        cold = run.blocks(3)
        cold_logits = run.prefill([_row(p, 0, L, run.blocks(-(-L // bs)), cold)])
        assert torch.equal(logits[i : i + 1], cold_logits), f"continuation {i}: logits differ from a cold prefill"
        run.assert_same_state(run.state(dests[i]), run.state(cold), f"continuation {i}")


@torch.no_grad()
@_parametrize
def test_decode_crossing_keeps_the_start_block(mesh_device, kv_bf8, monkeypatch, reset_seeds, ensure_gc):
    """Decode across a block boundary in mode "align": the step whose token crosses into a new column copies its start
    block into the destination block and decodes there. Two such rows (odd -> even and even -> odd block ids) give the
    same logits as a row that decodes in one block in place (mode "none"), and their start blocks keep the state of
    the boundary."""
    run = Run(mesh_device, monkeypatch, kv_bf8)
    bs = run.bs
    T = 2 * bs - 1  # the first decode step ends at 2 * bs tokens, the second crosses into column 2
    prompt = _prompts(1, T, seed=41)
    cols = 4
    pages = [run.blocks(cols) for _ in range(3)]
    col1 = [run.blocks(3, parity=1), None, run.blocks(3, parity=0)]
    col2 = [run.blocks(3, parity=0), None, run.blocks(3, parity=1)]
    in_place = run.blocks(3)
    dests = [col1[0], in_place, col1[2]]
    logits = run.prefill([_row(prompt, 0, T, pages[r], dests[r]) for r in range(3)])
    assert torch.equal(logits[0], logits[1]) and torch.equal(logits[2], logits[1])
    tok = logits[1].argmax(-1)
    pt = torch.zeros(BMAX, PAGE_COLUMNS, dtype=torch.int32)
    for r in range(3):
        pt[r] = run.page_row(pages[r])
    boundary = {}
    for step in range(3):
        if step == 0:
            starts, dests = [col1[0], "same", col1[2], None], [col1[0], in_place, col1[2], None]
        elif step == 1:
            starts, dests = [col1[0], "same", col1[2], None], [col2[0], in_place, col2[2], None]
        else:
            starts, dests = [col2[0], "same", col2[2], None], [col2[0], in_place, col2[2], None]
        t = torch.zeros(BMAX, 1, dtype=torch.int32)
        t[:3, 0] = tok
        pos = torch.full((BMAX,), -1, dtype=torch.int32)
        pos[:3] = T + step
        state = run.layout.step_state(dests, BMAX, start_blocks=starts)
        out, _ = run.gen.decode_forward(  # (host logits, logprobs)
            t,
            pos,
            page_table=pt,
            kv_cache=run.kv,
            enable_trace=False,
            read_from_device=True,
            mamba_state=state,
            **_decode_flags(),
        )
        lg = out[:3, -1, : run.vocab].float()
        for r in (0, 2):
            assert torch.equal(lg[r], lg[1]), f"decode step {step}: row {r} differs from the in-place row"
        tok = lg[1].argmax(-1)
        if step == 0:
            boundary = {r: run.state(col1[r]) for r in (0, 2)}
    for r in (0, 2):
        run.assert_same_state(run.state(col1[r]), boundary[r], f"row {r}: the start block of the crossing step changed")


@torch.no_grad()
@_parametrize
def test_block_operations_compile_nothing_after_warmup(mesh_device, kv_bf8, monkeypatch, reset_seeds, ensure_gc):
    """Restores, state writes and decode crossings address a block through ttnn.narrow views or write tables, so the
    programs that GdnBlockState.warmup compiles on the null block serve every block: after warmup, these operations
    on blocks of both parities add no program cache entry (a compile after a trace is captured can corrupt it)."""
    run = Run(mesh_device, monkeypatch, kv_bf8)
    gb = run.model.gdn_blocks
    gb.warmup()
    ttnn.synchronize_device(mesh_device)
    before = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for a, b in [(1, 2), (3, 6), (4, 7), (9, 10)]:
            state = run.layout.step_state([(b, b, b)], BMAX, start_blocks=[(a, a, a)])
            gb.restore(state, (a, a, a))
            gb.write_state(state, (b, b, b))
            gb.set_decode_blocks(state)
        ttnn.synchronize_device(mesh_device)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    after = mesh_device.num_program_cache_entries()
    assert after == before, f"{after - before} program(s) compiled after GdnBlockState.warmup"


@torch.no_grad()
@_parametrize
def test_checkpoint_inside_the_last_chunk(mesh_device, kv_bf8, monkeypatch, reset_seeds, ensure_gc):
    """A prefill whose checkpoint (vLLM's: the last KV block boundary before its end) lies inside its last prefill chunk
    computes the checkpoint in that chunk's forward (gdn_block_state.prefill_segments), for a partial (2000 tokens)
    and a whole (2048 tokens) last chunk. (1) Logits and end state equal the same prefill without a checkpoint
    exactly: the extra chunk_gated_delta_rule call does not touch the forward. (2) The checkpoint matches a cold
    prefill of the checkpoint's tokens, which is what a prefill that stops at the checkpoint writes there; the two
    forwards run different buckets, so by PCC. (3) A prefix hit on the checkpoint matches a cold prefill of its tokens
    (PCC and argmax) and leaves the checkpoint blocks unchanged."""
    run = Run(mesh_device, monkeypatch, kv_bf8)
    bs = run.bs
    if bs >= run.model.prefill_chunk_size:
        pytest.skip(f"{bs}-token KV blocks put every checkpoint on a prefill chunk boundary")
    n_cols = lambda T: -(-T // bs)
    for T in (2000, 2048):
        run.next_block = 1  # each length writes every block it reads first
        p = (T - 1) // bs * bs
        prompt = _prompts(1, T, seed=51 + T)
        ckpt, dest, pages = run.blocks(3, parity=1), run.blocks(3, parity=0), run.blocks(n_cols(T))
        logits = run.prefill([_row(prompt, 0, T, pages, dest, checkpoint=(p, ckpt))])
        state_ckpt = run.state(ckpt)

        cold = run.blocks(3)
        cold_logits = run.prefill([_row(prompt, 0, T, run.blocks(n_cols(T)), cold)])
        assert torch.equal(logits, cold_logits), f"T={T}: the checkpoint changed the prefill's logits"
        run.assert_same_state(run.state(dest), run.state(cold), f"T={T}: end state with vs without a checkpoint")

        cold_p = run.blocks(3)
        run.prefill([_row(prompt, 0, p, run.blocks(n_cols(p)), cold_p)])
        run.assert_same_state(state_ckpt, run.state(cold_p), f"T={T}: checkpoint vs cold {p}-token prefill", False)

        other = torch.cat([prompt[:, :p], _prompts(1, 700, seed=61 + T)], dim=1)
        end = p + 700
        hit_pages = pages[: p // bs] + run.blocks(n_cols(end) - p // bs)
        hit = run.blocks(3, parity=0)
        hit_logits = run.prefill([_row(other, p, end, hit_pages, hit, start_blocks=ckpt)])
        cold_hit = run.blocks(3)
        cold_hit_logits = run.prefill([_row(other, 0, end, run.blocks(n_cols(end)), cold_hit)])
        pcc = compute_pcc(hit_logits, cold_hit_logits)
        logger.info(f"T={T}: prefix hit on the checkpoint after {p}: logits PCC {pcc:.6f} vs cold prefill")
        assert pcc >= 0.99 and torch.equal(hit_logits.argmax(-1), cold_hit_logits.argmax(-1)), f"T={T}: PCC {pcc}"
        run.assert_same_state(run.state(hit), run.state(cold_hit), f"T={T}: prefix hit end state vs cold", False)
        run.assert_same_state(run.state(ckpt), state_ckpt, f"T={T}: the checkpoint blocks changed")


@torch.no_grad()
@_parametrize
def test_checkpoint_inside_the_last_chunk_compiles_nothing_after_warmup(
    mesh_device, kv_bf8, monkeypatch, reset_seeds, ensure_gc
):
    """After the masked-bucket warmup (which runs one checkpointed forward per bucket) and GdnBlockState.warmup (which
    writes that checkpoint into the null block), prefills with a checkpoint inside their last chunk, at other
    checkpoint positions and into blocks of both parities, add no program cache entry."""
    run = Run(mesh_device, monkeypatch, kv_bf8)
    bs = run.bs
    if bs >= run.model.prefill_chunk_size:
        pytest.skip(f"{bs}-token KV blocks put every checkpoint on a prefill chunk boundary")
    run.model.warmup_prefill_masked_buckets(torch.arange(PAGE_COLUMNS, dtype=torch.int32).reshape(1, -1))
    run.model.gdn_blocks.warmup()
    ttnn.synchronize_device(mesh_device)
    before = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for T, parity in ((2000, 0), (1500, 1), (2048, 1)):
            p = (T - 1) // bs * bs
            prompt = _prompts(1, T, seed=71 + T)
            row = _row(prompt, 0, T, run.blocks(-(-T // bs)), run.blocks(3, parity), checkpoint=(p, run.blocks(3)))
            run.prefill([row])
        ttnn.synchronize_device(mesh_device)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    after = mesh_device.num_program_cache_entries()
    assert after == before, f"{after - before} program(s) compiled after warmup"
