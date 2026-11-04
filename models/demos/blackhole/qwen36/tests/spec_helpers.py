# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the device spec-decode tests (not a test module): model build, fresh KV, page tables,
prompt pool, plain-greedy reference, near-tie comparison, memory snapshot and the plugin-runner mirror."""

import math

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE, _get_prompt
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _N_LAYERS, NEAR_TIE_GAP, _top2_gap
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAD = -1


def _build(device, num_blocks):
    from transformers import AutoTokenizer

    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        device, max_batch_size=1, max_seq_len=num_blocks * BLOCK_SIZE, n_layers=_N_LAYERS
    )
    assert model.mtp is not None, "MTP head not built"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    kv_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    # Verify runs the fused GDN op; plain decode must too (see test_spec_lossless.py).
    model.set_gdn_fused_decode(True)
    return model, tokenizer, kv_shape


def _fresh_kv(model, kv_shape):
    model.free_kv_caches()  # also shuts down any engine
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)


def _prompt_pool(tokenizer, n):
    """>= n real-text tokens: the source repeated (newline-separated) when it is shorter than n."""
    src = _get_prompt(n, tokenizer)[0].tolist()
    sep = tokenizer("\n", add_special_tokens=False)["input_ids"]
    pool = list(src)
    while len(pool) < n:
        pool += sep + src
    return pool[:n]


def _perm_table(num_blocks, seed, blocks=None):
    """[1, width] int32 table: a seeded permutation of `blocks` (default all of 0..num_blocks-1)."""
    ids = torch.arange(num_blocks) if blocks is None else torch.as_tensor(blocks)
    g = torch.Generator().manual_seed(seed)
    return ids[torch.randperm(len(ids), generator=g)].to(torch.int32).reshape(1, -1)


def _plain_greedy(model, kv_shape, prompt_ids, pt, n):
    """Free-running plain greedy (prefill + n-1 paged decodes) on fresh KV: (argmaxes, top-2 gaps)."""
    _fresh_kv(model, kv_shape)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.int32)
    logits_dev = model.prefill_for_spec(prompt, pt, len(prompt_ids), lambda hidden, chunk_start, valid_len: None)
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    ttnn.deallocate(logits_dev)
    row = lt.reshape(-1)[: model.vocab_size].float()  # logits are replicated; device 0's row
    toks, gaps = [int(row.argmax())], [_top2_gap(row)]
    pos = len(prompt_ids)
    for _ in range(1, n):
        row, hidden = model.decode_step_paged(toks[-1], pos, pt)
        ttnn.deallocate(hidden)
        toks.append(int(row.argmax()))
        gaps.append(_top2_gap(row))
        pos += 1
    model.free_kv_caches()
    return toks, gaps


def _assert_matches_ref(name, out, ref, gaps):
    """Equal to the plain reference up to the first mismatch, which must be a near-tie."""
    assert len(out) <= len(ref), f"{name}: {len(out)} tokens > reference length {len(ref)}"
    for i, t in enumerate(out):
        if t != ref[i]:
            assert (
                gaps[i] < NEAR_TIE_GAP
            ), f"{name}: position {i} spec {t} vs plain {ref[i]} with CONFIDENT plain gap {gaps[i]:.4f}"
            logger.info(f"[reuse] {name}: near-tie flip at {i} (gap {gaps[i]:.4f}); later tokens not compared")
            return
    logger.info(f"[reuse] {name}: {len(out)} tokens identical to plain greedy")


def _mem_state(mesh_device):
    return (
        ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank,
        ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank,
        len(ttnn._ttnn.reports.get_buffers(mesh_device)),
    )


class FakeRunner:
    """Host-side B=1 mirror of the plugin runner's use of the engine."""

    def __init__(self, eng, num_blocks, nb, seed):
        self.eng, self.K, self.nb = eng, eng.K, nb
        self.bs = BLOCK_SIZE
        # Seeded shuffle of every real block id; the MTP scratch block (== num_blocks) is never handed out.
        g = torch.Generator().manual_seed(seed)
        self.free = torch.randperm(num_blocks, generator=g).tolist()
        self.blocks = []

    def _table(self, last_pos):
        """[1, nb] table covering positions up to last_pos + (1+K) + (K+1) lookahead; unused entries 0."""
        need = min(self.nb, math.ceil((last_pos + (1 + self.K) + (self.K + 1)) / self.bs))
        while len(self.blocks) < need:
            self.blocks.append(self.free.pop())
        row = self.blocks + [0] * (self.nb - len(self.blocks))
        return torch.tensor([row], dtype=torch.int32)

    def run(self, prompt, max_new, n_policy):
        eng, K = self.eng, self.K
        cap = self.nb * self.bs
        P = len(prompt)
        logits = eng.prefill(0, prompt, self._table(P - 1))
        out = [int(logits.argmax())]
        stats = {"steps": [], "stop": "max_new" if len(out) >= max_new else None}
        proposal = torch.empty(0, dtype=torch.int32)  # empty after prefill
        acc = 1
        step = 0
        while len(out) < max_new:
            pos0 = P + len(out) - 1  # position of the last committed token
            if pos0 + 1 + K >= cap:  # next verify window would not fit the table
                stats["stop"] = "capacity"
                break
            n = min(len(proposal), n_policy(step))
            drafts = [int(t) for t in proposal[:n]]
            tokens = torch.full((1, 1 + K), PAD, dtype=torch.int32)
            start = torch.full((1, 1 + K), PAD, dtype=torch.int32)
            tokens[0, : 1 + n] = torch.tensor([out[-1]] + drafts, dtype=torch.int32)
            start[0, : 1 + n] = torch.arange(pos0, pos0 + 1 + n, dtype=torch.int32)
            pt = self._table(pos0)
            res = eng.decode_forward(
                tokens,
                start,
                num_valid_drafts=torch.tensor([n], dtype=torch.int32),
                accepted_counts=torch.tensor([acc], dtype=torch.int32),
                page_table=pt,
                spec_mode="argmax_ids",
            )
            ids = res.argmax_ids[0].tolist()
            assert all(i == PAD for i in ids[n + 1 :]), f"step {step}: ids past n={n} not -1: {ids}"
            assert all(i != PAD for i in ids[: n + 1]), f"step {step}: -1 inside valid ids {ids} (n={n})"
            m = 0
            while m < n and drafts[m] == ids[m]:
                m += 1
            new = drafts[:m] + [ids[m]]
            new = new[: max_new - len(out)]
            out.extend(new)
            acc = len(new)
            committed = torch.full((1, 1 + K), PAD, dtype=torch.int32)
            cpos = torch.full((1, 1 + K), PAD, dtype=torch.int32)
            committed[0, :acc] = torch.tensor(new, dtype=torch.int32)
            cpos[0, :acc] = torch.arange(pos0 + 1, pos0 + 1 + acc, dtype=torch.int32)
            d = eng.propose_draft_tokens(
                K, committed, cpos, torch.tensor([acc], dtype=torch.int32), self._table(pos0 + acc)
            )
            assert tuple(d.draft_token_ids.shape) == (1, K)
            nv = int(d.num_valid[0])
            assert 0 <= nv <= K
            proposal = d.draft_token_ids[0, :nv].clone()
            stats["steps"].append({"n": n, "m": m, "num_valid": nv})
            step += 1
        if stats["stop"] is None:
            stats["stop"] = "max_new"
        eng.release(0)
        return out, stats
