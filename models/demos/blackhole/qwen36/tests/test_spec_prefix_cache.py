# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of prefix caching on the MTP spec class (Qwen36MTPContractForCausalLM), driven like the vLLM plugin.

Clone of tests/test_prefix_cache.py for the spec path: a (1, N) mesh, max_batch_size=2 (the batched eager spec prefill is
the only path that resumes from a GDN snapshot), hand-built page tables that mimic vLLM APC (block size 64, block 0 =
null block, cached prefix blocks SHARED between requests, start_pos = cached tokens).

Each hit prefill (request B on blocks shared with request A, start_pos > 0) is compared with a REFERENCE full prefill of
the identical tokens on fresh blocks (model._prefix_cache detached, start_pos 0):
  * last-token logits: PCC + top-1,   * eng._last_hidden[slot] (MTP hand-off): PCC,
  * GDN rec_state of the decode slot (every GDN layer): PCC,
  * main-model attention KV of [0, L) and drafter (MTP) KV of [0, L-1) through each request's page table: PCC.
Scenarios cover divergence exactly on a 128 boundary (resume must step one 128-block back: MTP slot s-1 fuses a stale
token otherwise), divergence mid-block, an exact repeat, and no snapshot (full recompute).

    MESH_DEVICE=P150x4 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_spec_prefix_cache.py -svq
Requires QWEN36_SPEC_PREFIX_CACHE=1 (default) and QWEN36_PREFIX_CACHE_DEBUG=1 to log hits.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.spec_helpers import _prompt_pool
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM

BLOCK = 64
MAX_SEQ = 8192
BPU = math.ceil(MAX_SEQ / BLOCK)
NUM_BLOCKS = 640
WIDTH = 2  # max_num_seqs: slot 0 = request under test, slot 1 = reference
PCC_MIN = 0.99


def pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


class Harness:
    def __init__(self, gen, kv, corpus):
        self.gen, self.kv, self.corpus = gen, kv, corpus
        self.model = gen.model[0]
        self.eng = gen._engine()
        self.next_block, self.cursor = 1, 0
        self.gdn = [layer.attention for layer in self.model.layers if not layer.is_full_attention]

    def text(self, n):
        out = self.corpus[self.cursor : self.cursor + n]
        self.cursor += n
        assert len(out) == n, "corpus too short"
        return out

    def blocks_for(self, L, shared=()):
        need = (L + BLOCK) // BLOCK + 1
        ids = list(range(self.next_block, self.next_block + need - len(shared)))
        self.next_block += need - len(shared)
        assert self.next_block < NUM_BLOCKS - 1, "out of KV blocks"
        return list(shared) + ids

    def prefill(self, toks, blocks, start_pos, slot):
        """-> dict of everything the spec prefill leaves behind for this request."""
        L = len(toks)
        pt = torch.zeros(1, BPU, dtype=torch.int32)
        pt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        logits, _ = self.gen.prefill_forward(
            torch.tensor(toks, dtype=torch.int32).reshape(1, L),
            pt,
            self.kv,
            [L],
            start_pos=[start_pos],
            empty_slots=[slot],
            enable_trace=True,
        )
        comp = ttnn.ConcatMeshToTensor(self.model.mesh_device, dim=-1)
        return dict(
            L=L,
            blocks=blocks,
            logits=logits.reshape(-1).float(),
            hidden=ttnn.to_torch(self.eng._last_hidden[slot], mesh_composer=comp).float(),
            rec=[_dev0(dn.rec_state)[slot] for dn in self.gdn],
            main_kv=self._gather(blocks, L, main=True),
            mtp_kv=self._gather(blocks, L - 1, main=False),
        )

    def _gather(self, blocks, n, main):
        idx = torch.tensor(blocks, dtype=torch.long)
        tensors = []
        if main:
            tensors = [self.kv[0][0], self.kv[0][1]]  # first attention layer is enough to catch block mix-ups
        else:
            att = self.model.mtp.attention
            tensors = [att.paged_k, att.paged_v]
        out = []
        for t in tensors:
            x = _dev0(t)[idx]  # [nb, h, 64, d]
            out.append(x.permute(1, 0, 2, 3).reshape(x.shape[1], -1, x.shape[-1])[:, :n])
        return out

    def compare(self, tag, hit, ref, failures, s_expected=None):
        res = {
            "logits": pcc(hit["logits"], ref["logits"]),
            "hidden": pcc(hit["hidden"], ref["hidden"]),
            "rec_min": min(pcc(a, b) for a, b in zip(hit["rec"], ref["rec"])),
            "main_kv": min(pcc(a, b) for a, b in zip(hit["main_kv"], ref["main_kv"])),
            "mtp_kv": min(pcc(a, b) for a, b in zip(hit["mtp_kv"], ref["mtp_kv"])),
        }
        top1 = int(hit["logits"].argmax()) == int(ref["logits"].argmax())
        logger.info(f"RESULT {tag}: {res} top1={top1}")
        for k, v in res.items():
            if not v >= PCC_MIN:
                failures.append(f"{tag}: {k} pcc={v:.5f}")
        if not top1:
            failures.append(f"{tag}: top-1 differs")


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_prefix_cache_state(mesh_device, reset_seeds, ensure_gc):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    mesh_device.enable_program_cache()
    cls = Qwen36MTPContractForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=os.environ.get("HF_MODEL")),
        mesh_device,
        max_batch_size=WIDTH,
        max_seq_len=MAX_SEQ,
        tt_data_parallel=1,
        optimizations=None,
    )
    model = cls.model[0]
    tc = model.args.hf_config.get_text_config()
    n_kv = tc.num_key_value_heads
    kv_shape = (NUM_BLOCKS, n_kv // min(model.num_devices, n_kv), BLOCK, model.args.head_dim)
    kv = cls.allocate_kv_cache(kv_shape, ttnn.bfloat16, len(model._attention_layer_indices))
    cls.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    cls.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=False
    )
    cls.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    cls.warmup_model_decode(
        enable_trace=True, kv_cache=kv, max_batch_size=WIDTH, num_blocks=BPU, can_sample_on_device=False
    )
    cache = model._prefix_cache
    assert cache is not None, "spec warmup did not build the GdnPrefixStateCache (QWEN36_SPEC_PREFIX_CACHE=1?)"

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    h = Harness(cls, kv, _prompt_pool(tok, 30000))
    failures = []

    def reference(toks):
        model._prefix_cache = None  # plain full prefill, fresh blocks
        try:
            return h.prefill(toks, h.blocks_for(len(toks)), 0, 1)
        finally:
            model._prefix_cache = cache

    def scenario(tag, a_toks, a_blocks, b_toks, shared_blocks, start_pos, expect_hit):
        hits0, skipped0 = cache.stats["hits"], cache.stats["tokens_skipped"]
        hit = h.prefill(b_toks, h.blocks_for(len(b_toks), shared_blocks), start_pos, 0)
        got_hit = cache.stats["hits"] > hits0
        resumed = cache.stats["tokens_skipped"] - skipped0
        logger.info(f"{tag}: start_pos={start_pos} hit={got_hit} resumed_at={resumed}")
        if got_hit != expect_hit:
            failures.append(f"{tag}: expected hit={expect_hit}, got {got_hit}")
        if got_hit:
            cap = ((start_pos - 1) // 128) * 128
            assert resumed <= cap, f"{tag}: resumed at {resumed} > MTP-safe cap {cap}"
        h.compare(tag, hit, reference(b_toks), failures)

    shared = h.text(4352)  # 68 blocks = 34 x 128
    A = shared + h.text(700)
    a_blocks = h.blocks_for(len(A))
    h.prefill(A, a_blocks, 0, 0)  # request A: saves the last full-chunk snapshot (4096)
    logger.info(f"snapshots after A: {sorted(cache._positions)}")

    # divergence exactly on a 128 boundary (4352 = 34 * 128): resume must be <= 4224, here the 4096 snapshot
    B = shared + h.text(600)
    scenario("div@128-boundary", A, a_blocks, B, a_blocks[:68], 4352, True)
    # divergence mid 128-block but on a 64 block boundary (4288 = 67 * 64)
    C = shared[:4288] + h.text(900)
    scenario("div@64-block", A, a_blocks, C, a_blocks[:67], 4288, True)
    # prefix shorter than the first snapshot: no snapshot <= cap -> full recompute (hit = False), still correct
    D = shared[:1024] + h.text(1500)
    scenario("no-snapshot", A, a_blocks, D, a_blocks[:16], 1024, False)
    # exact repeat of A: vLLM hits all full blocks but the last token
    st = ((len(A) - 1) // BLOCK) * BLOCK
    scenario("repeat", A, a_blocks, A, a_blocks[: st // BLOCK], st, True)
    # second request with the same start: the learn snapshot taken by the first one at <= cap now exists
    E = shared[:2560] + h.text(1200)
    scenario("learn-1st", A, a_blocks, E, a_blocks[:40], 2560, False)
    scenario("learn-2nd", A, a_blocks, shared[:2560] + h.text(900), a_blocks[:40], 2560, True)
    logger.info(f"cache stats: {cache.stats}")
    assert not failures, "; ".join(failures)
