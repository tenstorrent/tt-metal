# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of MTP chunked prefill (QWEN36_SPEC_CHUNKED_PREFILL=1, work/notes/84). NOT run by the host suite.
Parametrized by QWEN36_SPEC_CHUNK_EXACT_RESUME: "1" = Option B (snapshots at the chunk ends, no e-128 / no recompute, drafter
slot e-1 warmed by the one-row warm at the start of the next chunk), "0" = Option A (snapshots at e-128).

One ~10k-token prompt is prefilled (a) in chunks like the vLLM plugin does it (intermediate_prefill_mask=[True] for every
chunk but the last, prompt_lens = chunk end e, start_pos = previous end; e.g. 4096 / 4096 / final for chunk 4096) on its own
blocks, and (b) in ONE shot on fresh blocks with the prefix cache detached (reference). Compared, with the harness of
test_spec_prefix_cache.py (Harness.compare): last-token logits (PCC + top-1 = first token), eng._last_hidden[slot],
GDN rec_state of the decode slot, main-model attention KV and drafter (MTP) KV through each request's page table.
Also checks that a non-final chunk registers no fresh row, returns a zero row and leaves the chunk-end snapshot (Option B;
no e-128 snapshot) or the e-128 snapshot (Option A) behind. Expected: logits / GDN / target KV bit-identical (PCC 1.0). Drafter
KV: every slot except e-1 of each chunk end is bit-identical; slot e-1 is written by the one-row drafter decode
(_warm_mtp_last) instead of the batched prefill warm (different matmul / rope programs, bf16): expect PCC >= 0.999 there
(checked separately), not bit equality.

    MESH_DEVICE=P150x4 HF_MODEL=... QWEN36_SPEC_CHUNKED_PREFILL=1 QWEN36_SPEC_PREFIX_CACHE=1 QWEN36_PREFIX_CACHE_DEBUG=1 \
        pytest models/demos/blackhole/qwen36/tests/test_spec_chunked_prefill.py -svq
Run it with QWEN36_SPEC_FAST_PREFILL=1 (traced segments) and with it unset (eager resumable path). Stop any server first:
one device user at a time.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.spec_helpers import _prompt_pool
from models.demos.blackhole.qwen36.tests.test_spec_prefix_cache import BLOCK, Harness
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM

MAX_SEQ = 12288
BPU = MAX_SEQ // BLOCK
NUM_BLOCKS = 640
WIDTH = 2  # slot 0 = chunked request, slot 1 = reference
PROMPT_LEN = 10000  # not a multiple of 128: the final chunk has a ragged tail


class ChunkHarness(Harness):
    def prefill_chunk(self, toks, blocks, start_pos, slot, final):
        """One prefill_forward call of the plugin's chunked flow: tokens[:e], start_pos, intermediate_prefill_mask."""
        e = len(toks)
        pt = torch.zeros(1, BPU, dtype=torch.int32)
        pt[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        logits, _ = self.gen.prefill_forward(
            torch.tensor(toks, dtype=torch.int32).reshape(1, e),
            pt,
            self.kv,
            [e],
            start_pos=[start_pos],
            empty_slots=[slot],
            enable_trace=True,
            intermediate_prefill_mask=[not final],
        )
        return logits.reshape(-1).float()

    def collect(self, toks, blocks, slot, logits, skip=()):
        """skip: drafter slots excluded from ``mtp_kv`` and reported in ``mtp_kv_skip`` (the one-row-warmed slots e-1)."""
        L = len(toks)
        mtp = self._gather(blocks, L - 1, main=False)
        keep = torch.tensor([i for i in range(L - 1) if i not in set(skip)], dtype=torch.long)
        sk = torch.tensor(sorted(skip), dtype=torch.long)
        comp = ttnn.ConcatMeshToTensor(self.model.mesh_device, dim=-1)
        from models.demos.blackhole.qwen36.tests.test_spec_prefix_cache import _dev0

        return dict(
            L=L,
            blocks=blocks,
            logits=logits,
            hidden=ttnn.to_torch(self.eng._last_hidden[slot], mesh_composer=comp).float(),
            rec=[_dev0(dn.rec_state)[slot] for dn in self.gdn],
            main_kv=self._gather(blocks, L, main=True),
            mtp_kv=[t[:, keep] for t in mtp],
            mtp_kv_skip=[t[:, sk] for t in mtp] if len(sk) else None,
        )


@run_for_blackhole()
@pytest.mark.timeout(3500)
@pytest.mark.parametrize("exact", ["1", "0"])
@pytest.mark.parametrize("chunk", [4096, 2048])
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_chunked_prefill_state(chunk, exact, mesh_device, reset_seeds, ensure_gc, monkeypatch):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    monkeypatch.setenv("QWEN36_SPEC_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("QWEN36_SPEC_PREFIX_CACHE", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHUNK_EXACT_RESUME", exact)
    # Harness.prefill (the one-shot reference) builds its page table with test_spec_prefix_cache's BPU (8k tokens):
    # widen it to this test's MAX_SEQ so the 10k reference fits.
    import models.demos.blackhole.qwen36.tests.test_spec_prefix_cache as spc

    monkeypatch.setattr(spc, "BPU", BPU)
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
    h = ChunkHarness(cls, kv, _prompt_pool(tok, 16384))
    eng = h.eng
    failures = []
    toks = h.text(PROMPT_LEN)
    blocks = h.blocks_for(PROMPT_LEN)

    # --- chunked: non-final chunks to [0,c), [c,2c), ..., final chunk to the end
    ends = list(range(chunk, PROMPT_LEN, chunk)) + [PROMPT_LEN]
    start, logits = 0, None
    for i, e in enumerate(ends):
        final = i == len(ends) - 1
        out = h.prefill_chunk(toks[:e], blocks, start, 0, final)
        if not final:
            assert not bool(out.any()), f"chunk end {e}: non-final chunk must return a zero row"
            assert not eng._fresh and not eng._active, f"chunk end {e}: non-final chunk registered a fresh row"
            keys = sorted(cache._positions)
            if exact == "1":
                assert e in keys, f"chunk end {e}: no GDN snapshot at the chunk end (have {keys})"
                assert (e - 128) not in keys, f"chunk end {e}: Option B must not snapshot e-128 (have {keys})"
                assert eng._chunk_prog()[0][0] == e, f"chunk end {e}: no chunk-resume progress for the row"
            else:
                assert (e - 128) in keys, f"chunk end {e}: no GDN snapshot at e-128 (have {keys})"
        else:
            logits = out
            assert eng._fresh == {0: PROMPT_LEN}, eng._fresh
        start = e
    skip = [e - 1 for e in ends[:-1]] if exact == "1" else []
    chunked = h.collect(toks, blocks, 0, logits, skip=skip)
    logger.info(f"chunk={chunk} ends={ends} snapshots={sorted(cache._positions)} stats={cache.stats}")
    assert getattr(eng, "resume_fallbacks", 0) == 0, "a chunk resumed below its snapshot (eviction?)"
    if exact == "1":
        assert getattr(eng, "chunk_exact_resumes", 0) == len(ends) - 1, "a continuation chunk did not resume exactly"
        assert getattr(eng, "chunk_exact_misses", 0) == 0
        assert eng._chunk_prog() == {}, "the final chunk must consume the progress entry"

    # --- reference: one shot, fresh blocks, prefix cache detached
    model._prefix_cache = None
    try:
        ref = h.prefill(toks, h.blocks_for(PROMPT_LEN), 0, 1)
    finally:
        model._prefix_cache = cache
    if skip:  # same exclusion as the chunked side: split the reference drafter KV into kept slots / the slots e-1
        full = ref["mtp_kv"]
        keep = torch.tensor([i for i in range(PROMPT_LEN - 1) if i not in set(skip)], dtype=torch.long)
        sk = torch.tensor(sorted(skip), dtype=torch.long)
        ref["mtp_kv"], ref["mtp_kv_skip"] = [t[:, keep] for t in full], [t[:, sk] for t in full]
    h.compare(f"chunk{chunk}-exact{exact}", chunked, ref, failures)
    if skip:
        from models.demos.blackhole.qwen36.tests.test_spec_prefix_cache import pcc

        p = min(pcc(a, b) for a, b in zip(chunked["mtp_kv_skip"], ref["mtp_kv_skip"]))
        logger.info(f"RESULT drafter KV slots e-1 {skip} (one-row warm vs batched warm): min PCC {p}")
        if not p >= 0.999:
            failures.append(f"drafter KV slots {skip}: pcc={p:.5f} < 0.999")
    assert not failures, "; ".join(failures)
