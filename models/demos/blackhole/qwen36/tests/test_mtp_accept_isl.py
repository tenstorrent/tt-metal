# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance-vs-prompt-length sweep for MTP speculative decode (scratch investigation).

Question: the demo shows accept 2.77/3 at ISL 128 but 2.16/3 at ISL 4k. Is that a bug in the
MULTI-CHUNK MTP warm (prompts > 2048 take more than one `_warm_mtp_chunk` call) or a genuine
length effect?

Isolation: hold CONTENT STRUCTURE fixed and vary only the length.
  * "rep128": the shared 128-token prompt repeated and clipped to the target length. Identical
    structure at every length, so any acceptance step at the 2048 chunk boundary is the warm path,
    not the text.
  * "long4k": prefixes of the demo's real 4k prompt file (same text, different truncation) — a
    realistic-content control for the same boundary.

A CLIFF between 2048 (one chunk) and 2176/2304 (two chunks) => multi-chunk warm bug.
A smooth decline with no boundary feature => genuine length/content effect.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_mtp_accept_isl.py -v -s
Override the sweep with QWEN36_ISL_SWEEP="rep128:512,rep128:2048,long4k:3968". An entry may carry a
per-case draft length as a third field, "src:plen:K" (e.g. "frank:3968:6"); without it the decoder's
default K applies. QWEN36_ISL_NUM_BLOCKS overrides NUM_BLOCKS (the KV / max_seq_len budget).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

MAX_NEW = 128
# 96 blocks x 64 = 6144 tokens: covers the 4096-row prefill bucket a 3968-token prompt uses (the
# MTP warm writes the whole tail bucket) plus MAX_NEW of decode. Matches the multiple-of-32 the
# demo's spec path rounds to. QWEN36_ISL_NUM_BLOCKS raises it for longer sweep entries.
NUM_BLOCKS = int(os.environ.get("QWEN36_ISL_NUM_BLOCKS", 96))

# long4k is the demo's OWN "ISL 4k" prompt file, which is only 2642 tokens — so the demo's 2.16/3
# "at 4k" was measured at 2642 tokens, i.e. exactly two warm chunks. Sweep it up to its own length
# (no repeat) so the 2048 boundary is crossed on the demo's real text.
_DEFAULT_SWEEP = (
    [("rep128", n) for n in (128, 512, 1024, 1536, 2048, 2176, 2304, 2560, 3072, 3968)]
    + [("long4k", n) for n in (128, 1024, 2048, 2176, 2560, 2642)]
    + [("frank", n) for n in (1024, 2048, 2176, 3072, 3968)]
)


def _sweep():
    """-> [(src, plen, draft_len_or_None)]. QWEN36_ISL_SWEEP entries are "src:plen" or "src:plen:K"."""
    spec = os.environ.get("QWEN36_ISL_SWEEP")
    if not spec:
        return [(src, n, None) for src, n in _DEFAULT_SWEEP]
    out = []
    for item in spec.split(","):
        parts = [s.strip() for s in item.split(":")]
        assert 2 <= len(parts) <= 3, f"bad QWEN36_ISL_SWEEP entry {item!r}: want src:plen or src:plen:K"
        out.append((parts[0], int(parts[1]), int(parts[2]) if len(parts) == 3 else None))
    return out


def _repeat_clip(ids, target):
    while ids.shape[1] < target:
        ids = torch.cat([ids, ids], dim=1)
    ids = ids[:, :target]
    assert ids.shape[1] == target, f"prompt is {ids.shape[1]} tokens, wanted exactly {target}"
    return ids


# --------------------------------------------------------------------------------------------- #
# Positive control: is the per-chunk MTP warm actually doing anything at chunk_start > 0?
#
# A no-op / mis-addressed warm for chunks after the first is indistinguishable from a working one
# by acceptance alone unless you can turn it off. `_warm_mtp_chunk` is wrapped (never edited) so a
# given chunk's warm can be skipped; if skipping the TAIL warm leaves acceptance unchanged, the
# tail warm was writing nothing useful (bug). If acceptance collapses, it is load-bearing.
# --------------------------------------------------------------------------------------------- #
_ABLATIONS = ("baseline", "skip_chunk0", "skip_tail", "skip_all")


class _WarmGate:
    """Context manager that skips selected `_warm_mtp_chunk` calls on the SpeculativeDecoder class."""

    def __init__(self, cls, mode):
        self.cls, self.mode, self.orig = cls, mode, cls._warm_mtp_chunk
        self.skipped, self.ran = [], []

    def __enter__(self):
        orig, mode = self.orig, self.mode
        skipped, ran = self.skipped, self.ran

        def gated(dec, hidden, chunk_start, valid_len, prompt_ids):
            skip = (
                mode == "skip_all"
                or (mode == "skip_chunk0" and chunk_start == 0)
                or (mode == "skip_tail" and chunk_start > 0)
            )
            (skipped if skip else ran).append(chunk_start)
            if skip:
                return None
            return orig(dec, hidden, chunk_start, valid_len, prompt_ids)

        self.cls._warm_mtp_chunk = gated
        return self

    def __exit__(self, *exc):
        self.cls._warm_mtp_chunk = self.orig
        return False


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_mtp_accept_vs_isl(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    device = mesh_device
    device.enable_program_cache()
    max_seq_len = NUM_BLOCKS * BLOCK_SIZE
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len)
    assert model.mtp is not None, "MTP head not built"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    # Prompt sources, built once so every length is a prefix/repeat of the SAME token stream.
    base128 = _get_prompt(128, tokenizer)  # exactly 128 ids, the shared question prompt
    long4k = _get_prompt(4096, tokenizer)  # demo's real "ISL 4k" prompt file (2642 tokens)
    logger.info(f"[isl] base128={base128.shape[1]} tokens, long4k={long4k.shape[1]} tokens")
    sources = {"rep128": base128, "long4k": long4k}

    def _build(src, plen):
        # "frank": the demo's long-context corpus rebuilt AT each length (prefix + context prefix +
        # the same instruction suffix), so structure is identical and only the context length moves.
        if src == "frank":
            ids = _get_prompt(8192, tokenizer, max_prompt_len=plen)
            assert ids.shape[1] == plen, f"frank prompt is {ids.shape[1]} tokens, wanted {plen}"
            return ids
        return _repeat_clip(sources[src], plen)

    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)

    rows = []
    for src, plen, k in _sweep():
        assert plen + MAX_NEW < max_seq_len, f"{plen}+{MAX_NEW} exceeds max_seq_len {max_seq_len}"
        prompt_ids = _build(src, plen)[0].tolist()
        model.free_kv_caches()
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        dec = SpeculativeDecoder(model, pt, draft_len=k)  # k=None -> the decoder's default K
        gen = dec.generate(prompt_ids, MAX_NEW)
        s = dec.stats()
        text = tokenizer.decode(gen, skip_special_tokens=True)
        chunks = 1 if plen <= 2048 else (plen // 2048 + (1 if plen % 2048 else 0))
        rows.append(
            dict(
                src=src,
                plen=plen,
                chunks=chunks,
                accept=s["accept_rate"],
                cond=s["conditional"],
                hist=s["hist"],
                zero=s["zero_accept_rate"],
                iters=s["iters"],
                ttft=dec.prefill_time,
                text=text[:80],
            )
        )
        logger.info(
            f"[isl] src={src} plen={plen} chunks={chunks} accept={s['accept_rate']:.3f}/{dec.K} "
            f"cond={[f'{x:.2f}' for x in s['conditional']]} hist={s['hist']} "
            f"zero={s['zero_accept_rate']:.1%} iters={s['iters']} ttft={dec.prefill_time:.2f}s"
        )
        logger.info(f"[isl] src={src} plen={plen} OUT: {text[:80]!r}")
        model.free_kv_caches()

    logger.info("[isl] ==================== SUMMARY ====================")
    logger.info(f"[isl] {'src':8s} {'plen':>5s} {'chk':>3s} {'accept':>7s} {'zero':>6s} {'ttft':>6s}  out")
    for r in rows:
        logger.info(
            f"[isl] {r['src']:8s} {r['plen']:5d} {r['chunks']:3d} {r['accept']:7.3f} "
            f"{r['zero']:6.1%} {r['ttft']:6.2f}  {r['text']!r}"
        )


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_mtp_warm_chunk_ablation(mesh_device):
    """Turn individual MTP warm chunks off and watch acceptance, on MULTI-CHUNK prompts.

    Expected if the multi-chunk warm is healthy: skipping the TAIL warm (the slots nearest the
    frontier, which the first drafts attend most) costs a lot; skipping only chunk 0 costs little.
    A tail skip that costs NOTHING would mean the chunk>0 warm was never writing usable KV.
    """
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    device = mesh_device
    device.enable_program_cache()
    max_seq_len = NUM_BLOCKS * BLOCK_SIZE
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    long4k = _get_prompt(4096, tokenizer)
    base128 = _get_prompt(128, tokenizer)
    # The prompt must be NON-repetitive (rep128 stores the same content in the first chunk and the
    # tail, so dropping either loses nothing) and NOT at the 3.000 ceiling (long4k@2560 is), or the
    # ablation cannot see a difference even when the warm is working.
    cases = [("long4k", 2642), ("frank", 3968), ("rep128", 3968)]

    def _build(src, plen):
        if src == "frank":
            return _get_prompt(8192, tokenizer, max_prompt_len=plen)
        return _repeat_clip({"long4k": long4k, "rep128": base128}[src], plen)

    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)

    rows = []
    for src, plen in cases:
        prompt_ids = _build(src, plen)[0].tolist()
        for mode in _ABLATIONS:
            model.free_kv_caches()
            model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
            with _WarmGate(SpeculativeDecoder, mode) as gate:
                dec = SpeculativeDecoder(model, pt)
                gen = dec.generate(prompt_ids, MAX_NEW)
            s = dec.stats()
            text = tokenizer.decode(gen, skip_special_tokens=True)
            rows.append((src, plen, mode, s["accept_rate"], gate.ran, gate.skipped, text[:60]))
            logger.info(
                f"[ablate] src={src} plen={plen} mode={mode:11s} accept={s['accept_rate']:.3f}/{dec.K} "
                f"warmed={gate.ran} skipped={gate.skipped} zero={s['zero_accept_rate']:.1%}"
            )
            logger.info(f"[ablate] src={src} plen={plen} mode={mode} OUT: {text[:60]!r}")
            model.free_kv_caches()

    logger.info("[ablate] ==================== SUMMARY ====================")
    for src, plen, mode, acc, ran, skipped, text in rows:
        logger.info(f"[ablate] {src:8s} {plen:5d} {mode:11s} accept={acc:6.3f}  warmed={ran} skipped={skipped}")


class _StopAfterWarm(Exception):
    """Raised from a stubbed `_warm_mtp_last` to freeze generate() right after the chunk warms."""


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_mtp_warm_kv_map(mesh_device):
    """Where does each MTP warm chunk's KV actually land?

    Freeze generate() after the last `_warm_mtp_chunk` (stub `_warm_mtp_last` to raise) and read the
    drafter's paged K cache block by block. A healthy multi-chunk warm leaves every block up to
    (T-2)//BLOCK_SIZE non-zero. Blocks that stay at the allocation's zeros mark slots the warm never
    wrote — and blocks that are non-zero past the prompt mark the bucket-padding junk.
    The base model's own K cache is printed alongside as the reference profile.
    """
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    long4k = _get_prompt(4096, tokenizer)
    base128 = _get_prompt(128, tokenizer)

    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)

    def _block_profile(cache):
        t = ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).float()  # [blocks, kvh, 64, hd]
        return t.abs().mean(dim=(1, 2, 3))

    def _runs(mask):
        """Compact run-length view of a per-block boolean (True = non-zero)."""
        out, i = [], 0
        while i < len(mask):
            j = i
            while j < len(mask) and mask[j] == mask[i]:
                j += 1
            out.append(f"{'NZ' if mask[i] else 'ZERO'}[{i}:{j}]")
            i = j
        return " ".join(out)

    for src, plen, ids in (("long4k", 2560, long4k), ("rep128", 3968, base128), ("rep128", 4224, base128)):
        prompt_ids = _repeat_clip(ids, plen)[0].tolist()
        model.free_kv_caches()
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
        orig = SpeculativeDecoder._warm_mtp_last

        def _stop(self, *a, **kw):
            raise _StopAfterWarm

        SpeculativeDecoder._warm_mtp_last = _stop
        try:
            SpeculativeDecoder(model, pt).generate(prompt_ids, 4)
            raise AssertionError("generate() never reached _warm_mtp_last")
        except _StopAfterWarm:
            pass
        finally:
            SpeculativeDecoder._warm_mtp_last = orig

        mtp_p = _block_profile(model._mtp_kv_cache[0])
        base_p = _block_profile(model._paged_kv_caches[0][0])
        last_real_block = (plen - 2) // BLOCK_SIZE
        logger.info(f"[kvmap] --- src={src} plen={plen} last real MTP block={last_real_block} ---")
        logger.info(f"[kvmap] MTP  k blocks : {_runs([float(v) > 0 for v in mtp_p])}")
        logger.info(f"[kvmap] BASE k blocks : {_runs([float(v) > 0 for v in base_p])}")
        logger.info(
            f"[kvmap] MTP  |k| per block (first 4 / around 32 / last real): "
            f"{[f'{float(mtp_p[i]):.4f}' for i in (0, 1, 2, 3)]} "
            f"{[f'{float(mtp_p[i]):.4f}' for i in (30, 31, 32, 33, 34)]} "
            f"{[f'{float(mtp_p[i]):.4f}' for i in (last_real_block - 1, last_real_block)]}"
        )
        logger.info(
            f"[kvmap] BASE |k| per block (same picks): "
            f"{[f'{float(base_p[i]):.4f}' for i in (0, 1, 2, 3)]} "
            f"{[f'{float(base_p[i]):.4f}' for i in (30, 31, 32, 33, 34)]} "
            f"{[f'{float(base_p[i]):.4f}' for i in (last_real_block - 1, last_real_block)]}"
        )
        model.free_kv_caches()
