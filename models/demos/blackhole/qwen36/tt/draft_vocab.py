# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_DRAFT_VOCAB_TOPK (default 0 = off): the MTP drafter's LM head over K kept token ids, with a full-head fallback.

Every draft step runs the drafter LM head over the whole 248320-id vocab (0.47 ms of ~2 ms per step at TP8) and then an
argmax over it. A draft only has to be a good guess: the verify arbitrates every draft, so greedy output does not
depend on which tokens the drafter can propose; acceptance does. With the flag set to K, the drafter head holds K
rows of the shared head weight (gathered once at load into a second, smaller weight), its argmax runs over those K
logits, and the winning pruned index is mapped back to the full token id on device. A kept-set miss (the verify wants
a token outside the set) rejects that draft as any wrong draft would be, so out-of-set text loses acceptance.

Acceptance regression, and why the fallback is on by default. No static set at a K that saves time covers every
language: one language alone needs tens of thousands of ids (ids for 99% of held-out WildChat assistant tokens: zh
46.8K, English 53.2K, ru 22.8K, es 22.5K, fr 17.7K). The first version (v1: counts of 3000 English/code Qwen
completions, K = 98304) kept 1.3% of the CJK, 2.8% of the Cyrillic and 1.1% of the Arabic ids; on device (chained
greedy engine, K=3, B=8, 50 steps) committed tokens per iteration fell zh 2.54 -> 1.41, ru 2.20 -> 1.52, fr 2.44 ->
2.27, es 2.68 -> 2.58 (en unchanged), so a zh user at 32K B=8 went from ~60 to ~34 tok/s. Hence:

  1. The ranking (v2, tt/draft_vocab_data/draft_vocab_rank_v2.npz, scripts/draft_vocab_counts.py) weights the
     normalized token frequencies of the Qwen completions (1), English chat (1, WildChat-4.8M + oasst2 + Aya assistant
     turns) and es / pt / fr / de / it / ru chat (1/4 each). Held-out assistant tokens outside the set at K = 131072,
     per 1000: Qwen completions 1.0, en 2.1, es 0.9, pt 0.7, fr 0.5, de 2.0, it 0.9, ru 0.8; but zh 234, ja 278, ko 312,
     ar 219, vi 512, tr 242, pl 246, id 234, uk 96 (v1 at 98304: en 15, es 105, fr 103, ru 401, zh 451).
  2. QWEN36_DRAFT_VOCAB_FALLBACK (default 1): DraftVocabGuard below watches every row's recent tokens (its prompt tail,
     then each commit) and the engine drafts with the FULL head (the flag-off drafter, its own captured traces) for
     every step in which any active row has GUARD_TRIGGER or more of its last GUARD_WINDOW tokens outside the set. The
     drafter is one batched matmul, so the head is chosen per step for the whole batch. A batch of covered languages
     keeps the pruned head; a batch with a zh / ja / ko / ar / ... row runs the flag-off drafter. Simulated on 250
     held-out replies per language (history-seeded, 6-token decision lag): out-of-set tokens drafted by the pruned head,
     per 1000 = the acceptance the fallback does not save: zh 2.7, hi 2.8, tr 2.0, fa 1.5, uk 1.4, ar 1.4, vi 1.7, ja
     0.6, ko 0.4; covered languages 0.4-1.1; an English row spends ~6% of its tokens on the full head.
  0 turns the fallback off: only for traffic known to stay inside the set (English / code / the five European
  languages / Russian). With it on, the speedup applies only to steps whose whole batch is covered.

Which K ids. The added/special tokens (always kept), then the ranking in order, then the unranked ids in ascending id
order (the byte-level BPE assigns ids in merge order, so a lower id is the more common token).
QWEN36_DRAFT_VOCAB_FILE points at another file: a v2 rank file (rank, always) or a v1 counts file (ids, completion,
prompt, always; ranked by completion + prompt count). For a deployment's own traffic, count its generations
(scripts/draft_vocab_counts.py --traces ... --groups qwen=1).

Layout on the mesh. The chosen ids are sorted ascending and split into TP equal contiguous blocks, block d on device d
(the same ShardTensorToMesh(dim=-1) the full head uses), so every device holds K/TP columns whatever the id
distribution. Pruned index p = d * K/TP + local column, and the fused per-shard argmax (QWEN36_SPEC_ARGMAX_FUSED,
tt/spec_argmax_fused.py, required) returns ids[p] from a replicated uint32 table. Because the table is sorted, the
lowest pruned index among equal maxima is the lowest token id among them, the tie rule of the full-vocab argmax: when
the drafter's full-vocab argmax lies in the set, the pruned head drafts that same token (up to the BFP rounding of the
regrouped weight tiles: block-float exponents are shared by 16 neighbouring columns, which differ in the gathered copy).
K is rounded up to a multiple of 32 * TP (whole tiles per device). K >= vocab leaves the flag off.

Choosing K (8-chip TP ring, r3 spec stack; tests/test_draft_vocab_prune.py timing): head + argmax per draft step
full 507 us, K = 131072 292 us (0.64 ms saved per K=3 iteration), 98304 215 us (0.88 ms), 65536 160 us. 131072 is the
recommended K with the v2 ranking: at 98304 the covered languages miss 6-9 per 1000 (rows trigger the fallback more and
lose more drafts before it does).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger

DRAFT_VOCAB_TOPK_ENV = "QWEN36_DRAFT_VOCAB_TOPK"
DRAFT_VOCAB_FILE_ENV = "QWEN36_DRAFT_VOCAB_FILE"
DRAFT_VOCAB_FALLBACK_ENV = "QWEN36_DRAFT_VOCAB_FALLBACK"
DEFAULT_RANK_FILE = Path(__file__).resolve().parent / "draft_vocab_data" / "draft_vocab_rank_v2.npz"
TABLE_WIDTH = 16  # uint32 entries per table row: one 64 B page per row, the DRAM read granule of the pick kernel
GUARD_WINDOW = 128  # DraftVocabGuard: the recent tokens of a row it looks at
GUARD_TRIGGER = 2  # ... and how many of them outside the set put the batch on the full head
_TILE = 32


def draft_vocab_topk():
    """QWEN36_DRAFT_VOCAB_TOPK as an int (0 = off; negative or empty = off)."""
    raw = os.environ.get(DRAFT_VOCAB_TOPK_ENV, "0").strip()
    return max(0, int(raw or 0))


def draft_vocab_fallback():
    """QWEN36_DRAFT_VOCAB_FALLBACK (default 1): the full-head fallback of a pruned drafter (DraftVocabGuard); 0 = off."""
    return os.environ.get(DRAFT_VOCAB_FALLBACK_ENV, "1").strip() != "0"


def draft_vocab_file():
    return Path(os.environ.get(DRAFT_VOCAB_FILE_ENV, "") or DEFAULT_RANK_FILE)


def round_topk(k, vocab_size, num_devices):
    """K rounded up to whole tiles per device (a multiple of 32 * num_devices); 0 when that reaches the full vocab."""
    if k <= 0:
        return 0
    step = _TILE * max(1, int(num_devices))
    kr = -(-int(k) // step) * step
    return 0 if kr >= int(vocab_size) else kr


def load_counts(path=None):
    """(ids int64 [n], score int64 [n], always int64 [m]) from a v1 counts file (np.savez: ids, completion, prompt?,
    always?). score = completion + prompt counts."""
    path = Path(path) if path is not None else draft_vocab_file()
    with np.load(path, allow_pickle=False) as d:
        ids = d["ids"].astype(np.int64)
        score = d["completion"].astype(np.int64)
        if "prompt" in d.files:
            score = score + d["prompt"].astype(np.int64)
        always = d["always"].astype(np.int64) if "always" in d.files else np.zeros(0, dtype=np.int64)
    assert ids.ndim == 1 and ids.shape == score.shape, f"{path}: ids {ids.shape} vs counts {score.shape}"
    return ids, score, always


def order_ids(vocab_size, always, ranked):
    """Every id < vocab_size once: ``always`` (ascending), then ``ranked`` in its order, then the rest ascending."""
    v = int(vocab_size)
    first = np.unique(always[(always >= 0) & (always < v)])
    ranked = ranked[(ranked >= 0) & (ranked < v)]
    taken = np.zeros(v, dtype=bool)
    order = []
    for block in (first, ranked):
        block = block[~taken[block]]  # an always-kept id that is also ranked is placed once, with the first block
        taken[block] = True
        order.append(block)
    order.append(np.nonzero(~taken)[0])
    out = np.concatenate(order)
    assert out.shape[0] == v
    return out


def rank_ids(vocab_size, ids, score, always):
    """v1 counts: every id < vocab_size once, in selection order: ``always`` (ascending), then the counted ids by
    (-score, id), then the remaining ids ascending."""
    assert np.unique(ids).shape == ids.shape, "counts file lists an id twice"
    keep = (ids >= 0) & (ids < int(vocab_size)) & (score > 0)
    ids, score = ids[keep], score[keep]
    return order_ids(vocab_size, always, ids[np.lexsort((ids, -score))])  # primary key -score, then id


def selection_order(vocab_size, path=None):
    """The selection order of every id < vocab_size from a v2 rank file (rank, always) or a v1 counts file."""
    path = Path(path) if path is not None else draft_vocab_file()
    with np.load(path, allow_pickle=False) as d:
        if "rank" in d.files:
            rank = d["rank"].astype(np.int64)
            always = d["always"].astype(np.int64) if "always" in d.files else np.zeros(0, dtype=np.int64)
            assert np.unique(rank).shape == rank.shape, f"{path}: rank lists an id twice"
            return order_ids(vocab_size, always, rank)
    return rank_ids(vocab_size, *load_counts(path))


def select_draft_vocab(k, vocab_size, path=None):
    """The K drafter ids, int64 sorted ascending (K <= vocab_size)."""
    order = selection_order(vocab_size, path)
    assert 0 < k <= order.shape[0], (k, order.shape)
    return torch.from_numpy(np.sort(order[:k]).copy())


def ids_digest(ids):
    """Short content hash of an id set (names the pruned weight's cache file, so another set never reuses it)."""
    return hashlib.sha1(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()[:10]


class DraftVocabGuard:
    """Host-side head switch of the pruned drafter (QWEN36_DRAFT_VOCAB_FALLBACK, see the module docstring).

    Per engine row, over its host-visible token stream (the prompt tail at prefill, then every committed token): a row is
    off-vocab while ``trigger`` or more of its last ``window`` tokens lie outside the kept set, and the engine runs the
    full drafter head for a step when any active row is off-vocab. Kept as the stream positions of each row's last
    ``trigger`` out-of-set tokens: off-vocab iff the oldest of them is within the last ``window`` positions, so a step
    without a miss costs a few vector ops. Host only (numpy)."""

    _NEVER = -(1 << 62)

    def __init__(self, keep_ids, vocab_size, rows, window=GUARD_WINDOW, trigger=GUARD_TRIGGER):
        self.keep = np.zeros(int(vocab_size), dtype=bool)
        self.keep[np.asarray(keep_ids, dtype=np.int64)] = True
        self.rows, self.window, self.trigger = int(rows), int(window), int(trigger)
        assert self.window > 0 and 0 < self.trigger <= self.window, (window, trigger)
        self.seen = np.zeros(self.rows, dtype=np.int64)  # tokens observed per row
        self.miss = np.full((self.rows, self.trigger), self._NEVER, dtype=np.int64)  # last misses' positions, oldest first

    def reset(self, row):
        self.seen[row] = 0
        self.miss[row] = self._NEVER

    def observe(self, row, ids):
        """Append token ids (oldest first; negative / out-of-range ids, e.g. -1 padding, are skipped) to row ``row``."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        ids = ids[(ids >= 0) & (ids < self.keep.shape[0])]
        pos = np.nonzero(~self.keep[ids])[0]
        if pos.size:
            self.miss[row] = np.concatenate([self.miss[row], self.seen[row] + pos[-self.trigger :]])[-self.trigger :]
        self.seen[row] += ids.size

    def observe_block(self, tokens, counts, rows=None):
        """A committed block: row r's first counts[r] tokens of ``tokens`` [R, T] (engine rows); ``rows`` limits it."""
        tok = torch.as_tensor(tokens).to(torch.int64).numpy()
        cnt = torch.as_tensor(counts).to(torch.int64).reshape(-1).numpy()
        valid = (np.arange(tok.shape[1])[None, :] < cnt[:, None]) & (tok >= 0) & (tok < self.keep.shape[0])
        if rows is not None:
            valid &= np.isin(np.arange(tok.shape[0]), list(rows))[:, None]
        missed = valid & ~self.keep[np.where(valid, tok, 0)]
        for r in np.nonzero(missed.any(axis=1))[0]:  # rare: rows with an out-of-set token in this block
            pos = self.seen[r] + np.cumsum(valid[r])[missed[r]] - 1
            self.miss[r] = np.concatenate([self.miss[r], pos[-self.trigger :]])[-self.trigger :]
        self.seen[: tok.shape[0]] += valid.sum(axis=1)

    def misses(self, row):
        """Out-of-set tokens among the row's last ``window`` tokens, counted up to ``trigger``."""
        return int((self.miss[row] >= self.seen[row] - self.window).sum())

    def off_vocab(self, row):
        return bool(self.miss[row, 0] >= self.seen[row] - self.window)

    def any_off(self, rows):
        r = np.fromiter(rows, dtype=np.int64)
        return bool(r.size) and bool((self.miss[r, 0] >= self.seen[r] - self.window).any())

    def remap(self, idx):
        """Row i takes the state of row idx[i] (a device slot remap moves the engine rows)."""
        idx = np.asarray(idx, dtype=np.int64)
        self.seen, self.miss = self.seen[idx].copy(), self.miss[idx].copy()


def guard_for(mtp, rows):
    """The DraftVocabGuard of an engine over ``rows`` rows: None without a pruned drafter head or with the fallback off."""
    vp = getattr(mtp, "_vocab_prune", None)
    if vp is None or not draft_vocab_fallback():
        return None
    logger.info(
        f"MTP drafter: full-head fallback on ({DRAFT_VOCAB_FALLBACK_ENV}=1): a step drafts with the full head while any "
        f"active row has >= {GUARD_TRIGGER} of its last {GUARD_WINDOW} tokens outside the K={vp.k} kept ids"
    )
    return DraftVocabGuard(vp.ids.numpy(), vp.vocab_size, rows)


class DraftVocabPrune:
    """The pruned drafter head of one Qwen36MTP: ``weight`` [dim, K/TP] per device (the drafter head's dtype),
    ``table`` the [K/16, 16] uint32 id table (replicated), ``fused`` the FusedShardArgmax over K that returns ids."""

    def __init__(self, mesh_device, ids, weight, num_devices, gather=None, vocab_size=None):
        import ttnn
        from models.demos.blackhole.qwen36.tt.spec_argmax_fused import FusedShardArgmax, all_gather_dim3

        self.ids = ids
        self.k = int(ids.shape[0])
        self.vocab_size = int(vocab_size) if vocab_size is not None else int(ids.max()) + 1
        self.num_devices = int(num_devices)
        assert self.k % (_TILE * self.num_devices) == 0 and self.k % TABLE_WIDTH == 0, self.k
        self.shard = self.k // self.num_devices
        self.weight = weight
        assert int(weight.shape[-1]) == self.shard, f"pruned head width {weight.shape[-1]} != K/TP {self.shard}"
        self.table = ttnn.from_torch(
            ids.to(torch.int32).reshape(self.k // TABLE_WIDTH, TABLE_WIDTH).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.fused = FusedShardArgmax(
            mesh_device, self.k, self.num_devices, gather=gather or all_gather_dim3, remap=self.table
        )

    def owns(self, logits):
        """True for a [.., rows, K/TP] logits shard of this pruned head (the full head's shard is vocab/TP wide)."""
        return int(logits.shape[-1]) == self.shard

    def ids_of(self, logits, mode):
        """Full-vocab token ids of the pruned ``logits`` shard (not freed) in the FusedShardArgmax ``mode`` contract."""
        why = self.fused.supports(logits)
        assert why is None, f"{DRAFT_VOCAB_TOPK_ENV}: pruned drafter logits cannot take the fused argmax ({why})"
        return self.fused(logits, mode)

    def release(self):
        import ttnn

        for t in (self.weight, self.table):
            ttnn.deallocate(t)


def log_coverage(ids, path=None):
    """One INFO line: the K and where its ids came from (always-kept, ranked, ascending fill) and the ranking's source."""
    try:
        path = Path(path) if path is not None else draft_vocab_file()
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
            ranked = d["rank"] if "rank" in d.files else d["ids"]
        inset = int(np.isin(ranked.astype(np.int64), ids.numpy()).sum())
        src = meta.get("groups") or meta.get("source") or path.name
        logger.info(
            f"MTP drafter: pruned vocab K={int(ids.shape[0])} keeps {inset} of the {len(ranked)} ranked ids of "
            f"{path.name} (groups {src}); the rest are always-kept or the lowest unranked ids"
        )
    except Exception as e:  # the coverage line is informational only
        logger.info(f"MTP drafter: pruned vocab K={int(ids.shape[0])} (coverage not computed: {e})")
