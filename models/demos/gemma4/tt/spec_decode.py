# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding for Gemma4 (it-assistant MTP/EAGLE drafter).

Two entry points share the drafter and the packed verify:

* ``generate`` / ``generate_batched``: the demo loops (batch=1, and the
  batched greedy loop), each driving one request set to completion.
* the serving API (``serving_prepare`` / ``serving_capture`` /
  ``serving_reseed`` / ``serving_step``): one captured trace per
  ``(rows, drafts)`` shape and verify width, replayed once per engine step
  for whichever rows the vLLM contract adapter hands it (see the
  "serving: batched fused records" section).

Loop (one iteration produces up to ``draft_len + 1`` committed tokens):

  1. Draft: the assistant proposes ``K = draft_len`` tokens autoregressively from
     a single fixed position, recurrently feeding its own projected hidden state
     and cross-attending into the target's last sliding / last full layer KV.
  2. Verify: the target runs ONE batched forward over ``[anchor, d1, ..., dK]``
     at consecutive positions (candidates in the batch dim, the user's page-table
     row replicated). This appends KV and yields per-position logits + hidden.
  3. Accept: greedy (argmax match) or speculative sampling. Committed = the
     matched prefix + one bonus/correction token. KV rollback at batch=1 is
     implicit — rejected positions are overwritten next iteration.

Correctness: the committed tokens are ALWAYS produced by the target verify, so
greedy speculative decode matches plain greedy decode and sampling matches the
target distribution — independent of the drafter's accuracy (which only affects
the acceptance rate / speed). The verify forward runs BATCHED (anchor + K
candidates), so its per-user RoPE + batched SDPA differ from batch=1 decode by
~1e-5; this flips only target near-ties (top-2 logit gap < ~1), so greedy
spec-decode is token-identical to plain greedy up to the first near-tie token
and produces an equally-valid greedy trajectory thereafter.
"""

import os
import time
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allgather

# Device argmax tile rule: the multicore argmax is exact for one 32-row tile,
# so a serving shape packs ``rows * (drafts + 1) <= 32`` verify rows.
SPEC_ROW_TILE = 32
# Seed-hidden pool: rows 0..31 hold the previous replay's verify hidden,
# rows 32..63 the last reseed's target hidden (serving_seed_index).
SPEC_POOL_ROWS = 2 * SPEC_ROW_TILE
SPEC_K_BY_ROWS_ENV = "GEMMA4_SPEC_K_BY_ROWS"
SPEC_K_BY_ROWS_DEFAULT = "5:5"  # K=5 up to 5 live rows; above the table the step is the plain decode (K=0)
# Additive mask value: -1e9, not -inf (flash kernels NaN on -inf).
SPEC_MASK_NEG = -1e9
# The in-trace mask builder works in key chunks of this width: its fp32
# intermediates are [B, 1, P, chunk] with P tile-padded to 32 rows, so one
# chunk costs B * 32 * chunk * 4 bytes (67 MB at B=16) instead of the full
# S_k (539 MB at 263168). Rungs and chunks are multiples of 1024 (tile-aligned).
SPEC_MASK_KEY_CHUNK = 32768
# Default verify-width ladder: x4 rungs from 2048 up to the top rung (one
# captured record per shape and rung). GEMMA4_SPEC_WIDTH_RUNGS (read by the
# adapter, passed as ``max_rungs``) keeps the LARGEST rungs.
SPEC_WIDTH_RUNG_SMALLEST = 2048
SPEC_WIDTH_RUNG_FACTOR = 4
SPEC_WIDTH_RUNGS_ENV = "GEMMA4_SPEC_WIDTH_RUNGS"
# GEMMA4_SPEC_VERIFY_CHECK=N (default 0): the first N SERVED replays of every (shape, rung) also run a plain target
# forward of each live row's anchor at its position and compare its argmax with the packed verify's row-0 argmax
# (the same keys in both: a disagreement past a bf16 near-tie means the packed verify read the wrong keys -- a
# mask pitch, table width or KV write-slot defect -- and is logged as a WARNING with the rows). One eager forward
# per checked replay; the forward rewrites the anchor's K/V in place with the same values. The zero-table replays
# of serving_measure_costs are never checked: with every key on block 0 the rows' own draft K/V alias into every
# earlier block, so the packed verify and a plain forward see different degenerate contexts and near-flat logits.
SPEC_VERIFY_CHECK_ENV = "GEMMA4_SPEC_VERIFY_CHECK"
# Measured 12B TP8 (3 GB region): 21.9-31.9 MB per captured record, width-independent.
SPEC_RECORD_MB_DEFAULT = 32.0


def _to_probs(logits_row, temperature, top_p, top_k):
    """torch logits [vocab] -> probability vector [vocab] under temp/top-p/top-k.

    temperature<=0 returns a one-hot (greedy) distribution.
    """
    logits_row = logits_row.float()
    if not temperature or temperature <= 0:
        probs = torch.zeros_like(logits_row)
        probs[int(torch.argmax(logits_row))] = 1.0
        return probs
    logits_row = logits_row / temperature
    if top_k and top_k > 0 and top_k < logits_row.numel():
        kth = torch.topk(logits_row, top_k).values[-1]
        logits_row = torch.where(logits_row < kth, torch.full_like(logits_row, float("-inf")), logits_row)
    probs = torch.softmax(logits_row, dim=-1)
    if top_p and 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    s = probs.sum()
    return probs / s if s > 0 else probs


def auto_draft_len(prompt_len, default=3):
    """Pick the speculative draft length for a prompt (``GEMMA4_SPEC_DRAFT_LEN=auto``).

    The optimum is INTERIOR and context-dependent, so neither the old fixed 3 nor
    "as large as possible" is right. Measured on 31B / native galaxy column tp=8,
    traced greedy (tok/s/user):

        ISL    K=3    K=5    K=7
        128    47.0   40.6    -
        4096   55.3   62.9   46.0

    Short prompts give the drafter little context, so acceptance is low and the
    extra verify rows of a long draft cost more than they return; with real
    context acceptance rises and a longer draft pays. Acceptance itself is
    non-monotonic in K (2.96/5 -> 2.67/7 at 4k) because the assistant drafts
    autoregressively from ONE fixed position, so late drafts degrade.

    Kept as a coarse threshold on purpose: the fused spec trace is captured PER
    draft length, so switching K at runtime costs a re-capture (seconds). A
    finer adaptive policy wants two pre-captured traces (cf. the qwen36 MTP
    "auto-K 11/7"); that is follow-up work.
    """
    env = os.environ.get("GEMMA4_SPEC_DRAFT_LEN")
    if env and env.strip().lower() not in ("auto", ""):
        return int(env)
    return 5 if prompt_len is not None and prompt_len >= 1024 else default


def auto_draft_len_batched(prompt_len, batch, default=3):
    """Batch-aware draft length. Returns 0 when speculation should be OFF.

    Speculation pays only while decode is bandwidth-bound; batch amortizes the
    weight read, and the verify's B*(K+1) rows also cross a hard 32-row
    tile/batch cliff. Measured on 31B / native galaxy column tp=8, short
    prompts, traced greedy, AGGREGATE tok/s vs the plain batched baseline:

        B   base    K=1    K=2    K=3    K=5    best
        8   189.9   239.1  255.6  263.5  144.8  K=3 (1.39x)  B*(K+1)=32
        16  357.7   460.9  260.5   -      -     K=1 (1.29x)  B*(K+1)=32
        32  682.9   515.5  454.8  450.6  392.8  NONE (<=0.75x)

    Every optimum sits exactly at B*(K+1) == 32, with a cliff beyond it, and at
    B=32 even K=1 (64 rows) loses -- past the compute knee no K wins (ceiling
    (1+aK)/(K+1) < 1). Policy: K = min(context_K, 32//B - 1); 0 => run plain
    batched decode instead.
    """
    if batch is None or batch <= 1:
        return auto_draft_len(prompt_len, default=default)
    env = os.environ.get("GEMMA4_SPEC_DRAFT_LEN")
    if env and env.strip().lower() not in ("auto", ""):
        return int(env)
    row_cap = 32 // int(batch) - 1
    if row_cap < 1:
        return 0
    return min(auto_draft_len(prompt_len, default=default), row_cap)


# ── serving shapes and widths (pure host arithmetic) ─────────────────────────
def spec_check_shape(shape):
    """Validate one serving shape ``(rows, drafts)`` against the 32-row argmax tile rule; returns it as ints."""
    try:
        rows, k = shape
    except (TypeError, ValueError):
        raise ValueError(f"a serving shape is (rows, drafts), got {shape!r}") from None
    rows, k = int(rows), int(k)
    if rows < 1 or k < 1:
        raise ValueError(f"serving shape {(rows, k)}: rows and drafts must be >= 1")
    if rows * (k + 1) > SPEC_ROW_TILE:
        raise ValueError(
            f"serving shape {(rows, k)} packs {rows * (k + 1)} verify rows, above the {SPEC_ROW_TILE}-row argmax tile"
        )
    return (rows, k)


def _same_device_buffer(a, b):
    """True when two ttnn tensors share one device buffer (ttnn returns the INPUT tensor, under a new Python
    wrapper, from to_memory_config / pad when nothing changes, so object identity is not enough and a forced
    deallocate of the "copy" would free the caller's tensor)."""
    return a is b or int(a.buffer_address()) == int(b.buffer_address())


def parse_k_by_rows(text):
    """``"5:5,8:3,16:1"`` -> ((5, 5), (8, 3), (16, 1)): the catalog ``max_rows:k`` table, rows strictly increasing."""
    entries = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        rows_s, sep, k_s = item.partition(":")
        if not sep:
            raise ValueError(f"{SPEC_K_BY_ROWS_ENV} entry {item!r} is not max_rows:k (table {text!r})")
        try:
            entry = (int(rows_s), int(k_s))
        except ValueError:
            raise ValueError(f"{SPEC_K_BY_ROWS_ENV} entry {item!r} is not max_rows:k (table {text!r})") from None
        entries.append(spec_check_shape(entry))
    if not entries:
        raise ValueError(f"{SPEC_K_BY_ROWS_ENV} table {text!r} has no entries")
    for prev, cur in zip(entries, entries[1:]):
        if cur[0] <= prev[0]:
            raise ValueError(f"{SPEC_K_BY_ROWS_ENV} max_rows must increase: {prev[0]} then {cur[0]} in {text!r}")
    return tuple(entries)


def spec_capture_shapes(k_by_rows, k_max):
    """The capture shapes of a ``k_by_rows`` table: ``(1, k_max)`` plus one ``(max_rows, k)`` per entry, sorted."""
    shapes = {spec_check_shape((1, k_max))}
    shapes.update(spec_check_shape(s) for s in k_by_rows)
    return sorted(shapes)


def serving_width_ladder(
    max_context,
    draft_len,
    bucket=1024,
    max_rungs=None,
    factor=SPEC_WIDTH_RUNG_FACTOR,
    smallest=SPEC_WIDTH_RUNG_SMALLEST,
):
    """The default verify-width ladder of one shape: ``smallest``, x``factor``, ... below the top rung, then
    the top rung ``round_up(max_context + draft_len + 2, bucket)`` (always present; the largest covering
    verify block for the last position).

    262144 context at K=5: 2048, 8192, 32768, 131072, 263168 (5 records per
    shape). A position below the smallest rung is still served: the rung just
    covers it. ``max_rungs`` keeps the LARGEST rungs (the adapter passes
    ``GEMMA4_SPEC_WIDTH_RUNGS`` here): dropping a small rung only widens short
    prompts, while dropping the largest would leave long ones uncovered.
    Pure: config-time arithmetic.
    """
    bucket, factor, smallest = int(bucket), int(factor), int(smallest)
    if int(max_context) < 1 or int(draft_len) < 1:
        raise ValueError(f"max_context and draft_len must be >= 1, got {max_context}, {draft_len}")
    if bucket < 64 or bucket % 64:
        raise ValueError(f"the width bucket must be a positive multiple of 64, got {bucket}")
    if factor < 2 or smallest < bucket:
        raise ValueError(f"the ladder needs factor >= 2 and smallest >= bucket, got {factor}, {smallest}")

    def _r(n):
        return ((int(n) + bucket - 1) // bucket) * bucket

    top = _r(int(max_context) + int(draft_len) + 2)
    ladder, rung = [], _r(smallest)
    while rung < top:
        ladder.append(rung)
        rung = _r(rung * factor)
    ladder.append(top)
    if max_rungs is not None:
        if int(max_rungs) < 1:
            raise ValueError(f"max_rungs must be >= 1, got {max_rungs}")
        ladder = ladder[-int(max_rungs) :]
    return ladder


@dataclass
class ServingTraceProjection:
    """Trace-region budget of a serving plan: ``records`` captures of ~``record_mb`` each on top of ``base_mb``
    (the traces already in the region), against ``region_mb`` when the device's region size is readable."""

    records: int
    per_shape: dict
    record_mb: float
    base_mb: float
    projected_mb: float
    region_mb: object  # float, or None when the region size is not readable

    @property
    def fits(self):
        """True/False against the region size; None when it is unknown."""
        return None if self.region_mb is None else self.projected_mb <= float(self.region_mb)

    def describe(self):
        per_shape = ", ".join(f"{shape}: {n}" for shape, n in self.per_shape.items())
        text = (
            f"{self.records} records ({per_shape}) x ~{self.record_mb:.1f} MB + {self.base_mb:.1f} MB in use "
            f"-> ~{self.projected_mb:.0f} MB of trace region"
        )
        if self.region_mb is None:
            return text + " (device region size not readable)"
        return text + f" of {float(self.region_mb):.0f} MB ({'fits' if self.fits else 'EXCEEDS the region'})"


def serving_trace_projection(shapes, widths_by_shape, record_mb=SPEC_RECORD_MB_DEFAULT, region_mb=None, base_mb=0.0):
    """Pure budget arithmetic for a validated plan (``serving_plan`` output)."""
    record_mb = float(record_mb)
    if record_mb <= 0:
        raise ValueError(f"record_mb must be > 0, got {record_mb}")
    per_shape = {shape: len(widths_by_shape[shape]) for shape in shapes}
    records = sum(per_shape.values())
    return ServingTraceProjection(
        records=records,
        per_shape=per_shape,
        record_mb=record_mb,
        base_mb=float(base_mb),
        projected_mb=float(base_mb) + records * record_mb,
        region_mb=None if region_mb is None else float(region_mb),
    )


def serving_plan(shapes, widths_by_shape, max_rows, max_context=None, max_rungs=None):
    """Validate a serving plan; returns ``(sorted shapes, {shape: sorted widths})``.

    Every shape fits ``max_rows`` rows and the tile rule; every shape has a
    non-empty width list of positive multiples of 64 (the SDPA key chunk);
    ``widths_by_shape`` names no other shape. With ``widths_by_shape=None``
    every shape gets the default ``serving_width_ladder(max_context, k,
    max_rungs=max_rungs)``.
    """
    max_rows = int(max_rows)
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    plan = sorted({spec_check_shape(s) for s in shapes})
    if not plan:
        raise ValueError("serving_prepare needs at least one shape")
    for rows, _k in plan:
        if rows > max_rows:
            raise ValueError(f"shape {(rows, _k)} has more rows than max_rows={max_rows}")
    if widths_by_shape is None:
        if max_context is None:
            raise ValueError("the default width ladder needs max_context (the ladder context in tokens)")
        widths_by_shape = {shape: serving_width_ladder(max_context, shape[1], max_rungs=max_rungs) for shape in plan}
    widths = {}
    for shape in plan:
        got = widths_by_shape.get(shape) if hasattr(widths_by_shape, "get") else None
        if not got:
            raise ValueError(f"no verify widths for shape {shape} (have {sorted(widths_by_shape)})")
        ws = sorted({int(w) for w in got})
        bad = [w for w in ws if w < 64 or w % 64]
        if bad:
            raise ValueError(f"shape {shape}: verify widths must be positive multiples of 64, got {bad}")
        widths[shape] = ws
    extra = sorted(set(widths_by_shape) - set(plan))
    if extra:
        raise ValueError(f"widths_by_shape names shapes that are not prepared: {extra}")
    return plan, widths


def serving_width_for(widths, max_position, k):
    """Narrowest width ``>= max_position + k + 2`` (a P = k + 1 row verify block at ``max_position``, plus one);
    ``None`` when no width covers it."""
    need = int(max_position) + int(k) + 2
    fits = [int(w) for w in widths if int(w) >= need]
    return min(fits) if fits else None


def serving_pad_rows(values, B):
    """``values`` for the live rows, then the first live row's value for every padding row of a ``B``-row record.

    Padding rows MIRROR live row 0 (same anchor, position, seed row and page
    tables) instead of standing on a zero page table: block 0 is vLLM's null
    block only in the full-attention pool, while a bounded sliding ring pool
    gives wire row 0 the blocks ``[0, ring/block_size)`` (the generator's
    bounded table install), so a padding row writing "block 0" would write into
    a live request's ring. A mirror row computes bit-identical drafts and K/V
    from identical inputs and writes identical bytes to the slots row 0 writes
    itself -- a benign race on the batched cache write.
    """
    values = list(values)
    if not values:
        raise ValueError("a record needs at least one live row to mirror")
    if len(values) > int(B):
        raise ValueError(f"{len(values)} live rows for a {B}-row record")
    return values + [values[0]] * (int(B) - len(values))


def serving_seed_index(shape, rows, seed_cols, previous, seeded):
    """Rows of the 64-row seed-hidden pool that each record row gathers its drafter seed from.

    ``previous`` is ``None`` or ``(K_prev, rows_prev)`` of the last replay;
    ``seeded`` maps a row id to its row in the last reseed. Live row r with
    ``seed_cols[r] = c`` gathers verify-hidden row ``r' * (K_prev + 1) + c``
    (r' = the row's index in the last replay); with ``None`` it gathers reseed
    row ``32 + seeded[row]``. Padding rows gather row 0's seed row
    (``serving_pad_rows``: a padding row mirrors live row 0).
    """
    B, _K = shape
    live = len(rows)
    if not 1 <= live <= B:
        raise ValueError(f"shape {shape}: {live} live rows, expected 1..{B}")
    if len(seed_cols) != live:
        raise ValueError(f"shape {shape}: {len(seed_cols)} seed_cols for {live} live rows")
    if len({int(r) for r in rows}) != live:
        raise ValueError(f"shape {shape}: duplicate rows in {list(rows)}")
    out = []
    for r, (slot, col) in enumerate(zip(rows, seed_cols)):
        slot = int(slot)
        if col is None:
            if slot not in seeded:
                raise ValueError(f"row {slot} has no seed: it was not in the last serving_reseed ({sorted(seeded)})")
            out.append(SPEC_ROW_TILE + int(seeded[slot]))
            continue
        if previous is None:
            raise ValueError(f"row {slot}: seed_cols[{r}]={col} refers to a previous replay, but none ran")
        k_prev, rows_prev = previous
        if slot not in rows_prev:
            raise ValueError(f"row {slot}: seed_cols[{r}]={col} but the last replay served rows {list(rows_prev)}")
        col = int(col)
        if not 0 <= col <= int(k_prev):
            raise ValueError(f"row {slot}: seed_cols[{r}]={col} outside 0..{int(k_prev)} (last replay K={k_prev})")
        out.append(list(rows_prev).index(slot) * (int(k_prev) + 1) + col)
    return serving_pad_rows(out, B)


def serving_masks_host(positions, P, window, S_k, ring=None):
    """Host additive masks for one packed verify of ``P`` rows per batch row.

    Returns ``(full [B, 1, P, S_k], sliding [B, 1, P, S_r])`` bf16, row p of
    batch b attending keys ``j <= c_b + p`` (full) and additionally
    ``j > c_b + p - window`` (sliding, unbounded). With ``ring`` (a bounded
    sliding pool) the sliding mask spans ring slots: ``S_r = ring`` and slot j
    holds position ``top_b - ((top_b - j) mod ring)`` with ``top_b = c_b + P
    - 1``, the ``_pv_host_inputs`` rule. The head repeat happens in-trace.
    """
    cs = torch.as_tensor([int(c) for c in positions], dtype=torch.int64)
    B = int(cs.numel())
    P, S_k, window = int(P), int(S_k), int(window)
    if B < 1 or P < 1 or S_k < 1 or window < 1:
        raise ValueError(f"masks need B, P, S_k, window >= 1, got {B}, {P}, {S_k}, {window}")
    zero = torch.tensor(0.0, dtype=torch.bfloat16)
    neg = torch.tensor(SPEC_MASK_NEG, dtype=torch.bfloat16)
    p = torch.arange(P, dtype=torch.int64)
    upper = (cs[:, None] + p[None, :])[:, :, None]  # [B, P, 1]
    j = torch.arange(S_k, dtype=torch.int64)[None, None, :]
    full = j <= upper
    mask_full = torch.where(full, zero, neg).reshape(B, 1, P, S_k)
    if ring:
        ring = int(ring)
        jr = torch.arange(ring, dtype=torch.int64)
        top = (cs + P - 1)[:, None]  # [B, 1]
        d = torch.remainder(top - jr[None, :], ring)[:, None, :]  # [B, 1, ring]
        lo = (P - 1 - p)[None, :, None]  # [1, P, 1]
        ok = (d >= lo) & (d < lo + window) & (d <= top[:, :, None])
        mask_slide = torch.where(ok, zero, neg).reshape(B, 1, P, ring)
    else:
        ok = full & (j > upper - window)
        mask_slide = torch.where(ok, zero, neg).reshape(B, 1, P, S_k)
    return mask_full, mask_slide


def serving_select_rows(table, rows, B, width, allow_clip):
    """Host ``[B, width]`` int64 page table: row r = ``table[rows[r]]`` fitted to ``width``, padding rows zero.

    A narrower source is zero-padded (zeros = block 0, never read past the
    live top). A wider source is clipped only with ``allow_clip`` -- the
    verify tables are width-matched to their mask on purpose; otherwise a
    live block past ``width`` is a capacity error and raises.
    """
    t = torch.as_tensor(table)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"page table must be [rows, blocks], got shape {tuple(t.shape)}")
    B, width = int(B), int(width)
    if len(rows) > B:
        raise ValueError(f"{len(rows)} rows for a {B}-row table")
    t = t.to(torch.int64)
    cols = int(t.shape[1])
    n = min(width, cols)
    out = torch.zeros(B, width, dtype=torch.int64)
    for r, slot in enumerate(rows):
        slot = int(slot)
        if not 0 <= slot < int(t.shape[0]):
            raise ValueError(f"row {slot} outside the {int(t.shape[0])}-row page table")
        row = t[slot]
        if cols > width and not allow_clip and bool(row[width:].any()):
            raise ValueError(f"page-table row {slot} holds live blocks past column {width} (table width {cols})")
        out[r, :n] = row[:n]
    return out


def serving_mask_plan(positions, P, ring=None):
    """The per-row scalars the in-trace mask builder compares the key iota against.

    ``q [B, P]`` = c_b + p (the verify positions, uploaded as ``v_pos``); with
    a ``ring``: ``top_b = c_b + P - 1``, ``tm = top_b mod ring`` and
    ``base = top_b - tm``, so that ring slot j holds position
    ``base + j - ring * [j > tm]`` (the ``_pv_host_inputs`` rule
    ``top - ((top - j) mod ring)`` without a device modulo).
    """
    cs = torch.as_tensor([int(c) for c in positions], dtype=torch.int64)
    P = int(P)
    plan = {"q": cs[:, None] + torch.arange(P, dtype=torch.int64)[None, :]}
    if ring:
        top = cs + P - 1
        plan["tm"] = torch.remainder(top, int(ring))
        plan["base"] = top - plan["tm"]
    return plan


def serving_mask_rows(positions, P, h):
    """The packed-row verify positions the in-trace mask builder compares the key iota against (uploaded as
    ``v_pos_rows``): ``[1, B*h*P]`` int64, row ``b*h*P + h'*P + p = c_b + p`` for every local head ``h'`` (the
    packed verify's row order, ``serving_masks_host`` row p of batch b repeated per head)."""
    P, h = int(P), int(h)
    if P < 1 or h < 1:
        raise ValueError(f"mask rows need P, h >= 1, got {P}, {h}")
    return torch.tensor([[int(c) + p for c in positions for _ in range(h) for p in range(P)]], dtype=torch.int64)


def serving_masks_from_plan(plan, S_k, window, ring=None, h=1):
    """Torch emulation of ``_srv_masks_device``, comparison for comparison (fp32 key - query, gtz/gez/ltz
    indicators cast to bf16, scaled by ``SPEC_MASK_NEG``) at the packed height ``h * P`` (row ``h' * P + p``
    reads row p); must equal ``serving_masks_host`` repeated ``h`` times along the rows."""
    q = plan["q"].to(torch.float32).repeat(1, int(h))[:, None, :, None]  # [B, 1, h*P, 1]
    B, P = int(q.shape[0]), int(q.shape[2])
    j = torch.arange(int(S_k), dtype=torch.float32)[None, None, None, :]
    diff = j - q  # key - query
    gt = (diff > 0).to(torch.bfloat16)
    mask_full = gt * SPEC_MASK_NEG
    if not ring:
        far = ((diff + float(window)) * -1.0 >= 0).to(torch.bfloat16)
        mask_slide = (gt + far) * SPEC_MASK_NEG
    else:
        jr = torch.arange(int(ring), dtype=torch.float32)[None, None, None, :]
        tm = plan["tm"].to(torch.float32)[:, None, None, None]
        base = plan["base"].to(torch.float32)[:, None, None, None]
        pos = jr + base + (jr - tm > 0).to(torch.float32) * (-float(ring))  # [B, 1, 1, ring]
        d = pos - q  # [B, 1, P, ring]
        gt_r = (d > 0).to(torch.bfloat16)
        far_r = ((d + float(window)) * -1.0 >= 0).to(torch.bfloat16)
        neg_r = (pos < 0).to(torch.bfloat16)
        mask_slide = (gt_r + far_r + neg_r > 0).to(torch.bfloat16) * SPEC_MASK_NEG
    return mask_full.reshape(B, 1, P, int(S_k)), mask_slide.reshape(B, 1, P, int(ring) if ring else int(S_k))


def packed_verify_kv_groups(h_local, nkv_local, P):
    """The packed-verify query rows each local KV head attends, as the SDPA decode kernel groups them.

    The packed Q is ``[1, B, h_local * P, hd]`` with row ``h * P + p`` (head-major);
    the kernel keeps, for KV head g, output rows ``[g * n, (g + 1) * n)`` with
    ``n = num_q_heads / num_kv_heads`` on the LOGICAL head count
    (``sdpa_decode_program_factory.cpp`` num_q_heads = q.logical_shape[2],
    ``writer_decode_all.cpp`` num_heads_to_write), written row by row. So a
    KV group is a contiguous block of ``h_local / nkv_local`` heads x P rows,
    which is the head-sharded GQA map ``h // (h_local / nkv_local)`` iff
    ``h_local`` divides by ``nkv_local``. The mask is ``[B, 1, h_local * P, S_k]``
    (read per batch for every KV head), the same rows.
    Returns ``[(kv_head, first_row, last_row_exclusive)]``.
    """
    h_local, nkv_local, P = int(h_local), int(nkv_local), int(P)
    if h_local < 1 or nkv_local < 1 or P < 1:
        raise ValueError(f"h_local, nkv_local and P must be >= 1, got {h_local}, {nkv_local}, {P}")
    if h_local % nkv_local:
        raise ValueError(f"{h_local} local query heads do not split evenly over {nkv_local} local KV heads")
    rows_per_group = (h_local // nkv_local) * P
    return [(g, g * rows_per_group, (g + 1) * rows_per_group) for g in range(nkv_local)]


def serving_kv_group_slots(layer_types):
    """vLLM's hybrid KV-cache grouping as one slot index per layer.

    ``_get_kv_cache_groups_uniform_page_size`` buckets layers by type, takes
    ``group_size = min(count per type)`` and gives group ``g`` of a type the
    layers ``layers_of_type[g::num_groups]``; every group owns its own block
    table with DISJOINT block ids. The 12B's (5 sliding, 1 full) x 8 therefore
    has five sliding groups (layers i, i+6, i+12, ...) and one full group, and
    a per-layer-TYPE table would hand 32 of the 40 sliding layers another
    group's blocks. One slot per (type, group), numbered by first appearance
    of the type then by group; a type whose count is not a multiple of the
    group size gets one slot per layer (always correct, only bigger). One
    uniform type is one slot. Returns ``(slot_per_layer, n_slots)``."""
    types = list(layer_types)
    if not types:
        return [], 0
    order = []
    for lt in types:
        if lt not in order:
            order.append(lt)
    counts = {lt: types.count(lt) for lt in order}
    group_size = min(counts.values())
    slots = [None] * len(types)
    base = 0
    for lt in order:
        idxs = [i for i, t in enumerate(types) if t == lt]
        if counts[lt] % group_size == 0:
            num_groups = counts[lt] // group_size
            for o, i in enumerate(idxs):
                slots[i] = base + (o % num_groups)
            base += num_groups
        else:
            for o, i in enumerate(idxs):
                slots[i] = base + o
            base += counts[lt]
    return slots, base


def serving_check_group_tables(installed, slots):
    """Raise unless every layer of a KV group slot carries the same installed table (or every layer of the slot
    has none): the serving records keep one table per slot, which is only lossless under that invariant."""
    by_slot = {}
    for i, (pt, slot) in enumerate(zip(installed, slots)):
        first = by_slot.setdefault(slot, (i, pt))
        fi, fpt = first
        same = (pt is None and fpt is None) or (
            pt is not None and fpt is not None and (pt is fpt or (pt.shape == fpt.shape and torch.equal(pt, fpt)))
        )
        if not same:
            raise ValueError(
                f"per-layer page tables of layers {fi} and {i} differ inside KV group slot {slot}: the serving "
                "records keep one table per vLLM KV-cache group (serving_kv_group_slots); the installed tables "
                "do not follow that grouping"
            )


def serving_verify_table_widths(s_k, block_size_by_type, ring_by_type, blocks):
    """Width (columns) of each layer type's width-matched verify page table at rung ``s_k``.

    The packed SDPA attends the TABLE width times the layer's block size, and
    reads the mask's second and later row tiles at that pitch, so a type's
    table must span EXACTLY its mask: ``s_k // block_size`` columns (or
    ``ring // block_size`` for a bounded ring type, whose sliding mask is
    ring-wide) -- wider than the ``blocks`` columns the adapter's tables hold
    when the top rung passes the model length (it always does: the rung rounds
    ``max_context + K + 2`` up), with the extra columns zero (block 0, keys the
    mask hides). The earlier cap at ``blocks`` left the 31B's sliding table at
    TP4 at 4096 columns under a 263168-wide mask (block 64, the plugin's
    4096-column table): the kernel then attended 262144 keys and read the
    mask's second row tile (packed rows 32-47: three of the eight local heads)
    32 tiles early, so those heads saw 1024 keys past their position -- the
    other draft rows and whatever sat beyond the top -- and the 256K verify
    accepted drafts the target would not have produced. ``block_size_by_type``
    is each type's EFFECTIVE block size (``effective_block_size`` of the
    layer's own view of its cache): under vLLM hybrid kv-cache groups the
    sliding and full-attention pools hold a different number of tokens per
    block (e.g. 256 and 128 for the 12B), so one block size for both types
    would size one table at half or double its mask. Pure: config-time
    arithmetic.
    """
    s_k, blocks = int(s_k), int(blocks)
    if s_k < 1 or blocks < 1:
        raise ValueError(f"s_k and blocks must be >= 1, got {s_k}, {blocks}")
    widths = {}
    for lt, bs in block_size_by_type.items():
        bs = int(bs)
        if bs < 1:
            raise ValueError(f"{lt}: block size must be >= 1, got {bs}")
        ring = ring_by_type.get(lt)
        span = int(ring) if ring else s_k
        if span % bs:
            raise ValueError(f"{lt}: {'ring' if ring else 'rung'} {span} is not a multiple of its block size {bs}")
        widths[lt] = span // bs
    return widths


def serving_mask_padded_bytes(shape, width, elem_bytes=2):
    """Device bytes of ONE ``[B, 1, P, width]`` mask tensor or intermediate: TILE pads P to 32 rows."""
    rows, _k = spec_check_shape(shape)
    return rows * SPEC_ROW_TILE * int(width) * int(elem_bytes)


@dataclass
class ServingStepResult:
    """One ``serving_step`` replay: the drafts and target argmax of the live rows, plus the device logits.

    ``logits`` is the decoder's ONE shared verify-logits buffer
    ``[1, 1, 32, V_padded]`` (rows ``[0, B_v * P)`` are this replay's packed
    rows, user-major), which every record's body overwrites at its end: the
    handle and ``logits_to_host`` are valid until the next ``serving_step`` or
    ``serving_reseed`` of this decoder, so an adapter that may need them reads
    them to host before it drives the decoder again.
    """

    shape: tuple  # (B_v, K_v) of the record that replayed
    width: int  # verify width (S_k rung) used
    live: int  # live rows = the first ``live`` rows of the shape
    drafts: torch.Tensor  # [live, K_v] int64 host: the assistant's drafts per live row
    argmax_ids: torch.Tensor  # [live, K_v + 1] int64 host: target argmax at [anchor, d1..dK]
    logits: object  # the decoder's shared verify-logits buffer [1, 1, 32, V_padded]; see the class docstring
    _reader: object = field(default=None, repr=False, compare=False)

    def logits_to_host(self):
        """[live, K_v + 1, V] bf16 host readback of ``logits`` (softcap and TP gather already applied)."""
        if self._reader is None:
            raise RuntimeError("this ServingStepResult carries no logits reader")
        return self._reader()


class SpeculativeDecoder:
    def __init__(
        self,
        target_model,
        assistant_model,
        mesh_device,
        tt_kv_cache,
        page_table_torch,
        stop_tokens=None,
        draft_len=None,
    ):
        self.target = target_model
        self.assistant = assistant_model
        self.mesh_device = mesh_device
        # ``Gemma4Generator.from_pretrained`` returns the cache wrapped per model
        # instance (``[model0_kv_cache]``); the generator unwraps ``kv_cache[0]``
        # before calling the model. We call the model forward directly, so peel a
        # single-model wrapper down to the per-layer ``[[k, v], ...]`` list. The
        # inner ``[0][0]`` type check disambiguates a wrapper from an already
        # per-layer cache (whose ``[0][0]`` is a ttnn.Tensor, not a list/tuple).
        if (
            isinstance(tt_kv_cache, (list, tuple))
            and tt_kv_cache
            and isinstance(tt_kv_cache[0], (list, tuple))
            and tt_kv_cache[0]
            and isinstance(tt_kv_cache[0][0], (list, tuple))
        ):
            tt_kv_cache = tt_kv_cache[0]
        self.tt_kv_cache = tt_kv_cache
        self.page_table_torch = page_table_torch
        self.stop_tokens = set(stop_tokens or [])
        self.draft_len = int(draft_len if draft_len is not None else os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
        if self.draft_len < 1:
            raise ValueError("draft_len must be >= 1")
        # Recurrent drafter-seed strategy across verify rounds (see generate()).
        self._seed_mode = os.environ.get("GEMMA4_SPEC_SEED_MODE", "reseed")
        if page_table_torch is None:
            raise ValueError("Speculative decoding requires paged attention (page_table_torch is None).")
        self._is_mesh = hasattr(mesh_device, "shape")
        self._mapper = target_model._replicate_to_mesh_mapper()
        self._tp = target_model.mesh_config.tp if target_model.mesh_config else 1
        # The drafter cross-attends to the target's last full / last sliding KV.
        # Build shared_kv from the KV cache ACTUALLY PASSED to this decoder --
        # the model's own on the metal harness, but vLLM's externally-owned
        # paged pool on the server. target_model.get_shared_kv_caches() reads
        # the model's internal tt_kv_cache attribute, which the server never
        # populates (vLLM hands KV in per-step), so the drafter would read an
        # unallocated buffer (TT_FATAL is_allocated in the paged SDPA). Indexing
        # the passed cache by the same last-layer-per-type map is identical on
        # metal and correct on the server.
        self._shared_kv = {lt: self.tt_kv_cache[idx] for lt, idx in target_model.last_kv_layer_by_type.items()}
        # Tracing: persistent I/O buffers + execute_trace replace per-op host
        # dispatch (the untraced loop is host-bound: ~77ms/decode vs a few ms
        # traced). Verify traces are keyed by batch (K+1 for verify, 1 for
        # seed/reseed). Captured lazily on first call at real inputs.
        self._use_trace = os.environ.get("GEMMA4_SPEC_TRACE", "0") == "1"
        self._verify_traces = {}
        # Single batch=1 drafter-step trace, replayed K times. The recurrent
        # hidden is kept ON DEVICE: between trace replays, ttnn.copy(next_hidden
        # -> h_in) refreshes the persistent recurrent buffer, so each replay
        # consumes the previous replay's hidden without a host round-trip. Only
        # the next draft *token* round-trips (host argmax -> tok_in), which is a
        # read + tiny scalar copy (no device allocation -> trace-safe).
        self._draft_trace = None
        # Single FUSED per-iteration trace (K drafter steps + verify in ONE
        # CCL-bearing trace). Replaying ONE trace per iter avoids interleaving
        # distinct CCL traces (the draft/verify-alternation deadlock). Requires
        # on-device argmax + re-embed for the draft recurrence and verify-input
        # assembly (see _fused_iter / _capture_fused_trace).
        self._fused_trace = None
        # Single FUSED batched (B>1) per-iteration trace (batched drafter chain +
        # batched packed verify). Captured once, replayed per iter (prefill never
        # traced). See _capture_fused_trace_batched / _generate_fused_traced_batched.
        self._fused_trace_batched = None
        # Seed mode for the fused greedy trace. "shift" (default) is the fast
        # production mode: it reuses the previous verify hidden row as an
        # approximate drafter seed and repairs the anchor KV at the next verify.
        # "reseed" is the exact assistant contract: run a batch=1 target seed
        # inside the fused trace before drafting, so the drafter sees the current
        # anchor's exact target hidden/KV. It raises acceptance, but the extra
        # target forward currently costs more than it saves; keep it as an A/B
        # correctness/reference mode via GEMMA4_SPEC_FUSED_RESEED=1.
        self._fused_reseed = os.environ.get("GEMMA4_SPEC_FUSED_RESEED", "0") == "1"
        # Cheap approximate seed-row selection for fused shift mode. `next` is
        # the original position-aligned row p+m+1. `current` uses row m (usually
        # the last matched target row), and `last_accepted` uses row m-1 when at
        # least one draft was accepted. These are A/B knobs for acceptance vs
        # seed quality without paying a full target reseed.
        self._fused_shift_seed = os.environ.get("GEMMA4_SPEC_FUSED_SHIFT_SEED", "current")
        # Persistent anchor-hidden buffer for the traced loop (allocated once).
        # The traced loop MUST be allocation-free: any ttnn.clone/slice between
        # execute_trace calls can land in memory a later trace replay uses as
        # scratch and corrupt it (manifests as a hang on a *re*-replay). seed()
        # writes into this buffer via ttnn.copy instead of cloning.
        self._anchor_buf = None
        # Packed-query verify: all K+1 candidates in the query-heads dim of ONE
        # batch=1 forward (one QKV/norm/RoPE over K+1 rows, one masked SDPA per
        # layer, loop-free staging KV write) instead of K+1 pseudo-users with
        # sequential per-candidate KV writes. This is the default multi-token
        # verify; single-token calls (seed/reseed) keep the plain batch=1 path.
        # (The fused single-trace paths keep the batch-dim verify.)
        # S_k (mask key length) is padded to this bucket so the verify trace
        # shape stays stable as the context grows; a new trace is captured when
        # the bucket rolls.
        self._pv_sk_bucket = 1024
        self._srv_check_steps = 0  # GEMMA4_SPEC_VERIFY_CHECK, read by serving_prepare
        self._srv_checked = {}  # (shape, rung) -> served replays checked so far
        self._srv_measuring = False  # serving_measure_costs is replaying (zero tables): no verify check
        self._srv_check_last = []  # the last check's disagreements (row, position, plain id, packed id)
        self._pv_ready = False
        self._pv_a_prev = -1  # last hot block index (-1 ⇒ staging unseeded)
        self._pv_traces = {}  # (P, S_k) -> persistent trace inputs/outputs
        # Serving records (see the "serving: batched fused records" section).
        self._srv_records = None  # {(shape, S_k): record} once serving_prepare ran
        self._srv_captured = False
        self._srv_shape_bufs = {}  # {shape: width-independent buffers}
        self._srv_per_layer = None  # adapter-installed per-layer host page tables
        self._srv_pool_v = None  # [1, 1, 32, backbone] last verify hidden
        self._srv_pool_s = None  # [1, 1, 32, backbone] last reseed hidden
        # The seed forward as a trace per shape, captured with the records (serving_capture): a reseed replays it
        # instead of dispatching the eager verify forward (108-186 ms on the 12B against a ~20 ms traced step).
        # GEMMA4_SPEC_TRACED_SEED=0 keeps every reseed eager.
        self._srv_seed_traces = {}  # shape -> trace id
        self._srv_traced_seed = os.environ.get("GEMMA4_SPEC_TRACED_SEED", "1") == "1"
        # The drafter's argmax on its TP-sharded logits (no vocab all-gather; the per-shard multicore argmax over
        # vocab / tp columns is ~0.05 ms against ~1 ms over the gathered vocab) with the shards' (max, index)
        # candidates exchanged over the TP axis. GEMMA4_SPEC_SHARDED_DRAFT_ARGMAX=0 keeps the gathered argmax.
        self._srv_sharded_draft_argmax = os.environ.get("GEMMA4_SPEC_SHARDED_DRAFT_ARGMAX", "1") == "1"
        self._srv_draft_consts = {}  # (gathered width, vocab slice) -> the candidate mask / column iota / slice offsets
        self._srv_logits = None  # [1, 1, 32, V_padded] the last replay's verify logits (shared by every record)
        self._srv_iota = {}  # key-chunk length -> [1, 1, 1, n] fp32 iota for the in-trace masks
        self._srv_uploaded = {}  # last uploaded small inputs, by key
        self._srv_last = None  # {"shape", "rows", "K"} of the last replay
        self._srv_phase_ms = {}  # the last serving_step's wall by sub-phase (serving_step_phases)
        # GEMMA4_SPEC_REPLAY_BLOCKING=1: block on the replay before its readback, so ``replay_exec`` is the record's
        # device time and ``replay_read`` the readback alone (attribution runs); off, the readback absorbs the wait.
        self._srv_replay_blocking = os.environ.get("GEMMA4_SPEC_REPLAY_BLOCKING", "0") == "1"
        self._srv_frozen = (
            False  # True once serving_prepare ran: the device set is complete, nothing allocates after it
        )
        self._srv_seeded = {}  # row id -> reseed-pool row
        self._srv_projection = None  # ServingTraceProjection of the prepared plan

    def _fused_shift_seed_row(self, accepted, K):
        if self._fused_shift_seed == "current":
            return min(accepted, K)
        if self._fused_shift_seed == "last_accepted":
            return max(min(accepted, K) - 1, 0)
        # Default: position-aligned next row. When all K drafts are accepted,
        # there is no row K+1, so fall back to row K.
        return min(accepted + 1, K)

    # ── device-tensor builders ────────────────────────────────────────────
    def _pos_tensors(self, positions):
        batch = len(positions)
        # int64 host source for the uint32 tensor: ttnn downcasts it host-side, avoiding
        # the int32->uint32 C++ conversion that emits the #18536 row-major warning.
        pu = torch.zeros((1, 32), dtype=torch.int64)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int64)
        pos_uint32 = ttnn.from_torch(
            pu, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper
        )
        pos_int32 = ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        return pos_uint32, pos_int32

    def _tokens_tensor(self, tokens):
        # int64 host source for the uint32 tensor (see _pos_tensors): avoids the
        # int32->uint32 conversion that triggers the #18536 row-major warning.
        t = torch.tensor(tokens, dtype=torch.int64).reshape(1, len(tokens))
        return ttnn.from_torch(
            t, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper
        )

    def _page_table(self, batch, user_idx=0):
        # Replicate ONE user's block row across the `batch` pseudo-users so every
        # candidate/verify position indexes the SAME physical KV blocks (the
        # batch-alias / single-user verify trick). ``user_idx`` selects which
        # real user's row to replicate (default 0).
        row = user_idx
        user_row = (
            self.page_table_torch[row : row + 1]
            if self.page_table_torch.dim() > 1
            else self.page_table_torch.unsqueeze(0)
        )
        pt = user_row.repeat(batch, 1).to(torch.int32)
        return ttnn.from_torch(
            pt, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=self._mapper
        )

    def _shared_kv_page_tables(self, fallback_pt):
        """Per-layer-type page tables for the KV-shared drafter.

        The drafter cross-attends the target's LAST layer of each type
        (get_shared_kv_caches / last_kv_layer_by_type). Unbounded, every layer
        shares one block-ID space, so one flat table serves both types. Under
        BOUNDED sliding the target runs HYBRID per-layer tables and each layer
        owns a DISTINCT block-ID range -- handing the drafter one flat table then
        points it at the right cache with the wrong blocks, and it drafts noise
        (measured acceptance 0.00/5 at 256k).

        Returns {layer_type: page_table}; falls back to the flat table whenever
        per-layer tables are not installed, keeping the unbounded path identical.
        """
        installed = getattr(self.target, "_active_page_tables_per_layer", None)
        by_type = getattr(self.target, "last_kv_layer_by_type", None)
        if not installed or not by_type:
            return {lt: fallback_pt for lt in self._shared_kv}
        cache = getattr(self, "_shared_pt_cache", None)
        if cache is None:
            cache = {}
            self._shared_pt_cache = cache
        key = id(installed)
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = {}
        for lt in self._shared_kv:
            idx = by_type.get(lt)
            pt = installed[idx] if (idx is not None and idx < len(installed)) else None
            if pt is None:
                out[lt] = fallback_pt
                continue
            if not isinstance(pt, ttnn.Tensor):
                row = pt[0:1] if pt.dim() > 1 else pt.unsqueeze(0)
                pt = ttnn.from_torch(
                    row.to(torch.int32),
                    device=self.mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.int32,
                    mesh_mapper=self._mapper,
                )
            out[lt] = pt
        cache[key] = out
        return out

    # ── re-stage the captured page tables (dynamic block tables) ─────────────
    def _restage(self, dev, row):
        """Copy ``row`` into an EXISTING device page-table buffer, in place.

        Widths are fixed at capture, so the row is clipped or zero-padded to the
        buffer's width -- the same contract the model's prefill traces use. A
        padded tail is only safe because it sits past the live top, where the
        mask is NEG; see _pv_tables_per_layer on why the table must never be
        WIDER than its mask.
        """
        if dev is None or row is None:
            return
        if not isinstance(dev, ttnn.Tensor):
            # A TORCH per-layer table (see _page_tables_per_layer): the model
            # converts it to a device tensor INSIDE the graph, so the copy the
            # trace replays was made at capture and cannot be reached from here.
            # Mutating the host tensor would change nothing. Reported once,
            # because it bounds what this refresh fixes -- see
            # refresh_page_tables.
            if not getattr(self, "_restage_warned", False):
                from loguru import logger as _lg

                self._restage_warned = True
                _lg.warning(
                    "MTP verify: per-layer page tables passed as torch tensors are "
                    "baked into the fused trace at capture and are NOT re-staged; "
                    "the drafter may read stale full-attention blocks as the "
                    "request grows (acceptance, not correctness -- the verify's "
                    "own tables are re-staged)"
                )
            return
        w = int(dev.shape[-1])
        rows = int(dev.shape[0]) if len(dev.shape) > 1 else 1
        flat = row.reshape(-1).to(torch.int64)
        if int(flat.numel()) < w:
            flat = torch.cat([flat, torch.zeros(w - int(flat.numel()), dtype=flat.dtype)])
        host = flat[:w].to(torch.int32).reshape(1, w).repeat(rows, 1)
        h = ttnn.from_torch(host, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=self._mapper)
        ttnn.copy_host_to_device_tensor(h, dev)
        h.deallocate(True)

    def refresh_page_tables(self, page_table_torch):
        """Re-stage every captured page table from the CURRENT block table.

        The fused verify trace binds its page tables at capture, which is correct
        for a caller whose table never changes (the demo pre-allocates an
        identity table for the whole run). A vLLM block table is not that: its
        WIDTH is fixed at max_num_blocks_per_req but its CONTENT grows, because
        the KV manager allocates a block only every ``block_size`` tokens. A
        session captured at prefill therefore holds the prompt's blocks and
        zeros past them, so once generation crosses out of the prompt's last
        block the verify reads -- and writes -- the null block.

        Two kinds of state go stale, not one:

        * the DEVICE tables the trace reads (``d_pt``/``v_pt`` flat,
          ``d_ptl``/``v_ptl`` per layer, ``pv_ptl`` packed per layer, and the
          drafter's shared-KV tables), and
        * ``_pv_pages_t``, the HOST map that resolves a hot-block index to a
          physical block id for the per-iteration ``pv_hot``/``pv_embed``
          uploads (_pv_block, _pv_host_inputs). Those uploads happen every
          replay, but they were computed through a map built once in _pv_setup,
          so refreshing only the device tables would leave the writes pointed at
          the wrong physical block.

        WHAT THIS DOES NOT REACH. ``_page_tables_per_layer`` returns TORCH
        tensors, which the model converts to device tensors inside the graph, so
        the copies the trace replays were made at capture. On the packed path
        that affects ``d_ptl`` only -- the drafter's own per-layer decode tables
        -- because the verify reads ``pv_ptl`` and the drafter's shared-KV
        tables, both of which are persistent device buffers and are re-staged
        here. So committed tokens follow the CURRENT blocks (the verify is
        authoritative), while the drafter can still draft against stale
        full-attention blocks as a request grows: that costs acceptance, not
        correctness. Fixing it means giving the fused trace device-resident
        per-layer tables, which is a change to a builder the demo path shares.

        Cheap enough for every step: a handful of small host->device copies of
        fixed width, and the caller only calls it when the row actually changes.
        Counterpart of DFlashFusedDecoder.refresh_page_tables.
        """
        if page_table_torch is None:
            return
        self.page_table_torch = page_table_torch
        flat = (page_table_torch[0] if page_table_torch.dim() > 1 else page_table_torch).to(torch.int64)
        installed = getattr(self.target, "_active_page_tables_per_layer", None)
        # 1. host map behind the per-iteration hot-block uploads
        if getattr(self, "_pv_ready", False):
            self._pv_pages_t = {lt: flat for lt in self._pv_ring}
            if installed:
                for lt, idx in (getattr(self, "_pv_ring_rep", None) or {}).items():
                    pt = installed[idx] if idx < len(installed) else None
                    if pt is not None and hasattr(pt, "dim"):
                        self._pv_pages_t[lt] = (pt[0] if pt.dim() > 1 else pt).to(torch.int64)
        # 2. the packed per-layer tables, cached by (type, width)
        for (lt, _w), dev in (getattr(self, "_pv_pt_cache", None) or {}).items():
            self._restage(dev, self._pv_pages_t.get(lt, flat) if getattr(self, "_pv_pages_t", None) else flat)
        # 3. the drafter's shared-KV tables (per layer type)
        for out in (getattr(self, "_shared_pt_cache", None) or {}).values():
            for lt, dev in (out or {}).items():
                src = flat
                if installed and getattr(self.target, "last_kv_layer_by_type", None):
                    idx = self.target.last_kv_layer_by_type.get(lt)
                    pt = installed[idx] if (idx is not None and idx < len(installed)) else None
                    if pt is not None and hasattr(pt, "dim"):
                        src = (pt[0] if pt.dim() > 1 else pt).to(torch.int64)
                self._restage(dev, src)
        # 4. the fused trace's own flat + per-layer tables
        tr = getattr(self, "_fused_trace", None)
        if not tr:
            return
        self._restage(tr.get("d_pt"), flat)
        self._restage(tr.get("v_pt"), flat)
        for key in ("d_ptl", "v_ptl", "pv_ptl"):
            lst = tr.get(key)
            if not lst:
                continue
            seen = set()
            for i, dev in enumerate(lst):
                if dev is None or id(dev) in seen:
                    continue
                seen.add(id(dev))
                src = flat
                if installed and i < len(installed):
                    pt = installed[i]
                    if pt is not None and hasattr(pt, "dim"):
                        src = (pt[0] if pt.dim() > 1 else pt).to(torch.int64)
                self._restage(dev, src)

    def _page_tables_per_layer(self, batch, user_idx=0):
        """Per-layer page tables replicated to ``batch`` verify rows (bounded KV).

        Under bounded sliding the target runs HYBRID per-layer page tables:
        sliding layers address a small bounded ring, full layers the full pool.
        The model stashes the batch-1 set in ``_active_page_tables_per_layer``,
        but the speculative verify puts its K+1 candidates in the BATCH dim, so
        each layer's table needs that user's row replicated across the
        candidates -- the same batch-alias trick ``_page_table`` uses for the
        flat table. Without this the per-candidate KV write slices a row the
        table does not have.

        Returns ``None`` when the target is not running per-layer tables, which
        keeps the unbounded path byte-identical.
        """
        installed = getattr(self.target, "_active_page_tables_per_layer", None)
        if not installed:
            return None
        cache = getattr(self, "_pt_layer_cache", None)
        if cache is None:
            cache = {}
            self._pt_layer_cache = cache
        key = (batch, user_idx, id(installed))
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = []
        for pt in installed:
            if pt is None:
                out.append(None)
                continue
            t = pt if hasattr(pt, "dim") else None
            if t is None:  # already a device tensor: leave it to the model
                out.append(pt)
                continue
            row = t[user_idx : user_idx + 1] if t.dim() > 1 else t.unsqueeze(0)
            out.append(row.repeat(batch, 1).to(torch.int32))
        cache[key] = out
        return out

    def _page_table_users(self, B, width=None):
        # Distinct per-user page-table rows [B, blocks] for a true B-user batched
        # forward (each user attends to its OWN physical KV blocks). Requires
        # page_table_torch to have >= B rows.
        #
        # ``width`` trims the column count for the PACKED verify: its non-causal
        # SDPA attends over the TABLE width, so columns past the mask's S_k are
        # effectively unmasked -- the unwritten tail dilutes softmax (measured:
        # the packed path drifted off greedy; width-matching restored exact
        # greedy and +0.07 accepted/iter single-user).
        pt = self.page_table_torch[:B]
        if width is not None:
            pt = pt[:, : min(int(width), int(pt.shape[1]))]
        pt = pt.to(torch.int32)
        return ttnn.from_torch(
            pt, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=self._mapper
        )

    def _page_tables_per_layer_users(self, B):
        """Per-layer page tables with DISTINCT rows for B real users (bounded KV).

        The batch-alias variant (``_page_tables_per_layer``) replicates ONE user's
        row across the candidate dim -- right for single-user verify, wrong for a
        true B-user forward where each user owns its own blocks. Returns ``None``
        when no per-layer tables are installed (unbounded path unchanged).
        """
        installed = getattr(self.target, "_active_page_tables_per_layer", None)
        if not installed:
            return None
        cache = getattr(self, "_pt_layer_users_cache", None)
        if cache is None:
            cache = {}
            self._pt_layer_users_cache = cache
        key = (B, id(installed))
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = []
        for pt in installed:
            if pt is None or not hasattr(pt, "dim"):
                out.append(pt)  # None, or an already-device tensor the model owns
                continue
            rows = pt if pt.dim() > 1 else pt.unsqueeze(0)
            out.append(rows[:B].to(torch.int32))
        cache[key] = out
        return out

    def _from_b(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, device=self.mesh_device, layout=layout, dtype=dtype, mesh_mapper=self._mapper)

    # ── host-side input tensors (for copy_host_to_device_tensor into traces) ──
    def _host_tokens(self, tokens):
        # int64 host source for the uint32 tensor (see _pos_tensors): avoids the
        # int32->uint32 conversion that triggers the #18536 row-major warning.
        t = torch.tensor(tokens, dtype=torch.int64).reshape(1, len(tokens))
        return ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)

    def _host_pos(self, positions):
        batch = len(positions)
        # int64 host source for the uint32 tensor (see _pos_tensors): avoids the
        # int32->uint32 conversion that triggers the #18536 row-major warning.
        pu = torch.zeros((1, 32), dtype=torch.int64)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int64)
        pos_uint32 = ttnn.from_torch(pu, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)
        pos_int32 = ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        return pos_uint32, pos_int32

    # ── traced verify ────────────────────────────────────────────────────────
    def _capture_verify_trace(self, tokens, positions):
        """Lazily capture a verify trace at the REAL first-call inputs.

        Persistent device inputs (x, pos_u, pos_i, page_table) are allocated once
        so trace capture binds stable addresses; later calls overwrite their
        contents via ``copy_host_to_device_tensor`` and ``execute_trace``. The
        compile + capture runs write the SAME KV positions with the SAME tokens
        as the first real call, so the KV writes are idempotent (no corruption).
        """
        from loguru import logger as _lg

        batch = len(tokens)
        _lg.info(f"[spec-trace] capture verify batch={batch}: building inputs")
        x_dev = self._tokens_tensor(tokens)
        pu_dev, pi_dev = self._pos_tensors(positions)
        pt_dev = self._page_table(batch)
        # Compile run (warm program cache before capture).
        _lg.info(f"[spec-trace] capture verify batch={batch}: compile run")
        logits, hidden = self.target.ttnn_verify_forward(
            x=x_dev,
            current_pos=pu_dev,
            current_pos_cache=pi_dev,
            page_table=pt_dev,
            kv_cache=self.tt_kv_cache,
            page_tables_per_layer=self._page_tables_per_layer(batch),
        )
        ttnn.synchronize_device(self.mesh_device)
        logits.deallocate(True)
        hidden.deallocate(True)
        _lg.info(f"[spec-trace] capture verify batch={batch}: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits, hidden = self.target.ttnn_verify_forward(
            x=x_dev,
            current_pos=pu_dev,
            current_pos_cache=pi_dev,
            page_table=pt_dev,
            kv_cache=self.tt_kv_cache,
            # Must match the compile run above: falling through to the model's
            # batch-1 ``_active_page_tables_per_layer`` under bounded sliding both
            # binds the wrong rows and allocates inside trace capture.
            page_tables_per_layer=self._page_tables_per_layer(batch),
        )
        _lg.info(f"[spec-trace] capture verify batch={batch}: end_trace_capture")
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info(f"[spec-trace] capture verify batch={batch}: DONE")
        self._verify_traces[batch] = {
            "id": tid,
            "x": x_dev,
            "pu": pu_dev,
            "pi": pi_dev,
            "pt": pt_dev,
            "logits": logits,
            "hidden": hidden,
        }

    def _verify_traced(self, tokens, positions):
        batch = len(tokens)
        if batch not in self._verify_traces:
            self._capture_verify_trace(tokens, positions)
        from loguru import logger as _lg

        tr = self._verify_traces[batch]
        # Overwrite persistent inputs (page_table row is constant -> write once at
        # capture). x/positions change every call.
        h_x = self._host_tokens(tokens)
        h_pu, h_pi = self._host_pos(positions)
        ttnn.copy_host_to_device_tensor(h_x, tr["x"])
        ttnn.copy_host_to_device_tensor(h_pu, tr["pu"])
        ttnn.copy_host_to_device_tensor(h_pi, tr["pi"])
        _lg.info(f"[spec-trace] verify replay batch={batch}: execute")
        ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)
        _lg.info(f"[spec-trace] verify replay batch={batch}: read logits")
        lh = self._logits_to_host(tr["logits"]).reshape(batch, -1)
        # Return the PERSISTENT hidden (no clone). It is overwritten on the next
        # execute_trace, so the caller MUST consume it before the next verify and
        # MUST NOT deallocate it (allocation-free loop invariant). seed() copies
        # it into self._anchor_buf immediately; generate() ignores it in the
        # default reseed mode.
        return lh, tr["hidden"]

    def _logits_to_host(self, logits):
        if self._tp > 1:
            t = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            t = ttnn.to_torch(logits)
        return t[..., : self.target.vocab_size]

    # ── packed-query verify ──────────────────────────────────────────────
    def _pv_setup(self):
        """Lazy one-time setup for the packed verify: allocate per-layer hot-
        block staging buffers (non-KV-shared layers only) and stash the shape
        constants the host-side index/mask builders need."""
        if self._pv_ready:
            return
        from models.demos.gemma4.tt.attention.kv_cache import PV_HOT_BLOCKS, init_kv_staging

        target = self.target
        # Paged cache shape: [max_blocks, nkv_local, block_size, head_dim].
        self._pv_bs = int(self.tt_kv_cache[0][0].padded_shape[2])
        self._pv_blk = PV_HOT_BLOCKS
        self._pv_s2 = self._pv_blk * self._pv_bs
        tp = self._tp
        first_cfg = target.layers[0].self_attn.config
        self._pv_h_local = first_cfg.num_attention_heads // tp
        self._pv_window = target.hf_config.sliding_window
        self._pv_nkv = {}  # layer_type -> nkv_local (staging/cache head count)
        # BOUNDED sliding support: a sliding layer's cache is a RING of
        # ``cache_position_modulo`` slots, so its hot-block index is
        # (c % ring) // bs (wrapping), its page-table row is the small ring
        # pool, and its SDPA mask spans ring slots rather than absolute
        # positions. Track ring + page row PER LAYER TYPE; the unbounded path
        # keeps ring=None and the flat table for both types (byte-identical).
        self._pv_ring = {}  # layer_type -> ring modulo (int) or None
        _ring_rep = {}  # layer_type -> a representative layer index for its pool
        for i, layer in enumerate(target.layers):
            cfg = layer.self_attn.config
            lt = target.hf_config.layer_types[i]
            mod = getattr(cfg, "cache_position_modulo", None)
            if lt in self._pv_ring and self._pv_ring[lt] != (int(mod) if mod is not None else None):
                # One mask + one hot table serve a whole TYPE; a mixed type
                # (e.g. the drafter-layer exemption putting the last sliding
                # layer on the full pool while its siblings ring) cannot be
                # expressed. Run with GEMMA4_SPEC_UNBOUNDED_DRAFTER_LAYER=0.
                raise NotImplementedError(
                    f"packed verify needs uniform pools per layer type; layer {i} ({lt}) has "
                    f"cache_position_modulo={mod} but its type resolved {self._pv_ring[lt]}. "
                    "Disable the drafter-layer exemption (GEMMA4_SPEC_UNBOUNDED_DRAFTER_LAYER=0)."
                )
            self._pv_ring.setdefault(lt, int(mod) if mod is not None else None)
            _ring_rep.setdefault(lt, i)
            self._pv_ring_rep = _ring_rep
            if i in target.kv_shared_layer_map:
                continue  # shares source layer's cache+staging; skips KV writes
            # GEMMA4_PV_FALLBACK_WRITE=1: debug bisect -- skip staging so
            # packed_decode_forward takes the per-position paged_update_cache
            # loop (the write mechanism decode_forward already proves correct),
            # isolating read-side (mask/page-table) defects from staging ones.
            if os.environ.get("GEMMA4_PV_FALLBACK_WRITE") == "1":
                layer.self_attn.kv_staging = None
                nkv_local = 1 if layer.self_attn.weights.kv_replicated else cfg.num_key_value_heads // tp
                self._pv_nkv[lt] = int(nkv_local)
                continue
            # Reuse an existing staging buffer if one is already attached to this
            # layer (serving re-runs _pv_setup per request via a fresh decoder;
            # the buffer is request-independent scratch of constant shape, so
            # reallocating it every request would orphan the previous allocation
            # and exhaust DRAM -- crash on the 2nd serving request). Allocate once.
            existing = getattr(layer.self_attn, "kv_staging", None)
            if existing is None or (hasattr(existing[0], "is_allocated") and not existing[0].is_allocated()):
                layer.self_attn.kv_staging = init_kv_staging(
                    self.mesh_device,
                    cfg,
                    max_batch_size=1,
                    block_size=self._pv_bs,
                    blk=self._pv_blk,
                    mesh_config=getattr(layer.self_attn, "mesh_config", None),
                )
            self._pv_nkv[lt] = int(layer.self_attn.kv_staging[0].shape[1])
        # Per-type EFFECTIVE block size: the token count one page-table column
        # spans for a layer of that type, i.e. what its paged_update_cache and
        # paged SDPA use (effective_block_size inverts the HMA byte-unified
        # view). Under vLLM hybrid groups the sliding and full pools differ
        # (12B: 256 vs 128 tokens); ``_pv_bs`` above is layer 0's physical
        # block dim and serves the single-pool demo paths only.
        from models.demos.gemma4.tt.attention.operations import effective_block_size

        self._pv_bs_t = {}
        for lt, idx in _ring_rep.items():
            layer = target.layers[idx]
            cfg = layer.self_attn.config
            nkv_local = 1 if layer.self_attn.weights.kv_replicated else cfg.num_key_value_heads // tp
            self._pv_bs_t[lt] = int(effective_block_size(self.tt_kv_cache[idx][0], cfg.head_dim, nkv_local))
        flat = (self.page_table_torch[0] if self.page_table_torch.dim() > 1 else self.page_table_torch).to(torch.int64)
        self._pv_pages = flat
        # Per-type page rows for the demo path: its build_hybrid_page_tables
        # gives every layer of a type the same block IDs, so one row per type
        # addresses every layer of that type (the flat ``_pv_pages`` makes the
        # same assumption for the unbounded case). NOT true of vLLM's hybrid
        # groups, which the serving records handle per group slot
        # (serving_kv_group_slots).
        self._pv_pages_t = {lt: flat for lt in self._pv_ring}
        installed = getattr(target, "_active_page_tables_per_layer", None)
        if installed:
            # Use each type's REPRESENTATIVE layer (uniform-pool guard above),
            # not last_kv_layer_by_type: the last sliding layer is exactly the
            # one a legacy exemption would put on the wrong pool.
            for lt, idx in _ring_rep.items():
                pt = installed[idx] if idx < len(installed) else None
                if pt is not None and hasattr(pt, "dim"):
                    row = pt[0] if pt.dim() > 1 else pt
                    self._pv_pages_t[lt] = row.to(torch.int64)
        self._pv_ready = True

    def _pv_tables_per_layer(self, S_k):
        """Per-layer page tables for the packed verify, each trimmed so its
        WIDTH matches its layer type's mask width.

        The packed non-causal SDPA attends over the PAGE-TABLE width, not the
        mask width: columns past the mask are effectively unmasked. Probe
        (pv_sdpa_probe): ring mask + exact 32-entry table matches unbounded to
        bf16 noise; the same mask over a zero-padded full-width table is
        rel-2.17 wrong -- the padded tail aliases physical page 0's REAL K/V.
        (Unbounded suffers a milder form: the un-written tail dilutes softmax
        with zero-K/V columns -- the packed path's small greedy divergence.)

        Sliding (ring) width is fixed; full width follows the S_k bucket.
        Device tensors cached by (type, width); layers of a type share one.
        """
        cache = getattr(self, "_pv_pt_cache", None)
        if cache is None:
            cache = {}
            self._pv_pt_cache = cache
        per_layer = []
        for lt in self.target.hf_config.layer_types[: len(self.target.layers)]:
            ring = self._pv_ring.get(lt)
            width = (ring // self._pv_bs) if ring else max(1, S_k // self._pv_bs)
            row = self._pv_pages_t[lt]
            width = min(width, int(row.shape[0]))
            key = (lt, width)
            t = cache.get(key)
            if t is None:
                t = self._pv_from_torch(row[:width].to(torch.int32).reshape(1, width), ttnn.int32)
                cache[key] = t
            per_layer.append(t)
        return per_layer

    def _pv_block(self, c, lt):
        """Hot-block index at position ``c`` for a layer type: absolute for an
        unbounded pool, wrapped into the ring for a bounded one."""
        ring = self._pv_ring.get(lt)
        slot = c % ring if ring else c
        return slot // self._pv_bs

    def _pv_nblocks(self, lt):
        """Valid blocks in this type's pool (ring blocks when bounded)."""
        ring = self._pv_ring.get(lt)
        if ring:
            return ring // self._pv_bs
        return int(self._pv_pages_t[lt].shape[0])

    def _pv_seed_staging(self, c):
        """Seed every layer's staging block-slot 0 with the committed content of
        the hot block at position ``c`` (read from the cache — the only
        committed-cache read in the design, once per generation)."""
        bs, S2 = self._pv_bs, self._pv_s2
        # prev tracks the ABSOLUTE block; per-type (possibly ring-wrapped)
        # indices derive from it in _pv_host_inputs / here.
        self._pv_a_prev = c // bs
        if c % bs == 0:
            return  # fresh block — zeros are fine
        for i, layer in enumerate(self.target.layers):
            if i in self.target.kv_shared_layer_map:
                continue
            lt = self.target.hf_config.layer_types[i]
            blk_phys = int(self._pv_pages_t[lt][self._pv_block(c, lt)])
            staging = layer.self_attn.kv_staging
            if staging is None:
                continue  # GEMMA4_PV_FALLBACK_WRITE: per-position writes, no staging
            cache = self.tt_kv_cache[i]
            for kv in (0, 1):
                stg = staging[kv]
                nkv, hd = stg.shape[1], stg.shape[3]
                block = ttnn.slice(cache[kv], [blk_phys, 0, 0, 0], [blk_phys + 1, nkv, bs, hd])
                tail = ttnn.slice(stg, [0, 0, bs, 0], [1, nkv, S2, hd])
                rebuilt = ttnn.concat([block, tail], dim=2)
                ttnn.assign(rebuilt, stg)
                for t in (rebuilt, block, tail):
                    ttnn.deallocate(t)

    def _pv_host_inputs(self, c, P, s_k=None, h_repeat=True):
        """Host tensors for one packed verify at anchor position ``c``.

        Returns dict of torch tensors: pos [1,P]u32, masks [1,1,H*P,S_k] bf16
        (full + sliding), embed_idx per layer type [1,nkv*S2]u32, hot_pt
        [1,BLK]i32, and S_k.

        ``s_k`` overrides the full mask's bucketed width -- the fused trace
        fixes S_k at capture to cover the whole generation so mask/table widths
        (and hence the captured program) stay stable across replays. Columns
        beyond the current ``c+P-1`` are NEG by construction, so a wider mask is
        exact, not diluted.
        """
        bs, S2, BLK = self._pv_bs, self._pv_s2, self._pv_blk
        a, off = c // bs, c % bs
        roll = 1 if a == self._pv_a_prev + 1 else 0
        H, W = self._pv_h_local, self._pv_window

        pos = torch.arange(c, c + P, dtype=torch.int32).reshape(1, P)

        # Additive masks, head-major rows h*P+p: causal upper bound c+p; sliding
        # adds the window lower bound. S_k is bucket-padded for trace stability.
        NEG = -1e9
        S_k = s_k if s_k else ((c + P + self._pv_sk_bucket - 1) // self._pv_sk_bucket) * self._pv_sk_bucket
        j = torch.arange(S_k)
        rows_full = torch.empty(P, S_k)
        for p in range(P):
            upper = c + p
            rows_full[p] = torch.where(j <= upper, 0.0, NEG)
        _rep = H if h_repeat else 1
        mask_full = rows_full.repeat(_rep, 1).reshape(1, 1, _rep * P, S_k).to(torch.bfloat16)

        ring = self._pv_ring.get("sliding_attention")
        if ring:
            # BOUNDED sliding: the mask spans RING SLOTS, not absolute
            # positions. After the hot fill everything up to c+P-1 is written,
            # so slot j holds pos_j = (c+P-1) - ((c+P-1-j) mod ring). Query p
            # (absolute q = c+p) may attend slot j iff q-W < pos_j <= q AND
            # pos_j >= 0 (early context: the ring's untouched tail is zeros,
            # never valid keys). In distance form with d = (c+P-1-j) mod ring:
            #   (P-1-p) <= d < (P-1-p)+W   and   d <= c+P-1.
            # Slots holding FUTURE candidates (pos_j > q) fall below the lower
            # bound and are masked, exactly like j > upper absolute-side.
            S_r = ring  # ring is a multiple of 64 (power of two >= 1024)
            jr = torch.arange(S_r)
            rows_slide = torch.empty(P, S_r)
            top = c + P - 1
            d = torch.remainder(top - jr, ring)
            for p in range(P):
                lo = P - 1 - p
                ok = (d >= lo) & (d < lo + W) & (d <= top)
                rows_slide[p] = torch.where(ok, 0.0, NEG)
            mask_slide = rows_slide.repeat(_rep, 1).reshape(1, 1, _rep * P, S_r).to(torch.bfloat16)
        else:
            rows_slide = torch.empty(P, S_k)
            for p in range(P):
                upper = c + p
                rows_slide[p] = torch.where((j <= upper) & (j > upper - W), 0.0, NEG)
            mask_slide = rows_slide.repeat(_rep, 1).reshape(1, 1, _rep * P, S_k).to(torch.bfloat16)

        # merge_idx over staging positions: committed prefix from staging
        # (identity, or +bs on a rollover — the prefix came from the spill
        # block), the P new rows from new_seq (concat index S2+p), stale tail
        # identity. embed_idx bakes the per-head flattened row offset in
        # (new_seq is padded to 32 rows in packed_decode_forward).
        m = torch.arange(S2, dtype=torch.int64)
        if roll:
            m[:off] += bs
        m[off : off + P] = S2 + torch.arange(P)
        p_pad = ((P + 31) // 32) * 32
        src_seq = S2 + p_pad
        embed = {}
        for lt, nkv in self._pv_nkv.items():
            off_h = (torch.arange(nkv, dtype=torch.int64) * src_seq).unsqueeze(1)
            embed[lt] = (m.unsqueeze(0) + off_h).reshape(1, nkv * S2).to(torch.int32)

        # -1, not 0: ``hot_pt`` feeds ``paged_fill_cache`` with no ``valid_seq_len``
        # cap (see attention/decode.py ``_packed_fill_kv_loopfree_embed``), so its
        # contract is "-1 = skip". Metal reserves no null page, so a 0 here makes
        # every idle slot write staging KV into physical page 0 and corrupts the
        # committed cache. Chunked prefill can pad with 0 because valid_seq_len
        # bounds that fill; this path has no such bound.
        # PER LAYER TYPE: a bounded (ring) pool wraps both the hot block index
        # and its spill successor; an unbounded pool uses absolute indices. The
        # unbounded/unbounded case yields two identical tensors — harmless.
        hot_t = {}
        for lt in self._pv_nkv:
            nblk = self._pv_nblocks(lt)
            a_t = self._pv_block(c, lt)
            h = torch.full((1, BLK), -1, dtype=torch.int32)
            h[0, 0] = int(self._pv_pages_t[lt][a_t])
            if off + P > bs:
                h[0, 1] = int(self._pv_pages_t[lt][(a_t + 1) % nblk if self._pv_ring.get(lt) else a_t + 1])
            hot_t[lt] = h

        return {
            "pos": pos,
            "mask_full": mask_full,
            "mask_slide": mask_slide,
            "embed": embed,
            "hot_t": hot_t,
            "S_k": S_k,
        }

    def _pv_from_torch(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
        return ttnn.from_torch(
            t,
            device=self.mesh_device if device else None,
            layout=layout,
            dtype=dtype,
            mesh_mapper=self._mapper,
        )

    def _pv_device_inputs(self, tokens, h):
        """Device tensors for one packed verify from host dict ``h``."""
        return {
            "x": self._tokens_tensor(tokens),
            "pos": self._pv_from_torch(h["pos"], ttnn.uint32),
            "mask_full": self._pv_from_torch(h["mask_full"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "mask_slide": self._pv_from_torch(h["mask_slide"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "embed": {lt: self._pv_from_torch(e, ttnn.uint32) for lt, e in h["embed"].items()},
            "hot_t": {lt: self._pv_from_torch(t, ttnn.int32) for lt, t in h["hot_t"].items()},
            "S_k": h["S_k"],
        }

    def _pv_call(self, dev, P):
        kv_write_idxs = None
        if os.environ.get("GEMMA4_PV_FALLBACK_WRITE") == "1":
            # Debug bisect: per-position write positions for the fallback loop
            # (paged_update_cache wraps them itself via cache_position_modulo).
            c = int(dev["c"])
            kv_write_idxs = [
                self._pv_from_torch(torch.tensor([c + p], dtype=torch.int32), ttnn.int32) for p in range(P)
            ]
        return self.target.ttnn_packed_verify_forward(
            x=dev["x"],
            position_idx=dev["pos"],
            attn_mask_full=dev["mask_full"],
            attn_mask_sliding=dev["mask_slide"],
            packed_p=P,
            page_table=dev.get("pt"),
            kv_cache=self.tt_kv_cache,
            kv_write_idxs=kv_write_idxs,
            embed_idx_full=dev["embed"].get("full_attention"),
            embed_idx_sliding=dev["embed"].get("sliding_attention"),
            hot_pt_full=dev["hot_t"].get("full_attention"),
            hot_pt_sliding=dev["hot_t"].get("sliding_attention"),
            # Width-matched per-layer tables (see _pv_tables_per_layer): the
            # packed SDPA attends the TABLE width, so each layer's table must be
            # exactly as wide as its type's mask. Applies unbounded too -- the
            # full-width flat table's unwritten tail diluted softmax.
            page_tables_per_layer=self._pv_tables_per_layer(dev["S_k"]),
        )

    def _verify_packed(self, tokens, positions):
        """Packed verify (eager). Same contract as ``_verify``."""
        self._pv_setup()
        P = len(tokens)
        c = positions[0]
        if self._pv_a_prev < 0:
            self._pv_seed_staging(c)
        h = self._pv_host_inputs(c, P)
        dev = self._pv_device_inputs(tokens, h)
        dev["pt"] = self._page_table(1)
        dev["c"] = c
        logits, hidden = self._pv_call(dev, P)
        self._pv_a_prev = c // self._pv_bs
        lh = self._logits_to_host(logits).reshape(P, -1)
        logits.deallocate(True)
        for t in (dev["x"], dev["pos"], dev["mask_full"], dev["mask_slide"], dev["pt"]):
            t.deallocate(True)
        for e in dev["embed"].values():
            e.deallocate(True)
        for t in dev["hot_t"].values():
            t.deallocate(True)
        return lh, hidden

    def _verify_packed_traced(self, tokens, positions):
        """Packed verify via trace, keyed by (P, S_k bucket). Persistent device
        inputs are refreshed per step via copy_host_to_device_tensor; a new
        trace is captured lazily when P or the S_k bucket changes."""
        self._pv_setup()
        P = len(tokens)
        c = positions[0]
        if self._pv_a_prev < 0:
            self._pv_seed_staging(c)
        h = self._pv_host_inputs(c, P)
        key = (P, h["S_k"])
        tr = self._pv_traces.get(key)
        if tr is None:
            dev = self._pv_device_inputs(tokens, h)
            dev["pt"] = self._page_table(1)
            # Compile run (warm program cache), then capture. Both runs write
            # the SAME tokens to the SAME positions, so KV writes are idempotent.
            logits, hidden = self._pv_call(dev, P)
            ttnn.synchronize_device(self.mesh_device)
            logits.deallocate(True)
            hidden.deallocate(True)
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            logits, hidden = self._pv_call(dev, P)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            dev.update({"id": tid, "logits": logits, "hidden": hidden})
            self._pv_traces[key] = dev
        else:
            ttnn.copy_host_to_device_tensor(self._host_tokens(tokens), tr["x"])
            for src, dst in (
                (self._pv_from_torch(h["pos"], ttnn.uint32, device=False), tr["pos"]),
                (
                    self._pv_from_torch(h["mask_full"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=False),
                    tr["mask_full"],
                ),
                (
                    self._pv_from_torch(h["mask_slide"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=False),
                    tr["mask_slide"],
                ),
            ):
                ttnn.copy_host_to_device_tensor(src, dst)
            for lt, e in h["embed"].items():
                ttnn.copy_host_to_device_tensor(self._pv_from_torch(e, ttnn.uint32, device=False), tr["embed"][lt])
            for lt, t in h["hot_t"].items():
                ttnn.copy_host_to_device_tensor(self._pv_from_torch(t, ttnn.int32, device=False), tr["hot_t"][lt])
            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)
        tr = self._pv_traces[key]
        self._pv_a_prev = c // self._pv_bs
        lh = self._logits_to_host(tr["logits"]).reshape(P, -1)
        # Persistent hidden — caller must consume before the next replay.
        return lh, tr["hidden"]

    # ── target forwards ───────────────────────────────────────────────────
    def _verify(self, tokens, positions):
        """Batched verify. Returns (logits_host [B,vocab], hidden_device [1,1,B,h])."""
        # Packed-query verify: all K+1 candidates in one batch=1 pass (positions
        # packed into the query-heads dim, loop-free staging KV write).
        # Single-token calls (seed/reseed) keep the plain verify.
        if len(tokens) > 1:
            if self._use_trace:
                return self._verify_packed_traced(tokens, positions)
            return self._verify_packed(tokens, positions)
        # Tracing: BOTH the batch=1 verify (seed/reseed) and the batch=K+1
        # speculative verify capture + replay correctly. The per-candidate
        # sequential_kv_write loop (the race fix for the shared paged block) is
        # fixed-count and fully trace-compatible — positions/tokens are bound as
        # data via copy_host_to_device_tensor, so a single trace per batch size
        # is reused across iterations as the anchor position grows. Traces are
        # keyed by batch (K+1 and 1), captured lazily on first call.
        if self._use_trace:
            return self._verify_traced(tokens, positions)
        x = self._tokens_tensor(tokens)
        pos_u, pos_i = self._pos_tensors(positions)
        pt = self._page_table(len(tokens))
        logits, hidden = self.target.ttnn_verify_forward(
            x=x,
            current_pos=pos_u,
            current_pos_cache=pos_i,
            page_table=pt,
            kv_cache=self.tt_kv_cache,
            page_tables_per_layer=self._page_tables_per_layer(len(tokens)),
        )
        lh = self._logits_to_host(logits).reshape(len(tokens), -1)
        logits.deallocate(True)
        for t in (x, pos_u, pos_i, pt):
            t.deallocate(True)
        return lh, hidden

    # ── traced drafting ──────────────────────────────────────────────────────
    def _capture_draft_trace(self, anchor_token, anchor_hidden, anchor_pos):
        """Capture ONE batch=1 drafter-step trace (replayed K times per iter).

        Persistent inputs ``tok_in`` (last token), ``h_in`` (recurrent hidden),
        ``pos_u/pos_i`` (fixed anchor position) and ``pt`` (page table) are
        allocated once. ``logits`` and ``h_next`` are persistent trace OUTPUTS.
        The recurrence copy ``h_next -> h_in`` is done BETWEEN replays (outside
        the trace) via ttnn.copy — an in-trace copy fails capture with "Writes
        not supported during trace capture".
        """
        from loguru import logger as _lg

        tok_in = self._tokens_tensor([anchor_token])
        h_in = ttnn.clone(anchor_hidden)
        pos_u, pos_i = self._pos_tensors([anchor_pos])
        pt = self._page_table(1)
        page_tables = self._shared_kv_page_tables(pt)
        _lg.info("[spec-trace] capture draft step: compile run")
        logits, h_next = self.assistant.step(tok_in, h_in, self._shared_kv, page_tables, pos_u, pos_i)
        ttnn.synchronize_device(self.mesh_device)
        logits.deallocate(True)
        h_next.deallocate(True)
        _lg.info("[spec-trace] capture draft step: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits, h_next = self.assistant.step(tok_in, h_in, self._shared_kv, page_tables, pos_u, pos_i)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info("[spec-trace] capture draft step: DONE")
        self._draft_trace = {
            "id": tid,
            "tok": tok_in,
            "h": h_in,
            "h_next": h_next,
            "pu": pos_u,
            "pi": pos_i,
            "pt": pt,
            "logits": logits,
        }

    def _draft_traced(self, anchor_token, anchor_hidden, anchor_pos, temperature=0.0, top_p=1.0, top_k=0):
        """Traced drafting: replay the batch=1 step trace K times. See _draft."""
        K = self.draft_len
        greedy = not temperature or temperature <= 0
        if self._draft_trace is None:
            self._capture_draft_trace(anchor_token, anchor_hidden, anchor_pos)
        tr = self._draft_trace
        # Per-iteration setup: position (anchor moves between iters), recurrent
        # hidden reset to this iter's seed, and the first token.
        h_pu, h_pi = self._host_pos([anchor_pos])
        ttnn.copy_host_to_device_tensor(h_pu, tr["pu"])
        ttnn.copy_host_to_device_tensor(h_pi, tr["pi"])
        ttnn.copy(anchor_hidden, tr["h"])  # device->device reset of recurrent buffer
        ttnn.copy_host_to_device_tensor(self._host_tokens([anchor_token]), tr["tok"])

        from loguru import logger as _lg

        drafts, draft_logits = [], []
        for step in range(K):
            _lg.info(f"[spec-trace] draft replay step {step}: execute")
            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)
            # Feed the recurrence: next step reads h_in, which the trace does NOT
            # update (in-trace copy is illegal). Copy the fresh output h_next into
            # h_in on-device (no allocation), queued after execute_trace.
            if step < K - 1:
                _lg.info(f"[spec-trace] draft replay step {step}: copy h_next->h")
                ttnn.copy(tr["h_next"], tr["h"])
            _lg.info(f"[spec-trace] draft replay step {step}: read logits")
            lh = self._logits_to_host(tr["logits"]).reshape(-1)
            draft_logits.append(lh)
            if greedy:
                tok = int(torch.argmax(lh))
            else:
                q = _to_probs(lh, temperature, top_p, top_k)
                tok = int(torch.multinomial(q, 1))
            drafts.append(tok)
            if step < K - 1:
                ttnn.copy_host_to_device_tensor(self._host_tokens([tok]), tr["tok"])
        return drafts, draft_logits

    # ── drafting ────────────────────────────────────────────────────────────
    def _draft(self, anchor_token, anchor_hidden, anchor_pos, temperature=0.0, top_p=1.0, top_k=0, user_idx=0):
        """Produce K draft tokens + their logits (host). anchor_hidden stays on device.

        Greedy (temperature<=0): the next draft is argmax(q). Sampling: the next
        draft is SAMPLED from the drafter distribution q under the SAME
        temp/top_p/top_k as the target — required for speculative sampling
        correctness AND acceptance (proposing argmax(q) maximizes q(d), which
        drives the accept ratio min(1, p(d)/q(d)) to ~0).

        ``user_idx`` selects which user's KV blocks the drafter cross-attends to
        (batched generate loops the drafter at batch=1 over each user)."""
        K = self.draft_len
        greedy = not temperature or temperature <= 0
        pos_u, pos_i = self._pos_tensors([anchor_pos])
        # Drafter page table: batch=1, this user's blocks for both layer types.
        pt = self._page_table(1, user_idx=user_idx)
        page_tables = self._shared_kv_page_tables(pt)

        drafts, draft_logits = [], []
        tok = anchor_token
        h_rec = anchor_hidden  # borrowed; do not deallocate the caller's anchor_hidden
        owns_h = False
        for _ in range(K):
            tok_tt = self._tokens_tensor([tok])
            logits_d, h_next = self.assistant.step(tok_tt, h_rec, self._shared_kv, page_tables, pos_u, pos_i)
            tok_tt.deallocate(True)
            lh = self._logits_to_host(logits_d).reshape(-1)
            logits_d.deallocate(True)
            draft_logits.append(lh)
            if greedy:
                tok = int(torch.argmax(lh))
            else:
                q = _to_probs(lh, temperature, top_p, top_k)
                tok = int(torch.multinomial(q, 1))
            drafts.append(tok)
            if owns_h:
                h_rec.deallocate(True)
            h_rec, owns_h = h_next, True
        if owns_h:
            h_rec.deallocate(True)
        pos_u.deallocate(True)
        pos_i.deallocate(True)
        pt.deallocate(True)
        return drafts, draft_logits

    # ── fully on-device fused iteration (greedy) ─────────────────────────────
    def _argmax_last(self, logits, rows):
        """argmax over the last (vocab) dim — fast path. Returns [1,1,rows] uint32.

        ``ttnn.argmax`` requires ROW_MAJOR input; passing a TILE tensor takes a
        single-core internal-untilize path that is catastrophically slow on a
        262144-wide vocab (~9 ms for 1 row, ~28 ms for 5). The multicore argmax
        is fast (~1.6 ms) but ROW-PARALLEL: it returns GARBAGE unless the row
        (batch) dim is EXACTLY one tile (32) — verified by a correctness probe
        (1/5 rows -> wrong; padded to 32 -> exact; and >32 rows in one call also
        returns garbage beyond the first tile). So process the rows in 32-row
        chunks: pad each ≤32-row chunk up to 32, run the multicore untilize +
        argmax, slice back, and concat. Net ~1.6 ms/chunk vs ~9-28 ms bare.
        """
        R32 = 32
        if rows > R32:
            # Batched packed verify (B*P > 32): argmax each 32-row tile separately
            # (offsets are multiples of 32, so the TILE slices are tile-aligned)
            # and concat — a single >32-row multicore argmax garbles later tiles.
            vocab = logits.shape[-1]
            chunks = []
            off = 0
            while off < rows:
                n = min(R32, rows - off)
                part = ttnn.slice(logits, [0, 0, off, 0], [1, 1, off + n, vocab])
                chunks.append(self._argmax_last(part, n))  # recurse (≤32 rows)
                if not _same_device_buffer(part, logits):
                    # a slice that ttnn served as a view of ``logits`` must not free it
                    part.deallocate(True)
                off += n
            out = ttnn.concat(chunks, dim=2)  # [1,1,rows]
            for c in chunks:
                c.deallocate(True)
            return out
        src = logits
        padded = None
        if rows < R32:
            padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, R32 - rows), (0, 0)], value=0.0)
            src = padded
        u = ttnn.untilize(src, use_multicore=True)
        if padded is not None and not _same_device_buffer(padded, logits):
            # ttnn.pad returns its input when the TILE padding already covers the
            # target rows; freeing that would free the caller's logits.
            padded.deallocate(True)
        idx = ttnn.argmax(
            u, dim=-1, keepdim=False
        )  # [1,1,32 or rows] uint32 RM (multicore by default on ROW_MAJOR last-dim)
        u.deallocate(True)
        if rows < R32:
            sliced = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])  # [1,1,rows]
            if not _same_device_buffer(sliced, idx):
                # the trimmed ids are a new buffer from prim::slice today; a ttnn that
                # serves the trim as a view of ``idx`` must keep ``idx`` alive
                idx.deallocate(True)
            idx = sliced
        return idx

    def _id_to_host(self, id_tt):
        """[*,1] uint32 device id -> python int (TP: read device-0 replica)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(id_tt)[0]) if self._tp > 1 else ttnn.to_torch(id_tt)
        return int(t.reshape(-1)[0])

    def _ids_to_host(self, ids_tt, n):
        """[1,1,n] uint32 device ids -> list[int] (TP: read device-0 replica)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(ids_tt)[0]) if self._tp > 1 else ttnn.to_torch(ids_tt)
        flat = t.reshape(-1)
        return [int(flat[j]) for j in range(n)]

    def _fused_iter(self, anchor_tok_tt, anchor_hidden, anchor_pos):
        """ONE greedy speculative iteration, fully on-device.

        Chains K drafter steps with ON-DEVICE argmax + re-embed (no per-step host
        round-trip), assembles the verify batch ``[anchor, d0..d_{K-1}]`` on
        device, runs the batched verify, and argmaxes the verify logits on device.
        Only the ``2K+1`` token ids are read to host (acceptance is a host int
        compare). This is the eager twin of the fused single-iteration trace.

        Args:
            anchor_tok_tt: [1,1] uint32 device token id to draft from (= committed token).
            anchor_hidden: [1,1,1,backbone] device drafter seed.
            anchor_pos: int absolute position p of the anchor token.

        Returns:
            (drafts:list[int], target_ids:list[int] len K+1, verify_hidden device [1,1,K+1,backbone]).
        """
        K = self.draft_len
        # Drafter queries a single fixed position (HF SinglePositionMTP).
        d_pu, d_pi = self._pos_tensors([anchor_pos])
        d_pt = self._page_table(1)
        page_tables = self._shared_kv_page_tables(d_pt)

        draft_id_tts = []  # [1,1] uint32 each
        tok_tt = anchor_tok_tt
        h = anchor_hidden
        owns_h = False
        for _ in range(K):
            logits, h_next = self.assistant.step(tok_tt, h, self._shared_kv, page_tables, d_pu, d_pi)
            if owns_h:
                h.deallocate(True)
            h, owns_h = h_next, True
            idx = self._argmax_last(logits, rows=1)  # [1,1,1] uint32 RM (fast multicore argmax)
            logits.deallocate(True)
            tok_tt = ttnn.reshape(idx, (1, 1))  # [1,1] uint32 RM
            draft_id_tts.append(tok_tt)
        if owns_h:
            h.deallocate(True)

        # Verify input = [anchor, d0..d_{K-1}] at positions p..p+K.
        verify_x = ttnn.concat([anchor_tok_tt] + draft_id_tts, dim=1)  # [1, K+1] uint32 RM
        v_pos = [anchor_pos + j for j in range(K + 1)]
        v_pu, v_pi = self._pos_tensors(v_pos)
        v_pt = self._page_table(K + 1)
        vlogits, vhidden = self.target.ttnn_verify_forward(
            x=verify_x,
            current_pos=v_pu,
            current_pos_cache=v_pi,
            page_table=v_pt,
            kv_cache=self.tt_kv_cache,
            page_tables_per_layer=self._page_tables_per_layer(K + 1),
        )
        vidx = self._argmax_last(vlogits, rows=K + 1)  # [1,1,K+1] uint32 RM (fast multicore argmax)

        drafts = [self._id_to_host(t) for t in draft_id_tts]
        target_ids = self._ids_to_host(vidx, K + 1)

        for t in draft_id_tts:
            t.deallocate(True)
        for t in (verify_x, v_pu, v_pi, v_pt, d_pu, d_pi, d_pt, vlogits, vidx):
            t.deallocate(True)
        return drafts, target_ids, vhidden

    # ── acceptance ────────────────────────────────────────────────────────
    def _accept_greedy(self, drafts, verify_logits):
        """Return (m, committed_tokens). verify_logits[j] = logits at anchor+j."""
        g = [int(torch.argmax(verify_logits[j])) for j in range(len(drafts) + 1)]
        m = len(drafts)
        for i, d in enumerate(drafts):
            if d != g[i]:
                m = i
                break
        committed = list(drafts[:m]) + [g[m]]
        return m, committed

    def _accept_sampling(self, drafts, draft_logits, verify_logits, temperature, top_p, top_k):
        """Speculative-sampling acceptance. Matches the target distribution."""
        K = len(drafts)
        committed = []
        m = K
        for i in range(K):
            p = _to_probs(verify_logits[i], temperature, top_p, top_k)  # target dist at anchor+i
            q = _to_probs(draft_logits[i], temperature, top_p, top_k)  # draft dist for d_i
            d = drafts[i]
            qd = float(q[d])
            ratio = 1.0 if qd <= 0 else min(1.0, float(p[d]) / qd)
            if torch.rand(()) < ratio:
                committed.append(d)
            else:
                resid = torch.clamp(p - q, min=0.0)
                s = resid.sum()
                corr = int(torch.argmax(p)) if s <= 0 else int(torch.multinomial(resid / s, 1))
                committed.append(corr)
                m = i
                break
        if m == K:
            p = _to_probs(verify_logits[K], temperature, top_p, top_k)
            committed.append(
                int(torch.multinomial(p, 1)) if (temperature and temperature > 0) else int(torch.argmax(p))
            )
        return m, committed

    # ── main loop ───────────────────────────────────────────────────────────
    def seed(self, anchor_token, anchor_pos):
        """Compute the target hidden at the anchor (after prefill filled the KV).

        Runs one batch=1 verify of the last committed/prompt token at its
        position — idempotent on the KV cache (prefill already wrote it) — and
        returns the post-norm hidden ``[1,1,1,backbone]`` used to seed the first
        drafter step.
        """
        _, hidden = self._verify([anchor_token], [anchor_pos])
        if self._use_trace:
            # hidden is the persistent batch=1 verify output (do NOT deallocate).
            # Copy it into the persistent anchor buffer (one-time clone to alloc).
            if self._anchor_buf is None:
                self._anchor_buf = ttnn.clone(hidden)
            else:
                ttnn.copy(hidden, self._anchor_buf)
            return self._anchor_buf
        h = ttnn.clone(hidden[:, :, 0:1, :])
        hidden.deallocate(True)
        return h

    def generate_fused(self, anchor_token, anchor_pos, max_new_tokens):
        """Greedy speculative decode using the fully on-device fused iteration.

        Each iteration reads back only the ``2K+1`` token ids; the drafter
        recurrence (argmax + re-embed) and the verify-input assembly stay on
        device. The next drafter seed is SHIFT-mode: the verify hidden at the new
        anchor position ``p+m+1`` (slice ``vhidden[m+1]``) — no extra reseed
        forward, so the whole iteration is one device program (this is the eager
        twin of the fused trace). Returns ``(generated_ids, accepts_per_iter)``.
        """
        if self._use_trace:
            return self._generate_fused_traced(anchor_token, anchor_pos, max_new_tokens)
        self._pv_a_prev = -1  # re-seed packed-verify staging for the new anchor/request
        self._last_fused_setup_s = 0.0
        K = self.draft_len
        out, accepts = [], []
        loop_t0 = time.perf_counter()
        anchor_hidden = self.seed(anchor_token, anchor_pos)  # [1,1,1,h] device
        anchor_tok_tt = self._tokens_tensor([anchor_token])
        while len(out) < max_new_tokens:
            if self._fused_reseed:
                new_anchor_hidden = self.seed(anchor_token, anchor_pos)
                anchor_hidden.deallocate(True)
                anchor_hidden = new_anchor_hidden
            drafts, target_ids, vhidden = self._fused_iter(anchor_tok_tt, anchor_hidden, anchor_pos)
            m = next((i for i in range(K) if drafts[i] != target_ids[i]), K)
            committed = drafts[:m] + [target_ids[m]]
            accepts.append(m)

            new_pos = anchor_pos + m + 1
            new_token = committed[-1]
            if self._fused_reseed:
                # The next iteration will run an exact batch=1 seed for
                # (new_token, new_pos), which also overwrites/repairs the target
                # KV at the new anchor before the drafter cross-attends to it.
                new_anchor_hidden = anchor_hidden
            else:
                # Fast approximate seed: hidden at position p+m+1 = vhidden row
                # (m+1); when all K accepted there is no row K+1, fall back to
                # the last available row (K). This is not the exact assistant
                # contract for correction/bonus anchors, but is useful for A/B.
                row = self._fused_shift_seed_row(m, K)
                new_anchor_hidden = ttnn.clone(vhidden[:, :, row : row + 1, :])
            vhidden.deallocate(True)
            if not self._fused_reseed:
                anchor_hidden.deallocate(True)
            anchor_hidden = new_anchor_hidden
            anchor_tok_tt.deallocate(True)
            anchor_tok_tt = self._tokens_tensor([new_token])
            anchor_pos = new_pos

            for tok in committed:
                out.append(tok)
                if tok in self.stop_tokens:
                    anchor_hidden.deallocate(True)
                    anchor_tok_tt.deallocate(True)
                    self._last_fused_replay_s = time.perf_counter() - loop_t0
                    return out, accepts
                if len(out) >= max_new_tokens:
                    break
        anchor_hidden.deallocate(True)
        anchor_tok_tt.deallocate(True)
        self._last_fused_replay_s = time.perf_counter() - loop_t0
        return out, accepts

    # ── fused single-iteration trace (greedy) ───────────────────────────────
    def _fused_body(self, tr):
        """The fused-iteration op graph over persistent buffers (capture + compile).

        Reads tr["anchor_tok"]/tr["h"] (persistent inputs) and the position /
        page-table buffers; returns the persistent OUTPUT handles
        (verify_x [1,K+1], vidx [1,1,K+1], vhidden [1,1,K+1,backbone]). Drafts are
        verify_x[1:]. All argmax/re-embed/concat are on device, so the K drafter
        steps chain in-graph (no inter-replay copy)."""
        K = self.draft_len
        page_tables = self._shared_kv_page_tables(tr["d_pt"])
        tok = tr["anchor_tok"]
        if self._fused_reseed:
            seed_logits, h = self.target.ttnn_verify_forward(
                x=tr["anchor_tok"],
                current_pos=tr["d_pu"],
                current_pos_cache=tr["d_pi"],
                page_table=tr["d_pt"],
                kv_cache=self.tt_kv_cache,
                page_tables_per_layer=tr["d_ptl"],
            )
            seed_idx = self._argmax_last(seed_logits, rows=1)
            seed_logits.deallocate(True)
        else:
            h = tr["h"]
        draft_ids = []
        for _ in range(K):
            logits, h = self.assistant.step(tok, h, self._shared_kv, page_tables, tr["d_pu"], tr["d_pi"])
            idx = self._argmax_last(logits, rows=1)  # [1,1,1] uint32 RM (fast multicore argmax)
            logits.deallocate(True)
            tok = ttnn.reshape(idx, (1, 1))  # [1,1] uint32 RM
            draft_ids.append(tok)
        # In exact-reseed mode, the seed forward already computed row 0
        # (target logits for the anchor) and repaired the anchor KV before the
        # drafter reads shared KV. Verify only the draft tokens at p+1..p+K and
        # prepend seed_idx to form target_ids[0..K]. In shift mode, keep the old
        # single verify batch [anchor, d0..dK-1].
        verify_inputs = draft_ids if self._fused_reseed else [tr["anchor_tok"]] + draft_ids
        verify_x = ttnn.concat(verify_inputs, dim=1)  # reseed: [1,K], shift: [1,K+1]
        if tr.get("pv"):
            # PACKED verify: all K+1 candidates in the query-heads dim -> ONE
            # KV read per layer instead of K+1 (the batch-dim verify re-reads
            # the whole KV per candidate row; at 256k that is 6x 5.2 GB/step).
            # Inputs are persistent buffers refreshed per replay by
            # _generate_fused_traced; page tables are width-matched to their
            # masks (see _pv_tables_per_layer).
            # Repeat the [1,1,P,S_k] host masks H x in-trace: dim-2 repeat tiles
            # the whole P-block per head -> row order h*P+p, the packed layout.
            _H = self._packed_H()
            mask_full = ttnn.repeat(tr["pv_mask_full"], ttnn.Shape([1, 1, _H, 1]))
            mask_slide = ttnn.repeat(tr["pv_mask_slide"], ttnn.Shape([1, 1, _H, 1]))
            vlogits, vhidden = self.target.ttnn_packed_verify_forward(
                x=verify_x,
                position_idx=tr["pv_pos"],
                attn_mask_full=mask_full,
                attn_mask_sliding=mask_slide,
                packed_p=K + 1,
                page_table=tr["v_pt"],
                kv_cache=self.tt_kv_cache,
                embed_idx_full=tr["pv_embed"].get("full_attention"),
                embed_idx_sliding=tr["pv_embed"].get("sliding_attention"),
                hot_pt_full=tr["pv_hot"].get("full_attention"),
                hot_pt_sliding=tr["pv_hot"].get("sliding_attention"),
                page_tables_per_layer=tr["pv_ptl"],
            )
            mask_full.deallocate(True)
            mask_slide.deallocate(True)
        else:
            vlogits, vhidden = self.target.ttnn_verify_forward(
                x=verify_x,
                current_pos=tr["v_pu"],
                current_pos_cache=tr["v_pi"],
                page_table=tr["v_pt"],
                kv_cache=self.tt_kv_cache,
                page_tables_per_layer=tr["v_ptl"],
            )
        tail_rows = K if self._fused_reseed else K + 1
        tail_idx = self._argmax_last(vlogits, rows=tail_rows)  # [1,1,K or K+1] uint32 RM
        if self._fused_reseed:
            vidx = ttnn.concat([seed_idx, tail_idx], dim=2)  # [1,1,K+1]
            seed_idx.deallocate(True)
            tail_idx.deallocate(True)
        else:
            vidx = tail_idx
        vlogits.deallocate(True)
        return verify_x, vidx, vhidden

    def _fused_packed_enabled(self):
        """Packed verify inside the fused trace. Auto: ON for BOUNDED targets in
        the 256k tier (max_seq_len > 131072), shift seed mode only.

        Measured (31B, greedy K=5, packed vs batch-dim verify, tok/s/u):
          32k  44.79 (2.28/5) vs 42.23 (2.40/5)  -- packed wins slightly
          128k 32.93 (1.70/5) vs 36.06 (2.78/5)  -- packed LOSES: its non-causal
               long-S_k softmax drifts off greedy (~tok 41) with the mesh op
               default; fp32 dest acc on mesh makes it far worse (0.06/5)
          256k 23.16 (1.63/5) vs 16.35 (1.70/5)  -- packed turns the datapoint
               positive vs the 16.97 baseline (KV re-read dominates batch-dim)
        So auto only enables where the KV amortization outweighs the acceptance
        cost. Reseed mode is excluded structurally: its seed forward writes the
        anchor KV around the staging, and the staged hot block goes stale.
        GEMMA4_SPEC_FUSED_PACKED=1/0 overrides."""
        env = os.environ.get("GEMMA4_SPEC_FUSED_PACKED")
        if env in ("0", "1"):
            return env == "1" and not self._fused_reseed
        bounded = bool(getattr(self.target, "bounded_sliding_kv_cache", False))
        max_seq = int(getattr(self.target, "max_seq_len", 0) or 0)
        return bounded and max_seq > 131072 and not self._fused_reseed

    def _capture_fused_trace(self, anchor_token, anchor_hidden, anchor_pos, max_new_tokens=None):
        """Capture ONE fused iteration at the real first-call inputs.

        Allocates persistent input buffers (anchor token, recurrent hidden, the
        drafter + verify position tensors, page tables), runs a compile pass, then
        captures the fused body. The compile + capture write the SAME KV positions
        with the SAME tokens (deterministic), so the writes are idempotent."""
        from loguru import logger as _lg

        K = self.draft_len
        v_pos = [anchor_pos + 1 + j for j in range(K)] if self._fused_reseed else [anchor_pos + j for j in range(K + 1)]
        d_pu, d_pi = self._pos_tensors([anchor_pos])
        v_pu, v_pi = self._pos_tensors(v_pos)
        tr = {
            "anchor_tok": self._tokens_tensor([anchor_token]),
            "h": ttnn.clone(anchor_hidden),
            "d_pu": d_pu,
            "d_pi": d_pi,
            "d_pt": self._page_table(1),
            "v_pu": v_pu,
            "v_pi": v_pi,
            "v_pt": self._page_table(K if self._fused_reseed else K + 1),
            # Bounded sliding KV runs HYBRID per-layer page tables, and verify puts
            # its candidates in the BATCH dim -- each layer's table needs the user's
            # row replicated across them. Without this the fused body fell through
            # to the model's batch-1 ``_active_page_tables_per_layer`` and
            # decode_forward sliced row b>=1 of a 1-row table (RuntimeError: bad
            # optional access). Resolved once here so compile and capture bind the
            # SAME persistent device buffers. None when unbounded (path unchanged).
            "d_ptl": self._page_tables_per_layer(1),
            "v_ptl": self._page_tables_per_layer(K if self._fused_reseed else K + 1),
        }
        if self._fused_packed_enabled():
            # PACKED verify inside the trace (shift mode: rows = anchor,d0..dK-1
            # at positions c..c+K, exactly _pv_host_inputs' layout). S_k is fixed
            # at capture to cover the WHOLE generation so mask/table widths (and
            # the captured program) never change across replays; columns beyond
            # the live top are NEG, so the wide mask is exact.
            self._pv_setup()
            self._pv_seed_staging(anchor_pos)
            P_v = K + 1
            horizon = anchor_pos + (max_new_tokens or 0) + P_v + 1
            s_k_cap = ((horizon + self._pv_sk_bucket - 1) // self._pv_sk_bucket) * self._pv_sk_bucket
            # h_repeat=False: masks arrive as [1,1,P,S_k] and are repeated H x
            # IN-TRACE (ttnn.repeat dim-2 tiles the P-block per head, exactly the
            # h*P+p row order the packed SDPA wants). Rows depend only on p, so
            # this cuts the per-replay host mask build+copy by H (25 MB -> 3.1 MB
            # at 256k). A device-computed mask (bcast add + clip) was probed and
            # ttnn's unit-tensor broadcast does NOT do scalar semantics -- 97%
            # wrong; do not re-try without a dedicated bcast op.
            h = self._pv_host_inputs(anchor_pos, P_v, s_k=s_k_cap, h_repeat=False)
            tr["pv"] = True
            tr["pv_S_k"] = s_k_cap
            tr["pv_pos"] = self._pv_from_torch(h["pos"], ttnn.uint32)
            tr["pv_mask_full"] = self._pv_from_torch(h["mask_full"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            tr["pv_mask_slide"] = self._pv_from_torch(h["mask_slide"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            tr["pv_embed"] = {lt: self._pv_from_torch(e, ttnn.uint32) for lt, e in h["embed"].items()}
            tr["pv_hot"] = {lt: self._pv_from_torch(t, ttnn.int32) for lt, t in h["hot_t"].items()}
            tr["pv_ptl"] = self._pv_tables_per_layer(s_k_cap)
            _lg.info(f"[spec-trace] fused verify: PACKED (S_k={s_k_cap}, ring={self._pv_ring})")
        _lg.info("[spec-trace] capture fused: compile run")
        vx, vidx, vh = self._fused_body(tr)
        ttnn.synchronize_device(self.mesh_device)
        vx.deallocate(True)
        vidx.deallocate(True)
        vh.deallocate(True)
        _lg.info("[spec-trace] capture fused: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        vx, vidx, vh = self._fused_body(tr)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info("[spec-trace] capture fused: DONE")
        tr["id"] = tid
        tr["verify_x"] = vx
        tr["vidx"] = vidx
        tr["vhidden"] = vh
        self._fused_trace = tr

    def _free_tensor_tree(self, v):
        """Deallocate a tensor, or every tensor in a dict/list of them."""
        if hasattr(v, "deallocate"):
            try:
                v.deallocate(True)
            except Exception:
                pass
        elif isinstance(v, dict):
            for t in v.values():
                self._free_tensor_tree(t)
        elif isinstance(v, (list, tuple)):
            for t in v:
                self._free_tensor_tree(t)

    def _hidden_row_to_device(self, row):
        """Read the verify hidden, slice row `row`, and copy it into tr["h"].

        Host round-trip for the recurrent seed: small ([1,1,1,backbone]) and
        allocation-free on device (copy_host_to_device into the persistent buffer),
        so it stays trace-safe (no device clone/slice that could alias trace
        scratch on a re-replay)."""
        tr = self._fused_trace
        vh = tr["vhidden"]
        vh_t = ttnn.to_torch(ttnn.get_device_tensors(vh)[0]) if self._tp > 1 else ttnn.to_torch(vh)
        sel = vh_t[:, :, row : row + 1, :].contiguous()  # [1,1,1,backbone]
        host_h = ttnn.from_torch(sel, layout=ttnn.TILE_LAYOUT, dtype=tr["h"].dtype, mesh_mapper=self._mapper)
        ttnn.copy_host_to_device_tensor(host_h, tr["h"])
        host_h.deallocate(True)

    def _generate_fused_traced(self, anchor_token, anchor_pos, max_new_tokens):
        """Greedy spec-decode via the single fused trace (one replay per iter)."""
        from loguru import logger as _lg

        K = self.draft_len
        # Eager seed (once, before the loop). Force the UNTRACED verify so we do
        # NOT capture a separate batch=1 verify trace — an active verify trace
        # would collide with the fused trace's eager compile-run allocations and
        # hang. The loop then replays ONLY the single fused trace (no interleaving
        # of distinct CCL traces).
        setup_t0 = time.perf_counter()
        self._use_trace = False
        anchor_hidden = self.seed(anchor_token, anchor_pos)
        self._use_trace = True
        self._capture_fused_trace(anchor_token, anchor_hidden, anchor_pos, max_new_tokens=max_new_tokens)
        anchor_hidden.deallocate(True)
        tr = self._fused_trace
        self._last_fused_setup_s = time.perf_counter() - setup_t0

        out, accepts = [], []
        cur_token, cur_pos = anchor_token, anchor_pos
        first = True  # capture already bound the first inputs (token/pos/hidden)
        replay_t0 = time.perf_counter()
        while len(out) < max_new_tokens:
            if not first:
                h_tok = self._host_tokens([cur_token])
                ttnn.copy_host_to_device_tensor(h_tok, tr["anchor_tok"])
                h_tok.deallocate(True)
                d_hpu, d_hpi = self._host_pos([cur_pos])
                ttnn.copy_host_to_device_tensor(d_hpu, tr["d_pu"])
                ttnn.copy_host_to_device_tensor(d_hpi, tr["d_pi"])
                d_hpu.deallocate(True)
                d_hpi.deallocate(True)
                v_pos = (
                    [cur_pos + 1 + j for j in range(K)] if self._fused_reseed else [cur_pos + j for j in range(K + 1)]
                )
                v_hpu, v_hpi = self._host_pos(v_pos)
                ttnn.copy_host_to_device_tensor(v_hpu, tr["v_pu"])
                ttnn.copy_host_to_device_tensor(v_hpi, tr["v_pi"])
                v_hpu.deallocate(True)
                v_hpi.deallocate(True)
                if tr.get("pv"):
                    # Refresh the packed-verify inputs for the new anchor: the
                    # positions, both masks (fixed widths -- S_k was capped at
                    # capture), per-type merge indices (they encode the hot-block
                    # roll) and per-type hot pages. _pv_a_prev must advance ONLY
                    # after the host inputs are built (the roll flag reads it).
                    h2 = self._pv_host_inputs(cur_pos, K + 1, s_k=tr["pv_S_k"], h_repeat=False)
                    ttnn.copy_host_to_device_tensor(
                        self._pv_from_torch(h2["pos"], ttnn.uint32, device=False), tr["pv_pos"]
                    )
                    ttnn.copy_host_to_device_tensor(
                        self._pv_from_torch(h2["mask_full"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=False),
                        tr["pv_mask_full"],
                    )
                    ttnn.copy_host_to_device_tensor(
                        self._pv_from_torch(h2["mask_slide"], ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=False),
                        tr["pv_mask_slide"],
                    )
                    for lt, e in h2["embed"].items():
                        ttnn.copy_host_to_device_tensor(
                            self._pv_from_torch(e, ttnn.uint32, device=False), tr["pv_embed"][lt]
                        )
                    for lt, t in h2["hot_t"].items():
                        ttnn.copy_host_to_device_tensor(
                            self._pv_from_torch(t, ttnn.int32, device=False), tr["pv_hot"][lt]
                        )
                    self._pv_a_prev = cur_pos // self._pv_bs
                # In reseed mode the trace computes the exact seed internally.
                # Otherwise tr["h"] already holds this iter's approximate seed
                # (set at the end of the previous iteration).
            first = False

            _lg.debug(f"[spec-trace] fused replay pos={cur_pos} execute")
            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)

            vx = (
                ttnn.to_torch(ttnn.get_device_tensors(tr["verify_x"])[0])
                if self._tp > 1
                else ttnn.to_torch(tr["verify_x"])
            )
            vx = vx.reshape(-1)
            drafts = [int(vx[j if self._fused_reseed else 1 + j]) for j in range(K)]
            target_ids = self._ids_to_host(tr["vidx"], K + 1)
            m = next((i for i in range(K) if drafts[i] != target_ids[i]), K)
            committed = drafts[:m] + [target_ids[m]]
            accepts.append(m)

            if not self._fused_reseed:
                # Shift seed: choose one of the verified hidden rows as a cheap
                # approximate seed for the next anchor (see _fused_shift_seed).
                self._hidden_row_to_device(self._fused_shift_seed_row(m, K))
            cur_pos = cur_pos + m + 1
            cur_token = committed[-1]

            for tok in committed:
                out.append(tok)
                if tok in self.stop_tokens:
                    self._last_fused_replay_s = time.perf_counter() - replay_t0
                    return out, accepts
                if len(out) >= max_new_tokens:
                    break
        self._last_fused_replay_s = time.perf_counter() - replay_t0
        return out, accepts

    # ── serving: batched fused records (vLLM contract adapter) ───────────────
    #
    # One captured trace per (shape, width) record, shape = (B_v, K_v). Every
    # width record of a shape binds the same width-independent buffers
    # (anchor tokens, seed index, drafter positions/tables, verify positions,
    # KV-write indices, ring scalars); a width record adds only the
    # width-matched per-type verify tables. Nothing about a request is baked
    # at capture: positions, tokens and page-table rows are uploaded before
    # every replay, so one record serves every request.
    #
    # Padding rows (record rows past the live count) mirror live row 0: the
    # same anchor, position, seed row and page-table rows, uploaded by this
    # file, whatever the adapter sends for them (it sends zero table rows,
    # which this file checks and then replaces). The interface's "block 0 =
    # null block" holds for the full-attention pool only; see serving_pad_rows.
    #
    # Masks. The packed verify's additive masks are built IN-TRACE from the
    # uploaded verify positions and a persistent key iota (the dFlash
    # ``pv_iota`` path: sub / gtz / gez / ltz / typecast / multiply in fp32,
    # H-repeated after), in key chunks of SPEC_MASK_KEY_CHUNK so the
    # tile-padded fp32 intermediates stay bounded; ``serving_masks_host`` is
    # the host reference and ``serving_masks_from_plan`` the op-for-op
    # emulation the host tests hold it to. No mask DRAM persists per rung and
    # nothing mask-sized is uploaded per step.
    #
    # Logits. Every record's body ends by copying its verify logits into the
    # decoder's ONE shared [1, 1, 32, V_padded] buffer (``ttnn.assign`` into a
    # persistent tensor, the same precedent as the hidden pool) and freeing
    # its own; ``ServingStepResult.logits`` is that buffer, valid until the
    # next serving_step or serving_reseed of this decoder.
    #
    # Seed hidden. The fused body starts by gathering each row's drafter seed
    # ON DEVICE from a shared 64-row hidden pool (rows 0..31 = the previous
    # replay's verify hidden, padded; rows 32..63 = the last serving_reseed's
    # target hidden) through the uploaded ``seed_idx`` [1, B_v] row index, and
    # ends by publishing its own verify hidden into pool rows 0..31
    # (``ttnn.assign`` into the persistent buffer, the dFlash ``fc_prev``
    # pattern). ``serving_seed_index`` builds the index on the host.

    def _srv_require(self, captured=None):
        if self._srv_records is None:
            raise RuntimeError("serving_prepare has not run on this SpeculativeDecoder")
        if captured is True and not self._srv_captured:
            raise RuntimeError("serving_capture has not run: the records are prepared but not captured")
        if captured is False and self._srv_captured:
            raise RuntimeError("serving_capture already ran; serving_release before preparing again")

    def _srv_shape(self, shape):
        shape = spec_check_shape(shape)
        if shape not in self._srv_shape_bufs:
            raise ValueError(f"shape {shape} was not prepared; prepared shapes: {sorted(self._srv_shape_bufs)}")
        return shape

    @staticmethod
    def _srv_check_rows(shape, rows, anchors, positions):
        """Validate one step's per-row inputs; returns the live-row count."""
        B, _K = shape
        live = len(rows)
        if not 1 <= live <= B:
            raise ValueError(f"shape {shape}: {live} live rows, expected 1..{B}")
        if len(anchors) != live or len(positions) != live:
            raise ValueError(
                f"shape {shape}: rows/anchors/positions lengths differ: {live}/{len(anchors)}/{len(positions)}"
            )
        if len({int(r) for r in rows}) != live:
            raise ValueError(f"shape {shape}: duplicate rows in {list(rows)}")
        if min(int(r) for r in rows) < 0:
            raise ValueError(f"shape {shape}: negative row in {list(rows)}")
        if min(int(p) for p in positions) < 0 or min(int(a) for a in anchors) < 0:
            raise ValueError(f"shape {shape}: negative anchor/position in {list(anchors)}/{list(positions)}")
        return live

    def _srv_layer_types(self):
        return list(self.target.hf_config.layer_types[: len(self.target.layers)])

    def serving_prepare(
        self, shapes, widths_by_shape, max_rows, record_mb=SPEC_RECORD_MB_DEFAULT, max_context=None, max_rungs=None
    ):
        """Allocate and compile every (shape, width) record; captures nothing.

        ``widths_by_shape=None`` takes the default x4 ladder per shape from
        ``max_context`` (``max_rungs`` keeps the largest rungs). The planned
        record count and the projected trace bytes (``record_mb`` per record,
        default the measured 32 MB, on top of what the region already holds)
        are logged against the device's trace region size when
        ``ttnn.get_memory_view`` reports it, as a projection only otherwise.

        Runs, eagerly and in this order: ``_pv_setup`` (ring detection; its
        one-slot staging is not consumed by the records, whose verify writes
        per position), then per shape the shape buffers, one target seed
        forward at B_v rows (the reseed graph) and one fused body per width
        (the replay graph). Every graph is therefore compiled before the
        adapter's ``serving_capture`` records the first trace. The assistant
        must have been built for ``max_rows`` rows
        (``create_assistant_model(max_local_batch_size=max_rows)``); its
        declared capacity is checked here. Nothing on the device is sized by
        ``max_rows`` itself: the per-shape buffers are B_v rows, the hidden and
        logits buffers 32 rows.

        Nothing allocates on the device after this call returns. Every
        buffer a reseed or a replay touches exists now: the per-shape inputs
        (anchors, seed rows, positions, drafter tables, the reseed forward's
        own token and position inputs), the per-record verify tables and
        outputs (the verify input and its argmax ids the body writes in
        place and the host reads back), the hidden pools and the logits
        buffer (made by the first eager seed and body) and the mask iotas
        (one per key-chunk length the ladder needs, made by the eager
        bodies). ``serving_reseed``, ``serving_step`` and
        ``serving_refresh_page_tables`` only upload into them; a buffer one
        of them would have to create is an error, never an allocation. The
        decode-trace guard releases and re-captures every decode trace when
        the allocator grows after a capture (0.7 s per re-capture), and the
        eager target forward of a reseed is the one serving call that still
        allocates: the model's own transients, freed before it returns.
        """
        from loguru import logger as _lg

        if self._srv_records is not None:
            raise RuntimeError("serving_prepare already ran; serving_release before preparing again")
        shapes, widths = serving_plan(shapes, widths_by_shape, max_rows, max_context=max_context, max_rungs=max_rungs)
        max_rows = int(max_rows)
        projection = serving_trace_projection(
            shapes,
            widths,
            record_mb=record_mb,
            region_mb=self._srv_trace_region_size_mb(),
            base_mb=self._srv_trace_region_mb(),
        )
        self._srv_projection = projection
        (_lg.warning if projection.fits is False else _lg.info)(f"[spec-serving] plan: {projection.describe()}")
        cap = getattr(self.assistant, "max_local_batch_size", None)
        if cap is None or int(cap) < max_rows:
            raise ValueError(
                f"the assistant declares max_local_batch_size={cap} but serving needs {max_rows} rows; "
                "build it with create_assistant_model(..., max_local_batch_size=max_rows)"
            )
        if self.page_table_torch.dim() != 2:
            raise ValueError(f"page_table_torch must be [rows, blocks], got shape {tuple(self.page_table_torch.shape)}")
        t0 = time.perf_counter()
        # Packed mode is the only verify the records run (the batched body is
        # packed-only), so the ``_fused_packed_enabled`` tier rule does not
        # apply here: unbounded targets run packed too.
        self._pv_setup()
        _lg.info(
            f"[spec-serving] KV geometry per layer type: effective block size {self._pv_bs_t}, "
            f"ring {self._pv_ring} (layer 0 physical block dim {self._pv_bs})"
        )
        self._srv_blocks = int(self.page_table_torch.shape[-1])
        self._srv_check_steps = int(os.environ.get(SPEC_VERIFY_CHECK_ENV, "0") or 0)
        self._srv_checked = {}
        self._srv_max_rows = max_rows
        self._srv_shapes = shapes
        self._srv_widths = widths
        self._srv_records = {}
        self._srv_shape_bufs = {}
        self._srv_pool_v = None
        self._srv_pool_s = None
        self._srv_seed_traces = {}
        self._srv_draft_consts = {}
        self._srv_logits = None
        self._srv_iota = {}
        self._srv_backbone = None
        self._srv_flat_type = "full_attention" if "full_attention" in self._pv_ring else next(iter(self._pv_ring))
        # One drafter/verify table per vLLM KV-cache group slot (not per layer
        # type): under hybrid groups the layers of one type are spread over
        # several groups with disjoint block ids.
        self._srv_slots, self._srv_n_slots = serving_kv_group_slots(self._srv_layer_types())
        self._srv_uploaded = {}
        self._srv_last = None
        self._srv_phase_ms = {}
        self._srv_seeded = {}
        self._srv_captured = False
        self._srv_frozen = False
        for shape in shapes:
            B = shape[0]
            S = self._srv_alloc_shape(shape)
            self._srv_shape_bufs[shape] = S
            # The reseed graph at B_v rows, on padding inputs: token 0 at
            # position 0 through zero tables, i.e. block 0 (vLLM's null block,
            # where the decode warmup already writes). It also fixes the pool
            # geometry (backbone width, hidden dtype) on the first shape.
            self._srv_seed_rows(shape, [0] * B, [0] * B)
            for w in widths[shape]:
                rec = self._srv_alloc_width(shape, S, w)
                self._srv_body(rec)  # writes the record's persistent outputs; nothing to free
                ttnn.synchronize_device(self.mesh_device)
                self._srv_records[(shape, w)] = rec
                _lg.info(f"[spec-serving] prepared record shape={shape} S_k={w}")
        self._srv_last = None
        self._srv_seeded = {}
        # The device set is complete: from here on a missing buffer is a bug, not an allocation.
        self._srv_frozen = True
        _lg.info(
            f"[spec-serving] prepared {len(self._srv_records)} records for shapes {shapes} "
            f"(max_rows={max_rows}) in {time.perf_counter() - t0:.1f}s"
        )

    def _srv_trace_region_mb(self):
        """Bytes allocated in the device's TRACE region, summed over the banks, in MB (capture budgeting)."""
        view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
        return float(view.total_bytes_allocated_per_bank) * float(view.num_banks) / 1e6

    def _srv_trace_region_size_mb(self):
        """The device's TRACE region size in MB (``MemoryView.total_bytes_per_bank`` x banks), or None when this
        ttnn's memory view does not expose the per-bank total (the projection is then logged on its own)."""
        view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.TRACE)
        total = getattr(view, "total_bytes_per_bank", None)
        if total is None:
            return None
        return float(total) * float(view.num_banks) / 1e6

    def serving_capture(self):
        """Capture one trace per prepared record (call once, after serving_prepare)."""
        from loguru import logger as _lg

        self._srv_require(captured=False)
        t_all = time.perf_counter()
        cost = {}
        used = self._srv_trace_region_mb()
        region = self._srv_trace_region_size_mb()
        _lg.info(f"[spec-serving] trace region before the record captures: {used:.1f} MB in use")
        captured = []  # (key, measured MB) in capture order
        for key, rec in self._srv_records.items():
            t0 = time.perf_counter()
            try:
                tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                verify_x, vidx = self._srv_body(rec)
                ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            except RuntimeError as exc:
                # The TT_FATAL of a full trace region surfaces as a RuntimeError;
                # re-raise it with the budget context (never swallowed: chained).
                last = f"last captured {captured[-1][0]} at {captured[-1][1]:.1f} MB" if captured else "none captured"
                region_text = f"{region:.0f} MB" if region is not None else "not readable"
                raise RuntimeError(
                    f"serving_capture failed on record shape={rec['shape']} S_k={rec['S_k']} after "
                    f"{len(captured)} of {len(self._srv_records)} records ({[k for k, _ in captured]}); {last}; "
                    f"trace region {used:.1f} MB in use of {region_text}; plan: {self._srv_projection.describe()}; "
                    f"fewer rungs ({SPEC_WIDTH_RUNGS_ENV}) or shapes shrink the record set. Device error: {exc}"
                ) from exc
            if verify_x is not rec["verify_x"] or vidx is not rec["vidx"]:
                raise RuntimeError(
                    f"serving_capture: the body of record shape={rec['shape']} S_k={rec['S_k']} returned outputs "
                    "that are not the record's persistent handles (an allocation inside the capture)"
                )
            rec["id"] = tid
            cost[key] = round(time.perf_counter() - t0, 2)
            now = self._srv_trace_region_mb()
            rec["trace_mb"] = now - used
            captured.append((key, now - used))
            _lg.info(
                f"[spec-serving] captured record shape={rec['shape']} S_k={rec['S_k']}: "
                f"+{now - used:.1f} MB trace, {now:.1f} MB in use"
            )
            used = now
        if self._srv_traced_seed:
            for shape in self._srv_shape_bufs:
                t0 = time.perf_counter()
                try:
                    tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                    self._srv_seed_body(shape)
                    ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
                except RuntimeError as exc:
                    region_text = f"{region:.0f} MB" if region is not None else "not readable"
                    raise RuntimeError(
                        f"serving_capture failed on the seed trace of shape={shape} after the {len(captured)} "
                        f"records; trace region {used:.1f} MB in use of {region_text}; GEMMA4_SPEC_TRACED_SEED=0 "
                        f"keeps the reseeds eager. Device error: {exc}"
                    ) from exc
                self._srv_seed_traces[shape] = tid
                now = self._srv_trace_region_mb()
                cost[("seed", shape)] = round(time.perf_counter() - t0, 2)
                _lg.info(
                    f"[spec-serving] captured seed trace shape={shape}: +{now - used:.1f} MB trace, {now:.1f} MB in use"
                )
                used = now
        self._srv_captured = True
        # The capture passes replayed nothing real: no previous verify, no seed.
        self._srv_last = None
        self._srv_seeded = {}
        _lg.info(f"[spec-serving] captured {len(cost)} records in {time.perf_counter() - t_all:.1f}s: {cost}")

    @staticmethod
    def serving_measure_positions(widths, k, top_position=None):
        """The positions each rung of a width ladder is timed at, ``[(width, first, last)]``: ``first`` is the
        position the rung starts serving (0 for the narrowest, else the first position the previous rung no
        longer covers) and ``last`` the highest it serves (``width - k - 2``), capped at ``top_position``."""
        ladder = sorted({int(w) for w in widths})
        positions, prev = [], None
        for w in ladder:
            first = 0 if prev is None else prev - int(k) - 1
            if serving_width_for(ladder, first, k) != w:
                raise ValueError(
                    f"position {first} selects rung {serving_width_for(ladder, first, k)} of {ladder}, not {w}"
                )
            last = w - int(k) - 2
            if top_position is not None:
                last = min(last, int(top_position))
            positions.append((w, first, max(first, last)))
            prev = w
        return positions

    def serving_measure_costs(self, top_position, repeats=2):
        """Time the serving sequence per record at the ends of its position range; the adaptive-K cost table.

        Returns ``{("reseed", shape): (0, ms, top, ms), ("step", shape, width): (first, ms, last, ms)}`` and,
        where a one-row replay is cheaper, ``("partial", shape, width)``,
        each value a cost line in the position (``spec_policy.CostLine``):
        per shape one ``serving_reseed`` of EVERY row of the shape at
        position 0 and at ``top_position`` (the eager target forward at
        ``B_v`` rows the batch pays when it enters speculation), and per
        record one ``serving_step`` of every row at the first and the last
        position its rung serves (``serving_measure_positions``) -- the
        ``step`` line is what the covering record costs the batch with all
        its rows live (the one-row timing of the earlier table undershot the
        served (8, 3) by 10 ms and let the policy enter below plain) -- plus
        the same step with one live row, kept as the ``partial`` line only
        when it is cheaper at every measured position (the cost of a partial
        replay over the continuing rows). Each number is what the adapter
        pays for that call on the device: the uploads, the replay and the
        host readback of the drafts and argmax ids. The minimum of
        ``repeats`` timings is kept. Runs after ``serving_capture`` and
        before any real step, through zero tables (block 0, vLLM's null
        block, which the eager and captured passes already wrote; a bounded
        target has its per-layer tables installed by the adapter first), and
        leaves the decoder as the capture left it: no seed, no previous
        replay.
        """
        from loguru import logger as _lg

        self._srv_require(captured=True)
        if int(repeats) < 1:
            raise ValueError(f"serving_measure_costs needs repeats >= 1, got {repeats}")
        top_position = max(0, int(top_position))
        t_all = time.perf_counter()
        costs = {}
        timings = 0

        def timed(call):
            best = None
            for _ in range(int(repeats)):
                t0 = time.perf_counter()
                out = call()
                ttnn.synchronize_device(self.mesh_device)
                dt = (time.perf_counter() - t0) * 1e3
                best = dt if best is None else min(best, dt)
            return best, out

        self._srv_measuring = True  # the zero-table measurement replays are not served verifies (no check)
        for shape in self._srv_shapes:
            B, K = shape
            rows = list(range(B))  # every row of the shape live: the covering record's cost
            table = torch.zeros((B, self._srv_blocks), dtype=torch.int32)
            at_zero, _ = timed(lambda: self.serving_reseed(shape, rows, [0] * B, [0] * B, table))
            at_top = at_zero
            if top_position > 0:
                at_top, _ = timed(lambda: self.serving_reseed(shape, rows, [0] * B, [top_position] * B, table))
                timings += 1
            timings += 1
            costs[("reseed", shape)] = (0, at_zero, top_position, at_top)
            for width, first, last in self.serving_measure_positions(self._srv_widths[shape], K, top_position):
                line, one = [], []
                for pos in (first, last) if last > first else (first,):
                    ms, result = timed(lambda: self.serving_step(shape, rows, [0] * B, [pos] * B, table, [None] * B))
                    timings += 1
                    if int(result.width) != int(width):
                        raise RuntimeError(
                            f"serving_measure_costs timed record shape={shape} at position {pos} on "
                            f"S_k={result.width}, expected the S_k={width} rung"
                        )
                    line.append((pos, ms))
                    if B > 1:  # one live row: a partial replay's cost, kept below only where it is cheaper
                        ms1, _ = timed(lambda: self.serving_step(shape, [0], [0], [pos], table, [None]))
                        timings += 1
                        one.append((pos, ms1))
                if len(line) == 1:
                    line.append(line[0])
                    one = one * 2
                costs[("step", shape, int(width))] = (line[0][0], line[0][1], line[1][0], line[1][1])
                if one and all(m1 < m for (_p1, m1), (_p, m) in zip(one, line)):
                    costs[("partial", shape, int(width))] = (one[0][0], one[0][1], one[1][0], one[1][1])
        self._srv_measuring = False
        self._srv_last = None
        self._srv_seeded = {}
        _lg.info(
            f"[spec-serving] measured {len(costs)} record cost lines ({timings} timings, min of {int(repeats)} "
            f"each) in {time.perf_counter() - t_all:.1f}s (ms at position; step and reseed with every row of the "
            f"shape live, partial = one live row where cheaper): "
            + ", ".join(
                f"{k[0]} {k[1]}" + (f" S_k={k[2]}" if len(k) > 2 else "") + f"={v[1]:.1f}@{v[0]}..{v[3]:.1f}@{v[2]}"
                for k, v in costs.items()
            )
        )
        return costs

    def serving_widths(self):
        """The resolved width ladder per prepared shape ({shape: [rungs]}), after serving_prepare."""
        self._srv_require()
        return {shape: list(ws) for shape, ws in self._srv_widths.items()}

    def serving_width_for(self, shape, max_position):
        """Narrowest prepared rung of ``shape`` covering a verify block at ``max_position``; None if none does."""
        self._srv_require()
        shape = self._srv_shape(shape)
        return serving_width_for(self._srv_widths[shape], max_position, shape[1])

    def serving_refresh_page_tables(self, page_tables_per_layer):
        """Install the adapter's per-layer page tables (one host [rows, blocks] table per target layer, or None
        for a layer that uses the flat table).

        The records bake no table content: every replay uploads the rows it
        serves, selected from these tables (and from the flat ``page_tables``
        passed per step) -- so a table that grows is picked up by the next
        upload, and this hook only has to keep the host tables current. A
        bounded-sliding target (ring pools per layer type) needs this install
        before its first reseed/replay; an unbounded target may never call it.
        ``None`` is a no-op, as for ``refresh_page_tables``.
        """
        if page_tables_per_layer is None:
            return
        n_layers = len(self.target.layers)
        if not isinstance(page_tables_per_layer, (list, tuple)) or len(page_tables_per_layer) != n_layers:
            got = type(page_tables_per_layer).__name__
            if isinstance(page_tables_per_layer, (list, tuple)):
                got = f"{got} of length {len(page_tables_per_layer)}"
            raise ValueError(f"serving_refresh_page_tables expects one table per target layer ({n_layers}), got {got}")
        installed = []
        converted = {}  # the layers of a KV group hand the same host table: converted once, kept shared
        for i, pt in enumerate(page_tables_per_layer):
            if pt is None:
                installed.append(None)
                continue
            if not hasattr(pt, "dim"):
                raise TypeError(f"per-layer page table {i} must be a host torch tensor, got {type(pt).__name__}")
            t64 = converted.get(id(pt))
            if t64 is None:
                t = pt.unsqueeze(0) if pt.dim() == 1 else pt
                if t.dim() != 2:
                    raise ValueError(f"per-layer page table {i} must be [rows, blocks], got shape {tuple(pt.shape)}")
                t64 = converted[id(pt)] = t.to(torch.int64)
            installed.append(t64)
        slots = getattr(self, "_srv_slots", None)
        if slots is not None and len(slots) == len(installed):
            serving_check_group_tables(installed, slots)
        self._srv_per_layer = installed

    def serving_reseed(self, shape, rows, anchors, positions, page_tables):
        """Seed the drafter for ``rows`` by a target forward at (anchor, position).

        One batch-dim verify at B_v rows (padding rows mirror the first
        reseeded row) re-writes each live anchor's KV in place and lands the
        post-norm hidden in reseed pool rows 32..63. The NEXT ``serving_step``
        reads it for every row whose ``seed_cols`` entry is None. The pool
        holds the rows of the LAST reseed only: reseed every pending row in
        one call, then step.

        This is the exact drafter contract (assistant/model.py ``step``: the
        first step pairs the anchor token with the target's post-norm hidden
        AT the anchor's own position; ``seed`` in this module): the forward
        consumes the anchor itself at its position.
        """
        self._srv_require()
        shape = self._srv_shape(shape)
        B, _K = shape
        live = self._srv_check_rows(shape, rows, anchors, positions)
        self._srv_upload_tables(shape, rows, live, page_tables, None)
        # padding rows mirror the first reseeded row (serving_pad_rows): their
        # anchor K/V write repeats row 0's, byte for byte, in row 0's slot
        self._srv_seed_rows(
            shape, serving_pad_rows([int(a) for a in anchors], B), serving_pad_rows([int(p) for p in positions], B)
        )
        self._srv_seeded = {int(slot): r for r, slot in enumerate(rows)}

    def serving_step(self, shape, rows, anchors, positions, page_tables, seed_cols):
        """One fused replay of ``shape`` at the narrowest rung covering max(positions) + P.

        Per live row r: seed hidden = the previous replay's verify hidden row
        ``r' * P_prev + seed_cols[r]`` (r' = the row's index in that replay) or
        the reseed-pool row when ``seed_cols[r]`` is None; then K_v assistant
        steps, one packed verify of [anchor, d1..dK] per row, argmax on device.

        Seed column convention: the adapter passes ``seed_cols[r] = committed
        count - 1 = accepted drafts``, i.e. the verify hidden of the LAST
        ACCEPTED token at its own position (column 0 = the anchor, column m =
        draft m), one position behind the new anchor (the bonus/correction
        token, whose own hidden no verify column holds). That is the
        pre-contract B=1 production mode ``_fused_shift_seed_row("current")``;
        column ``accepted + 1`` ("shift") would sit at the anchor's position
        but was computed from the REJECTED draft there. A reseed is exact.
        Returns a ServingStepResult; its ``logits`` is the decoder's shared
        verify-logits buffer, valid until the next serving_step or
        serving_reseed of this decoder.
        """
        self._srv_require(captured=True)
        shape = self._srv_shape(shape)
        B, K = shape
        P = K + 1
        live = self._srv_check_rows(shape, rows, anchors, positions)
        if len(seed_cols) != live:
            raise ValueError(f"shape {shape}: {len(seed_cols)} seed_cols for {live} live rows")
        top = max(int(p) for p in positions)
        width = serving_width_for(self._srv_widths[shape], top, K)
        if width is None:
            raise ValueError(
                f"shape {shape}: position {top} needs S_k >= {top + P + 1}, past the widest prepared rung "
                f"{max(self._srv_widths[shape])}; the adapter must decline proposals there"
            )
        rec = self._srv_records[(shape, width)]
        previous = None if self._srv_last is None else (self._srv_last["K"], self._srv_last["rows"])
        seed_idx = serving_seed_index(shape, rows, seed_cols, previous, self._srv_seeded)
        t0 = time.perf_counter()
        self._srv_upload(rec, rows, live, anchors, positions, page_tables, seed_idx)
        t1 = time.perf_counter()
        ttnn.execute_trace(self.mesh_device, rec["id"], cq_id=0, blocking=self._srv_replay_blocking)
        t2 = time.perf_counter()
        vx = ttnn.get_device_tensors(rec["verify_x"])[0] if self._tp > 1 else rec["verify_x"]
        vx = ttnn.to_torch(vx).to(torch.int64).reshape(B, P)
        gids = torch.tensor(self._ids_to_host(rec["vidx"], B * P), dtype=torch.int64).reshape(B, P)
        t3 = time.perf_counter()
        self._srv_phase_ms = {
            "replay_upload": (t1 - t0) * 1e3,
            "replay_exec": (t2 - t1) * 1e3,
            "replay_read": (t3 - t2) * 1e3,
        }
        if (
            self._srv_check_steps > 0
            and not self._srv_measuring
            and self._srv_checked.get((shape, int(width)), 0) < self._srv_check_steps
        ):
            zero_tables = not bool(torch.as_tensor(page_tables).any())
            self._srv_verify_check(shape, rec, rows, live, anchors, positions, gids, zero_tables=zero_tables)
        self._srv_last = {"shape": shape, "rows": [int(r) for r in rows], "K": K}
        return ServingStepResult(
            shape=shape,
            width=int(width),
            live=live,
            drafts=vx[:live, 1:].clone(),
            argmax_ids=gids[:live].clone(),
            logits=self._srv_logits,
            _reader=lambda rec=rec, live=live: self._srv_logits_to_host(rec, live),
        )

    def serving_step_phases(self):
        """The last ``serving_step``'s wall by sub-phase, in ms: ``replay_upload`` (the record's persistent inputs),
        ``replay_exec`` (the trace submit; the record's device time under GEMMA4_SPEC_REPLAY_BLOCKING=1) and
        ``replay_read`` (the drafts and argmax ids back to host, plus the device wait when the submit did not block).
        The adapter folds them into its stats line next to the ``replay`` phase they add up to."""
        return dict(self._srv_phase_ms)

    def serving_release(self):
        """Release every record's trace and buffer; idempotent."""
        recs = self._srv_records
        if recs is None:
            return
        self._srv_records = None
        self._srv_captured = False
        for tid in self._srv_seed_traces.values():
            ttnn.release_trace(self.mesh_device, tid)
        self._srv_seed_traces = {}
        for rec in recs.values():
            if rec.get("id") is not None:
                ttnn.release_trace(self.mesh_device, rec["id"])
            for key in ("v_ptl_slots", "verify_x", "vidx"):
                self._free_tensor_tree(rec.get(key))
        for S in self._srv_shape_bufs.values():
            for key in (
                "anchor_tok",
                "seed_idx",
                "seed_tok",
                "seed_pu",
                "seed_pi",
                "d_pu",
                "d_pi",
                "d_ptl_slots",
                "v_pos",
                "v_pos_rows",
                "write_idxs",
                "ring_base",
                "ring_tm",
            ):
                self._free_tensor_tree(S.get(key))
        self._srv_shape_bufs = {}
        self._srv_frozen = False
        for t in (
            self._srv_pool_v,
            self._srv_pool_s,
            self._srv_logits,
            *self._srv_iota.values(),
            *(t for c in self._srv_draft_consts.values() for t in c.values()),
        ):
            self._free_tensor_tree(t)
        self._srv_pool_v = self._srv_pool_s = self._srv_logits = None
        self._srv_iota = {}
        self._srv_uploaded = {}
        self._srv_last = None
        self._srv_phase_ms = {}
        self._srv_seeded = {}
        self._srv_checked = {}

    # ── serving internals ────────────────────────────────────────────────────
    def _srv_alloc_shape(self, shape):
        """Width-independent persistent inputs of one shape (shared by its width records)."""
        B, K = shape
        P = K + 1
        pu, pi = self._pos_tensors([0] * B)
        seed_pu, seed_pi = self._pos_tensors([0] * B)
        slots = self._srv_slots
        d_ptl_slots = [
            self._pv_from_torch(torch.zeros(B, self._srv_blocks, dtype=torch.int32), ttnn.int32)
            for _ in range(self._srv_n_slots)
        ]
        # The assistant reads each type's shared-KV layer: its table is that layer's slot.
        d_ptl = {lt: d_ptl_slots[slots[self.target.last_kv_layer_by_type[lt]]] for lt in self._pv_ring}
        ring = self._pv_ring.get("sliding_attention")
        ring_scalars = {}
        if ring:
            # per-row ring geometry for the in-trace sliding mask (serving_mask_plan)
            for key in ("ring_base", "ring_tm"):
                ring_scalars[key] = self._pv_from_torch(
                    torch.zeros(B, 1, 1, 1, dtype=torch.float32), ttnn.float32, layout=ttnn.TILE_LAYOUT
                )
        return {
            "B": B,
            "K": K,
            "P": P,
            **ring_scalars,
            "anchor_tok": self._tokens_tensor([0] * B),  # [1, B] uint32
            "seed_idx": self._tokens_tensor([0] * B),  # [1, B] uint32 rows of the hidden pool
            "d_pu": pu,  # [1, 32] uint32 drafter positions (B filled)
            "d_pi": pi,  # [B] int32
            # The reseed forward's own inputs (_srv_seed_rows): uploaded per reseed, never re-allocated.
            "seed_tok": self._tokens_tensor([0] * B),  # [1, B] uint32 anchor tokens
            "seed_pu": seed_pu,  # [1, 32] uint32 positions (B filled)
            "seed_pi": seed_pi,  # [B] int32
            "d_ptl": d_ptl,  # {layer_type: [B, blocks] int32} the shared-KV layer's slot table per type
            "d_ptl_slots": d_ptl_slots,  # one [B, blocks] int32 table per KV group slot
            "d_ptl_list": [d_ptl_slots[slot] for slot in slots],  # per target layer: its slot's table
            "v_pos": self._from_b(torch.zeros(1, B * P, dtype=torch.int64), ttnn.uint32),  # row b*P+p = c_b + p
            # the same positions per PACKED query row (b*H*P + h*P + p = c_b + p): the in-trace masks come out at
            # the packed height from these, with no row repeat of a key-wide tensor (_srv_masks_device)
            "v_pos_rows": self._from_b(torch.zeros(1, B * self._packed_H() * P, dtype=torch.int64), ttnn.uint32),
            "write_idxs": [self._from_b(torch.zeros(B, dtype=torch.int32), ttnn.int32) for _ in range(P)],
        }

    def _srv_alloc_width(self, shape, S, s_k):
        """Width-dependent persistent inputs of one (shape, S_k) record: the width-matched verify tables."""
        B, K, P = S["B"], S["K"], S["P"]
        # Width-matched to the type's mask with the type's OWN block size (see
        # serving_verify_table_widths): the packed SDPA attends the TABLE width
        # times that block size, and under hybrid groups the types differ.
        widths = serving_verify_table_widths(s_k, self._pv_bs_t, self._pv_ring, self._srv_blocks)
        slots = self._srv_slots
        layer_types = self._srv_layer_types()
        slot_type = {}
        for i, slot in enumerate(slots):
            slot_type.setdefault(slot, layer_types[i])
        v_ptl_slots = [
            self._pv_from_torch(torch.zeros(B, widths[slot_type[slot]], dtype=torch.int32), ttnn.int32)
            for slot in range(self._srv_n_slots)
        ]
        # The flat-table fallback reads each type's representative pool layer.
        v_ptl = {lt: v_ptl_slots[slots[self._pv_ring_rep[lt]]] for lt in self._pv_ring}
        return {
            "shape": shape,
            "S_k": int(s_k),
            "B": B,
            "K": K,
            "P": P,
            "v_ptl": v_ptl,  # {layer_type: [B, width_lt] int32} the representative layer's slot table per type
            "v_ptl_slots": v_ptl_slots,  # one [B, width of its type] int32 table per KV group slot
            "v_ptl_list": [v_ptl_slots[slot] for slot in slots],  # per target layer: its slot's table
            # The replay's outputs the host reads back (serving_step): the user-major verify input and its
            # argmax ids, allocated here, before any capture, and written in place by the body (ttnn.copy).
            # Taken from the body's own op outputs inside the capture they were the one allocation after the
            # decode-trace captures of the base warmup, and the guard's first armed check in serving released
            # every decode trace (12B Galaxy: 20 records x 2 = 30 x 128 B + 10 x 64 B = 4480 B per bank).
            "verify_x": self._from_b(torch.zeros(1, B * P, dtype=torch.int64), ttnn.uint32),  # [1, B*P] uint32
            "vidx": self._from_b(torch.zeros(1, 1, B * P, dtype=torch.int64), ttnn.uint32),  # [1, 1, B*P] uint32
            "id": None,
        }

    def _srv_publish_hidden(self, hidden, pool):
        """Copy a [1, 1, rows<=32, backbone] hidden into a 32-row pool buffer (padding rows zero)."""
        rows = int(hidden.shape[2])
        if rows > SPEC_ROW_TILE:
            raise ValueError(f"hidden has {rows} rows; the pool holds {SPEC_ROW_TILE}")
        src = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
        padded = (
            ttnn.pad(src, [(0, 0), (0, 0), (0, SPEC_ROW_TILE - rows), (0, 0)], value=0.0)
            if rows < SPEC_ROW_TILE
            else src
        )
        ttnn.assign(padded, pool)
        if not _same_device_buffer(padded, src):
            padded.deallocate(True)
        if not _same_device_buffer(src, hidden):
            src.deallocate(True)

    def _srv_eager_forward(self, shape, tokens, positions):
        """Eager (plain, one row per batch row) target forward of ``tokens`` at ``positions`` (B entries) through
        the shape's drafter tables, on the shape's persistent seed inputs; returns ``(logits, hidden)``."""
        S = self._srv_shape_bufs[shape]
        B = S["B"]
        if len(tokens) != B or len(positions) != B:
            raise ValueError(f"shape {shape}: seed needs {B} tokens/positions, got {len(tokens)}/{len(positions)}")
        # The forward's inputs are the shape's persistent buffers, refreshed from host staging tensors: a
        # reseed allocates nothing on the device itself (serving_prepare), only the eager forward's own
        # transients come and go.
        self._srv_seed_upload(shape, tokens, positions)
        return self._srv_seed_forward(shape)

    def _srv_seed_upload(self, shape, tokens, positions):
        """Refresh the shape's persistent seed inputs (tokens, positions) from host."""
        S = self._srv_shape_bufs[shape]
        host_tok = self._host_tokens([int(t) for t in tokens])
        ttnn.copy_host_to_device_tensor(host_tok, S["seed_tok"])
        host_tok.deallocate(True)
        hpu, hpi = self._host_pos([int(p) for p in positions])
        ttnn.copy_host_to_device_tensor(hpu, S["seed_pu"])
        ttnn.copy_host_to_device_tensor(hpi, S["seed_pi"])
        hpu.deallocate(True)
        hpi.deallocate(True)

    def _srv_seed_forward(self, shape):
        """The device-only seed forward of the shape's persistent inputs (what the seed trace captures)."""
        S = self._srv_shape_bufs[shape]
        return self.target.ttnn_verify_forward(
            x=S["seed_tok"],
            current_pos=S["seed_pu"],
            current_pos_cache=S["seed_pi"],
            page_table=S["d_ptl"][self._srv_flat_type],
            kv_cache=self.tt_kv_cache,
            page_tables_per_layer=S["d_ptl_list"],
        )

    def _srv_seed_body(self, shape):
        """The seed forward and its publish into the reseed pool: the body of the shape's seed trace."""
        logits, hidden = self._srv_seed_forward(shape)
        logits.deallocate(True)
        self._srv_publish_hidden(hidden, self._srv_pool_s)
        hidden.deallocate(True)

    def _srv_verify_check(self, shape, rec, rows, live, anchors, positions, gids, zero_tables=False):
        """GEMMA4_SPEC_VERIFY_CHECK: the packed verify's row-0 argmax of every live row against the plain forward
        of the same anchor at the same position (keys <= position in both; the forward rewrites the anchor's K/V
        with the same values), counted per (shape, rung) over served replays. Returns the disagreements
        ``(row, position, plain id, packed id)``; logs them (a zero-table replay is named as such: every key on
        block 0 makes the rows' own draft K/V alias into every earlier block, no evidence either way)."""
        S = self._srv_shape_bufs[shape]
        B = S["B"]
        logits, hidden = self._srv_eager_forward(
            shape, serving_pad_rows([int(a) for a in anchors], B), serving_pad_rows([int(p) for p in positions], B)
        )
        hidden.deallocate(True)
        idx = self._argmax_last(logits, rows=B)  # [1, 1, B] uint32
        logits.deallocate(True)
        plain = self._ids_to_host(idx, B)[:live]
        idx.deallocate(True)
        packed = [int(gids[r, 0]) for r in range(live)]
        bad = [(int(rows[r]), int(positions[r]), plain[r], packed[r]) for r in range(live) if plain[r] != packed[r]]
        from loguru import logger as _lg

        key = (shape, int(rec["S_k"]))
        n = self._srv_checked.get(key, 0) + 1
        self._srv_checked[key] = n
        self._srv_check_last = bad
        line = (
            f"[spec-check] shape {shape} S_k={rec['S_k']} served replay {n} of {self._srv_check_steps}: the packed "
            f"verify's row-0 argmax agrees with the plain forward on {live - len(bad)} of {live} rows"
        )
        if zero_tables:
            line += " on a zero-table replay (every key on block 0, the rows' own draft K/V alias into every earlier block: no evidence either way)"
        if bad:
            _lg.warning(line + "; disagreements (row, position, plain, packed): " + ", ".join(map(str, bad)))
        else:
            _lg.info(line)
        return bad

    def _srv_seed_rows(self, shape, tokens, positions):
        """Target forward of ``tokens`` at ``positions`` (B_v entries) through the shape's drafter tables; its
        post-norm hidden fills the reseed pool rows. The shape's seed trace once serving_capture made it, the
        eager forward before (the pools are made by the first seed of serving_prepare)."""
        tid = self._srv_seed_traces.get(shape)
        if tid is not None:
            if self._srv_pool_s is None:
                raise RuntimeError(
                    "the hidden pools do not exist after serving_prepare: a reseed never allocates (the first eager "
                    "seed of serving_prepare makes them)"
                )
            self._srv_seed_upload(shape, tokens, positions)
            # blocking: the reseed phase is the seed's device time (the cost table's entry cost), nothing overlaps it
            ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=True)
            return
        logits, hidden = self._srv_eager_forward(shape, tokens, positions)
        logits.deallocate(True)
        if self._srv_pool_v is None:
            if getattr(self, "_srv_frozen", False):
                raise RuntimeError(
                    "the hidden pools do not exist after serving_prepare: a reseed never allocates (the first eager "
                    "seed of serving_prepare makes them)"
                )
            if hidden.dtype != ttnn.bfloat16:
                raise TypeError(f"the verify hidden is {hidden.dtype}; the hidden pool (ttnn.assign) needs bfloat16")
            self._srv_backbone = int(hidden.shape[-1])
            zeros = torch.zeros(1, 1, SPEC_ROW_TILE, self._srv_backbone)
            self._srv_pool_v = self._pv_from_torch(zeros, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self._srv_pool_s = self._pv_from_torch(zeros, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._srv_publish_hidden(hidden, self._srv_pool_s)
        hidden.deallocate(True)

    def _srv_body(self, rec):
        """The fused record graph over persistent buffers (compile and capture).

        Returns the record's persistent outputs ``(rec["verify_x"], rec["vidx"])``
        (``[1, B*P]`` and ``[1, 1, B*P]`` uint32, allocated by ``_srv_alloc_width``),
        written in place by the body's last two ops: the graph allocates nothing
        that outlives it, so a capture binds only to buffers that existed before it.
        The verify logits and hidden go to the shared pools (``_srv_publish_*``).
        """
        S = self._srv_shape_bufs[rec["shape"]]
        B, K, P = rec["B"], rec["K"], rec["P"]
        # 1. seed gather from the hidden pools through the uploaded row index
        h = self._srv_seed_gather(S)  # [1, 1, B, bb]
        # 2. K batched drafter steps, argmax + re-embed on device
        tok = S["anchor_tok"]
        draft_cols = []
        for _ in range(K):
            sharded = self._srv_sharded_draft_argmax and self._tp > 1
            logits, h_next = self.assistant.step(
                tok, h, self._shared_kv, S["d_ptl"], S["d_pu"], S["d_pi"], gather_logits=not sharded
            )
            h.deallocate(True)
            h = h_next
            # [1, 1, B] uint32: the sharded argmax exchanges the shards' candidates, the gathered one scans the vocab
            idx = self._draft_argmax_sharded(logits, rows=B) if sharded else self._argmax_last(logits, rows=B)
            logits.deallocate(True)
            tok = ttnn.reshape(idx, (1, B))
            draft_cols.append(tok)
        h.deallocate(True)
        # 3. user-major verify input [1, B*P] (row b*P+p) from the P columns [1, B]
        verify_x = self._srv_assemble_verify_x(S, draft_cols)
        # 4. masks built in-trace at the packed height [B, 1, H*P, S] (row h*P+p
        # reads position p), then ONE packed verify.
        mask_full, mask_slide = self._srv_masks_device(S, rec)
        vlogits, vhidden = self.target.ttnn_packed_verify_forward(
            x=verify_x,
            position_idx=S["v_pos"],
            attn_mask_full=mask_full,
            attn_mask_sliding=mask_slide,
            packed_p=P,
            page_table=rec["v_ptl"][self._srv_flat_type],
            kv_cache=self.tt_kv_cache,
            kv_write_idxs=S["write_idxs"],
            embed_idx_full=None,
            embed_idx_sliding=None,
            hot_pt=None,
            page_tables_per_layer=rec["v_ptl_list"],
        )
        mask_full.deallocate(True)
        mask_slide.deallocate(True)
        # 5. publish this replay's logits (the shared readback buffer) and its verify
        # hidden (next replay's seed gather) before anything consumes them; the
        # record keeps neither.
        self._srv_publish_logits(vlogits)
        self._srv_publish_hidden(vhidden, self._srv_pool_v)
        vhidden.deallocate(True)
        vidx = self._argmax_last(vlogits, rows=B * P)  # [1, 1, B*P] uint32
        vlogits.deallocate(True)
        # 6. the outputs into the record's persistent handles (allocated before any capture)
        ttnn.copy(verify_x, rec["verify_x"])
        verify_x.deallocate(True)
        ttnn.copy(vidx, rec["vidx"])
        vidx.deallocate(True)
        return rec["verify_x"], rec["vidx"]

    def _srv_draft_consts_for(self, width, vocab_slice):
        """The sharded argmax's persistent rows for a gathered width ``width`` (``tp`` slices of ``width / tp``
        columns, a slice's maximum in its first column): the candidate mask (0 at a slice's first column, a large
        negative elsewhere), the fp32 column iota and the fp32 offset of each column's slice in the vocab."""
        key = (int(width), int(vocab_slice))
        consts = self._srv_draft_consts.get(key)
        if consts is None:
            if getattr(self, "_srv_frozen", False):
                raise RuntimeError(
                    f"no sharded-argmax rows for a gathered width of {width} after serving_prepare: the eager "
                    f"bodies made them ({sorted(self._srv_draft_consts)}); a replay never allocates"
                )
            per = int(width) // self._tp
            cols = torch.arange(int(width), dtype=torch.float32)
            mask = torch.where(cols % per == 0, 0.0, -30000.0).reshape(1, 1, 1, -1).expand(1, 1, SPEC_ROW_TILE, -1)
            iota = cols.reshape(1, 1, 1, -1).expand(1, 1, SPEC_ROW_TILE, -1)
            offset = ((cols // per) * float(vocab_slice)).reshape(1, 1, 1, -1).expand(1, 1, SPEC_ROW_TILE, -1)
            consts = {
                "mask": self._pv_from_torch(mask.contiguous(), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                "iota": self._pv_from_torch(iota.contiguous(), ttnn.float32, layout=ttnn.TILE_LAYOUT),
                "offset": self._pv_from_torch(offset.contiguous(), ttnn.float32, layout=ttnn.TILE_LAYOUT),
            }
            self._srv_draft_consts[key] = consts
        return consts

    def _draft_argmax_sharded(self, logits, rows):
        """argmax over the vocab of the drafter's TP-sharded logits ``[1, 1, rows, vocab / tp]`` (TILE, one slice
        per chip): ``[1, 1, rows]`` uint32 ROW_MAJOR, the same on every chip.

        Per chip the slice's argmax (the multicore ROW_MAJOR argmax over ``vocab / tp`` columns) and its maximum;
        both all-gathered over the TP axis (a 32-row tile per chip), the winning slice picked by an argmax over
        the gathered maxima with the non-leading columns masked down, and its slice index offset by the slice's
        first vocab id through a one-hot of the winning column. Ties between slices go to the lowest slice.
        """
        R32 = SPEC_ROW_TILE
        vocab_slice = int(logits.shape[-1])
        src, padded = logits, None
        if int(logits.shape[2]) < R32:
            padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, R32 - int(logits.shape[2])), (0, 0)], value=0.0)
            src = padded
        local_idx = self._argmax_last(src, R32)  # [1, 1, 32] uint32 RM: the index within the slice
        local_max = ttnn.max(src, dim=-1, keepdim=True)  # [1, 1, 32, 1] bf16 TILE
        if padded is not None and not _same_device_buffer(padded, logits):
            padded.deallocate(True)
        idx_t = ttnn.to_layout(ttnn.reshape(local_idx, (1, 1, R32, 1)), ttnn.TILE_LAYOUT)
        idx_f = ttnn.typecast(idx_t, ttnn.float32)  # [1, 1, 32, 1] fp32 TILE (exact: indices < 2**24)
        idx_t.deallocate(True)
        local_idx.deallocate(True)
        vals_g = ccl_allgather(local_max, self.assistant.mesh_config, self.assistant.ccl_manager, dim=3)
        idx_g = ccl_allgather(idx_f, self.assistant.mesh_config, self.assistant.ccl_manager, dim=3)
        local_max.deallocate(True)
        idx_f.deallocate(True)
        consts = self._srv_draft_consts_for(int(vals_g.shape[-1]), vocab_slice)
        masked = ttnn.add(vals_g, consts["mask"])  # only a slice's leading column can win
        vals_g.deallocate(True)
        best = self._argmax_last(masked, R32)  # [1, 1, 32] uint32 RM: the winning column
        masked.deallocate(True)
        best_t = ttnn.to_layout(ttnn.reshape(best, (1, 1, R32, 1)), ttnn.TILE_LAYOUT)
        best_f = ttnn.typecast(best_t, ttnn.float32)  # [1, 1, 32, 1]
        best_t.deallocate(True)
        best.deallocate(True)
        diff = ttnn.subtract(consts["iota"], best_f)
        best_f.deallocate(True)
        onehot = ttnn.eq(diff, 0.0)  # [1, 1, 32, W] fp32: 1.0 at the winning column
        diff.deallocate(True)
        picked = ttnn.multiply(idx_g, onehot)
        idx_g.deallocate(True)
        picked_idx = ttnn.sum(picked, dim=-1, keepdim=True)  # the winner's index within its slice
        picked.deallocate(True)
        offsets = ttnn.multiply(consts["offset"], onehot)
        onehot.deallocate(True)
        offset = ttnn.sum(offsets, dim=-1, keepdim=True)  # the winner's slice start
        offsets.deallocate(True)
        glob = ttnn.add(picked_idx, offset)  # [1, 1, 32, 1] fp32
        picked_idx.deallocate(True)
        offset.deallocate(True)
        glob_u = ttnn.typecast(glob, ttnn.uint32)
        glob.deallocate(True)
        glob_rm = ttnn.to_layout(glob_u, ttnn.ROW_MAJOR_LAYOUT)
        glob_u.deallocate(True)
        out = ttnn.reshape(glob_rm, (1, 1, R32))
        if rows < R32:
            sliced = ttnn.slice(out, [0, 0, 0], [1, 1, rows])
            if not _same_device_buffer(sliced, out):
                out.deallocate(True)
            out = sliced
        return out

    def _srv_seed_gather(self, S):
        """Stage 1 of the body: the drafter seed [1, 1, B, backbone] = rows ``seed_idx`` of [pool_v | pool_s].

        ``ttnn.embedding`` takes the table as [1, 1, num_rows, width], so the
        4D concat of the two 32-row pools is the table itself (the dFlash ctx
        merge pattern); row r of the result is pool row ``seed_idx[r]``, the
        previous replay's verify hidden ``r' * P_prev + col`` or the reseed
        hidden ``32 + reseed row`` (serving_seed_index).
        """
        table = ttnn.concat([self._srv_pool_v, self._srv_pool_s], dim=2)  # [1, 1, 64, backbone]
        h = ttnn.unsqueeze_to_4D(ttnn.embedding(S["seed_idx"], table, layout=ttnn.TILE_LAYOUT))  # [1, 1, B, bb]
        table.deallocate(True)
        return h

    def _srv_assemble_verify_x(self, S, draft_cols):
        """Stage 3 of the body: the user-major verify input [1, B*P] (row b*P+p = [anchor_b, d1_b..dK_b])
        from the anchor column and the K draft columns, each [1, B] uint32."""
        B = S["B"]
        P = len(draft_cols) + 1
        cols4d = [ttnn.reshape(S["anchor_tok"], (1, 1, 1, B))] + [ttnn.reshape(c, (1, 1, 1, B)) for c in draft_cols]
        stacked = ttnn.concat(cols4d, dim=2)  # [1, 1, P, B]
        um = ttnn.permute(stacked, (0, 1, 3, 2))  # [1, 1, B, P]
        stacked.deallocate(True)
        verify_x = ttnn.reshape(um, (1, B * P))
        if not _same_device_buffer(verify_x, um):
            # A same-volume reshape is a view of ``um``: freeing the source would free
            # the verify input (inside a capture, the replay then reuses the storage
            # the host reads the drafts from).
            um.deallocate(True)
        return verify_x

    def _srv_publish_logits(self, vlogits):
        """Copy the [1, 1, B*P, V_padded] verify logits into the shared 32-row logits buffer (allocated on
        the first eager body at the logits' own width and dtype)."""
        if self._srv_logits is None:
            if getattr(self, "_srv_frozen", False):
                raise RuntimeError(
                    "the shared logits buffer does not exist after serving_prepare: a replay never allocates (the "
                    "first eager body of serving_prepare makes it)"
                )
            if vlogits.dtype != ttnn.bfloat16:
                raise TypeError(
                    f"the verify logits are {vlogits.dtype}; the shared logits buffer (ttnn.assign) needs bfloat16"
                )
            zeros = torch.zeros(1, 1, SPEC_ROW_TILE, int(vlogits.shape[-1]))
            self._srv_logits = self._pv_from_torch(zeros, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self._srv_publish_hidden(vlogits, self._srv_logits)

    def _srv_iota_for(self, n):
        """Persistent [1, 1, 1, n] fp32 key iota 0..n-1 (dFlash ``pv_iota``; fp32 keeps every position exact)."""
        t = self._srv_iota.get(int(n))
        if t is None:
            if getattr(self, "_srv_frozen", False):
                raise RuntimeError(
                    f"no mask iota of length {n} after serving_prepare: the eager bodies made one per key-chunk "
                    f"length of the ladder ({sorted(self._srv_iota)}); a replay never allocates"
                )
            t = self._pv_from_torch(
                torch.arange(int(n), dtype=torch.float32).reshape(1, 1, 1, int(n)),
                ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            )
            self._srv_iota[int(n)] = t
        return t

    @staticmethod
    def _srv_mask_chunks(n_keys):
        """[(j0, n)] key chunks of SPEC_MASK_KEY_CHUNK plus the remainder (every piece tile-aligned)."""
        out, j0 = [], 0
        while j0 < int(n_keys):
            n = min(SPEC_MASK_KEY_CHUNK, int(n_keys) - j0)
            out.append((j0, n))
            j0 += n
        return out

    def _srv_masks_device(self, S, rec):
        """In-trace additive masks [B, 1, H*P, S_k] (full) and [B, 1, H*P, S_r] (sliding) for the record.

        Per key chunk, ``diff = iota - q`` (fp32, [B, 1, H*P, n] by the binary
        broadcast of [1, 1, 1, n] against the [B, 1, H*P, 1] packed-row query
        positions, row h*P+p = c_b + p, uploaded as ``v_pos_rows``); full =
        gtz(diff); sliding (unbounded) adds gez(-(diff + W)); under a ring the
        keys are ring slots whose position is ``base + j - ring * gtz(j - tm)``
        (the uploaded per-row ring scalars) and ltz(position) masks the
        untouched tail. Indicators are cast to bf16 before the NEG scale so each
        chunk holds at most two fp32 intermediates, and the chunks concatenate
        along the keys into the mask. Every wide tensor of the build is
        tile-aligned (rows pad to a tile, widths are multiples of 32): the
        earlier build made the mask at [B, 1, P, S_k] and repeated it H times
        along the padded row dim, which ttnn.repeat runs as untilize -> row-major
        repeat -> tilize over the full key width (526 KB rows at S_k 263168, two
        row tiles once H*P > 32); the 31B at TP4 (8 local heads x 6 rows) was the
        one served geometry on that path at the top rung, and the one whose 256K
        verify went wrong. ``serving_masks_from_plan(..., h=H)`` is this
        sequence in torch.
        """
        B, P = S["B"], S["P"]
        H = self._packed_H()
        W = float(self._pv_window)
        ring = self._pv_ring.get("sliding_attention")
        neg = float(SPEC_MASK_NEG)
        # query positions per packed row [B, 1, H*P, 1] fp32 from the uploaded [1, B*H*P] row positions
        q = ttnn.typecast(
            ttnn.to_layout(ttnn.reshape(S["v_pos_rows"], (B, 1, H * P, 1)), ttnn.TILE_LAYOUT), ttnn.float32
        )
        full_parts, slide_parts = [], []
        for j0, n in self._srv_mask_chunks(rec["S_k"]):
            iota = self._srv_iota_for(n)
            qc = ttnn.sub(q, float(j0)) if j0 else q  # chunk-relative query positions
            diff = ttnn.sub(iota, qc)  # [B, 1, P, n] fp32: key - query
            if qc is not q:
                qc.deallocate(True)
            gt = ttnn.gtz(diff)
            gt_b = ttnn.typecast(gt, ttnn.bfloat16)
            gt.deallocate(True)
            full_parts.append(ttnn.multiply(gt_b, neg))
            if not ring:
                shifted = ttnn.add(diff, W)
                flipped = ttnn.multiply(shifted, -1.0)
                shifted.deallocate(True)
                far = ttnn.gez(flipped)  # key <= query - W
                flipped.deallocate(True)
                far_b = ttnn.typecast(far, ttnn.bfloat16)
                far.deallocate(True)
                ind = ttnn.add(gt_b, far_b)  # exclusive conditions: 0/1
                far_b.deallocate(True)
                slide_parts.append(ttnn.multiply(ind, neg))
                ind.deallocate(True)
            gt_b.deallocate(True)
            diff.deallocate(True)
        if ring:
            for j0, n in self._srv_mask_chunks(ring):
                iota = self._srv_iota_for(n)
                jr = ttnn.add(iota, float(j0)) if j0 else iota  # absolute ring slot index
                past = ttnn.sub(jr, S["ring_tm"])  # [B, 1, 1, n]
                wrapped = ttnn.gtz(past)
                past.deallocate(True)
                wrap = ttnn.multiply(wrapped, -float(ring))
                wrapped.deallocate(True)
                shifted = ttnn.add(jr, S["ring_base"])
                if jr is not iota:
                    jr.deallocate(True)
                pos = ttnn.add(shifted, wrap)  # position held by each ring slot
                shifted.deallocate(True)
                wrap.deallocate(True)
                diff = ttnn.sub(pos, q)  # [B, 1, P, n]
                gt = ttnn.gtz(diff)
                gt_b = ttnn.typecast(gt, ttnn.bfloat16)
                gt.deallocate(True)
                shifted = ttnn.add(diff, W)
                diff.deallocate(True)
                flipped = ttnn.multiply(shifted, -1.0)
                shifted.deallocate(True)
                far = ttnn.gez(flipped)
                flipped.deallocate(True)
                far_b = ttnn.typecast(far, ttnn.bfloat16)
                far.deallocate(True)
                untouched = ttnn.ltz(pos)  # slots never written (position < 0)
                pos.deallocate(True)
                neg_b = ttnn.typecast(untouched, ttnn.bfloat16)
                untouched.deallocate(True)
                acc = ttnn.add(gt_b, far_b)
                gt_b.deallocate(True)
                far_b.deallocate(True)
                acc2 = ttnn.add(acc, neg_b)  # 0..2
                acc.deallocate(True)
                neg_b.deallocate(True)
                ind = ttnn.gtz(acc2)  # any condition -> 1
                acc2.deallocate(True)
                slide_parts.append(ttnn.multiply(ind, neg))
                ind.deallocate(True)
        q.deallocate(True)
        outs = []
        for parts in (full_parts, slide_parts):
            mask = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=3)  # [B, 1, H*P, S] tile-aligned
            if len(parts) > 1:
                for part in parts:
                    part.deallocate(True)
            outs.append(mask)
        return outs[0], outs[1]

    def _srv_copy_if_changed(self, key, host, dev, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        """Upload the small ``host`` tensor into the persistent ``dev`` buffer unless the last upload under
        ``key`` was equal (tables, positions and ring scalars repeat across steps)."""
        prev = self._srv_uploaded.get(key)
        if prev is not None and prev.shape == host.shape and torch.equal(prev, host):
            return
        h = self._pv_from_torch(host, dtype, layout=layout, device=False)
        ttnn.copy_host_to_device_tensor(h, dev)
        h.deallocate(True)
        self._srv_uploaded[key] = host.clone()

    def _srv_slot_table(self, slot, rows, page_tables):
        """Host [B, cols] int64 table of one KV group slot for the step's rows: the installed per-layer table of
        the slot's first layer (every layer of a slot carries the same table, serving_check_group_tables),
        rows selected; else the flat ``page_tables`` as given."""
        installed = self._srv_per_layer
        src = None
        if installed is not None:
            idx = self._srv_slots.index(slot)
            src = installed[idx] if idx < len(installed) else None
        if src is None:
            return page_tables
        B = int(page_tables.shape[0])
        return serving_select_rows(src, rows, B, int(src.shape[1]), allow_clip=False)

    def _srv_upload_tables(self, shape, rows, live, page_tables, rec):
        """Upload the drafter tables of ``shape`` (and the verify tables of ``rec`` when given), one per KV group slot."""
        S = self._srv_shape_bufs[shape]
        B = S["B"]
        if not hasattr(page_tables, "dim") or page_tables.dim() != 2 or int(page_tables.shape[0]) != B:
            got = tuple(page_tables.shape) if hasattr(page_tables, "shape") else type(page_tables).__name__
            raise ValueError(f"shape {shape}: page_tables must be [{B}, blocks], got {got}")
        pt = page_tables.to(torch.int64)
        if live < B and bool(pt[live:].any()):
            raise ValueError(f"shape {shape}: padding rows {live}..{B - 1} of page_tables must be zero (block 0)")
        if live < B:
            # Padding rows mirror live row 0 (serving_pad_rows): their verify
            # K/V then lands, byte-identical, where row 0 writes its own.
            pt = pt.clone()
            pt[live:] = pt[0]
        if self._srv_per_layer is None and any(self._pv_ring.values()):
            raise ValueError(
                "bounded sliding target: serving_refresh_page_tables(per-layer tables) must run before the first "
                "reseed/replay, or the sliding layers would read the flat table instead of their ring pool"
            )
        for slot in range(self._srv_n_slots):
            d_host = self._srv_slot_table(slot, rows, pt)
            d_host = serving_select_rows(d_host, list(range(B)), B, self._srv_blocks, allow_clip=False)
            if live < B:
                d_host[live:] = d_host[0]
            self._srv_copy_if_changed(("d", shape, slot), d_host.to(torch.int32), S["d_ptl_slots"][slot], ttnn.int32)
            if rec is None:
                continue
            v_host = self._srv_slot_table(slot, rows, pt)
            if live < B:
                v_host = v_host.clone()
                v_host[live:] = v_host[0]
            width = int(rec["v_ptl_slots"][slot].shape[-1])
            # Fitted, not checked: the verify table is width-matched to its mask
            # on purpose (serving_verify_table_widths); columns past the adapter's
            # table are zero (block 0) and columns past the mask are never attended.
            v_host = serving_select_rows(v_host, list(range(B)), B, width, allow_clip=True)
            self._srv_copy_if_changed(
                ("v", shape, rec["S_k"], slot), v_host.to(torch.int32), rec["v_ptl_slots"][slot], ttnn.int32
            )

    def _srv_upload(self, rec, rows, live, anchors, positions, page_tables, seed_idx):
        """Refresh every persistent input the record's trace reads for one replay."""
        shape = rec["shape"]
        S = self._srv_shape_bufs[shape]
        B, P = S["B"], S["P"]
        toks = serving_pad_rows([int(t) for t in anchors], B)
        pos = serving_pad_rows([int(p) for p in positions], B)
        for host, dev in ((self._host_tokens(toks), S["anchor_tok"]), (self._host_tokens(seed_idx), S["seed_idx"])):
            ttnn.copy_host_to_device_tensor(host, dev)
            host.deallocate(True)
        hpu, hpi = self._host_pos(pos)
        ttnn.copy_host_to_device_tensor(hpu, S["d_pu"])
        ttnn.copy_host_to_device_tensor(hpi, S["d_pi"])
        hpu.deallocate(True)
        hpi.deallocate(True)
        vpos = torch.tensor([[c + p for c in pos for p in range(P)]], dtype=torch.int64)
        self._srv_copy_if_changed(("vpos", shape), vpos, S["v_pos"], ttnn.uint32)
        vrows = serving_mask_rows(pos, P, self._packed_H())
        self._srv_copy_if_changed(("vrows", shape), vrows, S["v_pos_rows"], ttnn.uint32)
        for p in range(P):
            wi = torch.tensor([c + p for c in pos], dtype=torch.int32)
            self._srv_copy_if_changed(("widx", shape, p), wi, S["write_idxs"][p], ttnn.int32)
        ring = self._pv_ring.get("sliding_attention")
        if ring:
            plan = serving_mask_plan(pos, P, ring=ring)
            for key in ("ring_base", "ring_tm"):
                host = plan[key.split("_")[1]].to(torch.float32).reshape(B, 1, 1, 1)
                self._srv_copy_if_changed((key, shape), host, S[key], ttnn.float32, layout=ttnn.TILE_LAYOUT)
        self._srv_upload_tables(shape, rows, live, page_tables, rec)

    def _srv_logits_to_host(self, rec, live):
        """[live, P, V] bf16 host readback of the shared verify-logits buffer (rows [0, B*P) of its 32)."""
        B, P = rec["B"], rec["P"]
        t = self._logits_to_host(self._srv_logits)  # [..., :V], softcap + TP gather already applied on device
        V = int(t.shape[-1])
        rows = int(t.numel() // V)
        if rows != SPEC_ROW_TILE:
            raise ValueError(
                f"the shared logits buffer holds {rows} rows, expected {SPEC_ROW_TILE} (shape {tuple(t.shape)})"
            )
        return t.reshape(rows, V)[: B * P].reshape(B, P, V)[:live].to(torch.bfloat16)

    # ── batched (B>1) speculative decode ─────────────────────────────────────
    def _seed_batched(self, tokens, positions):
        """Batched drafter seed: one batch=B target verify of each user's anchor
        token at its position (idempotent re-write of the prompt's last KV).
        Returns the post-norm hidden [1,1,B,backbone]; slice row b for user b's
        recurrent drafter seed."""
        x = self._tokens_tensor(tokens)  # [1,B]
        pu, pi = self._pos_tensors(positions)  # pu [1,32] (B filled), pi [B]
        pt = self._page_table_users(len(tokens))  # [B, blocks] distinct
        logits, hidden = self.target.ttnn_verify_forward(
            x=x,
            current_pos=pu,
            current_pos_cache=pi,
            page_table=pt,
            kv_cache=self.tt_kv_cache,
            # Distinct rows per user: falling through to the model's batch-1
            # per-layer tables under bounded sliding slices row b>=1 of a 1-row
            # table (see _capture_fused_trace).
            page_tables_per_layer=self._page_tables_per_layer_users(len(tokens)),
        )
        logits.deallocate(True)
        for t in (x, pu, pi, pt):
            t.deallocate(True)
        return hidden  # [1,1,B,backbone]

    def _packed_H(self):
        """Local (per-device) query-head count for the packed additive mask."""
        return self.target.layers[0].self_attn.config.num_attention_heads // self._tp

    def _packed_mask_host(self, cs, B, P, H, window, S_k):
        """Host bf16 additive masks [B,1,H*P,S_k] for the packed verify (head-major
        rows h*P+p; per-user causal bound c_b+p, + sliding-window lower bound)."""
        NEG = -1e9
        j = torch.arange(S_k)
        mf = torch.empty(B, 1, H * P, S_k)
        ms = torch.empty(B, 1, H * P, S_k)
        for b in range(B):
            rf = torch.empty(P, S_k)
            rs = torch.empty(P, S_k)
            for p in range(P):
                upper = cs[b] + p
                rf[p] = torch.where(j <= upper, 0.0, NEG)
                rs[p] = torch.where((j <= upper) & (j > upper - window), 0.0, NEG)
            mf[b, 0] = rf.repeat(H, 1)
            ms[b, 0] = rs.repeat(H, 1)
        return mf.to(torch.bfloat16), ms.to(torch.bfloat16)

    def _verify_packed_batched(self, tokens_b, cs):
        """Packed verify for B users at PER-USER anchor positions (ragged).

        Folds each user's P=K+1 candidates into the query-head dim of a batch=B
        forward — B KV reads total (vs B*(K+1) for the batch-alias verify). Uses
        the fallback per-position ``paged_update_cache`` write (no staging), so
        no per-user staging is needed.

        Args:
            tokens_b: list[B] of P-length token-id lists ([anchor, d1..dK]).
            cs:       list[B] of anchor positions c_b (user b verifies c_b..c_b+K).

        Returns:
            (logits_host [B*P, vocab], vhidden device [1,1,B*P,backbone]); row
            u*P+p is user u's p-th packed position.
        """
        target = self.target
        B = len(tokens_b)
        P = len(tokens_b[0])
        H = self._packed_H()
        window = target.hf_config.sliding_window
        max_c = max(cs)
        # S_k (mask key length): cover positions 0..max_c+K, multiple of 64.
        S_k = ((max_c + P + 63) // 64) * 64

        x = self._from_b(
            torch.tensor([tok for toks in tokens_b for tok in toks], dtype=torch.int64).reshape(1, B * P), ttnn.uint32
        )
        position_idx = self._from_b(
            torch.tensor([[cs[b] + p for b in range(B) for p in range(P)]], dtype=torch.int64), ttnn.uint32
        )
        mf, ms = self._packed_mask_host(cs, B, P, H, window, S_k)
        mask_full = self._from_b(mf, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        mask_slide = self._from_b(ms, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        write_idxs = [
            self._from_b(torch.tensor([cs[b] + p for b in range(B)], dtype=torch.int32), ttnn.int32) for p in range(P)
        ]
        # Width-match the table to the masks' S_k (see _page_table_users).
        _bs = int(self.tt_kv_cache[0][0].padded_shape[2])
        pt = self._page_table_users(B, width=S_k // _bs)
        logits, vhidden = target.ttnn_packed_verify_forward(
            x=x,
            position_idx=position_idx,
            attn_mask_full=mask_full,
            attn_mask_sliding=mask_slide,
            packed_p=P,
            page_table=pt,
            kv_cache=self.tt_kv_cache,
            kv_write_idxs=write_idxs,
            embed_idx_full=None,
            embed_idx_sliding=None,
            hot_pt=None,
        )
        lh = self._logits_to_host(logits).reshape(B * P, -1)
        logits.deallocate(True)
        for t in (x, position_idx, mask_full, mask_slide, pt, *write_idxs):
            t.deallocate(True)
        return lh, vhidden

    def _draft_batched(self, anchor_tokens, anchor_hidden_b, anchor_positions):
        """Batched drafter: propose K tokens for ALL B users in ONE batch=B chain.

        Runs K batched drafter steps (each ``assistant.step`` at batch=B, the
        drafter cross-attending into every user's own KV via distinct page-table
        rows). The recurrent hidden [1,1,B,backbone] is carried across steps; the
        per-step argmax over B rows is read to host to feed the next step. This
        amortizes the (tiny) drafter over B users instead of B sequential batch=1
        loops — the key to batched throughput.

        Args:
            anchor_tokens: list[B] committed token ids to draft from.
            anchor_hidden_b: [1,1,B,backbone] batched drafter seed (borrowed).
            anchor_positions: list[B] anchor positions (fixed across the K steps).

        Returns:
            drafts_b: list[B] of K-length draft-token lists.
        """
        K = self.draft_len
        B = len(anchor_tokens)
        pos_u, pos_i = self._pos_tensors(anchor_positions)  # pu [1,32] (B filled), pi [B]
        pt = self._page_table_users(B)
        page_tables = self._shared_kv_page_tables(pt)

        drafts_b = [[] for _ in range(B)]
        tok_tt = self._tokens_tensor(anchor_tokens)  # [1,B]
        h = anchor_hidden_b
        owns_h = False
        for _ in range(K):
            logits, h_next = self.assistant.step(tok_tt, h, self._shared_kv, page_tables, pos_u, pos_i)
            if owns_h:
                h.deallocate(True)
            h, owns_h = h_next, True
            lh = self._logits_to_host(logits).reshape(B, -1)  # [B, vocab]
            logits.deallocate(True)
            toks = [int(torch.argmax(lh[b])) for b in range(B)]
            for b in range(B):
                drafts_b[b].append(toks[b])
            tok_tt.deallocate(True)
            tok_tt = self._tokens_tensor(toks)  # [1,B]
        if owns_h:
            h.deallocate(True)
        tok_tt.deallocate(True)
        pos_u.deallocate(True)
        pos_i.deallocate(True)
        pt.deallocate(True)
        return drafts_b

    # ── fused batched trace (greedy) ─────────────────────────────────────────
    def _fused_body_batched(self, tr):
        """Fused batched iteration op-graph: K batched drafter steps chained on
        device (argmax + re-embed in-graph), user-major packed-verify input
        assembled on device, ONE batched packed verify, on-device argmax. Returns
        the persistent output handles (verify_x [1,B*P], vidx [1,1,B*P], vhidden
        [1,1,B*P,backbone])."""
        K = self.draft_len
        B = tr["B"]
        P = K + 1
        page_tables = self._shared_kv_page_tables(tr["d_pt"])
        tok = tr["anchor_tok"]  # [1,B]
        h = tr["h"]  # [1,1,B,backbone]
        draft_cols = []
        for _ in range(K):
            logits, h = self.assistant.step(tok, h, self._shared_kv, page_tables, tr["d_pu"], tr["d_pi"])
            idx = self._argmax_last(logits, rows=B)  # [1,1,B] uint32 RM
            logits.deallocate(True)
            tok = ttnn.reshape(idx, (1, B))
            draft_cols.append(tok)
        # Assemble user-major x [1,B*P] (row u*P+p) from the P columns [1,B]:
        # stack -> [1,1,P,B] -> permute -> [1,1,B,P] -> flatten.
        cols4d = [ttnn.reshape(tr["anchor_tok"], (1, 1, 1, B))] + [ttnn.reshape(c, (1, 1, 1, B)) for c in draft_cols]
        stacked = ttnn.concat(cols4d, dim=2)  # [1,1,P,B]
        um = ttnn.permute(stacked, (0, 1, 3, 2))  # [1,1,B,P]
        stacked.deallocate(True)
        verify_x = ttnn.reshape(um, (1, B * P))
        if not _same_device_buffer(verify_x, um):
            # A same-volume reshape is a view of ``um``: freeing the source would free
            # the verify input (inside a capture, the replay then reuses the storage
            # the host reads the drafts from).
            um.deallocate(True)
        vlogits, vhidden = self.target.ttnn_packed_verify_forward(
            x=verify_x,
            position_idx=tr["v_pos"],
            attn_mask_full=tr["mask_full"],
            attn_mask_sliding=tr["mask_slide"],
            packed_p=P,
            page_table=tr["v_pt"],
            kv_cache=self.tt_kv_cache,
            kv_write_idxs=tr["write_idxs"],
            embed_idx_full=None,
            embed_idx_sliding=None,
            hot_pt=None,
        )
        vidx = self._argmax_last(vlogits, rows=B * P)  # [1,1,B*P] uint32 RM
        vlogits.deallocate(True)
        return verify_x, vidx, vhidden

    def _capture_fused_trace_batched(self, anchor_tokens, seed_hidden_b, anchor_positions, S_k):
        """Capture ONE fused batched iteration over persistent buffers."""
        from loguru import logger as _lg

        K = self.draft_len
        B = len(anchor_tokens)
        P = K + 1
        H = self._packed_H()
        window = self.target.hf_config.sliding_window
        cs = list(anchor_positions)
        d_pu, d_pi = self._pos_tensors(cs)
        v_pos_t = torch.tensor([[cs[b] + p for b in range(B) for p in range(P)]], dtype=torch.int64)
        mf, ms = self._packed_mask_host(cs, B, P, H, window, S_k)
        tr = {
            "B": B,
            "P": P,
            "H": H,
            "window": window,
            "S_k": S_k,
            "anchor_tok": self._tokens_tensor(list(anchor_tokens)),  # [1,B]
            "h": ttnn.clone(seed_hidden_b),  # [1,1,B,backbone]
            "d_pu": d_pu,
            "d_pi": d_pi,
            "d_pt": self._page_table_users(B),
            "v_pos": self._from_b(v_pos_t, ttnn.uint32),
            "mask_full": self._from_b(mf, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "mask_slide": self._from_b(ms, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            # Width-matched to S_k (see _page_table_users): the packed SDPA
            # attends the TABLE width, and the unwritten tail dilutes softmax.
            # The trace re-captures on S_k bucket growth, so the width is
            # stable for the life of this trace.
            "v_pt": self._page_table_users(B, width=S_k // int(self.tt_kv_cache[0][0].padded_shape[2])),
            "write_idxs": [
                self._from_b(torch.tensor([cs[b] + p for b in range(B)], dtype=torch.int32), ttnn.int32)
                for p in range(P)
            ],
        }
        _lg.info(f"[spec-trace] capture batched fused: compile run (B={B}, S_k={S_k})")
        vx, vidx, vh = self._fused_body_batched(tr)
        ttnn.synchronize_device(self.mesh_device)
        vx.deallocate(True)
        vidx.deallocate(True)
        vh.deallocate(True)
        _lg.info("[spec-trace] capture batched fused: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        vx, vidx, vh = self._fused_body_batched(tr)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        _lg.info("[spec-trace] capture batched fused: DONE")
        tr["id"] = tid
        tr["verify_x"] = vx
        tr["vidx"] = vidx
        tr["vhidden"] = vh
        self._fused_trace_batched = tr

    def _hidden_rows_to_device_batched(self, rows_b):
        """Shift-seed: gather packed-verify hidden row ``b*P+rows_b[b]`` per user
        into the persistent batched recurrent buffer tr["h"] (host round-trip)."""
        tr = self._fused_trace_batched
        B, P = tr["B"], tr["P"]
        vh = tr["vhidden"]
        vh_t = ttnn.to_torch(ttnn.get_device_tensors(vh)[0]) if self._tp > 1 else ttnn.to_torch(vh)
        sel = torch.cat([vh_t[:, :, b * P + rows_b[b] : b * P + rows_b[b] + 1, :] for b in range(B)], dim=2)
        host_h = ttnn.from_torch(
            sel.contiguous(), layout=ttnn.TILE_LAYOUT, dtype=tr["h"].dtype, mesh_mapper=self._mapper
        )
        ttnn.copy_host_to_device_tensor(host_h, tr["h"])
        host_h.deallocate(True)

    def _generate_fused_traced_batched(self, anchor_tokens, anchor_positions, max_new_tokens, max_seq_len):
        """Traced batched greedy spec-decode: capture one fused batched iteration
        (drafter chain + packed verify), then replay once per iteration, updating
        only host->device inputs (tokens, positions, per-user masks, KV-write
        indices, shift seed). Prefill is NOT traced (stays eager)."""
        B = len(anchor_tokens)
        K = self.draft_len
        P = K + 1
        S_k = ((max_seq_len + 63) // 64) * 64

        setup_t0 = time.perf_counter()
        # Eager (untraced) batched seed, then capture (mirrors the single-user path:
        # an active verify trace would collide with the fused capture allocations).
        self._use_trace = False
        seed_h = self._seed_batched(list(anchor_tokens), list(anchor_positions))
        self._use_trace = True
        self._capture_fused_trace_batched(anchor_tokens, seed_h, anchor_positions, S_k)
        seed_h.deallocate(True)
        tr = self._fused_trace_batched
        self._last_fused_setup_s = time.perf_counter() - setup_t0

        toks = list(anchor_tokens)
        pos = list(anchor_positions)
        outs = [[] for _ in range(B)]
        accepts = [[] for _ in range(B)]
        done = [False] * B
        pos_cap = max_seq_len - P
        H = tr["H"]
        window = tr["window"]

        first = True
        replay_t0 = time.perf_counter()
        while not all(done):
            if not first:
                cs = list(pos)
                h_tok = self._host_tokens(toks)
                ttnn.copy_host_to_device_tensor(h_tok, tr["anchor_tok"])
                h_tok.deallocate(True)
                d_hpu, d_hpi = self._host_pos(cs)
                ttnn.copy_host_to_device_tensor(d_hpu, tr["d_pu"])
                ttnn.copy_host_to_device_tensor(d_hpi, tr["d_pi"])
                d_hpu.deallocate(True)
                d_hpi.deallocate(True)
                vpos = ttnn.from_torch(
                    torch.tensor([[cs[b] + p for b in range(B) for p in range(P)]], dtype=torch.int64),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    mesh_mapper=self._mapper,
                )
                ttnn.copy_host_to_device_tensor(vpos, tr["v_pos"])
                vpos.deallocate(True)
                mf, ms = self._packed_mask_host(cs, B, P, H, window, tr["S_k"])
                hmf = ttnn.from_torch(mf, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=self._mapper)
                hms = ttnn.from_torch(ms, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=self._mapper)
                ttnn.copy_host_to_device_tensor(hmf, tr["mask_full"])
                ttnn.copy_host_to_device_tensor(hms, tr["mask_slide"])
                hmf.deallocate(True)
                hms.deallocate(True)
                for p in range(P):
                    wi = ttnn.from_torch(
                        torch.tensor([cs[b] + p for b in range(B)], dtype=torch.int32),
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        dtype=ttnn.int32,
                        mesh_mapper=self._mapper,
                    )
                    ttnn.copy_host_to_device_tensor(wi, tr["write_idxs"][p])
                    wi.deallocate(True)
            first = False

            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)

            vx = (
                ttnn.to_torch(ttnn.get_device_tensors(tr["verify_x"])[0])
                if self._tp > 1
                else ttnn.to_torch(tr["verify_x"])
            ).reshape(-1)
            gids = self._ids_to_host(tr["vidx"], B * P)

            rows_b = []
            for b in range(B):
                drafts = [int(vx[b * P + 1 + j]) for j in range(K)]
                g = [gids[b * P + j] for j in range(P)]
                m = next((i for i in range(K) if drafts[i] != g[i]), K)
                committed = drafts[:m] + [g[m]]
                rows_b.append(min(m + 1, K))
                if done[b]:
                    continue
                accepts[b].append(m)
                for tok in committed:
                    outs[b].append(tok)
                    if tok in self.stop_tokens or len(outs[b]) >= max_new_tokens:
                        done[b] = True
                        break
                pos[b] = pos[b] + m + 1
                toks[b] = committed[-1]
                if pos[b] >= pos_cap:
                    done[b] = True

            if not all(done):
                self._hidden_rows_to_device_batched(rows_b)

        self._last_fused_replay_s = time.perf_counter() - replay_t0
        return outs, accepts

    def generate_batched(self, anchor_tokens, anchor_positions, max_new_tokens, max_seq_len, temperature=0.0):
        """Greedy batched speculative decode for B independent users (ragged).

        Each iteration drafts ALL B users in one batch=B drafter chain, runs ONE
        batched packed verify over all B users (the KV-bandwidth win), and accepts
        per user — so users advance by their own matched-prefix+1 and their
        positions diverge. With ``self._use_trace`` the whole iteration is ONE
        replayed metal trace (prefill is never traced); otherwise it runs eager.

        Returns (outs: list[B] of generated token-id lists, accepts: list[B] of
        per-iteration accept counts).
        """
        if temperature and temperature > 0:
            raise NotImplementedError("batched spec-decode supports greedy only (temperature<=0)")
        if self._use_trace:
            return self._generate_fused_traced_batched(anchor_tokens, anchor_positions, max_new_tokens, max_seq_len)
        B = len(anchor_tokens)
        K = self.draft_len
        P = K + 1
        toks = list(anchor_tokens)
        pos = list(anchor_positions)
        outs = [[] for _ in range(B)]
        accepts = [[] for _ in range(B)]
        done = [False] * B
        # furthest position a verify touches is pos[b]+K; keep it in range.
        pos_cap = max_seq_len - P

        # Per-user recurrent drafter seed (clone immediately: _seed_batched's
        # hidden may alias a reused verify-output buffer).
        seed_h = self._seed_batched(toks, pos)  # [1,1,B,backbone]
        backbone = seed_h.shape[-1]
        anchor_h = [ttnn.clone(ttnn.slice(seed_h, [0, 0, b, 0], [1, 1, b + 1, backbone])) for b in range(B)]
        seed_h.deallocate(True)

        _draft_mode = os.environ.get("GEMMA4_SPEC_DRAFT_MODE", "batched")

        while not all(done):
            if _draft_mode == "loop":
                drafts_b = [self._draft(toks[b], anchor_h[b], pos[b], temperature=0.0, user_idx=b)[0] for b in range(B)]
            else:
                anchor_hb = ttnn.concat(anchor_h, dim=2)  # [1,1,B,backbone]
                drafts_b = self._draft_batched(toks, anchor_hb, pos)
                anchor_hb.deallocate(True)

            tokens_b = [[toks[b]] + drafts_b[b] for b in range(B)]
            vlogits, vhidden = self._verify_packed_batched(tokens_b, [pos[b] for b in range(B)])

            for b in range(B):
                row_logits = [vlogits[b * P + j] for j in range(P)]
                m, committed = self._accept_greedy(drafts_b[b], row_logits)
                # next drafter seed (shift): position-aligned hidden row min(m+1,K).
                row = min(m + 1, K)
                new_h = ttnn.clone(ttnn.slice(vhidden, [0, 0, b * P + row, 0], [1, 1, b * P + row + 1, backbone]))
                anchor_h[b].deallocate(True)
                anchor_h[b] = new_h
                if done[b]:
                    continue
                accepts[b].append(m)
                for tok in committed:
                    outs[b].append(tok)
                    if tok in self.stop_tokens or len(outs[b]) >= max_new_tokens:
                        done[b] = True
                        break
                pos[b] = pos[b] + m + 1
                toks[b] = committed[-1]
                if pos[b] >= pos_cap:
                    done[b] = True
            vhidden.deallocate(True)

        for b in range(B):
            anchor_h[b].deallocate(True)
        return outs, accepts

    def generate(
        self, anchor_token, anchor_pos, max_new_tokens, anchor_hidden=None, temperature=0.0, top_p=1.0, top_k=0
    ):
        """Run speculative decode from a prompt anchor (prefill must have filled the KV).

        Args:
            anchor_token: last committed token id (e.g. last prompt token).
            anchor_pos: its absolute position p.
            max_new_tokens: max tokens to generate.
            anchor_hidden: optional pre-seeded target hidden at p; computed via
                ``seed`` when None.

        Returns:
            (generated_token_ids, num_accept_per_iter) — the latter for accept-rate stats.
        """
        greedy = not temperature or temperature <= 0
        if self._use_trace:
            if greedy:
                return self.generate_fused(anchor_token, anchor_pos, max_new_tokens)

            # Sampling mode cannot use the single fused greedy trace because the
            # draft/verify token selection is non-deterministic (it depends on the
            # sampled token), so draft and verify must run as two SEPARATE traces
            # that interleave each iteration. That interleaving is unsafe here:
            #   1. Deadlock: draft and verify each capture their own multi-device
            #      CCL (all-gather/reduce-scatter) trace. Replaying two distinct
            #      CCL traces back-to-back on the same command queue makes one
            #      trace's collective wait on buffers the other trace still owns,
            #      so the mesh hangs (the same failure the fused path avoids).
            #   2. Allocation collision: capturing a second (verify) trace while a
            #      draft trace is live reuses persistent I/O buffers the draft
            #      trace still references, corrupting in-flight state.
            # Greedy sidesteps both by fusing draft+verify into ONE trace; sampling
            # has no fused equivalent yet, so fall back to the untraced host loop.
            from loguru import logger as _lg

            _lg.warning(
                "Disabling speculative trace for sampling mode; separate draft/verify "
                "trace interleaving deadlocks the mesh (see generate() for details)"
            )
            self._use_trace = False
            try:
                return self.generate(
                    anchor_token,
                    anchor_pos,
                    max_new_tokens,
                    anchor_hidden=anchor_hidden,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            finally:
                self._use_trace = True

        traced = self._use_trace
        self._pv_a_prev = -1  # re-seed packed-verify staging for the new anchor/request
        out = []
        accepts = []
        # Own (and free) only what we allocate: a caller-provided anchor_hidden
        # stays the caller's to manage; a seeded one is ours. In traced mode the
        # seeded anchor is the PERSISTENT self._anchor_buf (never deallocated).
        owns_anchor_hidden = (anchor_hidden is None) and not traced
        if anchor_hidden is None:
            anchor_hidden = self.seed(anchor_token, anchor_pos)
        draft_fn = self._draft_traced if self._use_trace else self._draft
        while len(out) < max_new_tokens:
            drafts, draft_logits = draft_fn(
                anchor_token, anchor_hidden, anchor_pos, temperature=temperature, top_p=top_p, top_k=top_k
            )

            verify_tokens = [anchor_token] + drafts
            verify_pos = [anchor_pos + j for j in range(len(verify_tokens))]
            verify_logits, hidden = self._verify(verify_tokens, verify_pos)

            if greedy:
                m, committed = self._accept_greedy(drafts, verify_logits)
            else:
                m, committed = self._accept_sampling(drafts, draft_logits, verify_logits, temperature, top_p, top_k)
            accepts.append(m)

            # New anchor token = committed[-1] = g[m] at position anchor_pos+m+1.
            # The drafter seed must be the target hidden for THIS token at THIS
            # position. The verify (tokens [anchor, d1..dK] at positions P..P+K)
            # only has hidden at index j = position P+j computed with the draft
            # token there as input — never g[m] at P+m+1. Seed quality drives the
            # acceptance rate (the first-step seed() is exact, PCC ~0.98):
            #   reseed: exact batch=1 verify of (g[m], P+m+1) — best quality, one
            #           extra small target forward (can be folded into the trace).
            #   shift : hidden[m+1] = position-aligned but input-approximate
            #           (computed with the rejected draft); no extra forward.
            #   cur   : hidden[m] = position P+m (input+position mismatched).
            new_pos = anchor_pos + m + 1
            new_token = committed[-1]
            # In traced mode `hidden` is the persistent verify output: never
            # deallocate it, and the default reseed path doesn't read it.
            if self._seed_mode == "cur":
                new_anchor_hidden = ttnn.clone(hidden[:, :, m : m + 1, :])
                if not traced:
                    hidden.deallocate(True)
            elif self._seed_mode == "shift" and m + 1 <= len(drafts):
                new_anchor_hidden = ttnn.clone(hidden[:, :, m + 1 : m + 2, :])
                if not traced:
                    hidden.deallocate(True)
            else:  # "reseed" (default) or shift fallback when m == draft_len
                if not traced:
                    hidden.deallocate(True)
                new_anchor_hidden = self.seed(new_token, new_pos)
            if owns_anchor_hidden:
                anchor_hidden.deallocate(True)
            anchor_hidden, owns_anchor_hidden = new_anchor_hidden, not traced
            anchor_pos = new_pos
            anchor_token = new_token

            for tok in committed:
                out.append(tok)
                if tok in self.stop_tokens:
                    if owns_anchor_hidden:
                        anchor_hidden.deallocate(True)
                    return out, accepts
                if len(out) >= max_new_tokens:
                    break

        if owns_anchor_hidden:
            anchor_hidden.deallocate(True)
        return out, accepts
