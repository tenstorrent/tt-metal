# `topk_threshold_select` — Design

**Status:** design, implementation-ready. Nothing in this document has been built; every
cost is either **MEASURED** (traceable to a perf/correctness artifact on branch
`nkapre/sorting`, cited by file) or tagged **ASSUMED**/**OPEN** per the branch discipline
(`SORTING.md` header). No speedup is promised beyond what the model in §5 derives, and §5
states where the model is weakest.

**One-paragraph summary.** A streaming threshold/histogram top-k for Blackhole that
escapes the bitonic critical path. The bitonic tree (`topk_large_indices`) has a derived
serial floor of a few µs (log-depth of 2-cycle `SFPSWAP` lattices, ~460–2100 cycles per
merge+rebuild step, `SORTING.md` §0a-quinquies) and hard constraints (`k % 16 == 0`,
`k <= 2048`, `topk_large_indices_device_operation.cpp:16,25`). Threshold selection is a
different complexity class: it is **bandwidth-shaped** — two streaming passes over the
data plus a tiny reduction — and k appears only as an integer compared against counts, so
**arbitrary k is free** (measured structural fact, `SORTING.md` §0a "Arbitrary k is
free"). The design composes four silicon-validated primitives from this branch: the
packer exponent histogram (zero-cost threshold search), the packer
`MIN_THRESHOLD_RELU` + zero-compression filter+compact (zero SFPU instructions),
`sfpu_count_above` (exact counting at the 2 cyc/vector architectural floor), and the
`SFPGT`+`SFPAND` negfilter (value-preserving signed filter at its proven 2-issue floor).

---

## 0. Grounding: the silicon-validated primitives this design is built from

Read these before touching the implementation. Every mechanism below is measured on
Blackhole silicon with correctness suites and mutation controls; the caveats listed are
themselves measured findings, not speculation.

| primitive | what it gives us | measured cost | caveats that shape this design | artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Packer exponent histogram** | per-exponent counts as a free side effect of any pack | **free** (−0.28% on the pack); `CLREXPHIST` = 1 cycle; full threshold search at N=32k ≈ **128 cycles** vs 24,876 for binary search (**194x**) | samples **1 datum in 8**, fixed pattern `p mod 64 < 8`; **32 buckets, `Exponent & 31` aliased**; `uint8` counters **saturate at 255**; `WhichPackers` ignored on read modes 6/7 (do NOT sum the four reads); mode-9 max-exp also subsampled; `CLREXPHIST` must issue from the **math** thread (pack-thread issue leaks ~39 counts); **sign-blind** (ranks \|x\|) | `tests/sources/pack_exp_histogram_{test,perf}.cpp`, 38/38 + 6/6; `SORTING.md` §0b A3–A6 |
| **Packer zero-compression** | compaction of filtered survivors with **zero SFPU instructions**, composes with `MIN_THRESHOLD_RELU` in one PACR sequence | pack cost +~0.41 cyc/vec flat; dense fp32 fused tile 4096 B → **640 B**; whole relucomp pipeline arm 4.034 cyc/vec L1_TO_L1 vs 6.879 for `_topk_xl_merge_` | max elision stride 16 ⇒ **≥ ceil(zeros/16) placeholder words survive** (~16:1 per pass ⇒ cascade to reach ~k); BH zero-run nibble counts **preceding** zeros (decoder trap — irrelevant here, see §1.4); `Downsample_mask` (THCON_SEC0_REG1 word 3) is a config escape that survives ELF reload — write it explicitly; use `TTI_PACR`, never `TT_PACR` (observed hang) | `tests/sources/pack_zero_compress_{test,perf}.cpp`, 35/35; `SORTING.md` §2.4 "zero-compression", §0b A1/C8/C9 |
| **Packer `MIN_THRESHOLD_RELU`** | threshold filter inside the pack, free | relucomp arm: the whole filter+compact costs **+0.097 cyc/vec (2.4%)** over the bare stream | compares in the **sign-magnitude total order** (same as `SFPGT` — the order top-k wants); **negative threshold is UB and measured-unusable** (mantissa ignored, \|T\| rounds up to the next power of two) ⇒ signed thresholds MUST take the SFPU path | `SORTING.md` §0a-ter "Bonus hardware finding", §0b A2 |
| **`sfpu_count_above`** | exact count of elements > T (sign-magnitude), bit-serial exact threshold search | **2.0 cyc/vec** — proven architectural floor for an SFPU reduction (macro Load+`SFPGT` + software `SFPIADD`); mask-map form 1.003 | count is a reduction and cannot ride a macro (`LReg[16]` write-only for ALUs); silent-undercount hazard `SFPLOADMACRO.md:149` ⇒ correctness suites need an exact all-above case; `ckernel_unpack_template::run` count ≤ 128 | `tests/sources/sfpu_count_above_{test,perf}.cpp`, 13 device cases; `SORTING.md` §2.4 Model B |
| **`topk_negfilter`** | value-preserving signed filter: `Dst[i] = (Dst[i] > T) ? Dst[i] : 0x00000000`, bitwise (denormals/NaN payloads preserved, losers become exact +0 for the compressor) | **2 issues/vec, proven floor** (two Simple ops, one Simple slot per macro); L1_TO_L1 6.415 vs `_topk_xl_merge_` 6.879 | **tie rule inverts at negative thresholds** (index magnitude makes fused ties *more* negative) — §1.4 turns this into an exact ≥-compare by choosing the fused threshold word per sign; needs `SFPENCC(0,EI)` at init to kill stale LaneFlags | `tests/sources/topk_negfilter_common.h`, `topk_negfilter_{test,perf}.cpp`, 9 passed; `SORTING.md` §0a-ter |
| **Stream floor (stock LLK)** | the number every per-pass cost sits on | fp32 `unpack_to_dest` **3.855–3.94 cyc/vec** (disputed band, both measured); stream+compressed-pack **4.132**; each software SFPU issue adds **~1.004** (measured linear law); PACK overlaps unpack (`max`), **SFPU does not** (same-Dest serialization) | the floor is the LLK per-tile handshake, **not hardware** — raw UNPACR streams at 1.257 cyc/vec with split Dest regions (3.32x headroom, unbuilt); numbers from the tt-llk harness are not directly transferable to a metal kernel (`.ttinsn` gathering differs, §0b B7) | `SORTING.md` §0a-bis, §0a-ter, `tests/sources/topk_pipeline_perf.cpp` |
| **Multi-core gather/tree idioms** | pairwise 2-semaphore ship/receive protocol, recv-CB address symmetry, ≤64 slices / 6 levels, root at rectangle (0,0), `valid_length` runtime-only (hash-excluded) | column-parallel `topk_large_indices`: 24–89 µs, single row, k∈[512,2048], W∈[32k,100k] (HANDOFF.md FINAL RESULTS) | reset-before-signal ordering on both sides; `noc.async_write_barrier()` before the data-semaphore bump; empty slices serviced by a prefilled −inf scratch | `ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/{topk_large_indices_program_factory.cpp, kernels/writer_tree.cpp}` |

Also load-bearing: the fused sort key `[bf16 value (hi16) | u16 index (lo16)]` compared as
one fp32 word in sign-magnitude order — the idiom `topk_xl` ships and the relucomp arm
measured (dense fused tile → 640 B). The 16-bit gap between value ulps and the index field
is what makes the tie handling in §1.4 *exact*, not heuristic.

---

## 1. Algorithm and multi-pass structure

### 1.0 Shape of the computation

Per row (rows are independent — see §3 for multi-row):

```
                      row of N bf16 elements, sliced across C cores
                                        |
   ┌────────────────────────────────────┼──────────────────────────────────┐
   |  PASS 1 (per core, streaming)      |                                  |
   |  unpack -> Dst -> pack-to-scratch  |   packer exponent histogram ON   |
   |  (MIN_THRESHOLD_RELU T=+0)         |   -> 32 sampled buckets/core     |
   └────────────────────────────────────┼──────────────────────────────────┘
                                        v
              GLOBAL REDUCE: ~300 B/core of counters -> root core
              root picks conservative threshold T_lo, broadcasts it
                                        |
   ┌────────────────────────────────────┼──────────────────────────────────┐
   |  PASS 2 (per core, streaming)      |                                  |
   |  unpack -> fuse [v|idx] -> exact count(> T') -> filter -> compressed  |
   |  pack   (packer relu path if T>=0, negfilter if T<0)                  |
   |  + cascade re-compression passes (16:1 each) down to ~candidates     |
   └────────────────────────────────────┼──────────────────────────────────┘
                                        v
              GATHER candidates (~k + slack fused words total) to root
              root: exact boundary refinement on candidates -> exactly k
              root: unfuse -> values (bf16) + indices (u32) -> DRAM
```

Exactness never depends on the histogram. The histogram is an **estimator** that makes
the common case cheap; the exact machinery is `sfpu_count_above` (2 cyc/vec, exact,
sign-magnitude) plus the per-core exact survivor counts from pass 2. If the estimate is
bad — sampling skew, bucket aliasing, adversarial data — the result is a **retry or a
fallback (§1.6), never a wrong answer**.

### 1.1 Key encoding (fixed before any pass runs)

- Input: bf16. In Dst (32-bit rows), a bf16 value unpacked to fp32 is its bit pattern
  shifted left 16 — the low 16 bits are zero. The **fused key** is
  `w = (bf16_bits << 16) | (local_idx + 1)` with `local_idx ∈ [0, 32767]` (chunk size
  32768 elements = 32 tiles, §1.4).
- The **+1 index offset** is load-bearing twice: (a) a survivor with value +0.0 at local
  index 0 would otherwise be the word `0x00000000`, which zero-compression silently
  elides — a dropped survivor; (b) it makes the tie arithmetic in §1.4 a clean ≥-compare.
- All comparisons (packer relu, `SFPGT`, `SFPSWAP` in the fallback) use the
  **sign-magnitude total order** `-NaN < -Inf < … < -0 < +0 < … < +Inf < +NaN`. Because
  bf16 ulp << 16 = 65536 > 32768 ≥ any index field, **distinct bf16 values never
  interleave**: the fused order restricted to values is exactly the bf16 sign-magnitude
  order, and ties order by index (descending index on the negative side — harmless, any
  tie set is valid, §2).
- **OPEN (must be settled in Phase A of the work plan):** whether the production
  bf16→Dst unpack path preserves denormals as the exact `bits << 16` pattern or flushes
  them. The negfilter suite proved the *SFPU* side is bitwise; the unpack conversion is
  the open half. If the convert path flushes, use a non-converting format pairing (unpack
  bf16 as raw 16-bit into the high half of 32-bit Dst rows) — the whole pipeline is
  bitwise sign-magnitude and never needs the datum to be a *valid float*, only a 32-bit
  key. A denormal-heavy test (§7) is the tripwire either way.

### 1.2 Pass 1 — per-core exponent histogram (streaming, one read of the slice)

Each core owns a contiguous slice of `ceil(N/C)` elements (chunk-aligned, same
`compute_slice_runtime` split as `topk_large_indices_program_factory.cpp:483`,
including the `valid_length` cut and empty-slice handling).

Per input tile (bf16, 1024 datums):

1. Reader: DRAM → L1 input CB (`TensorAccessor`, double-buffered).
2. Compute (math): `unpack_to_dest`; issue `TTI_CLREXPHIST` from the **math thread**
   before releasing Dst (measured requirement — pack-thread clear leaks ~39 counts,
   `SORTING.md` §0b A6).
3. Pack: `MIN_THRESHOLD_RELU` with threshold **+0.0** (legal: non-negative), histogram
   enabled (`ENABLE_ACC_STATS_Enable_ADDR32 = 45` on BH — the WH value 46 pokes the wrong
   register silently), zero-compression ON, into a small scratch CB that is immediately
   recycled. The pack exists to drive the histogram; its output is discarded. Compression
   is enabled anyway so `PackerTileSize` shrinks with sparsity and the scratch CB stays
   small.
4. Pack RISC-V: read back histogram halves via `SETDMAREG` modes 6/7 (one packer read —
   modes 6/7 ignore `WhichPackers`; summing the four reads 4x-counts) and accumulate into
   32 per-core `uint32` counters in L1. Readback batching: counters saturate at 255 and a
   tile contributes exactly 128 increments, so reading **every tile** is always safe;
   reading every 2 tiles is safe iff no bucket saturated (255 is then "≥255" — acceptable
   for an estimator; never acceptable to *miss* it, so treat 255 as saturated-flag).
   **OPEN:** the per-tile readback cost. The 128-cycles-for-32-tiles figure in
   `SORTING.md` prices `CLREXPHIST` + the SETDMAREG issues; the correctness probe's
   `drain_gprs` does a `tensix_sync` per read, which is NOT free per tile. Budgeted at
   10–40 cycles/tile (0.3–1.25 cyc/vec) pending measurement; the mitigation if it
   measures high is per-2-tile batching plus overlapping the drain with the next tile's
   unpack.

What the counters mean, and why relu(+0) is part of the trick:

- Relu T=+0 zeroes every **negative** datum before the histogram sees it, so negatives
  (and true ±0 and denormals) land in **bucket 0**, and buckets 1..31 are a clean
  alias-classed histogram of the **positive** magnitudes: bucket `b` counts exponents
  `{b, b+32, b+64, … , b+224}` — the `Exponent & 31` aliasing is real and measured.
- The counts are **1-in-8 sampled** on the fixed positional pattern `p mod 64 < 8`. The
  root multiplies by 8 and treats the result as an estimate with both statistical noise
  and (worst case) adversarial positional bias. §1.6 bounds what happens when the
  estimate is arbitrarily wrong.
- **Sign decision:** if `8 × Σ(buckets 1..31) ≥ k·α` the threshold is (estimated)
  positive and pass 2 takes the packer path. Otherwise the threshold is likely negative:
  run **pass 1b** on the slice — identical, but with an SFPU sign-flip map
  (`SFPXOR` of `0x80000000`, one macro-scheduled Simple riding the load ⇒ ~1 extra
  issue/vec) before the relu(+0) pack, yielding the histogram of **negative** magnitudes;
  the root then searches it from the *bottom* (want the (k − n_pos)-th smallest
  magnitude). Pass 1b costs one extra read of the slice and only runs when k reaches into
  the negative half — rare for the top-k-of-logits workloads this op targets, guaranteed
  for e.g. all-negative inputs (§7 covers both).

Alternative rejected: computing the histogram in software with the measured
`HistMacro`+`HistSum` corner (3.0 cyc/vec for 3 bits/pass) — correct and exact, but
3 cyc/vec against ~0 for the packer. It survives as the estimator inside the fallback
(§1.6), where exactness matters and the packer's sampling does not qualify.

### 1.3 Global threshold reduction (tiny, one NoC round)

Each core ships `{hist[32] (u32), hist_neg[32] (u32, pass-1b only, else omitted),
slice_len (u32), flags}` ≈ **140–270 B** to the root — direct unicast gather, not a tree:
64 cores × 270 B ≈ 17 KB total NoC traffic, far below the point where a tree pays.
Protocol: the pairwise ready/data semaphore idiom from `writer_tree.cpp` (reserve recv
CB → zero own data-sem → bump partner's ready-sem → wait data-sem), with the recv CB
sized `C × 512 B` on the root (≤ 32 KB — trivial). Reset-before-signal ordering and
`async_write_barrier()`-before-`data_sem.up()` are copied verbatim; they are the
correctness spine of the shipped op.

Root computes:

1. Global 32-bucket sums; cumulative sum from the top bucket downward (with the alias
   caveat: "top" within an alias class is resolved optimistically by the mode-9 max-exp
   hint where available, pessimistically by refinement — aliasing can only mislead the
   *estimate*).
2. `T_lo` = the **lower edge** (smallest magnitude) of the first bucket where the ×8
   cumulative estimate reaches `k·α + β` (slack policy: α = 1.5, β = 256; tunable
   compile-time constants). Conservative direction is *down*: too-low `T_lo` costs
   candidate volume, too-high costs a retry.
3. Broadcast `{T'_pos or T'_neg (fused 32-bit threshold word, §1.4), path flag
   (packer/negfilter), candidate cap}` to all cores: one semaphore-guarded scratch write,
   multicast, ~64 B.

The whole reduce is O(10 µs)-class latency, O(20 KB) traffic — it can never be the
bottleneck, which is the point of a 256-counter-class reduction (here 32+32 counters).

### 1.4 Pass 2 — filter + compact (streaming, second read of the slice)

Per chunk of 32 tiles (32768 elements — the u16-index span):

1. **Fuse:** unpack bf16 tile to Dst (value lands as `bits<<16`); SFPU map ORs in the
   per-lane local index `+1` (index vectors generated by an `SFPIADD` stride walk +
   `SFPOR`; both Simple ⇒ 2 macros ⇒ **~2.0 cyc/vec** measured-law cost on the stock
   path). This is the same fuse idiom the branch's `topk_large_indices` compute already
   performs ("lsb + index split", `topk_large_indices_program_factory.cpp:382`).
2. **Exact count:** `sfpu_count_above(T')` on the fused tile — **+2.0 cyc/vec** — into a
   per-chunk running count `n_c`. This is the count that makes the algorithm exact.
   (Optimization, later: move the count onto the 16x-smaller cascade stage-1 stream,
   saving ~1.9 cyc/vec; keep it inline in v1 for simplicity.)
3. **Filter + compact**, by sign of T:
   - **T ≥ 0 (packer path):** pack with `MIN_THRESHOLD_RELU` threshold word
     `T'_pos = (T_bits << 16) | 0x0000` and zero-compression ON. Zero SFPU cost —
     measured +0.097 cyc/vec over the bare stream.
   - **T < 0 (SFPU negfilter path):** the packer is **unusable** (measured UB: negative
     threshold ignores the mantissa and rounds \|T\| up to a power of two —
     `SORTING.md` §0a-ter). Run `topk_negfilter` (2 issues/vec ⇒ **+2.0 cyc/vec**) with
     threshold word `T'_neg = (T_bits << 16) | 0xFFFF`, then a plain compressed pack.
4. Survivors land in a **per-chunk compressed segment** in L1 (known base per chunk), so
   the chunk id — and with it bits 15+ of the global index — is positional. Global index
   `= slice_start + chunk_id·32768 + (idx_field − 1)`.

**Tie exactness (why those two threshold words).** The filter keeps
`w > T'` in sign-magnitude. For a datum with value exactly T and index field
`i ∈ [1, 32768]`:

- Positive side: `mag(w) = (|T|<<16) + i > (|T|<<16) = mag(T'_pos)` ⇒ **ties kept**. A
  value one bf16-ulp below T has `mag ≤ (|T|<<16) − 65536 + 32768 < mag(T'_pos)` ⇒
  dropped. Exact.
- Negative side: larger magnitude = *smaller*. `mag(T'_neg) = (|T|<<16) + 0xFFFF ≥
  (|T|<<16) + i = mag(w)` ⇒ `w` is less negative than `T'_neg` ⇒ **ties kept**. One ulp
  more negative: `mag ≥ (|T|<<16) + 65536 > mag(T'_neg)` ⇒ dropped. Exact.

This converts the measured **tie-rule inversion** (negative-threshold ties zeroed —
asserted in `test_topk_negfilter.py`) from a hazard into a controlled choice: both sides
implement exactly `keep iff value ≥ T`, using only the strict-`>` compare the hardware
provides. The `+1` index offset is what makes the positive side ≥ rather than >.

**Cascade.** Compression elides at most a 16-stride, so a chunk's compressed segment
still carries `ceil(zeros/16)` placeholder words (≈ 2048 words for an empty 32768-elt
chunk). Re-stream each segment through unpack → (no SFPU) → relu+compress pack with the
same `T'` — each pass shrinks ≥16:1 on the placeholders (survivors are untouched; they
are still > T'). Measured projection: 1024 → 88 → ~36 → ~33 ≈ K (`SORTING.md`
zero-compression section). Two cascade stages take a chunk's segment from ~2 K words to
~`n_c + 8`-ish. Geometric cost: `(1/16 + 1/256) ×` a bare relucomp pass ≈ **6–7% of one
pass 2**. Placeholders that survive the last stage are literal `0x00000000` words; the
gather step skips them by inspection — **no nibble/row-start decoding is ever needed**,
because the fused keys are self-describing. (This is why the BH
counts-preceding-zeroes decoder trap in §0 does not bite this design: we never decode.)

**Per-core outcome of pass 2:** exact survivor count `n_core = Σ n_c`, and a compacted
candidate list of `n_core` fused words (plus stragglers of placeholder zeros).

### 1.5 Bracket check, retry, gather, and exact boundary refinement

Cores ship `n_core` (4 B) to the root (same semaphore protocol). Root sums:
`n_total = count(value ≥ T_lo)` — **exact**, by construction.

- **`n_total < k`** — threshold too high (sampling lied high, or aliasing). Root lowers
  `T_lo` by one bucket (or to the negative path if already at +0's bucket floor),
  re-broadcasts, cores rerun pass 2. Each retry is one full pass; the retry count is
  bounded by the 32 buckets but in practice by the fallback trigger below.
- **`n_total > cap`** (global candidate capacity, `cap = max(4k, 4096)` words,
  compile-time-bounded by root L1) — threshold too low or heavy tie mass. Root raises
  `T_lo` one bucket and reruns **iff** the histogram says the next bucket up still
  brackets k; otherwise the mass is inside one bucket → **fallback** (§1.6).
- **`k ≤ n_total ≤ cap`** — proceed: cores ship their compacted candidates
  (`n_core × 4 B`, `Σ = n_total ≤ cap`) to per-core reserved offsets in the root's
  candidate CB (offsets computed by the root from the `n_core` gather and sent back with
  the "ship" broadcast — one extra tiny round; avoids atomics entirely).

Expected retries for non-adversarial data: **0** (the ×8 estimate at bucket granularity
with α=1.5, β=256 slack brackets k with large margin — a bucket is a full power-of-two of
magnitude).

**Root refinement to exactly k**, on `n_total ≤ cap` candidate words in L1:

1. Bit-serial descent with `sfpu_count_above` on the candidate buffer: find the exact
   fused boundary word. The key is 16 value bits (+16 index bits that never need
   descending — any tie set is valid): ≤ 16 count passes × `n_total/32` vectors ×
   2 cyc/vec. At `n_total = 8192`: ≤ 16 × 256 × 2 = **8192 cycles ≈ 6 µs-class — over-
   estimate**; in practice the exponent is already pinned by the bucket, leaving ≤ 8
   mantissa/sign passes. Alternative for small `n_total` (≤ 2048): one `topk_xl`-style
   bitonic sort of the candidates — the branch's own optimized kernel — then take k;
   choose per `n_total` at runtime (both paths ship, the bitonic one reuses
   `ckernel_sfpu_topk_xl.h` as `topk_large_indices`' compute does).
2. `m = k − count(> T_exact_fused)` ties at the boundary value: keep the first `m`
   encountered (any valid tie set — §2). The root's final filter pass over candidates
   emits exactly k fused words.
3. Unfuse: value = high 16 bits (bf16), index = chunk-positional reconstruction
   (§1.4). Output: UINT32 indices row (+ optional BFLOAT16 values row), streamed to DRAM
   exactly like `writer_tree.cpp`'s root path (contiguous or face-pair reorder does not
   apply here — output is ROW_MAJOR already; plain `TensorAccessor` page writes).
4. Sortedness: torch.topk returns values sorted descending. The refinement descent does
   not sort. For `k ≤ 2048`, run the candidate set (or just the k winners) through the
   bitonic sort before emission (cheap: one core, k words). For larger k, emit
   **unsorted** and document it, or pay a root-local merge — decided per §3's API
   (`sorted=` flag mirroring torch; default true ⇒ bitonic tail for k ≤ 2048, multi-pass
   merge above; the multi-pass merge is Phase D work, not v1 — v1 ships
   `sorted=false`-only above k=2048 and documents it loudly in validate()).

### 1.6 Fallback — REQUIRED, and specified

**Trigger:** the bracket logic above cannot make progress: the histogram places ≥ cap
candidate mass inside a **single bucket** (equivalently: `count(≥ bucket_hi) < k` and
`count(≥ bucket_lo) > cap`). The forcing worst case is **all-equal input**: every datum
in one bucket, `n_total = N`, no threshold at bucket granularity separates anything.
Two-value and exponent-cliff distributions (§7) hit the same trigger with k on the cliff.

**Fallback = exact bit-serial descent on the full stream, then positional tie take.**

1. The bucket pins the exponent (up to aliasing — the descent below resolves aliasing
   for free, since it descends the full 9 high bits when needed). Descend the remaining
   key bits with **full-slice `sfpu_count_above` passes**: each pass streams the slice
   (unpack + fuse + count = ~4.1 + 2 + 2 ≈ 8 cyc/vec) and produces one exact global
   count via the tiny reduce. ≤ 8 passes for sign+exponent disambiguation (usually 0 —
   the bucket gave it) + 7 mantissa passes ⇒ **≤ 8–15 streaming passes, hard bound**,
   after which `T_exact` (a bf16 value, not a fused word) satisfies
   `count(> T_exact) < k ≤ count(≥ T_exact)`.
2. Final pass: filter+compact survivors of `> T_exact` (they number `< k ≤ cap` by
   construction — **no capacity hazard exists in the fallback**) and, per chunk, the
   exact count of `== T_exact` ties (one extra `sfpu_count_above` fused into the same
   pass at +2 cyc/vec). Root takes all strict survivors plus the first
   `k − count(>T_exact)` ties **by position**: it walks chunks in index order using the
   per-chunk tie counts, and only the chunks straddling the take-boundary re-emit their
   tie *indices* (a trivial per-chunk pass: positions where `value == T_exact`, emitted
   as fused words with value field = T_exact). All-equal input resolves to "first k
   indices, all values equal" with **zero candidate storage** beyond k.
3. Cost bound: ≤ 15 passes × ~8 cyc/vec = ~3.8 cyc/element — about **2x the bitonic
   leaf cost** (§5) in the absolute worst case, i.e. the fallback degrades gracefully
   instead of failing. It is also exact-by-construction: it never consults the histogram
   again.

Alternative fallback considered and rejected as primary: bitonic sort of the heavy
bucket's contents — unbounded candidate volume (all-equal ⇒ N words) makes it a
non-starter as the *general* fallback; it survives only as the root-side small-`n_total`
refinement in §1.5.

**Decision rule summary (host-visible, deterministic):**

```
pass1 → reduce → T_lo
loop ≤ R_max (=4):
    pass2(T_lo) → n_total
    k ≤ n_total ≤ cap  → gather, refine, DONE
    n_total < k        → lower T_lo one bucket (or enter negative path); continue
    n_total > cap, next bucket brackets k → raise T_lo; continue
    else               → FALLBACK (≤15 exact passes), DONE
exceeded R_max         → FALLBACK
```

Every arrow terminates; the fallback is total. R_max exists so adversarial sampling skew
(§6-R2) cannot induce more than 4 wasted passes before the bounded path takes over.

---

## 2. Exactness contract (vs `torch.topk`)

Identical in kind to the contract the branch's bitonic ops ship (they compare with
`SFPSWAP`, which uses the same sign-magnitude order):

- **Values: exact.** The returned k values are bit-for-bit the k largest under the
  sign-magnitude total order — as a multiset, identical to `torch.topk(...).values` for
  any input free of NaN and of mixed-sign zeros at the boundary. No PCC anywhere in the
  test plan; comparisons are exact-integer / bit-exact (branch discipline).
- **Indices: any valid tie set.** When more than the needed number of elements equal the
  boundary value, the op returns *some* k-subset containing all strictly-greater elements
  — same contract as `ttnn.topk` / `topk_large_indices` (`stable=False` semantics). The
  fallback path happens to return lowest-index ties; the fast path returns
  index-order-within-chunk ties; neither is promised.
- **Documented divergences from torch, inherited from the hardware order:**
  - **NaN:** torch.topk sorts every NaN above +Inf regardless of sign bit. Sign-magnitude
    puts `+NaN` above `+Inf` (agrees) but `-NaN` **below `-Inf`** (diverges). Same
    divergence as every `SFPSWAP`-based op in the tree. NaN payload bits are preserved
    (bitwise pipeline).
  - **±0:** numerically equal, but the order separates them (`-0 < +0`); a boundary
    falling between them prefers `+0`. Values returned still compare torch-equal.
- **`valid_length`:** elements at positions ≥ valid_length are never candidates and never
  counted. Whole out-of-range chunks are sliced off at runtime (the
  `compute_slice_runtime` idiom); the partial tail chunk is masked in the fuse map by
  writing the sentinel word `0x00000000` (elided by compression, below every survivor)
  over the tail lanes — note an index field of 0 alone would NOT suffice, since a tail
  lane's value bits could still exceed `T'`; the whole word must be zeroed. Cost: one
  extra macro-scheduled `SFPAND` against a precomputed lane mask on the single straddling
  vector per row — negligible, exact.

---

## 3. Shapes, dtypes, API

```
ttnn.experimental.topk_threshold_select(
    input:  Tensor[bf16, ROW_MAJOR, interleaved DRAM/L1],   # rank >= 1
    k:      uint32,                # 1 <= k <= min(N, 2^20); NO %16, NO 2048 cap
    valid_length: Optional[uint32],# runtime-only, hash-excluded (idiom from
                                   # topk_large_indices_device_operation_types.hpp)
    return_values: bool = True,
    sorted: bool = True,           # v1: k<=2048 only when sorted=True (see §1.5.4)
    num_slices: Optional[uint32],  # column-split override, same contract as
                                   # topk_large_indices (loud error off-path)
) -> [indices: Tensor[u32, ROW_MAJOR], values?: Tensor[bf16, ROW_MAJOR]]
```

- **k: arbitrary.** The headline. k enters the device program only as runtime integers
  compared against counts (`SORTING.md`: k=5, 17, 100, 1000 are bit-identical kernels —
  a structural claim this op's test plan finally exercises, §7). Program hash does NOT
  include k (runtime arg), unlike the bitonic op where k shapes the lattice.
  Upper bound 2^20 in v1 from root-gather capacity + spill mode (below).
- **N up to 2^30** per row (same `max_row_elements` bound as
  `topk_large_indices_device_operation.cpp:17`; the u16-index scheme is chunk-local so N
  is bounded by u32 global-index math, not by the key encoding).
- **bf16 first.** fp32 later: same structure, 32-bit keys don't fit a fused u16 index ⇒
  parallel index tile (the `ttnn.topk` idiom) or 64-bit two-word keys — Phase E, out of
  v1 scope. u16/u32/int inputs: sign-magnitude ≠ two's-complement ordering — needs a
  premap; out of v1 scope.
- **Multi-row:** rows independent. `num_rows > 1` ⇒ row-parallel factory: rows split
  over cores (`split_work_to_cores`, as the existing row-parallel topk_large_indices
  path), each core running the whole per-row pipeline single-core (no reduce, no gather;
  its own slice is the whole row). Column-parallel (this document's main path) engages
  for `num_rows == 1` — same selection shape as
  `compute_model_column_split_config(...)` (`topk_large_indices_program_factory.cpp:357`).
  Middle ground (few rows × huge N: row groups × column split) is a factory-level
  extension, not v1.
- **Large-k spill mode** (k+slack beyond root-L1): cores write their compacted survivors
  directly to a DRAM scratch region at exact offsets (prefix sum of `n_core` computed at
  the root — already in hand from §1.5), refinement runs as a second tiny op over the
  scratch. v1 gates k ≤ 2^20 through this; the fast in-L1 path covers k ≤ ~16K.

Outputs mirror `topk_large_indices`: indices UINT32 ROW_MAJOR always; values BFLOAT16
ROW_MAJOR when requested; sentinel behavior for `valid_length < k` lanes copied from that
op (index sentinel `0xFFFFFFFF`, value exact bf16 −inf — `writer_tree.cpp:162-165`).

---

## 4. L1/CB budget per core, and NoC traffic

Per-core budget, column-parallel path, worst planning config (chunk = 32 tiles,
cap_core = 8192 candidate words). BH user L1 = 1.5 MB (WH 1.43 MB — this op is BH-only
v1, see §6-R9).

| CB / region | size | notes |
| :--- | ---: | :--- |
| `cb_in` input (bf16 tiles) | 2 × 8 × 2 KB = **32 KB** | double-buffered reader→compute |
| `cb_fused` scratch (fp32 tiles) | 2 × 8 × 4 KB = **64 KB** | fuse-map output staging for pass 2 |
| `cb_pass1_scratch` (compressed pack sink) | **8 KB** | recycled per tile; exists to drive the histogram |
| per-chunk compressed segments (pass 2 out) | 2 chunks in flight × 16 KB = **32 KB** | 32768 elts → ≤ (n_c + 2048 placeholders + headers) × 4 B ≤ 16 KB conservatively |
| cascade ping-pong | 2 × **8 KB** | stage sizes shrink 16:1 |
| `cb_candidates` (compacted survivors) | **32 KB** | cap_core = 8192 fused words |
| histogram counters + reduce staging | **1 KB** | 2×32 u32 + counts + flags |
| `cb_recv` (root only): count gather + candidate gather | 64 × 8 B + cap_global × 4 B = **64–256 KB (root)** | root candidate CB is THE sizing knob; cap_global = 4k bounded ≤ 64K words |
| broadcast scratch + semaphores | **<1 KB** | 2 semaphores (ready/data) + 1 (bcast), `CreateSemaphore` idiom |
| **Total, non-root** | **~180 KB** | 12% of L1 — comfortable |
| **Total, root** | **~250–440 KB** | still <30%; k ≤ 16K in-L1, beyond ⇒ spill mode |

**L1-resident regime (important):** when the slice fits L1 (`N/C × 2 B ≤ ~768 KB`, i.e.
N ≤ ~24M at C=64), the reader pulls each chunk from DRAM **once**; pass 1, pass 2, and
any fallback passes re-stream from L1. Beyond that, pass 2 (and each fallback pass)
re-reads DRAM — the DRAM term in §5 doubles (or ×P for fallback). The factory picks the
regime from the shape; both use the same kernels (the reader either retains or re-fetches).

**NoC traffic per row (column-parallel):**

| flow | bytes | pattern |
| :--- | ---: | :--- |
| histogram reduce | ≤ 270 B/core ⇒ ≤ 17 KB | unicast gather to root, 1 round (2 rounds if pass 1b) |
| threshold broadcast | ~64 B | root multicast, semaphore-guarded |
| count gather + offset return | 8 B/core each way | 1 round |
| candidate gather | Σ n_core × 4 B ≤ cap_global × 4 B ≤ 256 KB | unicast to precomputed offsets, no atomics |
| retry/fallback control | 8 B/core/round × ≤ (R_max + 15) | worst case only |

All flows use the ready/data two-semaphore protocol from `writer_tree.cpp` with its
reset-before-signal and barrier-before-bump ordering. Total is ≤ ~300 KB per row —
against ≥ 2 MB of input even at N=1M, the NoC never binds.

---

## 5. Performance model

**Per-pass per-vector costs (stock LLK path, measured laws from §0; bf16 input).**
The fp32 stream numbers are used as the planning numbers for bf16-to-32-bit-Dst streams
because the measured floor is the per-tile handshake (91 cyc/tile), not bytes —
**ASSUMED, re-measure in Phase A** (§8).

| pass | composition | cyc/vec | cyc/element |
| :--- | :--- | ---: | ---: |
| pass 1 | stream+pack 4.13 + hist readback 0.3–1.25 | **~4.5–5.4** | 0.15 |
| pass 1b (negative side, when taken) | + sign-flip map ~1.0 | ~5.5–6.4 | 0.18 |
| pass 2, T ≥ 0 | 4.13 + fuse ~2.0 + exact count 2.0 + relu-filter 0.10 | **~8.2** | 0.26 |
| pass 2, T < 0 | + negfilter 2.0 | **~10.2** | 0.32 |
| cascade | ~6 cyc/vec on 1/16 + 1/256 of the data | — | +0.013 |
| fallback pass (each) | 4.13 + fuse 2.0 + count 2.0 | ~8.2 | 0.26 |

**Totals (per element, per core, compute):**

- Fast path, T ≥ 0: `0.15 + 0.26 + 0.013` ≈ **0.42 cyc/elt** (signed: 0.51).
- Absolute worst case (fallback, 15 passes): ≈ **4.0 cyc/elt** — bounded, ~2x bitonic.

**Fixed costs (per row, column-parallel):** 2 (worst 4 + 15) global rendezvous at
O(µs)-class each (semaphore rounds + kernel-phase boundaries), plus root refinement
(≤ 8192 cycles ≈ 6 µs at 1.35 GHz for cap = 8192; §1.5), plus host dispatch (~5–10 µs,
same as any op). Call the fixed total **F ≈ 15–25 µs** fast-path — the same order as
`topk_large_indices`' measured 24–89 µs envelope, which is itself dispatch/tree-latency
shaped at these widths. **ASSUMED; the Phase 0 benchmark makes this real.**

**Bandwidth terms.** Compute at 0.42 cyc/elt = 76 elt/cyc across 32 lanes… per core:
0.42 cyc/elt ⇒ 2.4 B/cyc of bf16 per core on the compute clock; 64 cores ⇒ ~150 B/cyc
aggregate ⇒ at 1.35 GHz ≈ **200 GB/s of input consumption** — of the same order as
DRAM bandwidth. So at full grid the op is **genuinely bandwidth-shaped**: the DRAM read
(1 read L1-resident regime, 2 reads beyond) and the compute stream are within ~2x of each
other, and the 3.32x raw-unpack headroom (§0) would tip it fully DRAM-bound. That is the
structural contrast with bitonic sorting networks, which are 55x past the compute knee
(`SORTING.md` §2.3) and cannot be helped by bandwidth.

**Head-to-head model vs the branch's bitonic tree (`topk_large_indices`):**

Bitonic leaf cost ≈ 2 merge-units per chunk (its own cost model,
`topk_large_indices_program_factory.cpp:381-385`) ≈ 2×459/512 ≈ **1.8 cyc/elt** at
K=512, 2.1 at K=2048 (measured step costs, §0a-quinquies), plus its log-tree.

| (k, N, C) | threshold-select model | bitonic (measured op / model) | verdict |
| :--- | ---: | ---: | :--- |
| k=512, N=32k, C=32 | F + 1k elt/core × 0.42 ≈ **F + 0.3 µs** | **24–89 µs measured band** | wash — both fixed-cost-bound; no reason to switch |
| k=2048, N=1M, C=64 | F + 16k × 0.42 cyc ≈ F + 5 µs ≈ **~25 µs** | F' + 16k × 2.1 cyc ≈ F' + 25 µs ≈ **~50 µs** | **~2x win, growing with N** |
| k=2048, N=2^27, C=64 | 2M × 0.42 ≈ 880k cyc ≈ **~0.7 ms** (+DRAM 2-read ≈ 0.5–1 ms ⇒ ~1–1.7 ms) | 2M × 2.1 ≈ **~3.1 ms** | **~2–4x win; DRAM-read count is the swing term** |
| k=100 (or 17, or 5000), any N | **works** | **unsupported** (k%16, k≤2048) | the headline: capability, not speed |
| k=512, N=4096, C≤8 | F dominates (2 rendezvous) | single bitonic pass, tiny | **loses** — below N≈2048–8192/core the sorting networks win outright (`SORTING.md` §1298 "Where it loses"); the factory must route small N to the bitonic op |
| all-equal / adversarial | ≤ 4.0 cyc/elt bounded | 1.8–2.1 cyc/elt oblivious | **loses ≤ 2x** — bitonic is data-oblivious; that is its enduring advantage and the reason it stays the small-N and worst-case backstop |

**Honest caveats on this model, per branch discipline:** (1) every stream number is from
the tt-llk harness, which enables `.ttinsn` gathering that production metal kernels
disable (§0b B7) — metal-kernel per-pass numbers will differ and must be re-measured;
(2) the SFPU adders (+2, +2) assume the measured same-Dest serialization law — a
split-Dest kernel would hide them under the unpacker (measured max()-composition,
§0a-bis) and roughly halve pass 2, but that kernel does not exist yet; (3) F is not
measured. The Phase 0 gate in §8 exists because of exactly these three.

---

## 6. Risk register

| # | risk | severity | mitigation / disposition |
| :--- | :--- | :--- | :--- |
| R1 | **Heavy-bucket worst cases.** bf16 has 256 exponents but real data occupies ~40; the hardware histogram folds them to 32 aliased buckets. All-equal / two-value / exponent-cliff inputs put ≥cap mass in one bucket — no bucket-granular threshold exists. | correctness-fatal if unhandled | **Fallback path is REQUIRED and specified (§1.6):** ≤15 exact `sfpu_count_above` passes + positional tie take; bounded ~2x bitonic; zero reliance on the histogram. Trigger is exact (per-core counts), not estimated. |
| R2 | **1-in-8 sampling is positionally fixed (`p mod 64 < 8`)** — adversarial data correlated with position mod 64 can make the estimate arbitrarily wrong (all top-k placed in never-sampled slots). | perf only | Exactness never consumes the estimate. Wrong estimates cost retries, capped at R_max=4 before the bounded fallback. §7 includes a test that hides all winners in unsampled positions. |
| R3 | **Signed-threshold packer UB** (measured: negative T rounds \|T\| to the next power of two — unusable). | correctness-fatal if the packer path ever sees T<0 | Hard routing: `T < 0 ⇒ negfilter` (SFPU, value-preserving, its own 9-test suite). The packer relu threshold word is asserted non-negative in the kernel (`static_assert`-style runtime check on the broadcast word's sign bit). |
| R4 | **Denormals / ±0 / ±Inf / NaN.** Bitwise sign-magnitude pipeline handles all of them *if* data reaches Dst bit-exactly. -NaN ordering diverges from torch (documented, §2). +0-at-index-0 elision fixed by the +1 index offset (§1.1). | correctness | OPEN half: does bf16→Dst unpack flush denormals (§1.1)? Settle by test in Phase A; fallback mechanism (raw 16-bit unpack pairing) identified. §7 has denormal-heavy and specials suites. |
| R5 | **Counter saturation** (u8 @ 255, 128 samples/tile). | estimate quality | Per-tile (or per-2-tile with saturation flagging) readback; 255 treated as "≥255". Never a correctness input. |
| R6 | **Histogram readback cost unmeasured** (`tensix_sync` per drain in the probe kernel). | perf (pass 1 could grow 25%) | Phase A micro-measurement; batching + overlap mitigations specified (§1.2). Worst case: pass 1 costs ~5.4 not 4.5 cyc/vec — model absorbs it. |
| R7 | **Config escapes:** `Downsample_mask` (THCON word 3) survives ELF reload and silently decimates packs; `ENABLE_ACC_STATS_Enable` is per-thread, OR'd, and survives reload; stale relu config. | correctness, *observed live on this branch* | Every kernel writes word 3 and the ACC_STATS bit explicitly both ways (the probe kernels already model this — copy them). Add a CI canary test that runs after an unrelated op. |
| R8 | **`TT_PACR` hang** (runtime-issued PACR hung where `TTI_PACR` didn't; not root-caused). | hang | All PACR issue sites use `TTI_PACR`/recorded forms only. |
| R9 | **Arch scope.** `SFPGT`/`SFPLE` are BH-new; the exp histogram divergences are measured on BH only; `ENABLE_ACC_STATS` index differs (45 BH / 46 WH — wrong index pokes another register **silently**). | portability | v1 is **Blackhole-only**, gated in validate(). WH port = new measurement campaign, not a recompile. |
| R10 | **Candidate/root capacity vs huge k.** | functional gap | Spill-to-DRAM mode (§3) for k ≤ 2^20; validate() rejects beyond. |
| R11 | **Retry coordination across cores** (all cores must agree on rerun vs proceed; a straggler using a stale threshold corrupts counts). | hang / wrong answer | Single-writer control: only the root decides; decisions travel on the same ready/data semaphore pairs (monotone round counter in the broadcast word; cores assert round match). No core ever infers control state locally. |
| R12 | **LLK-harness → metal-kernel transfer** (§0b B7, gathering CSR) and the stock-LLK handshake floor. | perf model risk | Phase 0 op-level benchmark before any optimization claims; the 3.32x raw-unpack headroom is explicitly out of v1 scope (no kernel exists that removes the handshake — HANDOFF.md "a diagnosis, not a win"). |
| R13 | **Two DRAM reads beyond the L1-resident regime** (bitonic reads once). | perf at huge N | Model shows the win survives at ~2x; L1-resident regime boundary documented; single-read fused-pass variants (histogram from a *sampled subset of tiles*) noted as Phase E research, not promised. |

---

## 7. Test plan

House rules (branch discipline, non-negotiable): **no PCC — exact-integer or bit-exact
only**; every macro/config mechanism carries a **mutation control** (timing cannot
distinguish "free" from "silently not executed" — `SORTING.md` §0b D4); perf runs carry
the `SFPLOAD`/`SFPSWAP` control pair and are discarded unless the swap lands at 2.00x;
`PROFILER_SYNC()` closes every timed zone; producer+consumer under one
`flock /tmp/tt-device.lock`; never `scripts/run_safe_pytest.sh` for tt-llk tests.

**Level 1 — LLK kernel suites** (extend the four existing suites in
`tt_metal/tt-llk/tests/`):

- Fuse-map correctness: fused word bit-exact vs host for all lanes, index offset +1,
  tail masking; mutation control: drop the `SFPOR` → indices all read as +1-less ⇒ exact
  mismatch.
- Threshold-word tie semantics: for both `T'_pos` and `T'_neg`, three-point probes at
  value = T−ulp / T / T+ulp on both packer and negfilter paths — asserting **kept /
  kept / kept-or-dropped exactly per §1.4's table**, including the negative-side
  inversion case that `test_topk_negfilter.py` already asserts.
- Histogram accumulation: per-tile readback under multi-tile streams; saturation flag at
  a 3-tile single-bucket stream; math-thread `CLREXPHIST` ordering (the measured 39-count
  leak as the mutation control — issue from pack thread, assert the leak appears).
- Cascade: 32768-elt chunk with s ∈ {0, 1, 32, 33, 2048, 32768} survivors — assert exact
  fused-word multiset after 2 stages and that stage sizes follow the ≥16:1 law.

**Level 2 — op-level correctness** (`tests/ttnn/unit_tests/operations/experimental/`):
golden = `torch.topk` with the §2 contract applied (values as multiset; indices checked
by membership + validity, not order; NaN cases get bespoke goldens).

Adversarial distribution battery — each × {T≥0 path, T<0 path where reachable} ×
{single-core row-parallel, column-parallel C ∈ {2, 8, 64}} × valid_length ∈
{None, mid-chunk, < k}:

1. **all-equal** (forces fallback; asserts first-k-indices tie take and zero candidate
   storage growth),
2. **two-value** with k exactly on, one-below, one-above the cliff,
3. **exponent-cliff**: N−k values at exp e, k at exp e+1; then the aliased twin (e and
   e+32 — exercises alias-class resolution),
4. **denormal-heavy** (≥50% bf16 denormals, boundary inside the denormal range — the R4
   tripwire),
5. **specials**: ±0/±Inf/±NaN (payload-carrying) at and around the boundary; the
   +0-at-index-0 elision case explicitly,
6. **all-negative**, **mixed-sign with negative threshold** (pass 1b + negfilter path),
7. **sampling-adversarial**: all top-k placed at positions `p mod 64 ≥ 8` (never
   sampled) — asserts correct result within the R_max retry budget,
8. **randn / uniform / lognormal** bulk shapes: N ∈ {2^12 … 2^24}, k ∈ {1, 5, 16, 17,
   100, 1000, 2047, 2048, 2049, 5000, N−1, N} — the arbitrary-k headline gets its first
   real sweep here,
9. **program-cache round trip**: two calls, second with different k and valid_length,
   asserting cache hit + correct output (k and valid_length are runtime-only).

Mutation controls at op level: (a) flip the negfilter threshold word's low half on the
negative path → tie tests go red; (b) zero the compression-enable bit → candidate CBs
overflow loudly (asserts the overflow detection, not a hang); (c) drop the +1 index
offset → specials case 5 goes red. A **chained/multi-chunk config is mandatory in every
suite** (the branch's rebuild bug survived every single-chunk test — HANDOFF.md traps).

**Level 3 — perf** (only after Level 2 is green): add arms to
`tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py` (per-cell
subprocess, Tracy Device Kernel Duration, correctness-gated timing, HEAD-sha-stamped) —
`threshold` vs `op` (topk_large_indices) vs `routed` at the §5 table's (k, N, C) points,
plus the fallback worst case as its own labeled arm. 3+ trials, cache cleared, noise
floor reported; a delta inside the noise floor is not a result.

---

## 8. Implementation work-plan

Gated phases, per the branch's plan style (a phase may exit with a negative result).

**Phase A — micro-measurements that the model depends on (3–5 days, tt-llk harness).**
1. bf16 `unpack_to_dest` stream cost (the fp32 3.9/vec was the measured point; bf16
   assumed equal — verify).
2. Histogram per-tile readback cost with and without `tensix_sync` batching (R6).
3. Denormal preservation through the bf16→Dst unpack convert (R4) — a 20-line variant of
   `pack_zero_compress_test.cpp`.
4. Fuse-map macro (SFPIADD stride + SFPOR + store) at the predicted ~2.0 cyc/vec, with
   an exact-index correctness twin.
Gate: numbers land within 1.5x of the §5 model, or the model is re-derived before any
kernel work.

**Phase B — single-core end-to-end (the de-risking milestone, ~2–3 weeks).**
One core, one row, whole pipeline minus the reduce/gather (its own slice is the row):
pass 1 → local threshold pick → pass 2 → cascade → local refinement → output. Level 1 +
Level 2 (single-core column) suites green, including all-equal fallback. This retires
R1/R3/R4/R5/R7/R8 on silicon before any multi-core complexity exists.

**Phase C — multi-core factory (~2 weeks).**
Reduce/broadcast/gather + retry protocol + row-parallel path + spill mode + validate().
Level 2 full battery green.

**Phase D — perf + routing (~1 week).**
Canonical-sweep arms; small-N routing threshold to the bitonic op measured, not modeled;
`sorted=True` tail for k ≤ 2048 via `ckernel_sfpu_topk_xl.h` reuse.

**Files (new unless noted):**

```
ttnn/cpp/ttnn/operations/experimental/topk_threshold_select/
    topk_threshold_select.hpp / _nanobind.{hpp,cpp}                 (~150 lines)
    device/topk_threshold_select_device_operation.{hpp,cpp}         (~450)  validate(), specs, hash (k & valid_length runtime-only)
    device/topk_threshold_select_device_operation_types.hpp         (~80)
    device/topk_threshold_select_program_factory.{hpp,cpp}          (~900)  both factories, slice split, retry plumbing, spill mode
    device/kernels/
        reader.cpp                                                  (~150)  TensorAccessor stream, L1-resident retention mode
        compute_hist.cpp                                            (~250)  pass 1 (+1b sign-flip variant via compile-time arg)
        compute_filter.cpp                                          (~400)  pass 2: fuse + count + {relu | negfilter} + cascade
        compute_refine.cpp                                          (~300)  root: bit-descent / bitonic-tail refinement, tie take
        writer_reduce.cpp                                           (~350)  hist/count gather, broadcast, candidate ship (writer_tree idiom)
        writer_root.cpp                                             (~250)  candidate recv, output stream, spill mode
        threshold_select_common.hpp                                 (~200)  fused-key/threshold-word helpers, decision-rule constants
tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/
    ckernel_sfpu_threshold_filter.h                                 (~300)  negfilter promoted from tests/sources/topk_negfilter_common.h
    ckernel_sfpu_fuse_index.h                                       (~200)  fuse map + count_above entry points
    ckernel_pack_exp_histogram.h                                    (~250)  enable/clear/readback helpers (math-thread CLREXPHIST contract)
tt_metal/hw/ckernels/blackhole/metal/llk_api/  (edits)              (~200)  API wrappers for the three headers above
tests/  (tt-llk Level-1 suites + ttnn Level-2/3 suites)             (~2500)
```

**Estimated diff: ~6.5–7K lines** (roughly the size of `topk_large_indices` plus its
test surface — consistent with a two-pass op that carries a reduce tree AND a fallback).

**Explicitly out of v1 scope:** the raw-UNPACR/split-Dest handshake removal (3.32x
stream headroom — separate LLK project); fp32/integer dtypes; WH port; single-DRAM-read
fused-pass variants; `sorted=True` above k=2048.

---

*Grounding artifacts: `SORTING.md` (§0a, §0a-bis, §0a-ter, §0b, §2.3–2.4, §5.3),
`HANDOFF.md` (FINAL RESULTS, traps), `tt_metal/tt-llk/tests/sources/{pack_exp_histogram,
pack_zero_compress, sfpu_count_above, topk_negfilter}*`,
`ttnn/cpp/ttnn/operations/experimental/topk_large_indices/` — all on branch
`nkapre/sorting`, all measured on the Blackhole in this machine.*
