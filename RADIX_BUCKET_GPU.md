# RADIX_BUCKET_GPU.md — Radix/Bucket/Histogram Selection: the Literature, and What It Certifies on Blackhole

**Purpose.** Engineering dossier on the *selection-without-sorting* family — radix select,
bucket select, histogram-guided top-k — as (a) certification that the exponent-first
bucket-select design sketched in `SORTING.md` is a member of a well-studied, exact algorithm
family rather than an ad-hoc trick, and (b) design input for a Tenstorrent Blackhole
implementation. Every claim is cited to a verified external source, or explicitly tagged
**[local finding]** (measured on BH silicon in this repo, see `SORTING.md`) or
**[inference]** (this document's own reasoning, unverified).

Citation verification: all papers/repos below were located by web search on 2026-08-16;
the Alabi et al. paper was read directly (pp. 1–7 of the author PDF). Where a secondary
summary was the only source for a detail, that is flagged.

---

## 1. The GPU lineage

### 1.1 Comparison table

| Algorithm | Year / venue | Digit / bucket scheme | Passes over data | Refinement strategy | Exact? | Tie handling | Claimed perf vs sort-based |
|---|---|---|---|---|---|---|---|
| **radixSelect** (Alabi et al.) | 2012, ACM JEA | MSD-first, 4-bit hex digits → 16 bins | ≤ 8 (fp32) / ≤ 16 (fp64) digit passes; each pass touches only surviving candidates (or in-place saturates losers) | Recurse into the digit bin containing the k-th element (`KDigit`); terminate when bin has 1 element or LSB reached | Yes | Digits are counted exactly; equal keys share all digits, so any of the tied elements is a correct k-th value | Several× faster than `thrust::sort`+choose (Merrill LSD radix sort) |
| **bucketSelect** (Alabi et al.) | 2012, ACM JEA | 2^10 = 1024 equispaced buckets by linear projection `⌊B·(x−min)/(max−min)⌋` | Iterative: one assignment pass per level, re-projecting only the k-th bucket | Re-bucket the `Kbucket` with tightened [min,max] until bucket width < machine precision → all-equal bucket = answer | Intended exact, with a documented near-zero numerical failure in the published implementation | Terminates on a bucket of equal values; the k-th value is that value | 6× over GPU sort for fp32 n > 2^24; up to **19.1×** for fp64 n = 2^28; its adversarial “Bucket Killer” is dramatically slower than sort/radix select |
| **Merrill & Grimshaw radix sort** | 2011 (base of Thrust/CUB sort) | LSD, digit-binning with per-block counting + scan | one binning pass per digit, full data each pass, ~3n global traffic/pass | n/a (full sort) | Yes (sort) | Stable | The *baseline* the select papers beat |
| **Onesweep** (Adinets & Merrill) | 2022, arXiv 2206.01784 | LSD radix sort; single-pass chained-scan digit binning | ~2n global read/writes per digit pass (vs ~3n prior) | n/a (full sort) | Yes (sort) | Stable | 29.4 GKey/s on A100 (256M 32-bit keys); ~1.5× over CUB |
| **Dr. Top-k** (Gaihre et al.) | 2021, SC'21, arXiv 2109.08219 | *Delegate-centric pre-pass*: partition input into subranges, take each subrange's max ("delegate"), run top-k on delegates, then filter the full input against the resulting threshold | 1 delegate pass + 1 filter pass + second-stage top-k on a tiny survivor set | Delegate top-k gives an upper-bound threshold; the filter pass restores exactness | Yes | Inherited from the second-stage kernel (radix/bucket/bitonic — Dr. Top-k wraps all three) | Reduces second-stage workload "up to more than 99%"; speeds up radix/bucket/bitonic top-k baselines; multi-GPU |
| **AIR Top-k** (Zhang, Naruse, Li, Wang — NVIDIA) | 2023, SC'23, DOI 10.1145/3581784.3607062 | MSD radix select, multi-bit digits; *iteration-fused* — all digit passes in one kernel launch, no CPU↔GPU round-trips; adaptive strategy skips re-materializing candidates when the surviving set is still large (digit width in the RAFT implementation: 8/11-bit passes — **[inference]** from RAFT source, not re-verified here) | 1–⌈bits/digit⌉ fused passes; adaptive early-exit | Recurse on threshold bin; adaptivity chooses between re-scanning original input vs compacted candidate buffer by measured candidate fraction | Yes | Boundary bin: takes the first (k − count_above) elements of the threshold bin; among equal keys selection is arbitrary but count-exact | 1.98–21.48× (batch=1) and 8.01–574.78× (batch=100) over the prior radix top-k; 1.44–7.34× / 1.38–31.91× over SOTA overall |
| **GridSelect** (same paper) | 2023, SC'23 | Not radix: grid-wide shared priority queue, two-step parallel insertion, on-the-fly (single pass) | 1 | n/a — maintains exact running top-k | Yes | Queue order among equal keys arbitrary | Up to 882.29× over BlockSelect (warp-select style baseline), esp. batch=1, large n |
| **RadiK** (Li et al.) | 2024, ICS'24, arXiv 2501.14336, github.com/leefige/radik | MSD radix select; 12-bit first digit for fp32; optimized for memory traffic with block-local histograms, global aggregation, and a flush-efficient write buffer | ~3 passes for 32-bit keys | Recurse on threshold bin; **adaptive scaling subtracts a sampled input from all keys** to redistribute clustered float exponents | Yes | Count-exact boundary-bin split | Up to 2.5× (non-batch) and 4.8× (batch) over prior art; adaptive scaling worth up to 2.7× on adversarial distributions; supports large k |
| **sampleselect** (Ribizel & Anzt) | 2019 (ICL-UT tech report / IPDPS-W line) | Splitters chosen by *sampling* the input, buckets = sample-defined ranges | histogram pass + recurse | Approximate variant stops early (k-th value approximate); exact variant recurses into the boundary bucket | Both modes offered — sampling alone is approximate; exact mode restores the guarantee by recursion | Boundary-bucket recursion | Faster than sort-based selection on V100-class GPUs (their measurements) |
| **FlashInfer sorting-free sampling** (Ye et al.) | 2025 (blog 2025-03-10; FlashInfer: MLSys'25 best paper) | No histogram at all: fused **rejection sampling** — sample via inverse transform, use the sampled probability as a pivot, reject-and-resample; **dual-pivot** variant (v0.2.3) brackets with low/high pivots | Multiple rounds inside one kernel; each dual-pivot round halves the bracket → O(log 1/ε) rounds worst case | Pivot refinement per round | Yes *in distribution*: proven to sample token j with prob p_j/Z of the exact top-k/top-p-filtered distribution (it never materializes the top-k set) | >50% sampling-latency reduction vs sort-based top-k/top-p sampling on large vocabularies |

Sources: Alabi, Blanchard, Gordon, Steinbach, *Fast K-selection Algorithms for Graphics
Processing Units*, ACM J. Experimental Algorithmics 17 (2012), DOI 10.1145/2133803.2345676
(author PDF: blanchard.math.grinnell.edu/Research/ABGS_KSelection.pdf — read directly);
Adinets & Merrill, *Onesweep: A Faster Least Significant Digit Radix Sort for GPUs*,
arXiv:2206.01784 (2022); Gaihre et al., *Dr. Top-k: Delegate-Centric Top-k on GPUs*, SC'21,
arXiv:2109.08219, DOI 10.1145/3458817.3476141; Zhang, Naruse, Li, Wang, *Parallel Top-K
Algorithms on GPU: A Comprehensive Study and New Methods*, SC'23, DOI
10.1145/3581784.3607062; Li et al., *RadiK: Scalable and Optimized GPU-Parallel Radix Top-K
Selection*, ICS'24, DOI 10.1145/3650200.3656596, arXiv:2501.14336; Ribizel & Anzt,
*Approximate and Exact Selection on GPUs* (icl.utk.edu/files/publications/2019/icl-utk-1230-2019.pdf);
FlashInfer, *Sorting-Free GPU Kernels for LLM Sampling*, flashinfer.ai/2025/03/10/sampling.html.

### 1.2 Notes per algorithm

**Alabi et al. 2012 — the family's founding GPU paper.** Verified directly from the PDF.
Both algorithms are framed by the authors as one "Generic k-Selection Technique": *(1)
define bins and an assignment rule, (2) assign, (3) find the bin holding the k-th largest
via cumulative counts, (4) iterate on that bin.* radixSelect is explicitly "a selection
adaptation of a most-significant-digit radix sort" built on Merrill & Grimshaw's counting
code, with hexadecimal (4-bit) digits and a *bit-flipping* float↔uint correspondence
(§4.2 below). It has an **in-place** variant: instead of compacting the surviving bin,
losers below `KDigit` are zeroed and losers above are saturated to all-ones, so the next
digit pass runs over the same array — no auxiliary memory beyond 16 counters. bucketSelect
is "distributive partitioning" (Allison & Noga 1980) on the GPU: 1024 buckets chosen
empirically so counters fit in shared memory with `atomicInc`, a dummy bucket for
already-eliminated values, and re-projection of only the k-th bucket per iteration until
the bucket width falls under machine epsilon (all-equal bucket ⇒ exact answer). Uniform
data terminates in ~1 iteration; the paper explicitly tests adversarial distributions.
However, §4.2.5 reports that the implementation can fail for double inputs near machine
zero when `B/(max-min)` becomes a computational division by zero. Its adversarial “Bucket
Killer” is 135× slower than sort-and-choose and as much as 395× slower than radixSelect.
The algorithmic family is exact; this particular floating-point projection is not an
unconditional correctness primitive.

**Merrill lineage → Onesweep.** Onesweep is a *sort*, not a select — it is in this dossier
because (a) it is the modern form of the digit-binning machinery every radix select borrows
(per-digit histogram + prefix sum + scatter), and (b) `thrust::sort`/CUB descendants are the
baseline every select paper measures against. Its contribution is reducing per-digit global
traffic from ~3n to ~2n with a single-pass chained scan (decoupled-lookback style
inter-block prefix sum). The select papers win precisely by *not* paying even 2n per digit
for all digits: they touch only surviving candidates after pass 1.

**AIR Top-k / GridSelect (SC'23) — what ships in NVIDIA's stack.** AIR Top-k is the radix
select inside RAFT/cuVS (`matrix::select_k`), consumed by cuML/RAPIDS
(github.com/NVIDIA/raft). Its two ideas matter for TT: *iteration-fusion* (all passes in one
launch — the GPU analog of "don't round-trip to host between digit passes") and
*adaptivity* (choose whether to compact candidates or re-scan the original buffer based on
how many survived — compaction is a waste when the first digit barely discriminates).
GridSelect is the non-radix control: a single-pass priority-queue method that wins when k
is small and data arrives on-the-fly.

**Dr. Top-k (SC'21) — delegation as a *pre*-filter.** Its delegates are deterministic
subrange maxima, not a random or positional sample. Their top-k supplies a conservative
threshold, after which filtering and an exact second stage restore the result. The useful
analogy to the BH histogram is therefore limited: both are cheap threshold predictors whose
decisions must be verified by an exact stage; only the BH mechanism is sampled.

**RadiK (ICS'24) — radix select that takes clustered inputs seriously.** Its adaptive
scaling does not merely repartition work after observing a heavy bin: it subtracts a
randomly sampled input from every key, changing float exponents so clustered values spread
across radix buckets. That distinction matters on TT because it requires a full transform
and a defined policy for rounding and special values. The primary preprint confirms the
12-bit fp32 first digit, hierarchical histogram aggregation, flush-efficient write buffer,
and the reported 2.7× adversarial benefit.

**FlashInfer (2025) — the field's endpoint for LLM sampling.** When the consumer is a
*sampler* rather than a top-k *list*, FlashInfer shows you can skip selection entirely:
rejection sampling with pivot refinement produces a token distributed exactly as if top-k/
top-p filtering had been applied, in O(log 1/ε) fused rounds. Design input for TT: if the
op's contract is "sample a token," the certified family includes samplers that never
materialize the top-k set. If the contract is "return values+indices" (ttnn.topk), the
radix/bucket branch is the relevant one.

**GridSelect is real** — it is the second algorithm of the SC'23 paper, not folklore.

### 1.3 Missing comparators that matter for TT dispatch

- **Shanbhag, Pirk, Madden (SIGMOD'18)** directly models radix-select versus bitonic Top-K
  on massively parallel hardware. Their bitonic path wins through roughly k=256 in the
  evaluated regime and is robust to skew. This argues for dispatch by `(N,k,bucket mass)`,
  not a radix-only replacement.
- **FAISS WarpSelect** (Johnson, Douze, Jégou) is an exact, register-resident, single-pass
  selector carrying indices, designed for k≤1024 and producer fusion. It is the closest
  published reminder that TT's local bitonic/register path is a serious small-k competitor.
- **GPU Multisplit** (Ashkiani et al.) is the more direct source for histogram + prefix +
  hierarchical scatter/compaction mechanics. It is more relevant to a TT private-L1 plus
  NoC-reduction design than a full-sort paper alone.
- **RTop-K (ICLR'25)** is relevant for many short rows: its exact mode uses threshold
  search, while its early-stop mode is approximate. MoE-style short rows may therefore
  favor threshold/local methods rather than global radix machinery.

---

## 2. The CPU lineage (brief)

| Method | Year | Shape | Guarantee |
|---|---|---|---|
| Quickselect / FIND (Hoare) | 1961 | partition on pivot, recurse one side | exact; O(n) expected, O(n²) worst |
| Median-of-medians (Blum, Floyd, Pratt, Rivest, Tarjan) | 1973 | deterministic pivot | exact; O(n) worst case |
| Floyd–Rivest SELECT | 1975 | *sampling* to pick two pivots bracketing the k-th | exact; n + min(k, n−k) + o(n) comparisons expected |
| Introselect (Musser); `std::nth_element` implementations | 1997 | quickselect + depth-triggered fallback | exact; Musser's BFPRT fallback is worst-case linear, while the C++ `nth_element` contract requires average-linear comparisons and implementations vary |
| Distributive partitioning (Allison & Noga) | 1980 | linear-projection buckets — the direct ancestor of bucketSelect (cited as such by Alabi et al.) | exact |
| x86-simd-sort (Intel/NumPy) | 2022–24 | AVX-512 vectorized quickselect / partial sort; powers `np.partition`/`np.argpartition` | exact; up to ~25× (16-bit), ~17× (32-bit), ~8× (64-bit) over scalar (v3.0 release notes, github.com/numpy/x86-simd-sort) |
| Highway vqsort (Wassenberg et al., Google) | 2022 | portable vectorized quicksort; SIMD partition kernel is the same primitive a SIMD select needs | exact (sort) — opensource.googleblog.com 2022-06 |

Two CPU lessons transfer: **Floyd–Rivest** legitimizes *sampling* for pivot/threshold
estimation inside an exact algorithm (exactness comes from the verification/partition pass,
not the sample), and **Musser's introselect** legitimizes the *certified fallback* — run the
fast heuristic, count, and fall back to a heavier exact method when progress degenerates.
Do not infer Musser's worst-case-linear guarantee from the standard `std::nth_element`
contract; common library implementations use different fallbacks.

---

## 3. The common skeleton, its worst cases, and the standard mitigations

Every exact member of the family is the Alabi "generic technique" with different constants:

```
1. HISTOGRAM      count elements per digit/bucket (one streaming pass)
2. PREFIX-SUM     cumulative counts over the (tiny) histogram
3. LOCATE         the bucket b* where cum(b*−1) < n−k ≤ cum(b*)   ← the "threshold bucket"
                  everything in buckets above b* is a winner (count_above of them)
4. RECURSE/SCAN   need k' = k − count_above more winners from inside b* only:
                    - recurse until remaining digits identify an exact key class, or
                    - if |b*| is small, run an exact local selection/sort
5. VERIFY/EMIT    establish Cgt = #(key > T), Ceq = #(key == T), with
                  Cgt <= k <= Cgt+Ceq; emit all >T and exactly k−Cgt equals
```

Exactness argument (shared by all): steps 1–3 use *exact counts*, so `b*` provably contains
the k-th element and `count_above` winners are provably above it — no approximation anywhere
unless the histogram itself is approximate (then see §4.3). “Take the first k' from the
threshold bin” is valid only after all remaining key digits have been resolved or an exact
selection has run inside that bin. It is not valid for a coarse exponent bucket.

**Worst cases:**

- **Heavy boundary bucket / duplicate mass.** If one digit value dominates (all-equal
  inputs, saturated logits, quantized data), recursion makes no progress: |b*| ≈ n every
  level. This is the adversarial case Alabi et al. test explicitly, and the case RadiK's
  adaptive scaling exists for. For bucketSelect the built-in escape is analytic: once
  bucket width < machine precision the bucket is all-equal and *is* the answer. For radix
  select the escape is that equal keys share all remaining digits, so recursion terminates
  at the LSB with the tied value — correct, but only after paying every pass.
- **Skewed distributions vs equispaced buckets.** Linear-projection buckets (bucketSelect)
  degenerate on heavy-tailed data — most mass lands in few buckets. Mitigations: sampled
  splitters (sampleselect), or use *radix digits of the bit pattern* instead of value-space
  projection, which is exactly why radix select is preferred for floats: the exponent field
  is a built-in logarithmic bucketing of value space. **[inference]** on the last clause;
  the rest is per the cited papers.
- **Approximate front-ends.** Samples (sampleselect, Floyd–Rivest, the BH histogram) yield a
  prediction, not an answer. Dr. Top-k's deterministic delegates yield a conservative
  threshold, which is a related but stronger object. In either case, exact selection
  requires exact counting/refinement and a terminating fallback. A fixed positional sample
  need not even produce a guaranteed bracket, so one refinement pass is not a bound.

**Standard mitigations, as used in the literature:** multi-level digits (recurse with a new
digit rather than re-scanning full width — all radix selects); key-space scaling for
clustered inputs (RadiK); compact-vs-rescan decision by survivor fraction (AIR Top-k); fallback
sort of the boundary bucket when it is small (step 4's "scan" arm — universal); and the
introselect pattern of counting-then-falling-back for certified worst-case behavior.

---

## 4. Mapping to Tenstorrent Blackhole

### 4.1 The IEEE-bits monotone mapping, sign-magnitude order, and SFPGT

The trick underlying every float radix select (used by Alabi et al. as the "bit flipping
routine", and standard in GPU radix sorts since Herf's *Radix Tricks*, 2001,
stereopsis.com/radix.html): map the 32-bit IEEE-754 pattern `b` to

```
key(b) = b XOR ( (b >> 31) ? 0xFFFFFFFF : 0x80000000 )
```

i.e. **negative ⇒ flip all bits; non-negative ⇒ flip only the sign bit.** `key` is a
bijection and unsigned integer order on `key` is monotone for non-NaN numeric values, with
`−NaN < −Inf < … < −0 < +0 < … < +Inf < +NaN`. That total order is exactly the
**sign-magnitude integer order** of the raw bits (interpret bit 31 as sign, bits 30:0 as
magnitude). So "radix select on XOR-mapped keys" and "select in sign-magnitude order" are
the same algorithm with the map applied lazily or not at all. For NaNs this is a chosen,
deterministic bit order, not IEEE comparison semantics; payload/signaling order and the
public `ttnn.topk` contract must be established by differential tests.

**Blackhole's comparators natively implement this order.** `SFPGT` compares in the
sign-magnitude total order, and — **[local finding]**, `SORTING.md` §A2, found by a
*failing* test on the ±NaN specials — the packer's `MIN_THRESHOLD_RELU` does too: `+NaN`
survives a threshold, `−NaN` is zeroed, i.e. the packer orders
`−NaN < −Inf < … < −0 < +0 < … < +Inf < +NaN`, *not* IEEE compare semantics (where NaN
comparisons are false). This is a feature, not a bug, for this family: the hardware
compare order **is** the radix-on-IEEE-bits order, so threshold filtering, bucketing by
exponent, and the final compare all agree on one total order — the one top-k wants.
For non-negative data the XOR map degenerates to a sign-bit flip, so the raw biased
exponent field is already monotone in value.

### 4.2 The exponent as the first radix digit

For fp32/bf16 the biased exponent is an 8-bit field ⇒ **256 buckets**, and for non-negative
values, value order is lexicographic in (exponent, mantissa). An exponent histogram is
therefore precisely **pass 1 of an MSD radix select with an 8-bit first digit** — the same
shape as AIR Top-k / RadiK pass 1, just with the digit boundary aligned to the exponent
field instead of an arbitrary bit offset. The sign bit is a 1-bit digit *above* it
(handled first: with signs present, all positives outrank all negatives after the map;
count positives, and only if k exceeds that count does the negative half matter — where
exponent order *reverses*). **[inference]** on the composition; each ingredient is standard.

The mantissa (23 bits fp32, 7 bits bf16) is the remaining digit string: if the boundary
exponent bucket is heavy, recurse on mantissa bits (multi-level digits, §3) or fallback-sort
the bucket (`topk_local_sort` exists in-tree and was measured/optimized —
**[local finding]**, `SORTING.md`). For bf16 the boundary bucket has ≤ 2^7 distinct
magnitudes, so a single fallback level suffices.

### 4.3 The packer exponent histogram — free, but sampled and aliased **[local finding]**

All facts in this subsection are measured on BH silicon (`SORTING.md` §A2–A6 and
"MEASURED: the packer exponent histogram WORKS on Blackhole";
`tt_metal/tt-llk/tests/sources/pack_exp_histogram_test.cpp`, `pack_exp_histogram_perf.cpp`;
38/38 functional + 6/6 perf tests). No file in the tree used this hardware before.

- **Cost: zero.** Enabling it during a pack that happens anyway: 25.175 → 25.104 cyc/tile
  (−0.28%, i.e. noise); `CLREXPHIST` costs exactly 1 cycle.
- **It samples 1 datum in 8**, in the fixed positional pattern `p mod 64 < 8` — 128
  increments per 1024-datum tile, format-independent. The WH documentation
  (`Packers/ExponentHistogram.md`) says per-datum; on BH it is a **12.5% positional
  sample**. Proved by construction (marker patterns + mod-8 phase sweep).
- **Bins alias: `Exponent & 31`** ⇒ 32 bins, exp 127 and 159 collide (fp16's 5-bit
  exponent is clean). So the hardware gives a *5-bit sub-digit* of the 8-bit exponent
  digit, not the full 256-bucket histogram.
- **8-bit counters saturate at 255** (no wrap); `WhichPackers` is ignored on read modes
  6/7 (summing the four reads 4×-counts — the trap); max-exponent mode 9 is *also*
  subsampled (a 1-in-1024 outlier is missed); `CLREXPHIST` from the PACK thread does not
  fence in-flight PACRs (~39-count leak) — issue it from the math thread ordered by the
  dest semaphore.
- **It is sign-blind** (ranks |x|), so the sign digit must be handled before it (§4.2 —
  or via the `MIN_THRESHOLD_RELU` filter for the T ≥ 0 case).
- **Measured primitive, modeled pipeline:** enabling histogram collection costs no
  measurable PACK throughput and `CLREXPHIST` adds 1.079 cycles/tile in the measured table
  (approximately one architectural cycle). The often-quoted **128-cycle search** for
  N=32768 is a composition of per-tile clear/read assumptions, not an end-to-end timing:
  modes 6/7/9 readback, `tensix_sync`, RISC aggregation, prefix/locate, refinement,
  extraction, and final sorting were not in the timed loop. Likewise 3,072, 24,876, and
  1,267 cycles are component-based models, not measurements of a working Top-K operator.

**Where this sits in the literature:** a 1-in-8 positional sample of a 32-bin aliased
histogram is *not* the exact histogram of §3 step 1. It is a sampled threshold predictor,
closer to sampleselect/Floyd–Rivest sampling than to Dr. Top-k's deterministic delegates.
Because the fixed sampled positions can miss every extreme value, it does not guarantee a
bracket. The exact construction must therefore be: predictor → exact `(Cgt,Ceq)` decision
→ monotone digit refinement or exact local fallback → emit only after
`Cgt <= K <= Cgt+Ceq`. The predictor may improve expected work but cannot decide
correctness, termination, or a one-refinement bound.

### 4.4 The packer threshold filter — free compare-and-zero for T ≥ 0 **[local finding]**

`MIN_THRESHOLD_RELU` zeroes every datum below threshold *during the pack*, so the compare
costs zero SFPU work: the measured `relucomp` arm runs at 4.034 cyc/vector vs a 3.855
unpack-bound floor, a 4.6% delta. With packer zero-compression, a dense fp32 fused-key tile
`[bf16 value | u16 index]` emits 4096 → 640 B with 32 survivors.

That sparse PACR stream is **not yet a production compaction primitive**. Its row-start
metadata and 4-bit zero-run encoding are decoded only by host-side tt-llk tests; no in-tree
BH device decoder or ordinary CB/unpacker consumer was found. `DataflowBuffer` exposes
fixed-size entries, not variable-byte entries carrying compression metadata. Long zero
runs also emit placeholder datums, so `PackerTileSize` is a byte count, not an exact
survivor count. A device-to-device consume/round-trip path, or a separate explicit
gather/rescan compactor, is required before this mechanism belongs in an operator cost
model.

The tested fused key is also narrower than the public contract: it carries only a u16
index, orders equal positive values by the larger low 16 bits, and behaves differently for
negative values. It is suitable only for a deliberately `stable=false`, u16-scoped
prototype until sign-aware tie encoding and UINT32/global-index reconstruction are proven.

**The T < 0 hole is real and closed.** A negative threshold is documented UB and measured
to behave as |T| rounded up to the next power of two (mantissa ignored) — genuinely
unusable. The fallback is an SFPU filter (`SFPGT`(SET_VD) + `SFPAND`, 2 issues/vector,
proven to be the ISA floor). The reported 41.4% (non-negative packer arm) and 6.7% (signed
SFPU arm) deltas compare one filter pass with one `_topk_xl_merge_` primitive; they are not
complete Top-K speedups and must not be used for routing. Ties exactly on a negative
threshold are zeroed, not kept — asserted in tests.

### 4.5 Where TT differs from the GPU papers

| GPU mechanism (papers §1) | Blackhole reality | Consequence |
|---|---|---|
| Shared-memory atomics for block-local histograms (`atomicInc` — Alabi; hierarchical atomics — RadiK) | SFPU lanes cannot issue CUDA-style shared-memory atomics, but data-movement RISCs can issue NoC atomics to local/remote L1 | Benchmark per-core private L1 histograms + tree reduction against NoC-atomic accumulation. Do not assume either wins. **[inference]** |
| Global barrier / grid sync between digit passes; Onesweep's chained scan to avoid it | No global sync primitive; cross-core sync is semaphores (`noc_semaphore_*`), multicast | Digit passes become **semaphore levels**: cores publish partial histograms, one core (or a tree) prefix-sums and multicasts `b*`/threshold, cores proceed. Iteration-fusion (AIR) maps to keeping all levels inside one program launch — TT programs already are. **[inference]** |
| Candidate compaction to global memory with write buffers (RadiK) vs re-scan (AIR's adaptive choice) | Zero-compression emits a sparse proprietary PACR stream, but there is no production BH device consumer and byte length is not survivor count (§4.4) | Compact-vs-rescan remains open. Gate the sparse stream against explicit prefix/gather compaction and simple rescanning. **[local finding]** for emission, **[inference]** for any operator mapping |
| Histogram pass reads global memory at ~2n bandwidth (Onesweep bound) | Histogram is free *inside an existing pack* (§4.3); a dedicated pack pass costs 806 cycles for N=32k (win drops to 2×/12.5× but survives) | The "count pass" can piggyback on whatever op produced the logits (e.g. the matmul's own pack). **[local finding]** for costs, **[inference]** for fusion |
| fp32 keys XOR-mapped in registers before binning | Comparators natively sign-magnitude (§4.1); no map needed for compare/filter; exponent field directly addressable | One fewer transform; but the histogram hardware is |x|-only, so the sign digit is handled by control flow, not by the map. **[local finding]** + **[inference]** |

---

## 5. Suitability verdict and gated plan

**Verdict: suitable for an SFPU-assisted threshold-selection research prototype; not yet
suitable as a production exact radix Top-K.** The literature pedigree is valid for a
design that actually implements exact MSD refinement. Blackhole's strongest demonstrated
fit is the compare/filter stage: `SFPGT` supplies the desired deterministic order, the
negative-threshold bit-preserving filter reaches its two-issue floor, and the packer can
provide cheap exponent telemetry. The missing pieces are the selector itself: no exact
digit-refinement state machine, device-side sparse consumer, dense extraction, boundary-tie
allocator, or whole-op implementation exists.

The hardware histogram must be treated only as an optional predictor. A correct row-level
state machine needs the invariant

```
Cgt = #(key > T), Ceq = #(key == T), and Cgt <= K <= Cgt + Ceq
```

before emission. If the invariant is false, the kernel must monotonically refine another
sign/exponent/mantissa digit or invoke a bounded exact fallback. If it is true, emit every
strict winner and exactly `K-Cgt` equal keys using the requested tie policy, then sort the
K results when `sorted=true`.

### 5.1 Initial research scope

Start with **Blackhole + BF16 + `largest=true` + `stable=false` + K≤32/64 + long rows**.
Do not initially route FP32, UINT32-width indices, `largest=false`, stable ties, or sharded
outputs. Do not position this as a replacement for the current column-parallel
`topk_large_indices` path at K=512/1024/2048; its measured 24–89 µs whole-op baseline is
the competitor, not the historical `_topk_xl_merge_` micro-op.

### 5.2 Gates

1. **Contract/oracle.** Define raw-key order, NaNs and payloads, ±0, infinities,
   subnormals, padding, duplicate-boundary allocation, global indices, direction,
   `sorted`, and `stable`. Differential-test the shipping operation rather than assuming
   sign-magnitude order is its public semantic contract.
2. **Exact single-core reference.** Implement sign handling, exact radix digits,
   `(Cgt,Ceq)`, progress detection, boundary allocation, and a bounded `topk_local_sort`
   fallback without the packer predictor. Require bit-exact values/valid indices on random
   and adversarial inputs.
3. **Estimator-guided path.** Add the sampled histogram only after Gate 2. Histogram OFF
   and ON must return identical results. Measure refinement-count p50/p95/max and include
   positional anti-samples, `exp&31` collisions, saturation, all-equal, duplicate-mass,
   alternating-sign, all-negative, ±0/Inf/NaN, and tile-face-correlated inputs.
4. **Compaction decision.** Build a production-style device-to-device compressed-stream
   round trip carrying explicit `(offset,length,format)` metadata, or exclude packer
   compression from v1. Compare its complete consume/cascade cost against explicit
   prefix/gather compaction and rescanning. A producer-only packed-byte measurement is not
   sufficient.
5. **Whole-op A/B.** Measure Tracy Device Kernel Duration and total op duration against the
   best current path for every cell: stock local/multicore Top-K and column-parallel
   `topk_large_indices`. Charge standalone histogram packing, final extraction, index
   reconstruction, and output sorting. Route only an empirically won region.
6. **Multicore last.** Partition rows into width shards and A/B private L1 histograms plus
   tree reduction against data-movement-RISC NoC atomics. Keep digit levels in one program
   launch with versioned semaphore epochs. Require bounded progress, no hangs, exact
   single-core identity, and measured synchronization below the benefit.

### 5.3 Go/no-go criteria

Production routing requires: exact contract tests; no unbounded refinement; no sparse-stream
metadata ambiguity; no hangs under repeated/multicore launches; a whole-op win over the best
current per-cell baseline; and a guarded fallback with negligible regression outside the
won region. Until those gates pass, the accurate claim is **“promising hybrid research
hypothesis,” not “certified Blackhole implementation.”**

---

## References

1. T. Alabi, J. D. Blanchard, B. Gordon, R. Steinbach. *Fast K-selection Algorithms for
   Graphics Processing Units.* ACM J. Experimental Algorithmics 17 (2012).
   https://dl.acm.org/doi/pdf/10.1145/2133803.2345676 ·
   https://blanchard.math.grinnell.edu/Research/ABGS_KSelection.pdf
2. D. Merrill, A. Grimshaw. *High Performance and Scalable Radix Sorting.* Parallel
   Processing Letters 21(2), 2011. (Cited via [1]; basis of thrust::sort/CUB.)
3. A. Adinets, D. Merrill. *Onesweep: A Faster Least Significant Digit Radix Sort for
   GPUs.* arXiv:2206.01784 (2022). https://arxiv.org/abs/2206.01784 ·
   https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
4. A. Gaihre, D. Zheng, S. Weitze, L. Li, S. L. Song, C. Ding, X. S. Li, H. Liu.
   *Dr. Top-k: Delegate-Centric Top-k on GPUs.* SC'21. https://arxiv.org/abs/2109.08219 ·
   https://dl.acm.org/doi/10.1145/3458817.3476141
5. J. Zhang, A. Naruse, X. Li, Y. Wang. *Parallel Top-K Algorithms on GPU: A Comprehensive
   Study and New Methods.* SC'23. https://dl.acm.org/doi/10.1145/3581784.3607062
   (AIR Top-k, GridSelect; shipped in NVIDIA RAFT/cuVS: https://github.com/NVIDIA/raft)
6. Y. Li, B. Zhou, J. Zhang, X. Wei, Y. Li, Y. Chen. *RadiK: Scalable and Optimized
   GPU-Parallel Radix Top-K Selection.* ICS'24. https://arxiv.org/abs/2501.14336 ·
   https://dl.acm.org/doi/10.1145/3650200.3656596 · https://github.com/leefige/radik
7. T. Ribizel, H. Anzt. *Parallel Selection on GPUs.* Parallel Computing 91 (2020),
   102588. https://doi.org/10.1016/j.parco.2019.102588 · earlier technical report:
   https://icl.utk.edu/files/publications/2019/icl-utk-1230-2019.pdf
8. FlashInfer. *Sorting-Free GPU Kernels for LLM Sampling.* 2025-03-10.
   https://flashinfer.ai/2025/03/10/sampling.html ·
   https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html
9. C. A. R. Hoare. *Algorithm 65: FIND.* CACM 4(7), 1961. (Cited via [1].)
10. M. Blum, R. Floyd, V. Pratt, R. Rivest, R. Tarjan. *Time Bounds for Selection.*
    JCSS 7(4), 1973.
11. R. Floyd, R. Rivest. *Expected Time Bounds for Selection.* CACM 18(3), 1975.
12. D. Musser. *Introspective Sorting and Selection Algorithms.* Software: Practice and
    Experience 27(8), 1997.
13. S. Allison, M. Noga. *Selection by Distributive Partitioning.* Information Processing
    Letters, 1980. (Cited via [1] as bucketSelect's ancestor.)
14. numpy/x86-simd-sort (AVX-512 quickselect / partial sort; np.partition up to 25×).
    https://github.com/numpy/x86-simd-sort (v2.0/v3.0 release notes)
15. J. Wassenberg et al. *Vectorized and performance-portable Quicksort* (Highway vqsort).
    Google Open Source Blog, 2022.
    https://opensource.googleblog.com/2022/06/Vectorized%20and%20performance%20portable%20Quicksort.html
16. M. Herf. *Radix Tricks.* 2001. http://stereopsis.com/radix.html (IEEE-bits monotone
    XOR mapping.)
17. **[local findings]** `SORTING.md` (this repo): §A2 sign-magnitude packer order; §A3
    1-in-8 sampled exponent histogram, pattern `p mod 64 < 8`; §A4 `WhichPackers` ignored
    on modes 6/7; §A5 mode-9 max-exponent subsampled; §A6 `CLREXPHIST` fencing; §0a-ter
    negative-threshold behavior and SFPU fallback; measured primitive costs and composed
    payoff models. Tests:
    `tt_metal/tt-llk/tests/sources/pack_exp_histogram_test.cpp`,
    `pack_exp_histogram_perf.cpp`, `topk_negfilter_*.cpp`; host sparse decoder:
    `tt_metal/tt-llk/tests/python_tests/test_pack_compress_int32.py`; fixed-entry CB API:
    `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h`.
18. A. Shanbhag, H. Pirk, S. Madden. *Efficient Top-K Query Processing on Massively
    Parallel Hardware.* SIGMOD 2018. https://doi.org/10.1145/3183713.3183735 ·
    https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf
19. J. Johnson, M. Douze, H. Jégou. *Billion-scale similarity search with GPUs.* 2017.
    https://arxiv.org/abs/1702.08734 (FAISS WarpSelect.)
20. S. Ashkiani et al. *GPU Multisplit.* 2017. https://arxiv.org/abs/1701.01189
21. RTop-K. *A Top-K Operator for Efficient LLM Inference on GPUs.* ICLR 2025.
    https://xiexi51.github.io/assets/pdf/RTopK.pdf

---

## 6. Appendix — Swarm feasibility audit (2026-08-16)

**Method.** Seven adversarial personas (Verifier/SFPU-ISA, Expert/LLK, Auditor/citations,
Critic/skeptic+devil's-advocate, Supporter/steelman, Implementer/Gate-2, Random
Stranger/GPU fresh-eyes) audited this dossier in two rounds — independent analysis, then
all-see-all debate with mandatory re-verification against ground truth before any
challenge. 14 agents, 224 tool calls, ~1.55M tokens, at repo commit `4b3ebaef8e9`. Ground
truth consulted: `SORTING.md`, `tt_llk_blackhole/common/inc/sfpu/*` (incl. the working-tree
`ckernel_sfpu_topk.h`), `tt-isa-documentation/BlackholeA0` SFP*/Dst/LReg pages, the WH
`Packers/ExponentHistogram.md`, `pack_exp_histogram_*` / `topk_negfilter_*` test sources,
and the campaign memory. Every finding carries file:line evidence; every finding was voted
confirm/dispute/abstain by all seven personas after debate.

**Tally.** 40 findings: **39 Confirmed, 1 Discarded** (SUPP-2, killed 0/7 — see §6.4).
Severity: 2 CRITICAL, 14 HIGH, 14 MEDIUM, 10 LOW. Votes: 259 confirm / 15 dispute / 0
abstain across 274 ballots; 42 challenges issued, 33 positions revised.
**Anti-herd status: PASSED, borderline** — flip_rate 0.83 (above the 0.8 alarm),
vote entropy 0.31 (just above the 0.3 alarm). The convergence is evidence-driven rather
than social (the one groupthink candidate, SUPP-2, was killed by four personas
independently re-reading the same source lines), but minority positions are preserved
in §6.4 and should not be discarded.

### 6.1 Consensus verdict

All seven personas converged, from disjoint evidence, on the same recalibration:

1. **The primitive layer is certified.** Every load-bearing hardware claim in §4 traces
   to an ISA page or a silicon test: SFPGT's sign-magnitude total order is stated
   verbatim in the ISA (`SFPGT.md:3,55-66` — the exact XOR-map order of §4.1), the packer
   MIN_THRESHOLD_RELU order, the digit-extraction toolkit (SFPEXEXP/SFPEXMAN/SFPSETSGN —
   real, near-free), histogram zero marginal cost, and the 2-issue filter floor (scoped:
   for *value-preserving* filters; a 1-issue mask-materializing map exists) all verify.
   The dossier's epistemic tags are honest almost everywhere.

2. **The radix framing does not fit this hardware.** SFPU has no scatter, no indexed
   registers, no atomics; exact multi-bin digit counting was *measured and rejected* by
   this campaign's own data (HistNibble 5.0 cyc/vec unclamped is a loser once SFPSHFT
   mod-32 wrap clamping and SFPIADD drain are charged, `SORTING.md:1288-1292`; full-width
   software histogram 20,580 cyc at N=32k). The measured exact-refinement menu is:
   **1 bit @ 2.0 cyc/vec** (CountD1 predicated count, an architectural floor) or
   **3 bits @ 3.0 cyc/vec** (HistMacro+HistSum, `SORTING.md:1305-1342`), each
   data-dependent decision costing a **≥25.1-cyc RISC rendezvous** (PassSync). The exact
   arm is therefore *threshold bisection with quickselect economics* — not MSD radix
   digits. §4.2's "recurse on mantissa bits (multi-level digits)" and the "pass 1 of an
   8-bit MSD radix select" framing should be rewritten accordingly.

3. **The composed economics omit measured costs.** Uncharged in every cycle model:
   per-tile `CLREXPHIST` + `tensix_sync` pipeline drains forced by 255-saturating
   counters exactly on the chosen long-row scope; the count pass being additive to the
   3.938 cyc/vec unpack floor on the stock LLK path (with the §0a-bis disjoint-Dest
   concurrency escape as the measured way out); the 806-cycle standalone pack whenever
   producer fusion is absent — and no TTNN fusion mechanism exists (this campaign already
   hit the fused-only trap once).

4. **Gate 4, not Gate 5, is the go/no-go pivot.** With no device-side consumer for the
   compressed stream and no gather/compaction instruction, every emit path collapses to
   the full value-independent bitonic sort, making all selector passes strictly additive
   on most cells (the measured ~6,509-cyc cascade-to-K=32 model is the one modeled
   exception, and its emit leg is exactly the unconsumable compressed stream). The exact
   Gate-2 reference is a *correctness oracle*, provably slower than the incumbent until
   Gate 4 delivers compaction or count-guided tile skipping.

5. **Every quoted competitor number is stale, disputed, or strawman.** The 2×/12.5×
   ratios compare against internal threshold-search pipelines the incumbent never runs;
   the 3.855/6.879 denominators are tagged [DISPUTED] in their own source; the 24–89 µs
   `topk_large_indices` baseline predates the log-tree merge (now ~15–42 µs) and belongs
   to a K-regime §5.1 excludes; the §5.1 cell itself (BF16, K≤64, long rows) has no
   measured baseline at all (archived stock-multicore point: ~171 µs at N=32k/K=32); and
   the same cell measured 2.3× apart on the same day under different harnesses — Gate-5
   targets must pin the harness, not just the number.

6. **The scope is self-defeating as written.** K≤32/64 is precisely the regime this
   dossier's own §1.3 literature (Shanbhag SIGMOD'18, FAISS WarpSelect) assigns to
   bitonic/register selection; "long rows" maximizes counter saturation and is unbounded
   while the u16 fused key silently requires N≤65,536 (in-tree sweeps go to 131,072).
   For signed data the design loses both hardware assists at once (sign-blind histogram +
   T<0 packer UB).

7. **Two feasibility positives the dossier missed.** (a) The (Cgt,Ceq)-to-branch
   mechanism has a complete instruction sequence: SFPGT(SET_VD)+SFPIADD accumulate,
   SFPTRANSP + SFPSHFT2-SHFLROR1 cross-lane fold, SFPSTORE to Dst, then the TRISC reads
   the scalar via *memory-mapped Dst* at `0xFFBD_8000` (`Dst.md:103`) — no pack
   round-trip; cost unmeasured (honest prior 25–100 cyc/pass). (b) SFPGT's full-word
   sign-magnitude order lets the bisection run on raw bits directly, collapsing §4.2's
   sign-digit/exponent-reversal special-casing.

**Recalibrated verdict** (replacing §5's): *compare/filter primitives certified; the
"radix select" is actually sampled-pivot quickselect with a free-but-lossy predictor;
selector feasibility unknown pending a measured Gate-2 refinement loop; no win region
exists before device-side compaction (Gate 4); the strongest defensible subset is
predictor-threshold + free filter + exact count + existing bitonic finish (the Dr. Top-k
shape this dossier itself names at §1 — and the shape THRESHOLD_SELECT_DESIGN.md already
sketches).*

### 6.2 Ranked corrections queue

Edits to this dossier, ranked by consensus priority (severity × confidence × votes):

| # | Correction | Driven by |
|---|---|---|
| 1 | Rewrite §4.2 + Gate 2: exact arm = threshold bisection (1 bit @ 2.0 or 3 bits @ 3.0 cyc/vec; ≥25.1-cyc rendezvous per decision), not multi-bin radix digits; cite HistNibble as measured-and-rejected | IMPL-1, VERI-1, STRA-1 |
| 2 | Reorder §5: Gate 4 (device consumer for compressed stream, or count-guided tile skipping) is the load-bearing go/no-go; Gate 2 is a correctness oracle only | IMPL-2, EXPE-1 |
| 3 | Charge the omitted costs in §4.3/§4.5 models: per-tile sync drains from 255-saturation, count additive to unpack floor (stock path; §0a-bis disjoint-Dest is the escape), 806-cyc standalone pack absent fusion, no fusion mechanism exists | CRIT-2, EXPE-2, STRA-4 |
| 4 | Name the 2×/12.5× denominators; add that the shipping merge topk pays zero threshold-search cost; retag the "but survives" clause [model] | STRA-2, AUDI-2, CRIT-1 |
| 5 | Re-quote filter economics vs both 6.930 (shipping unfused) and ~5.4 (1.438 fused macro merge, fused-only pending SFPSWAP-index port): 41.4% edge shrinks to ~25%, the 6.7% signed edge inverts | CRIT-4 |
| 6 | Measure the actual §5.1 cell baseline before Gate-2 work; refresh 24–89 µs to post-log-tree numbers; pin the harness (canonical sweep, .so-mtime stamping, replay-STORE arm state) | STRA-3, EXPE-4, CRIT-4 |
| 7 | Justify K≤32/64 against §1.3's own literature (honest version: the *filter*, not radix, is the small-K candidate) and add the explicit N≤65,536 bound the u16 fused key requires | STRA-3, EXPE-3, SUPP-6 |
| 8 | Document the (Cgt,Ceq) engine (§6.1 item 7a) in §4.5 with a RISC-visible ordering primitive (semaphore/sentinel, not bare STALLWAIT) and a Gate-2 deliverable to measure its per-pass cost | IMPL-3, EXPE-1 |
| 9 | Footnote SFPGT VD-range gating (SET_VD writes only VD<8 or VD==16; VD 12-15 silent no-op under default LaneConfig); keep Gate-1 NaN differential tests mandatory | CRIT (VERI-3 challenge) |
| 10 | Flag [DISPUTED] inheritance on 3.855/6.879-derived deltas; retag §4.4's sparse-stream paragraph as repo-survey; attribute exp&31 aliasing + 255 saturation to the WH ISA page (confirmed on BH); add the compound wide-span + subsample-missed-max case to Gate 3 | AUDI-1/2/3, VERI-2/5 |

### 6.3 Post-debate persona verdicts

- **Verifier:** Every load-bearing hardware claim in the dossier traces to ISA pages or silicon tests, but its selector economics must be rewritten around the measured constant menu the debate surfaced — 3 bits/pass max (HistMacro 3.0 cyc/vec), 2.0 cyc/vec single-boundary count, ≥25.1-cyc data-dependent rendezvous, per-tile histogram readback drains, a stock-path-only (not architectural) count/unpack serialization, and no device-side compaction — which makes Gate 4, not the free histogram, the go/no-go pivot.

- **Expert:** The dossier's primitives (SFPGT order, filter floor, histogram telemetry) are certified ground truth, but the swarm's convergent economics — 2 cyc/vec counting floor additive to a 3.938 unpack floor, ≥25-cyc data-dependent rendezvous, 255-saturation forcing per-tile sync drains, and no device compaction making pre-Gate-4 radix passes strictly additive — mean the buildable Gate 2 is SFPGT threshold bisection feeding the existing bitonic emit (the Dr.Top-k shape), with the radix vocabulary, the 2x/12.5x ratios, and the stale 24-89 µs baseline all needing correction before any go/no-go.

- **Auditor:** The dossier's primitive-level claims and epistemic tags survive audit almost everywhere (SFPGT order, histogram divergences, filter costs all trace to primary sources), but its composed economics quote disputed and superseded figures uncaveated (3.855/6.879 [DISPUTED], stale 24-89us, strawman 2x/12.5x denominators), and the debate converges on one structural correction: the exact selector is threshold-bisection/quickselect economics with an uncosted extraction stage — not GPU-shaped radix digits — so the gates should be re-scoped around the filter+count+bitonic hybrid with a measured in-scope baseline before any build.

- **Critic:** The compare/filter primitives are certified real, but the selector's economics degenerate to uncosted quickselect on a hostile ISA, every quoted competitor number is stale or same-day-volatile, and the null hypothesis — the just-improved bitonic paths — remains unbeaten in every in-scope cell; treat the dossier as a primitive catalog, not a feasibility case.

- **Supporter:** The primitives (SFPGT order, packer filter, histogram telemetry) survive every attack and a Dr.Top-k-shaped subset — free T>=0 filter, byte-count bounds, existing bitonic finish, histogram demoted to router — remains a real, defensible win; but the exact-radix framing is dead (my own HistNibble lifeline withdrawn: the safe multi-bin histogram is a measured loser), the composed cycle models need serialisation and per-tile-sync re-pricing, and the compressed-stream consumer (plausibly a writer-RISC decoder) is the true load-bearing gate.

- **Implementer:** The primitives are certified but the economics are hostile: exact selection on BH degenerates to serialized ~2-bit threshold bisection (~8-10 cyc/vec per rescan once the unpack-serialization tax is charged) with a concrete but uncharged (Cgt,Ceq) engine (Dst-mapped RISC readback), and no win region exists before Gate-4 device compaction — so re-scope Gate 2 to a bisection correctness oracle, price the standalone 806-cycle pack as default, and treat Gate 4, not Gate 5, as the load-bearing go/no-go.

- **Random Stranger:** The dossier's primitives are real and honestly tagged, but the swarm converges: BH has no economical multi-bin counting (2 cyc/vec single-threshold floor, additive to the 3.938 unpack floor, 25-cycle rendezvous per decision), so the 'radix select' is actually quickselect with a free-but-lossy predictor — re-scope Gate 2 to filter+count+existing-bitonic, make consumer-side compaction (Gate 4) the load-bearing gate, and measure the real competitor (replay-step-ON ttnn.topk at the scoped K<=64 cell) before writing any selector code.


### 6.4 Findings register

Consensus labels: Confirmed ≥4/7 confirm votes. Priority = 0.4·severity + 0.2·confidence
+ 0.4·consensus. Debate lines record challenges that survived with counter-evidence.
Full debate transcripts: workflow `wf_1bd5caca-e12` in the session transcript directory.

#### CRITICAL and HIGH (full entries)

**IMPL-1 · CRITICAL · confidence HIGH · 6/7 confirm** — No exact multi-bin counting exists: 'exact radix digits' degenerates to ~2-bit bisection, 8 (bf16) to ~16 (fp32) full-data rescans

*Source persona: Implementer · Location: `RADIX_BUCKET_GPU.md:233-236, 366-369; ground truth: SFPGT.md:22-46, SFPIADD.md:19-44, WormholeB0/.../LReg.md:10`*

> **Evidence:** Doc line 235 frames the exponent histogram as 'precisely pass 1 of an MSD radix select with an 8-bit first digit', and Gate 2 (line 366) requires 'exact radix digits'. On BH SFPU the only exact counting sequence is compare-to-mask (SFPGT SET_VD writes -1/0, SFPGT.md:28-30) + SFPIADD accumulate — one accumulator LREG per bin boundary. With 8 general LREGs (LReg.md:10) minus data reg and threshold regs, at most ~3 thresholds fit per pass = ~2 bits of digit per full-data rescan. An exact 8-bit (256-bin) digit pass would need ~85-128 rescans; nobody builds that. There is no scatter, no indexed register file, no per-lane binning anywhere in the BlackholeA0 ISA directory. So Gate 2's real shape is radix-2/4 threshold bisection over sign-magnitude keys: bf16 (16-bit key) ~8 rescan passes worst case, fp32 (32-bit key) ~16 — each pass a full unpack+scan of all N at ~4 cyc/vec (dual count; single count measured 1.998 cyc/vec, SORTING.md:1045). The GPU family's per-pass information rate (8-12 bits, one pass to 256/4096-way partition via atomics) simply does not transfer.
>
> **Recommendation:** Rewrite section 4.2/5 to state the exact-path digit width achievable on this hardware (~2 bits/pass, LREG-bound) and recompute the pass-count budget for Gate 2 from that number; stop describing the sampled packer histogram as 'pass 1' of anything exact.

> **Debate (Verifier, partial):** Downgrade CRITICAL→HIGH and correct the constant: the exact-path digit width achievable is up to ~3 bits/pass via the measured HistMacro+HistSum macro histogram (1.00 cyc/bit, issue-port-bound), not ~2-bit LReg-bound bisection; the '85-128 rescans for an 8-bit digit' framing is a strawman since MSD recursion needs ~⌈bits/3⌉ levels. The recommendation to stop calling the sampled packer histogram 'pass 1 of an exact …

> **Debate (Expert, partial):** Conclusion confirmed — exact multi-bin digit counting does not transfer and the exact arm degenerates to ~2-3-bit bisection with full-data rescans. The LReg arithmetic is slightly conservative (constants can be banked in LReg[11..14]) and severity is better calibrated at HIGH than CRITICAL since the doc's invariant language permits the correct implementation shape; the recomputed pass-count budget recommendation …

> **Debate (Supporter, partial):** Confirm at CRITICAL — the conclusion (radix-256 digit passes are economically impossible; Gate 2's real shape is narrow bisection) is unaffected; the achievable digit width is ~2-3 bits/pass rather than exactly 2 once SFPCONFIG constant registers hold thresholds.


**IMPL-2 · CRITICAL · confidence HIGH · 6/7 confirm** — Without device compaction, Gate 2's emit must fall back to full bitonic sort, making all radix passes strictly additive — no win region exists pre-Gate-4, miscalibrating the section-5 plan ordering

*Source persona: Implementer · Location: `RADIX_BUCKET_GPU.md:295-303, 331-338, 354-359, 379-383; ground truth: SORTING.md:1050-1051, 1046, 53-55; ckernel_sfpu_topk.h:648-960`*

> **Evidence:** Once T* is found with Cgt<=K<=Cgt+Ceq, emitting 'all >T and exactly K-Cgt equals' with indices requires gathering survivors scattered across N/32 vectors. No gather/compaction instruction exists; the sparse PACR stream has no device consumer (doc admits, :295-303). The only in-tree emit is saturate-losers-then-bitonic-sort — and bitonic cost is data-independent (ckernel_sfpu_topk.h's phase/step structure never branches on values), so filtering saves zero sort cycles. Arithmetic from the doc's own sources: bitonic baseline = topk_local_sort 76.195 cyc/vec (SORTING.md:1051) + merge cascade (~2.8-10.6 cyc/vec, :1046-1049) ~= 90k cycles at N=32768; Gate 2 exact path = 8-16 rescans x ~4 cyc/vec x 1024 vec (32-65k cycles) + the SAME ~78k-cycle sort for emit = strictly worse everywhere in the initial scope (:354-359). The verdict (:331-338) lists 'dense extraction' as merely missing and orders Gate 5 A/B after Gate 4, but never states the decisive fact: Gates 2-3 cannot produce a winnable candidate, so the whole-op A/B is only meaningful after Gate 4 succeeds.
>
> **Recommendation:** Add to section 5: 'the exact reference (Gate 2) is a correctness oracle only — it is provably slower than the bitonic path on all cells until Gate 4 delivers device-side compaction or count-guided tile skipping; Gate 4 is the load-bearing gate, not Gate 5.'

> **Debate (Critic, partial):** Confirm the substance (Gate 4 is the load-bearing gate; exact reference is additive-cost everywhere pre-compaction) at MEDIUM severity as a verdict-wording fix, not a plan-invalidating flaw.

> **Debate (Supporter, partial):** Confirm the conclusion that Gate 4 (a compressed-stream consumer) is load-bearing and must precede a meaningful Gate 5 A/B; dispute 'strictly worse everywhere / no win region exists pre-Gate-4' and the CRITICAL severity — the measured 1.10x cascade composition and the writer-RISC decode option mean the win region is contingent on a bounded software deliverable, not architecturally absent. Downgrade to HIGH with the …

> **Debate (Random Stranger, partial):** Keep the recommendation ('Gate 4 is the load-bearing gate, not Gate 5') but reword the mechanism: without a device consumer for the compressed stream, the compaction that already works at emission has no downstream, so the only closable path is saturate+full-bitonic — which is where the strictly-additive arithmetic applies.


**CRIT-1 · HIGH · confidence HIGH · 7/7 confirm** — Histogram's value is priced against a strawman; measured marginal value over software predictors is ~800-2,900 cycles per 32k row

*Source persona: Critic · Location: `RADIX_BUCKET_GPU.md:255-277, 324; SORTING.md:1278-1286, 1510-1525`*

> **Evidence:** The doc's economics for the histogram inherit SORTING.md's '194x cheaper' framing, which compares 128 cycles against a 24,876-cycle 12-bit binary search (SORTING.md:1517) that no competitor uses — the shipping bitonic path needs no threshold search at all (SORTING.md:1294-1296: 'At N=256, do not threshold — sort'; crossover N~4000-8000 unmeasured). SORTING.md's own threshold-search table (1282-1284) shows 'prior + explicit verify (1 pass)' at 2,073 cycles and a fused per-token prior at ~25 cycles, vs the histogram total of 1,267 (1515) and software HistMacro at 4,211 (1516). So the hardware histogram buys at most ~2.9k cycles (~2 µs) per 32k row over pure-SFPU alternatives — and only when a pack happens anyway (a dedicated pack pass costs 806 cycles, SORTING.md:1523). The doc never runs this predictor-vs-predictor comparison; it lists only the binary-search and HistMacro columns (RADIX_BUCKET_GPU.md:272-277) and asserts the piggyback fusion as [inference] (:324).
>
> **Recommendation:** Add a predictor-cost table comparing the packer histogram against (a) no predictor + bitonic, (b) per-token prior + verify, (c) HistMacro — all charged with their readback/sync overhead. If the histogram's net edge over (b) is under ~1k cycles/row, drop it from the critical path and demote it to Gate-3-optional telemetry, which shrinks the pipeline-complexity bill dramatically.


**CRIT-2 · HIGH · confidence HIGH · 7/7 confirm** — Initial scope 'long rows' is self-defeating: 255-saturating counters force either wrong prefix sums or per-tile tensix_sync pipeline drains omitted from every cycle model

*Source persona: Critic · Location: `RADIX_BUCKET_GPU.md:263-265, 352-358; WormholeB0 ExponentHistogram.md functional model; pack_exp_histogram_test.cpp:166,243,261`*

> **Evidence:** The counters are uint8 with a !=255 saturation guard (ExponentHistogram.md functional model, confirmed on BH per SORTING.md:1505-1507). A 32-tile row contributes 4,096 sampled increments across 32 aliased bins; any realistic logit distribution concentrates in a few binades, so the boundary bin — the one the LOCATE step needs — saturates precisely on long rows and clustered data, the doc's own §3 worst case (:171-178). The only escape is CLREXPHIST + modes-6/7 readback per tile, but every readback in the working test requires a full tensix_sync() before regfile reads (pack_exp_histogram_test.cpp:166, 243, 261) — a pipeline drain per tile that the 128-cycle model excludes (the doc admits readback/sync were 'not in the timed loop', :272-277, but never states the saturation-vs-sync dilemma). The initial scope (:354, 'long rows') selects exactly the regime where this either/or bites hardest; Gate 3 lists 'saturation' as a test input (:373) but no gate prices the per-tile sync alternative.
>
> **Recommendation:** Amend §4.3 and §5.1 to state the dilemma explicitly and measure one end-to-end row: 32 tiles with per-tile CLREXPHIST + mode-6/7 readback + tensix_sync inside a streaming pack. If per-tile readback costs >~30 cycles/tile, the histogram's 128-cycle search model is off by an order of magnitude and the initial scope must move to short-row batches or drop the predictor.


**CRIT-3 · HIGH · confidence HIGH · 7/7 confirm** — The exact-selector hole is architectural, not incidental: every refinement pass is a data-dependent full pass + RISC rendezvous, and no winning on-device count primitive exists

*Source persona: Critic · Location: `RADIX_BUCKET_GPU.md:279-286, 331-350; SORTING.md:1266-1276, 1288-1296, 1367-1371`*

> **Evidence:** The doc correctly states the sampled positional histogram 'does not guarantee a bracket' and cannot bound refinement (:285-286), so exactness rests entirely on the unbuilt (Cgt,Ceq) machinery (:344-350). Ground truth shows every ingredient of that machinery is expensive on Tensix: an explicit verify pass costs 2,073 cycles at N=32768 (SORTING.md:1283); both count/reduce offloads LOSE (packer L1-accumulate 4-10x bottleneck, FPU reduce needs MOVD2A/B + fences and misreads SFPGT's 0xFFFFFFFF as -(2^31-1) in sign-magnitude Dst — SORTING.md:1266-1271); each inter-pass RISC rendezvous has a >=25.1-cycle floor and SORTING.md:1369-1371 flags data-dependence as 'an architectural disadvantage the cycle counts alone hide' versus the oblivious MOP/replay bitonic schedule. The verdict paragraph (:331-338) admits the pieces are missing but does not acknowledge that the missing 80% is the part where TT's architecture is weakest relative to GPUs — the §1 pedigree (atomics, grid sync, compaction) maps onto exactly the primitives BH lacks or loses on (:319-326, all [inference]).
>
> **Recommendation:** Recalibrate the verdict from 'suitable for a research prototype' to 'compare/filter primitives certified; selector feasibility unknown pending a measured Gate-2 refinement loop.' Require Gate 2 to report measured refinement-pass cost (verify + rendezvous) before any claim that the family transfers, since that number — not the histogram — decides feasibility.

> **Debate (Verifier, partial):** Keep the recalibrated verdict ('selector feasibility unknown pending a measured Gate-2 refinement loop') — that is right. Split the evidence: rendezvous floor + no compaction = architectural; count-pass additivity = stock-path artifact with a measured (unbuilt) split-Dest escape at ~1.26-1.5 cyc/vec.


**EXPE-1 · HIGH · confidence HIGH · 7/7 confirm** — Exact (Cgt,Ceq) stage has no execution home, and both candidate engines carry measured costs the doc never charges

*Source persona: Expert · Location: `RADIX_BUCKET_GPU.md:340-350; SORTING.md:1217,1220,1638-1640,1692-1695,1720`*

> **Evidence:** The doc mandates the Cgt/Ceq invariant before emission (RADIX:340-350) but 'Cgt'/'Ceq' appear nowhere in SORTING.md (grep: zero hits) — no primitive was ever measured for it. The two candidate engines are both compromised: (a) SFPU counting has a 2.0 cyc/vector architectural floor (CountD1 1.997, SORTING.md:1217,1256) AND the SFPU serializes against unpack_to_dest because math and unpack share the Dest register file (SORTING.md:1638-1640, 'that time ADDS to the unpacker'), so an exact count pass at N=32768 costs ~2048 cycles ON TOP of the 3.938 cyc/vec unpack floor — comparable to the entire 1,267-cycle model at RADIX:277; each data-dependent decision (refine vs emit) is a >=25.1 cyc rendezvous (PassSync, SORTING.md:1220). (b) PackerTileSize is a byte count with placeholder datums, not a survivor count (RADIX:299-301, doc's own admission), and can never yield Ceq (equality count) at all, since MIN_THRESHOLD_RELU is a >=T filter and ties on a negative threshold are zeroed (RADIX:315). Ceq requires a second SFPU compare pass, again additive on the unpacker. Also SFPSTORE has no compress-store and no lane-mask-to-scalar readback (SORTING.md:1720), so reducing 32 lane predicates to the scalar the refinement loop branches on needs a SFPTRANSP tree + store + RISC read of Dest — none of it costed.
>
> **Recommendation:** Add to Gate 2 an explicit deliverable that names the engine for (Cgt,Ceq) and prices the full loop — count pass (charged additive to the unpack floor per SORTING.md:1638), cross-lane reduce, Dest-to-RISC scalar readback, and the per-round >=25 cyc rendezvous — before the go/no-go. The section-5 verdict should say the exact stage is architecturally expensive, not merely unbuilt.

> **Debate (Verifier, partial):** Confirm the finding's deliverable (name and price the (Cgt,Ceq) engine in Gate 2) but scope the framing: the count is additive on the STOCK LLK path; §0a-bis split-Dest concurrency is the doc's own measured escape hatch, so the section-5 wording should be 'expensive on the stock path, with a measured but unbuilt concurrency escape', not 'architecturally expensive'.

> **Debate (Supporter, partial):** Confirm the finding's core (no measured (Cgt,Ceq) engine; SFPU count is additive on the unpack floor; lane-mask-to-scalar readback uncosted — though IMPL-3's Dst.md:103 mapping closes the readback mechanism). Amend the recommendation: Gate 2 should FIRST consider replacing the (Cgt,Ceq) invariant with 'survivor-superset + capacity bound + final-sort tie resolution', which eliminates the Ceq pass entirely for …

> **Debate (Implementer, partial):** Keep the finding at HIGH for its true content — the doc never charges the serialization-additive count cost or the per-round rendezvous — but replace 'no execution home / none of it costed' with the concrete IMPL-3 sequence and its ~tens-of-cycles-per-pass constant; the blocker is economics (additive rescans), not mechanism.


**EXPE-2 · HIGH · confidence HIGH · 7/7 confirm** — Histogram readback is a per-tile cross-thread RISC<->Tensix rendezvous that drains the pipeline; the 128-cycle model counts only instruction issues

*Source persona: Expert · Location: `RADIX_BUCKET_GPU.md:272-277; tt_metal/tt-llk/tests/sources/pack_exp_histogram_test.cpp:115-121,164-180,242-262; SORTING.md:1506-1511`*

> **Evidence:** The real readback path (pack_exp_histogram_test.cpp:164-180) is SETDMAREG modes 6/7 into Tensix GPRs followed by tensix_sync() and regfile[] reads on the issuing TRISC — tensix_sync waits for the RISC to catch up with the whole Tensix instruction stream (test:240-243), i.e. it drains in-flight packs, so the 'zero cost' histogram (RADIX:255) holds only while nobody reads it mid-stream. Per-tile CLREXPHIST+readback is MANDATORY, not optional: 8-bit counters saturate at 255 and the 1-in-8 sample yields 128 increments/tile, so a concentrated row saturates a bin within 2 tiles ('CLREXPHIST is required between tiles', SORTING.md:1507); a 32-tile row therefore pays 32 sync-drains. CLREXPHIST must additionally be issued from the MATH thread ordered by the dest semaphore (SORTING.md A6:515; test:115-121, 'CLREXPHIST is a MATH-resource instruction'), while the T>=0 threshold consumer is packer config (PACK thread) and the signed-arm consumer is an SFPU LREG (MATH thread) — a three-thread choreography per refinement round the doc never maps onto the TRISC split. tensix_sync appears nowhere in SORTING.md as a measured cost (grep: zero hits). The doc honestly flags readback/sync as 'not in the timed loop' (RADIX:276-277), so this is a mechanism gap, not a misrepresentation — but the 128-cycle figure structurally cannot survive the per-tile drain.
>
> **Recommendation:** In Gate 3, measure the histogram readback with a pipelined pack actually running (not a quiesced kernel), charging the per-tile tensix_sync drain; specify which TRISC owns readback+prefix-sum and how T crosses to the MATH thread (L1 mailbox + semaphore) for the signed arm.


**IMPL-3 · HIGH · confidence HIGH · 7/7 confirm** — The (Cgt,Ceq)-to-branch mechanism — the audit's #1 open dependency — has a complete known instruction sequence, including RISC-V count readback via memory-mapped Dst

*Source persona: Implementer · Location: `dependency-map.md:35-36 (open dep 1); RADIX_BUCKET_GPU.md:343-350; ground truth: Dst.md:101-115, SFPTRANSP.md (WH shared):38-49, SFPSHFT2.md (BH) SUBVEC_SHFLROR1 mode`*

> **Evidence:** Every step is expressible: per-vector SFPGT(SET_VD)+SFPIADD accumulate; cross-lane fold = 1x SFPTRANSP + 3x SFPIADD (collapses the 4 row-groups; SFPTRANSP.md functional model) then 7x (SFPSHFT2 SUBVEC_SHFLROR1 + SFPIADD) across the 8-lane groups (~18-20 instructions, once per pass, since accumulators persist across all tiles of a pass); SFPSTORE the folded count to a Dst row; STALLWAIT; then TRISC1 reads the scalar directly — 'RISCV T0/T1/T2 have Dst mapped into their address space, starting at address 0xFFBD_8000' (Dst.md:103), with per-thread RISC_DEST_ACCESS_CTRL_SEC format config. No pack round-trip, no tensix_sync per refinement level. This is a feasibility POSITIVE the doc never establishes: the single-core state machine is not an instruction-sequence blocker, and its per-pass overhead is tens of cycles, not hundreds.
>
> **Recommendation:** Document this sequence in section 4.5 as the exact-counting mechanism (closing dependency-map open dep #1), and use its ~4 cyc/vec + ~20-instr fold cost as the per-pass constant in any Gate 2 cost model.

> **Debate (Expert, partial):** Confirm with correction: document the sequence with an explicit RISC-visible ordering primitive (semaphore or polled sentinel) in place of the bare STALLWAIT, and note the one-time RISC_DEST_ACCESS_CTRL_SEC setup. Per-round overhead estimate of tens of cycles remains credible; this closes dependency-map open dep #1 as a mechanism, leaving only the ~2 cyc/vec count cost (additive to unpack per SORTING.md:1638) as the …

> **Debate (Critic, partial):** Confirm the sequence exists and closes dependency #1's MECHANISM; the COST remains open — Gate 2 should measure it, with ~25-100 cyc/pass as the honest prior, not 'tens'.


**STRA-1 · HIGH · confidence HIGH · 6/6 confirm** — Radix framing does not fit the hardware: exact multi-bin digit histograms are measured losers on SFPU, so the 'exact MSD refinement' arm degenerates to pivot-select (quickselect) economics

*Source persona: Random Stranger · Location: `RADIX_BUCKET_GPU.md:233-246, 347-349; SORTING.md:1279-1292, 1266-1270`*

> **Evidence:** Doc :235-236 sells the exponent histogram as 'precisely pass 1 of an MSD radix select with an 8-bit first digit' and :244-245 prescribes 'recurse on mantissa bits (multi-level digits, section 3)'. But GPU radix select is only cheap because pass-2+ digit histograms cost one shared-memory-atomic pass; SFPU has no scatter/atomics, and the doc's own source measured every multi-bin substitute losing: full-width 8-bucket 12-bit software histogram = 20,580 cyc (barely under the 24,876 binary search it was meant to beat, SORTING.md:1285-1286); the nibble histogram 'was measured and loses' once SFPSHFT mod-32 wrap is clamped (SORTING.md:1288-1292); packer L1-accumulate is a 4-10x bottleneck and FPU reduce needs MOVD2A round-trips plus sign-magnitude int32 traps (SORTING.md:1266-1274). The only cheap exact refinement is a threshold count Cgt/Ceq fused into a filter pass (~25 cyc/token prior, 2,073 with explicit verify, SORTING.md:1282-1283) — i.e. one bucket boundary per full unpack-bound pass, which is Floyd-Rivest/quickselect, not radix. When the sampled predictor misses, the exact fallback IS the 24,876-cycle binary search the doc's headline numbers are advertised against.
>
> **Recommendation:** Reframe section 4.2 and Gate 2: the certified-exact arm should be specified as sampled-pivot select with fused count-verify (one threshold per pass), with the radix vocabulary reserved for the single free exponent-predictor pass. State explicitly that per-digit multi-bin counting was measured and rejected (cite SORTING.md's table) so future implementers do not rebuild GPU-shaped pass structure the SFPU cannot execute economically.

> **Debate (Verifier, partial):** Keep the reframe (sampled-pivot select with count-verify as the primary exact arm; radix vocabulary reserved for the predictor) but correct the constants: the measured refinement menu is 1 bit @ 2.0 cyc (CountD1), 3 bits @ 3.0 cyc (HistMacro+HistSum), and the miss-fallback is ~3-4k cycles (HistMacro search), not 24,876.


**STRA-2 · HIGH · confidence HIGH · 6/6 confirm** — The 2x/12.5x figures compare against strawman denominators (software histogram search, binary search), never against the shipping merge-based topk — and whole-op time is overhead-dominated, making SFPU-cycle wins third-order

*Source persona: Random Stranger · Location: `RADIX_BUCKET_GPU.md:324, 272-277; SORTING.md:1515-1524`*

> **Evidence:** SORTING.md:1515-1524 shows the provenance: 2,073 cyc (histogram + dedicated 806-cyc pack + filter + finish) vs 4,211 (HistMacro arm, ~2x) and 25,903 (binary-search arm, ~12.5x). Both denominators are hypothetical threshold-select pipelines; the actual incumbent (merge-based bitonic topk) performs no threshold search at all, so '12.5x' measures how much better this design is than a bad version of itself. A GPU person's first check — 'faster than what, end to end?' — also exposes scale: the composed model is ~2,073 cyc (~1.5 us at BH clocks) while the measured multi-core ttnn.topk at N=32k/K=32 is 171 us and even the best in-tree path sits 13-23x above roofline (memory/topk-sorting-campaign.md:13, 26), i.e. the op is dispatch/data-movement bound and SFPU compare cycles are a small slice. Doc :272-277 honestly flags models-not-measurements and Gate 5 (:379-382) demands whole-op A/B, but :324 still prints 'win drops to 2x/12.5x but survives' in the architecture table without naming the comparator.
>
> **Recommendation:** Amend :324 (and anywhere 2x/12.5x appears) to name the denominators explicitly and add one sentence stating the shipping merge topk pays zero threshold-search cost, so these ratios are internal to the threshold-select pipeline. Add a roofline/overhead note: until the op is within ~2x of data-movement roofline, algorithm-level SFPU savings will not be visible at Gate 5.


**STRA-3 · HIGH · confidence HIGH · 6/6 confirm** — Initial scope (K<=32/64) picks exactly the regime the doc's own cited literature says bitonic/register selection wins, while forbidding the large-K regime where the radix family's advantage lives — and the quoted competitor baseline is already stale

*Source persona: Random Stranger · Location: `RADIX_BUCKET_GPU.md:111-117, 354-358, 30; memory/topk-sorting-campaign.md:13`*

> **Evidence:** Doc :354 scopes v1 to 'K<=32/64 + long rows'. But :111-114 (Shanbhag SIGMOD'18) reports bitonic beats radix select through roughly k=256, and :115-117 (FAISS WarpSelect) is cited as the small-k register-resident champion; :30 notes RadiK's differentiator is 'supports large k'. On GPUs nobody runs radix select at k=32 — that is WarpSelect territory, and TT's in-tree analog (bitonic topk_local_sort, multi-core ttnn.topk gated at k<=64) already occupies it. Meanwhile :356-358 bans competing at K=512/1024/2048 against topk_large_indices's '24-89 us' baseline — but that number predates the log-tree merge commit (8794fbb, after dossier commit 48119db): the path now measures ~15-42 us (k512@65536: 14.96 us; k2048@65536: 41.9 us per campaign memory), roughly halving the window in the one regime where radix-family methods are documented to win.
>
> **Recommendation:** Justify the K<=64 scope against the doc's own section 1.3 (the honest justification is 'the threshold FILTER, not radix, is the candidate win at small K'), or re-scope the radix arm to the large-K cells where the family's literature advantage applies; either way refresh the competitor baseline to the post-tree-merge numbers before Gate 5 target-setting.

> **Debate (Critic, partial):** Any Gate-5 target must name the harness, not just the number: canonical sweep (_canonical_topk_sweep.py), device flock, .so-mtime stamping per cell, and the replay-STORE arm state — otherwise the refreshed baseline drifts as fast as the stale one did.


**VERI-1 · HIGH · confidence HIGH · 7/7 confirm** — Section 4 omits the exact-counting primitive and its measured 2 cyc/vec floor — the cost that dictates the selector's shape

*Source persona: Verifier · Location: `RADIX_BUCKET_GPU.md:231-247,272-277 vs SORTING.md:1207-1292, SFPSHFT.md:44-50`*

> **Evidence:** SFPU has no scatter/atomics; the only exact counting primitive is a predicated count: SFPLOADMACRO(Load+SFPGT) + software SFPIADD, measured at 1.997-2.0 cyc/vector and proven an architectural floor (single shared SFPU issue port across all three threads, SORTING.md:1223-1256; macros cannot host reductions, LReg[16]/macroVD restrictions SORTING.md:938-944). Multi-bin exact digit counting is priced out: an 8-bucket SFPU nibble histogram measures 5.0 cyc/vec UNCLAMPED (SORTING.md:1221), SFPSHFT wraps mod 32 rather than saturating (SFPSHFT.md:44-50) so clamping makes it worse than binary search, SFPIADD wraps needing drains every 15 vectors (SORTING.md:1288-1292); a full-width software histogram costs 20,580 cyc for N=32k vs the doc's own ~1,267-cycle composed budget. RADIX_BUCKET_GPU.md §4.2:244-246 advises 'recurse on mantissa bits (multi-level digits)' without stating that exact multi-bin digit counting is economically infeasible on this ISA — the exact path must be threshold bisection via (Cgt,Ceq) at 2 cyc/vec per candidate plus a >=25.1-cyc data-dependent rendezvous per decision (PassSync, SORTING.md:1220), not radix binning.
>
> **Recommendation:** Add the counting-primitive cost model to §4 (2 cyc/vec predicated count floor, 5 cyc/vec 8-bucket histogram that loses, >=25 cyc data-dependent restart) and reword §4.2's mantissa recursion to 'threshold bisection over mantissa bits or bounded topk_local_sort fallback'; make Gate 2 explicitly require the bisection state machine rather than a literal multi-bin digit pass.


**CRIT-4 · HIGH · confidence MEDIUM · 7/7 confirm** — The null hypothesis moved: the campaign's own 1.438 macro merge inverts the quoted 6.7% signed edge, and the chosen initial-scope cell has no baseline number at all

*Source persona: Critic · Location: `RADIX_BUCKET_GPU.md:310-316, 352-358, 379-382; SORTING.md:26, 77, 283-287, 303-310, 1527`*

> **Evidence:** The doc quotes 41.4%/6.7% filter-vs-merge deltas (:313-314) against _topk_xl_merge_ at 6.879 — but the same campaign beat that primitive 2.844→1.438 cyc/vec with 71/71 correctness (SORTING.md:77, 1527). Under SORTING.md's own measured linear pipeline model L1_TO_L1 = 4.132 + 0.275 + 1.004×issues (:283-284), the improved merge lands near ~5.85 cyc/vec, below the signed SFPU filter's 6.415 (:280) — the 6.7% edge inverts to a loss against the current best in-tree primitive. Separately, §5.1 names topk_large_indices's 24-89 µs at k∈[512,2048] as 'the competitor' (:355-358; SORTING.md:26) while restricting scope to K≤32/64 — a different cell whose actual whole-op baseline (stock multicore topk at K≤64, long rows) is cited nowhere in either document, so Gate 5 (:379-382) has no ex-ante number to beat and the scope's winnability cannot be assessed even on paper. The honest end-to-end model is ~3.0x vs only the local-sort component (SORTING.md:1344-1356), before charging extraction, index reconstruction, final sort, and rendezvous.
>
> **Recommendation:** Re-quote all filter economics against the 1.438 macro merge, not the superseded 6.879 baseline, and measure the stock multicore topk whole-op time for the exact §5.1 cell (BH, bf16, K≤64, long rows) before Gate 2 work begins — if that number is already <2x the threshold-select paper model, stop.

> **Debate (Expert, partial):** Confirm the core finding — the 41.4%/6.7% deltas are quoted against a superseded denominator and the §5.1 cell has no ex-ante baseline. Add the caveat that the 1.438 merge is fused-only pending the unfused SFPSWAP-index port, so the honest framing is 'the 6.7% edge inverts against the campaign's own best fused merge and is at risk against the shipping path once the port lands', with the measured-baseline demand …

> **Debate (Implementer, partial):** Confirm at MEDIUM rather than HIGH: quote filter economics against BOTH 6.930 (shipping-unfused today) and ~5.4 (fused macro merge, pending unfused port), and cite the archived 171 us stock-multicore baseline for the exact 5.1 cell instead of claiming no number exists.


**IMPL-4 · HIGH · confidence MEDIUM · 7/7 confirm** — Data-dependent refinement pass counts break the fixed-stream tri-thread kernel model — a novel unpack/math coordination problem the doc and gates never mention

*Source persona: Implementer · Location: `RADIX_BUCKET_GPU.md:343-350, 366-369; ground truth: ckernel_sfpu_topk.h:648 (_bitonic_topk_phases_steps host params), :963 (_bitonic_topk_merge m_iter)`*

> **Evidence:** The invariant loop ('if false, monotonically refine another digit', doc :347-349) means the number of full-data rescans is decided at runtime by the math thread — but for long rows (N=32k does not fit in DEST's 8-tile fp32 budget) each rescan requires the UNPACK thread to re-stream ~1024 tiles. All existing topk LLK entry points take host-computed iteration controls (i_start_phase/i_end_phase at ckernel_sfpu_topk.h:648, m_iter at :963); no in-tree compute kernel runs a cross-thread data-dependent loop count. Mechanisms exist (RISC-RISC mailboxes, Tensix semaphores), so it is feasible — but it is a genuinely new kernel architecture with its own hang surface (the CB-credit protocol between reader and unpack must also become data-dependent). The escape hatch — host-fixed worst-case pass count (8 for bf16, 16 for fp32) — burns the full worst-case scan budget on every call, further worsening the IMPL-2 economics.
>
> **Recommendation:** Add to Gate 2's definition: choose and prototype the cross-thread pass-count mechanism (mailbox-signaled early exit vs fixed worst-case passes) before writing the selector, and add hang tests for it to Gate 3's list.


**SUPP-1 · HIGH · confidence MEDIUM · 6/7 confirm** — Verdict's 'missing selector' list ignores the exact second stage already in-tree; Gate 2 overshoots for the strongest subset

*Source persona: Supporter · Location: `RADIX_BUCKET_GPU.md:336-338, :366-369 (Gate 2) vs tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h:348-538 and RADIX_BUCKET_GPU.md:244-246`*

> **Evidence:** The verdict says 'The missing pieces are the selector itself: no exact digit-refinement state machine ... exists' (:336-338), and Gate 2 requires building 'exact radix digits, (Cgt,Ceq), progress detection, boundary allocation' before anything else. But ckernel_sfpu_topk.h already implements the full exact bitonic phase machinery (bitonic_topk_ph0_st1_to_1 ... ph3_st4_to_1, STABLE_SORT variants, bitonic_topk_step_N at lines 348-538), and the doc itself concedes topk_local_sort 'exists in-tree and was measured/optimized' (:245-246) with bf16 boundary buckets holding <=2^7 magnitudes so 'a single fallback level suffices' (:246). A Dr.Top-k-shaped subset (doc's own analogy, :85-89) — predict T, filter (measured, SORTING.md:1629), exact-count survivors (measured floor 2.0 cyc/vec, SORTING.md:1207-1218 CountD1=1.997), run existing bitonic sort on survivors if S>=K, else monotonically lower T over 32 exponent bins — needs NO new digit-refinement state machine; exactness reduces to 'all elements > T survive the filter'. Dependency-map open dependency #1 (exact counting cost) is in fact answered by SORTING.md's measured counting floor.
>
> **Recommendation:** Re-scope Gate 2 to the threshold-filter subset: filter + exact survivor count + existing bitonic_topk fallback, with monotone threshold-lowering as the bounded retry. Defer the full radix digit-refinement state machine to a later gate; it is only needed when the survivor set exceeds local-sort capacity, which the exponent-bin retry bound already limits.

> **Debate (Verifier, partial):** Confirm as a Gate-2 correctness-scope simplification (aligned with IMPL-5's direct SFPGT bisection); reject any implication that the filter+existing-bitonic subset is a PERFORMANCE candidate before Gate 4 delivers compaction or count-guided tile skipping.

> **Debate (Auditor, partial):** The re-scoping recommendation (filter + count + existing bitonic fallback as Gate 2, deferring the digit state machine) is sound and converges with IMPL-5/STRA-5, so the finding's recommendation survives. But the evidentiary claim that dependency #1 is 'answered' should be revised to 'the per-vector count primitive is priced; the loop cost (unpack serialization, >=25.1-cyc rendezvous per retry, survivor extraction, …

> **Debate (Critic, partial):** Gate 2 as written is roughly the minimal exact machinery for bf16, not an overshoot; SUPP-1's real contribution is naming that mantissa-level digits are already excluded by :244-246, which is a wording fix, not a re-scope.

> **Debate (Implementer, partial):** Adopt the re-scoped Gate 2 (bisection + count + bitonic fallback as correctness oracle) while explicitly stating it is provably slower than stock bitonic on every cell until Gate 4 lands device-side compaction or count-guided tile skipping.

> **Debate (Random Stranger, partial):** Adopt SUPP-1's Gate-2 re-scope, but state the counting dependency as 'floor measured in isolation, full loop unpriced' — EXPE-1's additive-to-unpack charge and rendezvous cost must be in the Gate-2 budget.


#### MEDIUM (condensed entries)

**AUDI-1 · 7/7** — Disputed source figures (3.855 floor, 6.879 merge) presented as settled — 4.6%/41.4%/6.7% deltas inherit unflagged dispute
`RADIX_BUCKET_GPU.md:291-292,314 vs SORTING.md:159,198,281 and SORTING.md:40-46`
> RADIX_BUCKET_GPU.md:291-292 states 'relucomp arm runs at 4.034 cyc/vector vs a 3.855 unpack-bound floor, a 4.6% delta' and :314 states the 41.4%/6.7% deltas, all as [local finding] with no caveat. SORTING.md:159 tags 3.855 '[DISPUTED — being re-measured 2026-08-16 ... elsewhere 3.938 (§0a, B3) and 4.175 (§0a headline)]'; SORTING.md:281 tags 6.879 '[DISPUTED ... §0a/B2 quote 6.930 for the same arm]'. SORTING.md:28 claims every DISPUTED tag is adjudicated in HANDOFF.md, but grep of HANDOFF.md finds no 3.855/6.879/41.4 — these rows are not adjudicated. If the floor is 3.938 the '4.6% delta' …
> **Fix:** Add the [DISPUTED]/re-measurement caveat to the 4.034-vs-3.855 and 41.4%/6.7% sentences (or quote the post-adjudication numbers once the canonical sweep lands), and rename '3.855 unpack-bound floor' to 'stock LLK-path floor (handshake-bound, under re-measurement)'. Directional conclusions survive; the doc's own every-claim-tagged standard requires …


**AUDI-2 · 7/7** — Internal epistemic contradiction: §4.5 tags the modeled 806-cycle / '2×/12.5×' win as [local finding] after §4.3 declared the same composed table 'models, not measurements'
`RADIX_BUCKET_GPU.md:324 vs :272-277; SORTING.md:1510-1524`
> RADIX_BUCKET_GPU.md:274-277 correctly states the 128/3,072/24,876/1,267-cycle figures 'are component-based models, not measurements of a working Top-K operator'. Yet :324 says 'a dedicated pack pass costs 806 cycles for N=32k (win drops to 2×/12.5× but survives)' tagged '[local finding] for costs'. 806 is not an independent measurement: it is 32 tiles × the measured 25.175 cyc/tile PACK_ISOLATE = 805.6 (SORTING.md:1482,1523), and the 2×/12.5× ratios (2,073 vs 4,211 / 25,903, SORTING.md:1524) are ratios of two modeled totals from the very table §4.3 demoted. 'But survives' is therefore a …
> **Fix:** Retag :324 as '[local finding] for the 25.175 cyc/tile pack cost; [inference/model] for the 806-cycle composition and the 2×/12.5× win', matching the standard §4.3 already set.

> **Debate (Random Stranger, partial):** Narrow the finding: the tag at :324 already splits correctly for 806 and fusion; only the '2x/12.5x but survives' clause needs an explicit [model] retag (and per STRA-2, its denominators named). Severity LOW rather than MEDIUM.


**CRIT-5 · 7/7** — Concrete hang/corruption vectors in the sketched operator: variable-byte sparse stream vs fixed-page CB credits, and cross-thread/cross-op histogram state discipline
`RADIX_BUCKET_GPU.md:295-303, 324; dataflow_buffer.h:113-118; test_pack_compress_int32.py:137; SORTING.md:1501-1508; pack_exp_histogram_test.cpp:115-123`
> (a) Sparse stream: CB/DFB flow control is denominated in fixed-size entries/pages (dataflow_buffer.h:113-118 get_entry_size; pages_reservable_at_back/pages_available_at_front at :193-194), but the compressed PACR stream's length is a data-dependent byte count that is not a survivor count (placeholder datums, RADIX_BUCKET_GPU.md:299-301); a consumer waiting on page credits for variable-byte payloads either deadlocks (CWFW) or reads stale bytes as data — and the only existing decoder is host-side Python (decode_compressed32, test_pack_compress_int32.py:137). (b) Histogram state: …
> **Fix:** Add to Gate 4/6: the sparse stream may only enter a CB behind an explicit (offset,length) header written to a fixed-size metadata page (restoring page-credit semantics), and any producer-fused histogram must specify which thread issues CLREXPHIST, its dest-semaphore ordering, and how counts cross the program boundary (L1 scratch + semaphore, never …


**CRIT-6 · 7/7** — For signed data the design loses both of its hardware assists simultaneously, and the 'count positives first' sign digit has no costed mechanism
`RADIX_BUCKET_GPU.md:237-240, 269-270, 309-316; SORTING.md:299-317, 1505-1508`
> The histogram is sign-blind (ranks |x|, :269-270; SORTING.md:1508) and the packer threshold filter is UB for T<0 (measured as ceil-pow2 of |T|, :309-315; SORTING.md:312-317), so on signed logits — the stated production motivation (MoE routing, vocab sampling, SORTING.md:235-236) — both packer assists drop out at once: the filter falls back to the 2-issue SFPU floor (6.415 cyc/vec, SORTING.md:280) and the predictor requires the sign digit resolved first. The doc's mechanism for that ('count positives, and only if k exceeds that count does the negative half matter', :239-240) is tagged …
> **Fix:** Either restrict the design's claimed applicability to provably non-negative inputs (post-softmax probabilities) in §5.1, or add a costed sign-count pass to the §4.2 composition and re-run the payoff model; as written the signed arm should be scored as predictor-free threshold-select, i.e., the SORTING.md ~25 cyc/vec model with zero histogram …

> **Debate (Supporter, partial):** Confirm the finding's severity for the spill case and the demand to cost the sign digit; dispute the specific claim that no cheap positive-count mechanism exists. The recommendation should offer a third option: route on a free T=0 filter-pass byte count (upper bound), falling back to the …


**EXPE-3 · 7/7** — Index survival forces the fp32 fused [bf16|u16] carrier, which erases BF16's unpack advantage and caps rows at 65536 — in tension with the 5.1 'BF16 + long rows' scope
`RADIX_BUCKET_GPU.md:293,304-308,352-355; SORTING.md:1638-1644; SFPSWAP.md:46-49`
> The only measured filter+compaction carrier that keeps indices attached through a filter pass is the dense fp32 fused key [bf16 value | u16 index] (RADIX:293; SORTING.md relu32 arm). That has two consequences the doc does not draw: (1) fused keys are 32-bit datums, so the unpacker floor is the 32-bit 3.938 cyc/vec that 'no 32-bit-fused Top-K can beat' (SORTING.md:1642-1644) — choosing BF16 in 5.1 buys no bandwidth on the radix path (bf16-only unpack would be ~half the bytes; inference); (2) u16 indices address at most 65536 positions, while 'long rows' is left unbounded and the competitor …
> **Fix:** In 5.1, bound 'long rows' explicitly (N_min from the unmeasured N~4000-8000 crossover, SORTING.md:1294-1296; N_max 65536 for the u16 carrier) and note that the fused-key path pays the 32-bit unpack floor regardless of BF16 input — or gate a chunked u16-local + chunk-id reconstruction design in Gate 2.


**EXPE-4 · 7/7** — The 5.1 competitor is misframed: the named 24-89 us baseline belongs to the excluded K regime, while the in-scope bitonic path just got 1.15-1.20x faster in this same working tree
`RADIX_BUCKET_GPU.md:355-359; tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h:66-75; SORTING.md:1294-1296,1659-1663`
> 5.1 scopes to K<=32/64 yet the only quoted competitor number, topk_large_indices 24-89 us, exists at K=512/1024/2048 (RADIX:356-359) — no baseline is quoted for the actual in-scope cell (stock single/multicore ttnn.topk at K<=32/64, long N). That competitor is a moving target: the uncommitted TOPK_REPLAY_STEP_STORE default-ON in ckernel_sfpu_topk.h (lines 66-75) measures ttnn.topk single-core 1.154x-1.202x across N in [4096,131072], k in [8,512] — exactly the scoped cell. The bitonic fallback the doc leans on (`topk_local_sort`, RADIX:245-246) is the same machinery: _bitonic_topk_phases_steps …
> **Fix:** Gate 5 already requires measuring stock Top-K; pull that forward: quote a Tracy whole-op baseline of ttnn.topk (replay-step change ON) at the exact 5.1 cells before committing to the scope, and either name that number in 5.1 or drop the out-of-scope 24-89 us figure from the scope paragraph.


**IMPL-5 · 7/7** — SFPGT's native sign-magnitude total order makes section 4.2's sign-digit/exponent-reversal decomposition unnecessary for the exact path
`RADIX_BUCKET_GPU.md:238-241; ground truth: SFPGT.md:3, 55-66`
> Doc :238-241 prescribes handling the sign as a 1-bit digit above the exponent, 'where exponent order reverses' for negatives. But SFPGT compares full 32-bit keys in exactly the target total order (-NaN < -Inf < ... < +Inf < +NaN; SFPGT.md:55-66 SignMagIsSmaller), so a Gate 2 implementation bisects raw-bit thresholds directly: negative-half reversal, sign handling, and NaN ordering all fall out of the comparator with one uniform code path, and the T<0 packer-UB hole (:310-312) never arises during counting because counting never uses the packer. SFPEXEXP-based digit extraction is only needed if …
> **Fix:** Amend section 4.2: for the exact reference, replace digit decomposition with direct sign-magnitude threshold bisection via SFPGT; keep the exponent-digit framing only for the sampled-predictor discussion in 4.3.


**IMPL-6 · 7/7** — Index ride-along is free during counting but inherits full bitonic cost and width limits at emit; fp32 doubles both pass count and DEST pressure
`RADIX_BUCKET_GPU.md:304-308, 354-359; ground truth: ckernel_sfpu_topk.h:237-250 (dst_indices_offset=128, paired index loads), WormholeB0/.../LReg.md:15`
> During (Cgt,Ceq) passes indices never move — position is implicit, and LReg[15] (lane i holds 2i, LReg.md:15) plus a per-vector offset materializes indices on demand at zero memory cost. But at emit, the only index-carrying machinery is the bitonic path's paired index tiles at DEST offset 128 with dual-SFPSWAP tracking (ckernel_sfpu_topk.h:237-250, 991-995), so the radix path's output stage inherits the full bitonic cost (reinforcing IMPL-2) and, for the fused-key shortcut, the u16-index/stable=false limits the doc already concedes (:304-308). For fp32+u32 indices the initial-scope arithmetic …
> **Fix:** State in 5.1 that fp32 is excluded because the exact path's pass count and DEST pressure both double, not just for contract simplicity; note LReg[15]-based index materialization as the free index mechanism for counting passes.


**STRA-4 · 6/6** — 'Histogram is free' depends on producer-op fusion that has no existing mechanism — and this project already hit the fused-only trap once
`RADIX_BUCKET_GPU.md:255-257, 324; memory/topk-sorting-campaign.md:15, 19`
> Doc :255-257 reports zero marginal cost 'during a pack that happens anyway', and :324 asserts the count pass 'can piggyback on whatever op produced the logits (e.g. the matmul's own pack)' — tagged [inference]. Three unpriced obstacles: (a) ttnn.topk is a standalone program; TTNN has no producer-epilogue fusion pass to enable histogram collection inside an upstream op's pack, and per-core counter state must survive across program launches with matching core-to-row sharding — none of this is designed or measured. (b) Inside a standalone topk the input arrives through the UNPACKER; …
> **Fix:** Demote fused-histogram from a section-4.5 table assumption to an explicit gated dependency ('Gate 0: demonstrate histogram counter readback across a producer-op program boundary with matching sharding'), and make the standalone 806-cycle pack pass the default cost in every composed model.


**SUPP-4 · 7/7** — The doc undersells its own end-to-end evidence: the filter arm was measured same-kernel end-to-end, not merely modeled
`RADIX_BUCKET_GPU.md:272-277, :313-315 vs SORTING.md:1617-1659 (topk_pipeline_perf.cpp, 'MEASURED END TO END')`
> The doc groups everything under 'component-based models, not measurements of a working Top-K operator' (:272-277) and bars the 41.4%/6.7% deltas from routing (:313-315). But SORTING.md:1617-1656 documents a pipelined multi-tile kernel (unpack -> math -> compressed pack) with _topk_xl_merge_ as an in-kernel comparison arm: filter+threshold+compaction costs 0.097 cyc/vec (2.4%) over the streaming base, and the 1.66-1.68x head-to-head vs the shipping merge primitive is 'real, measured side by side, in one kernel' — this is a per-pass pipeline measurement, a categorically stronger evidence class …
> **Fix:** Split §4.3's honesty paragraph into two evidence tiers: (a) measured pipeline facts — filter-pass cost, PACK concurrency, unpack floor, SFPU serialisation (SORTING.md end-to-end table); (b) composed models — cascade totals, threshold-search 128 cycles. Only tier (b) needs the 'not a working operator' disclaimer; the current blanket phrasing makes …


**VERI-2 · 7/7** — exp&31 alias-window resolution depends on the subsampled mode-9 max exponent — the two caveats interact but are listed separately
`RADIX_BUCKET_GPU.md:263-266 vs SORTING.md:1499-1508, WormholeB0 ExponentHistogram.md:23-30`
> The 32-bin histogram bins on the LOW 5 bits of the 8-bit exponent (BinNumber = Exponent & 31, ExponentHistogram.md:23-24), so bins are a monotone value partition only within one 32-exponent window; data spanning >32 octaves aliases non-adjacent magnitude ranges into the same bin. The natural disambiguator is the mode-9 max exponent, but that is itself subsampled on BH — a single exp-132 outlier among 1023 exp-127 datums is MISSED (SORTING.md:1499-1500). So the predictor can select the wrong 32-exponent window entirely, a distinct failure mode from within-window bracket misses. The doc lists …
> **Fix:** Add the compound case to Gate 3's adversarial inputs: exponent span >32 with the true max exponent placed only at non-sampled positions (p mod 64 >= 8), verifying the exact (Cgt,Ceq) path recovers from a wrong-window prediction.


**SUPP-3 · 6/7** — All three claimed hardware gifts verify against ground truth, including the BH-only novelty of SFPGT — the design's foundation is solid, not speculative
`RADIX_BUCKET_GPU.md:219-229, :255-257, :288-293 vs tt-isa-documentation/BlackholeA0/.../SFPGT.md:3,7-8,55-66; SORTING.md:493-498 (A2), :1482-1486, :276-278`
> (1) SFPGT.md:3 states verbatim the total order '-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN' and the functional model SignMagIsSmaller (SFPGT.md:55-66) is exactly sign-magnitude comparison; SFPGT.md:7-8 confirms the instruction is 'new in Blackhole', which independently validates §5.1's Blackhole-only initial scope. (2) MIN_THRESHOLD_RELU using the same order is a silicon finding from a failing test on exactly the 64 negative-NaN datums (SORTING.md:493-498). (3) Histogram cost: 25.175 -> 25.104 cyc/tile with histogram on, CLREXPHIST exactly 1 cycle (SORTING.md:1482-1486). (4) Filter: …
> **Fix:** Treat C3/C4/C6/C12 as settled ground truth in downstream audits; feasibility challenges should target composition and pipeline cost (B2 serialisation, multicore sync), not the primitives themselves.

> **Debate (Critic, partial):** Restate as: gifts 1-3 (SFPGT order, MIN_THRESHOLD_RELU order, histogram zero-marginal-cost) are settled; gift 4 (filter-vs-floor delta) is directionally right but quantitatively under re-measurement — cite it with AUDI-1's caveat.


**STRA-5 · 6/6** — The simpler architecture the evidence actually supports — predictor threshold + free filter + existing exact bitonic topk as second stage (Dr.Top-k pattern) — is under-weighted; Gate 2 mandates radix-digit machinery it does not need
`RADIX_BUCKET_GPU.md:366-369, 85-89, 244-246; SORTING.md:1282-1283`
> Gate 2 (:366-369) requires an 'exact radix digits' reference implementation with digit progress detection before any estimator work. But the doc's own measured components compose into a certified design with zero digit machinery: sampled threshold prior, count fused into the filter pass (~25 cyc, SORTING.md:1282), prior + explicit verify at 2,073 cyc (SORTING.md:1283), then the already-measured in-tree bitonic topk_local_sort (:245-246) as the exact second stage over survivors — with the introselect-style certified fallback (:141-144) being 'rerun stock topk on the whole row', which bounds …
> **Fix:** Restructure the gates so the filter+existing-topk hybrid is Gate 2 (fastest path to a whole-op A/B with the fewest new correctness obligations) and exact radix-digit refinement is a later, conditional gate entered only if the hybrid's p95 survivor count makes the second-stage sort dominant; add a cross-reference to THRESHOLD_SELECT_DESIGN.md.

> **Debate (Implementer, partial):** Confirm the restructuring recommendation, with the added sentence that the hybrid's whole-op win is gated on Gate-4 compaction (or count-guided tile skipping), so its first deliverable is a correctness oracle plus a measured survivor-count distribution, not a perf win.


**SUPP-5 · 5/7** — Histogram-as-router between existing shipped paths is a zero-correctness-risk deployment the gate plan never considers
`RADIX_BUCKET_GPU.md:109-123 (§1.3 dispatch argument), :329-394 (§5 gates lack a router milestone); SORTING.md:1482-1486 (free), :505-508 (A4 readback)`
> The doc's own §1.3 argues (via Shanbhag SIGMOD'18) for 'dispatch by (N,k,bucket mass), not a radix-only replacement' (:113-114), and the histogram is measured free during any existing pack (SORTING.md:1482-1486) with RISC readback available (modes 6/7, A4). Yet §5's gates only monetize the histogram inside the new operator (Gate 3), behind Gates 1-2. Used as a router/threshold-primer for paths that already exist and are exact (stock bitonic top-k, topk_large_indices at 24-89 us, the measured filter pass), every histogram pathology the doc lists — sampling misses, exp&31 aliasing, saturation …
> **Fix:** Add a Gate 0.5 / parallel track: wire histogram telemetry into dispatch or threshold priming for existing exact paths, A/B'd under Gate-5-style whole-op Tracy measurement. It de-risks the histogram plumbing (CLREXPHIST fencing, readback, aggregation) years before the exact selector needs it, and can ship value even if Gates 2-4 never pass.

> **Debate (Verifier, disagree):** Demote from 'Gate 0.5 zero-risk deployment' to 'a routing experiment contingent on STRA-4's Gate-0 fusion demonstration'; correctness-neutrality is real, but the cost and mechanism claims do not survive contact with the fusion gap and readback choreography.

> **Debate (Critic, partial):** Downgrade 'zero-correctness-risk router' to 'threshold-priming telemetry for non-traced paths, contingent on solving the same cross-program readback Gate 0 that STRA-4 identifies' — it is not cheaper than the gates it claims to bypass.


#### LOW — verified positives and polish

| ID | Votes | Finding |
|---|---|---|
| AUDI-3 | 7/7 | [local finding] tag on §4.4 sparse-stream paragraph covers repo-survey claims, not silicon measurements (content verified true) — *Split the §4.4 tag: keep [local finding] for the 4.034/4096→640B measurements, mark the no-device-consumer/fixed-entry-API sentences as [code survey] or [analysis]. No content change needed — the …* |
| AUDI-4 | 7/7 | Cross-document inconsistency resolves in the honest direction: SORTING.md's headline still pairs the modeled 128-cyc/194× search with 38/38+6/6 test evidence that only covers the … — *No change to RADIX_BUCKET_GPU.md. Fix the source: annotate SORTING.md:79's correctness cell as 'primitive tests only; 128-cyc figure is a model' so future documents quoting the headline table do not …* |
| AUDI-5 | 7/7 | SFPGT sign-magnitude ordering claim is asserted without citing the ISA page that proves it (claim verified true) — *Add tt-isa-documentation/BlackholeA0/.../SFPGT.md as a citation at :222 (and to the references list). This strengthens, not weakens, the design: the ordering claim rests on ISA documentation plus an …* |
| EXPE-5 | 7/7 | Verified: the order-theory and filter-primitive claims the design rests on are all correct against ISA and silicon ground truth — *None needed — retain these citations; they are the strongest part of the dossier and correctly separate [local finding] from [inference].* |
| STRA-6 | 6/6 | Verified positive: the load-bearing SFPGT ordering claim is exactly right per the ISA, and the doc's epistemic tagging is trustworthy — *Keep Gate 1's differential NaN/±0 testing mandatory before any routing decision; no change to the SFPGT claims themselves.* |
| VERI-3 | 7/7 | C3 CONFIRMED: SFPGT natively implements the sign-magnitude total order the radix-on-IEEE-bits map produces — *No change needed; optionally cite SFPGT.md's functional-model lines directly in §4.1 so the claim is doc-anchored rather than only SORTING.md-anchored.* |
| VERI-4 | 7/7 | C15 CONFIRMED with a scope caveat: 2-issue floor holds for value-preserving filters; a 1-issue destructive mask map exists — *In §4.4, qualify 'proven to be the ISA floor' with 'for a bit-exact value-preserving filter' — a 1-issue mask-tile map exists if a later pass can consume mask+value separately.* |
| VERI-5 | 7/7 | C6-C9 CONFIRMED end-to-end; note the exp&31 aliasing and 255-saturation are documented WH ISA behavior, not BH discoveries — *Attribute aliasing/saturation to the WH ISA page (confirmed on BH) rather than presenting them as silicon findings; keeps the genuinely undocumented BH divergences (sampling, WhichPackers, fencing) …* |
| VERI-6 | 7/7 | Digit-extraction toolkit CONFIRMED feasible and near-free; bounded fallback exists in-tree — *No change needed; §4.2 could cite SFPEXEXP/SFPEXMAN by name to anchor 'exponent field directly addressable' (:325) to the ISA.* |
| SUPP-6 | 5/7 | The u16 fused-key caution is non-binding inside the declared v1 scope — declared limitation already matches declared scope — *State in §5.1 that the u16 fused key satisfies the v1 index contract for rows <= 64K and record it as a Gate-1 already-passed item, reserving the UINT32/global-index work for the scope expansion …* |

#### Discarded

**SUPP-2 (0/7 — unanimous, including self-withdrawal).** Claimed SORTING.md's HistNibble
(5.0 cyc/vec SFPU exponent histogram) means "an exact §3 step-1 histogram is not missing —
it is priced." Four personas independently re-read `SORTING.md:1288-1292`: the measured
figure is *unclamped* (SFPSHFT wraps mod 32; a safe clamp adds +2-4 instructions and makes
it worse than binary search outright; SFPIADD counters need draining every 15 vectors) and
the full-width version costs 20,580 cyc. The primitive exists, is measured, and is a
**rejected loser** — the inversion of the claim. Salvage: one documentary sentence in §4.2
citing it as considered-and-rejected, so nobody rediscovers it. The debate around this
finding also surfaced the HistMacro+HistSum 3-bit/3.0-cyc arm (Verifier) that corrected
IMPL-1's per-pass constant — the swarm's clearest example of adversarial verification
outperforming any single persona.

### 6.5 Claim-cluster outcomes (audit scoreboard)

| Claim (dossier §) | Outcome |
|---|---|
| C2/C3/C4 sign-magnitude order, SFPGT, packer order (§4.1) | **Confirmed** against `SFPGT.md:3,55-66` + silicon tests; add VD-gating footnote |
| C5 exponent histogram = "pass 1 of MSD radix select" (§4.2) | **Refuted as framed** — no economical multi-bin pass 2+ exists; predictor only |
| C6–C9 histogram cost/sampling/aliasing/sign-blindness (§4.3) | **Confirmed** end-to-end; aliasing+saturation are WH-documented, BH-confirmed |
| C10 cycle figures are models, not measurements (§4.3) | **Confirmed** and worse: denominators are strawmen; per-tile sync drains uncharged |
| C11 predictor cannot decide correctness/termination (§4.3) | **Confirmed** — and the compound wrong-window failure (VERI-2) added |
| C12 filter costs (§4.4) | **Confirmed**; end-to-end tier (4.175, 1.66–1.68×) is stronger than the doc quotes |
| C13 sparse stream not production-ready (§4.4) | **Confirmed** — and promoted: this is the load-bearing gate |
| C14 u16 fused-key narrowness (§4.4) | **Confirmed** + explicit N≤65,536 bound required |
| C15 T<0 UB + SFPU fallback floor (§4.4) | **Confirmed**; floor scoped to value-preserving filters; signed deltas inherit [DISPUTED] denominators and the 6.7% edge inverts vs the fused macro merge |
| C16 TT-vs-GPU mapping (§4.5) | **Partially corrected**: count/unpack serialization is stock-path (§0a-bis escape measured); (Cgt,Ceq) readback mechanism exists via Dst-mapped RISC access |
| C17 verdict (§5) | **Recalibrated** — see §6.1 |
| C18 gates (§5.2) | **Reordered** — Gate 4 load-bearing; Gate 2 re-scoped to bisection + count + bitonic-finish correctness oracle |

*Generated by a 14-agent swarm audit (7 personas × 2 rounds), 2026-08-16. Composite
predict score: 620 (39 confirmed × 15 + full persona participation + both rounds +
anti-herd pass). The audit's scratch knowledge files were folded into this appendix.*
