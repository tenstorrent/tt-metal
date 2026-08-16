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
