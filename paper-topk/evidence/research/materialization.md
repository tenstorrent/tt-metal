# Gate 2 — Candidate Materialization Microbenchmark Design (THE go/no-go)

**Scope.** Given a *known exact threshold T* for a row of N bf16 values, decide whether the
~K survivors (with unique UINT32 global indices, bit-exact values) can be materialized
cheaply enough to beat the incumbent bitonic paths. Per RADIX_BUCKET_GPU.md §5.2 gate 2 and
the §6 swarm audit (IMPL-2: "Gate 4/materialization is the load-bearing go/no-go"), this is
the gate that decides whether any selector work continues. This document designs the
microbench set; nothing here was run on the device (read-only task).

Repo state: branch `nkapre/sorting`, Blackhole silicon box. All costs below are either
MEASURED (cited to SORTING.md / test sources), ISA-DERIVED (cited to
`~/tt-isa-documentation/BlackholeA0`), or PRIOR (this document's estimate, to be replaced
by the microbench).

---

## 1. The envelope to beat (incumbent numbers, pinned)

| quantity | value | source |
|---|---|---|
| Bitonic leaf cost | ~1.8 cyc/elt @K=512, ~2.1 @K=2048 (merge-unit model) | THRESHOLD_SELECT_DESIGN.md §5; SORTING.md §0a-quinquies |
| `topk_local_sort` end_phase 5 | 76.195 cyc/vec = 2.38 cyc/elt | SORTING.md:1355 (via §6 audit IMPL-2) |
| Whole-op `topk_large_indices`, post log-tree merge | k512@65536: **14.96 µs**@52c; k2048@65536: **41.9 µs**@26c; k512@262144: 32 µs@P64 | campaign memory (topk-sorting-campaign.md), tree merge commit 8794fbb |
| Routed `ttnn.topk` k512@65536 | 112.4 µs (values-native) / 134 µs | campaign memory, PR2 809cf5b |
| Stock multicore `ttnn.topk`, the §5.1 cell (K=32, N=32k, 65c) | **~171 µs** (archived Tracy baseline) | campaign memory `sweep/tracy_baseline.csv` |
| Stream floor (stock LLK, per pass) | 3.86–4.13 cyc/vec = **0.12–0.13 cyc/elt** | SORTING.md §0a-bis/§0a-ter [DISPUTED band] |
| Selection passes (pass1 + pass2 model, known-T excerpt: pass2 only) | pass2 ≈ 8.2 cyc/vec = **0.26 cyc/elt** (fuse 2.0 + count 2.0 + filter 0.1 on the 4.13 stream) | THRESHOLD_SELECT_DESIGN.md §5 table |
| SFPU exact count | 2.0 cyc/vec floor (CountD1); mask-map form **1.003 cyc/vec** | SORTING.md:1207 §"ARCHITECTURAL FLOOR" |
| Data-dependent rendezvous | ≥25.1 cyc per decision (PassSync) | SORTING.md:1220 (via audit) |
| Zero-compression | filter+compact in one PACR, +0.097 cyc/vec over bare stream; **max elision stride 16 ⇒ N_aug ≥ K + ceil((N−K)/16)**; dense inflates +28% | SORTING.md:1149–1186 |

**Pass/fail bar for this gate** (derived, stated up front):

- **PASS:** a complete materialization path — given T, per-core — produces exactly K
  unique in-range UINT32 indices and bit-exact gathered values at an **amortized added
  cost ≤ 0.5 cyc/element** over the selection passes, across the full adversarial battery
  (§6). That keeps the whole-selector model at ≤ ~1.0 cyc/elt vs the bitonic 1.8–2.1,
  i.e. ≥ ~1.8x compute headroom before fixed costs — enough to survive the §5.3 "repeatable
  whole-op win beyond pooled noise" requirement.
- **MARGINAL (capability-only):** 0.5–1.4 cyc/elt — the selector ties the incumbent;
  continue only for the arbitrary-k capability claim, make no perf claim.
- **FAIL:** > 1.4 cyc/elt amortized, or any correctness hole (silent truncation, tie
  mis-allocation, index collision) without loud in-path detection → **stop selector work**
  (RADIX_BUCKET_GPU.md §5.2 gate-2 stop rule).

---

## 2. RISC-side cost model prior (the constant every candidate shares)

**No measured scalar-RISC L1 scan rate exists anywhere in SORTING.md** (grep for
scan/cyc-per-element rates over BRISC/NCRISC comes back empty; SORTING.md's RISC content is
issue-rate of Tensix instructions, not data scans). The prior must come from the ISA docs:

- Baby RISCVs are **in-order single-issue, 1 instr/cycle, 1.35 GHz**, RV32IM + Zba + Zbb
  (`min/max/cpop/ctz/clz` available) + Zaamo (`amoadd.w` on local L1 only)
  (`BlackholeA0/TensixTile/BabyRISCV/README.md:3`, `InstructionSet.md:60–70`).
- **L1 load latency: 2 cycles on L0-dcache hit, ≥8 on miss** (more under bank/port
  conflicts). The L0 dcache is 64 B — 4 lines × 16 B — non-coherent, with a ~0.8%
  random-flush that `cfg0.DisLowCachePeriodicFlush` disables
  (`BabyRISCV/README.md:73–81,138–142`). A sequential scan takes 1 miss per 16 B line;
  misses are hideable with ≥7 independent instructions (8-entry retire queue, distinct
  dest registers — README.md:95).
- **Store throughput:** 1/cycle when the store queue coalesces into aligned 128-bit
  blocks; otherwise 1 per 5 cycles (README.md:83). Survivor writes are ~K only — noise.
- **`amoadd.w` to L1: ≥12 cycles** (README.md:81) — atomics are for cross-RISC counters,
  not per-element work.
- **RISCV T2 only: partial RVV 1.0, 32×128-bit vregs, SEW≤32**, with a false-dependency
  bug on destination registers and per-element micro-op splitting on some instructions
  (`InstructionSet.md:28–35,75–80`). B/NC (the DM RISCs) have **no** vector unit.

**Derived dense-scan prior (candidate (a) inner loop).** Load u32 (2 bf16), per element:
extract half (1), branchless in-range test (for T ≥ 0 the sign-magnitude compare collapses
to *one unsigned range check* — survivor ⟺ `(uint16)(b − T − 1) < 0x8000 − T − 1`, ~3
instrs; the general signed case needs the XOR map `key = b ^ (0x8000 + ((b>>15)<<15) −
(b>>15))`, ~5–6 instrs), predicted-not-taken branch (1). With 16x unroll and scheduled
loads:

- **T ≥ 0 fast path: ~4.5–6 cyc/elt per RISC; general signed: ~6–8 cyc/elt per RISC.**
- **Dual-RISC (BRISC+NCRISC on disjoint halves): ~2.3–4 cyc/elt aggregate.**

That is **parity-to-worse vs the bitonic leaf (1.8–2.1 cyc/elt)** even before selection
passes — the dense scan cannot be the primary path. It becomes decisive only when the
stream it scans has already been shortened (candidates (c)/(e): 16x–32x shorter), where the
same instr/word constant divides by the compression factor.

---

## 3. Candidate designs

### (a) Fixed-size BRISC/NCRISC dense emission — *the calibration control*

**Kernel structure.** One dataflow kernel source compiled twice (BRISC
`is_initializer=true`, NCRISC `false`), copying the scaffold of
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/device/kernels/reader_masked_bincount.cpp`
verbatim for: dual-compilation pattern (lines 20–23,77), row-range split with shared
boundary tile (107–115), `TensorAccessor` page reads (87–124), init/done semaphore
choreography (127–140,180–182), and the face-aware untile addressing
(`(r/16)*2*256 + (r%16)*16 + c`, lines 148–161) if the input is TILE layout. Preferred:
feed ROW_MAJOR/flat L1 so the scan is a straight pointer walk (the untile index math costs
~4 instrs/row otherwise).

Per RISC:
1. Read slice half into its input CB region (or scan in place if resident).
2. Tight loop: load u32, test both u16 halves against T in sign-magnitude
   (branchless range check for T≥0; XOR-map for signed), on survivor compute
   `global_idx = slice_start + elt_off` and store `(value:u16, idx:u32)` (or the fused
   `(bits<<16)|(local+1)` word) into a **fixed-size per-RISC output region** at a bump
   pointer; increment local count in a register.
3. Write count + overflow flag to L1; `done_sem.up()` (masked_bincount idiom).
4. Overflow policy: when bump pointer hits cap, **stop writing, keep counting**, set the
   flag — the caller (root) sees count > cap and retries/falls back. Never silent.

**Cost prior:** §2 → 2.3–4 cyc/elt dual-RISC aggregate (dense). At the §5.1 cell
(N=32k/core) that is 75–130k cycles ≈ 55–97 µs single-core, ~1–2 µs at 64-way split.

**Why build it anyway (first):** it is the ~150-line bench that measures the one constant
every other candidate's model divides by — instrs/word achieved, L0-hit behavior,
miss-hiding effectiveness, store-queue coalescing — and it *is* the consumer inner loop of
(c) and (e). Also the standalone winner for tiny N / short-row batches where fixed Tensix
pipeline costs dominate.

**Expected verdict:** standalone dense loses (parity at best). Value = calibration + the
sparse second stage.

### (b) Per-core count → exclusive offset → direct writes — *the multicore glue, not an alternative*

**Kernel structure.** This is not a per-element method; it is how any of (a)/(c)/(e)
becomes exactly-K across C cores without atomics:
1. Each core produces `(local_Cgt, local_Ceq)` — from the SFPU count pass
   (+2.0 cyc/vec, SORTING.md:1207) or as a free by-product of the RISC scan in (a).
2. Ship 8 B/core to the root — the masked_bincount **binary-tree reduction** (BRISC-only
   phase 3, `reader_masked_bincount.cpp:184–235`) or the `writer_tree.cpp` pairwise
   ready/data-semaphore idiom; note the tree code's hard-won ordering details:
   wait-all-children-before-reading (single counter can't identify which child, lines
   197–201) and the **`load_blocking` L1-drain before signaling the parent** (lines
   220–228 — MMIO store can race ahead of L1 stores).
3. Root computes exclusive prefix over `Cgt` and per-core tie quotas
   (`k − ΣCgt` allocated by core order using `Ceq`), returns
   `(out_offset, tie_quota)` per core (one multicast).
4. Cores `noc_async_write` their survivors directly to `out_offset` — disjoint ranges,
   no output allocator. This is exactly RADIX_BUCKET_GPU.md §5.2 gate 7's structure,
   pulled forward because Gate 2's "unique UINT32 global indices" requirement is
   meaningless single-core.

**Cost prior:** one extra tiny NoC round (µs-class latency, ~10 B/core traffic) + remote
writes of ~K entries. The *count* is the only per-element term and it rides pass 2.

**Adversarial trap specific to (b):** tie quotas. With `Cgt = K` exactly, every core's tie
quota is 0 — a core must not emit a tie just because it has one locally. With ties
spanning cores, quota allocation must be by deterministic core order (assert repeatability
across launches).

### (c) Explicit keep-mask/prefix/gather on SFPU+pack — *reject the literal form; keep the mask-byte-map variant as backup*

**Literal form (as named in §5.2): reject on paper.** SFPU has no gather/scatter, no
indexed register file (audit IMPL-1, `RADIX_BUCKET_GPU.md:695`); a cross-lane prefix costs
an SFPTRANSP + SFPSHFT2 fold of ~18–20 instructions per vector (audit IMPL-3 mechanism,
`RADIX_BUCKET_GPU.md:778`) — ≥0.5 cyc/elt for the prefix alone — and even then there is no
instruction to *place* a lane at a data-dependent Dst position. Any "gather" ends up done
by a RISC, which is candidate (a) with extra steps. Do not spend silicon time on it beyond
one confirmation arm.

**Reformulated (c′), worth one bench arm — "mask byte-map + RISC ctz gather":**
1. SFPU produces the keep mask (`SFPGT` SET_VD, **mask-map form measured 1.003 cyc/vec**,
   THRESHOLD_SELECT_DESIGN.md §0 table) — rides pass 2.
2. Pack the mask tile as Int8 (1 B/elt ⇒ a 1024-elt tile becomes 1 KB) instead of packing
   values.
3. Writer RISC scans the byte map 4 elts per u32 load, `bnez` skip on zero words
   (~1–1.5 cyc per 4 elts ≈ **0.25–0.4 cyc/elt single-RISC**), and for each hit gathers
   the value from the still-resident input tile (random L1 load ≥8 cyc — but only ~K of
   them) and computes the global index positionally.

Sign-safe (SFPGT handles T<0 natively — no packer UB), no compression quirks, no
placeholder ambiguity. Slightly worse prior than (e) because the map is 1 B/elt vs
~0.25 B/elt for the compressed stream, but it is the **fallback if any (e) quirk bites**.

### (d) Count-guided tile skipping — *distribution-conditional add-on, cannot be the gate decision*

**Kernel structure.** Use per-tile survivor counts (free by-product of the pass-2 count or
the RISC scan) to feed **only surviving tiles** into the incumbent bitonic finish:
- Writer/reader RISC reads the per-tile count array from L1, pushes the ≤C_t tiles that
  contain survivors into the compute CB, and **pads to exactly C_t with prefilled −inf
  tiles** (the `topk_large_indices` empty-slice idiom, THRESHOLD_SELECT_DESIGN.md §0
  table row "Multi-core gather/tree idioms"). Padding keeps the compute kernel's tile
  count host-fixed — this sidesteps the audit's IMPL-4 hazard (data-dependent pass counts
  breaking the tri-thread fixed-stream model) entirely.
- Compute = unmodified `topk_local_sort` + merge cascade over C_t tiles.

**Cost prior / win model.** Win = bitonic leaf cost × (1 − C_t/N_tiles). For K=64
survivors spread randomly over N=32k (32 tiles), expected occupied tiles ≈
32·(1−(31/32)^64) ≈ 28 → **~12% win, i.e. nothing**. Clustered survivors (attention
logits, sorted-ish data): up to N_tiles/C_t = 4–8x on the leaf term. Worst case: zero win
and the count pass was still paid.

**Verdict:** cheap-ish to implement given counts already exist, **zero worst-case win** —
it cannot clear a gate whose bar is the adversarial battery. Keep as an opportunistic
post-gate add-on and as the only materialization-free fallback that reuses 100% of the
incumbent.

### (e) Device consumer for the packer compressed stream — *survey result: the fused-key design makes the decoder nearly trivial; this is the strongest candidate*

**What RADIX §4.4 says is missing** — an in-tree device decoder for the sparse PACR
stream — **is only needed if positions must be reconstructed from the compression
metadata.** The stream format (spec: host decoder
`tt_metal/tt-llk/tests/python_tests/test_pack_compress_int32.py:137–171`, semantics
confirmed for 32-bit datums):

- `rss_units × 16` B header: u16 row-start-index array, `num_rows+1` entries;
- then groups of **32 datums (4 B each) + 16 B of 32 four-bit counters**;
- BH counter = number of zeros **preceding** its datum (divergence from the WH doc,
  SORTING.md:1181–1186);
- `PackerTileSize` readback = byte count in 16 B units (`diag[3]`,
  test_pack_compress_int32.py:196–197) — a length, **not** a survivor count.

**Key observation (aligns with THRESHOLD_SELECT_DESIGN.md §1.4 "no nibble decoding is
ever needed"):** if the packed datums are the fused self-describing keys
`(bf16_bits << 16) | (local_idx + 1)` — the exact pattern
`test_pack_compress_int32.py:59–82` already packs and decodes bit-exactly — then a
writer-RISC consumer needs **none** of the metadata:

1. skip the `rss_units×16` B header (compile-time constant);
2. linear-scan the augmented datum words, `bnez`-skip zeros (placeholders are literal
   `0x00000000`; the `+1` index offset guarantees no survivor is ever the zero word —
   test file lines 62–64);
3. skip the 16 B counter block every 32 datums (address arithmetic, no decode);
4. for each nonzero word: `global_idx = slice_start + chunk_id·32768 + (w & 0xFFFF) − 1`,
   value = `w >> 16`; write to the fixed-size output region ((a)'s emit tail).

**What a *full* decoder would additionally need** (only if values were packed unfused):
rsi walk + per-datum nibble read + zero-run accumulation ≈ 8–12 instrs/word — roughly 3x
the fused-key consumer; plus the BH zeros-precede semantics trap. Assess: buildable but
strictly dominated by fusing indices; do not build it.

**Cost prior.** After one relucomp pass, stream length `N_aug ≈ N/16 + K` words
(16:1 elision cap, SORTING.md:1171–1176). Consumer at (a)'s measured instrs/word
(prior 3.5–5 cyc/word): **≈ 0.22–0.31 cyc/elt single-RISC, ~0.12–0.16 dual-RISC** — under
the 0.42 cyc/elt selection-pass rate, i.e. **fully hideable behind pass 2's stream** in a
pipelined kernel, and 6–15x under the bitonic leaf. One optional cascade stage (re-pack
the segment, another 16:1, cost ≈ 1/16 of a pass) drops the consumer another 16x if it
measures slow.

**Producer cost** is already measured: `MIN_THRESHOLD_RELU` + zero-compression is
+0.097 cyc/vec on a pack that pass 2 performs anyway (SORTING.md:1149–1169; 5/5
bit-exact filter+compaction).

**Blockers this candidate must carry (all testable):**
- **T < 0 is packer UB** (measured: |T| rounds up to a power of two, RADIX §4.4) — the
  signed arm substitutes `topk_negfilter` (SFPU, +2.0 cyc/vec, value-preserving zero-fill)
  before a plain compressed pack; the consumer is unchanged.
- Segment length must cross PACK-thread → writer-RISC (mailbox/L1 word + semaphore; or
  conservatively scan the max-size region relying on a pre-zeroed buffer).
- `DataflowBuffer`/CBs are fixed-entry (RADIX §4.4) — the consumer treats the segment as
  raw bytes in a plain L1 region, not CB pages.
- Config escapes: `Downsample_mask` THCON word-3 survives ELF reload (observed live —
  SORTING.md:1190–1196); `TTI_PACR` only, never `TT_PACR` (observed hang).

---

## 4. Ranking (expected win × implementation cost)

| rank | candidate | expected added cost (prior) | worst case | impl cost | decision role |
|---|---|---|---|---|---|
| 1 | **(e) relucomp/negfilter compressed stream + fused-key writer-RISC consumer** | 0.12–0.31 cyc/elt | dense survivors: stream inflates +28% ⇒ consumer ≈ dense scan (bounded by (a)) | ~400 lines kernel + LLK enable bit + plumbing | **The go/no-go measurement.** Only candidate whose prior clears the 0.5 bar with margin |
| 2 | **(b) count→exclusive-offset→direct writes** | ~0 per-element; one µs-class NoC round | tie-quota bookkeeping bugs | ~200 lines, all idioms exist in-tree (masked_bincount + writer_tree) | Required glue for exactly-K/unique-u32 on any path; measure the rendezvous fixed cost |
| 3 | **(a) dual-RISC dense emission** | 2.3–4 cyc/elt dense (loses); its instrs/word constant prices (c)/(e) | — | **~150 lines — build FIRST** | Calibration control + the emit tail of (c)/(e) + small-N niche |
| 4 | **(c′) mask byte-map + ctz/bnez gather** | 0.25–0.5 cyc/elt + K random loads | none beyond (a)'s | ~250 lines | Sign-safe backup if (e)'s compression quirks bite; literal SFPU prefix/gather form: reject on paper |
| 5 | **(d) count-guided tile skipping (−inf padding)** | 0 added, win only on clustered data | **zero win on spread data** | ~200 lines reader logic | Post-gate opportunistic add-on; cannot decide the gate |

Build order ≠ rank: build (a) first (one afternoon, calibrates everything), then (e)'s
consumer against synthetic segments, then (e)'s producer+consumer end-to-end, then (b)
around whichever wins. (c′) only on (e) failure; (d) only after a PASS.

---

## 5. Where the microbenches live

1. **RISC scan-rate benches — (a), (e)-consumer, (c′)-consumer:** a standalone tt-metal
   C++ test (gtest under `tests/tt_metal/`, or a `programming_examples/`-style Program)
   with dual BRISC/NCRISC kernels cloned from `reader_masked_bincount.cpp`, input staged
   into L1 by the same kernel, timed with `c_tensix_core::read_wall_clock_l` on the RISC
   (the data-movement wall-clock API per docs/profiling.md), results + cycle counts
   written to an output buffer the host asserts on. **Not tt-llk:** the tt-llk harness
   exercises TRISC threads and its numbers don't transfer to metal kernels
   (`.ttinsn` gathering divergence, SORTING.md §0b B7); BRISC/NCRISC kernels are
   metal-native. Synthetic compressed segments for the (e)-consumer bench are generated
   host-side from the `test_pack_compress_int32.py` encoder logic inverted — no packer
   needed to measure the consumer.
2. **(e)-producer (compressed segment + PackerTileSize/length readback):** a tt-llk test
   extending `tests/sources/pack_zero_compress_test.cpp` with a perf twin patterned on
   `pack_exp_histogram_perf.cpp` (producer/consumer perf flow, mutation controls,
   `SFPLOAD/SFPSWAP` 2.00x control pair per branch discipline).
3. **Gate-2 end-to-end assembly (known T injected as a runtime arg):** a standalone
   **ttnn experimental op bench** (new `ttnn/cpp/ttnn/operations/experimental/` op or a
   bare Program), measured with Tracy **Device Kernel Duration** under the canonical-sweep
   discipline: per-cell subprocess, `flock /tmp/tt-device.lock`, `.so`-mtime stamping,
   3+ trials with noise floor — vs the pinned incumbent cells (`topk_large_indices`
   post-tree-merge and routed `ttnn.topk`), at N ∈ {4K, 32K, 256K} × K ∈ {16, 64, 512,
   2048} single-core and C ∈ {8, 64}.

---

## 6. Adversarial test battery (every candidate runs all of these)

Correctness assertions everywhere: **exactly K outputs; K unique, in-range UINT32 global
indices; gathered values bit-for-bit equal to the golden multiset; overflow/failure is a
loud flag, never truncation.** (RADIX §5.2 gate 2 verbatim requirements.)

1. **All-negative row** (T < 0): packer arm must be provably out of the path (assert on
   the broadcast word's sign bit); negfilter arm exact; **zeroed losers must not
   masquerade as candidates** — assert no +0.0 value appears among winners unless
   genuinely present in the input.
2. **T = +0.0** with genuine ±0 elements at the boundary: `−0 < +0` in sign-magnitude —
   boundary prefers +0; the `+1` fused-index offset case (survivor value +0 at local
   index 0 must not be elided as the zero word).
3. **All-equal row** (Cgt = 0, Ceq = N): emit exactly K by deterministic positional quota;
   fixed buffers must not overflow (the tie-quota path, not the strict-winner path, is
   exercised).
4. **K−1 / K / K+1 strict winners vs T:** Cgt = K−1 (need exactly 1 tie), Cgt = K (zero
   ties — no core may emit a local tie), Cgt = K+1 (violates the Gate-2 precondition
   `Cgt < K ≤ Cgt+Ceq` — must be *detected*, not absorbed).
5. **Capacity edges:** survivors = cap and cap+1 per core — overflow flag fires, counts
   stay exact past the write stop.
6. **Placement extremes:** all K survivors in one tile vs exactly 1 per tile (greenlights/
   kills (d); stresses (e)'s placeholder asymmetry); all survivors at positions
   `p mod 64 ≥ 8` (harness parity with the predictor tests).
7. **Stream extremes for (e):** 0 survivors in a 32k chunk (placeholder floor N/16 words —
   the consumer's worst empty-work case) and 1024/1024 dense (compression inflates +28%,
   SORTING.md table — consumer degrades to (a)'s dense rate; must stay bounded).
8. **Specials at the boundary:** ±Inf, ±NaN (payload-carrying; sign-magnitude order
   `−NaN < −Inf … +Inf < +NaN`), denormal-heavy rows (bitwise pipeline tripwire, R4 of
   THRESHOLD_SELECT_DESIGN.md).
9. **Index width:** N = 131072 (> u16 span) forcing chunk-based u16→u32 reconstruction;
   assert uniqueness across chunk and core borders.
10. **Repeat launches / program cache:** second launch with different T and K
    (runtime-only args), no hang, bit-identical goldens; repeated multicore launches for
    the (b) semaphore epochs.

---

## 7. Bottom line

The audit's framing survives contact with the sources: **materialization, not comparison,
decides the campaign.** The dense dual-RISC scan (a) is priced by the ISA at 2.3–4
cyc/elt — parity with the bitonic leaf, a non-win — but the packer's measured 16:1
filter+compaction (+0.097 cyc/vec, riding a pass that must happen anyway) shortens the
stream the RISC must scan by 16x, and the fused `[bf16|u16+1]` key (already proven to
round-trip the compressed format bit-exactly on silicon in `test_pack_compress_int32.py`)
reduces the "missing device decoder" of RADIX §4.4 to a *skip-zero-words loop* — no
metadata decode at all. Candidate (e)'s prior of 0.12–0.31 cyc/elt is the only one that
clears the 0.5 cyc/elt gate bar with margin, and everything it depends on is measured
except one number: **the achieved RISC instrs-per-word scan rate, which the ~150-line
candidate-(a) bench measures first.** If that constant comes back ≤ ~5 cyc/word, Gate 2
passes on the (e)+(b) composition; if it comes back ≥ ~10 (miss latency un-hidden,
L0 pathologies), one cascade stage buys 16x and the gate still passes; only a
compression-quirk correctness hole (§6 tests 1/2/7) or a blown rendezvous budget kills it —
and then (c′) is the sign-safe backup before declaring FAIL and stopping selector work.
