# Dual-RISC BF16 Radix Selector — Feasibility Analysis

**Task:** assess the "strongest unmeasured alternative" from the RADIX_BUCKET_GPU.md swarm audit: BRISC+NCRISC scan raw bf16 words, build exact 256-bin high-byte histograms in private memory on XOR-mapped keys, tree-reduce, locate the boundary byte, run a second exact low-byte histogram restricted to the boundary high byte, locate the threshold, then emit.

**Scope:** read-only research on `nkapre/sorting` @ Blackhole box, 2026-08-16. Nothing here was executed on device. Every number is tagged **[ISA-doc]** (tt-isa-documentation/BlackholeA0), **[measured]** (SORTING.md silicon results), or **[estimate]** (this document's derivation, unmeasured).

---

## 0. Executive summary

1. **The in-tree scaffold is real and complete.** `masked_bincount` (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/`) demonstrates every synchronization structure the selector needs: same-source dual compilation on RISCV_0/RISCV_1, in-tile face-aware row addressing, init/done semaphore handoff between the two RISCs, a binomial-tree cross-core reduction with a memory-ordering fence, and TensorAccessor-based I/O. Its *accumulation* strategy (NoC atomics into a shared L1 histogram) is the one part that must NOT be copied for a 64Ki-element scan.

2. **A first-principles instruction floor kills the single-core threshold-finder framing before any measurement.** The BabyRISCV is in-order, single-issue, 1 instr/cycle max **[ISA-doc]**. The best-case pass-1 inner loop is ~5.25 instructions/element; pass 2 adds ~3.2; so the dual-RISC two-pass threshold find costs ≥ N·(8.4)/2 ≈ **4.2 cyc/element of N even at a perfect IPC of 1** [estimate]. The bars it must beat on one core: shipping bitonic `topk_local_sort` ≈ 2.38 cyc/elem **[measured]**, optimized merge cascade ≈ 0.10–0.20 cyc/elem **[measured model]**, SFPU 3-bit bisection ≈ 0.56 cyc/elem isolate / ~1.3 charged **[measured components]**. Even the *most favorable honest comparison* (SFPU 1-bit bisection fully charged with the 3.938 cyc/vec unpack serialization ≈ 3.0 cyc/elem) requires X₁+X₂ ≤ 6.0 — barely above the instruction floor and below any realistic stall-inclusive estimate (9–14 cyc/elem for the two passes combined).

3. **What survives:** (a) the **multicore width-sharded shape** (64 cores × 2 RISCs; per-core slice N/64) where the scan shrinks to ~5–7 k cycles/core and the tree reduction (~6–10 k cycles critical path) dominates — plausibly 8–13 µs whole-row at N=65536, in the same band as post-log-tree `topk_large_indices` (~15–42 µs, different K regime) [estimate vs stale baselines]; (b) the **RISC scan rate X is load-bearing for Gate 2 regardless**, because 4 of 5 candidate materialization paths (dense emission, count→offset→direct-write, keep-mask/gather, tile skipping) are RISC scan loops of the same shape; (c) the RISC arm occupies **zero compute-engine resources**, so it composes with a bitonic/SFPU finish instead of competing with it for Dest/unpack.

4. **Recommended next artifact:** a ~250-line single-core programming-example microbench that measures the BRISC histogram inner-loop rate (5 variants, below) — reframed as *"measure the RISC scan/emit rate for Gate 2"*, not *"race the SFPU threshold finder"*. Go/no-go on the selector arm: measured pass-1 X ≤ 6 cyc/elem keeps the multicore + overlap variants alive; X ≥ 10 demotes the dual-RISC arm to correctness oracle and Gate-2 emit engine only.

---

## 1. `masked_bincount` anatomy (the in-tree precedent)

Files: `device/masked_bincount_program_factory.cpp` (host), `device/kernels/reader_masked_bincount.cpp` (single kernel source compiled twice per core). Modern D2.0 dataflow API (`Noc`, `CircularBuffer`, `Semaphore<>`, `TensorAccessor`).

### 1.1 Dual-RISC split
- The **same kernel source** is created twice per core: RISCV_0/NoC0 (BRISC) and RISCV_1/NoC1 (NCRISC), differentiated by ct-arg 7 `is_initializer` (factory lines 157–177, kernel line 77).
- Fixed 8×8 = 64-core grid; token rows split contiguously per core (`shard_height = tokens/64`), then halved on-core: `h_brisc = shard_height/2`, NCRISC takes the rest starting at `h_start + h_brisc` (factory lines 37–47, 227).
- Each RISC reads its own TILE pages into a **separate input CB** (c_0 for BRISC, c_2 for NCRISC), sized `(⌈h/32⌉+1)` pages because unaligned row ranges straddle one extra tile (factory lines 46–47, 64–83). Boundary tiles may be read twice (once per RISC) — each counts only its own rows (kernel comment lines 107–110).

### 1.2 L1 histogram layout and accumulation
- **One shared histogram** per core: CB c_1, a single page of `n_routed_experts` u32 (factory lines 71–76). BRISC zeroes it; both RISCs increment it.
- Increments are **NoC atomics to the core's own L1**: `noc_semaphore_inc(get_noc_addr(my_x, my_y, out_addr + idx*4), 1)` (kernel lines 165–172). The comment documents this as a deliberate fallback: D2.0 has no wrapper for atomic-increment on an arbitrary L1 word, and atomicity is required because both RISCs target the same bins. `noc.async_atomic_barrier()` closes the phase (line 175).
- Cost of that choice **[ISA-doc + estimate]**: an L1 atomic is ≥12 cycles latency (BabyRISCV README.md:82) and each `noc_semaphore_inc` is ~5 MMIO command-register stores plus address formation. Fine for its workload (per core ≤ `shard_height × num_experts_per_token` increments — dozens); fatal at 32Ki+ elements/core.

### 1.3 Semaphores and tree reduction
- **init_sem:** BRISC zeroes the histogram, fetches the expert mask, `init_sem.set(1)`; NCRISC waits (kernel lines 129–140).
- **done_sem:** each RISC does a NoC-atomic self-increment (`done_sem.up(noc, my_x, my_y, 1)` — required because both RISCs bump the same word), BRISC waits for ==2 (lines 180–186).
- **Tree (BRISC only):** binomial tree — core i's children are i+2^L for the levels where `i % 2^(L+1) == 0`; parent = clear lowest set bit (factory lines 184–209). A **single counting gather_sem** means the parent must `wait_min(num_children)` before reading *any* child, because the counter doesn't identify which child signaled (kernel lines 196–201). Parent NoC-reads each child's 1-page histogram into CB c_3 and does a scalar `local_hist[i] += remote_hist[i]` loop (lines 202–218).
- **Memory-ordering fence:** before signaling its parent, a non-root core issues `ckernel::load_blocking(local_hist + last)` to force the accumulation stores into L1 ahead of the semaphore MMIO write, which can otherwise race ahead (kernel lines 220–229, citing NoC/Ordering.md). The root writes the result via TensorAccessor (lines 231–234).

### 1.4 Per-element inner loop (lines 156–174) — instruction estimate [estimate]
Per row: face-aware address `row_base = tile_local*1024 + (within/16)*512 + (within%16)*16` — all shifts since tile_h=32 is constexpr. Per element (column w): volatile `lhu` from L1 CB; bounds compare+branch; volatile indexed `lw` of `mask[expert_idx]` (L1); compare+branch; `get_noc_addr` (~3–5 ALU ops); `noc_semaphore_inc` (~5 MMIO stores). **≈ 15–25 RISC instructions + one NoC atomic per counted element.** Unmeasured — no perf test exists for this op. This is 3–5× the instruction count of the private-histogram loop below, which is exactly why the selector must change the accumulation strategy while keeping the sync skeleton.

---

## 2. Scalar throughput priors (BabyRISCV, Blackhole A0)

Source: `/home/nachiket/tt-isa-documentation/BlackholeA0/TensixTile/BabyRISCV/README.md` (+ CSRs.md). **No measured RISC histogram loop rate exists anywhere in SORTING.md, RADIX_BUCKET_GPU.md, or the masked_bincount tree — everything in §2.2–2.3 is an unmeasured derivation from ISA-documented pipeline behavior.**

### 2.1 Hardware facts [ISA-doc]
| Fact | Value | Evidence (README.md) |
|---|---|---|
| Core type | in-order, **single-issue**, 1 instr/cyc max, 1.35 GHz | :3 |
| ISA | RV32IM + Zicsr/Zaamo/**Zba/Zbb** (so `sh2add` exists); only T2 has partial V — **BRISC/NCRISC have no vectors** | :7 |
| L1 load latency | **2 cyc on L0-dcache hit, ≥8 on miss** (+bank/port conflicts) | :75–83 |
| L0 dcache | **64 B total (4×16 B lines)**, non-coherent, ~0.8%/hit random full flush unless `cfg0.DisLowCachePeriodicFlush` (CSRs.md:72) | :138–142 |
| L1 store throughput | 1/cyc **only when coalescing to aligned 128-bit blocks; otherwise one coalesced store per 5 cycles** — random-address histogram stores to L1 pay ~5 cyc each | :85 |
| Local data RAM | **8 KiB on BRISC/NCRISC** (4 KiB on TRISCs); load latency 2; stores 1/cyc; second NoC-visible mapping at 0xFFB1_4000 (B) / 0xFFB1_6000 (NC) | :144–157, memory map :109–110 |
| L1 atomic | ≥12 cyc | :82 |
| Load-latency hiding | 8-entry retire queue; up to ~7 independent instrs can cover a miss; use distinct dest registers | :89–95 |
| Mispredicted branch | ~5 cyc effective | :55 |
| Store→load same-address | load overlapping the store queue **drains the queue** (stall) instead of bypassing | :71 |

**Two placement conclusions fall straight out of the table:** the histogram must live in **core-local data RAM** (1 KiB of 256×u32 fits the 8 KiB easily; 2-cyc loads, 1/cyc stores, zero contention), *not* in L1 (5-cyc random-store throughput ≈ doubles X) and *never* behind NoC atomics (≥12 cyc). The local RAM's slow-path NoC mapping (or a 1 KiB copy-out to a CB) still lets the masked_bincount tree reduction read it remotely.

### 2.2 Inner-loop instruction budget [estimate]

**XOR map is free.** `key = b XOR ((b&0x8000)?0xFFFF:0x8000)` acts on the high byte as a pure function of the raw high byte (`hb_key = hb ^ ((hb&0x80)?0xFF:0x80)`), and within a fixed boundary high byte the low-byte map is a fixed XOR constant. So both passes histogram **raw bytes** and apply the map as a 256-entry bin permutation during the (tiny) prefix/locate step. Zero in-loop cost.

**Pass 1** (high-byte histogram, counters in local RAM). High bytes of bf16 words sit at odd addresses, so `lbu` at stride 2 needs no shift/mask:
```
lbu   t0, 1(p)        # raw high byte           1 instr (2 cyc L0-hit / ≥8 miss)
sh2add t1, t0, hbase  # &hist[t0]  (Zba)        1
lw    t2, 0(t1)       # counter, local RAM      1 (2-cyc latency)
addi  t2, t2, 1       #                         1
sw    t2, 0(t1)       #                         1 (1/cyc to local RAM)
```
+ ~0.25/elem loop/pointer overhead at 8–16× unroll ⇒ **~5.25 instr/elem floor → X₁ ≥ 5.3 cyc/elem at perfect IPC**. Stall adders: one ≥8-cyc L0 miss per 16 B line (8 elements) partially hidden by unrolled scheduling (~+0.75/elem unhidden [estimate]); the counter `lw` is dependent on `lbu` and the `sw`→`lw` same-bin dependency chains. **Realistic X₁ ≈ 6–10 cyc/elem; adversarial same-bin input (clustered logits — the common case for real data, since one binade dominates) triggers the store-queue-drain rule every element, plausibly 10–15 cyc/elem** unless mitigated by 2–4 interleaved sub-histograms (+1–3 KiB local RAM, +256·k merge adds).

**Pass 2** (low-byte histogram restricted to boundary high byte): `lhu` + `srli` + predictable not-taken `bne` ≈ 3 instr/elem for non-matching elements (the vast majority), +5 for matches ⇒ **X₂ ≈ 3–5 cyc/elem**.

**Total two-pass threshold find: X₁+X₂ ≈ 9–14 cyc/elem per RISC [estimate]; dual-RISC wall ≈ 4.5–7 cyc per element of N.**

**Tree reduction (multicore only):** per level ≈ 1 KiB NoC read + 256×(lw,lw,add,sw) ≈ 1.0–1.3 k cycles; log₂64 = 6 sequential levels ⇒ **~6–10 k cycles critical path** + semaphore waits [estimate]. Negligible per-element at N=65536 single-row-per-core; dominant in the width-sharded shape.

### 2.3 The bar, stated honestly (N = 65536, single core)

Dual-RISC covers N in `N·X/2` cycles. With X = X₁+X₂ (both passes):

| Competitor | cyc/elem | Basis | Break-even X₁+X₂ | Verdict vs 9–14 estimate |
|---|---|---|---|---|
| Shipping bitonic `topk_local_sort` (76.195 cyc/vec → 78,024 cyc @ N=32768) | **2.38** | [measured] SORTING.md:1050, :54–55 | ≤ 4.76 | **lose ~2–3×** |
| Optimized merge cascade (~6,509 → ~3,334 cyc @ N=32768, K=32) | 0.10–0.20 | [measured model] SORTING.md:54, :203 | ≤ 0.2–0.4 | **lose 25–70×** |
| SFPU 3-bit bisection, isolate (6 passes × 3.0 cyc/vec) | 0.56 | [measured components] SORTING.md:1305–1342 | ≤ 1.13 | **lose ~10×** |
| SFPU 1-bit bisection, isolate (16 × 2.0 cyc/vec) | 1.00 | [measured] SORTING.md:1045 | ≤ 2.0 | **lose ~5×** |
| SFPU 1-bit bisection, charged with 3.938 cyc/vec unpack serialization (stock LLK path) + 16×25.1-cyc rendezvous | ~3.0 | [measured components] SORTING.md §0a-bis; RADIX_BUCKET_GPU.md §6.1 | ≤ 6.0 | **lose ~1.5–2.3×** (closest race) |

Equivalently in the task's units: SFPU bisection is 0.06–0.13 cyc/vec-element *per pass* and ~0.56–1.0 cyc/elem for a full 16-bit threshold; the dual-RISC arm cannot get under ~4.2 cyc/elem even at IPC=1. **Single-core, the dual-RISC threshold-finder loses to every competitor including the fully-charged stock-path SFPU bisection.** The instruction floor makes this a pre-measurement conclusion.

**What changes the frame:**
- **Multicore width-sharding** (the masked_bincount shape): per-core slice N/64 = 1024 → scan ≈ 1024·(9–14)/2 ≈ 4.6–7.2 k cyc, + 6–10 k reduction ⇒ **~11–17 k cycles ≈ 8–13 µs** whole-row [estimate]. Post-log-tree `topk_large_indices` is ~15–42 µs (different, larger-K regime; RADIX_BUCKET_GPU.md §6.1 item 5) and the archived stock multicore point was ~171 µs at N=32k. Both baselines are stale/mismatched — this comparison is *unsettled*, not lost.
- **Resource orthogonality:** the RISC arm uses no unpacker, no Dest, no SFPU, no packer. It can run in the shadow of the streaming/bitonic work; its *marginal* cost is `max(0, t_RISC − t_compute)`. At X_eff 4.5–7 vs compute 0.2–2.4 cyc/elem it still overshoots the shadow by 2–30× at large N single-core — overlap alone does not rescue it, but in the multicore shape it can hide most of the scan under the reader's NoC streaming.
- **Fixed pass count:** exactly 2 data passes and ~2 locate steps, versus bisection's 8–16 data-dependent decisions × ≥25.1-cyc rendezvous — the RISC arm's advantage grows as N shrinks (at N≤2048 the bisection rendezvous overhead dominates its own data passes), which is the opposite corner from where its throughput deficit hurts.
- **Gate 2 needs this number anyway:** dense emission, count→offset→direct-write, keep-mask/gather, and count-guided tile skipping are all RISC scan loops of the same shape (~4–7 instr/elem). Measuring X prices the load-bearing gate of the whole campaign, whichever engine finds the threshold.

---

## 3. Microbench sketch (the minimal next artifact)

**Goal:** measure X for the pass-1/pass-2/emit loops on one BRISC, then the dual-RISC overlap — *without* building the selector. Single core, no ttnn op, no tree reduction (its cost is a separate 30-line follow-on using the masked_bincount pattern verbatim).

**Artifact:** `tt_metal/programming_examples/risc_histogram_bench/` — one host `main.cpp` + one dataflow kernel, ~250 LOC total, modeled on the existing programming-example skeleton (the user's tree already has `eltwise_poly/` as a local precedent). No compute kernel at all.

**Host side:**
1. `CreateDevice(0)`, single core (0,0).
2. N = 65536 u16 words in an interleaved DRAM buffer; input distributions selected per run: (a) uniform random bytes, (b) realistic bf16 logits (one dominant binade — the same-bin adversary), (c) all-equal (worst-case store-queue drain), (d) tile-faithful layout of a real [32, 2048] TILE tensor for the traversal variant.
3. One L1 buffer/CB of N·2 = 128 KiB for the data; one small L1 result buffer for `{t0, t1, hist[256]}` readback.
4. Compile-time arg selects the variant; host validates the histogram bit-exactly against a golden and reports `(t1−t0)/N`.

**Kernel variants (one `#if` family, timed with two back-to-back runs, second one quoted, wall clock via `c_tensix_core::read_wall_clock_l()` around the loop only — data already resident and barriered):**
- **V1 (design point):** local-RAM histogram at `MEM_LOCAL_BASE`, `lbu` stride-2, 16× unrolled, distinct dest regs per the retire-queue rule.
- **V2:** identical loop, histogram in L1 — prices the 5-cyc non-coalesced store rule directly (predicted ≈ 2× V1).
- **V3:** V1 + 4 interleaved sub-histograms + final 256×4 merge — the same-bin drain mitigation, run against inputs (b)/(c).
- **V4 (control):** masked_bincount-style `noc_semaphore_inc` accumulation — anchors how far the in-tree pattern is from viable (predicted 15–25+ cyc/elem).
- **V5:** pass-2 loop (high-byte match + low-byte histogram) and the Gate-2 emit loop (compare vs threshold + conditional store of `(value,index)` to an L1 output cursor) — the number Gate 2 consumes.
- **V6 (second step):** add the NCRISC kernel, split halves, semaphore start-gate (init_sem pattern from masked_bincount), root records `max(t_brisc, t_ncrisc)` — measures dual-RISC scaling and L1 read-port contention in one shot.
- Toggle `cfg0.DisLowCachePeriodicFlush` (CSRs.md:72) in one A/B to quantify the 0.8% random-flush tax.
- **V7 (traversal):** V1 with the TILE face-run address pattern of §4.1 instead of a flat scan — prices the tiled-layout overhead.

**Decision rule:** X₁(V1, uniform) ≤ 6 and X₁(V3, clustered) ≤ 8 keep the multicore selector arm alive → proceed to the tree-reduction bench. X₁ ≥ 10 → the dual-RISC arm is a Gate-4 correctness oracle and Gate-2 emit engine only; the threshold finder stays SFPU bisection.

---

## 4. Risks

### 4.1 Tiled address traversal
`ttnn.topk` inputs are TILE_LAYOUT. A logical row r restricted to one 32×32 tile is **two contiguous 16-element (32 B) runs 512 B apart**: u16 offset `(r%32/16)·512 + (r%32%16)·16` for the left face and +256 u16 for the right (exactly the arithmetic masked_bincount uses, kernel lines 148–161). Per-tile address setup is ~4 ALU ops per 16-element run ⇒ ~0.25 instr/elem amortized — cheap, all shifts (no `div`: 6–33 cyc **[ISA-doc]** :51–54). Each 32 B run spans two L0 lines; the stride-2 `lbu` pattern gets 8 high-bytes per 16 B line, so the miss cost is one ≥8-cyc miss per 8 elements either way. **Recommendation: scan TILE directly with face-run traversal; do NOT require ROW_MAJOR input** — an untilize pass costs a full extra streamed pass through the compute engines and forfeits the arm's resource-orthogonality. Caveats: tile-padded tail columns must be excluded by loop bounds (garbage in padding would corrupt bins); a 32-row tile batch is 32 *independent* selection problems → 32 histograms = 32 KiB > 8 KiB local RAM, so batch rows are processed in groups of ~6 (u32 counters) or ~14 (u16 counters, N≤65535/row) per sweep, or per-row sequentially.

### 4.2 L1 read latency and the (near-)absence of dcache
There *is* a dcache, but it is 64 B, non-coherent, and randomly self-flushing **[ISA-doc]** :138–142. Sequential scans get 1 miss + 7 hits per 16 B line; the ≥8-cyc miss must be hidden by unrolled, distinct-register scheduling under the 8-entry retire queue — the single largest scheduling risk in hitting X≈6. Non-coherence is safe here (data is read-only after the NoC-read barrier). Disable the periodic flush via `cfg0` for the bench; measure with it on for production honesty.

### 4.3 Histogram-update hazards
- **L1-resident histogram:** ~5 cyc/store (non-coalesceable random addresses) **[ISA-doc]** :85 — avoid; local RAM only.
- **Same-bin bursts:** the store-queue drain-on-overlap rule **[ISA-doc]** :71 makes clustered data (the *typical* logits case, one dominant binade) the throughput adversary, not an edge case. Mitigation: 2–4 interleaved sub-histograms; must be in the bench (V3 vs input (b)/(c)).
- **NoC-atomic accumulation** (the masked_bincount pattern): ≥12 cyc L1 atomic + ~5 MMIO stores per increment — correct, and 3–5× too slow here. Private per-RISC histograms + one 256-add merge replace it.

### 4.4 Dual-RISC contention and role conflicts
Two scalar readers demand ≤ ~8 B/cyc aggregate against an L1 fabric WH-documented at 256 B/cyc (BH undocumented, listed as "more" — SORTING.md:813) — bandwidth is a non-issue; bank/port conflicts add latency jitter only, and splitting halves keeps the RISCs mostly in different banks. The real conflict is **role occupancy**: in a fused op, BRISC/NCRISC are the reader/writer. The scan must interleave with NoC issue for the next tiles (issue is cheap fire-and-forget; the barrier discipline of §1 applies), but the RISCs cannot scan and service a second op's dataflow simultaneously — the selector owns the core for its duration.

### 4.5 Estimate risk
All X values here are derived, not measured; the pieces that most commonly break such derivations are compiler codegen (volatile pointer semantics forcing reloads — masked_bincount's `volatile tt_l1_ptr` idiom would serialize the unrolled loop; the bench kernel must read via non-volatile pointers after the barrier), i-cache effects on the unrolled body, and the actual cost of the L0 miss under back-to-back stride-2 loads. That is precisely why the bench exists.

---

## 5. Verdict

Build the §3 microbench next — it is small, device-cheap, and its output (X) prices Gate 2's materialization loops no matter which arm wins. But re-scope the hypothesis it tests: the dual-RISC BF16 two-byte-digit selector is **arithmetically dead as a single-core threshold finder** (instruction floor ~4.2 cyc/elem vs an incumbent at 0.1–2.4); it remains live only as (a) the **multicore width-sharded selector** whose scan hides under NoC streaming and whose real cost is the 6-level tree reduction, and (b) the **Gate-2 emit/count engine** attached to an SFPU-bisection threshold finder. The masked_bincount scaffold should be reused for the sync/tree skeleton with the accumulation swapped to private local-data-RAM histograms.
