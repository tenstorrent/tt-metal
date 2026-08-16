# TT-Metal Sorting & Top-K: Audit, Roofline, and Gated Plan

> [!IMPORTANT]
> **This is a corrected revision of `SORTING.md`.** The original presented modelled
> cycle counts as measurements, cited experiments that were never run, and proposed
> as novel at least two optimizations that are already in production. Every claim
> below is tagged:
>
> - **VERIFIED** — traceable to a `file:line` in this tree or a named ISA document page.
> - **ASSUMED** — a modelling assumption, stated explicitly so it can be attacked.
> - **OPEN** — unresolved; the experiment that would settle it is named.
> - **`[UNVERIFIED ESTIMATE]`** — a number with no primary source.
>
> **No speedup figure appears anywhere in this document**, because no device
> measurement exists for any operation discussed. §7 Phase 0 specifies the baseline
> that must exist before one may be written down.

---

## 0. What Changed and Why

Five independent audits were run against this document — one per section, plus an
ISA lookup against `tt-isa-documentation/BlackholeA0/` and the craq-sim source.
Summary of what did not survive:

| Original claim | Reality |
| :--- | :--- |
| "Prune bottom 8 in Phase 2 → 138 to 76 cycles" | **Already shipped.** `bitonic_top8_ph3_st4_to_1`, `ckernel_sfpu_generalized_moe_gate_topk_single_face.h:158`. Real saving: **one `SFPSWAP`**. |
| "Softmax max-subtraction via free rank-0 max" | **Already shipped**, `:675-694`, with the "max is free" note upstream at `:672-673`. |
| "Cross-arch UINT16 pack wrapper" | **Already shipped**, `compute_kernel_api.h:833`. |
| "Excessive transposition in `ttnn.sort`, ~18% overhead" | **Describes code that does not exist.** The merge network is already transpose-free. The 18% appears nowhere in the repo. |
| "Streaming reject filter, ≤6 cyc/tile, 32.4× win" | **Dead as specified** (needs a ~50-100 cycle pipeline drain), **and 7.4× slower than the roofline floor** even if it worked. Replaceable by something better — see §5.3. |
| "OneSweep radix, 904×-20916×" | **No implementation, no design.** Speedups were formula-vs-formula. |
| "Zero compiler blockers, ship immediately" | **Uncompilable.** The pinned SFPI 7.69.0 lacks `sfpswap_indexed`/`sfptransp8`. |
| Benchmark results table | Every row was `ANALYTICAL_MODEL`/`SOTA_PROPOSED` with `accuracy_pcc` hardcoded to 1.0. Artifacts have since been deleted. |

Two fabrication modes are worth naming, because they recur:

1. **Real identifier, invented meaning.** Commit `8f943c2f8` exists, but it is a
   `-g`/DEBUG_BIND GIMPLE fix — not the "Pre-IRA sentinel tracking" the document
   described. Same pattern for `ENABLE_DEST_INDEX` (an ISA-doc name, not an in-tree
   symbol) and `CC_ALL_ZERO` (not an ISA name at all).
2. **Directionally-true claim wearing false precision.** OneSweep is bandwidth-
   efficient, but ">85% of peak DRAM" appears nowhere in the paper.
   `x86-simd-sort` is fast, but "0.25 cycles/element" is unsourced and better than
   the best published figure in that space.

---

## 1. Audit of TT-Metal Sorting & Top-K Implementations

```
+--------------------------+-----------------------+-----------------------------+------------------+
| Operation / Factory      | Primary Algorithm     | Hardware Primitives         | Index Tracking   |
+--------------------------+-----------------------+-----------------------------+------------------+
| ttnn.sort (Single Core)  | Bitonic Merge Sort    | topk_local_sort, topk_merge | Separate CB/Dst  |
| ttnn.sort (Cross-Core)   | Parallel Bitonic Sort | NoC unicast + semaphores    | Transposed L1 CB |
| ttnn.sort (Multi-Core)   | Hierarchical Bitonic  | Output-DRAM ping-pong, sems | Separate DRAM    |
| ttnn.topk (Single Core)  | Sliding-Window Insert | topk_local_sort, double buf | Double buf Dst   |
| ttnn.topk (Multi-Core)   | 2-Stage Divide/Conquer| NoC streaming, topk_final   | Staging L1 CBs   |
| topk_large_indices (BH)  | Fused Stride Bitonic  | MOP Expander, Replay Buffer | Fused [Val|Idx]  |
| moe_grouped_topk         | Grouped Bitonic Sort  | topk_local_sort, reduce_tile| Index-template CB|
| moe_hash_gate            | Hash Routing (0-Sort) | DRAM Lookup, L1 Gather      | Reader Synth     |
| generalized_moe_gate     | Single-Face Bitonic   | SFPSWAP/SFPTRANSP, SFPSHFT2 | LO16 companion   |
+--------------------------+-----------------------+-----------------------------+------------------+
```

### 1.1 `ttnn.sort`

Three program factories, selected in `sort_device_operation.cpp:51-70`.

**Strategy 1 — Single Row Single Core** (`sort_single_row_single_core.cpp`)
- **VERIFIED** Applicability: `Wt <= 64` (`SORT_WT_THRESHOLD`, `sort_device_operation.cpp:14`);
  `<= 32` for UINT16 ROW_MAJOR (`:23`), where RM value CBs are promoted to Float32.
- **VERIFIED** Pads `W` to the next power of two (`>= 64` elements) with `+inf`
  ascending / `-inf` descending; UINT16 uses `65535`/`0` (`sort.cpp:79-82`, `:110-129`).
- **VERIFIED** Substage 1 calls `topk_local_sort(0, dir, 5)`; wider substages call
  `topk_merge(0, m_iter, 64)` (`:236-241`). **Note the `k` differs per factory** —
  cross-core uses `topk_merge(..., 32)` (`sort_cross_core_data_exchange.cpp:165,260`).
- **VERIFIED — correction.** Transposes occur **twice per row**, not per tile pair:
  once building the bitonic sequence (`sort_common.hpp:66-78`) and once on exit
  (`transpose_and_pack`, `:128-162`). The entire merge network (`:202-272`) runs on
  CBs named `input_tensor_transposed` / `index_tensor_transposed` and issues **no
  transpose at all**.

**Strategy 2 — Cross-Core Data Exchange** (`sort_cross_core_data_exchange.cpp`)
- **VERIFIED** Applicability: `Wt <= N_cores * TilesPerCore`, **non-UINT16 only**.
- **VERIFIED — correction.** The NoC exchange runs in the **reader** kernel
  (`reader_cross_core_data_exchange.cpp:159`), not the compute kernel. Compute only
  stages tiles into `value_tensor_intermediate` / `index_tensor_intermediate`
  (`:198-228`). The handshake is `sem_self.up(...)` + `async_atomic_barrier()` +
  `noc.async_write` (`cross_core_data_exchange_common.hpp:77-103`).
- **VERIFIED** Partition kept per `select_lower = dir ^ (i < j)` (`:260-271`);
  global barrier by leader core 0 (`:40-41`, `:146`).

**Strategy 3 — Single Row Multi Core** (`sort_single_row_multi_core.cpp`)
- **VERIFIED — correction.** **No dedicated DRAM scratch buffer is allocated.** A
  coordinator core stages input into the *value-output* DRAM tensor and generates
  the index tensor beside it; workers then read/write those output buffers in place
  each stage (`reader_single_row_multi_core.cpp:30-33`,
  `sort_program_factory.cpp:1554-1556`). Three semaphores sequence the stages
  (`:1537-1539`). The DRAM round-trip is real; the "scratch buffer" is not.

### 1.2 `ttnn.topk`

**Single core** (`topk.cpp`) — **VERIFIED** sliding-window insertion merge over
`Kt = ceil(K/32)` double-buffered tiles (`:128-129, 136, 163-171`). Steady state
reads one tile and merges against the current worst (`:303-305, 352`). Serial
`O(Wt * Kt)` (`:231` x `:266`). Final K slice is **host-side** (`reduction/topk/topk.cpp:115-124`),
not in the kernel.

**Multi-core** — **VERIFIED** constraints, all of which must hold:
- `K <= 64` (`topk_device_operation.cpp:75`)
- **`W >= 8192`** — `multi_core_min_width` (`topk_constants.hpp:11`). **The original
  document said 2048. That is a 4x error on the most important dispatch boundary in
  the subsystem**: everything in 2048-8191 runs the serial single-core path.
- `W` power of two (`:72`); `W < 65535` (`:70`, `< numeric_limits<uint16_t>::max()`)
- **`verify_multi_core_cost`** (`:98-107`, `topk_utils.cpp:86-161`) — exact
  divisibility (`rem == 0`), split `>= 64` (`min_dim_per_core`), contiguous
  rectangular grid, `num_cores > 1`, per-core L1 fit. **In practice this rejects more
  shapes than the width gate.** The original document omitted it entirely.

Stage 2 aggregates on a single final core; `final_input_size = num_cores * max(k, 32)`
(`topk_utils.cpp:154`).

### 1.3 MoE Gates

**`moe_grouped_topk`** — **VERIFIED**, and the original description was wrong.
Grouped path is hardwired to DeepSeek-V3 geometry: `N = 256` in `G = 8` groups of 32;
each group scored by the sum of its **top-2** experts (`summed_experts_per_group == 2`,
`moe_grouped_topk_device_operation.cpp:92`); the **top-4 groups** survive
(`topk_groups == 4`, `:97`); 8 experts activated (`:101-105`). **The original said
"top-2 groups", conflating `topk_groups` with `summed_experts_per_group`.** An
un-grouped path (`n_groups == 1`) accepts any tile-aligned expert count with `K <= 64`
(`:72-87`) and is the flexible one.

**`moe_hash_gate`** — **VERIFIED** frozen `tid2eid[input_ids[token]]` DRAM lookup in
the reader (`reader_moe_hash_gate.cpp:82-113`); no compare-exchange network anywhere.
Normalization is **linear** (`s_i / sum s`), not softmax (`moe_hash_gate.cpp:29-41`).

**`generalized_moe_gate`** — **VERIFIED**, supports **E = 256 or 512 only**
(`h*w == 256` per block, `num_blocks in {1,2}`, `generalized_moe_gate_device_operation.cpp:89, 135-144`).
**The original claimed "N in [128, 1024]"** — 128 is below one block, 1024 is double
the maximum. `topk` is a compile-time parameter restricted to `{4, 6, 8}` with a
`static_assert` (`ckernel_sfpu_..._single_face.h:651`), which is the correct pattern:
fail loudly rather than silently mis-route.

### 1.4 LLK

**Index tracking** — **VERIFIED with correction.** `ENABLE_DEST_INDEX` is an
**ISA-documentation name**; it does not exist as a symbol in `tt_llk_blackhole/`
(only in the Quasar LLK). On WH/BH the mechanism is LaneConfig bit 2 via
`_sfpu_load_config32_(0xF,0x0,0x4)` (`ckernel_sfpu_topk.h:1015`) or
`TTI_SFPCONFIG(0x4, 0xF, 1)` (`..._single_face.h:551`). It carries **two documented
errata**: an LREG4-7 hardware bug forcing a disable/re-enable bracket around
`SFPSHFT2` (`:351`), and TEN-2932 — with tracking enabled, a result written to
LREG4-7 is replaced by the tracked index (`:592-599`). **"Zero overhead companion
indexing" does not survive these.** And `SFPSWAP` is a 2-cycle instruction, so the
index swap is free in *instruction count*, not "in the same clock cycle."

**`topk_xl` / `topk_large_indices`** — **VERIFIED.** Fused word
`[BF16 value b31..b16 | UINT16 index b15..b0]` with `InstrModLoadStore::INT32`
(`ckernel_sfpu_topk_xl.h:23-32`). The 5-slot MOP template firing 34 instructions per
`TTI_MOP` is real (`:62-73, 149-190`). **It has a shipping op** —
`ttnn.experimental.topk_large_indices`, Blackhole-only, `k in [16, 2048]` — which the
original never mentions. **UNSUBSTANTIATED:** the "50% traffic reduction" claim. It
halves *Dst words per element*; the measured merge-body saving is 16 vs 18
instructions (`:131-132`), ~11%.

### 1.5 Real Inefficiencies

| Component | Root cause | Status |
| :--- | :--- | :--- |
| `ttnn.sort` single-core | Fixed `2*Wt` transpose bracket per row (entry + exit). Merge network already transpose-free. | Low priority. **Not measured.** The original's "~18%" is invented. |
| `ttnn.sort` multi-core | Every bitonic stage round-trips tile pairs through the DRAM output buffers under coordinator lockstep. | **Real opportunity.** The only place the DRAM-spill premise holds. |
| `ttnn.topk` single-core | Serial `O(Wt*Kt)`; taken for all `W < 8192` and anything `verify_multi_core_cost` rejects. | **Real opportunity**, two ways: a cheaper per-tile path, *and* relaxing the cost model. |
| MoE gates | Score gather runs in the writer, normalization in compute — scores round-trip through L1 CBs. | Real, modest. "L1 fragmentation" is unsupported. |
| UINT16 mode-9 pack | Identical on **WH and BH** (`ckernel_sfpu_topk.h:107`/`:111`); no-op only on **Quasar** (`:709`). | **Dead code today** — the gating macro `TOPK_UINT16_FP32_DEST` is defined nowhere in-tree. Wire it up or delete it. The original claimed BH has a native fix; it does not. |

**Missing from the original audit entirely:** `ttnn.moe`; the `stable=True` split
(`ttnn.sort` hard-fails at `sort.cpp:247`, `ttnn.topk` supports it but Quasar
`static_assert`s it false); dtype/index-width coupling (FP32 and UINT16 inputs force
`fp32_dest_acc_en` and therefore UINT32 indices, doubling index CB footprint); and
roughly half the selection kernels in `sfpu/experimental/`.

---

## 2. ISA and Roofline

Cited to `tt-isa-documentation`. **Wormhole and Blackhole have separate trees on
purpose.** There is no Quasar tree, so every Quasar parameter is UNVERIFIABLE.

### 2.1 Hardware Parameters

| Parameter | Wormhole B0 | Blackhole A0/A1 | Source |
| :--- | :--- | :--- | :--- |
| Nominal Tensix clock | 1.00 GHz | 1.35 GHz | `BlackholeA0/.../VectorUnit.md:6`; `BlackholeA0/NoC/README.md:44` |
| **Is the clock a constant?** | **No** — AICLK is DVFS/thermal-managed, queried per device | **No** | `tt_metal/llrt/tt_cluster.cpp:757` (`get_device_aiclk`). tt-metal has **no** HAL constant fixing it. |
| Worker cores, **full die** | 80 (8x10) | 140 (14x10) | UMD `{wormhole,blackhole}_implementation.hpp` `TENSIX_GRID_SIZE` |
| Worker cores, **shipping part** | n150: usable **8x8 = 64**; n300: usable **8x7 = 56** (Tensix dispatch) | P100/P150/P300: 2 columns harvested, usable **11x10 = 110** (120 w/ ETH dispatch); BH Galaxy 13x10 = 130 | UMD `cluster_descriptor_types.hpp:237-244`; `tt_metal/core_descriptors/blackhole_140_arch.yaml:63-65` |
| SFPU lanes | 32, as a **4x8 grid** (4 rows x 8 cols) | same | `WormholeB0/.../VectorUnit.md:3`, `LReg.md:30-35` |
| `LReg` file | `LReg[0..7]` general; `[8..10]` read-only consts; `[11..14]` `SFPCONFIG`-only; `[15]` lane index; `[16]` `SFPLOADMACRO`-only | **identical** — BH page states so explicitly | `WormholeB0/.../LReg.md:6-16`; `BlackholeA0/.../LReg.md:3` |
| CC flag stack | 8 entries, per lane | same | `SFPPUSHC.md:3` (both trees) |
| `Dst` capacity | 1024 rows x 16 cols of 16-bit = **32 KiB** | same | `Dst.md:3-7`; `dst_size_alignment: 32768` in both soc descriptors |
| L1 per core (physical) | **1464 KiB**, 16 banks x 91.5 KiB | **1536 KiB**; bank organization **NOT DOCUMENTED** | `WormholeB0/TensixTile/L1.md:3,5` |
| Aggregate L1 BW | **256 B/cycle** = 256 GB/s | **UNVERIFIABLE** — BH lists "more L1 bandwidth" as a distinct upgrade | `L1.md:5`; `BlackholeA0/TensixTile/README.md:19` |
| Unpackers | 2. **One alone = 64 B/cyc.** Both = 80 B/cyc. **Only Unpacker 0 can write `Dst`.** | 2; rates NOT DOCUMENTED | `L1.md:37`; `Dst.md:78` |
| Packers | 4, 64 B/cycle aggregate | 4; rates NOT DOCUMENTED | `L1.md:43` |
| **NoC flit width** | **256 bits (32 B)** | **512 bits (64 B)** | `WormholeB0/NoC/README.md:3`; `BlackholeA0/NoC/README.md:3,44` |
| **Peak NoC BW / link** | 32.0 GB/s | **86.4 GB/s** | derived from above; `BlackholeA0/NoC/README.md:64-66` |
| Router-to-router hop | 9 cycles | 9 cycles | NoC perf tables, both trees |
| NIU-to-router legs | ~5 in, ~5 out | same | same |

Corrections to the original table: **BH flit is 64 B not 32 B** (so per-link peak is
86.4 GB/s, not 43.2 — a 2x error); `LReg` has **no documented banking** on either arch
("banks" in the ISA docs refer only to SrcA/SrcB); the "+ Staging Reg" is **not** a BH
addition (`LReg[16]` exists identically on both); "8 slices x 4 rows" inverts the
layout and uses a word that appears **zero times** in the ISA docs; Quasar L1 "2048 KiB"
is contradicted by `quasar_32_arch.yaml` (4096 KiB).

> **Topology.** Each NoC is a 2-D **torus**, not a mesh (`BlackholeA0/NoC/README.md:5`).
> Max hop distance is `floor(X/2) + floor(Y/2)` — **12 hops** across the BH worker span,
> 14 over the full NoC grid. The original's "7 hops" is wrong.

> **Never hardcode a core count.** Query `compute_with_storage_grid_size()`
> (`device.hpp:112`) or `logical_grid_size()` (`:91`).

### 2.2 Instruction Timing

- **VERIFIED `SFPSWAP`: 2-cycle latency, sustained IPC 0.5.** After an `SFPSWAP` the
  only instruction the SFPU accepts next cycle is `SFPNOP`; anything else auto-stalls
  the thread (`SFPSWAP.md:110`). **Two corrections to the original:** the bubble is
  inserted by *hardware* — software need not write `SFPNOP` — and it **cannot** be
  filled with an independent ALU op from the same thread. In `VEC_MIN_MAX` mode one
  `SFPSWAP` performs 32 lanewise compare-exchanges, so peak is **16 CAS/cycle**.
- **VERIFIED `SFPTRANSP`: 1 cycle, IPC 1.0**, no scheduling restriction. It is a
  **Simple**-column instruction, same column as `SFPSWAP` — **they cannot dual-issue.**
- **VERIFIED `SFPLOAD`/`SFPSTORE`: IPC 1**, moving 32 datums per instruction.
- **VERIFIED `SFPLOADMACRO` peak is 5 instructions/cycle** — and **inapplicable to a
  swap-bound sort.** If `SFPSWAP` is scheduled to Simple, `SFPNOP` must go to MAD the
  same cycle and both Simple and Round must idle the next (`SFPLOADMACRO.md:11`). A
  bitonic inner loop gains essentially nothing.
- **VERIFIED `SFPGT`/`SFPLE` are new in Blackhole** (`VectorUnit.md:6`). Any Wormhole
  threshold path must emulate via `SFPSETCC`/`SFPMAD`.

The original's pipeline diagram invented separate "Swap Datapath", "Transpose/Shift",
and "Round EXU + PRNG" units. The real sub-units are **load, simple, MAD, round,
store**, and the SFPU "can only accept one instruction per cycle from the outside
world, so by default four fifths is always idle" (`VectorUnit.md:99`).

### 2.3 Roofline

$$R_{\text{CAS}} = 32 \text{ lanes} \times 0.5 \text{ IPC} = 16 \text{ CAS/cycle}$$
$$T_{\text{unpack}} = T_{\text{pack}} = S_{\text{tile}} / 64 \text{ B/cycle} = 32 \text{ cycles/tile (BF16)}$$

Unpack and pack are separate hardware and overlap in a pipelined stream, so combined
traffic peaks at **128 B/cycle**. With `I_sort` defined over two-way traffic, the knee is

$$I_{\text{crit}} = 16 / 128 = \mathbf{0.125 \text{ CAS/byte}}$$

*(The original divided a two-way intensity by a one-way 64 B/cycle and got 0.25. Pick
one convention; they differ by 2x.)*

**Where the workloads actually sit — the original never checked this:**

| Workload | `I_sort` | Regime |
| :--- | ---: | :--- |
| Full bitonic sort, 1 tile | 28,160 / 4096 = **6.88** CAS/B | **Deeply compute bound**, 55x past the knee |
| Streaming threshold filter | 1 / 2 = **0.5** op/B | **Exactly balanced** — unpacker and `SFPGT` both 64 B/cyc |

**Consequence: none of these kernels are bandwidth bound.** The 32-cycle unpack/pack
terms are under 2% of the compute term and overlap with it anyway. Every proposal must
be measured against the **compute** roofline, not against the current implementation.

### 2.4 Execution Models

**Model A — single-tile bitonic sort (N = 1024).** Structure is right: 10 phases,
`10*11/2 = 55` steps, `55 * 512 = 28,160` CAS. **The phase cycle counts are asserted
with no derivation and are impossible under the document's own peak:**

| Phase group | Steps | CAS | Floor @ 16 CAS/cyc | Original claim |
| :--- | ---: | ---: | ---: | ---: |
| 1-4 | 10 | 5,120 | **320** | 136 |
| 5-6 | 11 | 5,632 | **352** | 220 |
| 7-10 | 34 | 17,408 | **1,088** | 480 |
| **Total** | **55** | **28,160** | **1,760** | **836** |

836 cycles for 28,160 CAS implies **33.7 CAS/cycle** — 2.1x the 16 CAS/cycle derived
two paragraphs earlier. Corrected floor: `32 + 1760 + 32 = ` **1,824 cycles**, versus
the claimed 900.

Two costs are still excluded: **`Dst` spills** (the `LReg` working set is 256 datums
against a 1024-datum tile, so >=4 residency windows) and **cross-lane movement** —
**40 of the 55 steps have stride <= 16**, requiring a preceding `SFPTRANSP` or
`SFPSHFT2`. Phases 1-4, which the original priced most aggressively and labelled
"Zero Dst Spill", are made entirely of these and are the *least* defensible number in
the section. `[UNVERIFIED ESTIMATE]` realistic range **2,300-2,800 cycles/tile**.

**Model B — streaming top-K (K <= 32, N = 32k).** Two bounds, and the binding one is
the SFPU, not the unpacker:

```
SFPU (correct count):  32,768 / 32 lanes x 2 cyc/vector = 2,048 cycles   <-- BINDING, by 4x
unpack (BF16 @ 128 B/cyc): 32,768 x 2 B / 128 B/cyc     =   512 cycles
unpack (FP32 @ 128 B/cyc): 32,768 x 4 B / 128 B/cyc     = 1,024 cycles
```

**Blackhole unpacker = 128 B/cycle**, double Wormhole's 64 B/cycle (L1 to SrcA/SrcB).
*Source: Tenstorrent hardware figure supplied by the author.* It is **not** stated in
`tt-isa-documentation`: the only unpacker throughput figure in either tree is the
Wormhole one (`WormholeB0/TensixTile/L1.md:37`, four 128-bit reads/cycle), the Blackhole
`L1.md` is a three-line redirect to it, and the only Blackhole statement on the subject
is the qualitative *"more L1 bandwidth"* (`BlackholeA0/TensixTile/README.md:19`).
Treat this as an authoritative external fact pending a doc update or a measurement.

**Consequence — this reverses an earlier conclusion in this document.** A previous
revision claimed threshold selection "sits exactly on the unpack roofline". It does not.
At 128 B/cycle the unpacker delivers a 32-lane vector in **1 cycle (FP32)** or **0.5
cycles (BF16)**, while a correct `SFPGT` + `SFPIADD` count costs **2 cycles**. Selection
on Blackhole is **SFPU-issue-bound by 2x (FP32) to 4x (BF16)** — decisively compute-
bound, not bandwidth-bound.

Two things follow. First, there **is** real headroom, but it is only reachable by cutting
SFPU instructions per element; data-movement tricks (better CB sizing, sharding, prefetch)
cannot help a kernel that is 4x off its bandwidth bound. Second, any future selection
design must be judged against **2,048 cycles**, not against the unpack floor, and the gap
to 512 is the prize — not evidence that the design is already optimal.

**Why 2 cycles per vector and not 1 — VERIFIED, and it corrects an earlier revision of
this section.** A threshold count needs `SFPGT` (Simple sub-unit) plus an accumulate.
`SFPLOADMACRO` cannot host the accumulate: a macro-scheduled instruction may only write
`macroVD` -- overwritten by the very next macro's load -- or `LReg[16]`, which is *"only
readable by `SFPSTORE` instructions scheduled via `SFPLOADMACRO`"* (`LReg.md:16`) and
whose index is not encodable in a 4-bit operand field. **`SFPLOADMACRO` expresses maps
at 1 cycle/vector; it cannot express register reductions, and counting is a reduction.**
Consistent with the only 1-cycle macro user in tree, `ckernel_sfpu_mul_int.h:36-41`,
which is a pure feed-forward Load + MAD + Store map.

Nor can the accumulate move to the MAD column: the `SFPGT` mask is the bit pattern
`0xFFFFFFFF`, which is a NaN in FP32 and permanently poisons any float accumulator
(`SFPMAD.md:60`); and the only integer MAD-column op, `SFPMUL24`, is a pure multiply
whose add path is explicitly `NonContractualBehavior` (`SFPMUL24.md:3`). `SFPIADD` would
work but is itself a **Simple** instruction, colliding with `SFPGT`.

The correct formulation is therefore `SFPLOADMACRO`(Load + `SFPGT`) plus a
**software-issued** `SFPIADD` on the following cycle -- two issue slots per vector.
Issue bandwidth is the binding constraint (the SFPU accepts one software instruction per
cycle, `VectorUnit.md:112`), so unrolling does not help.

**OPEN -- and it decides whether 2 cyc/vector is even a cost.** No datums/cycle or
bytes/cycle figure for the Blackhole unpacker exists anywhere in
`tt-isa-documentation/BlackholeA0/`; the 64 B/cycle above is the *Wormhole* figure
(`WormholeB0/TensixTile/L1.md:37`), and the BH soc descriptor records only
`unpacker: version: 2`. If BH unpack is 64 B/cycle, an FP32 32-lane vector is 128 B and
takes 2 cycles to unpack -- the correct kernel would be exactly unpack-bound and nothing
is lost. At 16-bit input a vector is 64 B and the SFPU becomes a 2x bottleneck.
**Measure the unpacker rate before re-deriving this floor; the answer flips the
conclusion.**

2,048 cycles is the number every streaming proposal must beat. See §5.3 for what the
original proposal scored against it.

#### MEASURED ON SILICON — the 2 cyc/vector figure is no longer a derivation

Measured on the Blackhole in this machine, via the tt-llk perf harness
(`tests/sources/sfpu_count_above_perf.cpp`, `python_tests/perf_sfpu_count_above.py`),
MATH_ISOLATE, two-point slope over ITER_COUNT ∈ {512, 2048} to cancel the profiler
marker overhead, 5 runs per point:

| Arm | What it issues per vector | cyc/vector | elem/cycle |
| :--- | :--- | ---: | ---: |
| Feed floor | recorded `SFPLOAD`, MOP-driven | **1.000** | 32.0 |
| `SFPSWAP` control | known-2-cycle instruction | **2.000** | 16.0 |
| Count inline ("D1") | `SFPLOADMACRO`(Load+`SFPGT`) + software `SFPIADD` | **1.998** | 16.0 |
| 3-sub-unit probe | `SFPLOADMACRO`(Load+`SFPGT`+`SFPMAD`) | **1.002** | 31.9 |
| **Mask filter ("D2")** | `SFPLOADMACRO`(Load+`SFPGT`+`SFPSTORE`) | **1.003** | **31.9** |

**The two macro arms are the result that matters.** Adding a Simple (`SFPGT`) and
either a MAD or a Store to the macro costs **nothing** — 1.002 and 1.003 against a
1.000 bare-load floor. Three sub-units genuinely co-issue, exactly as
`SFPLOADMACRO.md:13` claims. So:

- A **filter that materializes a mask tile** (`Load + SFPGT + SFPSTORE`, a *map*)
  runs at **1 cycle / 32-element vector = 32 elements/cycle**.
- The 2 cycles of the inline-count form are **entirely** the software-issued
  `SFPIADD` — the *reduction*, not the compare. Reductions cannot be macro-scheduled
  (destinations restricted to `macroVD` or write-only `LReg[16]`), so they cost a
  second issue slot.
- The free MAD slot means a **fused** gate (compare on Simple, scale/bias/exp on MAD)
  is also free relative to a bare load. That is the lever for MoE gating.

**Revised floor.** For a mask-materializing filter the N=32k figure is **1,024
cycles**, not 2,048. And the roofline verdict flips by data type:

| | unpacker delivers | SFPU filter consumes | verdict |
| :--- | ---: | ---: | :--- |
| FP32 | 32 elem/cyc (128 B/cyc) | 32 elem/cyc | **exactly balanced — on the roofline** |
| BF16 | 64 elem/cyc (128 B/cyc) | 32 elem/cyc | SFPU-bound by 2x |

An earlier revision of this section said selection was SFPU-bound by 2x-4x. That was
correct for the inline-count form and wrong for the mask form, which is the one a real
filter would use.

Preceded by `test_profiler_overhead.py` passing on the same device, which pins the
marker pair at 30 ± 5 cycles. The `SFPSWAP` arm is the methodology control: its 2.000
was predicted from `SFPSWAP.md:110` alone, so it confirms the harness measures SFPU
retirement rather than RISC-V instruction pushes.

**Correctness verified on the same loop**, not merely throughput
(`tests/sources/sfpu_count_above_test.cpp`, `python_tests/test_sfpu_count_above.py`):
13 device cases pass, including exact all-above assertions (`count == 1024` and
`== 4096`) that are the only thing capable of catching the silent-discard hazard of
`SFPLOADMACRO.md:149`, a per-lane accumulator driven to 131072 to expose any 16-bit
truncation, and the sign-magnitude total-order cases where `SFPGT` and IEEE disagree
(threshold `-0.0` vs data `+0.0` → 1024, not 0; `+Inf` vs `+NaN` → 1024, not 0).

**MOP/REPLAY is a prerequisite, not an optimization.** The Tensix frontend dequeues at
most one instruction per thread per cycle (`PushTensixInstruction.md:19`), so a
RISC-V-driven selection loop is frontend-saturated at 2 cycles/vector with zero margin.
Recording the body and driving it from a MOP takes the RISC-V off the critical path;
that is what makes the 1.000 floor reachable at all. Rule of thumb: a sequence averaging
>1 backend cycle per instruction (an `SFPSWAP` lattice) has frontend slack and gains
nothing from replay; one averaging ~1 (`SFPGT`, `SFPLOAD`, `SFPIADD`, `SFPSTORE`) is
frontend-bound and requires it.

#### MEASURED: the shipping micro-ops, same harness, same units

All on Blackhole, MATH_ISOLATE, two-point slope, 5 runs/point, with the same
`SFPLOAD`/`SFPSWAP` control pair carried in the same translation unit. The
`SFPSWAP` control landed at exactly **2.00x** the load floor, so the measurement
is trustworthy. `tests/sources/topk_micro_op_perf.cpp`, `perf_topk_micro_op.py`.

| Kernel | cyc / 32-element vector |
| :--- | ---: |
| CONTROL `SFPLOAD` (frontend floor) | 1.000 |
| CONTROL `SFPSWAP` (known 2 cycles) | 2.000 |
| **Candidate: mask filter (`Load+SFPGT+SFPSTORE`)** | **1.003** |
| Candidate: inline count (`+ software SFPIADD`) | 1.998 |
| `_topk_xl_merge_`, K=512, fused | **2.844** |
| `bitonic_top8_ph0_to_ph3` (MoE gate micro-op), bare | **11.000** |
| `bitonic_top8_ph0_to_ph3`, with load/store envelope | 15.500 |
| `_bitonic_topk_merge`, k=32 | 10.615 |
| `topk_local_sort`, end_phase 4 | 47.784 |
| **`topk_local_sort`, end_phase 5 — what `ttnn.topk` uses** | **76.195** |

**Verdict, by opponent.** A threshold selector needs at least two full passes
(count, then compact), so its floor is 2 x 1.003 ~= 2.0 cyc/vector.

- **vs `ttnn.topk` (`topk_local_sort` @ 76.2): the filter wins by a landslide**,
  at every N and K. In situ the shipping op consumes 4.76 cyc per *newly-consumed
  element* (its 64-wide window is only half new data), a ~76-pass budget. Even a
  20-pass threshold scheme is 4x ahead.
- **vs the MoE gate micro-op (11.0 bare / 15.5 in situ): marginal.** At N=256,
  k=8 the gate costs 188 cycles in situ; the filter costs 84 + 16P, so it wins
  only if it converges in <= 6 passes, and below N ~= 105 its own 84-cycle
  replay/MOP setup exceeds the entire micro-op. The gate is also already at its
  instruction floor: 18 `SFPSWAP` x 2 + 7 `SFPTRANSP` x 1 predicts 43 cycles,
  measured 44.
- **vs `_topk_xl_merge_` (2.844): the filter LOSES outright.** Two passes cost
  4.0 against 2.844, at every N and k. Not fixable by tuning -- `topk_xl` is
  already MOP/replay-fed at ~1.3 cyc per SFPU instruction, i.e. near the same
  frontend floor the candidate exploits, and it reduces 1024->512 in that one
  pass. **Large-K selection should keep the bitonic merge.**

#### The most valuable finding is a fix to the shipping op, not a new algorithm

`topk_local_sort` and `topk_merge` are **RISC-V-issue-bound, not SFPU-bound**:
~2.4 cycles per SFPU instruction against a ~1.1 backend-bound floor, because
their load/store helpers use runtime-encoded `TT_SFPLOAD`/`TT_SFPSTORE` rather
than replay-recorded bodies.

The proof is clean: setting `STABLE_SORT=Yes` adds one `TTI_SFPSWAP` per inner
iteration (32 per call) and changes the measured slope by **exactly 0.000** --
only a fixed +20 cycle intercept. Adding real SFPU work costs nothing, so the
SFPU is not the constraint.

Two consequences, both cheap and both independent of the filter:

1. **Replay-record the load/store bodies**, exactly as `ckernel_sfpu_topk_xl.h`
   already does. This recovers most of the gap without changing the algorithm,
   and benefits every existing caller.
2. ~~Stop hardcoding `end_phase`.~~ **TESTED AND INVALID — retracted.** See below.

#### RETRACTED: the `end_phase = log(K)-1` "free win" does not exist

An earlier revision of this document proposed deriving `end_phase` from K in
`topk.cpp:154`, citing `compute_kernel_api.h:706` ("should be set to log(K)-1").
**Tested on silicon: it breaks correctness for every K.**

| `end_phase` | configs exact (of 54) |
| ---: | :--- |
| 5 (shipping) | **54/54** |
| 4 | **0/54** — including K=32, the case the formula most directly prescribes |
| 3 | 0/54 |
| 2 | 0/54 |

**Why, and it was predicted before testing.** The doc's "K" is the *sorted-subsequence
length the caller wants out of the 64-datum window*, not the user's top-k. Phase `p`
leaves sorted runs of length `2^(p+1)`. The **multi-core** path (`topk_local.cpp:124`)
legitimately stops at length-k runs because it follows up with `topk_merge` and
`topk_rebuild` — the partial sort is input to a merge stage. The **single-core** path
contains **no `topk_merge` and no `topk_rebuild` anywhere in the file**;
`topk_local_sort` is its only sort call, used as a full 64-element merge-sort. Its
invariant (`topk.cpp:349-351`) is "DST0 = top 32, DST1 = bottom 32", which only a
complete 64-element sort produces: phases 0-4 sort the accumulator half and the fresh
half into an opposed bitonic pair, and **phase 5 IS the merge**. Drop it and the halves
are never compared, so no new element can ever enter the running top-K.

So `end_phase = 5` is correct *and* consistent with the doc: for this caller the
subsequence length is 64, and `log2(64)-1 = 5`. The failure signature confirms it — the
match fraction scales as ~1/k and falls to 0.000 as W grows, i.e. only elements that
happened to land in the first two tiles survive. Total loss of the merge, not a
boundary artefact.

**The doc line is the real defect** (`compute_kernel_api.h:706`). It should say that K
is the sorted-subsequence length the caller wants, which equals the user's top-k only
when the caller follows with `topk_merge`/`topk_rebuild`; a standalone 64-element
merge-sort must pass 5. Note the stated valid range is 1-5, so `log2(2)-1 = 0` for k=2
is out of range anyway.

#### The `end_phase` cost curve, and where the real headroom is

Measured, MATH_ISOLATE, controls exact (`CtrlLoad` 1.000, `CtrlSwap` 2.000):

| `end_phase` | cyc / 32-elt vector (desc, unstable) | vs ep5 |
| ---: | ---: | ---: |
| 0 | 5.880 | 12.96x |
| 1 | 12.495 | 6.10x |
| 2 | 20.620 | 3.70x |
| 3 | 29.996 | 3.70x |
| 4 | 47.784 | 1.59x |
| **5 (shipping)** | **76.195** | 1.00x |

The curve does expose genuine waste, just not the kind a constant can capture. In the
steady state DST0 **already holds a sorted 32-run from the previous iteration**, yet
phases 0-4 re-sort it from scratch; only DST1 (the fresh input tile) needs them. Phase 5
alone costs `76.195 - 47.784 = 28.41` cyc/vector, so phases 0-4 are 47.78 of the 76.20.
Scoping phases 0-4 to a single DST tile would cost roughly half of that, giving
**~52.3 vs 76.2 = ~1.46x**. That needs a new LLK entry point (the current API applies
every phase across all 64 datums), not a kernel constant.

#### MEASURED: packer zero-compression works on Blackhole — filter AND compaction with zero SFPU instructions

Verified on silicon, 35/35 tests, bit-exact round-trip through a software decoder.
`tests/sources/pack_zero_compress_test.cpp`, `python_tests/test_pack_zero_compress.py`.

**Enablement is one config bit.** `THCON_SEC0_REG1_Disable_zero_compress = 0` plus a
reserved `Row_start_section_size`. The LLK's `set_packer_config` currently forces it
*off* unconditionally (`config.f.uncompress = 1`). Nothing in the tree has ever used it.

| pattern (1024 bf16 datums) | survivors | packed bytes | decodes exactly |
| :--- | ---: | ---: | :--- |
| compression off | — | 2048 | yes |
| 32 scattered survivors | 32 | **384** (256 with `Concat`) | yes |
| all-zero | 0 | 304 (256) | yes |
| dense | 1024 | 2624 (+28%) | yes |

**And it composes with the packer's own threshold.** `MIN_THRESHOLD_RELU` zeroes
sub-threshold datums, compression then elides them — so **filter + compaction happen in
a single `PACR` sequence with no SFPU instructions at all**. Measured: 31 survivors →
384 B, 63 → 384 B, 127 → 544 B, each matching a golden. The SFPU is left entirely free,
and this runs on the packer concurrently with whatever the SFPU is doing next.

**The hard limit: ~16:1 per pass.** The maximum stride between emitted datums is 16, so
`N_aug >= K + ceil((N-K)/16)` regardless of sparsity. For bf16 that floors the rate at
~0.156 B per source element — **12.8x**, not the 32x a perfect compaction of 32/1024
would give. Full compaction therefore needs a **cascade of ~log16(N/K) passes**
(measured rule projects 1024 -> 88 -> ~36 -> ~33 ~= K). The cascade itself is analysis,
not yet measured.

**Format confirmed, with one divergence from the docs.** The layout matches
`WormholeB0/.../Packers/Compression.md` exactly — `uint16` row-start-index array, then
groups of 32 augmented datums followed by 32 four-bit counters. But **on Blackhole the
counter is the number of zeroes PRECEDING its datum, not following it.** A decoder
written to the documented "after" semantics is bit-perfect on symmetric patterns
(all-zero, dense, front-loaded) and garbage on asymmetric ones — a trap worth knowing.
The doc itself declines to specify the compressing side (`Compression.md:31`).

**`Downsample_mask` is a real vector-compress on BH but is NOT usable here.** Verified:
mask `0x5555` emits every even index, `0x0001` every 16th. But the mask is a static
backend config field with period 16 — it cannot depend on the data, and `PACR_SETREG`
cannot reach the backend CONFIG file. Making it data-dependent would cost an `RMWCIB`
plus a config stall per 16 datums, far worse than the 1.003 cyc/vector SFPU filter.

**Two hazards found, both worth fixing upstream:**

1. **`Downsample_mask` is a config escape that survives ELF reload.**
   `set_packer_config` writes `THCON_SEC0_REG1` words 0 and 2 but deliberately skips
   word 3 ("removed word 3 to avoid potential race condition") — and word 3 holds
   `Downsample_mask`. A mask left behind by an earlier kernel silently decimates every
   subsequent pack. This was observed live: downsample probes poisoned the packer and
   later baseline runs quietly packed half a tile.
2. **`TT_PACR` (runtime-issued) hung the packer where `TTI_PACR` with byte-identical
   field values did not.** `TT_PACR` is used nowhere in tt-llk. Observed, not root-caused.

Also unresolved: `Packers[0].AllZeroFlags` (`0xFFB1_1020`) read `0x00000000` in every
configuration including an all-zero tile, via two independent read paths. So per-row
emptiness is **not** available for free as hoped.

#### MEASURED: 2.0 cyc/vector is an ARCHITECTURAL FLOOR for SFPU counting

Four more arms, measured on silicon, every control on its predicted value across a
device reset:

| arm | cyc/vector | what it settles |
| :--- | ---: | :--- |
| `ReplayLoad` (control) | 1.000 | frontend floor |
| **`MacroExp`** | **1.000** | a macro-scheduled `SFPEXEXP` is **free** |
| `MaskStore` | 1.002 | filter / map = 1.0 |
| `CountD1` | 1.997 | count = 2.0 |
| `ReplaySwap` (control) | 1.999 | tripwire, matches `SFPSWAP.md:110` |
| **`MultiPass`** | 2.097 | blind pass restart = **6.4 cyc** |
| **`PassSync`** | 2.389 | data-dependent restart >= **25.1 cyc** |
| **`HistNibble`** | 5.000 | 8-bucket exponent histogram |

**Why 2.0 cannot be beaten by an SFPU-only count — and the cause is NOT what an earlier
revision of this document said.** It is not the per-thread frontend dequeue rate; that is
a red herring, since a *per-thread* limit would have permitted two threads at 1/cycle
each. The real cause is that **the SFPU has a single shared issue port at the frontend
mux, global across all three Tensix threads**:

- `Diagrams/Src/TensixFrontend.lua:126-136` annotates `uops = 3` on exactly the
  per-thread-replicated units (Sync, Configuration, Miscellaneous) and leaves Vector Unit
  (SFPU), Matrix Unit (FPU), Packers, Unpackers, Scalar Unit and Mover at the default
  **1**. All three per-thread Wait Gates feed one mux.
- `WaitGate.md:5` names the exact stall: *"Multiple threads are wanting to dispatch an
  instruction to the same backend execution unit, but the unit can only accept one
  instruction per cycle (so one thread will dispatch, and the others will wait)."*
- `SFPLOADMACRO.md:13`: *"`SFPLOADMACRO` is the **only** mechanism for attaining more than
  one instruction per cycle"* — unqualified, not "from a single thread". Macro-scheduled
  sub-unit ops are generated *inside* the SFPU and never traverse the mux; software-issued
  ops, from any thread, always do.

**So a two-thread split cannot break the floor.** MATH issuing the macro stream and a
second thread issuing the accumulates still presents 2 instructions/vector to a 1/cycle
port. The second thread does not add a dispatch slot, it takes one 1:1 — best case a wash,
realistically worse from arbitration jitter. Cross-thread synchronization is independently
fatal too: `SyncUnit.md:5-9` throughput-limits `SEMPOST`/`SEMGET`/`SEMWAIT`/`STALLWAIT` to
one per cycle *globally*, so a per-vector handshake costs >= 2 cyc/vector before any SFPU
work. (Note `ATGETM`/`ATRELM` are 3/cycle by contrast.) And unsynchronized is worse than
slow: `SFPGT` and `SFPIADD` are both Simple sub-unit, so a cross-thread collision is a
**silent undercount** per `SFPLOADMACRO.md:149`.

The remaining structural facts still hold: a macro schedules **at most one
Simple instruction** (`SFPLOADMACRO.md:5`), so compare and accumulate cannot share one;
and a macro-scheduled result lands in `macroVD` (clobbered by the next load) or
`LReg[16]`, readable only by a macro-scheduled `SFPSTORE` — never an ALU input
(`:112,:120`). **No macro-resident accumulator can accumulate onto itself.** So any
SFPU-only count is >= 2 instructions/vector, and `CountD1` at 1.997 is already optimal.
What the macro *can* do at 1.0 is an arbitrary 4-deep **map**; reduction is the one
thing it cannot express.

**A validation subtlety worth copying.** `SFPLOADMACRO.md:88-94` silently rewrites an
illegal instruction template to `SFPNOP` — which also measures 1.000. So the free
`SFPEXEXP` had to be proven by *correctness*, not timing: four cases where a dropped
`SFPEXEXP` returns 1024 instead of the golden. Timing alone cannot distinguish "free"
from "silently not executed".

**Both reduce-offloads LOSE, and one of my earlier claims was wrong.**
- Packer L1-accumulate: 128 cyc/tile non-atomic, 320 atomic, against the SFPU pass's
  32 cyc/tile — a 4-10x bottleneck, and elementwise-only.
- FPU reduce: `GAPOOL`/`MVMUL` read SrcA/SrcB and never `Dst`, so it needs 16x
  `MOVD2A/B` plus fences per tile (~50-90 cyc), and `Dst` int32 is sign-magnitude so
  `SFPGT`'s `0xFFFFFFFF` decodes as -(2^31-1).
- **"The FPU is idle, so use it" was wrong as stated.** FPU and SFPU are separate
  backend units that do run in parallel, but they **share one in-order frontend per
  thread**. Issuing ~30 FPU instructions per tile from MATH roughly doubles MATH issue
  against the SFPU's 32. It only helps from a *second* Tensix thread — possible via
  `semaphore::FPU_SFPU`, unmeasured.

**Threshold-search cost, N=32768 K=32** (mandatory filter pass = 1024 cyc):

| strategy | cycles |
| :--- | ---: |
| per-token prior, count fused into the filter pass | ~25 |
| prior + explicit verify (1 pass) | 2,073 |
| subsample 1/32 + confirm | 2,628 |
| full-width 8-bucket histogram, 12 bits | 20,580 |
| full-width binary search, 12 bits | 24,876 |

**The search must be held to ~1 full-width pass or it dominates everything.** The nibble
histogram was measured and loses: 1.67 cyc/bit vs binary search's 2.00 is only 1.2x, and
that is *unclamped* — `SFPSHFT` wraps mod 32 rather than saturating (`SFPSHFT.md:44-50`)
so a safe clamp costs +2-4 instructions and makes it worse than binary search outright.
`SFPIADD` also wraps on overflow, so nibble counters need draining every 15 vectors.

**At N=256, do not threshold — sort.** The 25-cycle data-dependent rendezvous alone
exceeds the entire 16-cycle data pass. The existing bitonic top-8 micro-op does it with
no threshold and no L1 round trip. Estimated crossover **N ~ 4000-8000** (not measured).

#### MEASURED: a four-sub-unit macro at 1.000 cyc/vector, and the honest end-to-end number

**Correcting a misleading figure used earlier in this document.** The "~38x faster than
`ttnn.topk`" claim compared a *single filter pass* (1.003) against a *complete*
`topk_local_sort` (76.195). That is not a like-for-like comparison. A complete
threshold-select needs many passes. **The honest end-to-end figure is ~3x**, derived below.

**The new primitive.** Every previous arm used at most a 2-deep macro. `HistMacro` uses
the full five-instructions-per-cycle corner — Load + Simple + MAD + Round + Store — by
exploiting the fact that `SFPLOADMACRO.md`'s (†) rule (*"if a Simple and a Round
instruction execute on the same cycle, one of them needs `VD == 16`"*) constrains
*counting* but not *mapping*: a Round instruction may legally target `LReg[16]` and exit
through the Store slot.

```
t+0  Load                macroVD = x
t+1  Simple   SFPEXEXP   macroVD = exp(x)
t+2  MAD      SFPMUL24   macroVD = (4*macroVD) & 0x7FFFFF
t+4  Round    SFPSHFT2   LReg[16] = SEED << macroVD      (VD=16 satisfies the (†) rule)
t+5  Store    SFPSTORE   Dst[load addr] = LReg[16]
```

Delays are 1 and 3 because `SFPMUL24` has 2-cycle latency. `macroVD` is live t..t+4, so the
register rotation period must exceed 4 and not divide 4 — **P = 8**, which uses all of
LREG0..7 and forces the loop constants into the programmable-constant file (LReg11/12).

| arm | cyc/vector | vs floor |
| :--- | ---: | ---: |
| `ReplayLoad` (control) | 1.000 | 1.000 |
| `ReplaySwap` (control) | 2.000 | **2.000** — exactly 2.00x, valid |
| `CountD1` | 1.998 | |
| `MaskStore` | 1.003 | |
| `HistNibble` (software histogram) | 5.000 | |
| **`HistMacro`** (new) | **1.000** | predicted 1.000 |
| **`HistSum`** (new) | **2.000** | predicted 2.000 |

**`HistMacro` + `HistSum` = 3.000 cyc/vector for the same 8-bucket exponent histogram
`HistNibble` computes in 5.000.** The entire gain is the three free sub-unit slots:
`HistNibble` issues four software instructions per vector, `HistMacro` issues none.

| scheme | cyc/vector | bits/pass | **cyc per bit** |
| :--- | ---: | ---: | ---: |
| bit-serial binary search (`CountD1`) | 2.000 | 1 | 2.00 |
| `HistNibble` | 5.000 | 3 | 1.67 |
| **`HistMacro` + `HistSum`** | **3.000** | **3** | **1.00** |

**End-to-end threshold-select, bf16 (16 key bits), per 32-element vector:**

| stage | cyc |
| :--- | ---: |
| premap to monotone-integer key | 1.0 |
| sign+exponent, 3x (`HistMacro`+`HistSum`) | 9.0 |
| 7 mantissa bits, bit-serial | 14.0 |
| final filter / emit | 1.0 |
| **total** | **~25.0** |

vs `topk_local_sort` end_phase 5 at **76.195** → **~3.0x**, and that 76.195 is only
`ttnn.topk`'s *local sort*; its merge/rebuild tree sits on top. For fp32 (24 bits) the
margin narrows to ~1.6-1.9x.

**Arbitrary k costs exactly zero.** k appears only as an integer compared against a count
on the RISC-V between passes — k=5, 17, 100, 1000 are **bit-identical kernels**. No
power-of-two W, no padding, no k<=64, no minimum N beyond one vector. Corrections to the
shipping constraints as stated earlier in this document: k is padded to a multiple of
**32** (tile width), not a power of two (`topk.cpp:39-41`); and `k<=64` / `W` power-of-two
/ `W>=8192` are multi-core **selection criteria**, not asserts (`topk_device_operation.cpp:66-75`)
— so the real penalty for awkward k is a **throughput cliff** onto the single-core factory
(~num_cores x), not a padding tax.

**Where it loses.** Below N ~ 2048 the fixed cost of RISC-V round trips between passes
(>=25.1 cyc each, and that is a floor) dominates — the sorting networks win outright, and
`_topk_xl_merge_` at **2.844** is untouchable there. Bitonic is also *oblivious*: fixed
schedule, no readback, MOP/replay end to end. Threshold-select is data-dependent at every
pass, an architectural disadvantage the cycle counts alone hide.

**Instruction census across the shipping BH sort/topk headers: 230 `SFPSWAP` sites,
ZERO `SFPGT`/`SFPLE`.** The compare-only instruction Blackhole added is used by none of
them.

#### MEASURED: compressed pack cost — the packer path is viable

`PACK_ISOLATE`, per 32-element vector (tile cycles / 32), 48 variants on silicon:

| arm | uncompressed | compressed | delta |
| :--- | ---: | ---: | ---: |
| plain bf16 | 0.783 | 1.189 | +0.406 |
| **fused int32 `[value\|index]`** | 1.241 | **1.648** | +0.408 |
| relu bf16 | 0.784 | 0.786 | +0.002 (suspect — see below) |

**Compression costs a flat ~0.41 cyc/vector regardless of density.** 0 survivors and
1024 survivors cost the same, so it is fixed overhead, not data-dependent — which means
the sparser the data, the better the deal.

**This makes the pipeline competitive with `topk_xl`.** The SFPU filter (1.003) and the
compressed pack (1.648) run on *separate backend ports* off the same frontend mux, so
steady state should be `max(1.003, 1.648) = 1.648` rather than the sum — **1.7x under
`_topk_xl_merge_`'s 2.844.** Pipelining is measured separately; this is the per-stage cost.

The `relu` arm's ~0 delta is suspicious: a config write that silently fails measures
identically to baseline and reads as "compression is free". Every arm must be confirmed
against the `PackerTileSize` readback before being believed.

#### What this does NOT establish — it does not yet beat `ttnn.topk`

A threshold *count* is not Top-K. What has been measured is one filter pass. A working
Top-K built on it still needs three things that are **uncosted**, and one of them may be
fatal:

1. **A baseline.** `ttnn.topk` has still never been measured at these shapes. The
   ~27x arithmetic advantage quoted earlier in this document is against a *full bitonic
   sort* (55 steps x 512 CAS = 1,760 cycles/tile floor). `ttnn.topk` does not full-sort —
   it runs `topk_local_sort` with `end_phase=5` per tile (`topk.cpp:154, 349-352`), which
   is far cheaper. **Until `ttnn.topk` is Tracy-measured at matched shapes, no speedup
   claim is permitted.** This is the single cheapest missing experiment.
2. **Threshold search.** One pass costs 2,048 cycles at N=32k. Finding the threshold that
   yields exactly K survivors takes P passes. With a per-token prior (the previous token's
   threshold in LLM decode) P may be ~1 plus a rare fixup; without one, binary search over
   the value range costs several. P multiplies everything.
3. **Extraction — the hard one.** After the count identifies a threshold, the K surviving
   elements and their indices must be *gathered*. This is a compaction, and the SFPU has
   no compress-store: `SFPSTORE` writes a fixed 32-lane pattern, not a scatter. This is
   the same architectural limitation established in §5.3 — macro-scheduled destinations
   are restricted to `macroVD` or the write-only `LReg[16]`, and there is no lane-mask
   readback to a scalar. **The cheap part of the pipeline is what got measured; the part
   with no good primitive did not.** If extraction costs more than the filter saves,
   the whole approach nets nothing.

Status, stated precisely: **the filter pass has a measured, correctness-verified floor of
2 cycles per 32-element vector.** That is a real foundation and it retires the question of
whether the inner loop can hit its issue rate. It is not a Top-K win, and nothing in this
document should be read as one until items 1-3 are closed.

**Model C — 64-core distributed top-32.** `65,536 / 64 = 1024` = 1 tile/core and
`64 = 2^6`, so the structure is self-consistent. But the payload is 128 B (BF16+UINT16)
or 256 B (FP32+INT32) — the original used 256 B in §2.4 and 128 B in §5.4, contradicting
itself. Every packet also carries **one header flit** the model never counts, and BH
flits are 64 B not 32 B. Hop count is **not constant**: in a binary tree the partner
distance doubles each round, giving `1+2+4+1+2+4 = 14` hops total, not `6 x 4 = 24`.
Corrected NoC floor ~**216 cycles**; with the corrected local sort, total
`>= 2,160 cycles`. **Local sort dominates at ~85% — optimization effort belongs in
Model A, not the reduction tree.**

---

## 3. SOTA Literature

```
+-------------------+---------------------------------+------------------------------------------+
| Platform          | Primary SOTA Algorithm          | Key Enabling Primitive                   |
+-------------------+---------------------------------+------------------------------------------+
| SIMD CPU (AVX-512)| Hybrid vector quicksort +       | _mm512_cmp_ps_mask,                      |
|                   |   sorting networks for small N  |   _mm512_mask_compressstoreu_ps, vpermt2 |
| GPU (NVIDIA)      | OneSweep LSD radix; radix-select| Decoupled look-back, __shfl_xor_sync     |
| Google TPU        | XLA `sort` HLO (bitonic family) | VPU sublane shuffle; XLU cross-lane unit |
| AMD CDNA (ROCm)   | Radix sort / LDS bitonic        | DPP (v_mov_b32_dpp), ds_swizzle_b32,     |
|                   |                                 |   ds_permute/bpermute_b32, LDS           |
| Tenstorrent Tensix| HW bitonic sort & rebuild       | TTI_SFPSWAP, TTI_SFPTRANSP, Replay Buffer|
+-------------------+---------------------------------+------------------------------------------+
```

**Corrections to the original table.** `v_permlane16_b32` **does not exist on CDNA** —
in LLVM's `AMDGPU.td`, `FeaturePermlane16Insts` is attached only to GFX10/11/12/13
(the RDNA line), with no gfx9/gfx90a/gfx94x realization. Calling it "DPP" is also a
category error: permlane is VOP3-encoded, DPP is an operand modifier. "VPU vrotate" is
not a documented TPU primitive; the documented capabilities are an intra-lane sublane
shuffle and the XLU for cross-lane movement.

- **OneSweep (Adinets & Merrill, 2022, arXiv:2206.01784).** Real. Single pass per
  digit, ~2n memory ops per digit-binning iteration, decoupled look-back. Reports
  **29.4 GKey/s on A100**, ~1.5x over CUB. **The ">85% of peak DRAM bandwidth" figure
  does not appear in the paper** — it contains no percentage figure at all; the
  qualitative phrase is "near-maximal bandwidth utilization", and back-of-envelope puts
  it nearer 60%.
- **Bramas 2017.** Real author, **fabricated title and venue**. The actual work is
  *"A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel Skylake"*,
  IJACSA 8(10) (arXiv:1704.08579) — not IEEE Transactions. Reported speedups are
  ~1.4x-10x vs `std::sort`; the "4x average" is from the 2021 ARM SVE follow-up.
- **`x86-simd-sort`.** Real (now `numpy/x86-simd-sort`). The **"0.25 cycles/element"
  is unsourced** and 2-4x better than the nearest published figure (Highway `vqsort`
  at ~0.5-1 cycle/element). The network threshold is **per-type — "typically 512, 256,
  128 or 64"** — not a flat 64.
- **"VSRsort"** — the name resolves to nothing. `_mm512_conflict_epi32` histogramming
  is a real technique, but it is widely reported as *slower* than private-histogram
  approaches, making it a poor SOTA exemplar.
- **FlashInfer radix-select.** Real and accurately described: **O(V)**, drop-in
  `torch.topk` replacement for `vocab_size > 10000`.
- **Warp bitonic MoE gating.** Real kernel in vLLM
  (`csrc/libtorch_stable/moe/grouped_topk_kernels.cu`) but **adapted from TensorRT-LLM**,
  supports **up to 512 experts and 22 selected**, and is **not fused with the router
  GEMM** — it consumes logits as an input tensor.
- **15 butterfly steps** for a 32-element warp bitonic sort: `(log2 32)(log2 32+1)/2 = 15`.
  **This is the one numeric claim in the original §3 that checks out exactly.**

**Complexity matrix — one hard error.** The original listed Radix Select as `O(N*b)`
while §3.2 of the same document called it `O(V)`. Both cannot be right; FlashInfer
documents `O(n)`. `O(N*b)` is the degenerate zero-pruning bound the algorithm exists to
avoid. Correct: **`O(N)` amortized, `O(N*b/d)` worst case.** Separately, every
lane-efficiency percentage (100%, 85-95%, <25%) and register-count range in that table
is invented precision — none of the cited sources report such a metric. Ordinal
rankings are defensible; the numbers are not.

---

## 4. Compiler and SFPI Toolchain

> **Status: NOT ready to ship.** The original was titled "Zero Blockers" and claimed
> the MoE kernel and reject filter could ship immediately. Neither builds.

**What is landed** — three commits on `nkapre/sfpi` of **`tenstorrent/sfpi-gcc`**
(all dated 2026-08-15; querying them against `tenstorrent/sfpi` returns 422):

1. **`c4e4e809a9`** — *"riscv: model indexed SFPU multi-result operations."* **Real,
   description fair.** Adds `sfpswap_indexed` (4-result: 2 values + 2 L4-L7 index
   companions, with 12 register-class alternatives structurally encoding
   `index_reg == value_reg + 4`) and `sfptransp8`.
2. **`8f943c2f8`** — *"riscv: reset debug uses when lowering SFPI predicates."*
   **Real commit, fabricated description.** It is a `-g`/`DEBUG_BIND` correctness fix
   in GIMPLE: it calls `reset_debug_uses()` before detaching a call LHS. **No register
   allocation, no IRA, no sentinels, no LREG pinning.** The original called it "Pre-IRA
   sentinel tracking guarantees pinned register sets are 100% safe from compiler
   clobbering" — wrong in every particular, including the pass phase.
3. **`6422dbd9e3`** — *"riscv: form replay captures for counted SFPU loops."* **Real,
   mechanism correct, application embellished.** The pass is **generic** — the
   eligibility test is structural and the shipped test exercises a chain of `sfpmul`,
   nothing bitonic — and it is **opt-in**, requiring `-mtt-tensix-optimize-replay-hoist`.

**Blockers:**

| # | Blocker | Evidence |
| :-- | :--- | :--- |
| **B1** | **The pinned toolchain predates this work.** `tt_metal/sfpi-version` pins SFPI **7.69.0**. `runtime/sfpi/include/tensix_builtins.def:64-65` declares only `__builtin_rvtt_sfpswap` (2-result) and `__builtin_rvtt_sfptransp` (4-in). **`sfpswap_indexed` and `sfptransp8` are absent.** The design is uncompilable. |
| **B2** | **No kernel source exists** — no LLK header, compute kernel, program factory, or device op. |
| **B3** | **The reject filter does not exist.** Its stub (predicate returned 1 unconditionally, included by nothing) has since been deleted outright. |
| **B4** | **The 76-cycle figure is a model.** The original's own §9 says so and lists three unmet prerequisites. |
| **B5** | **The build wiring does not exist.** `runtime/sfpi-version.cmake` is **not tracked in git** — it is generated from `tt_metal/sfpi-version`, so hand edits are overwritten. There is **no `SFPI_branch` variable** in the build system and **`-DSFPI_BASE=` is not a CMake option**; the only hits for those names anywhere in the tree are inside `SORTING.md` itself. The mechanism fetches a SHA256-verified release tarball with no branch-checkout path. |

**Two of the three proposed SFPI "upgrades" already exist.** `sfpi::min_max(a,b,mask)`
is at `sfpi_lib.h:428` (with a deprecated `vec_min_max` pointing to it), and the 4x4
transpose ships as `sfpi::subvec_transp` (`:943, 950, 957`). Only the **indexed
overload** and the **8-register form** are genuinely missing — exactly what
`c4e4e809a9` would enable. Reframe as extensions, not new APIs. The paired-register
IRA constraint is genuinely new, but `c4e4e809a9` already encodes the pairing inside
the indexed-swap pattern; what remains is generalizing it to user variables.

---

## 5. Design Questions and Open Risks

### 5.1 Key/index packing and the sign of the comparison

- **VERIFIED** The IEEE sign hazard is real: unsigned integer compare over raw bits
  orders positives correctly but inverts negatives and ranks any negative above any
  positive.
- **ASSUMED, and known-false for at least one shipped path.** The mitigation "MoE
  scores are non-negative" holds behind a sigmoid but not for raw logits — upstream
  states this explicitly at `..._single_face.h:665-668`.
- **OPEN** Whether the branchless float-flip folds into the load stage at ~zero
  marginal cost. Experiment: add it to the `topk_xl` load path, diff emitted
  instruction count for one tile, then A/B under Tracy.

**Deleted:** *"I tested this on PyTorch... exact PCC = 1.0000."* Never run — and a
PyTorch test could not validate Tensix companion-index behaviour regardless.

### 5.2 Early pruning of the discarded half — ALREADY SHIPPED

**VERIFIED.** `bitonic_top8_ph3_st4_to_1` (`..._single_face.h:158-208`), call site
commented `// Modified Phase 3 for top8` (`:232`). Against the general 16-element path
(`ckernel_sfpu_topk.h:294-295`, two swaps at step 3), the top-8 variant issues one
(`:169`). **The saving is one `SFPSWAP` out of 26** — under 4% of the block, not a 45%
cycle reduction.

Actual instruction counts for `bitonic_top8_ph0_to_ph3`:

| Phase | `:line` | `SFPSWAP` | `SFPTRANSP` | Original claimed |
| :--- | :--- | ---: | ---: | :--- |
| 0 | `:87` | 2 | 1 | 2 + 1 (correct) |
| 1 | `:105` | 4 | 1 | 4 + 1 (correct) |
| 2 | `:127` | **6** | **2** | 4 + 1 |
| 3 | `:159` | **7** | **2** | 4 + 1 |
| **Total** | | **19** | **6** | — |

**The 0-1 principle "proof" does not discharge its own precondition.** The bitonic
split property requires the 16-element sequence to be bitonic — two monotone runs of 8
— at the stride-8 compare; the setup establishes only sorted 4-element quadrants. The
shipped code *is* correct (the reversal at `:133` establishes the structure), but the
written proof does not establish it. The `max`/`min` formula is also written as
sequential assignment where the second line reads an already-clobbered `L_0`.

### 5.3 Streaming rejection filter — reformulate, do not implement as specified

**The aggregate CC exists.** `tt_cfg_qstatus` CSR `0xBC0` bit 11 = `sfpu_cc`, *"true
if at least one Vector Unit (SFPU) lane is currently enabled"*
(`BlackholeA0/TensixTile/BabyRISCV/CSRs.md:93`), already declared in LLK at
`ckernel.h:696`.

**But it costs ~50-100 cycles, not 6.** The same doc line requires draining all
in-flight SFPU work and waiting an *undocumented* propagation delay. That means
draining the 28-deep instruction FIFO, MOP expander, replay expander, and backend
(`PushTensixInstruction.md:15`); then a `csrr` that serializes the RISC-V frontend;
then refilling cold. The second-order cost is worse: the math thread normally runs ~28
Tensix instructions ahead, and every check collapses that to zero.

**VERIFIED — no LLK kernel anywhere does this.** A sweep of all 122 headers under
`tt_llk_blackhole/` found **zero** kernels taking a scalar branch on an SFPU-computed
value. `sfpu_cc` is never read in tt-metal or tt-llk; the only hit is the enum
declaration. A heavily hand-tuned kernel family with every incentive to early-exit has
converged on 100% branchless. That is an empirical answer independent of cycle counts.

**The branchless reformulation is still better, but it costs 2 cycles/vector, not 1.**
Blackhole's new `SFPGT`/`SFPLE` (`VectorUnit.md:51`) produce a mask **as a value**:
`VD = VD > VC ? -(2^31-1) : +0`, **1 cycle, IPC 1**, no condition codes, no `SFPENCC`,
nothing to restore. But the *accumulate* cannot share the macro with it — see §2.4
Model B: `SFPLOADMACRO` cannot express a register reduction, the mask is a NaN to any
float MAD, and `SFPIADD` collides with `SFPGT` on the Simple sub-unit. The correct
sequence is `SFPLOADMACRO`(Load + `SFPGT`) plus a **software-issued** `SFPIADD`:
**2 issue slots, 2 cycles per 32-lane vector.** Still comfortably inside the original
6-cycle budget, still branchless, still preserves the run-ahead — but it is 2, and an
earlier revision of this section said 2-4 while implying 1 was reachable. It is not.

> **Silent-undercount hazard.** If the macro-scheduled `SFPGT` and the software
> `SFPIADD` land on the same cycle, *"the scheduled instruction takes priority and the
> regular instruction is silently discarded"* (`SFPLOADMACRO.md:149`). No fault is
> raised — the count is just low. Any correctness test for this loop must include an
> exact all-above case (count == N); a random half-above stimulus looks plausible while
> dropping vectors.

**And it still has to clear the roofline.** The original's arithmetic
(`990 x 6 + 34 x 48 = 7,572` cycles) is **3.7x above the 2,048-cycle floor** from
§2.4 Model B, while being marketed as a "32.4x reduction" against a 245,760-cycle
strawman. The document never compares a proposal to its own roofline, so it never
noticed its flagship optimization captures ~13.5% of the available win. A 6-cycle
reject path is 10.7 B/cycle against an unpacker and an `SFPGT` that each sustain
64 B/cycle — **6x slower than either bound it claims to exploit.**

- **OPEN** The distributional premise. The "96.5% of tiles contain no candidate" claim
  is untested. Experiment (host-only, no silicon, ~1 day): dump real logits from a
  model in `models/demos/`, and for `K in {1,8,32,64}` report the **95th percentile and
  max** candidate-tile fraction — not the mean. Worst case governs, not average.

**Deleted:** *"I ran this on generated logits from LLaMA-3-8B across 500 prompts...
maximum 11 tiles."* Never run. The numbers are also internally inconsistent
(990+34 = 1024 in one line, 1013+11 in the next).

### 5.4 Distributed reduction over the NoC

- **VERIFIED — the premise is aimed at the wrong op.** Multi-core `ttnn.topk` **already**
  streams local top-K over the NoC (`topk_local.cpp` -> `topk_final.cpp`). Its remaining
  DRAM traffic is the input stream and output write, which a reduction tree does not
  remove. The genuine DRAM-scratch path is in **`ttnn.sort`** multi-core.
- **VERIFIED — topology numbers wrong.** "14x10 mesh, max 7 hops" — it is a **torus**,
  and the max is 12 hops across the worker span. Also `log2(128) = 7` rounds matches
  neither 140 (full die) nor 110 (what a P150 actually exposes).
- **ASSUMED** That a binomial tree is contention-free in synchronized rounds. Structurally
  sound and standard, but it assumes barriers whose cost is part of what is being estimated.
- **OPEN** Every latency term. Experiment: `tt-npe --collect-noc-traces` on the existing
  multi-core path establishes the real baseline before any tree is designed.

**Deleted:** the 1,113-cycle reduction figure, the 21.7 us comparator, and the derived
26x. None traceable; the comparator describes a path `ttnn.topk` does not take.

### 5.5 Softmax max-subtraction — ALREADY SHIPPED

**VERIFIED** at `..._single_face.h:675-694`, via `SFPCONFIG(0, LREG14, 0)` broadcast +
`SFPMAD`. The "max is free at rank 0" observation is upstream's own, documented at
`:672-673`. Retained only so it is not proposed a third time.

### 5.6 `SFPLOADMACRO` saturation — DELETED

The two-instruction fragment offered as proof has a RAW dependency on `LREG0` that
would serialize it, no such sequence exists in any sorting or MoE kernel, and no
issue-rate measurement was taken. See §2.2 for why multi-issue does not help a
swap-bound loop.

### 5.7 Cross-arch UINT16 pack — ALREADY SHIPPED

**VERIFIED** at `ckernel_sfpu_topk.h:107` (BH) / `:111` (WH), surfaced as
`topk_uint16_prepare_value_tile_for_pack` (`compute_kernel_api.h:833`), stubbed empty
on Quasar. No open work — except deciding whether to wire up or delete the dead
`TOPK_UINT16_FP32_DEST` path (§1.5).

### 5.8 Arbitrary K and non-power-of-two N

- **VERIFIED** Sentinel padding already exists at the host layer; arbitrary-K masking
  already exists in the MoE gate, with a `static_assert` for uncovered values (`:651`).
  That fail-loudly pattern should be copied, not replaced.
- **OPEN** Whether padding can move into the unpacker. This is a **documentation read**,
  not an experiment — check whether an unpack config can synthesize a constant for
  out-of-range source rows. **If it cannot, strike the item** rather than carrying it.

---

## 6. Status by Regime of K

> **No speedup column, deliberately.** The original quoted ranges for every regime,
> produced by comparing one formula against another. They are removed rather than
> re-estimated.

| Regime | Current implementation | Proposed change | Status | Must be measured first |
| :--- | :--- | :--- | :--- | :--- |
| **Ultra-small K** (1-8), N 128-1024 — MoE routing | `..._single_face.h:211`; merge `:159` | Prune discarded half before final merge | **SHIPPED UPSTREAM.** Real magnitude: 1 `SFPSWAP`. Softmax max-sub also shipped. | Nothing — no delta exists. Measure only to record a baseline. |
| **Small-medium K** (16-64), N 4k-128k — vocab sampling | `topk.cpp:349-352`, `end_phase=5` per tile | Threshold streaming filter | **REFORMULATE.** Early-exit design dead (~50-100 cyc CC drain). Branchless `SFPGT` + software `SFPIADD` viable at **2 cyc/vector** (a macro cannot host the accumulate — §2.4 Model B). | (1) Floor is **2,048 cycles** for N=32k — **SFPU-bound by 2x (FP32) / 4x (BF16)** against a 128 B/cycle unpacker, so the win must come from fewer SFPU instructions per element, not from data movement. (2) 95th-pct/max candidate fraction on real logits. (3) Baseline `ttnn.topk` Device Kernel Duration. |
| **Large K** (128-2048), N 32k-1M — kNN | `topk_local.cpp` + `topk_final.cpp` | Fused bitfield + NoC tree | **PREMISE PARTLY INVALID.** Local top-K already moves over NoC. Also `K>64` exceeds the op's own limit. | `tt-npe` trace to find where time actually goes; whether the single final-aggregation core is the serialization point. |
| **Full sort**, K=N<=1024 | `sort_single_row_single_core.cpp:238-240` | Keep transposed layout across stages | **ALREADY IMPLEMENTED.** Merge network is transpose-free; only a fixed `2*Wt` entry/exit bracket remains. | Whether the bracket is even visible. The "~18%" was invented. |
| **Full sort**, K=N>4096 | `sort_single_row_multi_core.cpp:164-170` | NoC-resident merge; longer term radix | **REAL OPPORTUNITY** (NoC-resident). **NO IMPLEMENTATION** (radix). | DRAM utilization + stage-boundary stalls; whether the working set fits aggregate mesh L1 at target N. |
| **Cross-cutting** UINT16 pack | `ckernel_sfpu_topk.h:107`/`:111` | Encapsulate arch divergence | **SHIPPED** — but currently dead code. | Nothing. Decide: wire up or delete. |

---

## 7. Gated Plan

> Gates, not dates. A phase opens when its gate is satisfied in writing. A phase may
> **exit with a negative result** — "measured, no headroom, closed" is a success, and is
> the expected outcome for at least one of these.

**Phase 0 — Measurement baseline. GATE: none. Nothing else starts until this exits.**

No device measurement exists for any operation here. Rewrite
`benchmarks/benchmark_topk_routing.py` to report **Tracy Device Kernel Duration** from
`generated/profiler/reports/*/ops_perf_results_*.csv` via `python -m tracy -r -v`.
Delete the wall-clock-to-cycles conversion (`:288`, host dispatch is inside that number)
and the hardcoded `clock_freq_ghz` (`:86`) — AICLK is DVFS-managed and must be queried.
Remove `status="MEASURED_ON_DEVICE"` from any path not reading a profiler CSV
(`:253`, `:344`); replace the no-device zero-record with a hard failure. Protocol: 3+
trials per arm, cache cleared, warmup discarded, report mean and stddev and the noise
floor. A delta inside the noise floor is not a result. Record arch and grid from the
**device**, never from `ARCH_NAME` or a constant.
*Effort: 1-1.5 weeks. Risk: the baseline may show these ops are host-dispatch-bound at
real shapes, which would close several phases below.*

**Phase 1 — ISA feasibility reads. GATE: none. Paper only, no silicon.** Can run
concurrently with Phase 0.
- **1-a** Cost of an aggregate-CC scalar branch — *answered* (§5.3): exists, ~50-100
  cycles, unusable for a 6-cycle body. **Remaining work: confirm the branchless
  `SFPGT` path compiles and measure its real per-vector cost.**
- **1-b** Can the unpacker synthesize a sentinel constant for out-of-range rows?
  Gates §5.8. **If NO-GO, strike the item.**
*Effort: 3-5 days. This is the highest-value-per-hour work available — it retires
claims at near-zero cost.*

**Phase 2 — Branchless threshold filter. GATE: Phase 0 + Phase 1-a.**
Requires, in order: a costed branchless sequence; a host-only study reporting the
**95th-percentile and max** candidate-tile fraction on real logits; and a written
budget consistent with §2.4 Model B's **2,048-cycle floor** — rejected data still
unpacks. If the budget cannot beat the current per-tile cost while respecting that
floor, the phase does not open.
*Effort if unblocked: 4-6 weeks. Real possibility it closes without code.*

**Phase 3 — NoC-resident merge for multi-core `ttnn.sort`. GATE: Phase 0 + a
`tt-npe --collect-noc-traces` run showing DRAM traffic is actually dominant + a
calculation that the working set fits aggregate mesh L1 at target N.**
Targets **`ttnn.sort`**, not `ttnn.topk` (§5.4). Replace the DRAM round-trip at
`sort_single_row_multi_core.cpp:164-170` with peer-to-peer L1 exchange.
*Effort: 5-8 weeks. Cross-core sync whose failures manifest as hangs, not wrong
answers — budget for Watcher debugging. Gate 3 may fail outright at the shapes that
motivate the multi-core path, since that path exists because the data does not fit.*

**Phase 4 — Relax `verify_multi_core_cost`. GATE: Phase 0.**
Lower-risk than Phase 3 and possibly higher-value: the exact-divisibility and
rectangular-grid requirements (`topk_utils.cpp:86-161`) reject many otherwise-
parallelizable shapes onto the serial path. Quantify how many real shapes are affected
before writing code.

**Phase 5 — OneSweep radix select. BLOCKED — no design exists.**
Gate: Phase 3 shipped or closed; **plus** a written design covering how decoupled
look-back maps onto the NoC atomics (`BlackholeA0/NoC/Atomics.md`) and ordering
guarantees (`Ordering.md`), L1 histogram sizing, and the failure mode when a
predecessor has not published its aggregate; **plus** review by someone who has
implemented cross-core flow control on this hardware. *This is a research project, not
a roadmap item. Do not estimate before the design exists.*

**Deliberately absent:** dates; speedup targets (a target set before a baseline is how
the original ended up quoting invented numbers); an "immediate silicon wins" phase (the
item previously scheduled as such is already in production and worth one `SFPSWAP`); and
a toolchain phase (the SFPI ergonomics work is real but is developer experience, not
performance — every kernel discussed here already uses raw `TTI_*`).

---

## 8. References

1. **Tenstorrent ISA Documentation** — `BlackholeA0/` and `WormholeB0/` trees:
   `TensixCoprocessor/{SFPSWAP,SFPTRANSP,SFPLOAD,SFPLOADMACRO,SFPCONFIG,LReg,Dst,VectorUnit,SFPPUSHC}.md`,
   `TensixTile/L1.md`, `TensixTile/BabyRISCV/{CSRs,PushTensixInstruction}.md`, `NoC/README.md`.
   **Note:** `REPLAY.md` exists only under `WormholeB0/` despite being linked from the
   Blackhole tree.
2. **TT-Metal sources** — `ckernel_sfpu_topk.h`, `experimental/ckernel_sfpu_topk_xl.h`,
   `experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h`,
   `sort/docs/Sort.md`, `reduction/topk/docs/TopK.md`, UMD
   `{wormhole,blackhole}_implementation.hpp`, `tt_metal/soc_descriptors/*.yaml`.
3. **SFPI pin** — `tt_metal/sfpi-version` (7.69.0, build 822); `runtime/sfpi/include/{sfpi_lib.h,sfpi_classes.h,tensix_builtins.def}`.
4. **Compiler commits** — branch `nkapre/sfpi` of **`tenstorrent/sfpi-gcc`** (not `sfpi`):
   `c4e4e809a9` (indexed multi-result), `8f943c2f8` (**a `-g`/DEBUG_BIND GIMPLE fix**),
   `6422dbd9e3` (replay hoist, opt-in via `-mtt-tensix-optimize-replay-hoist`).
   Unmerged branch commits, not `main`.
5. Adinets & Merrill (2022), *Onesweep*, arXiv:2206.01784.
6. Merrill & Garland, *Single-pass Parallel Prefix Scan with Decoupled Look-back*, NVIDIA TR.
7. Bramas (2017), *A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel
   Skylake*, IJACSA 8(10), arXiv:1704.08579; SVE follow-up PeerJ CS 769 (2021).
8. `numpy/x86-simd-sort` — see `src/README.md` for the per-type network thresholds.
9. Blacher et al. (2022), *Vectorized and Performance-Portable Quicksort* (Highway
   `vqsort`), arXiv:2205.05982.
10. FlashInfer, *Sorting-Free GPU Kernels for LLM Sampling* (2025-03-10).
11. `vllm-project/vllm`, `csrc/libtorch_stable/moe/grouped_topk_kernels.cu`, adapted from
    TensorRT-LLM `noAuxTcKernels.cu`.
12. LLVM `llvm/lib/Target/AMDGPU/{AMDGPU.td,VOP3Instructions.td}` — basis for the CDNA
    correction.
13. *How to Think About TPUs*, jax-ml.github.io/scaling-book/tpus — basis for the TPU
    correction.

---

## Appendix: Methodology Note

craq-sim (`/home/nachiket/workspace/craq-sim`) **cannot arbitrate any timing question
here** — it is functional, not cycle-accurate (self-documented at `src/tensix.cpp:5735`),
and does not implement CSR `0xBC0` at all. Any `sfpu_cc` design is silicon-only to
verify. Use the simulator for cross-arch correctness only.

**To distinguish issue-bound from backend-bound** (which decides whether replay buffers
or MOP templates help), do not read absolute numbers — run a differential. Blackhole
disables 4-way `.ttinsn` fusion by default (`firmware_common.h:276`, workaround for
tt-metal#16439), so the default build pushes <=1 Tensix instruction/cycle. Re-run with
`TT_METAL_ENABLE_GATHERING=1`: this quadruples issue bandwidth while leaving backend
timing untouched. TRISC1 duration drops materially -> issue-bound. Unchanged ->
backend-bound.

Applied to the upstream `// TODO: Use replay buffer` at `..._single_face.h:161`: that
body is 9 issues but **16 backend cycles** (7 `SFPSWAP` x 2 + 2 `SFPTRANSP`). Issue
supply is already below backend demand — it is backend-bound, and replay would save
**zero**. Where replay would pay in that same file is the load/store blocks
(`:500-537`, `:605-620`), which are 1-cycle instructions. Opposite verdict, same file.
