# MiniMax-H3 on Wormhole: the MVMUL-order space of the matmul K loop — exploration plan (2026-09-23)

Follow-on to [kloop_refill_reorder_handoff.md](kloop_refill_reorder_handoff.md), which landed one alternation (reuse-A / reuse-B
mirror between consecutive K tiles, `matmul_block_kloop`) and measured 210 -> 203 cycles per K-tile step on the 2x2 fp32 loop.
This document enumerates every way the MVMULs of a K loop can be ordered, says which of them can matter and by how much, and
lays out the experiments. Evidence base: [ff1.md](ff1.md) §3.1 exp 16-18b; the "Tensix Matmul Cycle" page
https://claude.ai/artifact/Bd4na6WorztxQMoNiMUBxv.

## 0. What bounds the whole exercise

Per K-tile step of a 2x2 subblock (4 unpacks, 4 tile-MACs of 32 cycles at HiFi2), measured on this galaxy (exp 18b):

| configuration | old order | alternating (landed) |
|---|---|---|
| full loop | 210 | 203 |
| no packer, real unpack + real math | 198 | 193 |
| unpack mocked (handshakes only) | 192 | 193 |
| math mocked (handshakes only) | 193 | 193 |

Three consequences, which decide where effort goes:

1. **At 2x2 the order question is answered to within noise.** With the packer out of the way the alternating loop already runs
   at the 193 floor the two mocks define, and that floor is four source-register handshakes per step, which every 2x2 order
   has. No 2x2 order can take another cycle from the no-pack loop. The order experiments at 2x2 (§2) are therefore a
   *confirmation* sweep with an upside bounded by the 203 - 193 = 10 cycles the packer adds in production, and only if some
   order happens to interact better with the DST handoff.
2. **The 10 remaining cycles are the packer's interaction with the loop**, not L1 (0% port refusals in every column of the
   exp 18b table) and not the MVMUL order (the mocks and the no-pack row do not move). That is a separate experiment family (§3).
3. **The order space that matters is the 2x4 / 4x2 one**, which opens when fp32 dest off is decided: 6 tiles per step for 8
   tile-MACs, and the reused operand has 4 tiles per step but only 2 banks, so its banks are refilled *inside* the step and the
   within-step order decides whether those refills are hidden. That is where a wrong order costs tens of cycles per step and
   a right one does not (§4).

Everything below is written so that one table-driven LLK entry point (§1) serves all three.

## 1. Make the order a table, once

Today `_llk_math_matmul_kloop_` (`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h:1060-1115`) hard-codes two
sequences. Every experiment below needs a different one, so step one is a generator, not a kernel edit:

- **Input:** for each step parity `p` in a period `P`, an ordered list of the `rt x ct` operand pairs `(a_i, b_i)` to multiply.
- **Derived per transition:** which source changes tile (flip that source's read bank with `SETRWC CLR_A/CLR_B`, with both
  `CLR_DVALID_Src*_Disable` bits set so a flip never releases), and which tiles have just had their last use (release with
  `CLEARDVALID`, whose flip is suppressed with its keep bit when the next MVMUL reads the same bank). `dst` index follows the
  pair (row = in0 tile = SrcB, column = in1 tile = SrcA), so numerics are unchanged by construction for any order.
- **Feasibility check:** the unpacker writes each source's tiles into alternating banks in a fixed order (tile 0 to bank 0,
  tile 1 to bank 1 per step at 2x2; `_llk_unpack_AB_matmul_`, `llk_unpack_AB_matmul.h:287-447`). A schedule is feasible without
  touching the unpacker only if, for each source, the read pointer is on the bank that holds the first tile the next step reads:
  each source flips an even number of times per step when consecutive steps start on the same tile index, odd when they
  alternate. Schedules that fail this need a matching period in the unpacker's write order (§2.4); the generator reports which.
- **Output:** a `constexpr` table the kloop entry point walks, selected by a template parameter, so every candidate compiles to
  straight-line `TTI_*` code like the two orders that exist today. Validate each emitted table on the tt-llk golden
  (`tests/python_tests/test_matmul_kloop.py`, 112 cases) before it is timed.

A ~200-line Python model alongside it (release time = end of the tile's second use; refill usable at release + L_clear + 32 +
L_valid, with the two per-source unpackers serialized on their own tiles; L fitted to the measured 198 / 193 / 210 / 203)
ranks schedules before they are measured. Its job is to prune, not to predict: exp 18 showed a cycle model can be off by a
factor of two on this loop, so every kept candidate is measured.

## 2. The 2x2 order space (confirmation sweep, upside <= 10 cycles per step)

### 2.1 Within one step: 24 orders, 3 classes

Four pairs {A0B0, A0B1, A1B0, A1B1}; 4! = 24 orders. Each tile is used twice, so the release profile of any order is fixed by
*which* MVMULs share operands, and only three patterns exist:

| class | example | consecutive MVMULs share | releases at 64 / 96 / 128 / 128 | banks flipped per step (A, B) |
|---|---|---|---|---|
| **row-major (reuse-A)** | A0B0, A0B1, A1B0, A1B1 | ①② A0; ③④ A1 | A0 / B0 / A1, B1 | A: 2 (even), B: 4 |
| **column-major (reuse-B)** | A0B0, A1B0, A0B1, A1B1 | ①② B0; ③④ B1 | B0 / A0 / A1, B1 | A: 4, B: 2 |
| **zigzag (Hamiltonian)** | A0B0, A0B1, A1B1, A1B0 | ①② A0; ②③ B1; ③④ A1 | A0 / B1 / A1, B0 | A: 2, B: 3 (odd) |
| **diagonal-containing** | A0B0, A1B1, A0B1, A1B0 | ②③ B1 only | — / A0, B1 / A1, B0 | both flipped twice at once |

Relabelings (start on tile 1, swap the roles of A and B) give the other 20. The profile {64, 96, 128, 128} is the best any
order can do: the second release cannot come before ③. The diagonal class releases nothing before 96 and is dominated; drop
it. So within a step the only free choice is **which tile is released first and second**, and which source has the odd flip.

### 2.2 Across steps: period 1, 2, 4 and the objective

The order matters only through the boundary: the next step's ① needs one A and one B, its ② a second tile, its ③ the third.
Let `u(t) = t_release + L` be when a refilled tile is usable. For a period-P schedule the objective is the sum over the
boundary of `max(0, u(needed tile) - t_needed)`, with the two unpackers each serializing their own two refills.

| schedule | what it is | status |
|---|---|---|
| **P1, row-major** | the legacy loop | measured: 210 / 198 (no pack) |
| **P2, row-major / column-major mirror** | landed `matmul_block_kloop` | measured: 203 / 193 (no pack) = floor |
| P2, row-major / row-major starting on tile 1 | flips the first-needed tile instead of the first-needed source | needs the unpacker to write tile 1 first on odd steps (§2.4) |
| P2, zigzag / mirrored zigzag | one source flips an odd number of times per step, so the two steps must alternate start tiles | needs a period-2 unpacker order, or a compensating flip |
| **P4** schedules | e.g. row / column / row' / column' with start tiles 0,0,1,1 | the only family that can make the *second*-needed tile the one released at 64 for both sources over the cycle; model says +0..3 cycles vs P2 |
| P1 with a compensating `SETRWC` flip at step start | any single order restarted on bank 1 | equivalent to P2 relabeled; no new profile |

Honest expectation: the P2 mirror already puts the no-pack loop on the floor, so **none of these can beat it on the no-pack
loop**; the sweep measures them in the *full* loop to see whether any interacts differently with the packer (§3). Budget: the
generator plus 6-8 candidates at ~2 min each with `mm_kloop_variants.py apply 0` (KLOOP zone) and `apply 3` (no pack) — one
afternoon. Report as one table; if no candidate beats 203 ± 1, close the 2x2 order question in ff1.md and move to §4.

### 2.3 Finer than a tile-MAC: fidelity phases and MOP structure

HiFi2 runs two 16-MVMUL phases per tile-MAC as the inner loop of one MOP. Splitting phases across pairs (phase 1 of all four,
then phase 2) would hold every bank to the end of the step: strictly worse. Interleaving K tiles at the MVMUL level
(③_k, ①_{k+1}, ④_k, ②_{k+1}) needs a third bank per source for the tile carried across the boundary; Wormhole has two.
Both are listed so nobody re-derives them.

### 2.4 The unpacker's side

Measured order-independent for the *issue* order (srcA MOP before or after the two srcB unpacks: 193.4 / 201.6 / 279.5 vs
193.7 / 201.7 / 279.5, handoff §0.1): unpack instructions queue per unpacker, so only the FPU's release order matters. Two
unpack-side changes remain relevant only as enablers of the math schedules above:

- **write tile 1 before tile 0 on odd steps** (per source): the srcB pair is two explicit `UNPACR`s with computed addresses
  (`llk_unpack_AB_matmul.h:341-393`), so swapping them per parity is a loop change; the srcA pair is the unpack MOP
  (`:39-168`, `:439`) with a fixed stride, so it becomes two explicit `UNPACR`s or a second MOP program.
- **split the two sources' first tiles across the step boundary differently** (write A0' before B1 in the current step): the
  unpackers run in parallel so this is already what happens; no experiment.

### 2.5 Result of the 2x2 sweep (2026-09-23 23:50 to 2026-09-24 00:06, this galaxy)

The generator of §1 was built as a compile-time schedule interpreter inside `_llk_math_matmul_kloop_`
(`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h`, `namespace kloop_sched`; `MM_KLOOP_SCHED` selects the table
row, 1 = production). For each schedule the bank operations are derived from the order (flip on a change with a later reuse,
`CLEARDVALID` on a last use, both handed back at step end), and the production and legacy orders come out instruction for
instruction as the hand-written code did. Runs: single device, ff1 plain (8,7,10) 2x2 fp32 HiFi2, `KLOOP` zone, 3 calls each,
full loop and no-pack; the header edit is paired with a stamp in the kernel text because the JIT hash covers only the kernel
source (`tt_metal/impl/kernels/kernel.cpp:666`).

| schedule | period | full loop, cycles per step | no packer | PCC | note |
|---|---|---|---|---|---|
| 0 row-major every step (legacy order) | 1 | 210 | 198 | 1.000000 | reproduces the per-tile `matmul_block` loop: the interpreter is validated |
| **1 reuse-A / reuse-B mirror (production)** | 2 | **201-202** | **193** | 1.000000 | |
| 2 column-major every step | 1 | 210 | 198 | 1.000000 | mirror image of 0, same numbers |
| 8 mirror, parity swapped | 2 | 201 | 193-194 | 1.000000 | = production within noise |
| 3 row-major / row-major from tile 1 | 2 | **hang** | — | — | see below |
| 4 zigzag; 5 zigzag / mirrored zigzag; 6 period-4; 7 diagonal | | not measured | | | same mechanism as 3, now rejected at compile time |

**The mirror stands, and the space above it is closed for the 2x2 K loop.** Both period-1 orders sit at 210 / 198, both
period-2 mirrors at 201 / 193, and 193 is the mock floor. Nothing else in the table could beat it without changing the
unpacker, and the packer's 8-10 cycles do not respond to the MVMUL order.

**Why schedule 3 hung, and what it closes.** Starting a step on tile 1 was implemented as a compensating `SETRWC` flip of
the FPU's read bank onto bank 1 before the first MVMUL. The unpacker writes tile 0 into bank 0 first and tile 1 into bank 1
second, so the FPU sat on a bank that had not been written while the unpacker waited for the release of a bank that would
never be read: a deadlock, device wedged, `tt-smi -r all` needed. Every schedule that starts a step on tile 1 or flips a source
an odd number of times (3, 4, 5, 6, 7) has the same structure and is now a `static_assert` in the interpreter. Making them
runnable needs the §2.4 enabler on the unpacker side (write tile 1 first on the matching steps), which is the only way the
read order and the write order can disagree. Given the bound in §0 that enabler is not worth building for 2x2; it becomes
relevant only if a 2x4 / 4x2 schedule in §4 needs it.

**The cycle model did not earn its keep** (`tools/kloop_order_model.py`): with a per-MVMUL overhead and a refill latency
fitted to 198 / 193 it puts every tile-0-start schedule at 196 and cannot separate the legacy order from the mirror, so the
two-unpacker serialization that made the mirror win is below its resolution. Rank by measurement at 2x2; for 2x4 the model
would need bank-level state to be useful.

## 3. The 10 cycles that are not the order: the packer's interaction

`nopack` 193 vs full 203 under the alternating order; 0% L1-port refusals; exp 12 measured `ACQ` 21 + `PWAIT` 26 cycles per
7-step subblock (~7 per step) on the DST-half handoff. Experiments, all on the single-device bench with the KLOOP zone:

| # | experiment | what it tests | expected |
|---|---|---|---|
| P1 | K_block 14 with `matmul_block_kloop` (was slower under the old loop, exp 1) | half the DST handoffs per tile-MAC | -3..-5 if the handoff is the cost; the old-loop result may not carry over |
| P2 | pack the previous subblock's 4 tiles in a different order (row-major vs the current dst order) | L1 write pattern vs unpacker reads | ~0 (port refusals are 0%) |
| P3 | `tile_regs_acquire` placement: acquire the next DST half before the last K tile of the current subblock is issued | overlap the ACQ wait with the last MVMULs | -2..-5 |
| P4 | fp32 dest off, 2x2 (bf16 intermediate, half the pack bytes), same order | pack bytes vs handoff count | separates the two |
| P5 | counters: `PACKER0_DEST_READ_REQ` vs `DEST_READ_GRANTED_*` and `MATH_NOT_STALLED_DEST_WR_PORT` on full vs no-pack | is the FPU ever stalled on the DST write port while the packer reads the other half | a number, today unread |

If P1/P3 recover most of the 10, the 2x2 loop is done at ~195; if not, the residue is the handoff latency itself and is out of
reach without `dst_full_sync` (which serializes math and pack; measured not a lever).

### 3.1 Results (2026-09-24 00:13-00:20, this galaxy, production schedule, KLOOP zone, single device)

| # | run | cycles per K-tile step | reading |
|---|---|---|---|
| P1 | (8,14,8) 2x2 fp32 on | **did not build**: `program.cpp:2097` (L1) on `minimal_matmul`; (8,7,8) reference 202-203 | K_block 14 needs a smaller M_block on this op; not pursued (exp 1 measured it slower on the AGMM) |
| P4 | (8,7,10) 2x2 **fp32 dest off** (bf16 intermediate, half the pack bytes), same order | **196-198** vs 202 with fp32 on | the packer's cost above the 193 floor halves with the pack bytes: it is pack traffic / pack duration, not the number of DST handoffs (unchanged between the two) |
| P5 | counters, full loop vs no-pack (from the exp 18b logs) | `MATH_NOT_STALLED_DEST_WR_PORT` = `MATH_INSTRN_AVAILABLE` (0% DST write-port stall), scoreboard stall 0%, packer reads DST 17-19% of cycles, pack thread waits on the math-commit semaphore 60% | the FPU is never blocked by the packer on DST; whatever the ~8 cycles are, they are not a DST port conflict |
| P2, P3 | pack order; earlier `tile_regs_acquire` | not run | P2 expected ~0 (0% L1-port refusals); P3 needs the subblock loop restructured and is bounded by the ~8 cycles |

So the residual above the handshake floor at 2x2 is ~8 cycles with fp32 packs and ~3 with bf16 packs; fp32 dest off removes most
of it as a side effect, which adds to that decision's case ([ff1.md](ff1.md) §3.2). No further pack-side experiment at 2x2 is
worth its run time.

## 4. The 2x4 / 4x2 order space (the one worth the effort; gated on fp32 dest off)

With bf16 accumulation DST holds 8 tiles per half, so a step is `rt x ct = 2x4` (2 A tiles, 4 B tiles) or `4x2`: 6 unpacks and
6 handshakes per 8 tile-MACs (measured 42 cycles per tile-MAC vs 52 at 2x2, exp 15). The reused source (the one with 4 tiles)
has only 2 banks, so **two of its refills happen inside the step**, and the within-step order decides whether each refill has
the ~60 cycles it needs:

- **Row-major over the 4-tile source** (A0B0 A0B1 A0B2 A0B3 A1B0 ...): B0 is used at ① and ⑤, so all four B tiles are live
  across the step: impossible with 2 banks. Infeasible, not merely slow.
- **Pair-wise (column-major over the 4-tile source)**: A0B0 A1B0 | A0B1 A1B1 | A0B2 A1B2 | A0B3 A1B3. B0's two uses are
  consecutive (released at 64), B2 is needed at 128: refill window 64 cycles against a ~60-cycle refill: hidden, barely. The
  two A tiles are live all step (2 banks, fine). Release profile at the boundary: B3 at 256 and A0/A1 at 256 with B2 at 192.
- **Pair-wise with the A tiles alternating** (A0B0 A1B0 | A1B1 A0B1 | A0B2 A1B2 | A1B3 A0B3): same B windows, A's bank flips
  halve. Candidate for lowest bank-op count.
- **Cross-step alternation** as at 2x2: which A tile and which B tile the next step needs first, chosen so the tile released
  earliest is needed first; with 6 tiles there are more choices and the boundary exposure can be driven to zero on the model
  (the last pair releases only two tiles, and the next step's ① needs exactly two).
- **K_block choice interacts**: at 2x4 the pack is 8 bf16 tiles per subblock (half the bytes of 4 fp32), so §3's handoff cost
  per step changes too; K_block 7 vs 14 is a first-class variable here.

**First look, 2026-09-24 (single device, fp32 dest off, production K loop, no alternation for non-2x2 blocks).**
(12,7,8) **4x2**: 11.64 ms of K loop per core = **45.8 cycles per tile-MAC** (2x2 fp32 on: 50.7; 2x2 fp32 off: 49.0), PCC
0.999732 / rel-RMSE 0.01066 (the fp32-dest-off numerics, as expected). Per 4x2 step (8 tile-MACs, 6 unpacks): 366 cycles =
256 of MVMUL + ~96 for six handshakes + ~14 of boundary exposure. (8,7,16) **2x4 did not build** on `minimal_matmul` at this
shape (`program.cpp:2097`, L1); exp 15 measured it on the mesh AGMM at ~42 cycles per tile-MAC.

**Correction to the paragraph above, from the 2x2 sweep (§2.5).** The FPU cannot start a step on tile 1 of a source (the
compensating read-bank flip deadlocks against the unpacker's write order), so at 4x2 every step must start on A0 and B0, and
the two A tiles are both used by every B pair and so are both live until the last pair. The boundary exposure is therefore
fixed by which A tile the last pair reads first: with the standard pair order (A0 B_j, A1 B_j) A0 is released at 224 and A1
at 256 of a 256-cycle MVMUL train, and the next step's ① (A0' B0') waits ~14 cycles for A0'. The only way to hide it is the
§2.4 unpacker enabler (write A1 before A0 on odd steps, so the odd step can start on A1, released earlier by an even step that
ends on A1 B3, A0 B3). That is worth at most ~14 of 366 cycles per step (~4% of the 4x2 K loop), the same order as the 2x2
mirror gain, and it needs the unpacker-side change the 2x2 work did not. The pair order itself (B0..B3 with consecutive uses)
is forced by the two-bank alternation, so the 4x2 within-step space is one bit per pair (which A first), not 8!.

Plan: the §1 generator with `rt x ct` as parameters; model-rank the pair-wise orders and their P2 alternations; measure the
top 3-4 in the no-pack loop first (the floor at 2x4 is 6 handshakes per 8 tile-MACs plus the intra-step refills; the mocks
of `mm_kloop_variants.py` extend to 2x4 with one more pair of set/clear per step), then the full loop, then the mesh op with
`--fp32-dest 0 --blocks 8,7,16,2,4`. Numerics are the fp32-dest-off decision's, not the order's; the order is bit-exact by
construction and checked by the extended golden.

## 5. What is not an ordering question (listed so it is not re-opened as one)

- Fewer handshakes per tile-MAC: only the subblock shape (§4) does that.
- Fewer bytes per handshake: bfp8 operands; independent of order, stacks with everything here.
- Unpacker throughput: writes at ~32 cycles per 2 KB tile when not refused (exp 18b, every real-unpack column); not a lever.
- L1 bandwidth: 0% port refusals in all six exp 18b configurations.
- Math-thread issue: 0.5% stalled, instructions always pending; the `matmul_block` per-call overhead question was closed by
  `rawunpack` / `fixedaddr` (exp 16) and by HiFi4 = 256 + 25.

## 6. Order of work

1. §1 generator — **done 2026-09-23 as the `kloop_sched` interpreter in `llk_math_matmul.h`**, validated by reproducing the
   legacy order at 210 / 198 and the production mirror at 202 / 193 with PCC 1.000000. The cycle model was written and set
   aside (§2.5).
2. §2 confirmation sweep at 2x2 — **done 2026-09-24 (§2.5): the P2 mirror stands, the question is closed for 2x2.**
3. §3 packer-interaction experiments — **done 2026-09-24 (§3.1)**: the residual is pack-byte traffic (halves with bf16 packs),
   not DST handoffs or ports; nothing further at 2x2.
4. §4 once fp32 dest off is decided for ff1 / to_qkv: **first look done (§4)**: 4x2 at 45.8 cycles per tile-MAC with ~14 cycles
   per step of boundary exposure, recoverable only with the unpacker-side enabler (write the other A tile first on odd steps).
   Remaining work if pursued: the enabler in `_llk_unpack_AB_matmul_` plus a 4x2 / 2x4 schedule table in `kloop_sched`
   (~2 days), for ~4% of the fp32-off K loop.
