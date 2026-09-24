# MiniMax-H3 on Wormhole: hiding the srcB refill in the 2x2 matmul K loop — plan (2026-09-23) and result (same evening)

A tt-llk change to the Wormhole matmul LLK so that the two source-register refills each K-tile step makes late land on
different unpackers: the K loop of every `matmul_blocks` kernel in the block (ff1, to_qkv, to_out AGMMs; ff2's plain and
fused matmul) runs one `matmul_block_kloop` call per subblock and, for a 2x2 subblock, alternates the MVMUL order between
consecutive K tiles. Landed 2026-09-23 on this host (UF-EV-B12-GWH02); §0 is the result, §1-§3 the evidence and the source
trace the plan was built on (still correct), §4 the design as built and where the original plan was wrong, §5 the
validation as run, §6-§7 risks and what this does not replace. Evidence for the plan: [ff1.md](ff1.md) §3.1 (exp 16, 17,
17b); reference write-up with figures: the "Tensix Matmul Cycle" page, https://claude.ai/artifact/Bd4na6WorztxQMoNiMUBxv
(§8-9 are the GWH02 reproduction and the counter reading). Block context: [README.md](README.md).

## 0. Result

**Code.** Compute API `matmul_block_kloop_init` / `matmul_block_kloop` (`tt_metal/hw/inc/api/compute/matmul.h:293-368`;
Blackhole / Quasar and `MM_THROTTLE != 0` fall through to `matmul_block_init` and a loop of `matmul_block`). Wormhole
wrappers `llk_math_matmul_kloop_init` / `llk_math_matmul_kloop` (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h:80-131`)
and `llk_unpack_AB_matmul_kloop` (`.../llk_unpack_AB_matmul_api.h:142-209`). LLK: `_llk_math_matmul_kloop_init_`
(`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h:1018-1035`: the standard init, then for 2x2 both
`CLR_DVALID_Src*_Disable` bits in one `SETC16`, `:1033`), `_llk_math_matmul_kloop_` (`:1060-1115`: even steps in the reuse-A
order, `:1080`, odd steps in the reuse-B order, `:1098`, all bank hand-backs explicit `CLEARDVALID`s; other blocks call
`_llk_math_matmul_` kt_dim times), `_llk_unpack_AB_matmul_kloop_` (`llk_unpack_AB_matmul.h:473-503`: `_llk_unpack_AB_matmul_`
per K tile, in0 advancing one tile and in1 `in1_kt_stride` tiles). Callers: `minimal_matmul/device/kernels/compute.cpp:381`
(init `:492`), `all_gather_minimal_matmul_async/device/kernels/compute.cpp:377` (init `:483`),
`minimal_matmul/device/kernels/fabric_bound_compute.cpp:445` (init `:556`). The SDPA kernels (`matmul_block_no_mop`) are
untouched.

**Measured, this host, 2026-09-23 evening.** Single device, plain ff1 (8,7,10) 2x2 fp32 dest, `KLOOP` zone per output
block (`tools/mm_kloop_variants.py parse`), cycles per K-tile step, same-session A/B by forcing the API's fall-through:

| fidelity | per-tile `matmul_block` loop | `matmul_block_kloop` | delta |
|---|---|---|---|
| LoFi | 200.3 | 193.8 | -6.5 (-3.3%) |
| HiFi2 | 209.7 | 202.3 | -7.4 (-3.5%) |
| HiFi4 | 281.5 | 279.6 | -1.9 |

**Hardware counters, before and after** (`--profiler-capture-perf-counters=fpu,pack,unpack,instrn --perf-counter-multipass`,
`tools/tensix_perf_counters.py`; HiFi2, per core over the whole kernel, K loop plus copy epilogue; same session, the
"before" reproduces the exp 17 capture of the earlier session to 0.1%):

| counter | per-tile order (before) | alternating (after) | reads as |
|---|---|---|---|
| elapsed cycles per core | 13.52 M | 12.94 M | -4.3% |
| FPU active cycles | 8,141,937 | 8,141,937 | identical work: 63,504 steps x 128.2 |
| **FPU util** (active / elapsed) | **60.2%** | **62.9%** | the whole gain is fewer idle cycles |
| math source data ready / elapsed | 60.2% | 62.9% | = FPU active in both: the FPU idles exactly when no operand is valid |
| math thread stalled | 0.5% | 0.5% | issue never the limit |
| unpacker 0 / 1 busy | 93.3% / 93.4% | 90.8% / 90.8% | busy includes holding a tile it may not write |
| srcA / srcB write-request cycles / elapsed | 72.7% / 62.6% | 64.3% / 57.7% | in absolute cycles 9.83 M / 8.46 M -> 8.32 M / 7.46 M: the unpackers spend 12-15% fewer cycles asking |
| **srcA / srcB requests refused by overwrite protection** | **57.6% / 52.0%** | **49.8% / 45.5%** | of requests; in absolute cycles 5.66 M / 4.40 M -> 4.14 M / 3.40 M, both down ~25% |
| srcA / srcB requests refused by the L1 port | 0% / 0% | 0% / 0% | L1 bandwidth is never the limit, before or after |
| packer busy | 31.7% | 33.5% | same pack work over fewer cycles |
| pack thread stalled | 13.0% | 14.0% | waiting for math to commit a DST half |

The FPU does the same 8.14 M cycles of MVMUL in 4.3% fewer elapsed cycles; every recovered cycle is one in which an
operand was valid earlier. On the unpack side both engines wait less: the refused fraction of write requests drops by
~8 points on each source and the absolute refused cycles by a quarter, and the waiting is now split evenly between the two
unpackers (the old order held srcB's banks longest, so srcB's second refill queued behind its first). The mock ladder
(below) puts the same result in cycles per step: the loop without the packer now sits on the handshake floor.

Mesh, `transformer_op_mesh_bench.py`, host ms per call, same-session A/B, PCC / rel-RMSE identical in each pair:
ff1 fused 16.04 -> **15.69** (-0.35), to_qkv 11.25 -> **11.03** (-0.22), ff2 fused MM/RS (8x7, (6,7,8) 2x2, window 2)
8.98 -> **8.68** (-0.30). Block level (`test_minimax_h3_transformer_block_perf`, 15 s / 768P, 4x8 ring, one run each,
same session, device time per block): the three AGMMs 31.53 -> **31.02** ms and the fused MM/RS 8.62 -> **8.21** with FSDP,
31.28 -> 30.48 and 8.69 -> 8.37 without; device-only total -0.92 / -1.24 ms per block (-0.4 / -0.5%), ~-50 ms per forward.

**Numerics.** Bit-exact by construction and by test: the tt-llk harness golden `tests/python_tests/test_matmul_kloop.py`
(`tests/sources/matmul_kloop_test.cpp`) passes 112 cases (2x2 at kt 4 and 32, 2x1 and 1x2 at kt 4; bf16 and fp32 inputs
and outputs; dest accumulation on and off; four fidelities), and a raw-result comparison of the standard
`matmul_test.cpp` against `matmul_kloop_test.cpp` over the same 128 parameter sets found 0 mismatches
(`torch.equal`, including kt 32 with 16-bit accumulation, which the golden's tolerance does not cover for either kernel).
The single-device bench reproduces the pre-change pcc 1.000000 / rel-RMSE 0.00438 (HiFi2), 0.999879 / 0.02614 (LoFi),
1.000000 / 0.00210 (HiFi4) to the digit.

**Engine isolation, before and after (the method of ff1.md §3.1 exp 16/17: mock one engine, keep its handshakes, read
the `KLOOP` zone).** Same session, `tools/mm_kloop_variants.py apply <v> [legacy]`, cycles per K-tile step, mean of four calls:

| variant | per-tile `matmul_block` (legacy) | `matmul_block_kloop` (alternating) | delta |
|---|---|---|---|
| full loop, HiFi2 | 210.2 | 202.2 | **-8.0** |
| full loop, LoFi | 200.2 | 194.1 | -6.1 |
| full loop, HiFi4 | 281.6 | 279.6 | -2.0 |
| no PACR (unpack + math, real data) | 197.9 | **192.8** | -5.1 |
| unpack mocked (no UNPACR: math + pack, handshakes only) | 192.6 | 192.9 | 0 |
| math mocked (no MVMUL: unpack + pack) | 193.3 | 193.2 | 0 |
| math mocked + no PACR (bare unpack stream) | 188.2 | 187.7 | 0 |

The mock rows do not move: the reorder changes nothing about the four valid/clear round trips per step, and the bare
unpack stream is the same 188. The row that moves is **no PACR: 197.9 -> 192.8**, i.e. with the packer out of the way the
alternating loop runs exactly at the 193 handshake floor the two mocks define. Under the old order the exposed srcB refill
was worth 5 cycles above that floor with no packer and 17 with it; under the new order the refill is fully hidden and the
9 cycles that remain in production (202 vs 193) are the packer's interaction with the loop, the same ~10 cycles exp 16
measured as the `nopack` gain on the old loop. The counters put L1-port refusals at 0% before and after, so it is not L1
bandwidth; the likely account is the DST-half handoff between subblocks (exp 12's `ACQ` 21 + `PWAIT` 26 cycles per 7-step
subblock is ~7 per step), see §0.1. So the loop is now: 128 of MVMUL + ~65 of handshake round trips + ~9 of pack
interaction.

### 0.1 Why the plan was wrong

The plan projected 208 -> ~190 cycles per step (-15..-20) from a reorder; the reorder delivers 210 -> 202 (-8), and only
5 of those 8 are the refill it set out to hide. Four separate errors, each of which the evidence in §2 already contradicted:

**1. It sized the lever from the wrong two numbers.** The plan took the gap between the real loop (208) and the "mock either
engine" floor (193) and called all 15 of it "the exposed srcB refill". But exp 16 and 17 had a third number that bounds the
refill directly: the loop with real unpack, real math and **no packer** ran at 196-200. The refill is, by definition, the
cost of real unpacked data arriving late for real MVMULs; that cost is present in the no-pack loop and absent in the mocks,
so it is at most 198 - 193 = **5 cycles**. The other ~10 of the gap appear only when the packer runs, so they are the
packer's interaction with the loop, not the refill. The plan's §2 dismissed the pack row in one clause ("L1 bandwidth is
idle in every configuration"): the counters do show 0% L1-port refusals, but that does not make the pack free. The no-pack
gain is real and measured; the most likely account is the DST-half handoff between subblocks (exp 12's per-subblock
`ACQ` 21 + `PWAIT` 26 cycles, spread over 7 steps, is ~7 per step), with any remaining L1 read-modify-write interference on
top. Whatever its exact composition, no MVMUL order touches it. The re-run ladder in §0 confirms this split exactly:
nopack 197.9 -> 192.8 (the refill, gone), mocks unchanged, full 210.2 -> 202.2 with ~9 still above the floor.

**2. Its statement of the mechanism was false.** §3's own release table says the production order releases A0 at 64,
B0 at 96, A1 and B1 at 128, and that the next step needs A0 and B0 first, then B1, then A1. The sentence under the table,
"the tiles the next step needs first are the ones the current step releases last", contradicts the table: the two tiles
needed first are the two released **first**. What the table does show is different: the next step's first MVMUL needs one
tile released at 64 (fine) and one released at 96 (late by the refill round trip), and its second MVMUL needs a tile
released at 128 (also late), and under the production order **both late tiles are srcB's**, so the one srcB unpacker
writes them back to back and the second waits for the first. A correct reading of the plan's own table would have
predicted a gain of one serialized refill, not the whole gap.

**3. The proposed pair of orders could not work.** The even/odd "snake" of the old §4 was checked only at the even-to-odd
boundary (odd ① = A1B0, released by even ② and ③: fine). At the odd-to-even boundary its even ① is A0B0, which is exactly
the pair the odd step's ④ (A0B0) released last: the worst possible first MVMUL. The pairing also holds one tile across
①..④ in each order (A0 in the even order, B0 in the odd), which means that source's read bank flips **three** times per
step, so the FPU starts every other step reading bank 1 while the unpacker, whose write bank strictly alternates, has
written the first-needed tile to bank 0. Making that line up needs a period-4 unpacker order (four different per-step
write orders), not the one swap the plan's step 2 describes. The mirror alternation used instead (§4) flips each source's
bank an even number of times in both orders, so the unpacker order never changes; it is the only alternation that both
splits the late refills across the two unpackers and keeps the bank pointers in lockstep.

**4. It did not distinguish "where the waiting happens" from "how much it costs".** The plan's strongest evidence was exp
17b: the mock unpacker spends 70% of its time waiting for a srcB bank to clear and 1% for srcA. That correctly identifies
*which* bank is held, and it is why alternating the held operand between srcA and srcB helps at all. But a mock that does
no data movement waits 70% of the time in *any* scheme (the FPU takes 128 cycles per step and the mock has nothing else to
do), so the 70% says nothing about the number of cycles the real loop loses. The only quantity that sizes the lever is the
step time of a loop with real data on both engines, and the plan never asked for it under a different order before
promising 7-9% of the K loop and ~2 ms per block. The honest projection from the data in hand was "at most 5 cycles per
step, possibly more if the srcB serialization also interacts with the packer", which is what was measured (5 without the
packer, 8 with it).

The cycle model the plan used (release + ~14 handshake + 32 write + ~14 valid = usable ~60 cycles after release) was not
wrong as a picture; it was never calibrated against the no-pack row, and the projection was carried from the picture
instead of from the measurement. The trace itself (§3) was correct and was what made the design in §4 possible; the
conclusions drawn from it in the old §4 and §1 were not.

**What the unpack side taught.** The unpacker's own issue order (srcA MOP before or after the two srcB unpacks) makes no
difference (193.4 / 201.6 / 279.5 vs 193.7 / 201.7 / 279.5): unpack instructions queue per unpacker, so only the FPU's
release order matters, which is why the fix lives entirely on the math side.

**Can a smarter or more frequent order do more? No.** The alternating loop without the packer is at the mock floor, so no
MVMUL order can take another cycle: the floor is set by four handshakes per step, and every 2x2 order has exactly four. The
remaining production gap is the packer's interaction with the loop (the DST-half handoff, not L1 bandwidth), which no order touches. Below 193 only fewer handshakes or fewer bytes
per tile-MAC help: a 2x4 / 4x2 subblock (6 unpacks and 6 handshakes per 8 tile-MACs; needs fp32 dest off) and bfp8
operands, and on the pack side fewer DST handoffs per step (a larger K_block halves them but measured slower for other reasons,
ff1.md exp 1; fp32 dest off halves the pack bytes). The alternation should be carried into the 2x4 / 4x2
LLK orders when fp32 dest off lands (the same held-bank structure at the step boundary), which is the one follow-up here.

## 1. Why this is worth doing

| what | size | numerics |
|---|---|---|
| this change (projected from exp 17b; **measured -3.8% of the K loop, §0, and why the projection was high, §0.1**) | -7..-9% of the K loop on four ops; ~-1% of the block | none |
| fp32 dest off, 2x4 subblock (exp 2/15, measured) | -9.6% of the ff1 K loop | accumulator error x2, needs a model-level check |
| bfp8 in1 (projected) | ~-25% of the K loop | weight precision |
| ff2 fused MM/RS (landed 2026-09-23) | -166 ms per forward | none |
| ff1/ff2 blocking sweep (landed) | -70 ms per forward | none |

It is the only matmul lever left that costs nothing in precision, it is the size of the fused MM/RS landing, and it stacks with
the precision levers (a 2x4 subblock has the same held-bank structure at its step boundary).

## 2. What is established (do not re-measure)

One K-tile step of the production loop (ff1 plain, (8,7,10) 2x2, fp32 dest, HiFi2, single device) is **208-209 cycles** for
4 unpacks and 4 tile-MACs; the FPU's own work is 4 x 32 = 128. Measured on both galaxies (ff1.md exp 16/17):

- Mock either engine's data movement, keep only its valid/clear handshakes: **193** either way. Bare unpack stream: 186-188.
- Hardware counters on the real loop: FPU active 60% = exactly "source data ready"; math thread stalled 0.5% with instructions
  pending 97%; unpacker write requests refused by **overwrite protection** 52-58%, by the L1 port **0%**.
- Counters on the mocks (exp 17b): unpack mocked -> refusals 0%, FPU 66% = 128/193, the mock unpacker waits for a **srcB** bank
  to clear 70% of the time and for srcA 1%; math mocked -> the real unpacker writes **32 cycles per 2 KB tile** with no refusals
  and is busy-but-not-writing ~16 cycles per tile.

So: 46 cycles per tile = 32 of data + ~14 of per-tile handshake; the ~185-193 floor is four handshakes per step; the extra
15-20 of the real loop is the srcB refill the 2x2 reuse scheme exposes. L1 bandwidth is idle in every configuration.

## 3. The current scheme, traced from the source

All line numbers are `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/` at `f1575653d72` + this working tree. For ct_dim = rt_dim = 2
the LLK takes `reuse_a = (ct_dim >= rt_dim)`, `t_dim = 2`, `rut_dim = 2`.

**Unpacker, `_llk_unpack_AB_matmul_` (`llk_unpack_AB_matmul.h:287-447`), per K tile:** one context acquire (`:335`), then two
srcB writes — B0 (`:358`) and "one more tile into srcB" B1 (`:362-386`) — then `TT_MOP` (`:439`) runs the unpack MOP program
(`_llk_unpack_AB_matmul_mop_config_`, `:39-168`) which writes `ct_dim` = 2 srcA tiles A0, A1. Banks alternate automatically
per write. Fixed order every step: **B0, B1, A0, A1**. Unpacker 1 serves srcB, unpacker 0 serves srcA, in parallel.

**Math init, `_llk_math_matmul_init_` (`llk_math_matmul.h:767-818`):** for `t_dim > 1` and `reuse_a` it **disables the
automatic srcB valid clear** (`:788`, `CLR_DVALID_SrcB_Disable`), so the MOP's end-op cannot release B tiles; only explicit
`CLEARDVALID`s do. `matmul_configure_mop` (`:334-491`) records the 16-MVMUL replay image once (`:364`) and one end-op
(`:476-486`).

**Math execute, `_llk_math_matmul_` (`:847-997`), the four MVMULs and bank operations:**

| # | MVMUL (srcA x srcB) | dst tile | bank op after it | releases | at (cycles, HiFi2) |
|---|---|---|---|---|---|
| ① | A0 x B0 | 0 = (r0,c0) | `SETRWC CLR_B, SET_ABD` (`:934`): switch B read bank, B0 stays valid (clear disabled) | — | 32 |
| ② | A0 x B1 | 2 = (r1,c0) | `SETRWC CLR_AB, SET_ABD` (`:977`): clear A0, switch A and B banks | **A0** | 64 |
| ③ | A1 x B0 | 1 = (r0,c1) | `CLEARDVALID CLR_B` (`:929`): clear B0, switch B bank | **B0** | 96 |
| ④ | A1 x B1 | 3 = (r1,c1) | `SETRWC CLR_A` + `CLEARDVALID CLR_B` (`:972-973`) | **A1, B1** | 128 |

**Why a load is exposed every step.** A bank can be rewritten only after its release; a rewrite is ~32 cycles of data plus
~14 cycles each way for the release and the valid to be seen, so a tile released at `t` is usable at about `t + 60`. The next
step needs A0' and B0' at 128, B1' at 160, A1' at 192:

| tile | released | usable | needed | slack |
|---|---|---|---|---|
| A0' | 64 | ~124 | 128 | ok |
| B0' | 96 | ~156 | 128 | **~28 late** |
| B1' | 128, and the srcB unpacker is still writing B0' until ~142 | ~188 | 160 | **~28 late** (overlaps the first) |
| A1' | 128 | ~188 | 192 | ok |

The tiles the next step needs first are the ones the current step releases last. This is the 70% "waiting for srcB clear" of
exp 17b. (The cycle latencies are inferred from the counters; **confirm this trace by hand before writing code**, §5 step 1.)

## 4. The change, as built

Alternate the standard reuse-A order with its A/B mirror between consecutive K tiles. dst indices follow the operand pair
(row = in0 tile = SrcB, column = in1 tile = SrcA), so every dst tile accumulates the same products in the same order:

| even step (reuse-A, `llk_math_matmul.h:1080`) | releases | odd step (reuse-B, `:1098`) | releases |
|---|---|---|---|
| ① A0 x B0 -> dst 0 | | ① A0 x B0 -> dst 0 | |
| ② A0 x B1 -> dst 2 | A0 @ 64 | ② A1 x B0 -> dst 1 | B0 @ 64 |
| ③ A1 x B0 -> dst 1 | B0 @ 96 | ③ A0 x B1 -> dst 2 | A0 @ 96 |
| ④ A1 x B1 -> dst 3 | A1, B1 @ 128 | ④ A1 x B1 -> dst 3 | A1, B1 @ 128 |

Even step needs, from the odd step before it: A0 (released @96, late), B0 (@64), then B1 @160 (released @128; srcB's
unpacker is idle, having written B0' early), then A1 @192 (released @128; srcA's unpacker is still writing A0'). Odd step
mirrors it with A and B swapped. Under the production order every step had both late refills (B0 and B1) on srcB's unpacker.

Bank operations (ISA: `SETRWC` with `CLR_A/CLR_B` always flips the FPU's read bank and hands the bank back only if the
matching `CLR_DVALID_Src*_Disable` bit is clear; `CLEARDVALID` always hands the current bank back and flips unless its
"keep reading same src" bit is set). With both disable bits set at init (`:1033`), even step: `SETRWC CLR_B` (flip to B1),
`CLEARDVALID CLR_A` + `SETRWC CLR_B` (A0 back, flip to A1 and back to B0), `CLEARDVALID CLR_B` (B0 back, to B1), then
`SETRWC CLR_NONE SET_ABD` + `CLEARDVALID CLR_AB`. Odd step: `SETRWC CLR_A`, `CLEARDVALID CLR_B` + `SETRWC CLR_A`,
`CLEARDVALID CLR_A`, `SETRWC SET_ABD` + `CLEARDVALID CLR_AB`. Each source's read bank flips an even number of times per
step in both orders (A: 2 / 4, B: 4 / 2), so the FPU reads bank 0 first in every step and the unpacker's fixed write
order (B0, B1 into SrcB; A0, A1 into SrcA via the unpack MOP, `llk_unpack_AB_matmul.h:287-447`) needs no change. The
disable word is `CLR_DVALID_SrcA_Disable_ADDR32` (= the SrcB one, WH address 5, masks 0x1 / 0x2); every other math op's
init writes it to 0 (datacopy, eltwise binary, reduce, transpose, tilize), and the standard matmul init rewrites it, so
the fused epilogues that follow the K loop in the three kernels are unaffected.

Scope: 2x2 only (`ct_dim == rt_dim == 2`, full tiles, no throttle); other blocks and Blackhole / Quasar take the standard
path inside the same API. A 2x4 / 4x2 variant is a follow-up once fp32 dest off is decided.

## 5. Validation ladder, as run (2026-09-23 evening)

1. **Feasibility, no code** — done by hand against `_llk_math_matmul_` (`llk_math_matmul.h:847-1005`) and
   `_llk_unpack_AB_matmul_` (`llk_unpack_AB_matmul.h:287-447`) plus the ISA pages for `SETRWC`, `CLEARDVALID`, `MVMUL` and
   `SrcASrcB` (tenstorrent/tt-isa-documentation, WormholeB0/TensixTile/TensixCoprocessor). Outcome in §0 and §4: the
   snake pairing dropped, mirror alternation chosen.
2. **LLK unit tests** — the tt-llk harness runs on this host without sudo: create `tt_metal/tt-llk/tests/.venv`, `uv pip
   install -r requirements.txt`, download the SFPI release `sfpi-info.sh` names into `tests/sfpi` (the two steps of
   `setup_external_testing_env.sh` that need sudo or install pre-commit hooks are the only ones skipped). Then
   `cd tests/python_tests && source ../.venv/bin/activate && pytest test_matmul_kloop.py -q`: 112 passed in 4 s. Raw-result
   comparison with the standard kernel over 128 parameter sets: 0 mismatches (§0). The matmul golden's tolerance does not
   cover kt 32 with 16-bit dest accumulation: the standard `matmul_test.cpp` fails those 16 cases identically, so the
   golden test runs kt 32 with fp32 accumulation only.
3. **Single device** — `python -m tracy -r -p tools/transformer_op_single_device_bench.py --op ff1 --no-fusion --fidelity
   LoFi,HiFi2,HiFi4 --cases "8,7,10,2,2,1" --iters 3` with the `KLOOP` zone of `tools/mm_kloop_variants.py` (which now
   targets the `matmul_block_kloop` call and the kloop LLK entry points for its mocks); baseline by forcing the API's
   fall-through (`#if 0` on the `ARCH_WORMHOLE` guard of `matmul.h:301` / `:348`). Numbers in §0. Counters:
   `TT_METAL_DEVICE_ARCH=wormhole_b0 python -m tracy -r -p --profiler-capture-perf-counters=fpu,pack,unpack,instrn
   --perf-counter-multipass ... --fidelity HiFi2 --iters 2`, then `tools/tensix_perf_counters.py generated/profiler/.logs/profile_log_device.csv`.
4. **Mesh** — `transformer_op_mesh_bench.py --op ff1`, `--op to_qkv`, `--op ff2 --fused --mm-grid 8x7 --blocks 6,7,8,2,2
   --window 2`, each under `timeout 600`, new and baseline back to back; no hang, numerics identical (§0).
5. **Block** — `scripts/run_safe_pytest.sh --profile "test_performance_minimax_h3.py::test_minimax_h3_transformer_block_perf[wormhole_b0-sp_sim1-15s_768p-4x8sp1tp0nl4_ring_is_fsdp1]" -o timeout=1500`
   (and `_fsdp0`), new and baseline back to back, compared with `tools/block_profile_stats.py compare`: numbers in §0.
   The 10-step pipeline A/B was not run.

## 6. Risks and recovery

- **Hangs.** A wrong bank switch or an un-cleared valid deadlocks unpacker and FPU inside the kernel. On a single device
  the process times out and the device recovers on the next open; on the mesh a hung ring op wedges the Ethernet heartbeat
  and needs `tt-smi -r all` + 75 s ([ff1.md](ff1.md) §6 Recovery). Do steps 2-3 before touching the mesh.
- **The clear-disable.** `CLR_DVALID_SrcB_Disable` (`:788`) is set once at init and is what makes B tiles survive the MOP
  end-op. The kloop init sets **both** bits (`:1033`), so any op run after the K loop in the same kernel must reset the word
  in its own init (all in-tree math inits that clear source banks do; a new op that does not would hang on its first
  `SETRWC CLR_A`). Any parity in which srcA is the operand reused in the last pair needs the same treatment on srcA, or fully explicit
  clears. Getting this wrong produces a hang or, worse, a silently stale operand (wrong numerics with no error) — hence the
  golden test at kt 32, not only kt 1.
- **DPRINT on the Galaxy**: `TT_METAL_DPRINT_CHIPS=all` plus a core range (never `CORES=all`); virtual core = logical + 18.
- **Do not attach hardware counters to the ring op** (hangs in its first call, ff1.md §3.1); single device only.
- **Numerics** are unchanged by construction (same products into the same dst tiles, same order per tile within a K tile);
  verify anyway at kt 32 with fp32 dest on and off.

## 6b. Follow-on: the full order space

Every other way of ordering the MVMULs within and across K tiles (the three within-step classes, period-1/2/4 schedules, the
unpacker-side enablers, the packer-interaction experiments that own the remaining 10 cycles, and the 2x4 / 4x2 space that opens
with fp32 dest off) is enumerated with expected gains and an order of work in
[kloop_order_space_plan.md](kloop_order_space_plan.md).

## 7. Out of scope, and the levers this does not replace

The handshake floor (~185-193 per step) stays; the alternating loop without the packer sits on it, and the ~9 cycles above it in production are the packer's interaction with the loop, most likely the per-subblock DST handoff (§0, §0.1). Below it only fewer handshakes and bytes per tile-MAC help: the 2x4 subblock
with fp32 dest off (measured, [ff1.md](ff1.md) §3.1/§3.2) and bfp8 operands (projected), both precision decisions for the model
owner. The SwiGLU LUT silu ([ff1_swiglu_lut_handoff.md](ff1_swiglu_lut_handoff.md)) is independent of all of this.
