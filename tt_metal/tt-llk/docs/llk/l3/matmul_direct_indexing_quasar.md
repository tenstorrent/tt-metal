# Matmul direct indexing on Quasar

## Purpose and scope

Quasar matmul can address its operands two ways:

1. **Counter-driven `MVMUL`** (baseline) — addresses come from the RWC counters, advanced by
   `addr_mod`s. Encoding the traversal this way needs **6 addr_mod slots**.
2. **Direct indexing, `MVMULDI`** — each instruction carries its own source and dest indices.
   Needs **3 addr_mod slots** (`ADDR_MOD_0..2`), leaving `ADDR_MOD_3..7` free.

This document explains both, why the slot count differs, and what the slot contract is. It also
records a third variant — encoding the *dest* index per instruction and unrolling a whole block into
the replay buffer — which was built, measured, and removed (§8).

**The reason DI exists in this codebase is the addr_mod slot count, not performance.** The
performance results are kept as an appendix (§7) because they are informative, not because they
motivate anything.

## Audience

LLK developers working on Quasar matmul, and anyone composing matmul with other ops in a fused
kernel where addr_mod slots are contended.

---

## 1. Register geometry

Three register files, each holding one 32×32 tile as **4 faces × 16 rows** of 16 datums = **64
rows**:

| face | rows | quadrant |
|------|------|----------|
| f0 | 0–15 | top-left |
| f1 | 16–31 | top-right |
| f2 | 32–47 | bottom-left |
| f3 | 48–63 | bottom-right |

The FPU computes `Dest = SrcB × SrcA`. The unpacker crosses the operands — `_llk_unpack_matmul_init_`
argument 0 drives UNPACR1/SrcB, argument 1 drives UNPACR0/SrcA — so with `in0` = A (M×K) and
`in1` = B (K×N):

| face | SrcB (from in0, M×K) | SrcA (from in1, K×N) | Dest (M×N) |
|------|----------------------|----------------------|------------|
| f0 | m0k0 | k0n0 | m0n0 |
| f1 | m0k1 | k0n1 | m0n1 |
| f2 | m1k0 | k1n0 | m1n0 |
| f3 | m1k1 | k1n1 | m1n1 |

One matmul instruction consumes **8 rows of SrcB** (`MATH_ROWS = 8`) against a **full 16×16 SrcA
face**, writing **8 rows of Dest**. So a tile product is **16 instructions per fidelity phase**
(`LoFi = 1`, `HiFi2 = 2`, `HiFi3 = 3`, `HiFi4 = 4` phases; each phase re-runs the same 16
instructions with more mantissa bits, accumulating into the same dest rows).

---

## 2. Baseline: counter-driven `MVMUL` — 6 slots

### 2.1 State

Three address counters (`srca`, `srcb`, `dest`), each with a **carriage-return (CR) base**, plus a
fidelity counter. A plain `MVMUL` carries **no addresses**. Each instruction names one of the
pre-programmed `addr_mod`s, applied *after* the multiply.

### 2.2 addr_mod semantics

| field pattern | effect |
|---------------|--------|
| `incr = N` | counter += N |
| `cr = 1, incr = 0` | counter ← CR base |
| `cr = 1, incr = N` | counter ← CR base + N, **and CR base := that value** |
| `clr = 1` | counter ← 0, CR base ← 0 |

Source addresses wrap modulo 64 rows (see the `cr=32 before, cr+48=16 after wrapping` comment on
`ADDR_MOD_4`).

### 2.3 The six addr_mods

Programmed by `_llk_math_matmul_addrmod_`:

| slot | srca | srcb | dest | fidelity | role |
|------|------|------|------|----------|------|
| `ADDR_MOD_0` | — | +8 | +8 | — | step within a face pair |
| `ADDR_MOD_1` | +16 | ← CR | +8 | — | next SrcA face |
| `ADDR_MOD_2` | ← CR | ← CR +32 | +8 | — | next SrcB face pair |
| `ADDR_MOD_3` | clr | clr | ← CR + `num_tile_incr` | clr | end of tile |
| `ADDR_MOD_4` | ← CR +32 | ← CR +48 | ← CR | — | k0 → k1 transition |
| `ADDR_MOD_5` | clr | clr | ← CR | +1 | end of fidelity phase |

`num_tile_incr = (ct_dim >= rt_dim) ? 64 : ct_dim * 64` rows.

**Six slots is not incidental** — it is the number of distinct counter-step patterns the traversal
requires. §2.4 shows why.

### 2.4 The 16-instruction traversal, traced

15 instructions sit in the replay buffer (`_llk_math_matmul_mop_config_`); the 16th comes from the
MOP. All counters and CR bases start at 0.

| # | addr_mod | SrcB rows | SrcA face | Dest rows | product | counters after |
|---|----------|-----------|-----------|-----------|---------|----------------|
| 0 | 0 | 0–7 | f0 @0 | 0–7 | m0k0·k0n0 | srcb 8, dest 8 |
| 1 | 1 | 8–15 | f0 @0 | 8–15 | m0k0·k0n0 | srca 16, srcb ←0, dest 16 |
| 2 | 0 | 0–7 | f1 @16 | 16–23 | m0k0·k0n1 | srcb 8, dest 24 |
| 3 | 2 | 8–15 | f1 @16 | 24–31 | m0k0·k0n1 | srca ←0, srcb 32 (CR:=32), dest 32 |
| 4 | 0 | 32–39 | f0 @0 | 32–39 | m1k0·k0n0 | srcb 40, dest 40 |
| 5 | 1 | 40–47 | f0 @0 | 40–47 | m1k0·k0n0 | srca 16, srcb ←32, dest 48 |
| 6 | 0 | 32–39 | f1 @16 | 48–55 | m1k0·k0n1 | srcb 40, dest 56 |
| 7 | 4 | 40–47 | f1 @16 | 56–63 | m1k0·k0n1 | srca 32, srcb 16, dest ←0 |
| 8 | 0 | 16–23 | f2 @32 | 0–7 += | m0k1·k1n0 | srcb 24, dest 8 |
| 9 | 1 | 24–31 | f2 @32 | 8–15 += | m0k1·k1n0 | srca 48, srcb ←16, dest 16 |
| 10 | 0 | 16–23 | f3 @48 | 16–23 += | m0k1·k1n1 | srcb 24, dest 24 |
| 11 | 2 | 24–31 | f3 @48 | 24–31 += | m0k1·k1n1 | srca ←32, srcb 48, dest 32 |
| 12 | 0 | 48–55 | f2 @32 | 32–39 += | m1k1·k1n0 | srcb 56, dest 40 |
| 13 | 1 | 56–63 | f2 @32 | 40–47 += | m1k1·k1n0 | srca 48, srcb ←48, dest 48 |
| 14 | 0 | 48–55 | f3 @48 | 48–55 += | m1k1·k1n1 | srcb 56, dest 56 |
| 15 | MOP | 56–63 | f3 @48 | 56–63 += | m1k1·k1n1 | see 2.5 |

Ops 0–7 build the k0 partial products across all four dest faces; op 7's `addr_mod 4` snaps dest
back to row 0 and advances both sources to their k1 faces; ops 8–15 add the k1 contributions into
the same dest rows. That is the K=32 reduction, in place.

Note how much work the `addr_mod`s carry. Op 7 alone must advance srca to f2, wrap srcb modulo 64 to
land on f1, *and* rewind dest to 0 — hence the `+48` with the wrap comment. **Every distinct
movement pattern in this table costs a slot.**

### 2.5 Fidelity loop and end of tile

`ckernel_template(1 outer, FIDELITY_PHASES inner, REPLAY(0,15), matmul_op)` with
`set_last_outer_loop_instr(matmul_op_last)`:

```
for phase in 0 .. FIDELITY_PHASES-1:
    REPLAY 15 instructions              # ops 0-14
    op 15 = matmul_op                   # addr_mod 5: srca/srcb clr, dest <- CR, fidelity += 1
  last phase: op 15 = matmul_op_last    # addr_mod 3: srca/srcb clr, dest <- CR + num_tile_incr,
                                        #             fidelity clr, clear_dvalid = CLR_A or CLR_B
```

With `outer_loop_len == 1` the last inner iteration is also the last outer iteration, so the final
phase's 16th instruction is `matmul_op_last`.

### 2.6 The block loop

`_llk_math_matmul_block_`:

```
_set_dst_write_addr_<Tile32x32>(0);
for t in 0 .. t_dim-1:                       # stationary dimension
    for rut in 0 .. rut_dim-1:               # reused dimension
        run_bank0_sw_cntl();                 # run the MOP - one per tile
        if (rut == rut_dim-1)
            SETRWC(CLR_B or CLR_A);          # release the held operand
    if (!reuse_a && ct_dim >= 2) {
        TT_SETRWC(..., 64*(t+1), SET_D);     # drag dest back to the next column
        TTI_SETRWC(..., C_TO_CR_MODE, SET_D);# re-sync the dest CR base
    }
_reset_counters_<SET_ABD_F>();
```

where `reuse_a = ct_dim >= rt_dim`, `t_dim = reuse_a ? rt_dim : ct_dim`,
`rut_dim = reuse_a ? ct_dim : rt_dim`.

The last pair exists because `addr_mod 3` advances dest by `ct_dim` tiles on every tile: after a
column of `rt_dim` tiles the counter sits past the end of the block and must be pulled back. Since
dest addressing uses `cr`, the CR base then has to be re-synced — that is what `C_TO_CR_MODE` does.

---

## 3. Direct indexing — 3 slots

### 3.1 Encoding

```
MVMULDI = 0x25<<24 | clear_dvalid<<22 | ins_mod<<19 | srcb<<15 | srca<<11 | addr_mode<<8 | dst<<0
```

Field widths: `srcb` 4 bits, `srca` 4 bits, `addr_mode` 3 bits, `dst` 8 bits.

### 3.2 Hardware behaviour

| fact | source |
|------|--------|
| index × 4 rows (`{field, 2'd0}`) | `tt_instruction_issue.sv:2305, 2321` |
| indices are **added to** the RWC counters, not substituted | `tt_instruction_issue.sv:2383` (dest), `:2377` (srcb) |
| math replay buffer = 64 entries (TRISC1/3 = 2⁶, TRISC0/2 = 2⁵) | `tt_instruction_thread.sv:1155` |
| `SETRWC` `rwc_val` is 12 bits (bits 6–17), max 4095 | `ckernel_ops.h:571` |

Because indices are **offsets added to the counters**, direct indexing does not make the stream
independent of counter state — it makes the stream *depend on that state being zero*. This is why
`_reset_counters_` still matters (§5).

### 3.3 The instruction table is the baseline trace, written down

Take §2.4's "SrcB rows / SrcA face / Dest rows" columns and divide by 4:

```
(srcb, srca, dest) =
  (0x0,0x0,0x0) (0x2,0x0,0x2) (0x0,0x4,0x4) (0x2,0x4,0x6)
  (0x8,0x0,0x8) (0xA,0x0,0xA) (0x8,0x4,0xC) (0xA,0x4,0xE)
  (0x4,0x8,0x0) (0x6,0x8,0x2) (0x4,0xC,0x4) (0x6,0xC,0x6)
  (0xC,0x8,0x8) (0xE,0x8,0xA) (0xC,0xC,0xC) (0xE,0xC,0xE)
```

Same 16 products, same order, same in-place K accumulation. Dest indices are even because an 8-row
op covers two 4-row index units. **The arithmetic is identical to the baseline** — only the way each
instruction learns its addresses changed. In the current code these values are written out literally
in `_llk_math_matmul_di_mop_config_`.

### 3.4 The three addr_mods

`_llk_math_matmul_di_addrmod_`:

| slot | srca / srcb / dest | fidelity | selected by |
|------|--------------------|----------|-------------|
| `ADDR_MOD_0` | all zero | — | every replayed op |
| `ADDR_MOD_1` | all zero | +1 | `matmul_op` — end of a non-final fidelity phase |
| `ADDR_MOD_2` | all zero, **dest +`num_tile_incr`** | clr | `matmul_op_last` — end of tile |

`ADDR_MOD_0` must be programmed even though it does nothing: every replayed instruction selects it,
and if left unprogrammed it inherits increments from a previous matmul kernel (a plain `MVMUL`
matmul leaves `dest += 8, srcb += 8` there), which shifts the whole dest addressing. This was a real
bug, not a hypothetical.

### 3.5 Stateful versus self-contained

Baseline instruction words carry no addresses; all sixteen are the same word apart from a 3-bit
`addr_mode` field. Replaying one twice does *different* work because the counter moved — they are
**stateful**, and the six slots are what encode the movement.

DI instruction words carry their addresses, so replaying one twice does the *same* work — they are
**self-contained**, and the only movement left to encode is the fidelity counter and the per-tile
dest step. Hence three slots.

---

## 4. The addr_mod slot budget

This is the section that motivates DI.

| fact | value |
|------|-------|
| slots per thread | **8** (`ADDR_MOD_0` … `ADDR_MOD_7`) |
| cost to program one | **2 MMIO config writes** (`src_val` + `dest_val`) |
| addressing | per-thread: `addr_mod_*_reg_addr[mod] + NUM_MATH_ADDR_MODS * thread_id` |

| matmul variant | slots used | config writes | slots left for other ops |
|----------------|-----------|---------------|--------------------------|
| baseline `MVMUL` (incl. its 2x variant) | 0,1,2,3,4,5 — **6** | 12 | 2 |
| **direct indexing** (incl. DI+X2) | **0,1,2 — 3** | 6 | **5** |

### The contract

**With `ENABLE_DIRECT_INDEXING`, matmul programs `ADDR_MOD_0..2` and never touches
`ADDR_MOD_3..7`.** Verified by inspection: the only `.set(ADDR_MOD_n)` calls reachable from the DI
path are the three in `_llk_math_matmul_di_addrmod_`. Nothing in `_llk_math_matmul_block_`,
`_llk_math_matmul_tile_` or `_set_dst_write_addr_` writes an addr_mod slot (`_set_dst_write_addr_`
writes the dest *section base*, a different config register).

A fused kernel can therefore rely on slots 3–7 surviving a matmul.

### One residual runtime dependence

`ADDR_MOD_2` carries `dest.incr = num_tile_incr`, which is derived from `ct_dim`/`rt_dim` — runtime
arguments. Its *content* therefore changes when the block shape changes, so a kernel alternating
between block shapes re-programs that one slot (2 config writes). Everything else in the DI table is
either all-zero or a compile-time fidelity constant.

If **rewrite frequency** ever becomes a requirement alongside slot count, the fix is small and does
not need dest-DI: drop `num_tile_incr` from `ADDR_MOD_2` and advance dest with an explicit
`TT_SETRWC(..., SET_D)` per tile in the block loop. That makes all three slots constant — programmed
once, never rewritten, for any shape and fidelity — at a cost of one instruction per tile
(≈0.25–1 cyc/tile, measured; see §7). Not implemented, because the stated requirement is slot count.

---

## 5. What the three slots still have to do

Worth knowing why the count is 3 and not lower:

| slot | why it cannot be dropped |
|------|--------------------------|
| `ADDR_MOD_0` (identity) | every instruction must name a slot, and that slot must not move a counter. Its content is generic — an all-zero identity is reusable by any op. |
| `ADDR_MOD_1` (fidelity +1) | multi-phase fidelity needs the counter advanced between phases. Unused at LoFi, where `FIDELITY_INCREMENT` is 0. |
| `ADDR_MOD_2` (fidelity clr + dest step) | ends a tile: clears fidelity and advances dest. |

At LoFi, `ADDR_MOD_1` is never selected and the fidelity clear is a no-op, so the *functional*
requirement collapses to one slot — but the count is still 3 because the slots are programmed
unconditionally. A future refinement could reduce it further; a fused kernel must plan for the
worst case (3) regardless.

`_reset_counters_<SET_ABD_F>()` at the end of a block is **not** vestigial under DI. Because DI
indices are offsets added to the counters, the whole scheme requires the counters to be zero on
entry — direct addressing makes the stream *more* sensitive to inherited counter state, not less.
The DI addr_mods also never clear the source counters (`clr = 0` everywhere), where the baseline's
did as a side effect of walking them.

---

## 6. Instruction accounting

Secondary to the slot count, but useful context. Overhead instructions per block, excluding the
`16 × phases` math instructions per tile:

`tiles + t_dim + (2 × t_dim if !reuse_a && ct_dim >= 2)`

Identical between the baseline and DI: DI changes where addresses come from, not how many
instructions run. An 8-tile tall block (`rt=4, ct=2`) costs 14; a 4-tile row costs 5.

---

## 7. Performance appendix

DI was evaluated for performance before the requirement was clarified as slot count. Kept because it
bounds what any addressing change can do.

### 7.1 Where the cycles are

`perf_matmul_quasar`, bfloat16, cycles per tile-pass (`TILE_LOOP` in `.post.csv`):

| fidelity | math | unpack | pack | end to end | gated by |
|----------|------|--------|------|------------|----------|
| LoFi | 16.8 | 33.2 | 20.3 | 36.3 | unpack |
| HiFi2 | 32.8 | 33.2 | 20.3 | 36.8 | tied |
| HiFi3 | 48.8 | 33.2 | 20.3 | 51.2 | math |
| HiFi4 | 64.8 | 33.2 | 20.3 | 67.2 | math |

End to end lands within ~2.5 cycles of the slowest stage, so the pipeline overlaps well. The gating
stage flips between HiFi2 and HiFi3; at LoFi the math thread has ~16 cycles per tile of slack.
Cycles ÷ instructions is 1.03 at LoFi and 1.007 at HiFi4 — the FPU issues one instruction per clock
and is saturated, so the only math-thread lever is issuing *fewer* instructions.

### 7.2 DI versus non-DI

32 paired configurations:

| run type | median Δ | range |
|----------|----------|-------|
| `MATH_ISOLATE` | **+0.02 %** | −0.14 % … +0.19 % |
| `L1_TO_L1` | −0.18 % | −6.76 % … −0.05 % |
| `UNPACK_ISOLATE` | −0.26 % | −0.87 % … −0.06 % |
| `PACK_ISOLATE` | +0.00 % | −0.12 % … +0.10 % |

No effect, as expected: same instruction count. `UNPACK_ISOLATE` "improved" 0.26 % in a thread DI
cannot touch — that is the noise floor of a single emulator run.

**DI is performance-neutral. It is adopted for the slot count.**

---

## 8. Dest-addressed DI: explored and dropped

A third variant was built and measured: encode the **dest tile index** in every `MVMULDI`'s 8-bit
`dst` field and unroll a whole output block into the replay buffer, so no instruction and no
addr_mod field is spent on dest addressing.

**Why it was tried.** Under a performance framing it removes the per-tile MOP trigger, the
source-release `SETRWC`s and the dest re-base fixups — an 8-tile tall block drops from 14 overhead
instructions to 6.

**Why it is not in the tree.**

1. **It does not improve the slot count.** It programs `ADDR_MOD_0..2`, exactly like plain DI. The
   requirement that motivates DI is already met without it.
2. **Measured performance did not justify the complexity.** The only regime that won end to end was
   single-tile blocks at HiFi2–HiFi4 (−1.0 cyc/tile math, −1.2 % to −2.5 % end to end), because that
   is the only case where math gates *and* the unrolled stream fits 64 entries. Multi-group blocks
   measured **0.13–0.51 cyc/tile slower** despite issuing fewer instructions.
3. **The cost model behind the design was wrong.** Cost tracks **replay invocations**, not
   instruction count — a `TT_REPLAY` carries materially more issue overhead than a MOP trigger. The
   variant that replaced 4 MOP replays with 1 replay won; the variant that replaced 8 MOP triggers
   with 2 `SETRWC` + 2 replays lost. This is the reusable finding, and it is inferred from three data
   sets rather than measured directly — a microbenchmark of a bare `TT_REPLAY` against a MOP trigger
   would settle it.
4. **It carried real complexity**: ~290 lines for a stream plan (group size, divisor rule, dest
   stride), a grouped executor, a fallback path, and a template flag threaded through the test
   harness.

**Where to find it.** The full exploration is preserved in history:

| commit | contents |
|--------|----------|
| `5152286d511` | dest-addressed DI with hybrid grouping: plan struct, group-size derivation, grouped executor, test/perf wiring |
| `31e2f0a5d5e` | restriction to the single-replay case, dedicated grouped correctness and perf modules |

`git checkout 31e2f0a5d5e -- tt_metal/tt-llk` recovers the working implementation, which passed
21 462 correctness items on the HiFi4 sweep plus 36/36 on dedicated multi-group shapes.

One durable by-product: those multi-group shapes proved that
`TT_SETRWC(p_setrwc::CLR_NONE, 0, tile * 64, p_setrwc::SET_D)` sets the dest counter to an
**absolute** row value. That was previously an inference from older code, and it is the mechanism
the constant-addr_mod refinement in §4 would use.

---

## 9. Code map

| symbol | role |
|--------|------|
| `_llk_math_matmul_addrmod_` | baseline's six addr_mods (`ADDR_MOD_0..5`) |
| `_llk_math_matmul_mop_config_` | baseline's 15-instruction recording + MOP |
| `_llk_math_matmul_di_addrmod_` | DI's three addr_mods (`ADDR_MOD_0..2`) |
| `_llk_math_matmul_di_mop_config_` | DI's `MVMULDI` recording; also the DI+X2 variant |
| `_llk_math_matmul_init_` | dispatch on `ENABLE_DIRECT_INDEXING`, `ENABLE_2X_FORMAT` |
| `_llk_math_matmul_block_` | block executor, shared by both paths |
| `_llk_math_matmul_tile_` | single-tile executor |

Test wiring: `ENABLE_DIRECT_INDEXING` in `test_variant_parameters.py`, swept in
`test_matmul_quasar.py` alongside the baseline, kernel dispatch in
`tests/sources/quasar/matmul_quasar_test.cpp`.

---

## 10. Reproduction

```bash
export NNG_SOCKET_ADDR=tcp://<emu-host>:<port> NNG_SOCKET_LOCAL_PORT=5555

# correctness - the DI axis is swept automatically
.claude/scripts/run_test.sh run --worktree "$PWD" --arch quasar \
  --test test_matmul_quasar.py --k "Float16_b and LoFi" \
  --sim-path <path>/build/emu-quasar-1x3 --timeout 900 --verbose

# performance
rm -f /tmp/tt-llk-build/temp_perf_data/*
.claude/scripts/run_test.sh run --worktree "$PWD" --arch quasar \
  --test perf_matmul_quasar.py \
  --sim-path <path>/build/emu-quasar-1x3 --timeout 900 --maxfail 99 --verbose
```

`TILE_LOOP` rows in `perf_data/<module>/<module>.post.csv` are cycles per tile-pass; `INIT` and
`KERNEL` rows are absolute. Compare rows differing only in `enable_direct_indexing`.

---

## 11. Open items

- **Slot assignment across ops.** DI frees `ADDR_MOD_3..7`, but the benefit only materialises if
  other ops in a fused kernel do not also target `0..2`. Slot *assignment* may matter more than slot
  count; worth agreeing a convention across ops.
- **Unpack and pack threads not audited.** This analysis covers the math thread. The matmul unpack
  MOP appears to use `UNPACR*_TILE_INC` without addr_mods, but that is unverified, and slots are
  per-thread.
- **Constant addr_mods** (§4) — removes the one shape-dependent slot. Small change, not implemented.
- **`TT_REPLAY` versus MOP trigger microbenchmark** (§8) — would confirm the cost model that
  explains all the dest-DI results.
- **2x formats with DI** (`ENABLE_2X_FORMAT` + `ENABLE_DIRECT_INDEXING`) use the same three slots;
  the 8-instruction 2x stream was not part of this evaluation.
