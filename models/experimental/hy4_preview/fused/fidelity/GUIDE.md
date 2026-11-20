<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fidelity of the fused row matmul: what the multiplier does, `hifi_mm`, and how to use it

Everything here was measured on one Blackhole chip of the Galaxy, one core, 2026-11-19, with the tests of this
directory (logs `mk9_*` in `generated/hy4_preview/logs`, raw data in `generated/hy4_preview/logs/mk9_probe`). "Exact" means
equal bit for bit to float64 arithmetic on the operands as stored.

## 1. What a fidelity phase multiplies (Blackhole MVMUL)

An MVMUL multiplies 8 rows of SrcB (the row operand, in0) by a 16 x 16 face of SrcA (the weights, in1) and adds the
8 x 16 sums to DEST. The multiplier does not take whole operands. A 2-bit phase counter (moved by address mods:
`addr_mod_t.fidelity`, `ckernel_addrmod.h:60`, cleared by `SETRWC` with `SET_F`) selects which significant bits of
each operand enter it. Counting significant bits from the leading one:

| phase | SrcA (weight) | SrcB (row operand) |
|---|---|---|
| 0 | bits 1..5 | bits 1..7 |
| 1 | bits 6.. (to 10 or 11, see below) | bits 1..7 |
| 2 | bits 1..5 | bits 8..11 |
| 3 | bits 6.. | bits 8..11 |

Both splits are truncations (no rounding). Each phase's product is exact and the phases add up in DEST, so
LoFi = phase 0, HiFi2 = phases 0, 1 (the whole weight, the row operand cut to 7 bits), HiFi3 = 0, 1, 2 (everything
but low x low), HiFi4 = the whole product. `MathFidelity` in the standard matmul is exactly this: the same MVMULs
repeated N times on the same SrcA / SrcB contents with the phase stepping (`llk_math_matmul.h:63, 89-145, 495`).

What the operands carry: BF16 8 significant bits, BFP8 at most 7, BFP4 at most 3 (both normalised per value: the
block's shared exponent does not matter to the multiplier), FP32 unpacked to SrcB 11 bits, to SrcA 10 bits.

Evidence (`tests/test_probe.py`, log `mk9_board_all_1`): 12,288 single products per operand pair and schedule
(BFP4 / BFP8 / BF16 weights x BF16, FP32 x FP32; LoFi, HiFi2, HiFi3, HiFi4, phases {0, 2}, and the tree's
`custom_mm`) equal this model in 100.00 %; 16,384 one-hot weight reads per format equal "stored, cut to 5
significant bits" at LoFi and "stored" at HiFi2 in 100.00 % (BFP8 with random weights, with a mantissa sweep under a
pinned block exponent, without the pin, and with no other non-zero weight in the face column; BFP4 is exact at
every schedule); 4,096 reads of x equal "x cut to 7 bits" at LoFi / HiFi2 and x at HiFi3 / {0, 2}.

So, for the operands of the fused pieces (BF16 rows):

| weights | LoFi | HiFi2 | {0, 2} | HiFi3 | HiFi4 |
|---|---|---|---|---|---|
| BFP8 | weight cut to 5 bits, x to 7 | x cut to 7 bits | weight cut to 5 bits | low x low missing (2^-12) | exact |
| BFP4 | x cut to 7 bits | same as LoFi | exact | exact | exact |

The sum of 16 products inside one MVMUL is not exact either: products far below the largest of the 16 are rounded
at about 2^-10 of it. This is the floor of every schedule (1.7e-4 to 4.1e-4 relative rms on real weights, the same
in `ttnn.matmul` at HiFi4); its mechanism is only partly modelled (`tests/test_dense.py` `test_one_face`).

## 2. Why `custom_mm` is LoFi only, and what else it loses

Nothing in its unpack or DEST layout depends on the fidelity: the unpacker sets data valid once per weight tile and
once per K tile and waits for the math thread however many MVMULs it runs. The restriction is in the math thread's
MOP: the template's two loop levels are spent on (1, ct_dim) and no address mod carries a phase step
(`llk_math_custom_mm.h:49-123, 173-234`). `hifi_mm` uses (ct_dim, passes) instead, which is what the standard
matmul's row MOP does.

`custom_mm` loses precision in a second place, at every fidelity: the SplitAcc layouts accumulate the second K face
of each K tile in separate DEST rows and `finalize` merges them by moving them through SrcB with
`MOVD2B(dest_32b_lo = 0)`, which takes the upper 16 bits of the FP32 datum: **the second accumulator is cut to BF16
(8 significant bits)**. Measured: 2.0e-3 to 2.7e-3 relative rms and a gain of 0.998 to 0.999 on the sums, whatever
the fidelity (`tests/test_dense.py`; with 3-bit x and 2-bit weights, 171 of 512 K = 6144 sums are off by up to 7
units of the last product bit, `tests/test_lab.py`). The plain layouts do not have it, but there every second MVMUL
waits for DEST rows the one before last wrote: 2 ns per MVMUL against 1 ns.

## 3. The weights the device holds are not `stored()`: the cause of the "unexplained" errors

`ttnn.from_torch` / `ttnn.as_tensor` with a **sharded** memory config and a block-format dtype convert float32 to
**BF16 first** and then to BFP (`ttnn/core/tensor/py_to_tt_tensor.cpp:372-391` `compute_host_dtype`: the host
buffer is requested as BFLOAT16 when the tile sizes differ; `as_tensor`'s cache is written the same way,
`ttnn/ttnn/operations/core.py:941-975`). The host round trip the goldens use (`layout.stored`, `bfp4_round`: no
memory config) rounds once. Read back from the board (`tests/test_stream_reference.py`, log `mk9_board_stream`):
a sharded tensor equals BF16-then-BFP in 100 % of the entries and the single rounding in 89.5 % (BFP8) / 99.3 %
(BFP4); the two differ by 0.73 % (BFP8) and 2.94 % (BFP4) relative rms.

That is the whole of the three findings that were open:
* projections' probe, BFP8 "93.65 % equal to stored cut to 5 bits" (95.13 % over all 256 weight rows here): against
  BF16-then-BFP8 the same program gives
  100.00 % (0 of 8,192 entries off), through `custom_mm` and through `hifi_mm` (`tests/test_projpath.py`
  `test_stream_probe`);
* projections' probe, "a BFP4 weight one step off": 0.8 % of the BFP4 entries of that weight are a different (and
  equally good) rounding; against BF16-then-BFP4 the readback is 100.00 % exact. BFP4 is exact through the
  multiplier at every fidelity;
* the MoE's 3 % rms of gate|up "against exact arithmetic on the same BFP4 weights" and the projections' 3.05 %:
  the reference used other weights than the device has. With the right reference a LoFi BFP4 matmul is off by
  0.4 to 0.55 %, all of it the 7-bit cut of x and the BF16 merge.
* (the probe's "x rounded to 7 bits, 1.99609375 comes back as 2.0" was the host: that value has 9 significant bits
  and BF16 conversion rounds it to 2.0. LoFi cuts x to 7 bits: 1.9921875 comes back as 1.984375.)

It costs nothing against the true weights for one term (BFP4 11.31 % against 11.29 %, BFP8 0.76 % against 0.70 %),
but it **breaks the two-term BFP4 form wherever the stream is created as a sharded tensor from float32 values that
are not BF16**: the second term is computed on the host for `BFP4(w)` while the device holds `BFP4(BF16(w))`, one
step away in 0.7 % of the entries. Host arithmetic on q_a of layer 1, chip 13 (`tests/test_two_terms_host.py`, log
`mk9_host_two_terms`): two terms as modelled 0.88 % from the float32 weights, **as stored 3.4 %**, with pre-rounded
terms 0.88 % (BFP8: 0.72 %). On the board (projections, random weights, logs `mk9_proj_x2`, `mk9_proj_x2_pre`): the
`bfp4x2` o_proj partial is 4.3e-2 from the reference as stored and 1.3e-2 with `ProjProgram(pre_rounded=True)`.

Who is affected:
* fused/projections: every stream (they are created with the sharded config, directly or through the cache, from
  weights with a norm weight folded in, which are not BF16 values). One term: only the golden was off
  (`layout.stored_stream` is the stream's value now). Two BFP4 terms: a real loss until the stream is built with
  `pre_rounded=True`.
* fused/moe with random float32 weights through `layout.to_device_stream` (its tests): the "3 % against exact
  arithmetic" of program.py's notes. `to_device_stream(host_round=True)` packs on the host.
* fused/moe with the prepared checkpoint streams: **not affected**. `prepare.py` packs on the host
  (`prepare.host_stream`, one rounding) and the checkpoint's weights are BF16 values, so are `w - BFP4(w)`: the
  prepared two-term streams already hold `BFP4(w)` and `BFP4(w - BFP4(w))`, and a sharded tensor of the same floats
  would too (`fused/moe/tests/test_prerounded_host.py`, log `mk9_host_prerounded`: equal in every entry, and equal
  to the files on disk). No new MoE streams are needed.

The packer is not a source of loss either: an FP32 DEST value packed into a BF16 buffer is rounded to the nearest
BF16, halves away from zero, not cut (`tests/test_lab.py` `test_packer_bf16`, log `mk9_board_packer`).

Fix (in `layout.py` of both pieces, no kernel change): give the stream tensor values that are already
representable, so that neither the BF16 hop nor the second rounding changes them:

    q1 = stored(w, "bfp4")                 # host, one rounding
    terms = [q1, stored(w - q1, "bfp4")]   # instead of [w, w - q1]
    stream = stored(stream, fmt)           # one-term streams: the same idea

and keep `stored()` as the golden. (Or keep the streams and make the golden `stored(bf16(w))`; then two terms must
use `q1 = stored(bf16(w))`: 0.93 %.)

## 4. `kernels/hifi_mm.h`

Include `models/experimental/hy4_preview/fused/fidelity/kernels/hifi_mm.h` (it includes `custom_mm.h`). Same
operands, circular buffers, DEST layouts, pack and `tile_regs_*` protocol as `custom_mm`; only the math thread's
instruction stream differs, the unpack and pack LLKs of the tree are called unchanged and no LLK header was copied.

| `custom_mm` call | `hifi_mm` call |
|---|---|
| `custom_mm_block_init<transpose, layout>(in0, in1, out, ct)` | `hifi_mm_block_init<passes, step, transpose, layout>(in0, in1, out, ct)` |
| `custom_mm_block_init_short<transpose, layout>(...)` | `hifi_mm_block_init_short<passes, step, transpose, layout>(...)` |
| `custom_mm_block<layout, finalize, read_transposed>(in0, in1, i0, i1, dst, kt, ct)` | `hifi_mm_block<passes, layout, false, read_transposed>(...)` for EVERY block |
| (finalize = true on the last block) | `hifi_mm_merge<ct, layout>()` after the last block, before any SFPU finishing and the pack |
| | `hifi_mm_merge_init()` once after the init (after an `init_short` too) |
| `custom_mm_block_uninit<layout>(out)` | `hifi_mm_block_uninit<layout>(out)` |

`passes`, `step`: 1, 1 LoFi; 2, 1 HiFi2; 3, 1 HiFi3; 4, 1 HiFi4; 2, 2 phases {0, 2}. `hifi_mm_merge` adds the split
accumulators of the `ct` output tiles (from DEST tile 0) in the SFPU in FP32; `ct` is a template argument. A
`hifi_mm_block` with finalize = true still exists and does `custom_mm`'s BF16 merge. After a merge and a pack the
next `hifi_mm_block` needs no new init (tested with several matmuls per launch).

CB / DEST usage: unchanged. No extra circular buffer, no extra DEST rows (the merge works in place on the rows the
SplitAcc layout already uses), the compute config stays `math_fidelity=LoFi, fp32_dest_acc_en=True,
dst_full_sync_en=True` (the kernel sets the phases itself).

Limits: weight tiles must be full [32, 32] tiles (the narrow [32, 16] form of `custom_mm` is not carried over);
`hifi_mm_merge` assumes the tiles start at DEST tile 0; math-thread code grows by the unrolled merge (4 SFPU
operations per 4 x 8 values, 8 per output tile).

`fused/projections/kernels/proj_compute_hifi.cpp` is that piece's `proj_compute.cpp` with exactly these changes
(`ProjProgram(fidelity=...)` selects it, section 8); `projpath.use_hifi(op, "hifi3")` swaps it into a `ProjProgram`'s descriptor without touching the
piece.

For the MoE kernels (done: `fused/moe/kernels/gate_up_compute_hifi.cpp`, `down_compute_hifi.cpp`, section 8):
* `gate_up_compute.cpp`: `hifi_mm_block_init<2, 2, transpose, dest_layout>`, `hifi_mm_merge_init()`; in the item
  loop every block `hifi_mm_block<2, dest_layout, false>`, then `hifi_mm_merge<2, dest_layout>()` before
  `tile_regs_commit()`.
* `down_compute.cpp`: the same with `hifi_mm_block<2, dest_layout, false, read_transposed>` for every slot and
  term, one `hifi_mm_merge<cols, dest_layout>()` after the last slot.

## 5. Accuracy on checkpoint weights (layers 0-5, chip 13's share; one routed expert on layers 1-5)

Relative rms error / gain of the FP32 sums against exact arithmetic on the stored weights
(`tests/test_accuracy.py`, log `mk9_board_accuracy`; ranges over the layers). `hifi_mm` = split accumulators with
the FP32 merge.

| form | q_a (BFP8) | gate (BFP8) | o_proj (BFP8) | expert gate\|up (BFP4) | expert down (BFP4) | expert chain |
|---|---|---|---|---|---|---|
| `custom_mm` as the pieces run it | 2.1-2.3e-2 / 0.982 | 2.1-2.2e-2 / 0.983 | 1.9-2.0e-2 / 0.985 | 5.4-5.6e-3 / 0.9957 | 4.8-6.4e-3 / 0.996 | 1.3-1.6e-2 / 0.987 |
| `hifi_mm` LoFi | 2.0-2.2e-2 / 0.984 | 2.0-2.1e-2 / 0.984 | 1.8-1.9e-2 / 0.986 | 4.0-4.1e-3 / 0.9972 | 3.6-4.7e-3 / 0.997 | 0.9-1.0e-2 / 0.991 |
| `hifi_mm` HiFi2 | 4.0-4.2e-3 / 0.9972 | 4.0-4.2e-3 / 0.9972 | 3.7-4.0e-3 / 0.9973 | as LoFi | as LoFi | as LoFi |
| `hifi_mm` {0, 2} | 1.8-2.0e-2 / 0.987 | 1.8-1.9e-2 / 0.987 | 1.6e-2 / 0.989 | 1.7-1.8e-4 / 1.0000 | 3.2-4.1e-4 / 1.0000 | 1.4-2.0e-3 / 1.0000 |
| `hifi_mm` HiFi3 | 3.1-3.4e-4 / 1.0000 | 3.1-3.4e-4 / 1.0000 | 2.9-3.3e-4 / 1.0000 | 1.7-1.8e-4 | 3.2-4.1e-4 | 1.4-2.0e-3 |
| `hifi_mm` HiFi4 | 3.1-3.3e-4 / 1.0000 | 3.1-3.3e-4 / 1.0000 | 2.9-3.2e-4 / 1.0000 | 1.7-1.8e-4 | 3.2-4.1e-4 | 1.4-2.0e-3 |
| HiFi4 with `custom_mm`'s merge | 2.0-2.6e-3 / 0.9987 | 2.3-2.4e-3 / 0.9987 | 2.2-2.4e-3 / 0.9986 | 2.2-2.4e-3 / 0.9986 | 2.2-2.7e-3 / 0.9985 | 4.7-6.1e-3 / 0.996 |
| `ttnn.matmul` LoFi / HiFi2 / HiFi4 | equal to `hifi_mm` LoFi / HiFi2 / HiFi4 to every printed digit | | | | | |
| the format alone (stored against checkpoint) | 6.9-7.7e-3 | 7.0-7.4e-3 | 6.7-7.4e-3 | 0.106-0.119 | 0.114-0.117 | |

PCC: `custom_mm` 0.99989 to 0.99993 (BFP8), 0.999992 to 0.999995 (BFP4); HiFi2 0.999995 to 0.999997; HiFi3 / HiFi4
/ BFP4 at {0, 2} 0.9999999 or better (expert chain 0.999998 to 0.999999, bound by the BF16 activation).

Inside the projections' own pre program (their units, stream, finishing and BF16 outputs, kernel swapped;
`tests/test_projpath.py`, log `mk9_board_path`), q_a against exact arithmetic on the weights the stream holds:
`custom_mm` 1.9-2.1e-2 / 0.985, HiFi2 5.0-5.5e-3 / 0.997, HiFi3 and HiFi4 3.3-3.5e-3 / 1.000 (what is left is the
unit's rounding of q_a to 7 bits); against the checkpoint's weights 2.1-2.3e-2, 0.9-1.0e-2 and 0.8-0.9e-2.

## 6. Cost

One matmul with its operands resident in L1 (no DRAM stream, no other core): the lab program run with 1 and 17
matmuls per launch, traced, 20 replays, difference / 16 (`tests/test_cost.py`, logs `mk9_board_all_2`,
`mk9_board_rows16`). us per matmul:

| unit | `custom_mm` | LoFi | HiFi2 / {0, 2} | HiFi3 | HiFi4 | HiFi2 plain layout | HiFi4 plain |
|---|---|---|---|---|---|---|---|
| K 6144 x 2 tiles, BFP8 (q_a, a gate unit) | 3.00 | 3.04 | 3.10 | 4.67 | 6.37 | 5.74 | 12.58 |
| K 6144 x 2 tiles, BFP4 (an expert's gate\|up pair) | 2.74 | 2.78 | 2.97 | 4.67 | 6.37 | 5.74 | 12.58 |
| 48 K tiles x 2, BFP4 (a TP 32 core's piece) | 0.73 | 0.77 | 0.83 | 1.25 | 1.68 | 1.47 | 3.18 |
| 2 K tiles x 8, BFP4, column-major (down) | 0.27 | 0.46 | 0.48 | 0.55 | 0.62 | 0.36 | 0.64 |
| 16 K tiles x 4, BFP8 (o_proj) | 0.54 | 0.63 | 0.67 | 0.95 | 1.24 | 1.02 | 2.16 |
| K 6144 x 4 tiles, BFP8 | 5.50 | 5.59 | 5.93 | 9.32 | 12.73 | 11.47 | 25.11 |

* LoFi is bound by the unpack of the weight tiles (7.1 to 7.8 ns per tile), not by the FPU: with split accumulators
  an MVMUL takes 1.04 ns, so **the second pass is free** (HiFi2 and {0, 2}: +0.06 to +0.2 us on a K = 6144 unit).
  From the third pass on the FPU bounds: 1.04 ns x 4 MVMULs x passes per (K tile, output tile).
* The FP32 merge costs 0.02 us per output tile (0.04 us for 2 tiles, 0.19 us for 8).
* In the projections' pre program (traced, 8 calls per trace): `custom_mm` 24.4 us, HiFi2 24.5, {0, 2} 24.4,
  HiFi3 24.9, HiFi4 25.3 (plain layout 28.1); with the DRAM reads switched off 18.7, 18.7, 18.8, 20.3, 21.8.

Rows: 1, 2, 4 and 8 token rows cost the same (an MVMUL always multiplies 8 rows). **16 rows in one pass over the
weights** is `hifi_mm16_block_init<passes, step, layout>` / `hifi_mm16_block<layout, read_transposed>`: in0 and out
are [16, 32] tiles (two faces of 16 rows), 8 MVMULs per (K tile, output tile) and pass, layout Dense or Default, no
split accumulators and no merge (the 8 MVMULs already rotate over four DEST row groups). Exact on the lab in 34
cases incl. K = 6144; accuracy as the 8-row form. us per matmul, 16 rows:

| unit | LoFi | HiFi2 / {0, 2} | HiFi3 | HiFi4 |
|---|---|---|---|---|
| K 6144 x 2 tiles, BFP8 | 3.89 | 6.32 | 9.75 | 13.16 |
| K 6144 x 2 tiles, BFP4 | 2.93 | 6.32 | 9.75 | 13.16 |
| 48 K tiles x 2, BFP4 | 0.78 | 1.63 | 2.48 | 3.33 |
| 2 K tiles x 8, BFP4, column-major | 0.28 | 0.43 | 0.57 | 0.71 |
| 16 K tiles x 4, BFP8 | 0.58 | 1.14 | 1.71 | 2.28 |

So the weight side of a pass is `max(unpack, 1.04 ns x 4 x ceil(rows / 8) x passes)` per weight tile: going from 8
to 16 rows costs +0.15 to +0.85 us on a K = 6144 x 2 unit at LoFi and doubles everything from two passes on (the
FPU is the bound: +0.4 us per additional row at two passes, +0.85 us at four). 32 rows would need [32, 32] row
tiles (four SrcB faces, 16 MVMULs per tile and pass, 64 DEST rows per output tile); that form is not built.
What limits the rows of one pass in this matmul: SrcB holds one K tile of at most 32 rows; DEST holds 16 FP32
output tiles of up to 16 rows (Dense) or 8 of 32 rows.

## 7. Recommended fidelity per matmul

| matmul | weights | left operand | recommended | accuracy bought (rel rms / gain) | us it costs |
|---|---|---|---|---|---|
| pre: q_a, pool, gate, index query | BFP8 | broadcast x, BF16 | HiFi3 + merge | 2.2e-2 / 0.983 -> 3.3e-4 / 1.0000 | +1.7 per K 6144 x 2 unit; pre program +0.5 |
| the same, if 0.5 us matters | BFP8 | BF16 | HiFi2 + merge | -> 4.1e-3 / 0.9972 (what `ttnn.matmul` HiFi2 gives) | +0.1; pre program +0.1 |
| query: q_b, absorption; post: o_proj | BFP8 | rounded to 7 bits by the unit before (RND_7) | HiFi2 + merge | exact operands: -> 3.0e-4 | +0.06 to +0.1 |
| post: value side of kv_b | BFP8 | ctx, BF16 | HiFi3 + merge | as q_a | +0.4 per 16 K x 4 unit |
| MoE gate\|up, down, shared expert, dense FFN | BFP4 (one or two terms) | BF16 | {0, 2} + merge | gate\|up 5.5e-3 / 0.9957 -> 1.8e-4 / 1.0000; chain 1.5e-2 / 0.987 -> 1.7e-3 / 1.0000 | +0.1 per 48 K x 2 piece, +0.2 per down block of 8 tiles |
| anything with two BFP4 terms | | | pre-rounded terms (section 3) | weight error 3.4 % -> 0.9 % | none |

With HiFi3 available the units no longer need to round their outputs to 7 bits for the next matmul (RND_7 is what
is left of q_a's error at HiFi3 in the pre program: 3.3e-3); BF16 rounding and HiFi3 downstream should be the more
accurate pairing (not measured here).

## 8. The setting in the two pieces (2026-11-19, one chip; defaults unchanged)

`RoutedExperts(..., fidelity="p02")` (fused/moe/program.py; the integration piece: `moe_kwargs={"fidelity": "p02"}`)
runs gate|up and down through hifi_mm at phases {0, 2} with the FP32 merge (kernels `gate_up_compute_hifi.cpp`,
`down_compute_hifi.cpp`; hub, readers, writers, finishers, streams, output layout and SEM_OUT count as before).
Checkpoint layers 1 to 5, the 32 shares summed (logs `mk9_moe_real2`, `mk9_host_real_exact_new`; today's numbers
from the earlier dumps, log `mk9_host_real_exact_old`):

| output | against | `custom_mm` (default) | `fidelity="p02"` |
|---|---|---|---|
| routed experts | exact arithmetic on the stored weights | PCC 0.99998, 1.5e-2, gain 0.9863-0.9866 | PCC 0.999996, 2.8-2.9e-3, gain 0.9991-0.9992 |
| routed experts | the reference | PCC 0.9821-0.9887, gain 0.973-0.976 | PCC 0.9821-0.9887, gain 0.985-0.989 (the BFP4 format's own 0.986-0.990) |
| shared expert (two terms) | exact arithmetic | 1.5e-2, gain 0.9861-0.9864 | 2.4-2.5e-3, gain 0.9992 |
| shared expert | the reference | PCC 0.99985-0.99988, gain 0.986 | PCC 0.99986-0.99990, gain 0.9989-0.9990 |
| chip program (router, experts, shared) | exact arithmetic | 1.5e-2, gain 0.9863-0.9865 | 2.3-2.5e-3, gain 0.9992 |
| chip program | the reference | PCC 0.9854-0.9934, gain 0.978-0.980 | PCC 0.9854-0.9934, gain 0.990-0.992 |
| picks | the reference's 8 per row | 32 / 32 rows | 32 / 32 rows, every layer |
| dense FFN, layer 0 | the reference | | PCC 0.99987, gain 0.9990 |
| traced, unions of 8 / 16 / 32 / 64 (+ shared) | | 100.2 / 116.6 / 151.2 / 207.9 us (same session; 98.0 / 116.6 / 148.4 / 202.4 earlier) | 99.4 / 115.8 / 154.9 / 205.5 us (99.2 / 117.2 / 148.5 / 198.7 in a second run) |

`ProjProgram(..., fidelity="recommended")` (fused/projections/program.py; also `pre_program`, `query_program`,
`post_program`) runs every unit through hifi_mm with the FP32 merge at the cheapest schedule that takes its operands
whole: HiFi3 for the BFP8 units fed by x or ctx, HiFi2 for those fed by tiles rounded to 7 bits (q_b, absorption,
o_proj, the Hadamard basis). Checkpoint layers 0 to 5, chip 13, BFP8, relative rms / gain (logs
`mk9_proj_prec_custom2`, `mk9_proj_prec_rec3`):

| output | default, vs stored weights | recommended, vs stored | default, vs reference | recommended, vs reference | `ttnn.matmul` HiFi2, vs reference |
|---|---|---|---|---|---|
| q_a | 1.9-2.1e-2 / 0.985 | 3.3-3.5e-3 / 1.000 | 2.1-2.3e-2 / 0.985 | 0.81-0.89e-2 / 1.000 | 0.82-0.90e-2 / 0.997 |
| pool tile | 1.9-2.1e-2 / 0.985 | 1.5-1.8e-3 / 1.000 | 2.1-2.3e-2 | 0.77-0.85e-2 | 0.77-0.97e-2 / 0.997 |
| gate (after sigmoid) | 3.7-4.3e-3 | 1.75-1.78e-3 | 3.9-4.6e-3 | 2.2-2.4e-3 | 1.4-1.7e-3 |
| q_a lane sums | 3.0-3.2e-2 / 0.970 | 3.0-4.7e-3 / 1.000 | 3.1-3.4e-2 | 0.52-0.83e-2 | |
| index query (full layers) | 1.9-2.2e-2 / 0.984 | 3.6-3.9e-3 / 1.000 | 2.0-2.4e-2 | 0.71-0.90e-2 | |
| q_abs (q_b, norm, absorption) | 2.5-2.7e-2 / 0.976 | 1.9-3.2e-3 / 1.000 | 2.5-2.8e-2 / 0.976 | 0.32-0.74e-2 / 1.000 | 0.69-0.94e-2 / 0.994 |
| q_rope | 1.4-1.5e-2 / 0.988 | 2.0-2.1e-3 / 1.000 | 1.4-1.5e-2 | 0.42-0.51e-2 | 0.53-0.60e-2 / 0.997 |
| o_proj partial (value side, gate, o_proj) | 3.1-3.2e-2 / 0.973 | 3.4-3.6e-3 / 1.000 | 3.3-3.4e-2 / 0.973 | 1.0-1.2e-2 / 1.000 | 1.2-1.3e-2 / 0.994 |

PCC against the stored weights: 0.9997-0.99999 at the default, 0.99993-0.999999 recommended. Two BFP4 terms
(`bfp4x2`): 1.6-4.1e-3 against the stored weights with "recommended" (2.0e-3 to 1.0e-2 at the default), but still
0.6-3.6e-2 against the reference until the streams are pre-rounded (section 3).
Traced, 8 calls per trace (logs `mk9_proj_lat_custom`, `mk9_proj_lat_rec`): pre 24.4 -> 24.8 us, pre of a full
layer 27.0 -> 27.6, query 24.2 -> 23.8, post 27.3 -> 27.3.

16 rows in the projections' units needs restructuring, not a parameter. `hifi_mm16` is a drop-in for the matmul call
and for the pack (Dense slots, the same SFPU unit numbering with four register groups a face instead of two), but
the piece has 8 rows built into: `layout.ROWS` as a module constant behind every buffer shape and `to_row_tiles`;
the parameter page (`scale[8]` at byte 256, `lanes[8][8]` at byte 288: 16 scales run into the lanes); the constant
page the writer builds for the SFPU (face 0 rows 0..7 the scales, rows 8..15 the lane sums) and the `U_LANES` /
`U_NORM` offsets of `proj_sfpu.hpp`; `finish`'s loops over two register groups. What has to change: ROWS as a
program argument (Python and the `ROWS` define already reach the kernels), a parameter and constant page laid out
for ROWS, `finish` looping over ROWS / 4 groups, `proj_compute*.cpp` calling `hifi_mm16_*` when ROWS is 16. The
matmul of a 16-row unit is measured in section 6 (K 6144 x 2 tiles: 3.9 us at LoFi, 6.3 at two passes, 9.8 at
three); a 16-row unit inside a streamed program was not run.

## 9. Tests

| `fused/moe/tests/test_real.py`, `test_real_exact_host.py`, `test_prerounded_host.py`, `test_latency.py`, `test_routed.py` with `HY4_MOE_FIDELITY=p02` | section 8 | `mk9_moe_real2`, `mk9_host_real_exact_new`, `mk9_host_prerounded`, `mk9_moe_latency2`, `mk9_moe_routed2` |
| `fused/projections/tests/test_precision.py`, `test_latency.py`, `test_pre / query / post.py` with `HY4_PROJ_FIDELITY=recommended`; `test_interface.py` | section 8; the arguments for a caller's program | `mk9_proj_prec_rec3`, `mk9_proj_lat_rec`, `mk9_proj_if`, `mk9_proj_if_rec`, `mk9_proj_xn_rec` |
