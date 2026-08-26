# Wormhole EMA kernel: scheduling change and measurement status

Change under evaluation: `_compute_ema_math_` in
`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_ema.h`.

> **Read this first: there are no cycle counts in this document.**
> I was not able to get a trustworthy hardware cycle measurement. §3 records exactly how
> far the attempt got and what is left, so it can be picked up rather than restarted.
> What *is* measured here is the instruction count and the bit-exactness of the output.

---

## 1. What changed

`EMA_new = alpha * EMA_old + beta * input`, chained across 4 rows per block.

**Before** — two MADs per row, both on the dependency chain, so each needs an `SFPNOP`
behind it (2-cycle `SFPMAD` write latency):

```
LREG7 = alpha * LREG4          ; SFPNOP    (carry in from previous block)
LREG0 = beta * LREG0 + LREG7   ; SFPNOP
LREG7 = alpha * LREG0          ; SFPNOP
LREG1 = beta * LREG1 + LREG7   ; SFPNOP
... x4 rows
SFPMOV LREG3 -> LREG4          (carry out)
```

**After** — scale the inputs by `beta` up front, leaving one fused MAD per row on the
chain. The four scaling multiplies are mutually independent, so three of them deal into
the chain's latency slots instead of stalling behind it:

```
LREG0 = beta * LREG0                    (pre-scale in0)
LREG1 = beta * LREG1                    (pre-scale in1)
LREG0 = alpha * LREG4 + LREG0           (row 0)
LREG2 = beta * LREG2                    (pre-scale in2 — covers row 0's latency)
LREG1 = alpha * LREG0 + LREG1           (row 1)
LREG3 = beta * LREG3                    (pre-scale in3 — covers row 1's latency)
LREG2 = alpha * LREG1 + LREG2           (row 2)
SFPNOP
LREG3 = alpha * LREG2 + LREG3           (row 3)
SFPNOP
SFPMOV LREG3 -> LREG4                   (carry out)
```

`LREG7` is no longer needed as a temp. No other register pressure change: LREG0-3 rows,
LREG4 carry, LREG5/6 alpha/beta.

## 2. What is measured

### 2.1 Issue slots — counted from the emitted instruction stream

| | before | after |
|---|---|---|
| SFPMAD | 8 | 8 |
| SFPNOP | 8 | **2** |
| SFPMOV (carry) | 1 | 1 |
| **total issue slots** | **17** | **11** |
| per-row critical path | 2 MADs | **1 MAD** |

The MAD count is unchanged — the win is entirely NOP removal plus the shorter chain.
This is a static count, **not** a cycle measurement; see §3.

### 2.2 Bit-exactness — measured on Wormhole n300

The reassociation is not bit-neutral in fp32 (the old form rounded `alpha*prev` alone and
fused `beta*input` into the add; the new one rounds `beta*input` alone and fuses
`alpha*prev`, so results can differ by ~2^-24 relative). It **is** bit-neutral at the
output, because DEST for this kernel is bfloat16, whose resolution is 2^-9 — three orders
coarser than the perturbation.

Verified by dumping raw output bits before and after and diffing:

| axis | values |
|---|---|
| seeds | 0-7 |
| input amplitude | 0.25, 4.0, 64.0 |
| tile counts | 1, 2, 4 |
| outputs compared | **172032** |
| **differing** | **0** |

`test_sfpu_ema.py`: 3 passed.

### 2.3 Scope

Wormhole only. The Blackhole copy of this kernel has no NOPs to remove (BH interlocks) and
both forms issue the same eight MADs, so there is nothing to win there.

---

## 3. Why there are no cycle counts, and what is left

The EMA kernel had **no perf coverage at all** — that is why the change could not be
quantified from the existing suite, and it is the root of everything below. Building that
coverage is the prerequisite, not an optional extra.

### Built, and working

- `tests/sources/sfpu_ema_perf.cpp` — perf driver mirroring `sfpu_ema_test.cpp`, tile loop
  wrapped in `START_PERF_MEASURE("TILE_LOOP")` with `LOOP_FACTOR` repetition. Compiles for
  all TRISCs.
- `tests/python_tests/perf_sfpu_ema.py` — `@pytest.mark.perf`, single variant
  (`dest_acc=No`, `loop_factor=16`, `[128, 64]` → `tile_cnt` 8).
- `helpers/perf/test_schemas.py` — `perf_sfpu_ema` entry, 20 columns, derived with
  `perf_schema_derive.py` rather than hand-written. `test_perf_header_gate.py`: 12 passed.
- `helpers/perf/wide_schema.py` — added `alpha_bits` / `beta_bits`, without which the
  CSV→Parquet conversion fails strict mode.

### Blocked

1. **`MATH_ISOLATE` hangs the unpack/math handshake.** `TENSIX TIMED OUT ... waited for
   Unpacker, Math`. Three attempts, each ruling something out:
   - `_perf_unpack_loop_set_valid(num_faces * TILE_CNT * LOOP_FACTOR)` was `num_faces`x
     too many valids for one `TTI_CLEARDVALID` per tile. Corrected to
     `TILE_CNT * LOOP_FACTOR`, matching `sfpu_reduce_row_max_perf.cpp`. Still hung.
   - `llk_math_ema_sfpu_tile` already brackets itself with
     `_llk_math_eltwise_sfpu_start_`/`_done_`, so calling it inside another bracket
     double-enters the SFPU state machine. Switched to bracketing once and calling
     `sfpu::_calculate_ema_tile_()` directly, which is also the right shape for
     measurement (a per-tile bracket would be charged to the number). Still hung.
   - `_llk_math_eltwise_sfpu_start_` does **not** wait on SrcA valid (it stalls only on
     `STALL_SFPU`/`MATH`), so the hang is not a missing SrcA valid.

   Remaining suspicion: this kernel is stateful across two dst tiles (input 0, output 1)
   and inits through the *ternary* SFPU path
   (`_llk_math_eltwise_ternary_sfpu_init_`), so the isolate path likely needs a different
   valid/dest discipline than the unary reduce source it was modelled on. The
   `quasar-perf-test` skill is the documented owner of `PerfRunType` path bugs.

2. **`L1_TO_L1` runs on hardware but the consumer stopped emitting a report.** Restricting
   `run_types` to `[PerfRunType.L1_TO_L1]` got a real hardware run that reached CSV write
   and failed only on the Parquet schema — fixed by (4) above. After that fix the consumer
   phase completes in 0.11 s with no worker artifacts in
   `/tmp/tt-llk-build/temp_perf_data/` and no `perf_data/perf_sfpu_ema/`, i.e. it is not
   executing the kernel. Not diagnosed.

3. **Each `MATH_ISOLATE` hang wedges the device.** Four `tt-smi -r` resets were needed;
   after the last one, topology discovery threw `IndexError: map::at` until a further
   reset. Budget for that when picking this up.

### Note on method, for whoever finishes it

`--speed-of-light` must match on both runs, and per the `perf-report` skill: move
`perf_data/<module>/` aside rather than trusting an in-place overwrite (a no-op rerun
leaves the previous CSV looking like the new result), and repeat a run before attributing
a small delta to the change. Given that `L1_TO_L1` includes unpack and pack, a math-only
change will show diluted there — `MATH_ISOLATE` is the number worth having, which is why
blocker 1 matters more than blocker 2.

---

## 4. Provenance

| | |
|---|---|
| Branch | `ldjurovic/sfpu_wh_nop_overlap` |
| Base | `f6b36f3b1be` |
| Silicon | Wormhole n300, `CHIP_ARCH=wormhole` |
| Bit-exactness | raw output-bit dump, before vs after, 172032 outputs, 0 differing |
| Functional | `test_sfpu_ema.py` 3 passed |
| Cycle counts | **not obtained** — see §3 |
