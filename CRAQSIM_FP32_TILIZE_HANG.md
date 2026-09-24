# Quasar fp32 tilize: illegal format conversion (real bug) — craq-sim masks it as a hang

> **UPDATED (emulator data): this is a REAL tt-metal/tt-llk bug on Quasar, not a sim-only gap.** My original
> "craq-sim modeling gap, works on real HW" verdict below was **wrong** and is retracted. Keeping the filename
> stable (craq-sim#401 links here).

**Verdict:** producing an **fp32 tile** (and **uint8**) via tilize on Quasar hits a real hardware fault. On the
ZEBU emulator (real RTL) the 32×32 repro fails fast:

```
Neo0TRISC0 hardware fault: UNPACKER_0 (ILLEGAL_FORMAT_CONVERSION)
Current kernel: ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp
```

The **craq-sim functional simulator does not model that trap** — instead of faulting it stalls forever (reader
NTW / compute WFW). So on craq-sim it looks like a hang; on RTL it's an illegal format conversion. The real
differentiator is **`enable_32_bit_dest=true`**: fp32-lossless and uint8 both enable it and fault; `bf16` does
not enable it and works. (NB: on Quasar there is **no separate fast tilize path** — `fast_tilize_block` is a
literal alias for `tilize_block` — so this is not a "fast vs slow path" distinction; it is purely 32-bit-dest.)

Two things to fix, tracked separately:
1. **The real bug (tt-llk):** the Quasar tilize LLK path with `enable_32_bit_dest=true` produces an
   `ILLEGAL_FORMAT_CONVERSION` on real RTL. Reproduced on the emulator via three paths — mainline `ttnn.tilize`
   (uint8 and fp32→bf16) and the canonical `ttnn.experimental.quasar.tilize` (fp32, config built via
   `to_compute_hardware_config`, byte-identical compute kernel) — so it is shared `tt_llk_quasar` LLK code, not a
   factory-specific config bug (rules out the mainline factory's `TODO(#52269)` manual Gen1→Gen2 copy).
2. **craq-sim (issue #401):** ttsim should surface/trap the `ILLEGAL_FORMAT_CONVERSION` like RTL instead of
   hanging; plus the separate `qsr_convert_pack_value` bf16→fp32 (`5→0`) gap.

## Root cause: CONFIRMED + FIXED (fp32) — Quasar tilize never actually unpacked to DEST

**Confirmed root cause:** the tilize factories request **unpack-to-DEST** for fp32 (`UnpackMode::UnpackToDest`), but
Quasar's tilize never implemented that route — it pushed the fp32 tile into a **source register** instead. Real RTL
rejects fp32 in a src register (`UNPACKER_0 ILLEGAL_FORMAT_CONVERSION`); and the math stage waited on an
UNPACK→MATH handshake the unpacker never posted, which is why **craq-sim hung instead of faulting**. It is a
route/handshake bug, **not** a Float32→Tf32 relabel.

**Fix (peer session, validated; tracked as tt-metal #57780 — not on this repro branch):** make Quasar tilize
actually unpack straight into DEST, one tile per DEST section, using the same unpack/math/pack handshake Quasar
`copy_tile` already uses; math becomes a sync-only forwarder (no `ELWADD`). 3 files: `llk_unpack_tilize_api.h`
(UNP_DEST tilize MOP + per-DEST-section execute), `tilize.h` (Quasar branch passes `UnpackToDestEn` to the math
datacopy), `llk_unpack_tilize.h` (runtime `full_ct_dim` overload; compile-time version forwards). Only the
`UnpackToDestEn=true` path changes (it could never work on Quasar before), so bf16/other paths are untouched.

**Validated:** fp32 32×32 (craq-sim: hung→PASS; ZEBU exact-value: fault→PASS), fp32 32×512 (16 tiles, bank flips,
per-tile DEST handoff: PASS), bf16 32×32 regression (PASS). The **exact-value** pass confirms it stays lossless —
which is exactly why the Tf32 relabel below would have been wrong (lossy).

**Still open:** **uint8** faults the same way but its config never requests unpack-to-DEST, so this fix doesn't touch
it — a separate bug (see the issue). Untested at time of writing: larger/multi-core shapes (e.g. 256×2048), tt-llk's
own `unpack_tilize_quasar_test`, metal LLK integration tests, pre-commit.

---

### Why the earlier Tf32-substitution candidate was wrong (kept — it is what pointed at the real route bug)
- The harness Tf32 substitution (`data_format_inference.py`, `infer_unpack_out`) is gated on
  **`not unpacking_to_dest and not unpacking_to_srcs`** — i.e. it applies only on the unpack-to-**SrcA/B** route.
  On the unpack-to-**DEST** route, `Float32` falls through to `return input_format` → **stays Float32**.
- **PR #50728** ("Add missing dest-format configs to stop Quasar tests passing by accident", merged 2026-07-27)
  says its Tf32 rule is for **Float16/Float16_b** inputs ("the HW unpacker has no fp16→Float32 conversion; Tf32 is
  the correct widened SrcS format") and explicitly: **"Native Float32 input stays Float32 (direct passthrough)."**
- **tilize is configured for unpack-to-DEST** for fp32: `tilize_multi_core_default_program_factory.cpp:152` sets
  `UnpackMode::UnpackToDest` (gated on `fp32_llk_acc`; `:149` `enable_32_bit_dest`, `:160` copied to Gen2). So by
  the harness's own logic tilize's `Float32` is the **correct** declared format — a relabel to `Tf32` is both the
  wrong route and **lossy** (it truncates, defeating the lossless path's whole purpose: exact fp32 before the
  bf16 typecast).

So the declared format is what the harness itself would infer as correct → the fault is probably **not** a missing
relabel. Leading remaining lead: a **route-vs-math mismatch** — tilize unpacks to **DEST**, but under
`enable_32_bit_dest` its "unary datacopy" (`llk_math_eltwise_unary_datacopy`) compiles to an **`ELWADD`**, a math op
that reads the **source** registers → data in DEST, math reading SRC. The harness file also notes Quasar unpacker
conversions are under-tested ("for now only conversions performed by the packer are tested"). **uint8 also faults**
(no Tf32 rule applies to it at all) — consistent with the bug being about the DEST/ELWADD path, not a Float32 format
label. **This lead was CONFIRMED** (see the top section): the fix makes tilize genuinely unpack to DEST with the
proper handshake, and fp32 passes on both craq-sim and ZEBU.

### Earlier hypotheses (checked + refuted / contradicted, kept for the record)

- **CONTRADICTED — "missing Float32→Tf32 unpack substitution"** (see above: wrong route — tilize uses UnpackToDest
  where Float32 stays Float32 per PR #50728; and lossy).
- **REFUTED — "fast vs slow path":** on Quasar `fast_tilize_block` aliases `tilize_block`; there is no separate
  fast path.
- **REFUTED — "binary-ELWADD MOP wired for tilize's unary consumer":** with 32-bit-dest,
  `llk_math_eltwise_unary_datacopy` compiles to an `ELWADD`, and `_llk_unpack_tilize_mop_config_`
  (`llk_unpack_tilize.h`, `EN_32BIT_DEST` branch) pokes a dvalid onto the opposite unpacker to feed it. This
  looked like a unary-op-wired-as-binary mismatch — but the codebase's **validated** datacopy reference
  `llk_unpack_unary_operand.h` (the exact case that branch's comment calls out as known-good) uses the

- **REFUTED — "fast vs slow path":** on Quasar `fast_tilize_block` aliases `tilize_block`; there is no separate
  fast path.
- **REFUTED — "binary-ELWADD MOP wired for tilize's unary consumer":** with 32-bit-dest,
  `llk_math_eltwise_unary_datacopy` compiles to an `ELWADD`, and `_llk_unpack_tilize_mop_config_`
  (`llk_unpack_tilize.h`, `EN_32BIT_DEST` branch) pokes a dvalid onto the opposite unpacker to feed it. This
  looked like a unary-op-wired-as-binary mismatch — but the codebase's **validated** datacopy reference
  `llk_unpack_unary_operand.h` (the exact case that branch's comment calls out as known-good) uses the
  byte-identical `TT_OP_UNPACR_NOP` and **also** never programs the opposite unpacker's buffer descriptor. So
  tilize matches the known-working pattern; this is not the difference.

Whatever differs is more subtle (init ordering, tilize's MOP loop structure vs. plain datacopy, or a genuinely
unvalidated RTL gap). Pinning it needs waveform-level diagnosis (the tt-llk `llk-wave-debug` skill / `llk-debugger`,
Quasar-only, requires the RTL simulator or emulator with waveform dumps — not doable from a craq-sim / WH host).

This branch exists only to hand a self-contained reproducer + the localization done so far. Nothing here is meant
to merge.

---

## TL;DR for the craq-sim investigator

- The wedge is the **packer never firing**: `qsr_execute_pacr_tile` (`src/tensix.cpp:17715`) returns false
  forever at `!qsr_dest_dvalid_ready(QSR_DVALID_CLIENT_PACK)`; the compute writer then spins on `wait_front`.
- The upstream cause: for fp32, the Quasar **MATH thread does the datacopy as `ELWADD`** (SrcA = tilized data,
  SrcB = a dummy dvalid set by `UNPACR_NOP`), writing 32-bit DEST; the sim **issues** the `ELWADD` (frontend
  trace shows 8× `ELWADD 0x28`) but **never retires it**, so the `MATH→PACK` semaphore is never posted
  (`sem=[0,...]`, `semmax=[0,2,...]`), DEST never becomes valid, and PACR0 stalls.
- Prime suspect: the dummy SrcB dvalid from `UNPACR_NOP` (`src/tensix.cpp:8760-8806`) is set on the *write*
  bank and does not match the `src_b_matrix_bank` that `ELWADD` reads (`src/tensix.cpp:6037-6046`) — a
  bank-flip / dvalid handshake mismatch — or `qsr_dest_dvalid_ready(MATH)` gating.
- **Second, separate, trivially-clean gap:** `qsr_convert_pack_value` (`src/tensix.cpp:16558`) has **no
  bf16-dest(5) → fp32-L1(0)** case → throws `UnimplementedFunctionality: qsr_convert_pack_value: qsr pack
  in_format=5 out_format=0`. This is the fastest, no-hang reproducer of a real fp32-pack gap.

The prebuilt `libttsim.so` used (Sep 22) is at/after craq-sim `quasar` HEAD `1596be84` (Sep 17), which already
carries the PACR_STRIDE / tile-counter fixes — so this is a *remaining* gap, not a stale binary.

---

## Discriminator matrix (all at 32×32 = 1 tile, 1 core, on craq-sim)

| test (`tests/ttnn/unit_tests/operations/test_quasar_tilize_isolate.py`) | in → out | result | isolates |
|---|---|---|---|
| `test_mainline_tilize_bf16`      | bf16 → bf16 | **PASS**  | fast_tilize path (no ELWADD datacopy) |
| `test_mainline_tilize_fp32`      | fp32 → fp32 | **HANG**  | production case |
| `test_experimental_tilize_fp32`  | fp32 → fp32 | **HANG**  | same LLK path, quasar-native op |
| `test_disc_fp32in_bf16out`       | fp32 → bf16 | **HANG**  | the fp32 **UNPACK / ELWADD** side |
| `test_disc_bf16in_fp32out`       | bf16 → fp32 | **ERROR** | the fp32 **PACK-convert** gap (immediate) |

Common factor of the two hangs = **fp32 input → the ELWADD lossless datacopy**. Not width-dependent (hangs at
32×32). Not mainline-vs-experimental.

---

## How to reproduce

Requires a tt-metal build for Quasar on this branch (the Metal-2.0 tilize port lives here) and a craq-sim
`libttsim.so`.

```bash
# 1. Build tt-metal for Quasar (heavy; ~30-60 min). build_metal.sh returns 0 even on -Werror, so grep for "error:".
source python_env/bin/activate
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD PYTHONPATH=$PWD:$PYTHONPATH
ARCH_NAME=quasar ./build_metal.sh --build-all

# 2. Point at the simulator + Quasar env (TT_METAL_SIMULATOR must be the FULL .so path).
export TT_METAL_SIMULATOR=/abs/path/to/libttsim.so
export TT_SIMULATOR_LOCALHOST=1 TT_METAL_SLOW_DISPATCH_MODE=1
export ARCH_NAME=quasar CHIP_ARCH=quasar MESH_DEVICE=N150 TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"

T=tests/ttnn/unit_tests/operations/test_quasar_tilize_isolate.py

# 3a. FASTEST clean gap (immediate error, no hang): fp32-PACK-convert unimplemented.
python -m pytest -q -p no:cacheprovider "$T::test_disc_bf16in_fp32out" -k 32x32

# 3b. bf16 baseline: PASSES in a few seconds.
python -m pytest -q -p no:cacheprovider "$T::test_mainline_tilize_bf16" -k 32x32

# 3c. The production hang (bound it — pytest's signal timeout can't interrupt the C++ device wait):
timeout -k1 90 python -m pytest -q -p no:cacheprovider "$T::test_mainline_tilize_fp32" -k 32x32
# -> wedges; kill at 90s. bf16 (3b) finished in ~4-6s, so >60s == hang.
```

After a killed hang: `pkill -9 -f 'python -m pytest'; pkill -9 -f libttsim` and confirm no `python` procs
remain before the next run. There is one sim device — never run two sim pytests at once.

### Scaffolding to watch the stall directly

- `TT_METAL_WATCHER=1` → `generated/watcher/watcher.log` names the wedged core/RISC and wait state
  (expect core (0,0), writer kernel at `WFW`).
- craq-sim built-in traces: `TTSIM_QSR_DFB_COUNTER_TRACE=1 TTSIM_TENSIX_FRONTEND_TRACE=1` → `[ttsim-qsr-pack]`
  (emitted only when PACR actually executes) appears **0 times** for the fp32 hang while the tile-counter poll
  fires hundreds of thousands of times (writer spinning); the frontend trace shows the math pipe issuing
  `ELWADD (0x28)` but the `MATH→PACK` semaphore never advancing.
- `TTSIM_SRC_VALID_TRACE` would pinpoint the SrcB-dvalid/bank-flip handshake but is compiled out of the release
  `.so` — pinpointing the production fix needs an instrumented libttsim rebuild.

---

## Root cause references

**tt-metal (correct — the standard lossless-fp32 path, shown here only to explain the sim's inputs):**
- `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:80-83` — fp32 forces `use_fast=false` (lossless path).
- `tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_unpack_tilize.h:76-81` — Quasar fp32 datacopy as `ELWADD`
  (SrcA = tilized data, SrcB = dummy dvalid via `UNPACR_NOP`), 32-bit DEST → PACR0.
- `ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_multi_core_default_program_factory.cpp` —
  `enable_32_bit_dest` + `UnpackToDest` for fp32 (standard config).

**craq-sim (the gap):**
- `src/tensix.cpp:17715` — `qsr_execute_pacr_tile` stalls at `!qsr_dest_dvalid_ready(PACK)`.
- `src/tensix.cpp:6037-6046` — Quasar `ELWADD` gate in `tensix_execute_elw_op` (never passes for the fp32
  datacopy → never retires → MATH_PACK never posted).
- `src/tensix.cpp:8760-8806` — `UNPACR_NOP` set-dvalid (dummy SrcB), suspected write-vs-read bank mismatch.
- `src/tensix.cpp:16529-16560` — `qsr_convert_pack_value`, missing the `5→0` (bf16-dest → fp32-L1) case.

## Proposed sim fixes

1. **Small / safe:** in `qsr_convert_pack_value` add `else if (in_format == 5 && out_format == 0) return
   uint32_t(value) << 16;` (mirrors the existing `5→1`). Fixes only `bf16→fp32`; does **not** fix the
   production `fp32→fp32` hang.
2. **Production fix:** make the sim retire the Quasar fp32 `ELWADD` datacopy so it posts `MATH_PACK` and sets
   DEST valid for PACR0 — investigate the `UNPACR_NOP` dummy-SrcB dvalid vs `src_b_matrix_bank` handshake and
   `qsr_dest_dvalid_ready(MATH)`. Needs an instrumented libttsim rebuild (`TTSIM_SRC_VALID_TRACE`).

## Impact

The llama32_1b weight upload `from_torch(fp32, dtype=bfloat16, TILE)` (upload rowmajor fp32 → tilize fp32 →
typecast bf16) is broken on Quasar: it **faults on real RTL/emulator** (`ILLEGAL_FORMAT_CONVERSION`) and **hangs
on craq-sim**. This is a real Quasar bug in the generic tilize path, NOT sim-only. Workaround until the LLK fix
lands: avoid the generic fp32/uint8 tilize path — cast to bf16 on host first (`from_torch(bf16, …, TILE)`), which
takes the `fast_tilize` path that works on both RTL and the sim.
