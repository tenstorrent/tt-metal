# Handoff: Blackhole NVFP4 / MXFP8 dequantization and the shared harness

Branch `feat/blackhole-nvfp4-dequantize`. This file is a separate commit; drop
it before opening a PR. No PR has been opened.

## State in one paragraph

Two experimental ops, `ttnn.experimental.dequantize_nvfp4` (optimized) and
`ttnn.experimental.dequantize_mxfp8` (exact reference SFPU decoder, not yet
optimized), share one kernel-optimization harness, `scripts/quant_harness.py
--format nvfp4|mxfp8`. MXFP8 passes full validation in the harness at
1477 cycles per output tile; optimized NVFP4 runs at about 530. The NVFP4
harness baseline must be re-measured, because the refactor changed the
evaluator hashes.

## Commits added in this round (on top of `e2bf1a08417`)

| Commit | What |
|---|---|
| `e0d5de6da21` | MXFP8 op scaffolding (host op, program factory, reader/writer, placeholder compute), tests, and the harness refactor |
| `ede68294f18` | Reference SFPU decoder for MXFP8 |
| `13ac6a4d10a` | Fix the scale-page race between compute UNPACK and MATH; 16-tiles-per-core regression test |
| `69d0850bc32` | Make the `MXFP8_MATH_SWEEP_*` marker text unique (the harness found it twice) |
| `247137241cf` | MXFP8 performance numbers in the README |

## Layout

| Path | Contents |
|---|---|
| `scripts/quant_harness.py` | Shared harness: `prepare`, `evaluate`, `compare`. One `FormatSpec` per format in `FORMATS` |
| `scripts/quant_harness_plugin.py`, `scripts/quant_loop_stats.py` | pytest plugin (kernel-root override, metrics) and loop statistics |
| `scripts/nvfp4_harness.py`, `nvfp4_harness_plugin.py`, `nvfp4_loop_stats.py` | Thin wrappers keeping the old NVFP4 CLIs |
| `tests/scripts/test_quant_harness.py` | Hardware-free harness tests (with the existing NVFP4 harness tests: 23 pass) |
| `ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4/` | NVFP4 op; `README.md`, `HARNESS.md` (harness principles), `ACCURACY.md`, optimization logs |
| `ttnn/cpp/ttnn/operations/experimental/dequantize_mxfp8/` | MXFP8 op; `README.md` (host/kernel contract, perf), `HARNESS.md` (differences from NVFP4) |
| `ttnn/ttnn/_experimental/mxfp8.py` | Host packer `ttnn.experimental.mxfp8.pack` |
| `tests/ttnn/unit_tests/operations/mxfp8_reference.py` | Independent decoder, exhaustive corpus, exact gate, `CONFIGS`, `PRIMARY` |
| `tests/ttnn/unit_tests/operations/test_mxfp8_*.py`, `test_dequantize_mxfp8.py` | Accuracy survey, candidate screen, perf workload, op tests |

## MXFP8 op

`output = E4M3(code) * 2^(E8M0(scale) - 127)`, one E8M0 scale per 32
horizontally adjacent values. Full contract in `dequantize_mxfp8/README.md`.

- Inputs: packed `UINT8` (1 code/word) or `UINT16` (2 codes/word, tile-plane
  packing as in NVFP4), TILE or ROW_MAJOR, interleaved DRAM/L1. Scales: `UINT8`
  ROW_MAJOR raw E8M0, `[packed_tiles, 32 * planes]`.
- Output: BF16 (default) or FP32, TILE or ROW_MAJOR.
- **Primary/benchmark configuration: `UINT16` input, BF16 DEST
  (`fp32_dest_acc_en=False`), BF16 TILE output.** Same container/DEST choice as
  NVFP4's benchmark.
- `UINT8` input requires `fp32_dest_acc_en=True` (enforced by `TT_FATAL`): the
  UINT8 datacopy produces Integer-32 in DEST. NVFP4 has the same rule. With BF16
  DEST it was 19% wrong.
- Numerics: exact. ≥ 2^128 → signed ∞; < 2^-126 flushes to zero (allowed).
  NaN codes are outside the contract; `pack` rejects them.

### Current kernel (`device/kernels/compute/dequantize.cpp` + `mxfp8_sfpu.hpp`)

Per packed tile:
1. UNPACK reads the scale words with `scale_cb.read_tile_value` (mailbox
   broadcast to MATH/PACK).
2. `copy_tile` puts the packed words in D0.
3. For each plane, the SFPU builds the FP32 bits from code and scale with
   integer exponent arithmetic (`mxfp8_decode_bits`) and writes D(plane+1).
4. Pack `planes` tiles.

The sweep between `// MXFP8_MATH_SWEEP_BEGIN` and `// MXFP8_MATH_SWEEP_END` is
what the harness's `--math-isolate` diagnostic repeats. Each marker must appear
exactly once in the file.

**Race to remember:** MATH must not read a CB page through an L1 pointer after
UNPACK's `pop_front`. UNPACK runs up to two tiles ahead and the reader then
refills the page. This was exact at 1 tile/core and wrong from 4 tiles/core.
`test_mxfp8_many_tiles_per_core` covers it.

### Performance (Blackhole p150b, 1350 MHz, 120 cores, primary config, L1, 26880x2048, 24 samples)

| Metric | Value |
|---|---|
| L1-to-L1 pipeline | 1477 cycles per output tile (MAD 0.09) |
| Isolated SFPU sweep (×128) | 1422 cycles per output tile |
| One call, 55.05 M values | ≈ 490 µs, ≈ 112 G values/s, ≈ 225 GB/s BF16 out |

Harness result: `generated/mxfp8_loop/sfpu-reference-v3/result.json`
(`status: passed`, `validation_level: full`; not committed, `generated/` is local).

### Optimization ideas (not started)

The kernel is ~96% SFPU, so all gains come from the decode sweep:
- Move the scales into DEST with a datacopy (as NVFP4's fast path does with its
  scale tile) instead of the mailbox + per-vector scalar broadcast.
- Drop the generic `v_if` chains: the subnormal path (`magnitude < 8`) and the
  overflow/underflow clamps can likely become branchless integer ops, or a
  LUT, as in NVFP4's `nvfp4_tti.hpp`.
- Hand-scheduled `TTI_SFP*` with replay, as in NVFP4 (`nvfp4_tti.hpp`).
- Decode both planes from one load of each packed vector.

## NVFP4

Unchanged functionally in this round. Behaviour, accuracy gate and harness
principles are documented in `dequantize_nvfp4/README.md`, `ACCURACY.md` and
`HARNESS.md`.

- After the refactor, the NVFP4 harness passed its screen end to end at
  531 cycles per output tile on **120 cores**. The recorded baseline (≈480)
  used **130 cores**, and old results carry different evaluator hashes:
  **re-measure the baseline** before comparing candidates.
- **Known latent bug, not fixed:** the non-fast kernel
  `dequantize_nvfp4/device/kernels/compute/dequantize.cpp` reads scales from
  the CB page in MATH, the same race as the one fixed in MXFP8. The fast path
  (`dequantize_fast.cpp`) copies scales through DEST and is unaffected.

## Running

```bash
source source-env.sh

# Hardware-free harness tests
pytest tests/scripts/test_quant_harness.py

# Op tests (hardware)
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/test_dequantize_mxfp8.py
scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/test_mxfp8_accuracy.py

# Harness loop (see dequantize_mxfp8/HARNESS.md for the full sequence)
python scripts/quant_harness.py --format mxfp8 prepare --path /tmp/mxfp8-candidate
python scripts/quant_harness.py --format mxfp8 evaluate \
    --candidate /tmp/mxfp8-candidate --output generated/mxfp8_loop/candidate-001 \
    --math-isolate --perf-repeats 4
```

Kernel-only changes need no rebuild. Host changes (program factory, validation,
nanobind) need `cmake --build build_Release --target install`.

## Environment caveats

- `/tmp/tt-device.lock` on the dev machine is owned by another user with mode
  `rw-rw----`. `scripts/run_safe_pytest.sh` still prints "Device lock acquired"
  when it can't open it, so runs are not serialized. Use the device job queue
  or fix the lock file; the script should fail when it can't take the lock.
- `python` is only on `PATH` after `source source-env.sh` (or activating
  `python_env`).

## Related branch

`feat/fp8-e4m3-tile-layout` (separate, see its `HANDOFF_FP8_E4M3_TILE_LAYOUT.md`)
lets FP8_E4M3 tensors use TILE layout. A UINT8-packed MXFP8 tensor is
byte-identical to an FP8_E4M3 tensor of the same layout, so with that branch,
MXFP8 data could later be passed as FP8_E4M3 directly. A native-unpack path
would still need 32-bit DEST because of the `program.cpp` FP8 check.
