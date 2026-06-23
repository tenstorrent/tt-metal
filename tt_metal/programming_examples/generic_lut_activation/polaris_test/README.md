<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Quasar generic-LUT eltwise flow

The Quasar generic-LUT activation kernels (polynomial, rational, and the ported
eval methods: parity, exponent-ALU, Newton-root) are tested through the
**tt-llk pytest harness** running under the craq-sim Quasar `ttsim`. The harness
itself owns the SFPI build and the on-sim execution
(`tt_metal/tt-llk/tests/python_tests/helpers/test_config.py` + `conftest.py`).

## Single entry point: `run_quasar.sh`

```
./run_quasar.sh [-m EVAL_METHOD] [-a ACTIVATION] [-c CSV] [--compile-only] [--no-compile] [-t PCC]
```

For one `(activation, eval_method)` it: builds the config (maps the method to its
Quasar test source + per-method env contract), runs an SFPI compile-only sanity
gate, runs the test under the pinned sim, and compares to the tt-polynomial-fitter
ground-truth golden (PCC / ULP).

| eval_method   | tt-llk test source                          | CSV env var      |
|---------------|---------------------------------------------|------------------|
| `polynomial`  | `generic_lut_activation_quasar_test.cpp`    | `QUASAR_LUT_CSV` |
| `rational`    | `generic_lut_rational_quasar_test.cpp`      | `QUASAR_LUT_CSV` |
| `parity`      | `generic_lut_parity_quasar_test.cpp`        | `QUASAR_LUT_CSV` |
| `expalu`      | `generic_lut_expalu_quasar_test.cpp`        | `QUASAR_LUT_CSV` |
| `newton_root` | `generic_lut_newton_root_quasar_test.cpp`   | `QUASAR_NR_CSV`  |

Sources live in `tt_metal/tt-llk/tests/sources/quasar/`; python drivers (goldens)
in `tt_metal/tt-llk/tests/python_tests/quasar/`.

### Pinned environment (override by exporting first)

```
TT_METAL_HOME=/localdev/nkapre/tt-metal
TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so
TT_METAL_SLOW_DISPATCH_MODE=1   CHIP_ARCH=quasar    # forced by run_quasar.sh
```
SOC descriptor: `tt_metal/soc_descriptors/quasar_32_arch.yaml` (selected by the
ttsim build via `CHIP_ARCH=quasar`).

### Examples

```
./run_quasar.sh -m polynomial  -a gelu
./run_quasar.sh -m rational    -a atanh -c <atanh_n8d8_s1_..._rational.csv>
./run_quasar.sh -m newton_root -a sqrt
./run_quasar.sh -m parity      -a tanh  --compile-only
```

## `compile_llk_quasar.sh` (compile-only gate)

`./compile_llk_quasar.sh [eval_method] [activation]` — compiles the selected
Quasar test source for all three TRISC threads with the real qsr32 SFPI
toolchain, **no device/sim**. A syntax / SFPU-intrinsic gate; mirrors the
include/flag set the pytest harness builds with. `run_quasar.sh` invokes it
unless `--no-compile` is passed.

## Multi-activation sweep

For a sweep across the fitter's best configs (poly + rational), use
`tt_metal/tt-llk/tests/python_tests/quasar/quasar_sweep.sh`.

## Legacy / deprecated scripts

The following predate the tt-llk pytest flow and are kept only for reference.
They reference paths that no longer exist (`tt_metal/third_party/tt_llk`,
`$HOME/sim/qsr`, `quasar_1_arch.yaml`) or rely on the JIT path that does not work
for Quasar. **Do not use them; use `run_quasar.sh`.**

- `compile_for_quasar.sh` — JIT compile attempt (designed to fail).
- `setup_quasar_ttsim.sh`, `run_on_quasar.sh` — old `$HOME/sim/qsr` JIT runs.
- `run_gelu_on_polaris.sh`, `run_gelu_p1_s4_on_polaris.sh`, `generate_polaris_config.py`,
  `config/`, `quasar_mock_cluster_desc.yaml` — polaris/neosom RTL-data plumbing.
