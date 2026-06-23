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

## Location

This driver is self-contained in the tt-llk tree at
`tt_metal/tt-llk/tests/quasar_eltwise/`. `run_quasar.sh` and
`compile_llk_quasar.sh` derive `TT_METAL_HOME` from their own location, so they
work when invoked from anywhere. Unrelated legacy polaris/neosom RTL-data scripts
(`compile_for_quasar.sh`, `setup_quasar_ttsim.sh`, `run_on_quasar.sh`,
`run_gelu_*_on_polaris.sh`, `generate_polaris_config.py`) remain under
`tt_metal/programming_examples/generic_lut_activation/polaris_test/` and are not
part of this flow.
