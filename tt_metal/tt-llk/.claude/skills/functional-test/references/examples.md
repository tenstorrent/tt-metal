# Functional test examples

These are structural templates. Copy operation-specific formats, constraints,
goldens, parameters, LLK calls, and synchronization from a current test for
the same pipeline and architecture.

## A. Add a new functional/perf family

### Functional module

`tests/python_tests/test_op.py` owns the sweep and runner:

```python
import pytest
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from helpers.perf.blaze import execute_llk

OP_SWEEP = dict(
    formats=...,
    input_dimensions=...,
    dest_acc=...,
)


def _run_op_test(
    formats,
    input_dimensions,
    dest_acc,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(...)
    golden = None if is_perf else get_golden_generator(OpGolden)(...)

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name="sources/op_test.cpp",
            formats=formats,
            templates=[...],
            runtimes=[...],
            variant_stimuli=StimuliConfig(...),
            dest_acc=dest_acc,
        ),
    )
    if is_perf:
        return

    result = result_tensor(outcome.result, formats.output_format, golden)
    assert passed_test(golden, result, formats.output_format)


@parametrize(**OP_SWEEP)
def test_op(formats, input_dimensions, dest_acc):
    _run_op_test(formats, input_dimensions, dest_acc)
```

Rules behind the template:

- Derive block counts, tile counts, formats, and every kernel parameter before
  config construction in the shared runner.
- A value used to configure the perf kernel must be computed in perf mode too.
  Only result collection and correctness checking are functional-only.
- If the operation needs custom pre/post-run device work, use
  `create_test_or_perf_config` directly, as `test_pack_l1_acc.py` does.

### Perf module

`tests/python_tests/perf_op.py` imports, rather than recreates, the program:

```python
import pytest
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_op import OP_SWEEP, _run_op_test


@pytest.mark.perf
@parametrize(**OP_SWEEP, **PERF_MEASUREMENT_AXES)
def test_perf_op(
    perf_report,
    formats,
    input_dimensions,
    dest_acc,
    run_types,
    loop_factor,
    is_perf,
):
    _run_op_test(
        formats,
        input_dimensions,
        dest_acc,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
```

The only new axes are measurement axes. If the full functional sweep is too
large, copy it and replace one key with a documented subset:

```python
OP_PERF_SWEEP = {**OP_SWEEP, "input_dimensions": PERF_INPUT_DIMENSIONS}
```

Every perf value must still exist in functional. Never write
`@parametrize(**OP_SWEEP, input_dimensions=...)`; duplicate keyword expansion
raises `TypeError`.

### C++ source

`tests/sources/op_test.cpp` serves both modes:

```cpp
#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

#ifdef LLK_TRISC_UNPACK
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        // Configure the functional unpack path.
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        // L1_TO_L1 and UNPACK_ISOLATE call the same real unpack helper.
        // MATH_ISOLATE uses only the perf-test-approved partner stand-in.
        PROFILER_SYNC();
    }
}
#endif

// Add the active MATH and PACK sections. A real Quasar fourth-thread stage
// uses an LLK_TRISC_ISOLATE_SFPU section.
```

Follow `perf-test` for the exact run-type branches, stand-ins, dvalid counts,
Dest release, and speed-of-light parameter handling. The functional
`L1_TO_L1` path is the baseline; do not write a second perf loop.

### Quasar form

Use the parallel paths and markers:

```text
tests/python_tests/quasar/test_op_quasar.py
tests/python_tests/quasar/perf_op_quasar.py
tests/sources/quasar/op_quasar_test.cpp
```

The common Quasar perf wrapper imports and calls the functional test callable
directly:

```python
from quasar.test_op_quasar import test_op_quasar as run_op

# Inside test_perf_op_quasar(...):
run_op(
    ...,
    run_types=run_types,
    loop_factor=loop_factor,
    is_perf=is_perf,
    perf_report=perf_report,
)
```

The functional module may expose one generator with an `is_perf` switch:

```python
def generate_op_combinations(*, is_perf=False):
    tile_sizes = (
        select_perf_tile_sizes(SUPPORTED_TILE_SIZES)
        if is_perf
        else SUPPORTED_TILE_SIZES
    )
    ...


OP_COMBINATIONS = generate_op_combinations()
PERF_OP_COMBINATIONS = generate_op_combinations(is_perf=True)
```

This is a documented generator narrowing, not two unrelated sweeps. Follow
`quasar-test-coverage` and the
[Quasar perf reference](../../perf-test/references/quasar.md).

## B. Add behavior under an existing functional module

### Another value of the same behavior

Edit the functional-owned sweep:

```python
OP_SWEEP = dict(
    formats=[EXISTING_FORMAT, NEW_FORMAT],
    input_dimensions=...,
    dest_acc=...,
)
```

Do not add a second function merely to hide one more value. Update the golden,
constraints, generated parameters, and kernel branch as needed. The importing
perf wrapper inherits the change; verify whether an existing documented perf
subset should inherit it too.

### A distinct paired function

Give the family a stable name and a separately visible sweep. First add a
`special_mode=False` keyword to the shared runner and pass it through the
shared config/parameter path; then both wrappers select it identically:

```python
OP_SPECIAL_SWEEP = {**OP_SWEEP, "formats": SPECIAL_FORMATS}


@parametrize(**OP_SPECIAL_SWEEP)
def test_op_special(formats, input_dimensions, dest_acc):
    _run_op_test(
        formats,
        input_dimensions,
        dest_acc,
        special_mode=True,
    )
```

Add the normalized perf peer:

```python
from test_op import OP_SPECIAL_SWEEP, _run_op_test


@pytest.mark.perf
@parametrize(**OP_SPECIAL_SWEEP, **PERF_MEASUREMENT_AXES)
def test_perf_op_special(
    perf_report,
    formats,
    input_dimensions,
    dest_acc,
    run_types,
    loop_factor,
    is_perf,
):
    _run_op_test(
        formats,
        input_dimensions,
        dest_acc,
        special_mode=True,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
```

`op_special` now pairs by function name. A generic `test_perf_op` does not
prove that this family is measured.

### An intentional functional-only regression

Keep a forced state or block shape out of the published sweep:

```python
@parametrize(formats=..., ...)
def test_op_two_forced_blocks(formats, ...):
    _run_op_test(formats, ..., blocks=(2, 4))
```

Do not add a perf peer unless the forced shape is itself a measurement users
need. Name the unmatched function and rationale in the PR. The concrete
pattern is
`tests/python_tests/test_pack_l1_acc.py::test_pack_l1_acc_two_blocked_packs_of_4`;
its normal operation family remains paired through
`perf_pack_l1_acc.py::test_perf_pack_l1_acc`.

## Current examples to inspect

| Need | Functional | Perf | Why |
|---|---|---|---|
| Shared sweep and runner | `test_pack_l1_acc.py` | `perf_pack_l1_acc.py` | Functional owns `PACK_L1_ACC_SWEEP`; regression stays separate |
| Several named families in one module | `test_eltwise_binary.py` | `perf_eltwise_binary.py` | Shows normalized per-function peers and shared runners |
| Quasar generated narrowing | `quasar/test_pack_quasar.py` | `quasar/perf_pack_quasar.py` | One generator creates functional and documented perf combinations |
| Shared tile axis with perf packaging | `test_unpack_tilize.py` | `perf_unpack_tilize.py` | Shows a legitimate dimension-grid narrowing; inspect manually |

Do not copy legacy defects such as the cross-stem tiny-tile import or
`math_transpose_perf.cpp`. A dedicated phase source is allowed only for a
documented `perf_<op>_phases.py` benchmark and does not replace normal paired
coverage. The block-max8 cleanup inventory tracks the existing benchmark
until it is renamed and paired coverage is added.

## Focused validation

```text
/run-test test_op.py --arch blackhole
/run-test perf_op.py --arch blackhole --perf
```

```bash
cd tests/python_tests
python3 compare_test_and_perf.py --arch blackhole test_op.py perf_op.py
```

Repeat the comparer for Wormhole when the top-level files support it. For
Quasar, pass `--arch quasar` and the `quasar/..._quasar.py` paths. Use
`tests/run_hardware_free.sh` when parameters or perf columns change.
