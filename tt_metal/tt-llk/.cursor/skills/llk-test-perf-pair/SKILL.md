---
name: llk-test-perf-pair
description: >-
  Adds or aligns LLK functional (test_*.py) and perf (perf_*.py) pytest modules
  so @parametrize axes match for comparison. Covers PerfConfig/TestConfig wiring,
  C++ *_perf.cpp templates, axis naming, derive-and-assert, and verification with
  compare_test_and_perf.py. Use when adding a new test/perf pair, aligning
  parametrize sweeps, or unifying functional and perf LLK tests.
---

# LLK functional + perf test pair

Add or align a **functional** module (`test_<name>.py`) with its **perf**
counterpart (`perf_<name>.py`) under:

`tt_metal/tt-llk/tests/python_tests/`

Perf C++ sources live in `tt_metal/tt-llk/tests/sources/<name>_perf.cpp`.

## Workflow checklist

```
- [ ] 1. Study the functional test — axes, skips, geometry, golden path
- [ ] 2. Create or update perf_<name>.py (subset sweep, same axis *names*)
- [ ] 3. Create or update sources/<name>_perf.cpp (wire templates/runtimes)
- [ ] 4. Every shared axis must be truthful (wired, derived+assert, or dropped)
- [ ] 5. Run compare_test_and_perf.py on the pair
- [ ] 6. Confirm variant counts unchanged unless expansion was requested
```

---

## Naming and pairing

| Artifact | Pattern | Example |
|----------|---------|---------|
| Functional module | `test_<name>.py` | `test_reduce.py` |
| Perf module | `perf_<name>.py` | `perf_reduce.py` |
| Perf C++ source | `sources/<name>_perf.cpp` | `sources/reduce_perf.cpp` |
| Functional test fn | `test_<name>` or descriptive | `test_reduce` |
| Perf test fn | `test_perf_<name>` | `test_perf_reduce` |
| Comparison tool | `compare_test_and_perf.py` | **not** `test_*.py` |

Subfolder pairs work the same way (`test_matmul_quasar.py` ↔ `perf_matmul_quasar.py`).

---

## Axis alignment rules

**Goal:** axis *names* line up so `compare_test_and_perf.py` can diff sweeps.
Perf is usually an **intentional subset** of functional — do **not** expand perf
sweeps or variant counts unless the user explicitly asks.

### Standard axis names (use these, not legacy names)

| Prefer | Avoid |
|--------|-------|
| `math_op` | `mathop` |
| `formats` | ad-hoc format tuples |
| `cpp_source` | `testname` |
| `input_dimensions` | hardcoded dims without an axis |
| `tile_dimensions` | only when geometry is swept |
| `dest_acc` | `dest_acc_and_dims` combos when axes can be split |

### Three ways to handle a shared axis on perf

Pick one — **never leave a silent no-op**:

1. **Wire** — pass through `templates=[...]` / `runtimes=[...]` and into C++
   compile-time symbols (`MATH_OP(...)`, `BROADCAST_TYPE(...)`, `MATH_FIDELITY(...)`,
   `REDUCE_TO_ONE(...)`, `NUM_FACES(...)`, etc.).
2. **Derive + assert** — compute from other axes and `assert` consistency:
   ```python
   derived_tile_count = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
   assert derived_tile_count == tile_count
   ```
3. **Drop** — remove the axis from perf if it isn't used and functional sweeps it
   (comparison will show `[T] FUNCTIONAL-ONLY`; that's OK).

### Perf-only measurement axes (OK to differ)

These are measurement knobs, not functional coverage:

- `loop_factor`, `iterations`

`compare_test_and_perf.py` ignores them via `IGNORED_AXES`.

### Intentional geometry divergence

Perf may use different tile geometry for throughput (e.g. functional
`input_dimensions=[[256,32]]` → 8 tiles, perf `[[512,32]]` → 16 tiles). Keep
perf internally consistent (derive + assert `tile_count`). Add a one-line comment
in the perf module explaining why geometry differs.

---

## Functional test pattern (`test_<name>.py`)

```python
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import MATH_OP, MATH_FIDELITY, ...

@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, ...]),
    math_op=[MathOperation.Elwadd, ...],
    math_fidelity=lambda formats, math_op: _get_valid_math_fidelities(formats, math_op),
    dest_acc=lambda formats: _dest_acc_for_format(formats),
    # ... other axes
)
def test_<name>(formats, math_op, math_fidelity, dest_acc, ...):
    # skips for unsupported combos
    configuration = TestConfig(
        "sources/<name>.cpp",  # functional kernel, not *_perf.cpp
        formats,
        templates=[MATH_OP(mathop=math_op), MATH_FIDELITY(math_fidelity), ...],
        runtimes=[...],
        variant_stimuli=StimuliConfig(...),
        dest_acc=dest_acc,
    )
    # generate stimuli, run, compare to golden
```

Use lambdas and helper filters (`_fidelities_for_format`, constraint helpers) to
encode hardware/format rules at parametrize time.

---

## Perf test pattern (`perf_<name>.py`)

```python
import pytest
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.llk_params import PerfRunType
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import MATH_OP, TILE_COUNT, ...

@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, ...]),  # subset of functional
    math_op=[MathOperation.Elwadd, ...],
    math_fidelity=[MathFidelity.HiFi4],  # subset OK
    dest_acc=[DestAccumulation.No],
)
def test_perf_<name>(perf_report, formats, math_op, math_fidelity, dest_acc, ...):
    tile_count = 16  # or derived + asserted

    configuration = PerfConfig(
        "sources/<name>_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],  # or a subset when the kernel only supports some isolates
        templates=[
            MATH_OP(mathop=math_op),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[TILE_COUNT(tile_count), ...],
        variant_stimuli=StimuliConfig(
            None, formats.input_format, None, formats.input_format, formats.output_format,
            tile_count_A=tile_count, tile_count_B=tile_count, tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
    )
    configuration.run(perf_report)
```

Key differences from functional:

- `@pytest.mark.perf` + `perf_report` fixture
- `PerfConfig` instead of `TestConfig`
- No golden generation — measures cycles/timing
- Narrower format/op/fidelity sweeps
- `pytest.skip(...)` for combos the perf kernel doesn't support (document why)

---

## C++ perf kernel wiring (`sources/<name>_perf.cpp`)

Template parameters come from Python `templates=[...]` as compile-time `#define`s
(e.g. `MATH_FIDELITY`, `BROADCAST_TYPE`, `REDUCE_DIM`, `POOL_TYPE`).

Runtime parameters come from `runtimes=[...]` (e.g. `TILE_COUNT`, `REDUCE_TO_ONE`,
`UNPACK_TRANS_FACES`) and arrive via `RUNTIME_PARAMETERS params`.

Rules:

- **Do not hardcode** values that Python exposes as axes — read from template or
  `params.*`.
- Non–speed-of-light builds: read from `params` (e.g. `params.num_faces`,
  `params.TILE_CNT`).
- `SPEED_OF_LIGHT` builds: use compile-time constants from `params.h` where needed.
- Match functional kernel behavior for the swept subset.

---

## Verification

From `python_tests/`:

```bash
# Single pair
python compare_test_and_perf.py test_<name>.py perf_<name>.py

# Full sweep → perf_comparison_output.txt
python compare_test_and_perf.py > perf_comparison_output.txt
```

### Reading comparison output

| Tag | Meaning |
|-----|---------|
| `[=]` | Identical value sets |
| `[~]` | One sweep is a subset of the other (usually OK for perf) |
| `[x]` | True mismatch — investigate |
| `[T]` | Functional-only axis — OK if perf deliberately omits |
| `[P]` | Perf-only axis — OK for measurement knobs |

Fix `[x]` and silent no-ops. `[~]` and `[T]` are often intentional.

### Variant count sanity check

```bash
# From tt-llk/.cursor/scripts/run_test.sh (see run-test.mdc rule)
./.cursor/scripts/run_test.sh count perf_<name>.py --arch quasar
./.cursor/scripts/run_test.sh count test_<name>.py --arch quasar
```

Perf count should stay stable unless you intentionally changed coverage.

---

## Adding a **new** pair from scratch

1. **Functional first** — implement correctness test with full sweep, golden, skips.
2. **Identify the perf kernel** — often an existing `*_perf.cpp` or a trimmed isolate
   of the functional kernel. Copy/adapt from the closest existing pair (see
   [llk-test-perf-pair-reference.md](llk-test-perf-pair-reference.md)).
3. **Mirror axis names** on perf — start with the axes the perf kernel actually uses.
4. **Subset values** — pick representative formats/ops/fidelities; copy skip logic
   from functional where the perf kernel has the same limitations.
5. **Wire or drop every axis** — grep perf function body for unused params.
6. **Compare** — run `compare_test_and_perf.py`; iterate until no accidental `[x]`.

---

## Common mistakes

- Leaving a parametrize axis on perf that is never wired → false alignment
- Expanding perf sweeps “to match functional” without being asked → blows up compile time
- Using `mathop=` in `MATH_OP(...)` call — parameter name is `mathop=` (API), axis is `math_op`
- Naming helper scripts `test_*.py` → pytest collects them (`pytest.ini` collects `test_*.py`, `perf_*.py`)
- Hardcoding HiFi4 / broadcast / num_faces in C++ while Python sweeps those axes
- Forgetting `@pytest.mark.perf` on perf tests

---

## Reference pairs (copy from these)

See [llk-test-perf-pair-reference.md](llk-test-perf-pair-reference.md) for a curated list of well-aligned pairs and
which patterns each demonstrates.
