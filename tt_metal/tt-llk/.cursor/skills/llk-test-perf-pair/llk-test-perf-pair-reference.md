# Reference pairs and patterns

## Well-aligned examples

| Pair | Demonstrates |
|------|--------------|
| `test_eltwise_binary.py` / `perf_eltwise_binary.py` | `broadcast_type` → C++ template; derive+assert `tile_count`; intentional geometry divergence |
| `test_reduce.py` / `perf_reduce.py` | `math_fidelity` + `pool_type` wired; `is_reduce_to_one` pinned + documented (kernel has no reduce-to-one branch to bind); perf drops `tile_dimensions` sweep |
| `test_unpack_tilize.py` / `perf_unpack_tilize.py` | single geometry axis `input_dimensions` (redundant `dimensions` axis removed); `num_faces` → `NUM_FACES`; derived + asserted |
| `test_unpack_A.py` / `perf_unpack_A.py` | `cpp_source` axis; transpose runtimes; format skip matrix |
| `test_pack_dest_bank.py` / `perf_pack_dest_bank.py` | `input_dimensions` derived from block geometry |
| `perf_sfpu_unary.py` (with functional SFPU unary) | `math_op` threading; `approx_mode`/`fast_mode`/`dest_acc`; `iterations`/`loop_factor` ignored by comparison |

## Key helper modules

| Module | Role |
|--------|------|
| `helpers/param_config.py` | Custom `@parametrize` decorator, `input_output_formats`, lambdas |
| `helpers/test_config.py` | Functional `TestConfig` — compile + simulate + golden |
| `helpers/perf.py` | `PerfConfig` — perf measurement pipeline |
| `helpers/test_variant_parameters.py` | `MATH_OP`, `TILE_COUNT`, `BROADCAST_TYPE`, template/runtime builders |
| `helpers/constraints.py` | `get_valid_math_fidelities`, `get_valid_dest_accumulation_modes` (perf) |
| `helpers/stimuli_config.py` | `StimuliConfig` tile counts and formats |
| `compare_test_and_perf.py` | Axis diff tool (not collected by pytest) |

## Template vs runtime parameters

**Templates** (compile-time `constexpr`, in `PerfConfig.templates`):

- `MATH_OP(mathop=...)`, `MATH_FIDELITY(...)`, `BROADCAST_TYPE(...)`
- `REDUCE_POOL_TYPE(...)`, `APPROX_MODE(...)`, `FAST_MODE(...)`
- `DEST_SYNC(...)`, `STABLE_SORT(...)`, `TILE_DST_CT_OFFSET(...)`, `ACC_TO_DEST(...)`

**Runtimes** (per-variant struct fields, in `PerfConfig.runtimes`):

- `TILE_COUNT(n)` (varies per variant / derived)
- `UNPACK_TRANS_FACES(...)`, `UNPACK_TRANS_WITHIN_FACE(...)` (multi-option sweeps)
- `LOOP_FACTOR(...)`, `ITERATIONS(...)` (perf measurement only)

**Single-option axes belong in `templates`, not `runtimes`.** A `RuntimeParameter`
(e.g. `NUM_FACES`, `TEST_FACE_DIMS`) can be placed in `templates=[...]` — it is
then emitted as a bare `constexpr` via `convert_to_cpp()` in both normal and
`SPEED_OF_LIGHT` builds, so the kernel reads it directly with no `params.*` access
and no `#ifndef SPEED_OF_LIGHT` guard. Runtimes become an empty struct under SOL,
so keeping a multi-option runtime requires a `#ifndef SPEED_OF_LIGHT` alias.

Use `runtime(...)` wrapper from `param_config` when a runtime axis should not
split compile keys.

## Minimal perf module skeleton

```python
@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    math_op=[MathOperation.Elwadd],
    dest_acc=[DestAccumulation.No],
)
def test_perf_foo(perf_report, formats, math_op, dest_acc):
    tile_count = 16
    PerfConfig(
        "sources/foo_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE],
        templates=[MATH_OP(mathop=math_op)],
        runtimes=[TILE_COUNT(tile_count)],
        variant_stimuli=StimuliConfig(
            None, formats.input_format, None, formats.input_format, formats.output_format,
            tile_count_A=tile_count, tile_count_B=tile_count, tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
    ).run(perf_report)
```

## Comparison commands

```bash
cd tt_metal/tt-llk/tests/python_tests

python compare_test_and_perf.py test_foo.py perf_foo.py
python compare_test_and_perf.py --full test_foo.py perf_foo.py
python compare_test_and_perf.py > perf_comparison_output.txt
```
