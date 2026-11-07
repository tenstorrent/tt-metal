# Blackhole versus Quasar alignment

Read with `perf-coverage`. Align Quasar test versus perf first (the
same-architecture workflow), then Blackhole versus Quasar.

## Checklist

```
Alignment:
- [ ] Inventory families and axes on BH functional, BH perf, QSR functional, QSR perf
- [ ] Split QSR perf 1:1 with QSR functional families
- [ ] Unbundle QSR composites onto BH axis names (constrained callables, not full Cartesian)
- [ ] Use shared MathOperation aliases where the kernel is the same
- [ ] Pin perf approx_mode to No except kernels that read it (atan2)
- [ ] Leave real ISA / datapath differences visible
- [ ] QSR test vs perf
- [ ] BH vs QSR perf, and BH vs QSR functional
- [ ] Host-only suite (header gate); optional LLK perf workflow run
```

## 1. Inventory

Read all four modules. For each family record: function name,
`@parametrize` axis names, value sets, functional-only `runtime(...)` axes,
perf-only `_PERF_AXES`, and the C++ `BinaryOp` or kernel path.

| Class | Examples | Action |
|---|---|---|
| Name pairing | One QSR perf versus six functional names | Split wrappers |
| Axis packaging | `formats_dest_acc` versus `formats` + `dest_acc`; tuple `binary_op_mathop_approx` versus `mathop` + `approx_mode` | Unbundle to BH names |
| Enum alias | QSR `SfpuElwmulInt` versus BH `SfpuMulInt32` | Keep the QSR member; the test maps it onto the WH/BH kernel |
| Real ISA | the Quasar int sweep has no shifts; BH int SUB; BH `bcast_dim` | Keep the difference |
| Family split | BH `div`/`atan2` are separate tests; QSR lumps them into `float` | Split QSR only if the kernel exists; do not invent BH families |

## 2. Unbundle onto Blackhole names

Blackhole typically sweeps flat `formats`, `dest_acc`, `mathop`,
`approx_mode`. Quasar often packs `QuasarSfpuVariant` or
`(binary_op, mathop, approx)` tuples. Explode the representatives into named
axes, keeping the same combinations with callable constraints
(`helpers.param_config._params_solve_dependencies` resolves callables from
argument names):

```python
def _dest_acc_for_float_formats(formats):
    return [v.dest_acc for v in _VARIANTS
            if v.formats.input_format == formats.input_format
            and v.formats.output_format == formats.output_format]

def _approx_modes_for_mathop(mathop):
    # Functional. Perf copies the sweep and pins DIV back to No.
    if mathop in (MathOperation.SfpuAtan2, MathOperation.SfpuElwdiv):
        return [ApproximationMode.No, ApproximationMode.Yes]
    return [ApproximationMode.No]

FLOAT_SWEEP = dict(
    formats=[v.formats for v in _VARIANTS],
    dest_acc=_dest_acc_for_float_formats,
    mathop=_MATHOPS,
    approx_mode=_approx_modes_for_mathop,
)
```

Resolve `QuasarSfpuVariant` and `binary_op = mathop.cpp_enum_value` inside
the test, not as a sweep axis.

## 3. Shared aliases, not Quasar-only enum members

`SfpuGtInt` / `SfpuLtInt` / `SfpuLeInt` / `SfpuGeInt` remain in
`helpers/llk_params.py` as aliases. They are not sweep values; sweeps use
`SfpuElwLt/Gt/Le/Ge` on every architecture. `SfpuElwmulInt` stays
Quasar-only (`BinaryOp::MUL`); the test maps it to `SfpuMulInt32`
(`MUL_INT32`) and must not treat them as one sweep value. `SfpuCopyDest`
(`COPY_DEST`) is declared only on Quasar. Leave `MathOpType.SFPU_BINARY` as
it is, and keep a guard that `COPY_DEST` is absent from the WH/BH `BinaryOp`
enums.

## 4. Blackhole versus Quasar perf

Equalize only packaging. Typical remaining differences after a good unbundle:

- **identical**: shared axes whose value sets match (`dest_acc`, Int32
  `formats`).
- **subset**: Quasar formats without Bfp8 or MX-only extras; float perf
  `approx_mode` when atan2 stays in the Quasar float family.
- **different `mathop`**: the intersection is the portable ops; the extras
  are ISA.
- **one architecture only**: `bcast_dim`, `dest_sync`, tagged `[B]` or `[Q]`.

Do not add a dummy `bcast_dim=[None_]`. Do not shrink both mathop lists to
the intersection. Do not add Bfp8_b to DIV perf: unpack expands it to BF16
before the SFPU, so the extra rows do not change the math kernel.

Optional packaging: split a lumped Quasar family so it pairs with an existing
Blackhole family name (`div`, `atan2`) when both have the kernel.

## 5. CI check

Workflow **LLK perf** (`llk-perf.yaml`) takes `architecture=all|blackhole|quasar`
and `speed-of-light`. Use `speed-of-light=false` when comparing against a
normal-build baseline, and never mix the two kinds
([perf-facts](../../perf-test/references/perf-facts.md), "Build kinds").

```bash
gh workflow run "LLK perf" --ref <branch> -f architecture=all -f speed-of-light=false
```

## Do not

- flatten `QuasarSfpuVariant` in `compare_test_and_perf.py` instead of
  unbundling the tests;
- add Quasar mathops missing from the `tt_llk_quasar` `ckernel_defs.h`
  `BinaryOp`;
- add Quasar `bcast_dim` without a kernel path;
- rename Quasar-only families (`quant`, `bf16_rne`, `max_min_*`) to
  Blackhole names;
- put `approx_mode` on the Quasar global `_PERF_AXES` when a family already
  constrains it;
- apply a Yes/No `approx_mode` product to kernels that ignore it.

Stand-in and accounting facts that affect cross-architecture numbers (`#1230`,
`tile_cnt` counting both operand tiles) are in
`perf-test/references/wormhole-blackhole.md`, `quasar.md`, and
`perf-facts.md`.
