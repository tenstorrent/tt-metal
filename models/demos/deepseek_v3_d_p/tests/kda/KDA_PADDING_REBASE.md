# Padding prototype rebased onto PR #56632

> Historical first rebase. See [the f20c rebase report](KDA_PADDING_REBASE_F20C.md)
> for current accuracy and performance measurements.

## Provenance

On 2026-09-18 the padding branch was rebased from PR head
`84ae832f1787fc1b8495e9a1b40f8090601a658d` onto its latest published head
`bbb915fa5c30023c7b9667d64a1b86d7c48ae91d`.
The original prototype and measurements remain reproducible at
`924c1b79ecb2a28836d07576665859c6ab12389b`, also preserved locally as
`backup/kda_pad_before_pr56632_rebase_20260918`.

The original three commits were replayed as `2d32e621104`, `a9f78f96756`, and
`b33f7988ee3`. The subsequent integration commit contains benchmark fixture
migration, formatting, and this evidence report. Tests exercised the working
tree during integration; the revision recorded in runner JSON alone does not
identify those uncommitted changes.

## Alignment with the parent PR

- Keep required caller-owned `actual_start` and required tail entry seeds.
  Preparation also requires explicit start metadata; there is no implicit-zero
  allocation or legacy dynamic-chronology mode.
- Extend the parent's `ChronologicalSelections`, shared chronology header, and
  reader bindings. Preserve its `first_rank` and `split_group` semantics.
- Keep `RecurrenceResult(output, final_state)` and construction-time layer and
  recurrence geometry. Cropped test oracles construct separate layers sharing
  weights, rather than changing the geometry of an existing layer.
- Use the canonical test utilities and scalar owner. Unsupported benchmark
  lengths receive explicit benchmark-owned grouping configurations; production
  tuning policy is unchanged.
- Keep optional `actual_end` as the padding extension. Device bounds can change
  during replay, are 32-aligned, and define a nonempty global interval. Empty
  ranks remain supported. Early exits preserve physical strides and collective
  participation. Returned state stops at the valid end; padded output is
  unspecified. Omitting the end means full constructed capacity.

## Validation

Checkout: `/localdev/mvasilijevic/tt-metal.worktrees/kda_pad_runtime`.
Native build and Python imports used this checkout's `python_env`:

```bash
source python_env/bin/activate
export TT_METAL_HOME="$PWD" PYTHONPATH="$PWD"
./build_metal.sh --enable-ccache --build-ttnn-tests
```

Final native build passed in 34.259 seconds. Import inspection resolved both
`ttnn` and `_ttnn.so` inside this checkout. Logs are under
`generated/kda_rebase/`, including `build-final.log` and `imports-final.log`.
Hardware tests used the serialized workspace `ai_workspace.tt.testing.test`
runner, which invokes `scripts/run_safe_pytest.sh` with exact collected node IDs.
Each named JSON below preserves the exact argv, timestamps, revision, individual
outcomes, and wrapper verdict; its matching `.log` preserves raw output.

| Evidence file | Scope | Result |
| --- | --- | --- |
| `native-2.json` | Padding preparation/direct scan and prefix widths 32/128 | 3 passed |
| `layer-2.json` | Padding trace replay on SP1/TP8, SP2/TP4, transposed SP4/TP2 | 3 passed |
| `offset-1.json` | Parent's changing-offset and production local trace cases | 10 passed |
| `prep-regression.json` | Preparation accuracy, cache reuse, rejection and scratch wrap | 27 passed |
| `contract-1.json` | Layer contracts and chronological selections | 24 passed |
| `perf-smoke.json` | All three benchmark harnesses, recurrence and layer | 14 passed |

All 81 selected cases passed. [Tracked validation manifest](KDA_PADDING_REBASE_VALIDATION.json)
preserves exact commands and outcomes. Benchmark smoke used
`KDA_FIXED_VARIANT=early KDA_FIXED_PADDING=0,1024 KDA_COST_VARIANT=control`;
the paired harness exercised valid lengths 4096/4896 and the attribution control
covered 4096/4896/5120.

The layer padding tests cover nonzero starts, changing bounds in one capture,
CPU references, cropped differential carries, padding invariance, and immutable
input carry. Native and layer tests passed before final formatting; the offset,
preparation, and contract regressions passed after final native build.

The first build exposed an rvalue use of `mesh_tensor()` in the optional-end
binding. Selecting the tensor by reference fixed it. Initial runner attempts
`native-1` and `layer-1` failed before device execution due to node-ID selection;
canonical collection supplied exact IDs for the successful runs. Model-only
collection must be used consistently: collecting alongside native tests changes
parameter-ID formatting through the native conftest. Logs retain these failures.
Collection emitted SWIG/Pydantic deprecation warnings.

## Performance applicability

The three historical reports and JSON datasets describe the original base and
prototype revision, not this rebase. Benchmark smoke checks only establish that
the harnesses execute with the new contracts. They do not establish a new
unpadded-PR versus padded-treatment performance comparison. That comparison has
not been rerun against `bbb915fa5c3`.
