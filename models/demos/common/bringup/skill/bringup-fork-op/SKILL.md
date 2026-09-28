---
name: bringup-fork-op
description: Fork an existing TTNN op into ttnn/ttnn/bringup (registered as ttnn.bringup.<op>) when a model bring-up needs the op changed, whether a feature or a bug fix. Covers checking INDEX.md for an existing fork, running fork_op.py, making the change behind an option, the changelog, the build and its usual errors, switching the model to the fork, and testing it. Use instead of ever editing an op under ttnn/cpp.
---

# Fork a TTNN op for a bring-up

A bring-up never edits an existing TTNN op. When a step needs an op to behave differently (a missing feature, a
precision option, a bug on this mesh shape), the op is copied to `ttnn/ttnn/bringup/<op>/` and changed there. The
copy registers as `ttnn.bringup.<python name>` beside the original. Later, a person decides whether the change is
ported back to the original, and then deletes the fork.

## 1. Reuse before you fork

Read `ttnn/ttnn/bringup/INDEX.md`. If the op is listed:
- call the fork (`ttnn.bringup.<op>`, and `ttnn.bringup.<Enum>` for its enums);
- read its `CHANGELOG.md`: the change you need may already be there;
- extend it if not.

Forks are shared by every model that uses them. A new behaviour must sit behind an argument or enum value whose
default keeps the fork's current behaviour, so the models already using it do not change. Never fork the same op
twice.

## 2. Fork

```
python ttnn/ttnn/bringup/fork_op.py <source op folder> --model <model> --task <task id>
# e.g. ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch
```

The source must be a C++ op folder with its own `CMakeLists.txt`. The script copies it and makes it a separate op:
- Namespaces: the op's C++ namespace moves under `ttnn::operations::bringup`. Every other namespace the copy declares
  (for example `ttnn::prim`, where device ops register their prim function) nests one level deeper, as `N::bringup`.
  Without this, the two copies clash at link time.
- CMake: the target becomes `ttnn_op_bringup_<folder>`, with alias `TTNN::Ops::Bringup::<Folder>`. Host `.cpp` files
  that the source had compiled from ttnn's central `sources.cmake` go into this target.
- Paths: kernel paths and fork-internal `#include`s point at the copy.
- Python: the op binds under `ttnn.bringup.`, registered in `bringup_nanobind.cpp`.
- Records: it writes `CHANGELOG.md` (source path and SHA) and adds an `INDEX.md` row.

It prints any line still pointing at the source's sibling ops. Those are dependencies the fork shares with the
original; leave them unless your change needs them.

## 2b. Carry the source op's tests (best-effort)

A fork must keep passing what its source op was tested on, not only the calls our models make. Otherwise a later model
with another shape or placement can hit a regression nobody saw. The source op's tests stay where they are: the fork
lists a selection in `tests/source.yaml` (fork_op.py writes the skeleton with the swap map), and
`testing/fork_source.py` runs them in place with the original op's name pointing at the fork.

An op's tests are scattered and mostly mixed with other ops' (rms_norm's sit in 8+ places, mostly in files that also
test layernorm), so pick a reasonable set, not an exhaustive one:
1. Find the call sites: grep the source's Python name (e.g. `ttnn.rms_norm(`,
   `ttnn.experimental.deepseek_prefill.dispatch(`) and its thin wrappers (e.g. `TtDispatchModule`) under
   `tests/ttnn/unit_tests`, `tests/ttnn/nightly`, and the op's own model folder (e.g.
   `models/demos/deepseek_v3_d_p/tests/op_unit_tests`). The CI lists in `tests/pipeline_reorg/*.yaml` show which of them
   the sanity and nightly tiers run.
2. Keep op-level tests that run on this box. Take whole files that test only this op. From mixed files, take this op's
   cases with a `k:` filter (a parametrize id or a test name). Skip:
   - model tests (they need weights);
   - tests for meshes this box does not have (T3K, Galaxy);
   - tests that are already skipped;
   - look-alike ops with their own kernels.
3. Aim for minutes, not hours. A check (fork only, warm kernel cache) should take a few minutes; recording runs
   everything twice, cold the first time. Prefer breadth (layouts, placements, dtypes, program configs) over many
   shapes of one kind: the sanity-tier file and the nightly file of the op usually cover most of it. Note in `why:`
   what each entry covers. For reference, `ttnn.bringup.rms_norm` carries 4 entries (263 tests: the sanity-tier
   interleaved and sharded files, the nightly file, and the rms cases of the nightly ULP file). Recording took about
   6 minutes in all; a check takes about 1 minute.
4. Complete the `swap:` map. Every Python name the tests reach, including enum types they pass, points at the fork.
5. Record the baseline: `python -m models.demos.common.bringup.testing.fork_source --fork <fork> --record`. It runs the
   selection on the original op and on the fork, and writes `tests/source_baseline.json`. Everything runs in the
   foreground, so when the selection is long, record one entry at a time (`--record --entry N`; the results merge into
   the baseline). The swap also hands the fork the source op's golden function, because upstream tests take their
   reference from `ttnn.get_golden_function(<source op>)`. The two should agree right
   after forking. Every case the original passes and the fork does not is a gap: fix it, or explain it in the
   changelog. Commit the manifest and the baseline.

The same applies to an AI-generated drop-in (a new op that replaces an existing one, such as `ttnn.bringup.rms_norm`
for `ttnn.rms_norm`): the replaced op's tests are its source tests.

## 3. Change it

Make the change in the fork only. Keep it as small as the need; the smaller the diff, the easier the port back.
Forks are shared, so extending one must never change what it already does. Follow this recipe for every change to an
existing fork, whether it is a new feature, a new argument, a new output or a bug fix:

1. **Know the baseline; don't rerun it.** The fork's last known state is on record: the source tests in
   `tests/source_baseline.json`, and the model cases, which passed at their last O.1. Rerun a baseline before
   changing anything only if that record is missing or older than the fork's last change.
2. **Behind an option, default off.** The new behaviour is an argument or enum value whose default keeps today's
   behaviour, return type and program. When it is off, the program must be the one it was: the same kernels
   (e.g. new kernel code only behind a define that is absent when off), the same compile-time and runtime args, and
   the same circular buffers. If the option changes what the program compiles to, add it to the program-cache key
   (the op's `compute_program_hash` / attributes).
3. **Iterate on the minimum.** A kernel edit changes the kernel's compile hash, so every program using it recompiles,
   and a full suite would cost many minutes per iteration. While developing, run only:
   - a handful of targeted cases of the new setting (`-k`);
   - the case of the model that needs the change (`tests/test_*.py -k <model>`).
   Write the new tests (step 4) early, so that this loop exercises them. When they pass, run the model's own
   gate that uses the op (its component test, or the short ladder rung). The change is correct in the op and inside
   the model only once that passes.
4. **New tests for the new setting.** Put them in the fork's own tests (`tests/unit/` for an op-level feature). Check
   the new output against a torch reference, cover the shapes, layouts and placements you claim, test the refusals,
   and show once, by hand, that a test fails when the new output is corrupted. A model that starts calling the new
   setting then gets its case through task O.1 (skill `bringup-fork-tests`).
5. **Then the full regression, once, option off.** Only after step 3 passes:
   - the fork's unit suite (`tests/unit/`, if it has one);
   - the source-test check, `python -m models.demos.common.bringup.testing.fork_source --fork <fork>`;
   - every model's cases, `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/<fork>/tests/test_*.py`.
   The results must match the baseline, and `fork_source` must report no regressions. Where the fork has a
   program-level check (a parity or descriptor test), it must show the option-off program is unchanged. If step 5
   finds a regression, fix it, go back to step 3 for the fix, then repeat step 5.
6. **Record it.** Add a `CHANGELOG.md` entry (below) and update the `INDEX.md` row.

A bug fix that has to change the default behaviour is the one exception to step 2. Say so in the changelog, and
rerun every model's cases in step 5: they are exactly the models the fix changes.
- Kernels compile at run time, so kernel-only edits need no build. Host code (program factory, device operation,
  bindings) does.

Record the change in the fork's `CHANGELOG.md` under `## Changes`, one entry per change:

```
### <short title>
- What: the change, and the option that turns it on (default = previous behaviour).
- Why: the symptom it fixes or the feature it adds, with numbers if you measured any.
- Needed by: <model> <task>
- Files: <paths inside the fork>
```

Update the `INDEX.md` row: the Changes column (a few words per change) and Used by.

## 4. Build

Run `./build_metal.sh` from the repo root in the foreground, and fix any errors. An incremental build rebuilds only
the fork and relinks ttnn. If a build dies halfway, run it again before reading anything into the error.

| Error | Cause | Fix |
|---|---|---|
| `'ttnn/operations/...' file not found` | the target lost the source's `ttnn/cpp` include dir | `target_include_directories(<target> PRIVATE ${FixmeOpAPIDir})` (fork_op.py adds it) |
| `use of undeclared identifier 'ccl'` (or another sibling namespace) | an unqualified lookup relied on the source's namespace | the fork must stay inside `ttnn::operations`; qualify the name fully |
| `duplicate symbol` at link | the copy declares a namespace fork_op.py did not nest | nest it as `N::bringup` and qualify the references to its symbols |
| `undefined symbol ... bringup::<op>` on `import ttnn` | a host `.cpp` is not in the fork's target | add it with `target_sources(<target> PRIVATE <file>)` |

## 5. Use it from the model

- Call `ttnn.bringup.<op>` where the model called the original. Enums and constants the op binds are there too, for
  example `ttnn.bringup.RoutedExpertActivation`. The originals' enums are different Python types; do not mix them.
- If the model reached the op through a Python wrapper that hard-codes the original (e.g. `TtDispatchModule`), keep the
  wrapper for its setup, and call the fork directly with the wrapper's attributes (see `_dispatch` / `_combine` in
  `models/demos/mimo_v2_6_d_p/tt/experts.py`). Do not edit the wrapper.

## 6. Test

- Run the model's own gate: its accuracy must not change beyond what the change intends.
- Give the fork a test case for every call the model makes. That is the `bringup-fork-tests` skill
  (`models/demos/common/bringup/skill/bringup-fork-tests/SKILL.md`), with random inputs.
- Run the fork's whole test folder, not only your cases: `scripts/run_safe_pytest.sh --run-all
  ttnn/ttnn/bringup/<fork>/tests`. Another model's case failing means your change broke that model. When you extended
  an existing fork, this is step 5 of the recipe in section 3: run it once, after the targeted tests and the model's
  gate pass, not after every edit.

## 7. For the reviewer

`git diff -w --no-index <source folder> ttnn/ttnn/bringup/<fork>` shows everything. That is the mechanical renames,
plus whitespace from the repo's formatters, plus the real changes. To see only the real changes, fork the unchanged
source again under a scratch name (`--name <fork>_chk`), diff the two, then delete the scratch fork and restore
`INDEX.md` and `bringup_nanobind.cpp`.
