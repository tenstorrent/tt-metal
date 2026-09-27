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

## 3. Change it

Make the change in the fork only. Keep it as small as the need; the smaller the diff, the easier the port back.
Forks are shared, so extending one must never change what it already does. Follow this recipe for every change to an
existing fork, whether it is a new feature, a new argument, a new output or a bug fix:

1. **Baseline first.** Before touching anything, run the fork's whole test suite and write down the counts: its own
   unit suite (`tests/unit/`, if it has one) and every model's cases (`tests/test_*.py`), with
   `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/<fork>/tests`. Known failures stay known; a new one after
   your change is yours.
2. **Behind an option, default off.** The new behaviour is an argument or enum value whose default keeps today's
   behaviour, return type and program. When it is off, the program must be the one it was: the same kernels
   (e.g. new kernel code only behind a define that is absent when off), the same compile-time and runtime args, and
   the same circular buffers. If the option changes what the program compiles to, add it to the program-cache key
   (the op's `compute_program_hash` / attributes).
3. **Old tests again, option off.** Rerun the same suite. The counts must match the baseline. Where the fork has a
   program-level check (a parity or descriptor test), it must show the option-off program is unchanged.
4. **New tests for the new setting.** Put them in the fork's own tests (`tests/unit/` for an op-level feature). Check
   the new output against a torch reference, cover the shapes, layouts and placements you claim, test the refusals,
   and show once, by hand, that a test fails when the new output is corrupted. A model that starts calling the new
   setting then gets its case through task O.1 (skill `bringup-fork-tests`).
5. **Record it.** Add a `CHANGELOG.md` entry (below) and update the `INDEX.md` row.

A bug fix that has to change the default behaviour is the one exception to step 2. Say so in the changelog, and
rerun every model's cases: they are exactly the models the fix changes.
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
  an existing fork, this is step 3 of the recipe in section 3: the counts must match your baseline.

## 7. For the reviewer

`git diff -w --no-index <source folder> ttnn/ttnn/bringup/<fork>` shows everything. That is the mechanical renames,
plus whitespace from the repo's formatters, plus the real changes. To see only the real changes, fork the unchanged
source again under a scratch name (`--name <fork>_chk`), diff the two, then delete the scratch fork and restore
`INDEX.md` and `bringup_nanobind.cpp`.
