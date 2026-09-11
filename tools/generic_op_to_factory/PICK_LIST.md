<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Migration workflow pick list

Candidate material from this tt-metal checkout's `.cursor/`, `.github/`, and
`scripts/`, inspected at `dfc0dae18e45`. This is a selection aid for the future
eval-run → C++ program-factory workflow, not an adopted workflow or a record of
completed checks. [MAPPING.md](MAPPING.md) remains the translation reference.

Use the IDs to select workflow components. All selection boxes start empty;
**Recommended** is a recommendation, not an approval. Existing repository
checks are listed separately because omitting them from a workflow document
does not disable their enforcement.

The `.cursor` recipe starts with an existing C++ operation, not an eval run.
The `.github/instructions` file is review guidance; [AGENTS.md][agents]
explicitly says review files are not the specification for authoring changes.
We extract useful questions and adapt them to the verified implementation.

## 1. `.cursor/`: implementation and verification procedure

Sources: [descriptor migration recipe][recipe] and [hash audit guide][hash].

| Pick | ID | What we can take | Concrete contribution to our workflow | Recommendation / qualification |
|---|---|---|---|---|
| [ ] | C01 | Operation scaffold — recipe §§1.1–1.2, 1.6 | A file plan for the public API, device operation, attributes, tensor arguments, descriptor factory, and Python binding integration | **Recommended.** Adapt to a named native operation; `_new` naming is not necessary when importing an eval candidate |
| [ ] | C02 | Reuse existing kernels — recipe §1.3 | A source-to-target kernel/helper inventory with the chosen paths and any verified include changes | **Recommended.** Preserve file/inline source type and verify helper equivalence under MAPPING U6 |
| [ ] | C03 | Deliberately choose the cache-hit mechanism — recipe §§1.2–1.4 | For each varying arg/CB address: its writer on a cache hit, plus the expected adapter branch | **Recommended.** Use MAPPING §2's actual aliasing and CB-only fallback conditions, not the recipe's simplified fast-path description |
| [ ] | C04 | Trace structural values to their inputs — hash guide Steps 1–5 | A hash audit: kernels, defines, compile args, core/work sets, CB configuration, semaphores, source attributes/specs, and intentional exclusions | **Recommended.** Prefer the default hash when complete; do not mechanically add a custom hash or duplicate already-hashed derived values |
| [ ] | C05 | Compare old and new behavior — recipe §2.1 | Original Python candidate and native operation run against the same golden cases, with results and refusal classes compared | **Recommended.** Use the actual golden suite's tolerances; the recipe's example C++ benchmark is not an eval-harness adapter |
| [ ] | C06 | Cache transition tests — recipe §2.1; hash guide Test Patterns 1–4 | Same specs/new allocations hit; structural changes miss; excluded scalars hit and affect results; supported optional/alias cases refresh correctly | **Recommended.** Keep allocations alive where needed to establish that addresses actually differ; test CB-only cases when applicable |
| [ ] | C07 | Measure dispatch overhead — recipe §§2.1–2.3 | A reproducible comparison separating cache-miss setup, cache-hit host overhead, and device execution | **Conditional:** select when performance is an objective. Choose representative cases and an acceptance budget; do not inherit the example's 3%/5% thresholds or fixed old/new ordering as a validated method |
| [ ] | C08 | Descriptor patching parity mode — recipe §2.1 | A diagnostic build with `ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON` comparing patched runtime state against reconstruction | **Conditional:** useful when cache patching is complex. Verify the loaded build; use deterministic inputs/seeds. It adds reconstruction overhead and is not a performance measurement or a replacement for golden correctness |
| [ ] | C09 | Check consumers before replacement — recipe §3.7 | A reference search for old operation/factory users, with an explicit integration or compatibility decision | **Conditional:** needed when replacing an existing native API. Importing a new eval candidate does not by itself authorize deleting an existing operation |

Suggested core: **C01–C06**. C07–C09 depend on the migration's objective and scope.

## 2. `.github/instructions/`: review questions to turn into evidence

Source: [TTNN operation review guidance][review]. These are candidate review
items, not a blanket adoption of that file's wording.

| Pick | ID | What we can take | Concrete contribution to our workflow | Recommendation / qualification |
|---|---|---|---|---|
| [ ] | R01 | Input validation and functional equivalence | An input/output contract: shapes, dtype, layout, device/storage constraints, optional tensors, output reuse, and rejection classes | **Recommended.** Derive checks from the source operation; preserve Python refusal types rather than replacing every exception with `TT_FATAL` |
| [ ] | R02 | Tracing compatibility and placement of host work | An inventory of allocation, host-generated tensor data, and device-buffer I/O, with their location outside descriptor construction | **Recommended.** Ordinary planner arithmetic and descriptor/vector construction are valid host work. Actual tracing compatibility requires an appropriate device test |
| [ ] | R03 | Scope discipline and descriptor pattern | A review that the port uses the selected descriptor design and introduces only necessary integration changes | **Recommended.** Do not bundle unrelated operation migrations, cleanup, or kernel modernization into the host-factory port |
| [ ] | R04 | Public API documentation | A native signature and binding docstring documenting TTNN behavior and supported options | **Recommended.** Review guidance asks for API docs without external-framework references; Python binding mechanics still need a concrete target example |
| [ ] | R05 | Kernel ordering, startup placement, and API regression checks | A focused comparison of changed kernel code: initialization, tile-register lifecycle, CB/NOC/semaphore ordering, and include placement | **Conditional:** when kernel code changes. This is a review scope, not a requirement to rewrite preserved kernels into another API |
| [ ] | R06 | Reshape and layout invariants | Targeted tests for logical-volume preservation, inferred dimensions, padding, and the operation's real alignment constraints | **Conditional:** when the wrapper/planner reshapes or changes layout. Do not turn the review file's fixed 32×32 tile assumptions into universal validation |
| [ ] | R07 | Composite-operation overhead | Count intermediate operations/allocations and measure their effect on representative cases | **Conditional:** when the source wrapper is composite or the port introduces composition; combine with C07 |
| [ ] | R08 | CCL buffer/semaphore consistency | Check EDM/kernel configuration agreement, vector lengths, and size recalculation when relevant CCL parameters change | **Defer:** the current MAPPING scope excludes coordinate-dependent/workload-scoped mesh migration; revisit with a dedicated mapping |

Suggested core: **R01–R04**, with R05–R07 selected by the source operation.

## 3. `scripts/` and CI: checks already configured in the repository

These are existing checks to include in the workflow's verification stage, not
optional coding policies. Their scope and limitations matter. Passing them
does not prove functional equivalence or establish the absence of cache-hit
reconstruction.

| ID | Existing check | What we get | Scope / limitation |
|---|---|---|---|
| G01 | [detect_smuggled_rta.py][rta] | Flags recognized raw buffer-address expressions/locals flowing into descriptor runtime-arg sinks | [Pre-commit][precommit] scopes it to `^ttnn/.*/device/.*\.(cpp|hpp)$`. Text heuristic, not full dataflow analysis; documented `smuggled-rta-ok` suppressions exist |
| G02 | [detect_override_rebuild.py][rebuild] | Flags recognized descriptor reconstruction/bulk-application calls inside `override_runtime_arguments` | Same `device/` scope. Text heuristic with `override-rebuild-ok` suppressions and an existing-violation baseline; it does not detect every indirect rebuild or the adapter's fallback path |
| G03 | [detect_legacy_device_op.py][legacy] | Flags non-static `validate`, `compute_output_specs`, `create_program`, or `create_program_at` patterns associated with `OldDeviceOperation` | Hook is `detect-legacy-device-op`, scoped to `^ttnn/.*\.hpp$`, with `--check-new-only`. New-file detection depends on comparison refs. It does not prove that a new class implements the intended modern concept |
| G04 | [clang-tidy CI][tidy] | Compiler-backed analysis using `.clang-tidy` and `--warnings-as-errors=*` | Eligible changes route through [pr-gate][gate] and [code-analysis][analysis]. Reporting and inline suggestions are conditional; formatting hooks alone do not run this analysis |
| G05 | [Formatting and basic hooks][precommit] | clang-format for C++, gersemi for CMake, applicable Python/YAML formatting, whitespace and conflict-marker checks | Run the hooks applicable to touched files; each hook's actual filters determine coverage |
| G06 | [Build wrapper][build] and [authoring instructions][agents] | Compile the registered native operation and report the exact command/result | Follow applicable build instructions, choose a narrow build, and report environment blockers. The wrapper invokes `build_metal.sh --enable-ccache`; compilation does not replace on-device tests |

For a migrated implementation, the first three checks can be invoked through
their configured hooks with the relevant changed paths:

```text
pre-commit run detect-smuggled-rta --files <changed-device-cpp-and-hpp-files>
pre-commit run detect-override-rebuild --files <changed-device-cpp-and-hpp-files>
pre-commit run detect-legacy-device-op --files <new-ttnn-hpp-files>
```

These are command templates, not commands executed for this pick list. Use
explicit comparison refs when needed to make the legacy check's new-file scope
reproducible. Follow the user's safe-pytest runner instruction for Python tests.

## 4. Material to adapt or leave out

| Source advice | Selection decision |
|---|---|
| Recipe's three-stage `_new` → replace → delete workflow | Keep comparison and staged verification; decide naming, replacement, and cleanup separately for an eval import |
| Recipe's automatic buffer-patching descriptions | Use MAPPING §2 and adapter evidence. Aliasing and CB-only descriptors can still rebuild |
| Review file's no-op override suggestion | Do not import blindly. On the ordinary descriptor path an override disables automatic binding patching, so a no-op can leave stale addresses |
| Review file's recommendation to use `DynamicRuntimeArg` | Do not introduce it as the migration target. The recipe recommends the factory override, and the adapter rejects combining the two hooks |
| Hash guide's unconditional custom-hash/factory-index advice | Audit whether the existing key already determines the factory; choose a custom hash only for a concrete reason |
| Review file's blanket alignment numbers and automatic branch deduplication | Derive alignment from the real source contract. Preserve planner arithmetic/branches under MAPPING's no-refactor policy |
| Recipe's `--build-all`, raw pytest in the hash guide, and device-reset examples | Do not copy as workflow defaults. Follow current authoring and user instructions for build scope, Python environment, safe pytest, and device operations |
| Recipe's deletion of comparison tests | Decide what temporary scaffolding to remove; retain useful regression coverage and baseline evidence |
| Recipe's fixed benchmark iteration counts, thresholds, and anecdotal timing claims | Take the measurement objective, not unverified measurements or a universal performance budget |
| Recipe's workload-descriptor instructions | Outside this workflow's current scope. In particular, its claim that the per-program override is never called is inconsistent with the checked adapter's workload branch; investigate that path separately |

## 5. What these sources do not supply

They do not specify how to select an eval-run candidate, export its exact
artifacts and tuning configuration, pin its source/harness revisions, reproduce
the original baseline, or route the golden suite to the new native entry point.
Those steps need the eval repository and a pilot run. Selecting C01–C06 and
R01–R04 provides the implementation/review portion; G01–G06 supplies existing
verification checks. It still does not constitute an end-to-end eval workflow.

[recipe]: ../../.cursor/commands/ttnn/descriptor-migration-recipe.md
[hash]: ../../.cursor/commands/ttnn/verify-device-operation-hash.md
[review]: ../../.github/instructions/ttnn-ops.instructions.md
[rta]: ../../scripts/detect_smuggled_rta.py
[rebuild]: ../../scripts/detect_override_rebuild.py
[legacy]: ../../scripts/detect_legacy_device_op.py
[precommit]: ../../.pre-commit-config.yaml
[tidy]: ../../.github/workflows/clang-tidy-reusable.yaml
[gate]: ../../.github/workflows/pr-gate.yaml
[analysis]: ../../.github/workflows/code-analysis.yaml
[build]: ../../.github/scripts/copilot-build.sh
[agents]: ../../AGENTS.md
