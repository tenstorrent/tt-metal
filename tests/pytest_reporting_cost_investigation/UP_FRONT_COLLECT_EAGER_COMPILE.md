# `up_front_collect`'s parallel precompile is dead code for Metal 2.0 ops

**Status:** diagnosed, not fixed. Needs runtime-team input — the fix is tt_metal-side.
**Found:** 2026-09-07, during a production-TTNN reference run of `rms_norm` (eval DB run `1006`).
**Scope:** every op whose workload is built through a Metal 2.0 `ProgramSpec` factory, not just `rms_norm`.

---

## Summary

`tests/plugins/up_front_collect.py` exists to make a broad test run compile its kernels
**once, up front, in parallel across N workers**, instead of inline and serially behind the
test bodies. For Metal 2.0 `ProgramSpec` ops it silently does neither: every kernel is
compiled inline, one at a time, during the collect pass, and the parallel step at the end
finds nothing left to do.

The machinery is not broken. Its *input* is already cooked by the time it runs.

## Measured impact

Production `ttnn.rms_norm`, full golden cartesian, 40,828 cases, Blackhole, JIT server at
`bgdepyc02:54210`:

| phase | wall time |
|---|---|
| precompile warm pass | **6,924 s** (115 min) |
| graded run (cache warm) | **~8 min** |

The warm pass reported:

```
UP_FRONT_COLLECT: compiled 3420 programs in 1.7s (workers=32, errors=0)
```

`1.7 s` for 3,420 programs is not speed — it is 3,420 cache hits. All the compilation had
already happened, serially, during the preceding 115 minutes.

**Concurrency proof.** Distribution of kernel ELFs written per active second in the JIT
server's cache for this build key (9,075 ELFs across 4,749 distinct seconds):

| ELFs in one second | 1 | 2 | 3 | 4 | 5 | 6–10 |
|---|---|---|---|---|---|---|
| seconds | 2,534 | 1,324 | 340 | 185 | 190 | 176 |

A compute kernel emits 3 ELFs, a dataflow kernel 1. So the steady state is **~one kernel in
flight at a time** — never the 32 the pass was configured for.

## Root cause

Three facts that are individually fine and jointly broken.

**1. Metal 2.0 spec factories compile at construction.**

`tt_metal/impl/metal2_host_api/program_spec.cpp:3376`

```cpp
distributed::MeshWorkload MakeMeshWorkloadFromSpecs(...) {
    for (const auto& [device_range, program_spec] : program_specs) {
        workload.impl().add_program(device_range, BuildProgramFromSpec(...));
    }
    workload.impl().compile(&mesh_device);   // ← eager, at construction
    return workload;
}
```

Same in `MakeMeshWorkloadFromSpec`. `tt_metal/impl/program/program.cpp:2737` states it plainly:

> *"Eager callers (MakeProgramFromSpec/MakeMeshWorkloadFromSpecs) reach compile() directly"*

Introduced by `7b36e769200` — *[Feature] Metal 2.0 MakeMeshWorkloadFromSpecs (#49437)*.

**2. The collector still assumes the opposite.**

`ttnn/api/ttnn/device_operation.hpp:341`

```cpp
// Up-front parallel precompile: in collect mode, move the freshly
// built — but not-yet-compiled — workload into the collector keyed by its
// program hash, and skip caching + dispatch. The kernels are JIT-compiled
// later, in parallel, by ttnn::up_front_compile::parallel_compile
if (auto* collector = ttnn::up_front_compile::ProgramCollector::active()) {
    collector->collect(program_key.hash, std::move(cached_workload.workload));
    return;
}
```

**"not-yet-compiled" is no longer true** for spec factories. `#49437` silently falsified this
invariant; the comment was never updated.

**3. So the parallel step is a no-op.**

`ttnn/core/up_front_compile.cpp::parallel_compile` fans `detail::CompileProgram` across a
thread pool. `ProgramImpl::compile` (`program.cpp:2693`) early-outs on
`compiled_.contains(build_env.build_key())`. Every collected program is already in that set,
so all 32 threads return immediately.

**Confirmed reachable from a real op:** `ttnn.rms_norm` → `rmsnorm.cpp:63` →
`ttnn::prim::layer_norm`, whose factory is `ProgramSpec`-based
(`layernorm_op_multi_core.cpp:167`, `layernorm_op_multi_core_sharded.cpp:32`).

**Not affected: the ProgramDescriptor path.** The descriptor adapter only ever calls
`add_program` (`mesh_device_operation_adapter.hpp:279, 573, 595, 601`) — no `compile()` anywhere
on that path — so the collector still receives uncompiled workloads and precompile works
normally. This defect is **spec-path-only**.

## Why it stayed hidden

The pass reports success identically whether it compiled 3,420 programs or zero:

```
compiled 3420 programs in 1.7s (workers=32, errors=0)
```

There is no counter distinguishing *built* from *already built*. A ~30× regression looks like
a fast, healthy run.

## Proposed fix

### A. A "build, don't compile" hatch (the actual fix)

All that is wanted is for the spec factories to return an **uncompiled** workload while the
collect pass is running, so `parallel_compile` has real work to do.

Add a `defer_compile` parameter to the three entry points in
`tt_metal/api/tt-metalium/experimental/metal2_host_api/program.hpp:37,48,62`; when set, skip the
final compile in `program_spec.cpp:3372 / :3391 / :3401`. The callsite makes this structurally
trivial: `compile()` is the **last statement** in each, and nothing inside consumes the compiled
state.

Then pass `ProgramCollector::active() != nullptr` at the two call sites:

- `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:924` — the **only** adapter call site, and the
  one that covers every op written as a `create_program_artifacts` spec factory (layernorm /
  rms_norm included).
- `rotary_embedding_indexed_device_operation.cpp:616` — the one op today that hand-rolls its own
  `MeshWorkloadFactory` and calls `MakeProgramFromSpec` itself.

**Why a parameter and not a thread-local flag.** An earlier draft of this document proposed a
thread-local `ScopedDeferProgramCompile` on the grounds that ops call `Make*` directly and a
parameter would have to be threaded through every factory. **That was wrong.** Spec factories
return a `ProgramSpec`; the *adapter* calls `Make*`, at exactly one line. A parameter is a
two-line change with no global state.

The only genuine argument for a thread-local is reach over hand-rolled `MeshWorkloadFactory` ops
like `rotary_embedding_indexed`, which a parameter cannot cover without editing the op. With
exactly one such op in the tree, that does not yet justify hidden global state — revisit if the
hand-rolled pattern spreads.

**Why deferring is safe.**

- *The workload is discarded.* Collect runs under `NO_DISPATCH`; only the on-disk kernel cache
  survives. So the allocate/finalize half of `compile()` never needs to happen at all.
- *The halves already line up.* `MeshWorkloadImpl::compile` (`mesh_workload.cpp:106`) →
  `compile_and_allocate` (`program.cpp:2911`) is a superset: JIT **plus** CB allocation, DFB
  allocation, scratchpads, RTA pre-sizing, `finalize_offsets`. `detail::CompileProgram`
  (`tt_metal.cpp:1286`), which `parallel_compile` calls, is `ProgramImpl::compile` — the JIT half
  alone. Exactly what the warm pass wants.
- *The `SetProgramRunArgs` that follows at `adapter:928` does not depend on it.* Metal already
  documents this ordering as order-agnostic — `program_run_args.cpp:611` and `program.cpp:1826`
  both state that on the legacy order (run-args before allocation) scratchpad addresses are 0 and
  get patched later.
- *Bookkeeping works out for free.* Skipping compile leaves `compiled_` empty, so the later
  `detail::CompileProgram` does real work instead of hitting the `program.cpp:2716` early-out.

No existing seam to reuse — `skip_compile` / `defer_compile` / `lazy_compile` do not exist in the
tree today.

### B. Make the failure loud (land independently, first)

Have `up_front_compile` report how many collected programs were **already compiled**:

```
3420 collected · 0 built · 3420 already compiled     ← today's silent bug
3420 collected · 3420 built · 0 already compiled     ← healthy
```

Zero design risk, and it catches the entire class — including the next path that starts
compiling eagerly. This alone converts a two-hour investigation into a ten-second one.

### C. Considered and rejected for now

Collecting `ProgramSpec`s rather than `MeshWorkload`s would defer *construction* as well as
compilation, and is arguably more correct. But specs may not be self-contained, and it is a
much larger change. Not the first move.

## Open questions

- Does anything between `BuildProgramFromSpec` and the collector's interception assume
  compiled state? The funnel `return`s immediately after `collect()`, so probably not —
  **not traced**.
- Does `MakeProgramFromSpec` (non-mesh sibling) need the same treatment? `program.cpp:2737`
  names it as an eager caller too.
- Are there other eager-compile paths? Don't answer this by reading code — land guard (B)
  and run it across a few op families.

## Reproducing

```bash
# any Metal 2.0 spec-factory op, broad selection
pytest -p tests.plugins.up_front_collect <golden_dir>
```

Watch the `UP_FRONT_COLLECT: compiled N programs in Ts` line. `T` near zero with large `N`
means every program was pre-compiled inline and the parallel pass did nothing.

## Diagnostic trap (cost two wrong diagnoses)

`stat` on the JIT server's `<cache>/<build_key>/` directory shows only **immediate-child**
mtime, not deep writes. It reads as "the server is idle for my build key" while the server is
in fact busy several levels down. Count recursively instead:

```bash
find <cache>/<build_key> -name '*.elf' | wc -l
find <cache>/<build_key> -name '*.elf' -printf '%T@\n' | sort -rn | head -1
```

Two further dead ends worth recording, so nobody re-walks them:

- **Not host-side work.** The plugin already stubs `torch.randn` and friends (`_fast_randn`,
  `up_front_collect.py:218`) and allocates `ttnn.from_torch` shape-only.
- **Not "C++ ops bypass the collector".** They don't — `ttnn.rms_norm` goes through
  `ttnn::prim::layer_norm` on the new primitive infra and *is* intercepted. It collected all
  3,420 programs correctly. They were simply already compiled.

## What this is *not*

It is **not** an argument against `--precompile`. The batching machinery is correct and its
thread pool works; it is being handed an empty problem. Disabling precompile would hide the
defect, not address it, and would leave every other consumer of the pass slower for no reason.
