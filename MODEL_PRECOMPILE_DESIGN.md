# Design: Whole-Model Up-Front Parallel Precompile

**Status:** proposal / design sketch
**Author context:** generalizes the eval golden-test precompile (`PRECOMPILE_EXPLAINED.md`) from the Python `ttnn.generic_op` boundary down to the C++ op-dispatch convergence point, so it works for *any* ttnn op — including the C++ ProgramDescriptor-migrated ops.
**Anchors:** verified against origin/main `e04fae81c7f` (worktree at `…/wt_origin_main`).

---

## 1. One-paragraph summary

Run a model's `forward` **once** in a new **"collect" mode** where each op builds its program but does **not** dispatch it — instead the freshly-built program (and its cache hash) is stashed, and the op's already-allocated output tensor is returned so the forward keeps flowing. After the pass you hold the model's whole **distinct program set**. Then **compile that set in parallel** (`detail::CompileProgram` across a thread pool), warming the on-disk JIT kernel cache. The subsequent real run — or trace capture — then runs **warm**, paying compile once, up front, with the cores filled, instead of serially-inline on the critical path. This is the exact capture→dedup→parallel-compile architecture of the eval tool, lifted into C++ so it is op-agnostic.

---

## 2. Background: what we already proved, and the limitation

The eval golden-test tool (`PRECOMPILE_EXPLAINED.md`) established empirically (overnight A/B, `precompile_overnight/SUMMARY.txt`):

- On a cold cache, **~99% of wall time is JIT kernel compilation**, and tests run sequentially so only ~3.8/8 cores are busy.
- Capturing the distinct `ProgramDescriptor` set and compiling it in parallel up front cut the full groupnorm suite **88 min → 46 min (~1.9×)**, identical results.
- Parallelism **saturates at ~4 workers** (the process-wide build executor / `jit_server` `RemoteCompileCoordinator` is the limiter, not the worker pool). Expect **~2×, not N×**.

**The limitation:** that tool monkeypatches the **Python** `ttnn.generic_op`. Production ttnn ops are migrating to **C++** ProgramDescriptor factories ("Metal 2.0 / Contract-1/Contract-2") — slice, attn_matmul, conv2d/3d, halo, plusone, reshape_tiled, pad, nlp_concat_heads, … These build their descriptor in C++ and never call the Python `generic_op`, so the Python intercept is **blind** to them. A whole-model tool must hook in C++.

---

## 3. The architectural facts that make this possible (with anchors)

All paths are in the worktree; line numbers are origin/main `e04fae81c7f`.

**F1 — Every op converges on one dispatch funnel.** `generic_op` and every migrated op are device operations dispatched via `ttnn::device_operation::launch<Op>()`, whose single body wraps the op in a mesh adapter and calls `launch_operation_with_adapter`:
- `ttnn/api/ttnn/device_operation.hpp:488` — `detail::launch_operation_with_adapter<MeshDeviceOperationAdapter<device_operation_t>>(...)`
- `ttnn/api/ttnn/device_operation.hpp:370` — `launch_operation_with_adapter(...)` definition.
- `generic_op` is just `GenericOpDeviceOperation` with `operation_attributes_t = MeshProgramDescriptor` and `program_factory_t = GenericMeshProgramFactory` — it rides the same funnel. So a hook here **subsumes the current Python tool**.

**F2 — Output allocation happens *before* the program runs.** In `launch<>`:
- `ttnn/api/ttnn/device_operation.hpp:457` — `create_output_tensors(...)` (which internally calls the op's pure `compute_output_specs`) allocates outputs **before** `:488` dispatches anything.
- ⇒ In collect mode, even if we never build/run the program, the op already returned a spec-correct output tensor, so **the next op can build its descriptor and the forward continues**. This is the C++ equivalent of the harness intercept that returns `tensors[-1]`.

**F3 — The program is materialized at one identifiable spot, separately from compile and from enqueue.** Inside `create_and_cache_mesh_workload`:
- `ttnn/api/ttnn/device_operation.hpp:332` — `create_mesh_workload_from_workload_factory<...>(...)` builds the workload (the `Program` from the descriptor) — **structure only; kernels are not JIT-compiled here.**
- `:347-357` — on a normal run: insert into program cache **and** `enqueue_mesh_workload`.
- `:358-365` — else: `enqueue_mesh_workload` without caching.
- ⇒ The built program at `:332` is exactly what we want to stash; compile + enqueue are downstream and skippable.

**F4 — The dedup key is already computed.** `:391` — `program_hash = compute_mesh_workload_hash(...)`. For `generic_op` this is `compute_program_descriptor_hash` (`generic_op_device_operation.cpp:48`); for migrated ops it's their (possibly shape-aware) `compute_program_hash`. Either way it is the device program-cache key — the right dedup key, free.

**F5 — A "build-but-don't-cache-and-don't-dispatch" path already exists** (NO_DISPATCH graph capture):
- `:335-346` — detects the `GraphTracker` hook and sets `hook_blocks`; `:346` `should_cache = enabled && !hook_blocks`.
- Buffers in this mode have **address 0** (`:335-337` comment). Because compile + the program hash are **address-independent** (`compute_program_descriptor_hash` excludes addresses), address-0 is fine for our compile goal.
- ⇒ ~80% of "collect" plumbing exists. We extend the hook to **stash** the program built at `:332` instead of discarding it.

**F6 — The parallel-safe compile primitive exists.** `tt_metal/api/tt-metalium/tt_metal.hpp:198` — `detail::CompileProgram(IDevice*, Program&, bool)` JIT-compiles a program's kernels to the on-disk cache **without enqueue**, touching no command queue. Safe to call concurrently for distinct programs (this is what the eval tool's `precompile_program_descriptor` wraps).

**F7 — Coverage can be *verified*.** `:393-396` — `set_program_cache_misses_allowed(false)` makes any cache miss `TT_THROW`. Run the real model under it after precompile to prove every program was warmed.

---

## 4. What a real model looks like (ResNet-50)

Grounding the orchestration in `models/demos/vision/classification/resnet50/`:

- **Model = plain Python class**, weights preprocessed to `ttnn.Tensor`s; forward is a static call sequence of `ttnn.*` ops.
  - Entry: `ttnn_resnet/tt/ttnn_functional_resnet50.py:598` `__call__(self, input_tensor, device, ops_parallel_config)` → `:605` `run(...)`: `fold → conv2d(stem) → max_pool2d → 16 bottleneck blocks (each 3-4 conv2d + add_) → avg_pool2d → linear → untilize`.
- **Op sequence is static given input shape.** All branching is on Python constants fixed at construction (arch, batch, stride); the 16 blocks are fully unrolled; no control flow on tensor *values*. ⇒ **A single collect-mode forward with any input (even zeros) produces the complete, correct program set.**
- **~53 `conv2d` invocations**, but blocks within a layer share shapes ⇒ **~30-35 unique programs**. The device cache already dedups by (op, shape, dtype, mem_config, compute_config); our phase-2 dedup uses the same `program_hash` (F4).
- **Current AOT pattern** (`models/tt_cnn/tt/executor.py:159` `compile()`): **first forward = serial inline JIT** (`_run_model_for_compilation`), **second forward = trace capture** (`begin/end_trace_capture`). The first forward's serial cold compile is exactly the bottleneck the eval tool showed dominates — **this is what we replace.**

---

## 5. The design

### Two phases, mirroring the eval tool

```
Phase 1 — COLLECT (serial, cheap: no JIT, no dispatch)
  begin_collect()                                  # set a process-global CollectMode hook
  output = model.forward(dummy_input, device)      # runs once; each op:
       launch<Op>:
         create_output_tensors            (:457)   # spec-correct output (addr may be 0) → forward continues
         launch_operation_with_adapter:
           program_hash = compute_…hash   (:391)   # dedup key, already computed
           create_…workload_from_factory  (:332)   # builds Program from descriptor (NO kernel compile)
           >>> COLLECT HOOK <<<                     # stash {program_hash: Program}; SKIP cache + SKIP enqueue
  programs = end_collect()                          # dict hash → Program, naturally deduped

Phase 2 — COMPILE (parallel: the expensive part)
  ThreadPool(workers ≈ 4):                          # >4 saturates the build executor (proven)
      for prog in programs.values():
          detail::CompileProgram(device, prog)      # (:tt_metal.hpp:198) warms on-disk kernel cache

Then: real run / trace capture is WARM — JIT is a disk hit.
```

### Why the forward keeps flowing without executing

`create_output_tensors` (F2) already produced the output before any program work. The collect hook skips the build's *use* (compile/enqueue) but the op still returns its allocated output. The next op consumes it for its own `compute_output_specs` / descriptor build. Address-0 buffers are sufficient because everything we do in both phases is **structural** (F5) and the hash is address-free (F4). This is precisely the harness's `return tensors[-1]` trick, now intrinsic to the C++ machinery.

---

## 6. Where the code goes

### C++ (the load-bearing part)

1. **A collect registry + hook.** Extend `ttnn::graph::ProcessorHooks` (or add a sibling `CollectHooks`) reached via `GraphTracker::instance().get_hook()` — the same mechanism already consulted at `device_operation.hpp:340-345`. Add a callback like `on_workload_built(program_hash, const Program&)` and a thread-safe `unordered_map<uint64_t, Program>` collector keyed by `program_hash` (F4).
2. **One stash call** in `create_and_cache_mesh_workload`, right after the workload is built at `device_operation.hpp:332` and before the `should_cache`/enqueue branch (`:347`): if a collect hook is active, register the program (move or descriptor-copy) under `program_hash` and **return early** (skip both cache insert and `enqueue_mesh_workload`). This is a ~10-line localized change at the single funnel (F1, F3).
3. **A parallel compile driver.** Either C++ (`std::thread`/taskflow over `CompileProgram`) or expose the collector to Python and drive a `ThreadPoolExecutor` (GIL released inside `CompileProgram`, exactly like the eval tool). Cap workers ~4.
4. **Python bindings:** `ttnn.begin_collect()` / `ttnn.end_collect() -> handle`, `ttnn.parallel_compile(device, handle, workers=4)`. (The eval branch already added `precompile_program_descriptor` + `compute_program_descriptor_hash`; this generalizes them off the descriptor and onto the built program.)

### Reuse

`ttnn.graph.begin_graph_capture(NO_DISPATCH)` already gives you: build-but-don't-cache, don't-dispatch, output-spec propagation. The delta is *stashing the program* (today it's discarded) and the parallel compile. Prototype by extending `ProcessorHooks` rather than greenfielding.

### Python orchestration (grounded in resnet)

```python
device = ttnn.open_device(...)
test_infra = create_test_infra(device, batch_size=16, ...)        # builds model + preprocessed weights
host_in, mem_cfg = test_infra.setup_l1_sharded_input(device)       # any input; zeros fine (static sequence)
test_infra.input_tensor = host_in.to(device, mem_cfg)

h = ttnn.begin_collect()
test_infra.run()                                                   # ONE collect-mode forward
programs = ttnn.end_collect()                                      # ~30-35 unique programs (resnet)

ttnn.parallel_compile(device, programs, workers=4)                 # warm on-disk kernel cache, parallel

# now the existing AOT path is warm:
pipeline.compile(host_in)   # executor.py:159 — its first forward is now a disk-hit, not cold JIT
```

This slots in *ahead of* `TracedModelExecutor.compile` (`executor.py:159`), turning its serial cold first-forward into a warm one.

---

## 7. Scope tiers (be honest about the cliff)

**Tier 1 — warm the on-disk KERNEL cache (recommended, low-risk).** Exactly the design above. The real run still constructs `Program` objects and populates the in-memory device program cache, but **kernel JIT becomes a disk hit** — the dominant cost. This is what the eval tool does (it never touches the device program cache) and is where the proven ~2× comes from. No memory blowup: collect mode allocates no real intermediates (addr-0, like NO_DISPATCH).

**Tier 2 — also pre-populate the in-memory device `ProgramCache` (stretch).** Would skip even the `Program` rebuild on the real run. Open problems, all real:
- **Memory:** to insert cache entries with valid runtime args you need real buffer addresses; allocating every intermediate of a whole model simultaneously (collect never frees) risks OOM. Real execution allocates/frees along the dataflow.
- **Typed insertion:** `program_cache.insert` needs a `CachedProgramFactory{ cached_workload, factory_index }` whose `shared_variables_t` is a per-op template type — there is no generic insert path (the agent survey confirmed this gap).
- Likely **not worth it**: Tier 1 already removes the JIT cost; Program reconstruction is cheap by comparison.

Recommend shipping Tier 1; treat Tier 2 as a later optimization only if profiling shows Program-rebuild is material.

---

## 8. Caveats & risks

- **~2× ceiling, not N×.** The build executor / `jit_server` saturates ~4 workers (overnight-proven). The win is *parallel + up-front* vs *serial + inline*, not raw compiler throughput.
- **Static-sequence assumption.** ResNet's op sequence is fully determined by input shape (verified). Models with data-dependent control flow (value-dependent branching, dynamic shapes) would have collect mode capture only the path the dummy input takes; document this and let callers run collect with representative inputs (or multiple passes).
- **Contract-2 ops own intermediate device buffers.** Their `create_workload_descriptor` may allocate scratch tensors whose addresses the kernels read. Building such a workload at `:332` under addr-0/NO_DISPATCH needs validation — confirm the build path tolerates it (NO_DISPATCH already exercises `:332`, so likely yes, but test specifically: conv2d, halo, reshape_tiled).
- **Mesh/multi-device:** compiling for `devices[0]` warms the whole homogeneous mesh (one `build_key`), per the `precompile_program_descriptor` comment in the eval binding. Confirm for heterogeneous meshes.
- **Address-0 correctness:** safe for compile and hashing (both structural/address-free); do **not** attempt to enqueue a collect-mode workload.

## 8.1 Execution-state dependence — two hard limits (verified against the source, 2026-06-04)

Both limits below share **one root cause: NO_DISPATCH faithfully reproduces program *structure*, but nothing that depends on real device *execution*.** Under NO_DISPATCH the capture hook mocks every buffer at address 0 (`ProcessorHooks::hook_allocate` returns `do_block`, set true for NO_DISPATCH — `graph_processor.cpp:884,694`) and blocks dispatch (`hook_program` — `:900`). So any program *selection* or test *logic* that reads back the live runtime state of the device diverges from a real run. Both degrade **gracefully** (a cold recompile, never a wrong result — the on-disk kernel cache is content-addressed and the real run always rebuilds the program it actually needs), and both are **Phase-2 / whole-model concerns that do not affect the `generic_op` golden marriage**.

### (a) Ops that select their program from *live allocator state*

Some ops branch their program on how much L1 is currently free — exactly the "L1-saving path vs fast path" decision. They read it via `device->lowest_occupied_compute_l1_address()` or `allocator()->get_statistics(L1)`:

- `matmul/device/utilities/matmul_utilities.cpp:76-82` (`get_max_l1_space`): `auto lowest = device->lowest_occupied_compute_l1_address(); max_l1 = lowest.has_value() ? lowest.value() : device->l1_size_per_core();`
- `reduction/generic/device/common.cpp:215-223`: same idiom, returning `negate_cb_bytes <= max_l1_space` to pick a reduction strategy.
- Also: `transpose.cpp:147`, `data_movement/common/common.cpp:431`, `topk` (`largest_free_block_bytes`), `sliding_window/op_slicing`, several CCL ops.

**Under NO_DISPATCH nothing is really allocated**, so `lowest_occupied_compute_l1_address()` returns `nullopt` for the *whole* collect pass → these ops see **empty/full L1** and are systematically biased toward the **fast / L1-resident** path. A real run, under genuine L1 pressure, may pick the **L1-saving** path → a *different compiled program* (different CBs / compile-time args → different kernel hash) → the precompiled variant is a **cold miss** at run time for that op.

- **Not a new correctness risk.** The in-memory device program cache *already* keys on `compute_mesh_workload_hash` = op attrs + tensor specs, **not** live allocator state (`device_operation.hpp:399`). An op whose *kernel structure* truly varied with transient L1 would already risk the program cache serving a stale variant across runs; well-behaved ops therefore confine the allocator-derived choice to things that don't change the compiled kernel (e.g. *where* to place a scratch buffer = runtime addresses). The ones that genuinely fork kernel structure on live L1 are precisely the ones that miss-and-recompile under collect — slower, never wrong.
- **No impact on `generic_op` ops** (incl. `groupnorm_sc_N_1_HW_C`): their program is a pure function of the `ProgramDescriptor` (shapes/dtypes/layout/CB config), with no allocator query. Collected program == real program.
- NO_DISPATCH **cannot** reproduce a realistic allocator state (it mocks allocation), so these ops are inherently outside its clean coverage. Accept the cold-recompile fallback.

### (b) Tests / bodies that themselves drive trace or graph capture

A trace records **real dispatches into a real trace buffer** and replays them; ttnn has ~39 such tests (matmul/CCL: `test_rs_matmul_1d_gather_in0.py`, `test_new_matmul_reduce_scatter.py`, `test_llama_all_gather_matmul.py`), ~6 graph-capture tests, and one eval golden suite (tilize's `test_deepseek_v3_mla_tilize_trace_mode`). Running such a body **inside** a collect window breaks:

- NO_DISPATCH blocks the dispatch (`hook_program`) and the trace-buffer allocation (`hook_allocate`) that recording needs → the captured trace is empty/invalid → `end_trace_capture`/`execute_trace` error (swallowed best-effort).
- **Worse:** it can leave device/CQ state (CQ stuck in trace-record mode, a dangling trace buffer) that **poisons the rest of a shared collect window** — one bad body breaks every dispatcher after it in the single-`begin_collect` design. (A per-test hookwrapper would contain the blast radius, but the test still fails.)
- Graph capture nests as a **stack** (`push_processor`/`pop_processor`, `graph_processor.cpp:844-851`) with an **install-once hook** (`:691` adds a hook only if none exists) → an inner `begin_graph_capture` inherits our NO_DISPATCH blocking and sees an empty/wrong graph.
- The funnel hook is *upstream* of trace-record/CQ-enqueue, so the ops' programs would still be collected up to the point the body crashes — but you can't rely on it getting there.

**Phase 1 is safe by construction.** groupnorm has zero capture tests, and the eval plugin's `_eligible` (`precompile_plugin.py:37-47`) only re-invokes the canonical `test_op(inputs, axes, device)` parametrization. The one tilize trace test is a bespoke function without `inputs`/`axes` params → filtered out, never fed to NO_DISPATCH. The eligibility filter is an accidental but effective guard.

**Phase 2 must explicitly detect + skip capture-driving tests** (static scan for `begin_trace_capture`/`begin_graph_capture`, or a runtime guard that aborts a test's collection if a capture is entered while collect is active). The cost is ~nil: they're ≈45 of thousands, and a trace test *already* compiles everything in its warmup loop before capturing, so it gains nothing from in-region precompile. The correct integration for a trace workload is to run collect **before** the test's capture (warm the kernel cache), then let the test's own warmup+trace run warm — the §4 pipeline. **Precede the trace region with collect; never wrap it.**

---

## 9. Validation plan

1. **Correctness of the program set:** run resnet once NORMALLY with program cache on; record the set of `program_hash`es inserted. Run collect mode; assert the collected hash set **equals** the normal-run set. (Same key, F4.)
2. **Coverage:** after collect + parallel-compile, run the real forward under `device.set_program_cache_misses_allowed(false)` is *not* what proves kernel-cache warmth (that's the in-memory cache) — instead assert **zero `jit_build` events** in `JIT cache stats` telemetry on the first real forward (the same telemetry line the overnight logs use: `JIT cache stats: H/N hits`).
3. **Speed:** time existing `pipeline.compile()` (cold) vs `collect + parallel_compile + compile()`; expect the first-forward JIT term (`profiler.get("compile")` in `perf_e2e_resnet50.py`) to collapse toward the warm number.
4. **Saturation:** sweep workers ∈ {1,2,4,8}; expect knee at ~4 (reproduces the overnight w4≈w8 result).

---

## 10. Milestones

1. **M0 — feasibility spike (Python-only, no C++ build):** confirm a collect-style walk via `ttnn.graph.begin_graph_capture(NO_DISPATCH)` over resnet enumerates the expected ~30-35 ops + specs. Validates F2/F5 and the static-sequence claim before touching C++.
2. **M1 — C++ collect hook + registry** (extend `ProcessorHooks`, stash at `device_operation.hpp:332`), Python `begin/end_collect`.
3. **M2 — parallel compile driver** over `CompileProgram`; ship Tier 1.
4. **M3 — validate on resnet** (§9); compare against the trace-AOT baseline.
5. **M4 — generalize** to another model (a transformer block) to stress non-conv ops and the static-sequence assumption.

---

## Appendix: key anchors

| What | File:line |
|---|---|
| Universal dispatch funnel | `ttnn/api/ttnn/device_operation.hpp:488` (`launch`→adapter), `:370` (adapter def) |
| Output allocated before dispatch | `ttnn/api/ttnn/device_operation.hpp:457` |
| Program built (structure, no compile) | `ttnn/api/ttnn/device_operation.hpp:332` |
| Dedup hash already computed | `ttnn/api/ttnn/device_operation.hpp:391` |
| NO_DISPATCH build-no-cache-no-dispatch | `ttnn/api/ttnn/device_operation.hpp:335-346` |
| Cache-miss-forbidden enforcement | `ttnn/api/ttnn/device_operation.hpp:393-396` |
| Compile w/o enqueue (parallel-safe) | `tt_metal/api/tt-metalium/tt_metal.hpp:198` |
| generic_op = device op, descriptor hash | `…/generic/device/generic_op_device_operation.cpp:48`; `…/generic_op.cpp:13,26` |
| ResNet forward | `models/demos/vision/classification/resnet50/ttnn_resnet/tt/ttnn_functional_resnet50.py:598,605` |
| ResNet current AOT (warmup+trace) | `models/tt_cnn/tt/executor.py:159` |
