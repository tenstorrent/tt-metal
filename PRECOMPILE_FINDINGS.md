# Up-Front JIT Precompile + Remote Compile Farm — Experiments & Findings

**Status:** working prototypes, validated on real Wormhole hardware + a 128-core deviceless
compile farm. Last updated 2026-06-03.

This document captures what we built, what we reused, every experiment we ran, the numbers,
the key findings (including a few non-obvious gotchas), and the roadmap. It is meant to be
the single reference for picking this work back up.

---

## 0. TL;DR

The dominant cost of a cold tt-metal test/model run is **JIT kernel compilation**, done one
program (~5 RISC-V kernels) at a time behind sequential dispatch — so the host CPU is
starved. We attack that by **collecting the distinct program set up front and compiling it
in parallel**, optionally **offloading the compile to a remote multi-core farm**.

Headline results (full `groupnorm_sc_N_1_HW_C` golden suite, 2948 distinct programs,
identical test outcomes throughout):

| Configuration | Compile phase | Total run |
|---|---|---|
| Local, **sequential inline** — no precompile (today's default, 8-core box) | — (compiled inline) | **5325 s (88 min ≈ 1.5 h)** |
| Local, **parallel precompile** w8 (8-core box) | 2557 s | 2774 s (46 min) |
| Remote farm (128 cores), workers=64 | 483 s | 718 s (12 min) |
| **Remote farm (128 cores), workers=128** | **128.5 s** | **356.8 s (5:56)** |

→ Compile phase, **local-parallel-w8 → farm-w128: 19.9×** (8 → 128 cores, parallel vs parallel).
→ End-to-end vs the **true sequential-inline default (88 min): 14.9×**; vs local-parallel-w8
(46 min): 7.8×. Zero change in correctness (140 / 2808 / 3552 / 737 throughout).

> ⚠️ Note: `2557 s` is the *local parallel precompile* (w8) **compile phase**, **not** sequential
> inline. The no-precompile sequential baseline is **5325 s / 88 min** (overnight Run A). Don't
> conflate the two when quoting speedups.

---

## 1. What we built / reused

We ended up with **two distinct collection mechanisms** and **one reused execution backend**.

### Built #1 — C++ op collector *across one run* (e.g. a whole model)

A device-operation-level hook that, during a single forward pass, captures **every program**
the run would dispatch, then compiles the distinct set in parallel.

- **Where:** `ttnn/api/ttnn/up_front_compile.hpp`, `ttnn/core/up_front_compile.cpp`, a ~6-line
  stash in `ttnn/api/ttnn/device_operation.hpp` (`create_and_cache_mesh_workload`), nanobind
  bindings in `ttnn/core/graph/graph_nanobind.cpp`, re-exported in `ttnn/ttnn/graph.py`.
- **Branch:** `mstaletovic/up_front_compile` (based on current `origin/main`).
- **API:** `ttnn.graph.up_front_begin_collect()` → run `model.forward(...)` →
  `ttnn.graph.up_front_end_collect()` → `ttnn.graph.up_front_compile(device, max_workers)`.
- **How it works:**
  1. `begin_collect()` enters **NO_DISPATCH** graph capture — buffers are mocked (address 0)
     and nothing is dispatched, so the pass uses *no real device memory*.
  2. Each op flows through the universal funnel `ttnn::device_operation::launch<Op>` →
     `launch_operation_with_adapter<MeshDeviceOperationAdapter<T>>` →
     `create_and_cache_mesh_workload`, where the `MeshWorkload` (the built-but-uncompiled
     `Program`) is materialized. Our hook `std::move`s it into a thread-safe collector keyed
     by the program-cache hash, then returns (skips caching + enqueue).
  3. `up_front_compile()` drives `tt::tt_metal::detail::CompileProgram` over a thread pool.
- **Why it's op-agnostic:** the funnel handles all four factory concepts
  (`ProgramFactoryConcept` = classic `create()`/`override_runtime_arguments`,
  `ProgramDescriptorFactoryConcept` = Contract-1, `MeshWorkloadFactoryConcept` = Contract-2,
  `ProgramSpecFactoryConcept`). `create_output_tensors` runs *before* dispatch, so the forward
  keeps flowing on spec-correct (address-0) outputs without executing anything. Address-0 is
  fine because compilation is address-independent (kernel sources + compile-time args + CB
  structure; the program hash even excludes addresses).

### Built #2 — cross-parametrization collector *across a pytest session*

Collects one program per test case across **all parametrizations** of a suite, dedups, and
parallel-compiles before the test bodies run. (This is the eval golden-test precompile plugin;
it predates this session but is the second of the two mechanisms and we leaned on it heavily.)

- **Where:** `.claude/eval/precompile_plugin.py` (registered as `pytest_collection_finish`),
  `.claude/eval/precompile.py` (`collect_and_compile`).
- **API:** env-gated — `EVAL_PRECOMPILE=1`, `EVAL_PRECOMPILE_WORKERS=N`.
- **How it works:** monkey-patches the **Python** `ttnn.generic_op` to capture each case's
  `ProgramDescriptor` (raising to abort before compile/enqueue), dedups by
  `compute_program_descriptor_hash`, and parallel-compiles via
  `ttnn.precompile_program_descriptor` (a compile-only binding → `CompileProgram`).
- **Scope/limit:** only sees ops that dispatch through the **Python** `ttnn.generic_op` — i.e.
  the eval-generated ops. It is blind to C++ ProgramDescriptor-migrated ops, and it captures
  exactly **one** op per test case (the first `generic_op` call).

### How the two relate

Same architecture (collect distinct programs → dedup by structural hash → parallel compile) on
**orthogonal axes**:

- **#2 = cross-parametrization, one op per case** (the cartesian of shapes/dtypes/layouts).
- **#1 = cross-op within one invocation** (every op of a model forward).

For golden tests, where each case is a single `generic_op`, #2 is the full set. For a model
(many ops per run), you need #1. They compose: running #1 inside each case of a parametrized
session would capture *all ops of all cases* — which is exactly **Next Target A** below.

### Reused — the remote JIT compile server (jit_server)

Pre-existing experimental feature; we deployed and drove it, no code changes.

- **Where:** `tt_metal/tools/jit_compile_server/jit_compile_server.cpp` (the daemon),
  `tt_metal/impl/jit_server/` (`RemoteCompileCoordinator` client, `rpc.capnp`,
  `JitCompileService`). Routing is automatic in `tt_metal/impl/program/program.cpp:2016`
  (`JitCompileRpcClient::enabled()`).
- **Enable:** `TT_METAL_JIT_SERVER_ENABLE=1` + `TT_METAL_JIT_SERVER_ENDPOINTS=host:port`.
- **Properties:** deviceless (pure compiler farm — never opens a device, forbidden from
  `MetalContext`); **kernel-parallel** (thread pool sized to `hardware_concurrency`);
  **program-agnostic** (it has no notion of "program" — a program is a client-side grouping);
  process-wide kernel dedup + a `build_key`-partitioned on-disk cache. Endpoints can be sharded
  (`kernel_hash % N`) across multiple server boxes.

---

## 2. Key findings

1. **One universal C++ funnel.** Every device op — `generic_op`, the C++-migrated descriptor
   ops, *and* classic program-factory ops — converges at `launch_operation_with_adapter` →
   `create_and_cache_mesh_workload`. A single hook there is op-agnostic. (Confirmed structurally
   and empirically: the SDXL block's 18 programs span a mix of factory types and all compiled
   correctly.)

2. **The Python `generic_op` intercept is blind to migrated ops.** The "Metal 2.0" migration
   moves ops to C++ ProgramDescriptor factories that never call the Python `generic_op`. So
   mechanism #2 only covers eval-generated `generic_op` ops; mechanism #1 (C++ hook) is what
   generalizes.

3. **The farm parallelizes at the KERNEL level and is program-agnostic.** Cross-program
   concurrency is purely a *client-side* property driven by the worker count. Stock sequential
   dispatch feeds the farm one program's ~5 kernels at a time → ~5/128 busy → starved. Our
   collect-and-fan-out layer is the thing that fills the farm.

4. **FILE_PATH kernel gotcha (important).** ttnn references kernels as `KernelSource::FILE_PATH`,
   so the generated wrapper just `#include`s the kernel by path (`genfiles.cpp:56`). **Only the
   wrapper is uploaded**; the actual kernel `.cpp` is read from the *server's* filesystem at
   that path. Therefore the server's kernel sources must match the client's. Committed code on a
   matching checkout is fine (this is why stock SDXL kernels worked — md5 identical). **But an
   uncommitted/unpushed kernel edit makes the server silently compile stale code.** We hit this
   exactly: the groupnorm compute kernel had an uncommitted dst-corruption fix → with the stale
   kernel the farm gave 154 pass / 22 fail (catastrophic NaN/Inf on `1x1x32x320`, Cg=10, the
   `Cg%32 != 0` bug); after committing+pushing+pulling the fix it was 176 / 0.

5. **Saturation: workers must ≈ server cores.** Each client worker blocks in `finish()` until
   *all* its program's kernels return, and the `groupnorm_compute` kernel is the long pole
   (~4.5 s each). So **#concurrent compiles ≈ #workers**. Measured on the 128-core farm:
   - **w64:** ~60% util, run-queue `r` ≈ 48–98 → **under-fed**.
   - **w128:** ~88% util, run-queue `r` ≈ 120 → **well-fed**.
6. **Why it tops out ~88%, not 100%, and "spikes."** It is *not* IO-stall-bound (`wa` ≈ 0–3%).
   The residual is per-compile **fork/exec of `riscv-tt-elf-g++` + `ld`** plus heavy file IO
   (object/ELF/dep/genfile/ccache writes hit 270–900 MB/s) → ~12–25% **system** CPU and ~55k
   context-switches/s. The **spiking** is bursty submission (workers emit a pulse of ~5 kernels
   then wait ~4.5 s on the compute kernel) plus brief page-cache stalls during write bursts
   (blocked-process spikes `b` ≈ 14–35).
7. **The bottleneck shifts.** At w128 the compile is only 128.5 s, but the total is 356.8 s →
   **~228 s is non-compile**: the *serial* descriptor **collect** on the client + warm test
   execution of ~3000 cases + device init. That floor (≈ same as the overnight's ~220 s) now
   dominates. The compile is no longer the long pole.
8. **Single SSH tunnel had headroom** (~53% of one core at w128) but would become the next
   bottleneck at higher throughput → multiple endpoints/tunnels to scale past one farm box.

---

## 3. Experiments & results

### 3.1 Overnight local A/B (pre-session baseline)

Full groupnorm suite, cold cache, 8-core box. (Recorded in `precompile_overnight/SUMMARY.txt`.)

- A: cold sequential inline = **88 min**. B: cold → parallel precompile (w8) → warm = **46 min**
  (~1.9×). C (w4) ≈ B → **parallel compile saturates ≤4 workers on 8 cores; the build executor
  is the limiter**.
- Memory peaked at 125.9 GB but it was **reclaimable page cache** (RSS only 3.97 GB), peak during
  Run A, **zero OOM kills**. The SUMMARY's "FAILED / likely OOM" verdict is a scripting artifact —
  `exit=1` is just the suite's 140 pre-existing precision failures (Run A also exit=1).

### 3.2 Mechanism #1 validated locally (correctness)

- **Eltwise smoke** (`tests/ttnn/unit_tests/test_up_front_compile.py`): 3 ops → 3 programs,
  parallel compile 0 errors, warm-run PCC 0.999992.
- **Falcon-7B MLP** (`test_up_front_compile_mlp.py`, real `TtFalconMLP`, offline random weights):
  collect 2 programs; cold-inline first forward **1.72 s** vs precompile **1.41 s** up-front +
  **1.8 ms** warm forward; PCC 0.9997. (Only 2 programs, so parallelism barely matters — pure
  correctness check that the funnel captures matmul + fused gelu.)
- **SDXL `BasicTransformerBlock`** (`test_up_front_compile_transformerblock.py`, real
  `TtBasicTransformerBlock`, offline random weights): self-attn + cross-attn + GEGLU + 3
  layernorms → **18 distinct programs**. Local A/B (fresh cold cache): cold-inline first forward
  **15.64 s** vs precompile (w4) **10.54 s** compile + **12.7 ms** warm; PCC 0.9991. First
  evidence the hook handles a complex real block (matmul, sdpa/softmax, layernorm, geglu).

### 3.3 SDXL block on the remote farm (first remote proof + speedup)

Built `jit_compile_server` on `bgdepyc01` (128 cores, deviceless) at the identical path,
ran it on `localhost:9876`, tunneled from the client. Server log confirmed the block's real
kernels (`sdpa`, `bmm_large_block`, `nlp_create_qkv_heads`, reader/writer) compiled remotely.

A/B/C on the farm, cold server cache each, PCC 0.9991 throughout:

| Scenario | Compile cost | vs stock |
|---|---|---|
| stock (no precompile, sequential remote) | 9.41 s | 1.0× |
| precompile w4 | 3.40 s | 2.8× |
| precompile w18 (= the block's program count) | **1.97 s** | **4.8×** |

### 3.4 Groupnorm on the remote farm (the real workload)

- **Subset (176 programs):** surfaced the FILE_PATH stale-kernel gotcha (§2.4): 154/22 with the
  stale kernel → **176/0** after committing+pushing+pulling the fix. Confirmed fresh remote
  compiles of `groupnorm_sc_N_1_HW_C_compute` in the server log.
- **Full suite (2948 programs):** see the TL;DR table. **w64 → 483 s compile / 718 s total;
  w128 → 128.5 s compile / 356.8 s total.** Baselines: local *sequential inline* (no precompile,
  overnight Run A) = **5325 s / 88 min**; local *parallel precompile* w8 (Run B) = 2557 s compile
  / 2774 s total. So farm-w128 is **14.9× end-to-end vs the sequential-inline default**, and its
  compile is 19.9× the local-w8 precompile (8 → 128 cores). Identical outcomes (140 / 2808 / 3552
  / 737). Server vmstat at w128: avg **88% util, run-queue ~120**.

---

## 4. Limitations & gotchas (current state)

- **Path identity.** The client sends absolute `-I`/`gpp` paths; the server reads headers + runs
  the compiler from *its own* FS at those paths. So the repo + sfpi toolchain must exist on the
  server at the **identical absolute path**. We satisfied this by cloning the same branch to the
  same path and building just the `jit_compile_server` target.
- **Kernel-source sync (FILE_PATH gotcha, §2.4).** Uncommitted kernel edits compile stale on the
  server. Today's workflow: commit + push + pull on the server (or copy the file).
- **Worker sizing.** The `up_front_compile` default uses *client* `hardware_concurrency`; for the
  farm you must pass a high `max_workers` (≈ server cores). Eval plugin default caps at
  `min(cpus, 4)` (tuned for an 8-core cgroup) — set `EVAL_PRECOMPILE_WORKERS` explicitly.
- **Coordinator-per-program overhead.** Each `CompileProgram` spins its own
  `RemoteCompileCoordinator`; process-wide dedup + firmware gating make it correct but slightly
  wasteful. A single batch coordinator would be cleaner.
- **NO_DISPATCH doesn't cache.** Mechanism #1 warms only the *on-disk kernel cache* (the dominant
  win). It does not pre-populate the in-memory device program cache (hard — typed
  `shared_variables_t`; probably unnecessary).
- **Security.** jit_server is experimental: no auth, no encryption. Use a trusted network / SSH
  tunnel (we tunneled over the existing SSH).

---

## 5. Next targets

### A. Make mechanism #1 collect across parametrizations of *normal* ttnn tests

Today the cross-parametrization collection (mechanism #2) is shaped around the eval golden tests
and is `generic_op`-only. Real ttnn tests have **multi-op bodies** — e.g. a conv test isn't just
`conv`, it's `interleaved→sharded → halo → move → conv`. The goal: drive **mechanism #1** (the
op-agnostic C++ collector) across an entire pytest session so it captures *all* ops of *all*
cases, regardless of how the op dispatches or how many ops a test body runs. Concretely: a
pytest plugin that wraps each collected test in `up_front_begin_collect()/end_collect()` (or runs
all bodies in collect mode first), accumulating one global distinct-program set, then one
parallel compile — independent of the golden-test structure.

### B. Make it dramatically easier to use / more universal

The current friction: identical absolute paths on client+server, and commit/push/pull to sync.
Investigate removing these:
- **Path-independence** on the server (remap client paths, or upload the full kernel source set
  instead of `#include`-by-path so the server needs no matching tree).
- **Source sync without git** (the server pulls/receives the working tree, or a content-addressed
  upload of all referenced kernel files).
- **Zero-config client** (auto-discover/launch a farm, set env once). Aim for "point at a farm,
  run your tests" with no per-branch deployment dance.

### C. More thorough testing

Beyond the 3 module tests + groupnorm: multi-op test bodies, conv/sharded paths, more ops/shapes,
correctness parity (remote vs local) at scale, and failure-mode coverage (server down, stale
kernel detection, partial compile errors).

### D. The broader vision — beyond the eval system

This is generally useful, not eval-specific. **Any** cold tt-metal run — CI suites, model
bring-up, sweep tests — pays the JIT-compile tax. A universal "collect distinct programs → farm
compile" layer could cut cold-run time across the board. The op-agnostic C++ hook + the existing
jit_server are the building blocks; the work is making collection universal (A) and deployment
frictionless (B).

Also worth noting once the compile is cheap (§2.7): the **serial collect + warm execution** become
the bottleneck, so parallelizing those is the next lever for end-to-end (not just compile) wins.

---

## 6. Appendix — where everything lives

### Branches / commits

- **`mstaletovic/up_front_compile`** (origin/main-based; pushed): C++ mechanism #1 + 3 tests +
  `UP_FRONT_WORKERS` knob. Commits `945d2e1`, `0e30031`, `f10f2e5`, `2199678`, `8fceacf`.
- **`mstaletovic/group_norm_jit_profiling`** (pushed): groupnorm kernel dst-corruption fix
  (`e4bde58`) + the docs (`22df8d2`). Has local WIP not committed (build.cpp, run_safe_pytest.sh,
  the tt_ops_code_gen submodule pointer).

### Key files

| Purpose | File |
|---|---|
| C++ collector API/impl | `ttnn/api/ttnn/up_front_compile.hpp`, `ttnn/core/up_front_compile.cpp` |
| Funnel hook | `ttnn/api/ttnn/device_operation.hpp` (`create_and_cache_mesh_workload`) |
| Bindings | `ttnn/core/graph/graph_nanobind.cpp`, `ttnn/ttnn/graph.py` |
| Eval cross-param plugin | `.claude/eval/precompile_plugin.py`, `.claude/eval/precompile.py` |
| jit_server daemon | `tt_metal/tools/jit_compile_server/jit_compile_server.cpp` |
| jit_server client/RPC | `tt_metal/impl/jit_server/` |
| Remote routing | `tt_metal/impl/program/program.cpp` (~:2016) |
| Companion docs | `PRECOMPILE_EXPLAINED.md`, `MODEL_PRECOMPILE_DESIGN.md`, `REMOTE_JIT_SERVER_SETUP.md` |

### Reproduce the remote farm

1. Server (deviceless, e.g. `ssh -p 49210 bgdepyc01`): clone the branch to the *identical* path,
   `cmake -B build_Release ... -DWITH_PYTHON_BINDINGS=OFF`, `ninja -C build_Release
   jit_compile_server` (downloads sfpi at configure; ~3 min on 128 cores).
2. Run it: `TT_METAL_JIT_SERVER_ENDPOINT=localhost:9876
   TT_METAL_JIT_SERVER_CACHE_ROOT=<path>/jit_server_cache setsid ./build_Release/tools/jit_compile_server &`
   (verify with a `/dev/tcp` connect; `ss` can be blind in this env. Never
   `pkill -f jit_compile_server` — it matches your own remote shell.)
3. Tunnel: `ssh -f -N -L 9876:localhost:9876 -p 49210 bgdepyc01`.
4. Client: `EVAL_PRECOMPILE=1 EVAL_PRECOMPILE_WORKERS=128 TT_METAL_JIT_SERVER_ENABLE=1
   TT_METAL_JIT_SERVER_ENDPOINTS=localhost:9876 TT_METAL_CACHE=<fresh> scripts/run_safe_pytest.sh
   --run-all <golden_test.py>`. For a clean cold measurement, clear `<path>/jit_server_cache`
   between runs.

### Running infrastructure (as of this writing)

Two jit servers on `bgdepyc01` (`:9876` for the `up_front_compile`/origin-main tree, `:9877` for
the groupnorm tree) + two client SSH tunnels. Tear down when done.
