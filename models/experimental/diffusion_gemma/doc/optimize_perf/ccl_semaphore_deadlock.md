# A cache-missing `all_reduce` can permanently wedge the device (2026-07-30)

Status: current. A production-class hang, distinct from the `AllBroadcast` variant already
recorded in the [stage hub](README.md#trace-hazards-and-the-trace-lifetime-rule).
Owns: the `GlobalSemaphore` / `finish_nolock` deadlock, its two observed triggers, the
recovery recipe, and the reason "no `BuildKernels` lines" does not rule out a program-cache
miss on this path.
See also: [stage hub](README.md) · [work log](work_log.md) · [plan](../../plan.md) section 5.

## The symptom

The process stops making progress **at 100% CPU on one core**, with no child compiler
process and no further log output. It is not a slow compile and not an idle device wait:
`top` shows CPU time accumulating (observed 11–14 minutes) while the log mtime is frozen.
It never recovers. `tt-smi -r` is required.

Because the spin is at 100% CPU rather than 0%, it does not look like the familiar
"`open_mesh_device` hangs at 0% CPU" trace-FATAL signature, and because it emits no log
line it is easy to mistake for a cold-JIT-cache compile.

## The stack (gdb, live, `thread apply 1 bt`)

```
ttnn::all_reduce
  ttnn::experimental::all_reduce_async
    ttnn::reduce_scatter  ->  ttnn::prim::reduce_scatter
      ReduceScatterDeviceOperation::ReduceScatterProgram::create_mesh_workload   <-- cache MISS
        ttnn::global_semaphore::create_global_semaphore
          tt::tt_metal::GlobalSemaphore::GlobalSemaphore
            GlobalSemaphore::setup_buffer
              GlobalSemaphore::reset_semaphore_value
                MeshCommandQueueBase::enqueue_write_mesh_buffer
                  MeshCommandQueueBase::enqueue_write_shard_to_sub_grid
                    FDMeshCommandQueue::finish_nolock       <-- pthread_cond_wait, forever
```

## The mechanism

`create_mesh_workload` on the reduce-scatter path means the program cache **missed**. A miss
mints a fresh `GlobalSemaphore`, and `GlobalSemaphore::setup_buffer` initialises it with a
**blocking** `enqueue_write_mesh_buffer`, which calls `finish_nolock` — a full command-queue
drain. So *every cache-missing `all_reduce` synchronously drains the command queue*, and if
the queue cannot drain, the process waits forever.

Two consequences worth separating:

- **Latency.** A DiffusionGemma prefill forward issues roughly **90 `all_reduce`s**
  (3 per layer x 30 layers; the DG denoise call sites are `denoise_attention()` in
  `tt/diffusion_attention.py`, `shared_mlp_forward()` in `tt/expert_operations.py` and
  `concat_experts_forward()` in `tt/concat_moe.py` — named by symbol because all three line
  numbers this file used to carry had drifted — and the prefill forward runs the
  shared gemma4 collectives). If those miss, the prefill pays ~90 forced drains. This is the
  leading candidate for the request-13 prefill cliff (`prefill_s` 0.5 -> 12 s), because it is
  shape-independent, block-count-independent, and — decisively — **non-traced-only**: a
  replayed trace does not re-create programs, which is exactly why the traced denoise step
  stayed flat at 197.4 ms across the same 5-hour run in which prefill degraded 24x.
  **NOT YET CONFIRMED** — see "What is still open".
- **Deadlock.** The same drain can never complete, and then the device is unusable.

### Why the "zero `BuildKernels` lines" check did not catch this

A program-cache miss here creates a **semaphore**, not a kernel. `grep`ping a production log
for `BuildKernels` / `Compiling` returns nothing and yet the cache is missing. Do not use the
absence of compile lines to rule out a program-cache miss on a CCL path.

## Two observed triggers

1. **A second `session.prefill()` on one session while the 48 up-front traces are resident.**
   Reproduced by `repro_prefill_percall.py` (13 sessions x 2 prefills). The first prefill
   succeeds; the second wedges as above. This is the same class as the recorded
   second-request `AllBroadcast` stall — creating a program while traces are resident
   violates trace address stability and corrupts CCL state — but it surfaces on
   ReduceScatter's global-semaphore write instead.
2. **Trace capture itself, after a previous run was killed mid-capture.** Both runs launched
   after a `pkill` of a run that was inside `warmup_model_prefill` hung in
   `warmup_model_prefill` (`generator_vllm.py:568`) with the identical 100% CPU signature,
   while the two runs before that `pkill` completed warmup+capture in **21 s**. A `tt-smi -r`
   plus a `(1,4)` mesh open/close smoke was **not sufficient** to clear it — the mesh smoke
   passes and capture still spins, because the smoke never captures a trace.

**OPERATIONAL RULE: never `pkill` a run that is inside trace capture.** Let it finish or
reach a clean exit. A mid-capture kill leaves state that survives a device reset and that a
mesh-open smoke will not detect.

## Recovery

```bash
pkill -9 -f <harness>                       # the spin ignores SIGTERM
sudo /home/zni/.local/bin/tt-smi -r
# mesh smoke -- necessary but NOT sufficient: it does not exercise trace capture
MESH_DEVICE=P150x4 pytest models/demos/gemma4/tests/unit/test_model.py::test_single_layer_model -k "1x4"
```

If capture still spins after that, clear the JIT cache (`~/.cache/tt-metal-cache`, ~14 GiB on
`bh-qbge-06`) and accept one cold compile. **Unverified** — it is the documented remedy for
JIT-cache staleness and it is the remaining difference, but this specific spin was not
confirmed to clear that way, and the cache is shared with other sessions on the box.

## What is still open

- **Why the misses begin at request ~13.** A program cache is normally monotonic, so
  "misses start at 13" needs a reason. Three candidates, not yet distinguished:
  **(A) unbounded growth** — the program hash includes the `GlobalSemaphore` identity, so
  every call mints a new entry and every request always missed (then the puzzle becomes why
  requests 1–12 are *fast*, for which "less outstanding work to drain early" is a testable
  hypothesis); **(B) eviction** at some ceiling; **(C) invalidation** by something clearing
  the cache. `doc/optimize_perf/repro_prefill_instrumented.py` was written to separate these
  — it logs `num_program_cache_entries()` per request plus per-`all_reduce` timing and
  whether each call grew the cache — but it has not produced a curve yet: both attempts hit
  trigger 2 above.
- Whether the 8–12 s is ~90 drains of ~100 ms or one long stall. The instrumented harness
  answers this directly once it runs.
- The fix direction, if the cache-miss story confirms: make the `all_reduce`
  workload/semaphores **persistent across requests** rather than rebuilt. Scope warning —
  `ccl_allreduce` and `CCLManager` live in `models/demos/gemma4/tt/ccl.py`, which is
  **shared and off-limits**; use the copy-into-DG pattern. That file already carries a
  `topology` parameter `CCLManager` accepts and never reads, so read it fully before copying.
