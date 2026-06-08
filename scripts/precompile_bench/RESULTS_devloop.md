# Results (round 2) — parallelism scaling, JIT-cache staleness, and the dev loop

Follow-on to `RESULTS.md`. Same WH box (`nproc=8`), 75-test layernorm suite (490 kernel TUs /
92 unique programs). Produced by `worker_sweep.sh` and `devloop.sh`; raw in `*.example.txt`.

## 1. How far does parallel compile scale? (`worker_sweep.sh`)

Hardware-free warmup, fresh ccache + JIT each run (full cold compile), sweeping the compile worker count:

| workers | compile_wall | cores (getrusage) | speedup vs w1 |
|---|---|---|---|
| 1 | 98.0s | 3.3 | 1.00× |
| 2 | 57.5s | 5.9 | 1.70× |
| **4** | **49.9s** | **7.0** | **1.96×** |
| 8 | 51.2s | 7.0 | 1.91× |
| 16 | 52.2s | 7.0 | 1.88× (oversubscribed) |

* Even **workers=1 runs at 3.3 cores** — each program's reader/writer/compute(×3 TRISC) kernels already
  compile in parallel *within* a program. Worker count stacks concurrency *across* programs on top.
* Parallel compile **saturates at ~7 cores around 4 workers (~1.95×)**; this 8-core box keeps ~1 core for
  the driver. Past 4 workers there's no gain and oversubscription (16) slightly regresses.
* Implication: the default `--precompile-workers nproc` is at/above the knee — fine, but 4 would do here.
  The whole precompile speedup is bounded by this ~2× compile-parallelism ceiling on an 8-core host;
  a 32-core CI runner would have far more headroom (the cold inline run can't use it; the warmup can).

## 2. Does a warm JIT cache pick up a kernel edit, or stale-hit? (`devloop.sh` Part 1)

| run | wall | JIT telemetry |
|---|---|---|
| fresh JIT, first run | 119.5s | 0/490 hits (all compiled) |
| rerun, **no edit** | **10.0s** | **490/490 hits (100%)** |
| rerun, **after editing `layernorm.cpp`** | 37.5s | 418/490 hits (85.3%) → **72 recompiled** |

* **Not stale.** A warm `TT_METAL_CACHE` correctly recompiles exactly the edited kernel's TUs (72) and
  serves the other 418 from cache. (The per-kernel cache key reflects kernel content, so the edit is
  picked up — no manual cache clear needed.)
* The **true execution floor is ~10s** (import + device + run 75 bodies, zero compile). Note this is far
  below precompile's 27s warm run in `RESULTS.md` — that gap is precompile's 81% meta-collect coverage
  (it cold-compiles ~90 kernels in the warm run); a *fully* warm JIT cache has none of that.

## 3. Is precompile worth it in the dev loop? (`devloop.sh` Part 2 + supplemental)

After editing the op you're developing, three ways to rerun the suite (measured):

| edit size (TUs recompiled) | just rerun¹ (warm JIT, incremental) | precompile² (parallel warmup+warm) | cold³ (fresh JIT, inline) |
|---|---|---|---|
| **one compute kernel** (`layernorm.cpp`, ~72 TU) | **37.5s** ✅ | 51.6s | 85.5s |
| **whole compute path** (8 files, ~213 TU) | 101.7s | **53.9s** ✅ | 109.0s |

¹ keep the warm JIT cache, just rerun — only the edited kernel recompiles, inline at ~3.4 cores.
² fresh JIT + warm ccache: `probe 8.6s + parallel warmup + warm run` (warmup compiles at ~7 cores).
³ fresh JIT + warm ccache, inline compile.

**This is the crossover, and it's the real answer to "does precompile help during development":**

* **Small edit (one kernel):** *just rerunning wins* (37.5s). The warm JIT cache already recompiles only
  your kernel; precompile (51.6s) throws the warm cache away and pays ~8.6s probe + a full re-warm, so it
  *loses* — even though it beats a cold-from-scratch run (1.66×).
* **Big edit (whole compute path / a shared header):** *precompile wins decisively* (53.9s vs 101.7s,
  **1.9×**). When the edit invalidates a large fraction of the suite, the incremental rerun has to compile
  ~213 TUs inline at 3.4 cores (101.7s); precompile parallelizes that same work to ~7 cores (53.9s).

So the original intuition — "ccache can't help the kernel I'm editing, but precompile would" — is **right
for large edits and wrong for small ones**. The deciding factor is how many TUs your edit invalidates:
a leaf-kernel tweak recompiles too little to beat a plain warm-JIT rerun; editing the whole compute path
(or a shared `kernel_lib`/llk header, which busts the entire suite) is squarely precompile's win.

## 4. Why warm reuse capped at 81%, and the fix (81% → 99.4%)

The warm run consistently cold-compiled ~90/490 kernels (81% reuse) — much worse than conv2d's ~99%.
Root-caused by snapshotting the JIT cache after the warmup and diffing what the warm run newly
compiled, plus an env-gated per-body diagnostic (`UP_FRONT_LOG_SWALLOWED=1`) in the collect plugin:

* The 53 missed kernels were dominated (30, ~57%) by **`fill_pad` reader/writer/compute** — input
  padding kernels — plus a tail of compute/reader variants and the 3 `cq_*` dispatch kernels.
* The per-body log showed **10 bodies stashed ZERO ops** — all `test_layer_norm_with_padding[...]`
  — each throwing `RuntimeError: Tensor.item() cannot be called on meta tensors` on the test's first
  line: `non_zero_columns = torch.randint(1, w+1, (1,)).item()`. Under `UP_FRONT_META_COLLECT` the
  tensor is on torch's meta device and `.item()` is unsupported, so the body aborted **before** any
  ttnn op was stashed → those tests (and their `fill_pad` + `layer_norm` programs) were never collected.
  That one `.item()` call accounted for the bulk of the 19% gap (the suite has many non-tile-aligned
  widths → heavy padding; conv2d's aligned shapes barely hit `fill_pad`, hence its ~99%).

**Fix** (`up_front_collect_plugin.py`, `_meta_host_ops`): make `.item()`/`.tolist()` return a
deterministic stand-in on meta tensors. The extracted scalar only feeds input *data*, never the ttnn
program shape/config that keys the cache, so it's safe — and the content-hashed cache means any
mis-collected variant simply misses and recompiles (never a wrong result).

| | unique programs collected | warm-run reuse | tests passing |
|---|---|---|---|
| before | 92 | 397/490 (81.0%) | 75/75 |
| **after** | **112** | **487/490 (99.4%)** | **75/75** |

The remaining 3 misses are the `cq_prefetch`/`cq_dispatch*` kernels — the hardware-free (slow-dispatch)
warmup has no command queue, so the fast-dispatch CQ kernels are an irreducible ~3-kernel floor.
**Implication:** with ~99% coverage the warm run drops toward the ~10s execution floor (only 3 inline
compiles instead of ~90), which improves precompile's standing in every comparison above — the
north-star and dev-loop numbers in this doc were measured at the old 81% and are now conservative.

### Practical guidance for the dev loop
* Iterating on one kernel file: **don't bother with precompile** — keep your JIT cache warm and just rerun
  (~37s here); the cache recompiles only what you changed.
* Reworking a whole op's compute path, or touching a shared header: **use `--precompile`** — it parallelizes
  the large recompile (~1.9× here) instead of grinding through it inline.
* Either way, ccache only ever helps the files you *didn't* edit (see `RESULTS.md` §dev / the ccache probe).
