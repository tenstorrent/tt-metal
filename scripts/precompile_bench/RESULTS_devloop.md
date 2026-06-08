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

### Practical guidance for the dev loop
* Iterating on one kernel file: **don't bother with precompile** — keep your JIT cache warm and just rerun
  (~37s here); the cache recompiles only what you changed.
* Reworking a whole op's compute path, or touching a shared header: **use `--precompile`** — it parallelizes
  the large recompile (~1.9× here) instead of grinding through it inline.
* Either way, ccache only ever helps the files you *didn't* edit (see `RESULTS.md` §dev / the ccache probe).
