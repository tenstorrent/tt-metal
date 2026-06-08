# Results — precompile cold vs warmup+warm, 75-test layernorm suite

Run: WH (`bgd-lab-t3003`), branch `mstaletovic/precompile-bh-ci-validate`, `nproc=8` (cgroup ceiling;
32 logical host CPUs), 75-test layernorm suite (490 kernel builds / 92 unique programs), median of 2
repeats (variance was <3%). Raw: `SUMMARY.example.txt` (regenerate with `summarize.py /tmp/lnbench`).

Numbers below are **authoritative getrusage** (whole reaped process tree) for wall/CPU/utilization; the
0.25 s `/proc` sampler is used only for time-shape, peak cores, and a *lower bound* on the compiler's
CPU share — it cannot integrate the sub-250 ms gcc swarm, so its absolute CPU-seconds undercount.

## North star — total end-to-end wall

| ccache condition | COLD (inline) | PRECOMPILE (probe+warmup+warm) | speedup |
|---|---|---|---|
| **off** (default/CI) | **1m56s** | **1m28s** (pr5 + mk3 + wu52 + wm28) | **1.33×** |
| **on / deleted** (cold compiler cache) | **2m09s** | **1m38s** (pr5 + mk3 + wu58 + wm31) | **1.32×** |
| **on / warm** (reused compiler cache) | **48.6s** | **47.4s** (pr5 + mk3 + wu22 + wm17) | **1.03×** |

## The five findings

**1. The biggest single lever is a warm ccache, not precompile.** A warm compiler cache alone takes the
cold pass from **116–129 s → 48.6 s (2.4–2.65×)** — a larger win than precompile delivers, with none of
the precompile machinery. `TT_METAL_CACHE` is still empty (all 490 kernels are still "jitted"), but each
underlying C++ compile is a ccache hit, so the inline compile nearly vanishes (cold-pass compiler CPU
drops 394 → 154 CPU-s; comp-share 77% → 32%).

**2. Precompile is a steady ~1.32× when ccache is cold/off** — by moving compile off the test's critical
path *and parallelizing it*. Cold inline compile runs at **3.4 cores (42 % of 8)**; the warmup compile
burst runs at **6.9–7.0 cores (86–87 %)**. Same compile work (~355–406 CPU-s either way), but ~7 cores
instead of 3.4 ⇒ roughly half the compile wall. This is the core mechanism the precompile system buys.

**3. When ccache is warm, precompile is a wash (1.03×).** Warm ccache already makes each compile
near-free, so there is little compile left to hide or parallelize; both paths converge to a ~48 s floor and
precompile's fixed overhead (≈8.6 s of device probes + a warmup that still re-walks the suite + a warm run
that still inline-compiles its misses) eats the margin. **ccache-warm and precompile are substitutes here,
not additive.**

**4. ccache ON but cold is ~10 % *slower* than ccache OFF** (cold 2m09s vs 1m56s; precompile 1m38s vs
1m28s). An empty ccache turns every compile into a miss *plus* the store/hash overhead. ccache only pays
once it's warm — a cold ccache is a small net tax.

**5. The precompile win is capped by meta-collect coverage on this op (81 %).** The warm run hits
**397–400 / 490 (≈81 %)** and still inline-compiles ~90 kernels. That residual is exactly why the warm run
is 17–31 s rather than near the ~28 s execution floor, and why the net speedup is 1.32× rather than higher.
(On conv2d the same mechanism reaches ~99 %; layernorm bodies do more host-side / address-dependent work
the NO_DISPATCH meta-collect can't reproduce.) **Closing this coverage gap is the highest-leverage
improvement for the precompile path on layernorm.**

## Where the cold pass spends its time (ccache off)

* total cold wall **116 s**, of which **≈89 s (77 %) is inline kernel compile** (cold − warm wall, exact);
  the remaining ~28 s is the framework + execution floor (import ttnn, open device, run 75 bodies).
* total cold CPU **394 CPU-s** at **3.4 cores** — i.e. the box is <half utilized during the cold pass:
  inline compile is only ~3-way parallel, so 4–5 of 8 cores sit idle while tests serialize behind compiles.
* the warmup re-spends that same ~355 CPU-s at **6.9 cores**, finishing the compile in ~half the wall, and
  then the warm run reuses it.

## Per-phase utilization (getrusage, ccache off)

| phase | wall | cores (of 8) | util | peak cores | peak RSS | JIT |
|---|---|---|---|---|---|---|
| COLD | 1m56s | 3.4 | 42 % | 6.3 | 1.72 GB | 490 jitted / 0 hit |
| probe_real | 5.3s | 1.6 | 20 % | 3.7 | 0.46 GB | (device open) |
| probe_mock | 3.4s | 1.3 | 16 % | 3.5 | 0.46 GB | (hw-free open) |
| **warmup** | 51.8s | **6.9** | **86 %** | 8.0 | 0.59 GB | collect 6.6s + compile 45.2s (92 progs) |
| warm | 27.6s | 3.4 | 42 % | 6.6 | 1.73 GB | **400/490 hit (81.6 %)**, 90 jitted |

The warmup is the only phase that meaningfully uses the machine (86 %); everything else (cold included)
leaves the box mostly idle. The warmup compile subphase is the nproc-parallel JIT burst (peaks at all 8
cores); the ~6.6 s collect is single-process meta shape-propagation (low CPU).

## Takeaways / recommendations

* **For a developer or CI box that keeps a warm ccache**, the precompile path is not worth its overhead on
  a suite this size — just run cold (≈48 s). Keep ccache warm (`TT_METAL_CCACHE_KERNEL_SUPPORT=1` + a
  persistent `CCACHE_DIR`); that is the cheapest 2.5× available.
* **For a cold/ephemeral environment (fresh container, no ccache)** — i.e. today's CI — precompile's
  parallel warmup gives a reliable ~1.3× and, more importantly, would scale better than inline on larger
  suites (compile is ~2× more parallel). Combining precompile *with* a persisted ccache is redundant here:
  they target the same cost.
* **Raise meta-collect coverage on layernorm (81 % → ~99 %)** to convert more of the warm run's residual
  inline compiles into hits — the single change that would most increase the precompile speedup on this op.
* **Don't enable ccache without persisting it** — a cold ccache is a ~10 % tax with no benefit.
