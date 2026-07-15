# Mt=8 Regime-A characterization — findings & status

System under test: the productized `ttnn.experimental.regime_a_matmul` op (the trustworthy SUT).
All op numbers are median steady-state kernel time (min+spread retained), % of 512 GB/s.

## Status by part
| part | status |
|---|---|
| P1 harness timeout+cache cleanup + offline tests | **DONE** (committed `120ef24a951`; 4/4 offline tests) |
| P2 Mt=8 characterization + M-scaling | **DONE** (`0ebee33a82d`) |
| P3 comparable baselines (exhaustive practical sweeps) | **DONE** (`0ebee33a82d`) |
| P4 same-core factorization | **DONE** (`0ebee33a82d`) |
| P5 oracle `--skip*/--noreduce/--nsbcontig` ablations | **BLOCKED** — retained oracle hangs (cause not yet isolated: LLK/build vs firmware; see below) |
| P6 extended oracle profiler (role map + phase timing) | **BLOCKED** (same); COREMAP logging landed, inert until oracle runs |
| P7 K-depth via `--in0direct/--mshard` (oracle modes) | **BLOCKED** for oracle modes; op-side deep-kb sweep is doable |
| P8/P9 decision + gated change + final report | pending the above / orchestrator decision |

## Oracle blocker — NOT yet fully isolated (firmware not confirmed as cause)
The retained oracle `test_regime_a_mm --unified` (rebuilt in `build_Release`; `build/` is a symlink to it)
**hangs during execution on every geometry** — Mt=1 and Mt=8, Sm=1 and Sm>1. Each run reaches the
config-setup log line then never completes (timeout 124). What IS isolated:
- **Not oracle source drift.** `git log d4c8df34b19..HEAD -- .../regime_a_mm/` shows NO commit touched the
  oracle or its kernels except this branch's inert COREMAP edit; reverting COREMAP to HEAD + `ninja
  test_regime_a_mm` still hangs. So d4c8df34b19's oracle source == what I tested.
- **Not corrupt metal libs.** The productized op runs 4000+ configs fine against the SAME build_Release
  libs, so `libtt_metal`/`_ttnncpp` are healthy.
Still UNCONTROLLED (so firmware is NOT confirmed): `tt_metal/third_party/tt_llk` is **untracked / unpinned**
(currently `2d79fa0a`); the frozen parity binary was overwritten, so the exact original build artifact is
gone; and I did not do a full clean-from-scratch build. Whether the hang is LLK drift, a full-clean-build
difference, or firmware is unresolved.
Prescribed isolation exercise (to run before any kernel decision): (1) test the exact frozen binary if
recoverable; (2) rebuild from `d4c8df34b19` in a clean worktree with the original build config; (3) compare
vs the current-source oracle; (4) if the frozen oracle works, run the prepared ablations; (5) if not,
implement **test-only ablations around the current PRODUCT kernels** (not the prototype) rather than drawing
causal conclusions from whole-op timing.
Secondary hazard observed & handled: a SIGTERM'd oracle leaves a live process holding the UMD
`CHIP_IN_USE_3_PCIe` lock, wedging the next device init ("failed to initialize FW"); recovery =
`pkill -9 -x test_regime_a_mm` (NOT `-f` — that self-matches the caller's command line) + `rm -f
/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_3_PCIe` + `tt-smi -r`.

## Results (op, reliable)

### Ideal DRAM floor vs achieved (all Mt=8)
| shape | N | ideal@512 us | best op us | med/ideal | %512 | best cfg (Ns,Pk,Sm,kb,nsb) | class |
|---|---|---|---|---|---|---|---|
| 256x2048x1024 | 1024 | 11.3 | 30.1 | 2.67x | 37% | (1,4,2,2,2) | KxM |
| 256x6144x768 | 768 | 25.3 | 54.3 | 2.14x | 47% | (1,12,1,2,1) | pure-K |
| 256x6144x2304 | 2304 | 63.7 | 94.1 | 1.48x | 68% | (1,12,1,2,1) | pure-K |
| 256x6144x4608 | 4608 | 121.3 | 154.6 | 1.27x | 78% | (1,12,1,2,1) | pure-K |

**Efficiency correlates with total DRAM work (ideal runtime), NOT with N.** It is NOT monotonic in N:
N=768 reaches 47% while N=1024 reaches 37% — but K differs (6144 vs 2048), so N=768 has the larger ideal
runtime (25.3us vs 11.3us). Ordered by ideal runtime the efficiency is monotonic (11.3->37%, 25.3->47%,
63.7->68%, 121.3->78%): a larger in1/total workload amortizes the excess over the floor.

### M-scaling counterfactual (op) — the excess over the floor is M-DEPENDENT
| K,N | metric | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|
| small-N (2048,1024) | measured us | 14.6 | 16.8 | 21.6 | 30.1 |
| small-N (2048,1024) | **excess over ideal us** | **6.0** | 7.8 | 11.9 | **18.8** |
| wide-N (6144,4608) | measured us | 119.0 | 122.8 | 131.9 | 154.6 |
| wide-N (6144,4608) | **excess over ideal us** | **7.1** | 9.5 | 15.9 | **33.3** |

The excess over the DRAM floor is NOT fixed and NOT Mt-independent: it grows with M (small-N 6.0->18.8us,
wide-N 7.1->33.3us across M=32->256). **Supported conclusion: there is an M-dependent cost that is
increasingly amortized as the in1 workload (and total DRAM work) grows** — this is why % efficiency rises
with ideal runtime even though the absolute excess also rises. It is NOT a purely tiny-shape fixed overhead.

### Reduction-cost counterfactual — factorization at fixed cores (op; the `--noreduce` analog)
256x2048x1024 (shallow K, Kt=64):
- 64 cores: pure-K (1,8,1,1,1) **33.9us** vs K×M (1,4,2,2,2) **30.1us**  → trading split-K depth for M-parallelism saves ~12%.
- 96 cores: pure-K Pk12 (1,12,1,1,1) **53.1us** vs K×M (1,4,3,2,4) **30.3us** → deep split-K reduction chain *regresses* by ~75%.

pure-K scaling on this shape: Pk8 33.9us → Pk12 **53.1us** (worse). The split-K reduction chain cost grows
with depth once the per-slice K work (Ktl) is small. For deep-K shapes (Kt=192) Ktl stays healthy (=16 even
at Pk12), so pure-K Pk12 wins there. => **the winning factorization is set by K-slice depth, not Mt.**

## Report skeleton (P9) — filled where op data allows
- **Ideal DRAM floor:** table above (logical bytes / 512 GB/s).
- **Compute-only floor:** requires the oracle `--skipin0 --skipin1` ablation → BLOCKED. Op lower bound: the
  best achieved (e.g. 30.1us on 256x2048x1024) is an upper bound on the compute-only floor.
- **Delivery/reduction overhead:** op factorization shows split-K reduction is the dominant *variable* cost
  for shallow-K small-N (deep pure-K +75%); direct `--skipfwd/--noreduce` attribution BLOCKED.
- **M-dependent excess:** grows with M at fixed K,N (small-N 6.0->18.8us, wide-N 7.1->33.3us; table above);
  increasingly amortized as the in1/total workload grows. NOT a purely fixed per-invocation cost.
- **Best current config:** per shape table above; picker already selects these (auto==best within noise).
- **Demonstrated improvement:** 256x2048x1024 picker fix (+15%, shipped earlier this branch: N-split→M-split).
- **Remaining hypothesis (evidence-limited):** the excess is an M-dependent pipeline cost + a split-K
  reduction contribution (clear on shallow-K shapes: deep pure-K +75%). The **exact split among compute,
  in0 delivery, reduction, and synchronization is UNMEASURED** — whole-op timing cannot separate them.
  This does NOT yet point specifically to a tiny-shape fixed-overhead problem, so no kernel redesign is
  justified on current evidence. **Next experiment:** per-stage ablations (see below) on 256x2048x1024 and
  256x6144x768 to attribute the excess, then port to the op only after >=8-10% stable gain on 2 Mt=8 shapes.
