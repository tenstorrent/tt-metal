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
| P5 oracle `--skip*/--noreduce/--nsbcontig` ablations | **BLOCKED** — retained C++ oracle hangs on this firmware (see below) |
| P6 extended oracle profiler (role map + phase timing) | **BLOCKED** (same); COREMAP logging landed, inert until oracle runs |
| P7 K-depth via `--in0direct/--mshard` (oracle modes) | **BLOCKED** for oracle modes; op-side deep-kb sweep is doable |
| P8/P9 decision + gated change + final report | pending the above / orchestrator decision |

## Oracle infrastructure blocker (isolated)
The retained oracle `test_regime_a_mm --unified` **hangs during execution on every geometry** — Mt=1 and
Mt=8, Sm=1 and Sm>1 — on the current board (BH p150b, firmware bundle 19.5.0, KMD 2.4.1). Each run reaches
the config-setup log line then never completes (timeout 124). Isolated to NOT be caused by this branch's
changes: reverting the COREMAP edit to HEAD and rebuilding still hangs on Mt=1 (a shape the prototype
historically ran at 94%). The productized op's (newer) kernels run fine — 4000+ configs swept — so this is
the old prototype-kernel path vs current firmware, not a device fault. Secondary hazard observed & handled:
a SIGTERM'd oracle leaves a live process holding the UMD `CHIP_IN_USE_3_PCIe` lock, which wedges the next
device init ("failed to initialize FW"); recovery = `pkill -9 -x test_regime_a_mm` (NOT `-f` — that matches
the caller's own command line) + `rm -f /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_3_PCIe` + `tt-smi -r`.

## Results (op, reliable)

### Ideal DRAM floor vs achieved (all Mt=8)
| shape | N | ideal@512 us | best op us | med/ideal | %512 | best cfg (Ns,Pk,Sm,kb,nsb) | class |
|---|---|---|---|---|---|---|---|
| 256x2048x1024 | 1024 | 11.3 | 30.1 | 2.67x | 37% | (1,4,2,2,2) | KxM |
| 256x6144x768 | 768 | 25.3 | 54.3 | 2.14x | 47% | (1,12,1,2,1) | pure-K |
| 256x6144x2304 | 2304 | 63.7 | 94.1 | 1.48x | 68% | (1,12,1,2,1) | pure-K |
| 256x6144x4608 | 4608 | 121.3 | 154.6 | 1.27x | 78% | (1,12,1,2,1) | pure-K |

**Efficiency rises monotonically with N (37→47→68→78%).** The Mt=8 shortfall is a SMALL-N fixed overhead,
amortized as sustained DRAM work (N) grows — NOT an Mt-dependent cost.

### Fixed-overhead counterfactual — M-scaling (op)
| K,N | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|
| small-N (2048,1024) us | 14.6 | 16.8 | 21.6 | 30.1 |
| wide-N (6144,4608) us | 119.0 | 122.8 | 131.9 | 154.6 |

Wide-N is ~bandwidth-bound (8x the M adds ~30%); small-N has a large fixed floor at M=32 (14.6us vs an
11.3us ideal for M=256) with only modest M growth. => fixed overhead dominates the small-N shapes.

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
- **Fixed overhead:** small-N M=32 floor ~14.6us (vs 11.3us ideal at M=256); N-scaling 37→78%.
- **Best current config:** per shape table above; picker already selects these (auto==best within noise).
- **Demonstrated improvement:** 256x2048x1024 picker fix (+15%, shipped earlier this branch: N-split→M-split).
- **Remaining hypothesis:** small-N shapes are fixed-overhead + reduction-chain bound (NOT DRAM-BW bound).
  A tiny-/narrow-N kernel path (lower per-invocation setup, shallower/tree reduction) is the candidate; a
  wide-N kernel is already ~78-97%. **Next experiment (needs working oracle):** the `--skip*/--noreduce`
  counterfactuals + compute-only floor on 256x2048x1024 and 256x6144x768 to confirm the split between fixed
  setup and reduction, before any kernel change (port to the op only after >=8-10% stable oracle gain on 2
  Mt=8 shapes).
