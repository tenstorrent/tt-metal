# Qwen3.5-2B sequence-parallel prefill on QuietBox 2 (4x Blackhole p300c)

Prefill splits the 4096-token prompt into four 1024-token spans, one per die, and runs a
layer-wise wavefront: die d+1 receives the GDN state and the K/V prefix from die d over
MeshSockets. Decode stays TP=4. Measured TTFT at ISL 4096: **63 ms** (68 ms including logits
readback), vs 106.6 ms for TP=4 chunked prefill. Logits PCC vs TP=4 prefill 0.9994, argmax equal.

## Reproduce

```bash
cd tt-metal && source python_env/bin/activate
export PYTHONPATH=$PWD LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 HF_MODEL=Qwen/Qwen3.5-2B
unset TT_METAL_CACHE
T=models/demos/blackhole/qwen36/tests

# TTFT: prints wavefront mean/min, total incl. readback, per-die finish times
timeout 900 pytest "$T/test_sp_prefill.py::test_sp_prefill_traced_ttft" -q -s

# Correctness: PCC vs TP=4 prefill and argmax match
timeout 900 pytest "$T/test_sp_prefill.py::test_sp_prefill_matches_tp4[4096]" -q -s
```

Expected: wavefront ~63 ms, total ~68 ms, per-die finish ~56 / 58 / 60 / 63 ms; PCC ~0.9994.

Optional, LoFi on the in/out projections (~60.6 / 65.6 ms, PCC 0.9988):

```bash
QWEN36_SP_PROJ_LOFI=1 timeout 900 pytest "$T/test_sp_prefill.py::test_sp_prefill_traced_ttft" -q -s
```

Rules: run each test in its own pytest process (socket teardown wedges a later mesh open in the
same process), keep the `timeout`, and keep `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` on this
box (issue 55957). Code: `tt/sp_prefill.py` (orchestrator), `tt/sp_handoff.py` (SP -> TP=4 decode
cache handoff), `tests/test_sp_prefill.py`.

## Blackhole Galaxy (32 chips): SP x TP

`SPPrefill(tp=N)` gives each span an N-die tensor-parallel group, so SP splits the prompt
across groups and TP splits each layer within a group. Measured on Qwen3.5-2B, ISL 4096,
warm traced replay:

| config | chips | span | d0 (span work) | staircase | wavefront |
|---|---|---|---|---|---|
| TP=4 alone | 4 | 4096 | - | - | ~106 ms |
| SP=4, tp=1 | 4 | 1024 | 55.83 ms | 6.9 ms | 62.75 ms |
| SP=8, tp=1 | 8 | 512 | 40.06 ms | 11.1 ms | 51.15 ms |
| SP=16 x TP=2 | 32 | 256 | 30.88 ms | 20.9 ms | 51.76 ms |
| SP=8 x TP=4 | 32 | 512 | 33.86 ms | 9.7 ms | 43.58 ms |
| **SP=4 x TP=8** | **32** | **1024** | **38.51 ms** | **4.27 ms** | **42.79 ms** |

Best (2026-09-23): `SP_DIES=4 SP_TP=8 QWEN36_NO_AGMM=1` on FABRIC_1D, everything else default -- **37.55 ms** wavefront, **38.25 ms true TTFT** (on-device argmax; the full-logits readback that used to cost 26 ms is now outside the timed path), PCC 0.9993 vs the TP=4 oracle, argmax equal. Default-on flags: `QWEN36_ATUPE_OPTS` (Aniruddha's round-4 GDN/SDPA/L1 work) and `QWEN36_REPL_RESIDUAL` (replicated residual). Set either to 0 to back it out. Overheads NOT in the wavefront -- and the ~70 MB host handoff to decode -- are itemised in `PERF_ROOFLINE.md`.
argmax equal. `QWEN36_CCL_BF8=1` adds ~0.5% (42.59 ms) for PCC 0.9990 -- opt-in.

There is a real optimum, not a monotone trend: each extra span costs a ~1.4 ms hop, and past
SP=8 the staircase grows faster than the span work shrinks. SP=16 x TP=2 has the LOWEST span
work measured and is still the slowest of the 32-chip configurations.

### Things that do not work, and why

* **AGMM is a pessimisation here.** 1D + AGMM on is 44.19 ms vs 42.79 off. It was also
  e2e-neutral at plain TP=8. Keep `QWEN36_NO_AGMM=1`.
* **FABRIC_1D routes ONE axis on this mesh.** A same-column group hop succeeds; a same-row one
  fails with fabric.cpp:174 "Could not find any forwarding direction". SP=4 x TP=8 is unaffected
  (4 groups = 4 rows, every hop vertical). SP=8 x TP=4 needs FABRIC_2D, which in turn needs
  QWEN36_NO_AGMM=1 because all_gather_minimal_matmul_async's dm_in0_sender kernel is written
  against the linear (1D) fabric API and will not compile under 2D.
* **More ethernet links are not available.** num_links=4 errors: only 2 channels exist between
  adjacent dies, and both are already used.
* **Widening collective grids does nothing.** The AGMM grid 72 -> 90 -> 108 cores measured
  0.18 s TTFT at every width; these ops are bandwidth-bound, not core-bound.

### Where the time goes (SP=4 x TP=8, signposted eager pass, 1845 ms over 32 dies)

| category | share | note |
|---|---|---|
| SP sockets (Send/Recv) | 42.6% | 2 cores (= 2 eth links), but OVERLAPPED -- staircase is only 4.27 ms of 42.8 ms wall |
| CCL (AllGather + ReduceScatter) | 27.4% | 12 cores each, at the link roofline |
| compute (Matmul + SDPA) | 13.7% | Matmul already 108/120 cores, SDPA 120/120 |
| GDN (Scan + Prep + Conv) | 7.3% | ChunkGdnScan on 8 cores is the one real occupancy gap |
| layout churn | 5.9% | 12 Slice + 6 BinaryNg per layer per die |

Excluding the overlapped sockets the critical path is roughly 48% CCL, 24% compute, 13% GDN,
10% layout. Per layer per die: 4 AllGather, 2 ReduceScatter, 5 Matmul, 12 Slice, 6 BinaryNg.
Compute kernels are already near full occupancy, so kernel fusion/vectorisation addresses ~24%
of the path at best -- the dominant on-path cost is the TP collectives inside each span, which
is also why SP=8 x TP=4 and SP=4 x TP=8 land within 2% of each other.
