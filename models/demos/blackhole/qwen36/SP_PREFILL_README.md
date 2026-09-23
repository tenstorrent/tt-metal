# Qwen3.5-2B sequence-parallel prefill on QuietBox 2 (4x Blackhole p300c)

Prefill splits the 4096-token prompt into four 1024-token spans, one per die, and runs a
layer-wise wavefront: die d+1 receives the GDN state and the K/V prefix from die d over
MeshSockets. Decode stays TP=4. Measured TTFT at ISL 4096: **~50 ms** wavefront, **~50 ms**
to first token (on-device argmax, no 16 MB logits readback), vs 106.6 ms for TP=4 chunked
prefill. PCC vs TP=4 prefill ~0.9995, argmax equal. Per-die finish ~44 / 45 / 47 / 49.5 ms.

Numbers above are the 2026-09-22 state, uncommitted on top of commit `d8a6857013c` (working
tree only). See `SP_PREFILL_HANDOFF.md` section 4b for the per-change breakdown.

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

Expected: wavefront ~50 ms, total ~50 ms (on-device argmax, no separate readback total),
per-die finish ~44 / 45 / 47 / 49.5 ms; PCC ~0.9995.

Optional, LoFi on the in/out projections (adds ~-2.5 ms on top of the round-4 numbers above,
PCC 0.9994 -> 0.9988 measured at round 3; not re-measured on round 4):

```bash
QWEN36_SP_PROJ_LOFI=1 timeout 900 pytest "$T/test_sp_prefill.py::test_sp_prefill_traced_ttft" -q -s
```

Opt-out flags (round-4 defaults, 2026-09-22, uncommitted): `QWEN36_SP_SDPA_LEGACY=1` reverts
the SDPA program config to q_chunk=k_chunk=64; `QWEN36_SP_KDA_CONV=0` reverts the fused KDA
conv to conv1d + separate SiLU; `QWEN36_SP_L1_RES=0` reverts the MLP/GDN/KDA L1-residency
defaults; `QWEN36_SP_QKVZAB_DRAM=1` reverts qkvzab-in-L1 (round 2). Opt-in:
`QWEN36_SP_PROJ_LOFI=1` (above); `QWEN36_SP_SOCKET_FIFO_MB=N` (DRAM-backed async socket FIFO,
measured negative for TTFT, kept as a knob). All default to the current on-by-default behavior
when unset. See `SP_PREFILL_HANDOFF.md` section 4b for measurements per flag.

Rules: run each test in its own pytest process (socket teardown wedges a later mesh open in the
same process), keep the `timeout`, and keep `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` on this
box (issue 55957). Code: `tt/sp_prefill.py` (orchestrator), `tt/sp_handoff.py` (SP -> TP=4 decode
cache handoff), `tests/test_sp_prefill.py`.

---

## Where the 50 ms TTFT goes (recorded 2026-09-23)

Source: 4-die Tracy trace of the final round-4 code, ISL 4096, die 3 (the die that finishes
last), last traced replay. CSV: `~/atupe/sp_prefill_reports/2026-09-23_e2e_sp_tp_4k_osl8/`
(`b_ops_perf_results_2026_09_23_00_32_21.csv`); per-layer split script
`scripts/analyze_sp4_per_die.py` (usage: `python scripts/analyze_sp4_per_die.py <ops_perf_results.csv>`). Wall-clock time to first token in the
same run: 49.75 ms.

### Timeline view (die 3)

| Block | ms | Share | What it is |
|---|---:|---:|---|
| 18 GDN layers | 32.2 | 65% | 29.2 compute + 3.0 receiving state from die 2 |
| 6 attention layers | 10.3 | 21% | 5.2 attention over 4096 keys (SDPA 4.2) + 4.2 projections/MLP + 0.9 receiving K/V |
| Pipeline fill | 4.0 | 8% | Die 3 idles until dies 0, 1, 2 have each run layer 0 and passed the state down |
| Tail | 1.6 | 3% | Last-token select, final norm, LM head over the 248k vocabulary (1.5), argmax |
| Dispatch gaps and token readback | 1.6 | 3% | ~1 us between each of ~940 ops, plus reading one token back to the host |

### Work-type view (die 3, same 50 ms)

| Op type | ms | Share |
|---|---:|---:|
| Matmuls (projections, MLP, LM head) | 13.1 | 26% |
| Communication: fill wait + per-layer receives | 7.9 | 16% |
| GDN kernel (prep + scan) | 7.8 | 16% |
| Elementwise (SiLU, gating, residual adds) | 4.3 | 9% |
| SDPA | 4.2 | 8% |
| Fused conv1d (KDA op) | 3.8 | 8% |
| Layout ops (untilize, slices, head reshapes) | 3.6 | 7% |
| Norms | 2.3 | 5% |
| Gaps, readback, misc small ops | 3.0 | 6% |

### Per-layer numbers vs TTFT

Per-layer cost on die 3, including comm: GDN 1.79 ms, attention 1.72 ms. The layers alone
do not add up to the TTFT because two one-time costs sit outside them:

| | ms |
|---|---:|
| 18 GDN x 1.79 + 6 FA x 1.72 | 42.5 |
| Pipeline fill before layer 0 | 4.0 |
| Tail: LM head, norm, argmax | 1.6 |
| Dispatch gaps and token readback | 1.6 |
| Total | 49.7 |

Per-die measured means (compute-only / comm, ms per layer): die 0 GDN 1.616 / 0.159, FA
1.041 / 0.751 (send waits); die 3 GDN 1.622 / 0.166, FA 1.569 / 0.153. SDPA per die 0.146 /
0.330 / 0.514 / 0.699 ms (1024 to 4096 keys). Fill wait on die 3: 3.96 ms.
