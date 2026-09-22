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
