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
