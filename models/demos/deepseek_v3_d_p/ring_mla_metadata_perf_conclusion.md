# ring_mla metadata-path perf: is the regression genuine, and can it be <1%?

## TL;DR
The metadata-path regression vs the scalar path is **genuine but a small FIXED per-call cost** (the
on-device metadata read + derivation that replaces host-passed scalars). It does **not scale with KV
length**, so its *relative* size shrinks as the call grows:

- **At realistic prefill KV sizes it is already <1%** (steady-state, tiny-dim micro-bench):
  kv5120 (= the real prefill chunk) → **+0.72%**; kv16384 → +0.49%; kv49152 → ~0% (noise).
- It is only >1% at **tiny KV** (kv256 +3.7%, kv1024 +2.7%) — sizes that do **not occur** in real
  prefill (the smallest real call processes a 5120-token chunk → logical_n ≥ 5120). The headline
  +2–9% in `ring_mla_eq_perf.log` was measured at kv64–320 (the equivalence tests' tiny rotation
  inputs) and is a small-KV artifact, not the production regime.
- **Production (real Kimi dims, 8×4, production-50k+5k, worst-device mean over 11 growing chunks):**
  line **+0.9–1.2%**, ring **+0.4–0.9%** (the metric has ~±0.4% run-to-run noise; worst-device is the
  dev-16 dispatch outlier — the median device is comfortably lower). So production hovers around /
  just under 1%, driven upward only by the small first chunk.

## How it was determined (genuine vs noise)
`tests/perf/test_ring_mla_microperf.py` loops ring_mla N=30×/mode (scalar vs metadata) on identical
inputs, signpost-bracketed; the driver takes the per-device median (drops 6 warmup) over 3 runs.
Per-device-pairwise Δ with tiny std (±0.1–0.4%) ⇒ genuine, not noise. Scaling sweep
(kv 256→49152) shows the Δ shrinks with KV ⇒ FIXED cost, not a KV-scaling derivation loop.

## What it is NOT (ruled out)
- **Not the on-device derivation loop.** `build_ring_work_masks_device` is `O(ring_size ×
  num_local_k_chunks)`, but the overhead does not grow with KV (num_local_k_chunks ∝ KV) — at
  kv49152 (num_local_k_chunks ~190) the Δ is ~0. The integer-math loop is effectively free. (A
  binary-search rewrite is possible but pointless — the loop isn't the cost.)
- **Not the SDPA-reader metadata read.** Hoisting it before the op-signal wait (so the NoC fetch
  overlaps the all-gather-completion wait) was a **no-op** — that read already overlaps the wait.
- **Not DRAM-read latency / bank contention (at production).** Placing the metadata tensor in **L1**
  (`META_L1=1`) helped a little at tiny KV (kv256 +3.7%→+2.8%, ~0.7µs) but made **no reliable
  difference at production** (line +0.9% DRAM vs +1.2% L1 — within noise). Not adopted.

## What it IS
A small fixed per-call cost (~3µs at the tiny micro-bench dims; ~30–40µs at full Kimi dims, where many
more cores each read the replicated metadata) from the trace-safe metadata reads in the all-gather
reader/writer (which gate the gather and have no setup window to overlap) + field extraction. This is
inherent to the design (move per-chunk scalars on-device so one captured trace replays across chunks)
and is dwarfed by the host-dispatch op2op tax the resulting trace eliminates (~200ms/chunk).

## Recommendation
The new path already meets <1% per-call at realistic prefill KV (≥ the 5120 chunk size). The only place
it touches ~1% is the worst-device MEAN over a full prefill, weighted up by the tiny first chunk and a
single dispatch-outlier device. Cheap optimizations (SDPA-read hoist, L1 metadata, binary-search) do
**not** reliably move it. Getting the worst-device mean *solidly* under 1% would require reducing the
multi-core metadata reads in the all-gather (e.g. read-once-and-share across the reader/writer on a
core, or a multicast of the 16B) — deep, hang-risky kernel surgery for a sub-0.5% gain on a ±0.4%-noisy
metric. Assessed as low ROI; left as a documented follow-up.

Repro: `PYTHONPATH=. python_env/bin/python models/demos/deepseek_v3_d_p/tests/perf/ring_mla_microperf_driver.py 3`
(per-call steady-state) and `.../ring_mla_metadata_perf.py` (production worst-device). `META_L1=1` toggles
the L1 metadata experiment in the micro-bench.
