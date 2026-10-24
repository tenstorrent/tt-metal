# Nemotron-3-Nano-30B-A3B — Status Q&A

Answers to the three questions, with sources. Numbers are as-measured; where a
metric is not yet measured, that is stated explicitly rather than estimated.

---

## 1. Is the model implemented end-to-end on TT-NN? Command + branch?

**Yes.** The full 52-layer NemotronH (Mamba2 + MoE hybrid, ~60 GB bf16) runs
end-to-end on TT-NN on a single Blackhole ASIC, weight-streamed (TP=1), and
passes correctness:

- Per-component PCC ≥ 0.99 (7 components).
- End-to-end greedy token-match vs the HuggingFace reference.

| | |
|---|---|
| **Branch** | `apande/nvidia_nemotron_3_nano_30b_a3b_bf16` (commit `2181607ca27`) |
| **Topology** | Single Blackhole ASIC, TP=1, weights streamed from host |
| **Demo** | `./python_env/bin/python -m models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.demo.demo_text_generation --prompt "The capital of France is" --max-new-tokens 5` |
| **Correctness pytest** | `pytest models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py` |

> Note: the `tt-hw-planner-cc-shard` branch is a **separate** sharded (multi-chip)
> experiment for the cc engine — not this single-chip model.

---

## 2. Current perf in T/S/U? Which pytest generates the perf numbers?

**We do not have a T/S/U number yet.** What is measured today is
**on-device kernel time (`device_ms`)** for the full end-to-end pipeline
(prefill + bounded 4-token decode), profiled with **Tracy** in **eager**
execution mode.

| Metric | Baseline → Optimized | Speedup | What it is |
|---|---|---|---|
| `device_ms` | 877.5 → 359.2 ms | **2.44×** | Sum of on-device kernel time, full pipeline |
| end-to-end compute (`device_ms + host_ms`) | 942.9 → 392.7 ms | 2.4× | Kernel time + host op-to-op gaps in the compute region |

- **Perf pytest:** `tests/e2e/test_perf.py` (Tracy-profiled `device_ms`; the
  on-disk variant caps layers via `TT_PERF_LAYERS` as a device-time proxy).
- **Dominant win:** eliminating layout churn (64.5% of baseline `device_ms`),
  incl. host-tilizing the 128-expert MoE weights.

**Why this is not T/S/U:** `device_ms` is summed kernel time (host dispatch and
the generation loop are deliberately excluded), measured **eager**, at batch=1
over a 4-token decode window. T/S/U is wall-clock *tokens ÷ seconds* on the
**production (trace-replay)** path. Different quantity → cannot be converted.

**To get T/S/U:** run **trace-replay** (record the dispatch sequence once,
replay without the Python host loop) at a defined batch/seq, emitting
tokens/sec/user, tokens/sec, prefill (TTFT), and per-token decode latency.
This run has **not** been done yet.

---

## 3. How does it compare to GPU perf/benchmarks?

**Not comparable yet — by quantity, not by hardware.** GPU vendors publish
**wall-clock** T/S/U, throughput, prefill/decode latency on the production path.
We currently have **eager end-to-end kernel time (`device_ms`)**. Three mismatches:

1. **Wall-clock vs summed kernel time** — `device_ms` excludes host dispatch/overhead that GPU latency includes.
2. **Eager vs trace-replay** — GPU numbers are graph/production path; ours is eager.
3. **No throughput normalization** — batch=1, 4-token decode is a profiling window, not steady-state throughput.

A like-for-like GPU comparison becomes possible **after** the trace-replay T/S/U
run in Q2.

---

### One-line summary

Validated **end-to-end single-chip** Nemotron on TT-NN with a verified **2.44×**
kernel-time optimization (877.5 → 359.2 ms `device_ms`). The
production-representative **T/S/U** and the resulting **GPU comparison** require
a **trace-replay** run that has not been done yet.
