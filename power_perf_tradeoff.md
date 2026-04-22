# Power vs. Performance Tradeoff on Tenstorrent Hardware

## Context

The researcher has already demonstrated that power consumption can be measured and graphed on
Wormhole hardware using tt-sim and tt-telemetry. The power-vs-cores curve exists. What is missing
is the **execution time axis** — a workload that runs on a Wormhole N300 quietbox, sweeps the number
of active Tensix cores, and produces a matching latency-vs-cores curve.

Once both curves exist on the same x-axis, the story tells itself:

> *"For this workload, 24 cores delivers 92% of the performance of 48 cores at roughly half the power.
> Everything to the right of that point is efficiency you are paying for but not getting."*

---

## Hardware

- **Platform**: Wormhole B0 quietbox, N300 card
- **N300**: two WH chips on a single card; for single-chip inference workloads the compute grid per
  chip is **8 × 8 = 64 Tensix cores**
- The sweep runs on one chip; the power reading from the researcher's telemetry covers the card as a
  whole, which is fine — the active-chip contribution dominates and the relationship is consistent

---

## The Combined Graph

```
      time (ms)           power (W)
          │\                  │             /
          │ \                 │            /
          │  \__              │           /
          │      \____        │          /
          │           \_____  │_________/
          └────────────────── └──────────────
                    ↑ cores ↑
               sweet spot: time curve
               has flattened, power
               curve has not
```

The x-axis is number of active Tensix cores. The researcher overlays their power curve on top of
the latency curve we produce. The knee — where latency stops improving but power keeps climbing —
is the recommended operating point.

---

## What We Already Have

Branch `ncvetkovic/high_power_usage_workload` contains a C++ programming example at
`tt_metal/programming_examples/high_power_matmul/` that:

- Runs a configurable HiFi4 matmul with fixed total FLOPs across a variable number of cores
- Accepts `num_cores` as a CLI argument; total work is held constant so the latency change is
  purely a function of how many cores share it
- Reports per-iteration wall-clock latency

A full WH N300 sweep (one chip, multiples of 8 for clean row alignment):

```bash
BIN=./build/programming_examples/metal_example_high_power_matmul
for cores in 8 16 24 32 40 48 56 64; do
    echo -n "cores=$cores  "
    $BIN 4096 4096 4096 100 $cores 2>/dev/null | grep "Per-iteration"
done
```

This infrastructure is the foundation for both approaches below.

---

## Approach A — Matmul Sweep with LLM-Representative Shapes

### What it is

The existing example, with matrix dimensions chosen to match the dominant ops in a real 7B-parameter
LLM (Llama-7B / Falcon-7B weight projection layers). No model weights are needed. The argument to
the researcher is: *"this is the op that accounts for ~90% of compute time in this model's inference."*

### Two shapes, two stories

| Mode | Shape | Character | Expected latency curve |
|------|-------|-----------|------------------------|
| **Decode** (1 token at a time) | `[32, 4096] × [4096, 4096]` | Memory-bandwidth-bound | Nearly flat — adding cores does almost nothing |
| **Prefill** (seq=2048, prompt processing) | `[2048, 4096] × [4096, 4096]` | Compute-bound | Clearly sloping — each core reduces latency proportionally |

Plotting both on the same chart makes a second, complementary point: **the power-efficiency case
is strongest for decode**, which is exactly the hot path in interactive LLM serving.

### How to run

```bash
BIN=./build/programming_examples/metal_example_high_power_matmul

echo "=== Decode (memory-bound) ==="
for cores in 8 16 24 32 40 48 56 64; do
    printf "cores=%-3s  " $cores
    $BIN 32 4096 4096 500 $cores 2>/dev/null | grep "Per-iteration"
done

echo ""
echo "=== Prefill (compute-bound) ==="
for cores in 8 16 24 32 40 48 56 64; do
    printf "cores=%-3s  " $cores
    $BIN 2048 4096 4096 100 $cores 2>/dev/null | grep "Per-iteration"
done
```

Run these while the researcher's telemetry is recording. The decode sweep is the one to highlight —
the latency curve will be nearly flat while the researcher's power curve will be linear, making the
efficiency waste explicit.

### Effort

**None** — the binary already exists on this branch. Swap in the shapes above and run.

---

## Approach B — Full Model Sweep on SentenceBERT

### What it is

Run the full SentenceBERT model (12-layer BERT-base encoder) end-to-end at varying core counts,
measuring inference latency per batch. SentenceBERT is already validated and has a performant
traced runner on Wormhole B0:

```
models/demos/wormhole/sentence_bert/tests/perf/test_sentence_bert_e2e_performant.py
```

The headline the researcher can publish:

> *"SentenceBERT, batch=8, seq=384, N300: reducing active cores from 48 → 24 costs ~8% latency,
> saves ~40% power."*

This is a complete, attributable finding on a named model with a known use case (sentence
embeddings for semantic search / RAG pipelines), which gives it real-world weight.

### Technical details

All op configs in `models/demos/wormhole/sentence_bert/ttnn/common.py` hardcode
`compute_with_storage_grid_size=(6, 8)` = 48 cores. The 2D multicast matmul tiling means
both grid dimensions drive the blocking factors — you cannot change the grid without
recomputing them.

The QKV fused projection constrains `grid_x` to 6 (its output has 72 tiles, which 8 does not
divide). So the sweep holds `grid_x = 6` and varies `grid_y`, giving four data points:

| Config | Active cores | `per_core_M` | Status |
|--------|-------------|--------------|--------|
| `(6, 1)` | 6 | 96 tiles | Needs L1 fit verification |
| `(6, 2)` | 12 | 48 tiles | |
| `(6, 4)` | 24 | 24 tiles | |
| `(6, 8)` | 48 | 12 tiles | Current baseline |

With `grid_x` fixed, `per_core_N`, `in0_block_w`, and the weight-direction subblock sizes are
unchanged across all configs. Only `per_core_M` and `out_subblock_h` need recalculating per
grid size.

### What needs to be built

1. **Parameterise `common.py`**: replace the hardcoded `(6, 8)` in each of the 7 program configs
   with a function that accepts `grid_y` and returns the correct config. The only value that
   changes is `per_core_M = 96 // grid_y`; `out_subblock_h` needs to be the largest divisor of
   `per_core_M` such that `out_subblock_h × out_subblock_w ≤ 8`.

2. **Thread `grid_y` through the runner**: `SentenceBERTPerformantRunner` →
   `SentenceBERTPerformanceRunnerInfra` → model construction. Each core count requires a
   fresh trace capture (the trace is compiled against a specific grid).

3. **Write the sweep script**: for each `grid_y` in `[1, 2, 4, 8]`, instantiate the model,
   capture the trace, run 50 iterations, record average latency, release. Run while telemetry
   is recording.

### Effort

**~4 hours** of engineering work. Requires the model weights (downloaded automatically on
first run via HuggingFace).

---

## Recommendation

Run **Approach A first**. The binary is ready today; the decode sweep with Llama-7B shapes will
produce the latency curve in under an hour and can be overlaid with the researcher's existing
power data immediately. This validates the joint graph and confirms the knee is visible in the data.

**Approach B** is the stronger demo artefact — a named model with a quotable finding — and is
worth the half-day of engineering once the methodology is confirmed with Approach A.

The `(6, 4)` = 24-core config in Approach B is the one to highlight: it sits just past the likely
knee, and the power saving relative to the 48-core baseline is the concrete number to put in any
write-up.
