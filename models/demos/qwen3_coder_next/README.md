# Qwen3-Coder-Next

Text generation pipeline ([Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)) on Tenstorrent hardware.

An ~80B-parameter (159 GB bf16) hybrid Mixture-of-Experts causal LM. 48 decoder layers where three of every four use a **gated delta net** (linear attention) and the fourth uses **GQA attention**, and *every* layer's MLP is a **512-expert sparse MoE** (top-10, plus a shared expert; hidden size 2048). All ten graduated components are native TTNN — the pipeline runs with zero host aten ops in the model math.

## Status

- **On device:** the full 48-layer model, resident on 8 chips. All ten components native TTNN, verified by `test_gate1_stubs_are_native_ttnn` (no torch host compute) and `test_host_op_selftest` (zero host aten ops).
- **Correct:** worst-step PCC **0.973384** against a 0.95 gate.
- **Not yet optimized.** The 9989.6 ms figure below is an honest *baseline*, not a shipped number. Two things dominate it and neither is tuned: every decode step re-runs the whole prefix, and `batch=1` leaves 24 of the 32 chips carrying no weights.
- **Next: optimize.** A first op-level profile puts ~40% of device time in layout conversion (tilize/untilize) and only 27.7% in matmul. The two largest addressable items are `dense_routing` (48.4 ms against a 0.105 ms roofline floor — 1016 slice dispatches) and the vocab-wide `ArgMax` (74.8 ms against 0.386 ms). The expert matmul (1.9× off floor) and the 622 MB untilize (1.07×) are already near roofline and are *not* where the time is.

## Hardware

- **Board:** Blackhole 6U Galaxy
- **Mesh:** (8, 4) — 32 chips over `FABRIC_1D`

## Parallelism Strategy

| Component | Parallelism |
|---|---|
| Decoder stack (48 layers) | TP=8 × DP=4 — 8 chips carry weights |
| `gated_delta_net` (36 layers) | TP=8, head-sharded (16 k-heads / 32 v-heads) |
| `attention` (12 layers) | **Replicated** — `kv_heads=2` caps head-wise TP at 2, but it is only 0.65 GB (0.4% of the weights) |
| MoE experts | Sharded by expert index (512 experts, top-10) |
| `lm_head` | Column-parallel over the vocabulary, one `all_gather` |

`kv_heads=2` caps TP at 2 for `attention` alone. Replicating it and sharding everything else 8 ways gives **21.58 GB/chip against 28.6 usable**, which is what makes the full stack resident. The SystemMesh is (8, 4) and a submesh must divide it *elementwise*, so a TP=8 group is carved as **(2, 4)** — asking for `(1, 8)` raises `TT_FATAL`.

## Supported

- **Text → text**, greedy causal generation
- Deterministic: greedy argmax on device, no sampling

## Prerequisites

The checkpoint (~159 GB) downloads on first run. Set `HF_TOKEN` to avoid unauthenticated rate limits, or point at a local snapshot:

```bash
export TT_QWEN3_SNAPSHOT=/path/to/models--Qwen--Qwen3-Coder-Next/snapshots/<sha>
```

## How to Run

All commands from the tt-metal repo root.

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
```

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `TT_QWEN3_LAYERS` | 48 | **Build** depth. Set to 4 for the fast stack — the smallest depth whose `layer_types` covers both token mixers. |
| `TT_QWEN3_TP` | 8 (from `parallelism_manifest.json`) | Tensor-parallel degree |
| `TT_QWEN3_CAPACITY` | 64 | Pinned sequence capacity `C` |
| `TT_QWEN3_MAX_NEW_TOKENS` | model's `generation_config` | Decode length, clamped by `C - prompt_len` |
| `TT_QWEN3_MESH` | ladder | Force a mesh shape, e.g. `8x4` |
| `TT_QWEN3_SNAPSHOT` | HF cache | Local checkpoint path |
| `TT_PERF_LAYERS` | build depth | **Caps the profiled window** — see Profiling |

### Text generation

```bash
python models/demos/qwen3_coder_next/demo/demo_text_generation.py \
    --prompt "Write a Python function that returns the nth Fibonacci number."
```

Add `--compare` to run the HF golden alongside and report PCC. `--layers 4` for a fast run.

## Tests

```bash
# per-component PCC vs the HF reference — single-chip and TP=8 sharded
pytest models/demos/qwen3_coder_next/tests/pcc -s

# the correctness gate: PCC only, deliberately blind to HOW the answer was computed
pytest -o timeout=0 models/demos/qwen3_coder_next/tests/accuracy/test_model.py -s

# structural gates: native-ttnn sources, TP shard widths, zero host aten ops, trace capture
pytest -o timeout=0 models/demos/qwen3_coder_next/tests/e2e/test_e2e_pipeline.py -s

# the stubs AS THE PIPELINE — re-routing regression net against captured goldens
pytest -o timeout=0 models/demos/qwen3_coder_next/tests/e2e/test_captured_parity.py -s
```

**Point an optimizer at `tests/accuracy/test_model.py`, not the e2e file.** The e2e gate also asserts native-ttnn stub sources, exact TP shard widths, exact per-module invocation counts and zero host aten ops. Those are right for bring-up and wrong for optimization: fusing two modules or changing a shard width are legitimate speedups that leave PCC untouched yet trip a structural assertion. The accuracy gate asserts one thing — the logits still track HuggingFace — and prints `PCC: <float>` on its own line.

`_captured/` (the HF golden tensors the per-component tests compare against) is gitignored. Regenerate with:

```bash
python -m scripts.tt_hw_planner capture-inputs Qwen/Qwen3-Coder-Next
```

## Profiling

The demo harness carries Tracy signposts (`PREFILL_START/END`, `DECODE_START/END`), so a capture can be split into phases:

```bash
TT_QWEN3_LAYERS=4 TT_QWEN3_MAX_NEW_TOKENS=2 \
python -m tracy -r -p --op-support-count 40000 -o <outdir> -m pytest -- \
    models/demos/qwen3_coder_next/demo/text_demo_signpost.py::test_qwen3_signpost -x
```

Look for `Device only OPs csv generated at: .../ops_perf_results_*.csv` — and **check its size**; a run that produced no device work still writes a 2-byte header.

**Cap the profiled window.** A capture costs ~260 MB per layer-execution, so 48 layers × 5 forwards is ~62 GB. Above a few layers the device profiler also overflows and logs `Profiler DRAM buffers were full, markers were dropped!` — an *incomplete* profile that still yields a confident-looking ranking. `TT_PERF_LAYERS` caps the window without changing the build depth.

## Performance

48 layers, TP=8 × DP=4, `C=32`, `batch=1`, on one Blackhole 6U Galaxy.

| Stage | Time |
|---|---|
| **End-to-end decode step** | **9989.6 ms** (trace+1cq) |

A **baseline, not an optimized number.** Each "decode step" runs the whole model over the entire pinned capacity and consumes one row of the result, so the loop does O(`C`) times more work than a decode step needs. See Status.

## Accuracy

| Check | PCC | Gate |
|---|---|---|
| End-to-end generation, worst step (48 layers) | **0.973384** | ≥ 0.95 |
| End-to-end generation, per-step range | 0.9734 – 0.9915 | ≥ 0.95 |
| End-to-end, TP=8 × DP=4 | 0.9869 | ≥ 0.95 |
| `gated_delta_net` vs HF, seq 32–256 (1–4 chunks) | ≥ 0.99993 | ≥ 0.99 |

## Known limits

- **Decode re-runs the prefix.** `gated_delta_net` walks the delta-rule state across 64-token chunks, so there is no sequence-length ceiling, but the conv cache, the `seq==1` recurrent kernel and an attention KV cache are not built yet. `initial_state=` / `output_final_state=` are exposed for them; a split run currently scores 0.9907 against one-shot's 0.99994, and that gap is the missing conv cache (kernel 4 needs three timesteps of history).
- **`batch=1`**, so 24 of 32 chips carry no weights. `tt/pipeline.py` has no batch dimension.
- **The MoE computes all 512 experts per token and keeps 10.** Real, but bounded by matmul's 27.7% share of device time — see Status.

## Layout

```
_stubs/            native-TTNN components: gated delta net, attention, experts,
                   top-k router, sparse MoE block, MLP, both norms, rotary, decoder layer
tt/pipeline.py     THE chained forward pass — demo and tests both import this
tt/mesh.py         chip placement: TP groups carved from the 32-chip mesh
tt/reference.py    HF config / tokenizer / depth-capped reference checkpoint
device_harness.py  the package's SOLE device opener
demo/              text-generation demo and the signposted profiling harness
tests/accuracy/    PCC-only correctness gate — the one to point an optimizer at
tests/e2e/         structural gates, captured-parity regression net
tests/pcc/         per-component PCC, single-chip and TP=8 sharded
```
