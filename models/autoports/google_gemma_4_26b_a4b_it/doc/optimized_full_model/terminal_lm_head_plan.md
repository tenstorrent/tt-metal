# Precision-preserving LM-head experiment

Status: measured; local K-block-4 head selected from the precision-locked sweep. Full-model validation is recorded by the stage owner.

The inherited head computes a BF16 input × BF16 checkpoint embedding transpose,
HiFi4 with FP32 destination accumulation. Each of four Blackhole devices owns
65,536 vocabulary columns; K is 2,816 (88 tiles). No vocabulary gather belongs
in this experiment. Decoder dtype/fidelity, residual, and CCL policy stay fixed.

| Path | Tensor ownership | Local data movement | Purpose |
|---|---|---|---|
| Inherited `ttnn.linear` | Replicated hidden, vocab-sharded head | Interleaved DRAM input/output | Same-process precision-locked baseline |
| Chunked DRAM-sharded head | Identical vocab ownership; weights split locally | One hidden reshard, per-chunk output conversion, local padding removal, concat | Faster weight reads while bounding L1 circular buffers |
| One/two/three readers | Identical padded weights and input shard within each K block | More reader/compute workers per physical bank | Per-role Blackhole reader comparison |

`models/common/modules/lm_head/lm_head_1d.py` supplies the chunked DRAM-sharded
execution family. Its generic configuration can express the fixed precision;
its normal compatibility factory chooses HiFi2 and BFP8 defaults, so those
must not be inherited. The probe mirrors the common module's sequence while
also slicing each padded local chunk, preserving the original sampler's exact
power-of-two 65,536-wide vocabulary shards. There are no invalid vocab IDs in
the resulting tensor: padding columns are removed before sampling.

The probe compares local chunk widths 8,192 and 16,384 and legal K blocks
1/2/4/11/22. Because 88 has factors 11 and 22, restricting K blocks to powers
of two would miss relevant geometries. Activation storage grids adapt to keep
a complete K block in each shard. Local weight widths are padded to a common
multiple of bank counts × readers (1/2/3) × tile width and storage cores, so
reader comparisons use exactly the same storage. This padding can increase
physical weight traffic; complete-path replay timing includes all resulting
conversions and padding removal. Known allocation failures are recorded with
exact messages; runtime/assertion/timeout failures abort instead of being
silently classified as illegal configurations.

`tests/probe_full_lm_head.py --capture` obtains actual normalized terminal
activations from checkpoint layers 0 and 5 under a chat-template prompt. This
is reduced-model real-input evidence, not all-layer quality evidence. The
subsequent benchmark loads real tied-embedding weights and holds BF16/HiFi4
fixed. All four device vocabulary outputs are checked against the original
head with PCC >= 0.999; top-1 equality and maximum absolute differences are
recorded. Adoption still requires complete-model readiness and token-out timing.

Timing uses warmed nonblocking replay with synchronization once per replay
batch. Reported microseconds are explicitly host-wall replay time, never
substituted for device time. Tracy signposts `LM_HEAD_<candidate>` surround the
complete measured path, enabling separate device profiling when serialized
hardware access is available. Candidate timings are not full-model results.

## Measured result (2026-09-27)

All jobs closed the TP4 Blackhole mesh successfully (exit 0). Device profiler and
watcher were unset; `TT_METAL_TRACE_ALLOC_TRACKING=0`. Numbers below are warmed
host-wall nonblocking trace replay microseconds including the complete isolated
head path. They are not device-only or complete-model metrics.

| Candidate | Head replay us | Outcome |
|---|---:|---|
| Existing 11x10 interleaved, K block 8 | 955.30 | Baseline |
| 8x8 interleaved, K block 4 / subblock 4 | 1053.25 | Slower |
| 8x10 interleaved, K block 4 / subblock 2 | 1040.39 | Slower |
| 11x10 interleaved, K block 4 / subblock 1 | 936.85 | Selected: ~18 us isolated-head saving |
| 11x10 interleaved, K block 11 | 963.74 | Slower |
| DRAM 8192 chunks, K block 1, two readers | 1407.20 | Slower |
| DRAM 32768 chunks, K block 1, two readers | 1245.46 | Best DRAM candidate still slower |
| DRAM 1024 chunks, K block 22 / 44 / 88, three readers | 2624.95 / 2790.33 / 3098.01 | Adapted larger blocks work but lose whole-path latency |

The K-block-4 candidate reproduced across two processes: 937.57 and 936.85 us.
The later five rounds ranged 935.32–937.99 us; same-process K-block-8 control
ranged 954.05–956.87 us. Smaller coherent grids permit larger subblocks but
lose despite that local advantage. The selected local runtime change sets `head_program.in0_block_w=4` in
`tt/model.py`. BF16 weights/input/output, HiFi4, FP32 destination accumulation,
vocabulary ownership, grid, and output memory remain unchanged.

Every measured candidate retained all four devices' local top-1 tokens and
PCC > 0.999999 against the inherited head. This is one actual terminal input,
not an all-layer readiness gate. The exact fixture provenance is in each JSON;
`terminal_input.pt` is the binary source artifact and must not enter telemetry.

A single 65536-column DRAM chunk exceeded L1 for all attempted readers at
K blocks 1 and 2. The adapted 32768-column chunks ran with two or three
readers; reducing to 1024 columns made all larger K blocks through 88 run.
Thus DRAM sharding was not rejected on the first error. The larger interleaved
K-block candidates requested 2,030,592 bytes (22) or 3,832,832 bytes (44/88),
above maximum L1. Exact allocation traces remain in the JSON/log artifacts;
`terminal_lm_head_candidates.csv` keeps compact error excerpts. The measured
small-chunk penalty includes reshard, conversion, padding removal and concat.

Commands used the common prefix:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_full_lm_head \
  --fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/terminal_input.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/<artifact>.json
```

Capture added `--capture`; the five timing jobs added these exact arguments:

| Artifact | Additional arguments |
|---|---|
| terminal_lm_head | `--chunks 8192 --blocks 1 2 4 11 --readers 1 2 3` |
| terminal_lm_head_large | `--chunks 32768 65536 --blocks 1 2 --readers 1 2 3` |
| terminal_lm_head_interleaved | `--interleaved-only --blocks 4 8 11 22 44 88` (native 11x10 grid) |
| terminal_lm_head_adapted | `--chunks 1024 --blocks 22 44 88 --readers 3 --rounds 2 --replays 5` |
| terminal_lm_head_grids | `--interleaved-only --grids 8x8 8x10 11x10 --blocks 4 8 --rounds 5 --replays 20` |

Each timing command ran under `timeout 300` or `timeout 600` with stdout/stderr
in the matching `.log`. The first three default to 3 rounds of 10 replays.

The selected K-block change does not add weights, caches, persistent tensors,
trace inputs, or CCL buffers. It reduces the transient K-block working set;
the existing context-contract byte accounting remains valid. The binding
explicitly exposes a writable `in0_block_w` (`matmul_nanobind.cpp:423`).
The sweep leaves 44 candidate/control rows: 29 measured passes and 15 exact
allocation rejections. `terminal_lm_head_candidates.csv` covers all five runs.
