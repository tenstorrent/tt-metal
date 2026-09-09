# Qwen3-TTS single-window Tracy reports

Two self-contained per-op device reports, one pytest command each:

| test | window | what it profiles |
|---|---|---|
| `test_qwen3_tts_perf_prefill_single_layer.py` | `prefill_single_layer_<bucket>` | one Talker `DecoderLayer`, prefill, at a demo TRACE bucket (32 / 64 / 128, default **64**) |
| `test_qwen3_tts_perf_decode_single_step.py` | `decode_single_step` | **one decode step, block by block** — Talker decode layer + CP prefill layer + CP decode layer + one device sampling call, in a single capture |

Each test spawns its own Tracy capture of its own file's `main()`, then writes
`reports/<window>/`:

```
ops_list.md          full per-op list + rollups (primary artifact)
tt-perf-report.txt   the ranked view, when tt-perf-report is installed
ops.csv              raw ops_perf_results CSV, every column, every op
totals.json          ops / device_ms / gap_ms / chips
run.log              the tracy run
```

`reports/` is gitignored — these are measurements, not sources.

## Running

N150 (see the repo's device-pinning convention; N300 swaps the descriptor and
`MESH_DEVICE`):

```bash
export TT_VISIBLE_DEVICES=1
export TT_METAL_CACHE=$HOME/.cache/tt_metal_n150_1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto
export MESH_DEVICE=N150
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=wormhole_b0

python_env/bin/python3 -m pytest -s -q models/demos/qwen3_tts/tests/perf/
```

~35 s for prefill, ~60 s for decode. Do **not** wrap these in `python -m tracy`
yourself — the test is the driver and starts tracy itself.

Knobs:

| env var | default | effect |
|---|---|---|
| `QWEN3_TTS_PERF_PREFILL_BUCKET` | `64` | prefill bucket: 32, 64 or 128 |
| `QWEN3_TTS_PERF_BUDGET_US` | unset | fail if window device kernel time exceeds this |
| `QWEN3_TTS_PERF_OP_SUPPORT_COUNT` | `2000` | profiler program budget |

## Measured baseline (N150, `QWEN3_TTS_BF8_WEIGHTS` default on)

| window | device ops | device kernel | matmul share |
|---|--:|--:|--:|
| `prefill_single_layer_32` | 29 | 0.416 ms | 58.9 % over 5 matmuls |
| `prefill_single_layer_64` | 23 | 0.472 ms | 62.1 % over 5 matmuls |

Bucket 32 is 29 ops against 64's 23: at seq<=32 QKV takes the DRAM-sharded path,
which is a different op sequence, not a cheaper one. Compare a bucket only against
itself.

`decode_single_step`, one capture, 112 ops / 1.199 ms total:

| block | ops | device kernel | per frame |
|---|--:|--:|--:|
| `talker_decode` (Talker layer) | 30 | 386 µs | ×28 |
| `cp_prefill` (CP layer, seq=2) | 30 | 248 µs | ×5 |
| `cp_decode` (CP layer, seq=1) | 33 | 248 µs | ×70 (5 layers × 14 steps) |
| `cp_sampling` (topk + gumbel + sampling) | 7 | 267 µs | ×15 |
| **blocks total** | **100** | **1.149 ms** | |
| per-window buffer setup (see below) | 12 | 52 µs | |

**One sampling call costs more device time than an entire CP layer** (267 µs vs 248 µs),
and `TopK` alone at 218 µs on **9 cores** outranks every matmul in the combined report.
It belongs to no layer, so a layers-only report is blind to the frame's largest
non-layer cost. Cross-check against a real frame capture: 15 × 218 µs = 3.27 ms
against that frame's measured 3.281 ms `TopK` total.

Data movement is 1.8 % of the prefill layer and 7.4 % of the Talker decode block —
both are matmul-bound, so per-layer wins come from the QKV / MLP program configs, not
from chasing reshards. Sampling is the exception: it is one `TopK` on 9 cores.

## Reading the report

- **`decode_single_step` totals 112 ops, not the 100 its blocks sum to.** The extra 12
  (52 µs, ~4 %) are per-window buffer setup — RoPE table resharding, mask tilize and
  typecast, KV cache init — which each window body does before its own signposts open,
  so they land in the outer window but in no block. The demo hoists this work once per
  frame rather than per layer. Quote the per-block slices; treat the outer window as
  the ranked cross-block view.
- **It is not a frame.** The blocks repeat 28 / 5 / 70 / 15 times, and a frame also
  carries the 15 LM heads, 15 CP final norms, the Talker codec_head, the accumulated
  codec embed, and host-side D2H/H2D that no device report shows. Use
  `../qwen3_tts_perf_report.sh -w decode_frame` for a real frame.

- **Ignore the op-to-op gap column.** These windows are *untraced* forwards, so
  every op waits on host dispatch and the gap (~4.3-4.9 ms) is dispatch latency,
  not anything on the chip. Device kernel time is the number to rank ops by and to
  A/B against another capture. For gap that means something, profile a trace
  replay — `../qwen3_tts_perf_report.sh`.
- **Per layer, not per frame.** The Talker has 28 layers, and one AR frame runs the
  CodePredictor 15 times against 1 Talker decode. Never scale one of these windows
  into a frame number; use `../qwen3_tts_perf_report.sh -w decode_frame` for that.
- Under TP (N300) the CSV holds one row per chip per op; `ops` is the merged
  per-chip count and each op's time is the max across chips.

## Relationship to the existing tests

The profiled device graph lives in
[`../test_qwen3_tts_profile_single_layer.py`](../test_qwen3_tts_profile_single_layer.py)
and is **called**, not copied — that module owns the layer construction, the
deployed-decode buffers and the `start` / `stop` signposts. Its docstring records
what a stale second copy costs: it once profiled bfloat16 gate/up at 116 us
against the 92 us the model actually ran.

What this folder adds over running that test under tracy by hand is the report
itself, plus the three checks that a hand-run capture silently skips: a profiler
DRAM overflow (partial CSV, no error), a CSV left over from an earlier run
(a plain pytest run writes no new report dir), and a window whose signposts never
opened.

Whole-model windows — Talker prefill, a full AR frame, the speaker encoder — are
[`../test_qwen3_tts_perf_report.py`](../test_qwen3_tts_perf_report.py)
plus its shell driver, and are not duplicated here.
