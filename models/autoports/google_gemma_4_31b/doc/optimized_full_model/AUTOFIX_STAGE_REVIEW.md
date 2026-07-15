# AutoFix Report: Stage 07 review findings

## Starting evidence

- Original failing gate: `stage_review.md`, verdict `more-work-needed`.
- Source inspected: `tt/model.py`, `tt/generator.py`,
  `tests/test_full_model.py`, `tests/run_full_model_qualitative.py`, and the
  Stage 07 candidate/profiler/performance artifacts.
- The inspection was read-only and did not open TT devices. This report is the
  only file written by this pass.
- The worktree also contains unrelated deletion/untracked state. Remediation
  and the eventual commit must remain path-scoped to the Stage 07 files.

## Hypothesis experiments

### H1: the selected four-input-shard geometry has untested legal K blocks

**Hypothesis:** measurements from the earlier eight-input-shard geometry do
not adjudicate the selected four-input-shard geometry.

**Inspection:** the hidden dimension is 5,376 = 168 tiles. Four input shards
therefore contain 42 K tiles each. `Gemma4FullModel.__init__` accepts exactly
the positive divisors of 42 as `in0_block_w`; the legal set is
`1, 2, 3, 6, 7, 14, 21, 42`. The selected value is 2, so the unmeasured larger
set is `3, 6, 7, 14, 21, 42`. The program config uses this value directly in
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, with `per_core_M=1`
and `per_core_N=64` for split 8,192/four cores. Weight placement remains eight
physical Blackhole DRAM views; `lm_head_num_cores` controls logical input
shards, not physical DRAM banks.

The legal set is larger than the material hardware set. In
`ttnn/cpp/ttnn/operations/matmul/device/factory/`
`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:181-212`, the
BF16 weight CB contains `per_core_N_in1_sender * W` tiles and is triple
buffered. For this 8,192-column split, 256 N tiles divide over eight Blackhole
DRAM readers, so this CB alone is
`32 * W * 3 * 2,048 = 196,608 * W` bytes. Widths 14, 21, and 42 therefore
require 2,752,512, 4,128,768, and 8,257,536 bytes before any other CB; each is
physically above the evidenced 1,572,864-byte L1 capacity. These are exact
source-derived blockers and do not need device time. Width 7 consumes
1,376,256 bytes for this CB alone and is the material boundary compile probe;
width 6 consumes 1,179,648 bytes and is the largest plausibly fitting larger
candidate. The factory independently enforces `Kt % in0_block_w == 0` at
lines 1006-1012.

The reduced performance test is the current focused candidate harness: unlike
the broader reduced functional test, it reads all four LM-head overrides at
`tests/test_full_model.py:461-464`, executes real prefill, exact split greedy,
trace capture/replay, and asserts the no-host-boundary counters. Use it for
legality/blocker and warmed trace latency. Run fresh matched width-2, width-3,
and width-6 measurements, plus width 7 as the boundary compile probe, in
separate serialized processes (no watcher or profiler). Widths 14+ are closed
by the exact CB arithmetic above:

```bash
D=models/autoports/google_gemma_4_31b/doc/optimized_full_model
W=2  # repeat for 3, 6, and boundary probe 7
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig \
  GEMMA4_31B_FULL_MODEL_RUN_REDUCED_PERF=1 \
  GEMMA4_31B_FULL_MODEL_REDUCED_PERF_TOKENS=12 \
  GEMMA4_31B_FULL_MODEL_REDUCED_PERF_OUT="$D/candidates/lm_head_dram4_split8192_block${W}_perf.json" \
  GEMMA4_31B_LM_HEAD_DRAM_SHARDED=1 \
  GEMMA4_31B_LM_HEAD_NUM_CORES=4 \
  GEMMA4_31B_LM_HEAD_IN0_BLOCK_W="$W" \
  GEMMA4_31B_LM_HEAD_SPLIT_SIZE=8192 \
  pytest -q -s \
    models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_full_model_token_out_perf_signposts \
    --junitxml="$D/candidates/lm_head_dram4_split8192_block${W}.xml"
```

Retain the exact width-7 construction/L1/CB result. For widths 2, 3, and 6
that pass, repeat the timing three times and compare median steady t/s/u, not a
single setup sample or the historical width-2 repeat. Then run the same-hidden
full-60-layer diagnostic below. It projects
the identical normalized prompt-5 hidden state through legacy interleaved and
candidate heads and records every BF16 logit before and after softcap, block
ordering, top candidates, and device/host greedy choices. `max-new-tokens=1`
keeps the adjudication focused while still exercising all six full-model
prefills and first greedy token.

```bash
D=models/autoports/google_gemma_4_31b/doc/optimized_full_model
HF=/home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3
W=3  # each passing larger candidate
env LD_LIBRARY_PATH=$PWD/build/lib MPLCONFIGDIR=/tmp/mplconfig \
  GEMMA4_31B_LM_HEAD_NUM_CORES=4 \
  GEMMA4_31B_LM_HEAD_IN0_BLOCK_W="$W" \
  GEMMA4_31B_LM_HEAD_SPLIT_SIZE=8192 \
  python models/autoports/google_gemma_4_31b/tests/run_full_model_qualitative.py \
    --model-dir models/autoports/google_gemma_4_31b \
    --hf-model "$HF" \
    --prompt-source models/common/readiness_check/vllm_prompts.txt \
    --output-dir "$D/candidates/lm_head_dram4_split8192_block${W}_aligned" \
    --max-new-tokens 1 \
    --lm-head-aligned-ab

jq -e '
  .pre_softcap.aligned.exact_bf16_fraction == 1 and
  .post_softcap.aligned.exact_bf16_fraction == 1 and
  .sampler.legacy_device_token == .sampler.optimized_device_token and
  .sampler.optimized_device_token == .sampler.optimized_host_argmax
' "$D/candidates/lm_head_dram4_split8192_block${W}_aligned/lm_head_aligned_ab.json"
```

**Prediction:** a candidate is eligible only if its program is legal, its
aligned pre/post-softcap logits remain bit-identical to legacy, its sampler
agrees with host argmax, and its median warmed trace latency improves on block
2. A faster numerically different accumulation geometry is rejected under the
existing Stage 06 trajectory contract. After selecting the winner, rerun the
full 64-token qualitative control and the final profile from that default.

**Owner-side result:** complete. Block 3 passes at 337.545868 steady t/s/u,
slower than selected block 2, and its full-model same-hidden diagnostic changes
the greedy winner 669 -> 108 with only 0.160728 exact pre-softcap BF16 fraction.
Block 6 clashes between an L1 allocation at 1,351,040 and static-CB end
1,381,120. Blocks 7/14/21/42 grow static CBs to
1,581,824/2,986,752/4,391,680/8,606,464 bytes against 1,572,864-byte L1.
All logs, JUnit, JSON, and aligned-logit evidence are under `candidates/`.

**Verdict:** fixed. Block 2 is the only passing correct winner.

### H2: the top-level compact summary belongs to rejected block 3

**Hypothesis:** an output-suffix mistake left selected raw/report artifacts
beside a stale rejected stacked summary.

**Experiment:** hashes and totals show:

- selected raw `profiler_raw_ops.csv` SHA-256:
  `7997ff98469f6dad6f3dbb4cfb7b7f058910791671105a363f94f8db14f2e133`;
- selected `tt_perf_report.csv` SHA-256:
  `42c2e660816bc783a5c91abbdfa1300550cd69de97b1c85f3c1ae88a3ea1ee8c`;
- top-level `tt_perf_summary.csv` is byte-identical to the rejected block-3
  summary (SHA-256 `43fb8f...`);
- selected `tt_perf_summary.csv.csv` sums to 2,833.08 us/155 ops and agrees
  with the selected detailed report's 2,833 us total.

The CLI appends `.csv` to `--summary-file`; passing a name already ending in
`.csv` created the doubled suffix. Regenerate without hardware after the final
default profile, passing a suffix-free summary path:

```bash
D=models/autoports/google_gemma_4_31b/doc/optimized_full_model
env MPLCONFIGDIR=/tmp/mplconfig tt-perf-report "$D/profiler_raw_ops.csv" \
  --start-signpost GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY \
  --end-signpost GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY_END \
  --no-color \
  --csv "$D/tt_perf_report.csv" \
  --summary-file "$D/tt_perf_summary" \
  > "$D/tt_perf_report.console.log"
env MPLCONFIGDIR=/tmp/mplconfig tt-perf-report "$D/profiler_raw_ops.csv" \
  --start-signpost GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY \
  --end-signpost GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY_END \
  --no-color --no-summary > "$D/tt_perf_report.txt"
sha256sum "$D/profiler_raw_ops.csv" "$D/tt_perf_report.csv" \
  "$D/tt_perf_summary.csv" "$D/tt_perf_report.txt" \
  > "$D/profiler_sha256.txt"
```

**Owner-side result:** regenerated with a suffix-free `--summary-file` target.
The selected summary is 2,833.08 us/155 ops and `profiler_sha256.txt` binds the
losslessly compressed selected raw CSV and all compact reports.

**Verdict:** fixed; no runtime change was implicated.

### H3: the reported 0.904983 ms terminal cost subtracts unlike regimes

**Hypothesis:** full-trace end-to-end timing minus standalone per-layer host
medians cannot isolate terminal kernels.

**Experiment:** sorting the selected block-2 detailed CSV by `Global Call
Count` restores program order and produces non-overlapping spans:

| Span | Global calls | Ops | Device-time sum |
|---|---:|---:|---:|
| embedding/orchestration input | 595970--601091 | 6 | 25.503 us |
| one sliding layer | 602114--665603 | 63 | 448.78175 us |
| one full-attention layer | 666624--720898 | 54 | 479.633 us |
| final norm, LM head, softcap, positions | 721922--747523 | 26 | 1,565.191 us |
| exact greedy and token feedback | 754688--759808 | 6 | 313.981 us |

The terminal-plus-sampler device work is therefore 1.879172 ms, including
1.425920 ms for the eight LM-head matmuls and 0.298933 ms for exact greedy.
This directly refutes the 0.904983 ms terminal label.

Preserve the contract-requested Stage 05 standalone layer-stack arithmetic,
but label its regime accurately:

```text
50 * 0.463813 + 10 * 0.5166275 = 28.356925 ms/token
```

It is a sum of standalone optimized-layer medians and a decoder-stack target,
not an additive physical lower bound for a captured full trace. The selected
profile supplies the like-regime operation-sum model:

```text
profiled stack = 50 * 0.44878175 + 10 * 0.479633
               = 27.2354175 ms
terminal       = 1.565191 + 0.313981
               = 1.879172 ms
embedding      = 0.025503 ms
modeled full   = 27.2354175 + 1.879172 + 0.025503
               = 29.1400925 ms/token = 34.31698 t/s/u
observed full  = 29.254761467 ms/token = 34.18247 t/s/u
unmodeled gap  = 0.114668967 ms/token = 0.39351% of modeled full
```

This closes the 10--15% trigger without impossible accounting: the selected
full path is only 0.42% slower than the profile-derived full-path operation
model. Also report that naively adding the standalone stack target and measured
terminal gives 30.236097 ms, which is slower than the observed full trace and
therefore demonstrates the scheduling/regime mismatch rather than a negative
terminal cost.

**Verdict:** fixed reporting bug; corrected arithmetic is propagated to the
README, work log, and `perf_summary.json`.

### H4: the full-path TTFT regression lacks a matched control

**Hypothesis:** the single Stage 06/Stage 07 samples cannot distinguish a real
prefill regression from cache/setup variance.

**Inspection:** `benchmark_token_out_no_readback` is suitable once run under a
matched harness. It resets explicit cache/trace state for every sample, starts
TTFT timing immediately before the 149-token prefill, captures the same
full-60-layer model and sampler traces, runs 98 no-readback steady replays for
a 100-token request, synchronizes once at the end, and returns both overall and
steady throughput plus trace counters. Model construction is outside TTFT.

The matched A/B must compare source-current configurations, not the old
reduced baseline artifact:

- baseline: `Gemma4FullModelConfig(lm_head_dram_sharded=False)`;
- default: `Gemma4FullModelConfig(lm_head_dram_sharded=True,
  lm_head_num_cores=<selected>, lm_head_in0_block_w=<selected>,
  lm_head_split_size=8192)`;
- same HF snapshot, tensor-cache path, P150_X4/FABRIC_1D mesh, prompt 0 from
  `doc/full_model/readiness_aime24_plain.refpt` (149 tokens), batch 1, context
  262,144, and 100 output tokens;
- one unrecorded warmup followed by at least five recorded calls on the same
  constructed generator for each configuration;
- serialize processes and alternate configuration order if a second set is
  needed; never enable watcher or profiler for headline timing.

The smallest harness extension is to expose `lm_head_dram_sharded` beside the
existing three LM-head environment overrides and add benchmark warmup/repeat
arguments. Store every sample, median, min/max, and the per-sample trace
counters in separate baseline/default JSON files. Required counters remain 99
model replays, zero token/page/full-logit host traffic, one seed-token readback,
and only setup/final synchronization. Compare medians for TTFT, overall t/s/u,
and steady t/s/u. Fix a reproducible median regression; otherwise report its
absolute range and classify the old +8.42% single-sample result as uncontrolled
measurement noise. Reduced-run TTFT must not be used for this decision.

**Owner-side result:** complete. One warmup plus five recorded full-60-layer
calls per constructed generator give selected versus unsharded medians of
452.409/444.008 ms TTFT, 24.98654/24.88885 overall t/s/u, and
34.18247/33.87453 steady t/s/u. Changes are +1.89% TTFT, +0.39% overall, and
+0.91% steady. All ten samples retain the clean trace counters.

**Verdict:** fixed. The earlier +8.42% TTFT/-1.74% overall single-sample result
is classified as setup/cache-sensitive historical evidence, not the headline.

## Final status

Status: resolved, fresh independent rereview pending.

All four review findings have isolated owner-side evidence: the compatible
K-block frontier is exhausted, block 2 remains the only correct winner, compact
profile provenance is repaired and hashed, lower-bound accounting is
like-regime, and matched full-path medians replace uncontrolled samples. The
selected default itself did not change, so the source-current full accuracy,
qualitative, watcher, and selected profiler gates remain applicable.
