# Optimized multichip decoder work log

Date: 2026-07-18 UTC

## Scope and authoritative starting point

- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Starting HEAD: `a84e536ec56`.
- Runtime under optimization: the real TP4 `MeshShape(1,4)` path in
  `tt/multichip_decoder.py`; no single-chip or replicated-compute fallback is
  an acceptable target.
- Starting multichip implementation/evidence commit: `164cd8dff60` with the
  docs bookkeeping commit at starting HEAD.
- The unrelated pre-existing edit to
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` is preserved and
  excluded from this stage.
- Full-model, generator, and vLLM work are not being started.

## Required workflow inputs read

The complete `$optimize`, `$tt-device-usage`, `$graph-rewrite`, and
`$shard-advise` instructions were read, together with shard-advise `SETUP.md`
and section 4 of `tech_reports/LLMs/llms.md`.  The current multichip source,
tests, profiler reports, candidate tables, context contract, and prior
clean-pass stage review were inspected before making runtime changes.

## Hardware health

`timeout 60 tt-smi -ls --local` showed UMD IDs 0 through 3 as Blackhole P300c.
A bounded source-backed `MeshShape(1,4)` open/close returned
`MESH (1, 4) 4 blackhole` and `MESH_SMOKE_OK`.  The known 64 MiB `/dev/shm`,
firmware-version, motherboard-discovery, and nanobind diagnostics recurred but
did not affect mesh discovery or close.

## Fresh unchanged baseline

The canonical ordered performance pair was run with batch 1, logical prefill
length 18, 50 warmed prefill iterations, and 1,000 traced decode replays.
Both tests passed.

| Path | Prefill (ms) | Traced decode (ms) | Output PCC |
| --- | ---: | ---: | ---: |
| single-chip control | 1.243493 | 0.581238 | control |
| TP4 unchanged default | 0.746841 | 0.320079 | 0.9999998070672766 |

An initial narrower selector invoked only the second test and correctly failed
its harness precondition because the module-scoped single-chip reference had
not run.  No model or device failure occurred; the canonical `-k warmed_perf`
selector was then used.

## Topology-first finding

The retained current-source TP4 decode report shows that each of the two
nominal all-reduces is actually expanded into:

`sharded partial -> sharded-to-interleaved -> reduce-scatter -> all-gather -> interleaved-to-sharded`.

Across decode, reduce-scatter is 20.76%, all-gather is 12.67%, and explicit
data/tensor movement is about 14.46%.  Current TTNN also exposes a true minimal
all-reduce overload that consumes the existing L1 width-sharded partial,
requires a persistent 4x intermediate buffer plus one global semaphore, and
returns the same replicated L1 width-sharded residual.  This is the first
candidate family because it can remove four material rows at each row-parallel
boundary without changing the inter-layer residual contract.

Detailed audit and the complete family matrix are in `topology_audit.md`.

## Dedicated minimal all-reduce rewrite

The decode-only rewrite uses the overload which takes a sharded input, a
globally allocated persistent buffer, one cyclic global semaphore, the mesh,
and `cluster_axis=1`.  The buffer shard is four times an output shard, as
required by the ring.  It is allocated after prefill and reused sequentially
for attention and MLP because the reductions are data-dependent.

Initial one- and two-buffer runs both measured about 0.2666 ms decode.  One
buffer was retained because it has the same performance with half the L1
reservation.  The long BF16 result was 0.266609 ms versus 0.320035 ms for an
adjacent composite control.  Default worker placement won over the special
Llama placement (0.267443 ms); normal NOC won over NOC1-only (0.272685 ms);
two links won over one (0.283190 ms).  Decode-only BFP8 reached 0.263860 ms
with DRAM-sharded compute but lost once the faster advisor compute family was
applied (0.260163 versus 0.250921 ms BF16).

## Shard advisor

The exact TP4-local batch-one decode capture is
`shard_advise/advise_tp4_local.py`.  The advisor's environment must be sourced
from its own checkout directory; the first invocation sourced it from the
tt-metal directory and therefore produced a namespace-only `ttnn` import.
The corrected command was:

```bash
cd /home/mvasiljevic/tt-mlir
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}:/home/mvasiljevic/tt-metal:/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages
ttnn-advise capture \
  /home/mvasiljevic/tt-metal/models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/shard_advise/advise_tp4_local.py:decode \
  --out /home/mvasiljevic/tt-metal/models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/shard_advise/result
```

The run completed with 24 ops, 21 final choices, a completed spill pass, and
zero spills.  The capture represents the two CCL boundaries as identity
dependencies because the advisor does not model TTNN CCL.  Hardware evidence
covers the actual minimal all-reduces.

The first exact 64-core residual-chain attempt failed because its 11x6 output
CoreRangeSet was not contained by the buffer's rectangular 8x8 grid.  The
buffer was adapted to the advisor CoreRangeSet and retried.  The next error
showed that a 22-way norm shard has six K tiles and cannot satisfy
`in0_block_w=8`; applying the advisor's explicit L1-interleaved reshard at the
norm-to-gate/up and activation-to-down boundaries made the whole family pass.
It measured 0.251809 ms BF16 and 0.272064 ms BFP8, slower than the retained
16-core residual.  No family was rejected on its first TTNN error.

## Topology and program-config family decisions

The fresh advisor 1-D family plus BF16 minimal all-reduce initially reached
0.250921 ms.  A phase-specific layout was necessary to preserve prefill:
DRAM-sharded weights remain live for prefill; interleaved decode weights are
allocated by `prepare_decode()` after prefill.  There is no runtime weight
conversion inside the measured region.

The initial profiler marked QKV's 48-core 1x1 subblock actionable.  Precision-
locked retries showed:

| Candidate | traced decode |
| --- | ---: |
| advisor seed, 48-core QKV / block 8 | 0.253410 ms |
| all projections `in0_block_w=16` | 0.252076 ms |
| 24-core QKV / block 8 / 1x2 | 0.251352 ms |
| 24-core QKV / block 16 / 1x2 | **0.247563 ms** |
| 24-core QKV plus block 16 on other projections | 0.253397 ms |
| 24-core QKV, other projections only at block 16 | 0.256953 ms |

The 24-core QKV/block-16 configuration was promoted.  Packed gate/up with the
minimal/BFP8 family measured 0.264506 ms versus 0.263860 ms for separate
projections and was rejected.  The full candidate table is in
`candidate_results.csv`.

## Prefill candidate and paired timing

The final prefill graph is intentionally unchanged.  A sharded-norm candidate
owned tile padding for logical seq 18 and restored DRAM only before each
projection.  It passed PCC but measured 0.991491 ms versus the adjacent
0.711551 ms control, so the two conversion boundaries cost more than the
one-core norm saved.

Prefill E2E varied significantly across process runs.  With one invariant
single-chip warm-up, the median of three 100-iteration legacy controls was
0.773172 ms.  Three pre-review block16-default samples were 0.779536,
0.814313, and 0.740450 ms (median 0.779536, +0.82%).  Device-profiler prefill
time is effectively identical before and after at about 1.266 ms for three
executions.  Decode did not show this drift: the corresponding pre-review
samples were 0.247608, 0.247549, and 0.247538 ms.  The promoted block32 final
default is recorded in the repair cycle below.

## Initial final profiler, retained as pre-review topology evidence

The pre-review block16-default capture command was:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_VARIANT=default \
MULTICHIP_DECODER_BATCH=1 MULTICHIP_DECODER_SEQ_LEN=18 \
MULTICHIP_DECODER_PREFILL_REPEATS=3 MULTICHIP_DECODER_TRACE_REPLAYS=1 \
timeout 300 python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/tracy \
  -m pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k warmed_perf
```

Raw ops provenance is the `2026_07_18_13_09_56` report.  Each signposted
region was processed twice with advice enabled: once with `--csv` and
`--summary-file`, and once with `--no-summary` for the human-readable table.
The pre-review TP4 decode summary is 46.94% 1-D matmul and 18.16% minimal
all-reduce.  The decode report contains no reduce-scatter or all-gather.  QKV
is measured as 24 cores, block 16, 1x2, BF16 x BFP4 to BF16, LoFi.  This is
superseded by the replay-only block32 profile described below.  O,
gate/up, and down are block 8, 1x2, BF16 x BFP4 to BF16, LoFi.

## Pre-review gates

Correctness and fallback audit:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
timeout 300 pytest \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k 'runtime_path or context_capacity or stack_shares or correctness' -q -s
```

Result: 5 passed.  Final-default PCC includes prefill 0.9999993671, decode
0.9999885868, stacked decode 0.9999543911, and page-boundary positions
63/64/65 above 0.999969.

The first watcher command failed during mesh setup because ACTIVE_ETH watcher
code was 27,920 B for a 25,600 B kernel config buffer.  The prescribed scoped
retry was:

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
RUN_MULTICHIP_DECODER_WATCHER=1 timeout 300 pytest \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k watcher_stress -q -s
```

Result: 1 passed at batch 32 with paged BFP8 caches and ten deterministic
trace replays at PCC 1.0.  Watcher was run separately from Tracy.

The context contract was updated for the second persistent BFP4 weight layout:
14,847,836,160 B/device conservative BF16 plan versus 34,178,731,008 B/device
physically probed allocator capacity.  The full 131072-token contract remains.

## Independent-review repair cycle

The first independent review returned `more-work-needed` with four concrete
issues.  Each was repaired rather than dismissed.

### Stack-shared CCL workspace

The reviewer found that each layer retained a 1-MiB/device minimal-all-reduce
buffer on the same 16 workers.  Thirty-two copies would consume 2 MiB/core,
above Blackhole's 1.5 MiB/core before runtime CBs.  Buffer ownership now lives
in a shape/dtype/layout-keyed pool on the already stack-shared `TT_CCL` owner.
The 32-layer unit calls `prepare_decode()` on 32 distinct instances and proves
one allocation.  The hardware correctness test now creates two distinct full
decoders, proves they share the same buffer object, and directly chains their
prefill and decode outputs.  The final pool uses 1,048,576 B/device, or 65,536
B on each of 16 workers, independent of layer count.  `context_contract.json`
records this physical L1 evidence.

### Final-topology packing and per-role geometry

Packed gate/up was first adapted to the selected advisor/BF16-minimal family rather
than being rejected by the advisor's separate-projection capture.  The working
candidate uses one 56-core packed projection with `per_core_N=4`, converts the
packed result to L1 interleaved, slices gate/up on device, performs fused SiLU
multiply, and feeds the retained down projection.  It passes PCC
`0.9999998069059012` but is slower: 0.249415 ms versus the separate family.
The second review correctly required this and the near-winning role geometry
to be rerun cumulatively under the subsequently selected QKV32 policy; those
results are recorded below.

O, the identical-shape gate/up group, and down then received independent
core-grid, `per_core_N`, output-subblock, matching output-memory, and K-block
sweeps.  Down explicitly tried non-power blocks 14, 28, and 56.  O tried its
maximum legal block 32; gate/up tried 32/64.  One-tile output partitions are
physical blockers (O/down need 128 workers and gate/up needs 112, versus 110
available).  `geometry_sweep.md` and `candidate_results.csv` contain every
configuration, PCC, and timing.

The follow-on QKV sweep found block 32 faster than the previous block 16.
Three 100/1000 candidate samples were 0.246738, 0.246784, and 0.246758 ms, so
block 32 was promoted.  Three runs through that first promoted `default` path
then produced this now-superseded QKV32/O8 control:

| final default | prefill ms | traced decode ms | PCC |
| --- | ---: | ---: | ---: |
| sample 1 | 0.703614 | 0.246749 | 0.9999998068156027 |
| sample 2 | 0.774450 | 0.246765 | 0.9999998068156027 |
| sample 3 | 0.809573 | 0.246752 | 0.9999998068156027 |
| median | 0.774450 | 0.246752 | 0.9999998068156027 |

Against the paired legacy median, this intermediate default was +0.17%
prefill (within the observed process noise) and -22.89% traced decode.

### Replay-only profile and gap audit

`_trace_latency` now emits nested replay-only signposts after eager warm-up and
trace construction.  The first replay-only QKV32/O8 capture used ten replays:

```bash
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_VARIANT=default \
MULTICHIP_DECODER_BATCH=1 MULTICHIP_DECODER_SEQ_LEN=18 \
MULTICHIP_DECODER_PREFILL_REPEATS=1 MULTICHIP_DECODER_TRACE_REPLAYS=10 \
timeout 300 python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/tracy_replay \
  -m pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k warmed_perf
```

The advice-enabled report filters `PERF_MULTI_DECODE_REPLAY` through its END
signpost.  It contains 2,364.770 us device time across ten replays = 236.477
us/replay, and the same run reports 275.304 us/replay E2E.  A targeted
unprofiled 10-replay control is 249.346 us, assigning 25.958 us/replay to
profiler instrumentation; an unprofiled 100-replay run reaches 247.157 us by
amortizing the final sync.  Replay already uses nonblocking `execute_trace`
plus one final sync, leaving about 10.680 us of TTNN enqueue/cross-run
accounting rather than a model host fallback.

A 100-replay profiler retry passed functionally but overflowed the profiler
DRAM marker buffers and dropped markers, so its device data was rejected.  Its
compact failure is in `logs/profiler_replay100_failure.md`; 3.2 GB of invalid
scratch output was deleted after the failure was preserved.  The valid raw ops
CSV is losslessly compressed under `tracy_replay/reports/`.  The exact final
QKV32/O32 profile below supersedes it for final numbers.

### Final post-repair gates

The final `throw_exception_on_fallback=true` suite passes 5/5.  It covers the
real TP4 path, 131072-token context accounting, 32-instance CCL pooling,
distinct-layer stacked execution, non-aligned seq 7, contiguous and adversarial
paged caches, positions 63/64/65, and trace replay.  After the final O32
promotion the suite was rerun: PCC includes prefill 0.9999993671, decode
0.9999886769364014, and stacked decode 0.9999549274032734.  Machine output is
retained in `logs/correctness_fallback.xml`.

The final watcher command is:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
RUN_MULTICHIP_DECODER_WATCHER=1 timeout 300 pytest \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k watcher_stress -q
```

It passes batch-32 paged BFP8 cache stress and ten trace replays at PCC 1.0.
The level-10 log is `logs/watcher_level10.xml`; only ACTIVE_ETH is disabled for
the already documented 27,920-B versus 25,600-B image-size blocker.
The final post-run `tt-smi -s` health check also reports all four P300c devices
with DRAM healthy, zero corrected and uncorrected GDDR errors, and no thermal
trip counts.

## Second independent-review repair cycle

The fresh rereview returned one `more-work-needed` finding: packing and the
near-winning non-QKV role candidates had been isolated under QKV16, so the
evidence did not yet prove the best coherent family under the actual QKV32
default.  Four cumulative variants were added and first screened at 20
prefill iterations / 500 trace replays.  All passed performance-output PCC at
or above 0.9999998068.  Every family was then repeated in three independent
processes at 100 / 1000:

```bash
for variant in cumulative_qkv32_packed_gate_up \
  cumulative_qkv32_output_k32 cumulative_qkv32_output_subblock1 \
  cumulative_qkv32_down_subblock1; do
  for run in 1 2 3; do
    TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
    RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_VARIANT="$variant" \
    MULTICHIP_DECODER_BATCH=1 MULTICHIP_DECODER_SEQ_LEN=18 \
    MULTICHIP_DECODER_PREFILL_REPEATS=100 \
    MULTICHIP_DECODER_TRACE_REPLAYS=1000 timeout 300 pytest -o addopts='' \
      --junitxml="models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/logs/candidate_${variant}_long${run}.xml" \
      models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
      -k warmed_perf -q
  done
done
```

| QKV32 coherent family | decode samples, ms | median | decision |
| --- | --- | ---: | --- |
| packed gate/up | 0.248682 / 0.248581 / 0.248601 | 0.248601 | reject |
| O block32 | 0.246689 / 0.246691 / 0.246665 | **0.246689** | promote |
| O subblock1 | 0.246817 / 0.246810 / 0.246818 | 0.246817 | reject |
| down subblock1 | 0.247242 / 0.247234 / 0.247238 | 0.247238 | reject |

O block32 was promoted into `MultiChipConfig`.  Six independent runs through
the exact resulting `default` path produced:

```bash
for run in 1 2 3 4 5 6; do
  TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_VARIANT=default \
  MULTICHIP_DECODER_BATCH=1 MULTICHIP_DECODER_SEQ_LEN=18 \
  MULTICHIP_DECODER_PREFILL_REPEATS=100 \
  MULTICHIP_DECODER_TRACE_REPLAYS=1000 timeout 300 pytest -o addopts='' \
    --junitxml="models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_multichip_decoder/logs/final_cumulative_default_long${run}.xml" \
    models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
    -k warmed_perf -q
done
```

| final default | prefill ms | traced decode ms | PCC |
| --- | ---: | ---: | ---: |
| sample 1 | 0.841712 | 0.246689 | 0.9999998070869076 |
| sample 2 | 0.828560 | 0.246689 | 0.9999998070869076 |
| sample 3 | 0.726828 | 0.246682 | 0.9999998070869076 |
| sample 4 | 0.748395 | 0.246753 | 0.9999998070869076 |
| sample 5 | 0.790864 | 0.246635 | 0.9999998070869076 |
| sample 6 | 0.797251 | 0.246679 | 0.9999998070869076 |
| median | 0.794058 | 0.246686 | 0.9999998070869076 |

Against the paired legacy median this is +2.70% prefill, inside the observed
large process-to-process spread for the decode-only policy change, and -22.91%
traced decode.  The final default passed the five-test
`throw_exception_on_fallback=true` suite again, then passed the level-10
watcher stress again.  Updated machine logs are `correctness_fallback.xml` and
`watcher_level10.xml`.

The exact-final replay profile used the same command above with output changed
to `tracy_replay_final`.  Its advice-enabled filtered report contains 360 ops
and 2,367.64 us across ten replays = 236.764 us/replay.  The raw final-profile
CSV retains signpost timestamps 11,428,587,354 through 11,431,500,008 ns,
which bracket the ten profiled replays at 291.265 us/replay.  Exact-default
unprofiled controls are 248.833 us at ten replays and 246.850 us at 100
replays, leaving 10.086 us/replay after assigning 42.432 us to profiler plus
signpost instrumentation and 1.983 us to final-sync amortization.  The
explicitly named `logs/replay_only_profile_qkv32_o8.xml` belongs to the preceding QKV32/O8
capture and is not used for exact-final E2E accounting.  The retained
raw CSV SHA-256 is
`7a18efa2929df638e6d2e1f95874ca83f23072b78da0c5740024602dec7c1001`.
The generated 482-MB `.logs`, duplicated device-marker CSV, and host trace
scratch were removed after the compressed raw ops CSV and all filtered report
artifacts were preserved; those scratch copies are not recoverable.

## Review and commit

The final fresh independent rereview returned `clean-pass` with no required
work, other concerns, or hard-check gaps.  Its verdict and the preceding
repair cycles are summarized in `stage_review.md`.  The stage-owned local
commit SHA is appended immediately after checkpoint creation.  No push is
performed.
