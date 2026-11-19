# AutoFix: device-only eval top-k 40

Status at 2026-11-17 05:22 UTC: implementation applied after the previous
server and baseline probe closed. Native build, focused native/device checks,
and Qwen sampler lifecycle pass. Reduced-model greedy and eval-profile watcher checks pass; final served gates
remain pending; no final serving acceptance is claimed.

The required `lm_eval_reasoning` profile uses temperature1, top-p0.95, top-k40,
seed42. Qwen advertised32, the common formatter clamped32, and native sampling
retained/scanned32. The plugin correctly chose host sampling. The diagnosis is
[AUTODEBUG_eval_sampling.md](AUTODEBUG_eval_sampling.md), obtained by fresh
source-only AutoDebug. No plugin or vLLM change is needed for this repair.

Selected hypothesis: an opt-in native `input_sorted=True` path accepts64 paired,
descending candidates and applies actual logical k40 before softmax/nucleus.
The common sampler performs local64, gathers candidates, corrects greedy ties,
selects global64 and gathers exact UINT32 token IDs. Native sampling bypasses
the fixed32 sorter, processes two probability tiles and reads paired IDs
directly. Its default sorter/writer/RNG branch is preserved. Public Qwen support
will be40; physical capacity64 does not advertise untested additional requests.

Rejected as incorrect: raising only the advertised limit, clamping40to32,
widening only local candidates, or changing one native constant. Replacing the
entire RNG/draw with generic device operations would add unnecessary contracts.
The optional global topk-with-labels route could remove the integer gather, but
is not selected without exact high-ID hardware evidence.

One common sampler owns both persistent candidate-offset widths and the shared
seed/token sinks. Width joins trace identity. Qwen retains authoritative host
slot parameters and installs neutral k1/p0/temp1 for inactive physical rows.
Updates are deferred until actual sampling membership is known; unchanged
membership/parameters cause no upload. Both families are warmed before model
or sampling capture. Width changes do not reset request counters. Final replay
remains CQ0/nonblocking; only diagnostic token reads wait for completion.

Scratch components:

- `/tmp/qwen-stage10-presorted-native/presorted64.patch`: native API, attributes,
  validation, factory and three kernels. Static proof checked2048 tile
  coordinates and43008 scan cases; original default branches compared.
- `/tmp/qwen-stage10-wide-common/wide_common.patch`: common formatter, capacity,
  persistent offsets, graph selection, trace identity. CPU evidence:33 formatter
  cases,9 width transitions, identical17-call mocked narrow dispatch.
- `/tmp/qwen-stage10-wide-model/models/autoports/qwen_qwen3_8_flash_next/tt/generator.py`:
  public40 guard, active-slot parameter normalization and both-family warmup.
  Scratch AST tests:9 passed,16 subtests; the unrelated relative-import helper
  test is deferred until the live patch is applied.

The first scratch CPU test run had one fixture error: omitted required top_p
when constructing SamplingParams. The corrected fixture passed; no production
assertion was relaxed. The native statistical probe's TV threshold was corrected
before running: finite-sample TV scales with support size. It uses marginal
variance plus a family-wise bounded-difference term, with a one-candidate BF16
nucleus-edge allowance; exact top-k support, high-ID pairing, second-tile
reachability and deterministic checks remain separate assertions.

Required discriminating evidence, pending hardware availability:

1. Build native/API changes, then test integer global top64 pairing on cache
   hits and changed-input trace replay; IDs exceed65535.
2. Native H1/H32, every admitted k33..40 plus boundaries1/16/17/31/32/48/64;
   p0/p1/p0.95, second-tile support/distribution, excluded-tail masking before
   normalization, changed persistent k/IDs.
3. Actual Qwen `_sample` seeded lifecycle, capture/replay, pause/remap/prefill,
   narrow→wide→narrow transitions, stale inactive k40, same-schedule reset
   reproduction and full TP4 seed-state equality.
4. Full-watcher reduced-model immediate/queued async histories for greedy and
   exact eval profile, followed by real-vLLM mixed traffic, sampling, quality,
   latency, no-host-fallback and eval-load gates.

All device work remains serialized. The current sweep first measures256/33/17
on the existing greedy path. Final-source selected-budget evidence must be
rechecked after sampler/trace changes; existing results are not silently
relabeled as final implementation acceptance.

## Applied repair and focused results

Applied the four preserved patches in `repair_eval_sampler/` with devices free.
`cmake --build build_Release --target ttnn --parallel 8` passed; installed
metalium-runtime, ttnn-runtime and tt_pybinds locally. Source/binary identity is
[wide_sampler_build.json](wide_sampler_build.json). No plugin/vLLM source or
default changed.

- [wide_native_v2.json](wide_native_v2.json): PASS. Exact global64 value/UINT32
  pairing in eager and changed-input replay, native H1/H32, k33..40 and physical
  boundary cases, p0/1/0.95 distributions over1024 draws per physical row,
  second-tile support, excluded-tail output invariance, changed k/ID inputs.
- [narrow_sampler_regression.json](narrow_sampler_regression.json): PASS. All
  token/transition records exactly match capacity32 before versus production
  capacity64 after, logical k32, synthetic vocabulary248320, TP4, three active
  slots,41 logical sampling replays.
- [wide_sampler_lifecycle_v2.json](wide_sampler_lifecycle_v2.json): PASS. Actual
  Qwen `_sample`, k40 with best40 concentrated on one shard, pause/remap/prefill,
  width32→64→32, dormant k40 neutralization, persistent parameters, repeated
  schedule seed/token equality,51 sampling replays and6 total traces.
- Model host tests:47 passed,55 subtests. Common live host tests:123 passed, including six rejected-reset cases that
  failed before moving width selection after validation. All source hooks pass.

Preserved failed diagnostic invocations:

- narrow_sampler_before_v1 omitted token_feedback.prepare in its setup. Added
  the same preparation used by production; before_v2 passes.
- wide_native_v1 reached every required second-tile k in p1, but its p0.95
  reachability check incorrectly required an uncertain nucleus-edge candidate.
  For k34, host BF16 cumulative mass through32 is0.9481201171875 versus
  quantized p0.94921875: only0.0010986 separates the edge. The distribution
  oracle already permits one BF16 cutoff candidate; reachability now applies
  the same allowance for p<1, while p1/k33 still requires candidate33. No
  native source changed. Corrected v2 passes all checks.
- wide_sampler_lifecycle_v1 used PyTorch.max on UINT32, unsupported on the
  host. Cast the already-read diagnostic parameter vector to int64; v2 passes.
- Existing prefill lifetime fixture omitted sample_slots/sample_mask captured
  from the real enclosing function. Supplied that fixture metadata. A quiet
  pytest invocation also omitted the repo's importlib mode, causing one
  relative-import setup error. The final invocation explicitly restores it.

## Chunk sweep configuration correction

The original alignment128/budget256 run completed4096/32768 input prompts with
worst ITL19680.6/19980.6ms, respectively, one concurrent decoder, resident2,
context262144. It fails the1000ms bound. Budget33 then failed startup because
the existing plugin prefill_chunk_alignment default128 exceeds33. Devices
closed and no owned API/EngineCore process remained. These are preserved in
`chunk_sweep_alignment128_execution.json`, `chunk256_alignment128.json` and
`readiness_vllm/stage10_chunk_33/server.log`.

The corrected launch and benchmark config explicitly use the existing
`prefill_chunk_alignment=0` option. This permits non-aligned arbitrary chunks;
it introduces no plugin interface or default change. Rerun at33/17/9 with
new evidence directories after the sampler repair checks. Final matched
performance arms will both use the selected budget and alignment0. The old
alignment128 sweep result was moved out of the active P5 sweep directory so
it cannot be silently combined with the new regime.

05:30 UTC: both full-watcher immediate/queued async controls pass (greedy and
eval k40); device histories, RNG and final state agree. Invalid reset preserves
width/offset/trace family. Patch and123-test log retained in repair_eval_sampler.
