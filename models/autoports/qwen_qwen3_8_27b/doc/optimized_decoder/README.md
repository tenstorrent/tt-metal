# Qwen3.8-27B optimized decoder (stage03)

Stage03 complete. Final hardware validation, profiling and independent [stage review](stage_review.md) pass (`clean-pass`).

`tt/optimized_decoder.py` implements the single-device layer directly. The default constructor uses `p300c_bfp4_lofi_phase_specific_v1`; tests import `OptimizedDecoder`, and runtime never calls the functional or fused classes. The explicit test-only `--baseline-fused` switch produces controls. This stage contains no TT multichip, full-model or vLLM implementation.

## Hardware and workload

One physical device0, a Blackhole p300c ASIC with8 DRAM banks and110 available storage/compute cores; firmware19.8.1, KMD2.10.0, clock1350MHz in profiler headers. Four ASICs were available at startup, but only one participates in these measurements. Allocator DRAM capacity is34178731008 bytes. The theoretical whole-ASIC denominator uses120 physical cores and512GB/s GDDR6, not the storage-core count displayed on individual matmul rows.

Headline:4096 input positions,128 traced decode steps,B1,one request. This is a single-layer supplied-hidden-activation workload, not end-to-end token generation. Layer0 is `linear_attention`; layer3 is `full_attention`. Timings never average the two kinds. Final decode is the128th step, at context4224. HF comparisons cover initial decode, changed input/position and the final128th output. Reported device time is the complete firmware timestamp envelope, including normalization, attention, movement, all other operations and internal gaps. Host wall times have separate names.

## Selected policy

`selected_policy.json` and the constructor constant are the canonical configuration. Nonempty `policy` arguments replace it for explicit experiments; arbitrary experimental configurations are not advertised as validated defaults.

| Tensor/operation | Selected configuration |
| --- | --- |
| Attention projections; gate/up; down | BFP4 weights, BF16 activations/output, LoFi, FP16 destination accumulation, packer L1 accumulation |
| Norms, residual, convolution history | BF16; sensitive norm/attention operations use HiFi4/FP32 accumulation |
| Linear recurrence | FP32 state and arithmetic; final-only in-place update for the supported per-core head assignment |
| Full KV cache | BFP8; prefill fill tensors explicitly cast to cache dtype, decode updates remain BF16 |
| Decode mix and packed gate/up |80 input storage cores, K block2; two readers per each of8 DRAM banks |
| Decode output / down |64 cores/K3;68 cores/K8; two readers per bank for each role |
| Decode residual/norm | L1 width-sharded residual at B1;8-core norm; L1 interleaved mixer; packed gate/up output retains L1 sharding at B1; multiply output is L1 interleaved |
| Batched MLP decode | Gateup/product `[1,B,N]` until down projection, avoiding32 padded rows per user; public output remains `[B,1,H]` |
| Prefill |1280-token linear-attention /1024-token full-attention internal chunks,11x10 2D matmul grids; K10 mix/gate/up and K32 output/down; output subblock height2 when legal; original width1 becomes width5(linear)/3(full) when divisible, with legal height adjustment |
| Prefill MLP | Separate gate-with-SiLU and up; packed gate/up only for decode. Both weight representations are resident |
| Prefill norm | Complete rectangular block sharding, up to8x8 and1280/1024 logical token rows respectively; internal row padding for awkward tails; interleaved for larger token batches |
| Full prefill SDPA | Approximate exp, query128/key256 blocks; internal compatible chunk selection for tails and128-aligned continuation |
| Full decode SDPA | Key chunk256, maximum32 cores/head/batch, exact exp, HiFi4/FP32 |

Two-reader settings are explicit per role in this model policy. The generic primitive default remains one. Weight padding is explicit and zero-valued; padded projection channels are sliced before affecting outputs. DRAM reader/compute cores (16) are distinct from input storage cores (80/64/68).

The norm's nonrectangular peer-gather defect was isolated with AutoDebug. The decoder uses complete rectangles; no shared upstream kernel was changed. `AUTODEBUG_prefill_norm.md`, `AUTOFIX.md`, and `prefill_norm_probe.json` record the diagnosis, discriminating experiment and fix. Invalid original norms produced enormous finite values; FP32 statistics overflowed, so the probe now uses FP64 statistics. This was not a precision-policy rejection.

## API, state and context

Prefill takes `[B,S,5120]` and returns every logical output; decode takes `[B,1,5120]`. All positive valid logical lengths are supported, including1/2/3/4,31/32/33,127/128/129,511/512/513 and262142. Internal per-kind chunks do not impose a public divisibility constraint. Tests also cover1279/1280/1281 (linear),1023/1024/1025 (full), and odd-batch norm cutoffs426/427 and341/342. The inherited continuation start contract is128-aligned, independent of logical prompt length; state snapshots and resume at128 are tested.

Full attention uses paged `[physical_pages,4,32,256]` K/V and caller-supplied absolute-position RoPE tensors. Tests use randomized physical page ownership, change ownership under a live decode trace, and check output plus exact optimized-cache equivalence. Raw cache PCC against BF16 HF is diagnostic: it is not substituted for model-output correctness when BFP4 projection weights legitimately change cache values.

Linear state is row-major BF16 `[B,3,10240]` history and tiled FP32 `[B,48,128,128]` recurrence:3207168 bytes/user. Decode history requires one untilize to the fused convolution's row-major contract. For aligned prefill chunks, only the final32-row tile is converted before selecting the last3 logical history rows. Tiny/unaligned cases retain correct logical history handling.

Context remains262144; no capability reduction. At that context, one full-attention user's BFP8 KV occupies570425344 bytes, versus1073741824 with BF16. BFP8 storage includes exponent headers (2176 bytes/token for both K/V). The phase-specific MLP adds100270080 resident bytes versus packed-only; allocated decoder setup (weights plus constants and chunk-position tensors) is recorded per layer in the final JSONs and context contract. At the planning workload of32 users×32768 tokens, one full-attention layer needs2281701376 KV bytes; one linear layer needs102629376 recurrent/history bytes for32 users. These are layer-local estimates, not a full-model serving capacity claim. Memory snapshots cover decoder construction and request setup before output allocation, decode backups and trace capture; they are not peak-memory or trace/output accounting.

Long-capacity fixtures repeat recorded HF activations. TT computes every output; the long HF reference constructs state from the entire prefix and compares the final128 prefill positions plus the final two decode positions. This establishes capacity and layer semantics, not long-conversation full-model accuracy.

## Evidence and reproduction

Commands run in the existing `qwen38-functional-build` container, whose `/workspace` is this checkout. Hardware operations are serialized. Watcher is disabled for every profiler run; correctness uses `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, with no allocation exemptions.

```bash
# CPU fixture preparation from the locally available checkpoint/tokenizer:
python3 models/autoports/qwen_qwen3_8_27b/tests/prepare_optimized_weights.py /data/mvasiljev/weights/Qwen3.8-27B
docker exec qwen38-functional-build python -m models.autoports.qwen_qwen3_8_27b.tests.record_optimized_inputs --token-ids-only
python3 models/autoports/qwen_qwen3_8_27b/tests/prepare_optimized_weights.py /data/mvasiljev/weights/Qwen3.8-27B
docker exec qwen38-functional-build python -m models.autoports.qwen_qwen3_8_27b.tests.record_optimized_inputs

# Complete default-path correctness, then final profiles:
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/final_hardware.sh

# One explicit same-harness control:
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_run.sh profile baseline_accepted_l0 --layer 0 --recorded --baseline-fused --trace-prefill

# Matched final-default4096/fused-default513 screens, reader profiles and exact-runner fused baselines:
docker exec qwen38-functional-build python -m models.autoports.qwen_qwen3_8_27b.tests.final_comparisons

# Alternating one/two/three-reader role profiles (final default geometry):
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/profile_readers.sh accepted
```

`validate_optimized.sh` runs pytest odd-batch/tail/continuation/remap coverage; ragged full attention with8 distinct positions/pages; separate watcher4096+128 runs for both kinds; B8/B16/B32 workloads; exact262144 prefill and262142+two final decodes. Three identical restored-state trace replays are bitwise compared. Warmed prefill is bitwise compared with initial prefill, including traced replay. Larger-batch timing also performs100 warmed replays. `NoTorchRuntime` and patched TTNN conversion functions reject any torch/from_torch/to_torch call within audited forwards.

`record_optimized_inputs.py` uses real embedding rows for a repeated task-oriented text fixture and CPU HF preceding layers0..2 for the layer3 fixture. `input_std` in result JSON is the synthetic-input argument and is not applied when `activations` records HF inputs. Raw tensors/weights remain in ignored `.cache/qwen38-weights`; no tensors or credentials enter telemetry. A Transformers CPU-reference fallback warning refers to missing optional FLA kernels, not TTNN runtime fallback.

`collect_optimized_perf.py` invokes advice-enabled `tt-perf-report` separately for prefill and decode. Human tables are `*_perf_report.txt`; CSVs are `*_perf_report.csv`. Complete-window timestamps are `whole_layer_timing.json`. Raw large CSVs are compressed locally after extraction; compact reports and JSONs are committed. The Tracy WASM copy warning does not invalidate device CSVs; no web UI is started.

## Search, rejected choices and remaining boundaries

The initial topology audit precedes experiments in `work_log.md`. `screen_results.csv` and every candidate JSON/policy retain concrete measurements; sweep scripts preserve commands. Host screens are not mislabeled as device performance.

- Real recorded activations support BFP4/LoFi attention and all MLP weights above .995. Random-input failures were investigated through recorded inputs and did not veto real-weight wins. BFP8/HiFi2, BFP8/LoFi, per-group BFP4 and per-group fidelity candidates are retained.
- Packed decode beats tuned legal separate gate/up at the same precision and material80/40-core layouts. Separate mixer projections lose. Prefill chooses the faster separate MLP, accepting the measured resident-weight cost.
- Reader counts1/2/3 were crossed with8/16/32/40/80-scale input geometries and repeated in alternating role microbenchmarks. Those report complete role device time, physical/logical bandwidth, RISC durations, exact replay and reader agreement. Final full-layer integration is the deciding metric.
- K2 on80-core mix/gateup is the largest input-shard divisor. K4/5/10 on smaller grids and output/down K6/16/17/32/68 candidates were measured or have exact L1 failures. Legal larger blocks were not dismissed by the first failure. `role_*` isolates each material role. DRAM-sharded output subblocks are selected internally by this TTNN factory; the public config exposes no output-subblock override.
- Wider decode norm grids are slower. L1 intermediates win; packed-sharded gate/up output has a small consistent gain in the final paired test and is retained. BF8 activation conversions lose. Minimal-matmul fused SwiGLU was adapted at multiple block sizes, passes but is slower and lowers PCC; it is not selected.
- Large-prefill grids/chunks, L1 input, subblocks and K blocks were measured. Initial packed K8 L1 failure was adapted, and separate-prefill weights permit wider legal final K blocks. SDPA exact/approximate-exp candidates are compared separately. Traced prefill removes host dispatch gaps.
- Remaining conversions serve explicit consumers: row-major convolution history; width-sharded projection inputs; interleaved QKV/composite consumers; height-sharded paged-cache updates; public logical shape/tail handling. The residual is kept sharded where its consumers benefit. No CCL, MoE, logits, sampling or LM head exists in this dense single-layer scope.

The completed evidence mapping is in [checklist.md](checklist.md); independent signoff is in [stage_review.md](stage_review.md). Full-model quality, token-out generation and serving performance are not claimed by this layer stage.

## Final measurements

All headline device values use4096+128,B1,one request. Every layer kind is reported separately. `performance.json` contains formulas, useful FLOPs/estimated bytes, theoretical peaks and exact source references. Lower precision explains the small PCC change; all results exceed the functional .995 bar.

| Layer kind | Fused prefill us | Final prefill us | Speedup | Fused decode us | Final decode us | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|linear_attention|82204.06|16057.13|5.12x|2324.99|815.30|2.85x|
|full_attention|92975.46|17831.05|5.21x|2188.54|704.05|3.11x|

| Layer kind | Fused / final prefill PCC | Fused / final initial decode PCC | Fused / final128th decode PCC |
| --- | ---: | ---: | ---: |
|linear_attention|0.99999675 / 0.99980716|0.99999921 / 0.99981594|0.99999724 / 0.99982184|
|full_attention|0.99999493 / 0.99986445|0.99999938 / 0.99959877|0.99999933 / 0.99963450|

| Layer kind | Ideal compute prefill us | Actual device prefill us | Host prefill us | Prefill chip FLOPs % | Ideal DRAM decode us | Actual device decode us | Host last decode us | Decode DRAM % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|linear_attention|4766.83|16057.13|17127.32|29.69|435.93|815.30|837.14|53.47|
|full_attention|4909.23|17831.05|18478.79|27.53|426.96|704.05|722.73|60.64|

The ideal times are lower bounds, not mixed-kernel attainable ceilings. DRAM estimates count physical BFP4 weight tiles/headers/padding and state/KV traffic, omitting spills and rereads. Device gaps remain inside the denominator. Host values here are from the same profiler-enabled run; profiler-off screens below provide selection comparisons. Trace replay removes per-op host dispatch; remaining firmware gaps and single replay synchronization are included, not subtracted.

| Layer | Length | Fused prefill / decode host us | Final prefill / decode host us |
| --- | ---: | ---: | ---: |
|linear_attention|513|12914.56 / 2317.01|3001.62 / 794.99|
|linear_attention|4096|See device baseline above|16955.36 / 795.19|
|full_attention|513|12232.90 / 2146.77|2755.47 / 670.38|
|full_attention|4096|See device baseline above|18291.67 / 688.60|

These profiler-off decode screens use100 static-position replays after one initial restore for candidate comparison. The headline uses the valid advancing128-step workload. Larger-batch values below use its final valid step, not the static-position diagnostic loop.

| Layer kind | Batch | Prefill PCC | Final128th decode PCC | Final valid-step host us |
| --- | ---: | ---: | ---: | ---: |
|linear_attention|8|0.99981065|0.99984509|2283.34|
|linear_attention|16|0.99980923|0.99983479|3822.00|
|linear_attention|32|0.99980924|0.99982258|6915.85|
|full_attention|8|0.99987515|0.99984434|1324.42|
|full_attention|16|0.99987140|0.99976380|1991.90|
|full_attention|32|0.99987388|0.99982765|3266.98|

## Dominant matmul search

Role shares below are kernel time divided by the final complete layer device window; layout and internal gaps are outside the numerator but inside the denominator. Search latencies are whole-layer traced host screens at4096. Exact per-role configs/PCC and other precision/topology candidates are in `screen_results.csv` and matching JSON/policy files. Historical role rows are the precision-locked BFP4/LoFi geometry sweep before the final paired packed-output layout refinement; only the named role changes within that sweep. Final shares and accepted-default metrics come from the new final profiles.

| Kind | Role | Final layer time % | Config candidate | Traced decode us | Minimum output PCC | Decision |
| --- | --- | ---: | --- | ---: | ---: | --- |
|linear_attention|mix|13.09|control|797.33|0.99982408|Keep|
|linear_attention|mix|13.09|mix40k4|816.27|0.99982040|Reject: slower|
|linear_attention|out|6.84|control|797.33|0.99982408|Keep|
|linear_attention|out|6.84|out32k6|802.33|0.99979927|Reject: slower|
|linear_attention|gateup|26.59|control|797.33|0.99982408|Keep|
|linear_attention|gateup|26.59|gateup40k4|840.73|0.99982182|Reject: slower|
|linear_attention|gateup|26.59|gateup16k10|—|—|Reject: L1 capacity; legal smaller alternatives measured|
|linear_attention|down|12.95|control|797.33|0.99982408|Keep|
|linear_attention|down|12.95|down34k16|850.05|0.99981934|Reject: slower|
|linear_attention|down|12.95|down17k32|852.93|0.99981047|Reject: slower|
|linear_attention|down|12.95|down8k68|—|—|Reject: L1 capacity; legal smaller alternatives measured|
|full_attention|mix|12.79|control|690.18|0.99954898|Keep|
|full_attention|mix|12.79|mix40k4|707.62|0.99956438|Reject: slower|
|full_attention|out|7.92|control|690.18|0.99954898|Keep|
|full_attention|out|7.92|out32k6|695.45|0.99955198|Reject: slower|
|full_attention|gateup|31.18|control|690.18|0.99954898|Keep|
|full_attention|gateup|31.18|gateup40k4|733.08|0.99954952|Reject: slower|
|full_attention|gateup|31.18|gateup16k10|—|—|Reject: L1 capacity; legal smaller alternatives measured|
|full_attention|down|14.98|control|690.18|0.99954898|Keep|
|full_attention|down|14.98|down34k16|744.15|0.99955263|Reject: slower|
|full_attention|down|14.98|down17k32|748.12|0.99955699|Reject: slower|
|full_attention|down|14.98|down8k68|—|—|Reject: L1 capacity; legal smaller alternatives measured|

Final mix/gateup remain K2 on80 input-storage cores: two tiles/shard is the largest divisor. The crossed16/32/40/80-core1/2/3-reader search and isolated40/K4,16/K10 trials earn this choice. Output64/K3 beats32/K6; down68/K8 beats34/K16 and17/K32;8/K68 fails by12928 L1 bytes. The DRAM factory chooses output subblocks internally; its public program config exposes no override. Prefill has explicit subblock height2 where legal, wider K10/K32, and11x10 grids.

Final dtype is also verified directly above from measured CSV columns. `readers_accepted_l0.device.json` and `readers_accepted_l3.device.json` recheck one/two/three readers per role on the selected policy, in alternating order, with complete role envelopes and physical byte estimates.
