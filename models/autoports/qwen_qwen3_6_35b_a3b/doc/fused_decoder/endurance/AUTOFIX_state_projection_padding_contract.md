# Ordered K/V/A/B projection padding: full-chain contract

Source/CPU inspection only. No production edits, test edits, native edits, TTNN imports or device runs. Smaller tiles are blocked. The existing scalar-broadcast layout is already minimally padded; a different partial-lane packing can reduce payload, but requires an additional adapter and separate gate reduction. It is not a demonstrated speedup.

## Why smaller tiles and a different rectangle do not work

The current logical 6,208 columns are K2,048 + V4,096 + A32 + B32. A partial uses a full-tile plane of 32×224 =7,168 lanes. It therefore occupies seven 1,024-element tiles, the minimum possible for a plane containing 6,208 independent output lanes. The excess is 960 lanes per term: 15.46% over useful data, or 13.39% of the current product payload. All 2,048 input terms together produce 56 MiB of FP32 data versus an ideal 48.5 MiB without tile padding.

- `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:480–501` explicitly requires **32×32** tiles for both tiled inputs and any preallocated output. This applies to mixed dtype and accurate SFPU multiply too. 16×32, 8×32, 32×16, or 32×8 cannot reach this producer.
- FastReduce independently uses fixed full-tile geometry: `experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp:48–64` divides H/W by 32 and constructs full-plane reader strides; `:101–112` uses full-format tile byte sizes and divides physical output volume by 1,024. `fast_reduce_nc_device_operation.cpp:121–130` produces default TILE pages. A smaller preallocated output does not change those input reader/CB assumptions.
- FastReduce accepts only outer reduction axes (`fast_reduce_nc_device_operation.cpp:74–84`). Moving K into H/W to share padding would require a different reduction, changing the ordered FP32 fold or violating this API.
- H=16 with ordinary tile metadata still pads H to32. `[194,32]` or `[32,194]` both pad to seven full tiles. Splitting K/V and A/B while keeping separate outer partial axes still consumes six tiles plus one tile per partial; it saves nothing.
- A row-major multiply can reduce producer padding, but FastReduce reads tiled pages. The required tilization restores full-tile planes. Mixing RM coefficients with TILE weights does not avoid coefficient tilization: public binary dispatch calls `to_layout(..., TILE)` on both inputs in that case (`eltwise/binary/binary.cpp:845–851`). Both-RM dispatch is supported, but native RM mode requires RM output (`binary_ng_program_factory.cpp:1288–1292`); it does not fuse directly into a tiled FastReduce input.

These are contract blockers, not reasons to attempt a custom-tile device probe.

## Supported alternate packing, if a larger structural control is wanted

The one concrete full-tile alternative is to separate K/V and A/B and **share one A/B tile among the independent partial sums**. This avoids the otherwise almost-empty A/B tile per partial. It preserves the ordered K fold and the subsequent ordered partial fold. No custom tile is needed.

Let `B=2048/K`, with `(B,K)=(4,512)` for decode or `(2,1024)` for the long-prefill policy. Start from the same pair-permuted checkpoint matrix `W` and pair-permuted normalized input coefficients used today.

Setup-only weights:

```python
kv = W[:, :6144].reshape(B, K, 32, 192)  # six tiles per partial
ab = W[:, 6144:6208].reshape(B, K, 64)
ab = ab.permute(1, 0, 2).contiguous().reshape(K, 2 * B, 32)
# Upload both as ordinary BF16 TILE weights; ab has physical H32.
```

Runtime sequence for one token, using only existing public operations:

1. Preserve the existing scalar coefficients `[B,K,1,1]` for K/V and its accurate BF16×BF16→FP32 multiply. Reduce with the same `fast_reduce_nc(..., dims=[0,1], compute_kernel_config=self.compute)`; cast to BF16 only after both FP32 reductions. Reshape to 6,144 channels.
2. Use the existing **RM pair-ordered input** before its scalar-coefficient tilization. RM reshape to `[1,B,K]`, transpose the last two dimensions to `[1,K,B]`, reshape to `[K,B,1]`, repeat the last dimension twice to `[K,B,2]`, then RM reshape to `[K,2*B,1]` and tilize. Each coefficient now occupies its partial's two A/B rows. RM repeat supports BF16 and last-dimension repeats (`data_movement/repeat/device/repeat_device_operation.cpp:33–46`); the final operands both use ordinary tiles.
3. Multiply `ab [K,2*B,32]` by coefficients `[K,2*B,1]` with explicit `dtype=FP32, fast_and_approximate_mode=False`. Native binary column broadcast supports this shape (`binary_ng_device_operation.cpp:217–246`); padding is only in H and does not mix valid rows.
4. FastReduce dimension0 gives FP32 `[1,32,32]`, whose first `2*B` rows contain the independent partials in block order. Untilize this small result, crop to `[1,2*B,32]`, RM reshape to `[B,2,32]`, and tilize. Then FastReduce dimension0 again. This is the same left-associated sum of B partials, not a spatial reduction. Keep FP32 through both stages; only then cast BF16. RM crop/reshape the first two rows into 64 channels, tilize, and concatenate after K/V.

The explicit small RM repack is necessary: interpreting the first-stage physical 32-row tensor directly as `[B,2,32]` would mix padding with the independent partials. For an initial discriminator, keep this candidate decode-only; leave the current prefill path unchanged rather than adding token-batch packing before a benefit is established.

The native reduction wrapper sorts multiple dims and reduces them in descending order (`fast_reduce_nc.cpp:38–62`). Its FP32 compute loop copies the first term and sequentially adds every later term (`device/kernels/reduce_nc.cpp:42–59`). Thus the candidate K/V path and two explicit A/B reductions preserve the original arithmetic association. CPU checks for both K512/B4 and K1024/B2 mapped synthetic BF16 products through these exact ordered loops and found all 64 FP32 A/B codes identical. This proves indexing and association, not device correctness.

| Policy | Current FP32 products | Packed K/V plus A/B products | Product saving | Additional tiled A/B coefficients |
|---|---:|---:|---:|---:|
|K512, B4|56 MiB|48 + 2 =50 MiB|6 MiB (10.71%)|1 MiB BF16|
|K1024, B2|56 MiB|48 + 4 =52 MiB|4 MiB (7.14%)|2 MiB BF16|

These are physical payload sizes, not measured transfers. Additional coefficient tilization, two A/B reductions, small layout conversions and output concat all count against the saving. K/V keeps 24 (or12) first-stage output tile owners; compact A/B has one owner scanning K terms. Its separate dispatch cannot overlap automatically with K/V. The current evidence therefore **does not justify implementing this during the final queue**. If pursued later, an isolated complete projection probe must compare BF16 outputs and both FP32 partial stages, include all coefficient/layout/concat operations in traced latency, and then pass the full seeded decoder gates. A multiply-only bandwidth result would be insufficient.

No smaller-tile or zero-overhead padding-removal candidate is supported by the existing producer/consumer chain. The supported split A/B packing above is a materially larger control with a bounded, modest payload benefit and substantial extra dispatch risk.

## Prepared isolated control

At the root owner's request, the supported A/B packing is now implemented in [packed_ab_partials.py.snapshot](packed_ab_partials.py.snapshot), against immutable [packed_ab_baseline.py.snapshot](packed_ab_baseline.py.snapshot). Production is untouched. Baseline SHA256 is `61ac211a397a4ab24737581b931ea318a4ed83d2e32d8eff67a4f26acda0a178`; candidate SHA256 is `7b089e25007b002bd574a4d0214b29ccdbb311be3ffc527eb662a4d2e240f974`.

The candidate prepares both K512/B4 and K1024/B2 weight layouts. Singleton projection respects the existing `projection_k` argument, including internal prefill singleton K1024. Multirow projection retains 128-row chunking and K1024 for every chunk, including the final one-row internal tail. Only the ordered state-projection setup and its linear branch change; the rest of the frozen decoder is retained. K/V reductions remain two stages through the existing multidimension wrapper. Packed A/B performs the same K fold and block fold explicitly, with FP32 retained between them. BF16 packing occurs only after the complete sum.

[packed_ab_projection_probe.py](../../../tests/packed_ab_projection_probe.py) performs lightweight setup of real checkpoint K/V/A/B weights and calls the **actual `linear` methods from both snapshots**. Default cases are S1/K512, S1/K1024, S2, S33, S128 and S129; all multirow cases use K1024, and S129 exercises an internal one-row tail. Each compares all 6,208 output codes at both FP32 and BF16 boundaries, with separate K/V and A/B summaries. Inputs are deterministic normalized synthetic BF16 rows, explicitly not claimed to be captured activations. Weight/input/source/snapshot hashes and standard provenance are recorded.

All pair gathers, coefficient creation/repetition/tilization, reductions, output repacks and concatenations remain inside the traced `linear` call. No adapter is precomputed or removed from timing. Timings use 3 warmups, 15 samples and 5 replays per sample by default. A shape whose old/new codes differ is recorded and not timed. The diagnostic neither promotes a candidate nor replaces model-level endurance validation.

CPU setup passed ordered A/B checks using the real checkpoint weights for both K policies. A symbolic execution of the candidate's actual methods, modeling logical reshape/broadcast and FastReduce's padded output dimensions, also passed S1/K512, S1/K1024, S2/S33/S128/S129/S257. This catches adapter shape and tail indexing errors; it does not validate native kernels or timing. The test was formatted with Black23.10.1/isort5.13.2. No device result was available at preparation; subsequent root-owned results follow below.

Root-owned probe command, under the normal exclusive-device wrapper:

```bash
timeout 1200 env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.packed_ab_projection_probe
```

The output is `endurance/packed_ab_projection_probe.json` plus provenance. `--lengths 1` provides a bounded initial S1/K512 and S1/K1024 check; it does not cover multirow or tail behavior. Full default coverage remains required before drawing a candidate decision. `--oracle-only --output /tmp/packed_ab_projection_cpu.json` loads only CPU weights and runs mapping checks without importing TTNN.

Matched complete-decoder benchmarks, including every adapter inside the decoder:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/packed_ab_baseline.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/packed_ab_baseline_s256.json

env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/packed_ab_partials.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/packed_ab_partials_s256.json
```

The root-owned full-path measurements below complete the performance decision. A winning isolated projection latency still needs a whole-decoder improvement and the unchanged seeded endurance, continuation, determinism and final-reference gates.

## Root-owned isolated device result

[packed_ab_projection_probe.json](packed_ab_projection_probe.json), with [provenance](packed_ab_projection_probe.json.provenance.json), reports **`device_exact=true`**. All six cases have zero old/new code differences across the complete 6,208-channel projection at both FP32 and BF16 boundaries, including separate K/V and A/B comparisons. Both first and last timed replay outputs are exact. This covers K512 singleton, K1024 singleton, multirow inputs and the S129 internal tail.

Complete projection adapter medians, including every coefficient, gather, layout, reduction and concat operation, from 15 samples with 5 replays each:

| Rows | K | Original (ms) | Packed A/B (ms) |
|---:|---:|---:|---:|
|1|512|0.308488|0.532975|
|1|1024|0.445225|0.838966|
|2|1024|0.994403|1.357816|
|33|1024|14.486368|15.640972|
|128|1024|55.652020|60.090735|
|129|1024|56.421543|61.209566|

The candidate is slower at every tested shape despite reducing product payload. These measured complete-adapter results support rejection on performance; the subsequent full-decoder result below confirms that decision. No production adoption, endurance completion or final stage closure is claimed. The smaller-tile native blockers remain unchanged; passing this ordinary-tile alternative does not establish support for smaller tiles.

## Final candidate decision: reject packed A/B on performance

The root-owned [complete packed-A/B decoder benchmark](packed_ab_partials_s256.json) confirms the isolated adapter regression. Its matched comparator is [conv_first_tap_baseline_s256.json](conv_first_tap_baseline_s256.json): the convolution and packed-A/B baseline snapshots are **byte-identical**, SHA256 `61ac211a397a4ab24737581b931ea318a4ed83d2e32d8eff67a4f26acda0a178`. The suggested separate `packed_ab_baseline_s256.json` was not run; the existing identical-source baseline supplies this comparison.

| Complete decoder | Prefill (ms) | Decode (ms) |
|---|---:|---:|
|Baseline|154.292234|1.458949642|
|Packed A/B|163.169548|1.686173957|

Both use layer0, S256/context512, 10 warmed prefill samples and 60 warmed decode samples, with deterministic replay, passing runtime-fallback audit and allocation tracking disabled during timing. PCC is identical: prefill0.999962395060449, prefill contribution0.9999594863488576, eager/traced decode0.9999779064754709.

Packed A/B increases decode by **0.227224 ms (15.575%)** and prefill by **8.877313 ms (5.754%)**. **Reject this candidate on measured full-path performance.** Lower product payload did not compensate for its complete coefficient/repacking/reduction sequence. Its exact probe remains useful evidence of the legal layout and arithmetic association, but this slower candidate does not merit an endurance run for promotion and is not selected for production. This closes this candidate investigation, not the overall optimization or final-validation stage.
