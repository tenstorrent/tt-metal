# Frozen-v2 adapter candidates: shared history cast and prefill norm HC reduction

Prepared against frozen source `0b96b6204c5c1b276b14c5f29d8dd4789569608e092cec90b7a3898ee1cfa26c`. No production/native/core-harness edits, TTNN imports or hardware execution were performed by this investigation agent. Root owns the serial validation/profile queue and subsequent measurements. Neither candidate is selected or accepted yet.

## Independent snapshots

- [Baseline](v2_adapter_baseline.py.snapshot): `0b96b6204c5c1b276b14c5f29d8dd4789569608e092cec90b7a3898ee1cfa26c`.
- [Shared convolution history widening](v2_conv_shared_history.py.snapshot): `03643b919f9288008267b088fcf48369cae4fdfca9c96fa04eb8c3074ff744d6`.
- [Prefill inputnorm HC reduction](v2_prefill_norm_hc.py.snapshot): `5214366e3ebcc6c28c2e3cf9958d5ebdb02fa2c17998a0e8d2b87c0fa4d047af`.

Each changes only its named method relative to the baseline. They are not combined. The separate agent's mixed-input normalization candidate changes operand widening; this HC candidate retains the existing FP32 input and square operations and changes only the multirow sum layout/order adapter.

## Convolution: one widening shared by three taps

The selected mixed-input first-tap multiply is retained. After it, widen complete BF16 `history [1,S+3,8192]` once; taps1–3 slice that FP32 tensor and call the same ordered FP32 addcmul operations. Preserve the final rounding/SiLU chain, Q/K/V slices and BF16 three-row carry update.

This replaces three widening dispatches with one. The addcmul same-input-dtype restriction remains satisfied; it does not revive the unsupported mixed addcmul proposal. Conversion commutes with coordinate selection for ordinary BF16 values, but actual extreme/subnormal codes are compared in the probe rather than assuming device behavior. No arithmetic operation is reassociated.

The tradeoff is concrete: FP32 slices transfer more data, and unaligned H slices can each untilize the widened history. In the previous same-adapter [prefill profile](../profile_moe_padding_cached/tracy/layer_0/prefill_op_sequence.csv), steps100/105/110 are the three FP32 casts, about33.5µs each for S256, accompanied by separate slice adapters. Those times are from the earlier61ac source with cached metadata, not a fresh v2 attribution. The cast savings alone do not predict complete-adapter performance. The new probe keeps every slice/conversion inside timing and allows this candidate to lose honestly.

## Prefill input norm: move tokens into the tile height

For rows>1, replace the existing cumsum16/select/reshape/cumsum4/select sequence with:

```python
squares = ttnn.reshape(ttnn.multiply(wide, wide), (1, rows, 64, 32))
squares = ttnn.permute(squares, (0, 2, 1, 3))  # native HC
squares = ttnn.reshape(squares, (4, 16, rows, 32))
total = ttnn.empty((1, 1, rows, 32), dtype=ttnn.float32,
                   layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                   memory_config=DRAM)
total = ttnn.experimental.fast_reduce_nc(
    squares, dims=[0, 1], output=total, compute_kernel_config=self.compute)
```

The logical output allocation preserves `rows` when native intermediate tensors use padded tile height. It is the same supported preallocated-output mechanism already used by the singleton norm. The remaining four-group/eight-lane sums, mean scale, epsilon, inverse refinement, gamma and BF16 output boundary remain unchanged. Rows1 follow the unchanged existing path.

For each token and lane, the first reduced axis still visits its16 terms in their original order; the second visits its four partials in order. HC merely relocates independent tokens into H. This avoids the `[rows,4,16,1,32]` scalar-plane alternative, which would inflate storage32×. Source contracts:

- `data_movement/transpose/device/transpose_device_operation.cpp:25–53` explicitly keeps logical C/H and rounds the new H to32 for tiled HC. C need not be tile aligned. Its input H64/W32 satisfy validation at396–419.
- `experimental/reduction/fast_reduce_nc/fast_reduce_nc.cpp:38–62` sorts axes and performs the larger axis first, preserving16 then4.
- `device/kernels/reduce_nc.cpp:42–59` copies the first FP32 term then adds subsequent terms sequentially.
- `fast_reduce_nc_device_operation.cpp:118–121` accepts the preallocated output's logical spec. The snapshot uses the existing explicit HiFi4/nonapproximate/FP32-destination config.

The prior same-adapter profile's steps6–15 total about229µs for S256, including reshapes, cumsums, untilize/slice/tilize operations. The HC adapter also has real movement cost, so this is a source-backed experiment rather than a speed claim. It is distinct from the rejected normalization-L1 placement trial and from the retained singleton norm outer reduction.

## Complete-adapter diagnostic

[v2_adapter_fusion_probe.py](../../../tests/v2_adapter_fusion_probe.py) invokes each snapshot's actual `norm` method. For convolution it extracts the actual AST block from history concat through carry copy, including the output Q/K/V slices, and records its AST hash. It does not hand-reimplement either device adapter. Setup loads the real layer0 gamma and four convolution taps only.

Default row counts are1/2/33/128/129/256. Convolution adds an untimed case containing all65,280 finite BF16 codes across the actual8,192-channel history geometry; the 256 excluded nonfinite inputs are replaced with zero padding. Norm inputs are deterministic synthetic BF16, or actual saved oracle inputs when `--oracle-cache PATH` is provided. Oracle loading is setup-only and never enters a device callback.

Comparisons cover:

- Norm: exact FP32 shifted-mean codes and exact final output codes.
- Convolution: exact FP32 preactivation codes, final output codes, and the raw BF16 next-history values.
- Trace replay: first/last outputs against the baseline; convolution's carry is also compared. Four warmup calls stabilize singleton history before timing. Carry writes remain inside every replay.

Every allocation, coefficient-free layout adapter, concat, slice, cast, arithmetic operation, and carry write remains in the timed function. Default timing is15 samples ×10 replays after warmup. Cases with code differences are retained and skipped for timing, not filtered out of the exactness result. Existing output is protected. The diagnostic does not replace full-decoder benchmarking or endurance gates.

CPU setup passed all13 default cases. The norm indexing proof performs explicit FP32 sequential16-then4 folds through both layouts, with exact codes for every selected row. The extracted convolution ASTs parse correctly. The new test uses pinned Black23.10.1/isort5.13.2 formatting. Root-owned device results subsequently completed; see below.

Root command after the current queue releases hardware:

```bash
timeout 1200 env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v2_adapter_fusion_probe
```

`--kind norm` or `--kind conv` isolates either experiment; use distinct `--output` names. `--oracle-only --output /tmp/v2_adapter_cpu.json` performs CPU setup without importing TTNN. The default device opens only physical chip0.

For complete-path timing, invoke `benchmark_fused_snapshot` independently for baseline and each candidate. All use layer0/S256/context512,10 prefill and60 decode samples:

```bash
QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_adapter_baseline.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_adapter_baseline_s256.json

QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_conv_shared_history.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_conv_shared_history_s256.json

QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_prefill_norm_hc.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_prefill_norm_hc_s256.json
```

Use the same clean timing environment as the probe. Smaller tile layouts, final-reduce output-dtype fusion unavailable in the installed binding, packed A/B, native output-only adapters, and already rejected L1 norm/weight staging are not reopened. The final stop rule remains a separate evidence decision; these two experiments target concrete repeated work rather than consuming device time for its own sake.

## Root-owned exact adapter results

[v2_adapter_fusion_probe.json](v2_adapter_fusion_probe.json), with [provenance](v2_adapter_fusion_probe.json.provenance.json), reports `device_exact=true` for all13 cases. FP32 shifted means/preactivations, final outputs and applicable carry values agree exactly, including the finite-code convolution case. Complete traced adapter medians include all conversion and carry work:

| Rows | Norm baseline (ms) | Norm HC (ms) | Conv baseline (ms) | Shared FP32 history (ms) |
|---:|---:|---:|---:|---:|
|1|0.174439|0.174159|0.180631|0.191556|
|2|0.198027|0.184162|0.184756|0.196407|
|33|0.265824|0.228195|0.330007|0.349070|
|128|0.331864|0.254252|0.692829|0.730215|
|129|0.339775|0.259664|0.852501|0.890834|
|256|0.462955|0.319796|1.063533|1.146855|

The norm S1 path is unchanged; its small timing variation is not attributed to the multirow candidate. Norm HC improves every changed shape. Shared FP32 convolution history loses at every shape, consistent with the wider unaligned slice adapters outweighing removed casts.

## Complete decoder decision

All three source-bound artifacts use layer0/S256/context512,10 warmed prefill samples and60 warmed decode samples:

| Snapshot | Prefill (ms) | Decode (ms) |
|---|---:|---:|
|[Frozen-v2 baseline](v2_adapter_baseline_s256.json)|154.133333|1.406199066|
|[Shared FP32 history](v2_conv_shared_history_s256.json)|154.356289|1.416974468|
|[Prefill norm HC](v2_prefill_norm_hc_s256.json)|153.877292|1.405388117|

Each reports deterministic replay and a passing runtime fallback audit. PCC is identical: prefill0.999962395060449, prefill contribution0.9999594863488576, traced decode0.9999779064754709.

**Reject shared FP32 history on measured performance:** decode increases0.010775ms (0.766%) and prefill increases0.222956ms. **Advance prefill norm HC for integration/endurance qualification:** prefill decreases0.256041ms (0.166%). Its singleton code is unchanged, so the0.000811ms decode difference is not attributed to the prefill-only fusion. These short controls do not establish final acceptance or close the overall stage.

## Source-only follow-up: share BF16 untilization

One distinct convolution adapter avoids the rejected candidate's wider slicing cost. Keep BF16 `history` and the selected first-tap multiply unchanged. Compute `history_rm = ttnn.to_layout(history, ROW_MAJOR_LAYOUT)` once. For taps1–3, slice this shared BF16 RM tensor, explicitly tilize the slice, and then run the existing FP32 typecast and addcmul. Keep the final chain and carry copy unchanged.

This removes **two redundant BF16 untilizes**, not the three required FP32 casts. The source proves that tiled slice switches to RM for an unaligned begin (`data_movement/slice/slice.cpp:233–235`), untilizes at382 and restores the original input layout in the return adjustment at193. All three tap begins1/2/3 take that path. The prior S256 profile shows separate UntilizeWithUnpadding operations at97/102/107,49.552/49.910/49.579µs, each consuming the same BF16 history. Sharing that materialized BF16 RM result preserves each subsequent slice's width and arithmetic.

This is a concrete untried repeated-work elimination supported by the existing layout/slice operators, not an unsupported unaligned view or a repeat of the wider-history trial. The remaining three tilizes, casts and addcmul calls must remain in any complete-adapter timing. An optional BF16-RM→FP32-TILE tilize could also fold a widening, but its conversion behavior is a separate precision question and is not bundled into this source recommendation. No new device result or production change is claimed.

## Prepared shared-BF16-RM convolution control

Root authorized the concrete untilize-sharing follow-up. [v2_conv_shared_rm_history.py.snapshot](v2_conv_shared_rm_history.py.snapshot), SHA256 `0773879d65745498a841816ccb214a8821e85a5c81bbd7ec540d34f7772dc2af`, changes only the convolution block of frozen v2. It leaves first-tap multiplication, all three per-tap FP32 casts, addcmul order, activation chain and original carry-write expression unchanged. Only the BF16 history's RM conversion is shared; each tap's BF16 RM slice is explicitly returned to TILE before its existing widening.

Before extending the diagnostic, its exact measured source was preserved as [v2_adapter_fusion_probe.original.py.snapshot](v2_adapter_fusion_probe.original.py.snapshot), SHA256 `1e1ca5382d896cc5b2211cbba50ad7333799f4658b927344b0440b30a7387e6f`. This equals the `source_sha256` in the original device report, so that measured harness remains reproducible. The original JSON and provenance are unchanged. The live test adds only optional `--conv-snapshot`; default behavior still selects the original shared-FP32 candidate. A new result name prevents mixing runs.

CPU setup and AST extraction passed all seven convolution cases for the new snapshot. Subsequent root-owned device results are recorded below. Root-owned command:

```bash
timeout 1200 env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v2_adapter_fusion_probe \
  --kind conv \
  --conv-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_conv_shared_rm_history.py.snapshot \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_conv_shared_rm_history_probe.json
```

If exact and faster, measure the complete decoder with the same clean timing environment:

```bash
QWEN_DECODER=fused python_env/bin/python \
  -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
  --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_conv_shared_rm_history.py.snapshot \
  --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v2_conv_shared_rm_history_s256.json
```

All shared-history conversion, slices, tilizes, casts, arithmetic, output slices and carry writes remain inside the actual extracted adapter and the complete decoder. No production/core-harness edits or hardware activity were performed by this investigation agent.

## Shared BF16 RM history: exact and faster in the complete decoder

The root-owned [shared-RM probe](v2_conv_shared_rm_history_probe.json), with [provenance](v2_conv_shared_rm_history_probe.json.provenance.json), reports `device_exact=true` for all seven cases. FP32 preactivations, final outputs and raw next-history codes match, including the exhaustive finite-BF16 case. Complete adapter timing retains all three casts, every tilize and the carry write:

| Rows | Baseline (ms) | Shared BF16 RM (ms) |
|---:|---:|---:|
|1|0.182098|0.173080|
|2|0.184579|0.177115|
|33|0.329320|0.304744|
|128|0.690802|0.610122|
|129|0.851016|0.768538|
|256|1.063795|0.964252|

The subsequent [complete decoder benchmark](v2_conv_shared_rm_history_s256.json) records prefill **154.246572ms** and decode **1.396986423ms**, versus [frozen-v2 baseline](v2_adapter_baseline_s256.json) **154.133333ms / 1.406199066ms**. Both use layer0/S256/context512,10 prefill and60 decode samples. Decode improves **0.009213ms (0.655%)**. Prefill is **0.113240ms (0.073%) higher** in this full-path run despite the faster isolated S256 convolution adapter; no whole-prefill speedup is claimed.

PCC is identical (prefill0.999962395060449, contribution0.9999594863488576, traced decode0.9999779064754709), replay is deterministic and runtime-fallback audit passes. **Advance shared BF16 RM history as a measured candidate**, subject to the unchanged integration/endurance/final-reference gates. This result validates reuse of the existing BF16 layout conversion, rather than the rejected shared-FP32 slicing approach. Production remains frozen v2 at this update. Root plans a separately measured combination with mixed/HC inputnorm and filled-row token-height projection; that combination must retain its own source identity and qualification rather than inherit acceptance from these isolated controls.
