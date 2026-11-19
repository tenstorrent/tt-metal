# v4 decode graph: remaining source-supported controls

Inspection base: [v4_serial_l1_compact.py.snapshot](v4_serial_l1_compact.py.snapshot), SHA256 `deb1fa8f3fb3020bfb0f983cc04c43e45f96f245535e0e8e7b5741fe417cf4ad`. Root reports this candidate passes linear 8192 streams for seeds 913/42/154, around 1.352 ms decode and 118 ms prefill. Production remains qualified v3. These reported results do not qualify the new controls below. No device execution, production edit or core-harness edit was performed for this investigation.

Reviewed [pattern_assessment.json](../pattern_assessment.json), the current source, accurate [v3 decode sequence](../profile_v3/tracy/layer_0/decode_op_sequence.csv), and earlier convolution, recurrence, MoE, native fusion and v2/v3 adapter memos. The checklist's historical assessments do not imply every current producer/consumer adapter has been measured. The following are concrete changes to current adapters rather than repeats of the rejected native arithmetic paths.

## Prepared isolated convolution controls

| Snapshot | Change | SHA256 |
| --- | --- | --- |
| [v4_conv_tilize_widen.py.snapshot](v4_conv_tilize_widen.py.snapshot) | Fold each tap1–3 BF16 RM→BF16 TILE→FP32 TILE pair into BF16 RM→FP32 TILE | `f2340c1e41a248935865eb1e46677f8729cde72aa474caf276233005b3b890e4` |
| [v4_conv_direct_rm_history.py.snapshot](v4_conv_direct_rm_history.py.snapshot) | Form history directly in RM, tilize only selected tap rows and carry | `4c05e246e14a8cd3c1afb8e232b388016d675e9a98e7208f8a18941b0129c5a6` |

Both use the compact serial v4 base, remain independent, and do not include root's newer RM prefill variants. Only `_linear_attention` changes. The ordered four-tap product/addcmul sequence, FP32 accumulator, BF16 preactivation/FP32 SiLU/BF16 output chain, logical Q/K/V slices and three-row BF16 TILE carry remain unchanged.

### 1. Fold tilization and widening

Current tap1–3 code tilizes a BF16 RM slice, then casts the complete padded tile tensor to FP32 before `addcmul`. Candidate:

```python
tap_input = ttnn.to_layout(history_rm[:, j:j+s, :], ttnn.TILE_LAYOUT, dtype=ttnn.float32)
accum = ttnn.addcmul(accum, tap_input, self.taps_fp32[j])
```

`core/to_layout/to_layout_op.hpp:21–27` exposes the output dtype. `to_layout_op.cpp:156–164,222–229` forwards it to ordinary or padded tilize. `tilize_with_val_padding_device_operation.cpp:108–124` accepts BF16 RM inputs; its output spec uses the requested output dtype at lines 220/230. Thus this is a supported producer-format change; the addcmul same-dtype restriction remains satisfied.

One precision caveat is explicit: `tilize_with_val_padding_multi_core_default_program_factory.cpp:33–34` chooses FP32 destination from FP32 input (or particular special formats), not merely FP32 output. BF16 input therefore follows the normal BF16 tilizer unpack path and packs to FP32. Ordinary BF16 values are exactly representable in FP32, but extreme/subnormal/sign-zero conversion behavior must be measured, not inferred. This is why the diagnostic includes every finite BF16 input code and compares each tap's FP32 operand before accumulation.

V3 profile tap pairs at steps 68/69, 72/73 and 76/77 cost 6.220+5.042, 6.146+5.024 and 6.123+5.177 us. These are opportunities to replace two dispatches with one, not predicted savings: the FP32-output tilizer may cost more than the BF16-output tilizer. This control is distinct from rejected whole-history FP32 widening, which increased slice/untilize traffic.

### 2. Produce shared RM history directly

Current `concat([TILE carry,TILE qkv], dim=1)` has a padded height input (three carry rows), so the public concat wrapper untilizes both inputs, concatenates in RM, then tilizes the full history. The convolution immediately untilizes that history again for shared tap slicing. Candidate explicitly untilizes each input and concatenates RM, then tilizes the first tap slice and preserves all later tap conversions. Carry is cropped in RM and tilized, including aligned lengths.

Contract: `concat/concat.cpp:96–134` defines this untilize→RM concat→retilize fallback. Its direct unaligned tiled optimization is for width concat (`:296–305`), not this height concat. No unsupported view of unaligned history is used. The selected v3/v4 shared-RM-history and RM-carry improvements are retained; this removes their upstream round trip.

V3 decode steps 60–66 expose the exact sequence: carry untilize 3.554 us, current QKV untilize 2.497 us, RM concat 1.765 us, full history tilize 7.266 us, first TILE slice 3.405 us, and later shared history untilize 3.779 us. The candidate retains input untilizes and RM concat, replaces the first TILE slice with RM slice plus tap-only tilize, and avoids full-history retilize/untilize. All replacement costs must remain timed. For aligned lengths the old carry slice is already cheap; the new RM carry conversion could offset gains, so aligned cases are mandatory.

### Complete actual-block diagnostic

[tests/v4_conv_adapter_probe.py](../../../tests/v4_conv_adapter_probe.py) is new; earlier measured probes and reports are unchanged. It AST-extracts each actual snapshot convolution from history construction through carry copy, including Q/K/V output slices. It records source/dependency/snapshot/extracted-AST hashes and asserts only `_linear_attention` differs.

Cases use real layer-0 checkpoint taps and seeded BF16 history at S=1/32/33/127/128/129/256, plus a history tensor containing all 65,280 finite BF16 codes (and only padding zeros to fit the width). It checks every FP32 tap1/2/3 operand, FP32 preactivation, BF16 convolution output and next carry against baseline. No tolerance suppresses tiny differences. Complete trace timing includes input/history conversions, concat, all slices/tilizes/casts, arithmetic, activation and carry copy; four warmups stabilize repeated S1 history. First/last replay outputs and carry are checked again. The exhaustive-code case is correctness-only.

[CPU setup evidence](v4_conv_adapter_probe_cpu.json) passes all eight cases and source-scope assertions. It is not device evidence.

Root-only commands after the current serialized queue:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_conv_adapter_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_conv_adapter_probe.json

for candidate in serial_l1_compact conv_tilize_widen conv_direct_rm_history; do
  python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
    --snapshot "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_${candidate}.py.snapshot" \
    --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
    --output "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_${candidate}_conv_control_s256.json"
done
```

Acceptance remains unchanged: exact adapter discrimination, complete full-path timing, then required endurance/other gates if retained. No measured benefit is claimed for either control.

## Other concrete current-graph controls

### Router sharded slice directly into DRAM

Replace each `to_memory_config(gate_output, DRAM)[:,0,:8]` with:

```python
ttnn.slice(gate_output, (0, 0, 0), (1, 1, 8), memory_config=DRAM)
```

Retain original score tilize and final TILE routing reshape; do not repeat the measured slower RM score geometry. Native gate outputs are aligned BF16/UINT16 RM HEIGHT shards `[1,32,32]`. `slice.cpp:53–69` does not require a composite unshard for this aligned HEIGHT input. `slice_device_operation.cpp:436–453` selects general RM slice for an interleaved output; its reader `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:41,107–109` uses TensorAccessor and `noc_async_read_sharded`, supporting the sharded source directly. This can replace two S2I+slice pairs with two slices. V3 steps155/156 and158/159 cost 1.488+0.894 and1.489+0.904 us. Those are bounds on removed work, not wins.

Discriminator: actual native gate outputs from checkpoint-derived logits, exact score/id comparisons and complete post-gate adapter traces. Preserve actual gate's physical 32×32 output buffers. Compacting its output allocation is unnecessary and risks its hardcoded shard contract. Root owns any resulting candidate/probe.

### Metadata-only logical row crop

Native head concatenation and MoE expert reduction return a padded 32-row physical tensor whose first row alone is consumed. Current code launches a slice to reduce logical height before reshape. Use the explicit `ttnn.reshape(tensor, logical_shape, padded_shape)` overload with the original physical padded shape and logical H=1, then the existing rank reshape. This retains logical cropping while removing a physical copy.

Source: `reshape_nanobind.cpp:41,117–130` exposes explicit logical+padded shapes; `reshape.cpp:778–789` permits tile-preserving logical-volume changes when padded tile geometry and last dimension match, dispatching to metadata view. `ttnn/core/tensor/tensor_ops.cpp:465–479` aliases the original device buffer for TILE layout. The stricter public `ttnn.view` requires equal logical volume and is not a substitute. The experimental Python view also lacks the explicit padded-shape overload; use the documented reshape overload.

Prove source/input immutability and exact consumed row0, including positive/negative finite, infinity and actual verified NaN padding. Include downstream consumer output so unchanged logical shape is not the sole check. V3 step143 concat-head crop costs1.931 us; step172 MoE crop costs1.680 us. Root assigned these isolated artifacts to the recurrence agent; they are not bundled with this memo's convolution snapshots.

## Exact blockers / do not repeat

- **Remove recurrent V widening with mixed subtraction:** mathematically valid but current Python binding blocks the required accurate mixed SFPU configuration. `binary_ng_device_operation.cpp:25–46` rejects explicitly `fast_and_approximate_mode=False` for FP32 ADD/SUB output. Python `binary_nanobind.cpp:1314–1396` wraps subtract with a plain `bool`, and `:2185–2195` defaults it to true; omitting it is not C++ `nullopt`. Mixed BF16/FP32 with true does not select the accurate SFPU path (`binary_ng_device_operation.cpp:51–58`). `None` is not the wrapper contract. Therefore do not remove the cast while quietly choosing FPU. Existing FP32/FP32 subtraction remains SFPU even with that default. This differs from mixed multiply, whose accurate flag permits FP32 output.
- **Mixed addcmul for remaining conv taps:** same-input-dtype validation still blocks BF16 history with FP32 accumulator/taps. Tilize-output folding above satisfies this constraint rather than bypassing it.
- **State multiply/add→addcmul:** already rejected for changed FP32 rounding; no new evidence reopens it. Preserve separate multiply then add and exact contraction order.
- **Native causal QKV convolution:** no activation-off/preactivation output flag; existing fused SiLU crosses the required BF16 boundary. Exact stock FP32 depthwise adapter was already measured slower. Do not rerun either unchanged.
- **Native recurrent GDN / contraction matmul:** FP32-state narrowing/reduction-order failures already established. Output-only variants were also slower. No new matching arithmetic contract was found.
- **MoE compute / postcombine / minimal SwiGLU / output-projection residual:** native weight-format restriction or complete-adapter losses are already documented in pattern assessment. The direct router slice and row metadata crop above alter different data-movement boundaries.

These tests are justified by surviving graph boundaries and existing native APIs. The open stop rule is not itself a reason to manufacture extra stress runs or claim a source-only speedup.

## Initial convolution device result

Root completed [v4_conv_adapter_probe.json](v4_conv_adapter_probe.json). Direct RM history is exact across all eight cases and improves the measured complete adapter. Folded tilize/widen improves the seven seeded-shape timings but differs in exhaustive-case FP32 tap operands; preactivation/output/carry happen to match that fixture. The report correctly remains `device_exact=false`. See [boundary follow-up](AUTOFIX_v4_conv_widen_boundary.md) for exact timings, source diagnosis and the separate adversarial diagnostic. No acceptance inference is made from final fixture equality.

## Completed widening propagation discriminator

The [follow-up](AUTOFIX_v4_conv_widen_boundary.md) now records36/144 crafted cases with FP32 preactivation differences and24/144 with BF16 output differences, at maxima1.274576e−38 and1.653039e−38; all carry codes match. No global bitwise-equivalence claim is made. Raw and combined full-model controls retain short PCC, the combined72d4 candidate improves decode1.332007→1.317329ms, and all three single-repeat8192+33 selection streams pass with unchanged contribution minima. Diagnostic false flags remain unchanged. These synthetic differences are not an automatic rejection or a new acceptance threshold; final required model gates and root selection remain open.
