# Final-precision attention geometry probe

`tests/run_attention_geometry_probe.py` is a new source-only probe for the corrected linear-attention projections. It uses `evidence.decoder_class()` and `decoder_kwargs()` without changing a candidate or its policy. Select attention keys explicitly with `--key`; MLP is excluded. The default capture is B32 compact decode step 3 following a real S33 prefill with checkpoint weights. `ttnn.linear` is intercepted by identity against the selected `layer.weights` objects, so the correction-L1 path that bypasses `_linear_raw` is captured correctly. It verifies actual FP32 partials, BFP8 primary/residual weights, HiFi4, and disabled packer L1 accumulation.

The native C32/R4/block1 pair is measured first with its original captured programs, configs, weights and input layout. Every candidate uses that same compute configuration and both FP32 outputs. Timing includes input reshard, both linear calls, BF16 addition, final DRAM conversion and logical slice, matching the corrected projection's runtime topology. Setup, host capture, weight readback, references, and correctness reads are outside timing. Three eager and three traced warmups precede 20 blocking traced measurements; JSON retains every timing sample and the median/minimum. Trace replay is compared bitwise with eager output.

The bounded plan covers C8/16/32 input shards crossed with 1/2/4 readers per bank. All cases use the multicore pipeline, where four readers are supported; the separate non-pipeline reader helper's limit of three does not apply. Physical N is rounded to a multiple of `32*lcm(banks*readers, cores)`, and `per_core_N=physical_N/(32*cores)` exactly. Each geometry includes block 1, the entire activation shard width in tiles, and twice that width. C32/R4 also includes the largest proper shard divisor, four shards, and full K. Thus packed input includes blocks 5/10/20/160; output includes 3/6/12/24/192. There are 29 packed and 30 output candidates, plus each native baseline. Large blocks are intentionally attempted: exact L1/API errors are evidence, not a preemptive rejection. Both ordinary block-divisibility rules also hold, even though this pipeline allows ragged K blocks.

Weight preparation starts from the captured device BFP8 weights, never the original checkpoint values. Tile-aligned device slice/pad/shard is attempted first. If that API path fails, its exact error is saved, and setup retries by uploading the already dequantized stored values. Every prepared primary and residual is then read back and checked for exact element equality to the original prequantized values, with zero physical padding; a mismatch prevents benchmarking. The output reports FP64 references from both actual stored weight pairs and the original BF16 checkpoint, partial PCC/error metrics, final PCC/error metrics, and bit equality against the native control. Packed-GDN subprojection metrics are reported separately. No decoder acceptance threshold changes.

Example serial hardware command for the parent (not run while authoring):

```sh
QWEN_DECODER=optimized QWEN_OPT_CANDIDATE=grouped96 \
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_attention_geometry_probe \
  --key packed_gdn --key linear_attn.out_proj.weight \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/attention_geometry_results.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/attention_geometry_results.log 2>&1
```

Existing `QWEN_OPT_POLICY` overrides still apply through `decoder_kwargs`; omit them to test the chosen frozen candidate's defaults. `default81` and `exact82` are also valid choices when their effective attention policy remains the required one. Device policy setup, watcher/profiler separation, and serial ownership remain the parent runner's responsibility.

Artifacts are JSON with exact errors/tracebacks, a compact CSV beside it, and `.inputs.pt` containing actual captured compact activations. The script emits only one completion line per key; redirect the backend's own logs as above. Native control must succeed before candidates are compared. Syntax and the hardware-free plan passed; no TTNN execution or device claim was made. The generated source-only plan is `attention_geometry_plan.json`:

```sh
python models/autoports/qwen_qwen3_6_27b/tests/run_attention_geometry_probe.py \
  --key packed_gdn --key linear_attn.out_proj.weight --plan-only \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/attention_geometry_plan.json
```
