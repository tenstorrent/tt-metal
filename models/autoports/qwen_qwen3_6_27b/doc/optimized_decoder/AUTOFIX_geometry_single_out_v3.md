# Single-term BF16 output projection geometry

Geometry111 V2 measured corrected output projection with two FP32 matmuls and a BF16 addition. The parent reports native C32/R4/block1 at 0.1586 ms versus block3 at 0.1640 ms. Those results do not establish the best geometry after removing output correction. The new immutable `tests/run_attention_geometry_probe_v3.py` targets only `linear_attn.out_proj.weight` with its actual single BFP8 matrix and native BF16 matmul output.

V3 requires the selected decoder/policy to have no output-projection correction matrix. It uses `evidence.decoder_class/decoder_kwargs`, captures the actual `ttnn.linear` call by device-weight identity, and asserts BF16 output, BFP8 weight, HiFi4 policy and disabled packer L1 accumulation. The current candidate107 configuration must be supplied with `weight_correction: ["packed_gdn"]` in the existing policy. V3 does not alter the policy itself. Its native control reuses the exact captured C32/R4/block1 program and compute configuration.

The measured projection contains activation reshard, exactly one BF16-output matmul, the policy-selected output memory conversion, and logical slicing only if needed. There is no correction add. `l1_mlp` determines whether the output remains in L1 exactly as `_linear_raw` does for compact decode. FP32 DST accumulation is retained; FP32 output is not requested. Host/reference reads and upload of captured BF16 activations stay outside timing. V2's immediate host-snapshot capture fix is retained so L1 activation/output lifetimes do not extend into the MLP.

The bounded sweep has 15 candidates plus a native control:

- C8/16/32 crossed with R1/2/4 at block1: nine cases.
- R4 at each C additionally tests block3 and the full activation-shard width: 24 tiles at C8, 12 at C16, 6 at C32.

Physical-N padding, exact `per_core_N`, and K/shard block divisibility follow the validated V2 rules. All cases use the multicore pipeline. Weight transformations start from the actual prequantized device BFP8 values and must pass exact readback equality before timing. References include both the stored single BFP8 matrix's FP64 product and the original BF16 checkpoint matrix's FP64 product. JSON records PCC/errors/native bit equality, exact failures, and traced timing samples; adjacent CSV and `.inputs.pt` artifacts remain available.

With the parent's current candidate107 environment and single-term policy already set, run:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_attention_geometry_probe_v3 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_single_out_results.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_single_out_results.log 2>&1
```

If setting the environment from scratch, use `QWEN_DECODER=optimized`, `QWEN_OPT_CANDIDATE=candidate107`, and the selected full `QWEN_OPT_POLICY` with `weight_correction` restricted to `packed_gdn`. Do not replace other final candidate policy settings with an incomplete example policy.

Syntax, hardware-free generation of `attention_geometry_v3_plan.json`, and the host-only `candidates/check_geometry_single_v3_cpu.py` pass. The latter executes the extracted projection against sentinel tensor APIs for both output residency settings and padded/unpadded widths, verifying exactly one BF16 linear call and no addition. `geometry_single_v3_cpu.json` records the source hash and result. No hardware was accessed while creating V3; selection requires this topology's own device correctness and timing evidence.
