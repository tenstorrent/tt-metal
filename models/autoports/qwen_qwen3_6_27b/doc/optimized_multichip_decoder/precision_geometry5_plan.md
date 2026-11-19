# Integrated5 precision × geometry plan

`precision_geometry5.json` contains30 unique, unmeasured jobs on the ordinary TP4 decoder runner. All use B32/S33, layer0 or3, real recorded activations and the corresponding `bfp4_v2_ref_l{0,3}_32x33.pt` oracle. The exact integrated5 policy is retained, including joint GDN/conv, packed full-attention input, persistent CCL, DRAM collective placement, larger prefill MMRS pieces and the new decode packed epilogue. Two new real-activation controls are necessary: the prior integrated5 six-case results used synthetic inputs.

| Scope | Geometries | Precision/fidelity | Jobs |
|---|---|---|---:|
| Integrated5 controls | Incumbent c8/r2/K2 | BF8; GDN MLP HiFi2, other roles LoFi |2|
| MLP, both kinds | Gate/up c4/r2/K2 + down c8/r2/K2; both c8/r3/K2; gate/up c8/r2/K16 + down c8/r2/K8 | BF8/HiFi2, BF8/LoFi, BF4/LoFi |18|
| Attention, both kinds | Input c4/r2/K2 or c8/r3/K2; output geometry unchanged | BF8/LoFi, BF8/HiFi2 |8|
| Full attention accuracy-risk rows | Same two input geometries | BF4/HiFi2 |2|

The c4/r2 and c8/r3 choices already have passing `readers2` role evidence. Larger MLP blocks have passing `geometry3_gateup_k16_*` and GDN `geometry3_down_k8_*` evidence; the full-attention downK8 combination still requires hardware verification. New combinations and BF4 geometry are not asserted legal merely because the BFP8 component passed. c4/r2 was favorable for full-attention input/gate-up; it was not a universal GDN winner. The larger-K case is retained because crossing dtype with only core/readers would leave the material K-block question unresolved.

Every row has `audit.matched_control`, actual role scope, source hashes and `expected_decode_matrices`: K/N, padded N, activation shard shape, storage cores, readers, K block, per-core N, expected dtypes/fidelity. This is **expected** source-derived configuration, not proof a device executed it. The unchanged runner records actual matrix allocations/selected policy and the driver retains the entire job in `candidate_commands.jsonl`; compare those, then profile meaningful survivors to prove activation/weight/output dtype and fidelity reached the operations. Attention role fidelity also reaches its output projection; MLP role dtype changes gate/up, down and the paired prefill copy together. There is no assertion of per-matrix precision isolation.

CPU validation confirmed30 distinct `(layer, effective requested policy)` combinations,30 new names, every referenced fixture exists, all policy keys resolve in the main/inherited/helper sources, and all K/core, tile-padding and K-block divisibility arithmetic. No TTNN import/device call was made. In particular, main `_weight` consumes only `<role>_dtype`, role compute configs consume `<role>_fidelity`, `_prepare_matmuls` consumes per-weight geometry, and joint GDN inherits its geometry from `packed_gdn`. No unsupported `weight_dtypes` key is used.

## Attention BF4 limitations are explicit

GDN joint setup requires **both primary and correction BFP8**. Global `attention_dtype=bfloat4_b` with integrated5 joint enabled would fail setup. There is no supported per-weight dtype override in `MultichipDecoder._weight`. Therefore this manifest does not launch a known-invalid GDN joint BF4 job, disable joint silently, or pretend an output-only BF4 override works. BF4 MLP remains tested for both kinds. A GDN attention BF4 extension needs either an explicitly adapted joint kernel/packing contract or a separately named separate-projection family with its own matched controls.

Full attention packs Q/K/V/gate into one BF4 matrix under the current role policy. Q/K cannot be kept BFP8 through an existing policy key. `precision3_attn4_hifi2_l3_b32_s33.json` already fails unchanged KV-state PCC, approximately0.9938/0.9947, while output PCC passes. The final two rows cross that policy with new geometries; they are expected failure risks, not a promised accuracy recovery. No source change or threshold relaxation is included. They are last so a first accuracy failure does not prevent the useful BF8/MLP rows. After a failure preserve its file and launch any remaining missing row via a fresh subset; the driver stops at the first failure and `--resume` does not skip failures.

All BF4 rows are stage8 candidates only. They must not become the optimize-stage default, even if this layer test passes.

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_search \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/precision_geometry5.json --resume
```

## Native-pack recurrence is a separate precision candidate

Inspection of existing artifacts found:

- `native_pack_exact_b32_bf16.json`: strict state equality fails at the first eager step with484 total differing BF16 state elements across the recorded rank/user rows, while that step's output has zero bit mismatches. It is not a bitwise-equivalent storage optimization.
- `native_pack_exact_b32_fp32_control.json`: exact FP32-state control passes.
- `native_pack_precision_l0_b32.json`: ordinary single-step whole-layer PCC, cache-state and changed-input trace checks pass; prefill/decode recurrent-state PCC is approximately0.999419/0.999434. Its effective base is the integrated4 policy, without the later packed SwiGLU epilogue. This does not establish512-step autoregressive state behavior.

Keep `cumulative_exp_scalar` in the new precision×geometry manifest. Before native-pack can be selected, use the existing long-sequence path rather than weakening the exact kernel probe. The following **unexecuted** three commands generate a512-step oracle, measure the matching current recurrence control, and exercise the native-pack adapter on identical recorded inputs. Fresh names preserve all prior artifacts. `--stress512` in this harness runs two repeated sequences, checks finite values and exact repeat hashes, and compares final output/state with the oracle at the unchanged0.995 threshold. This is a precision gate, not a final uninstrumented latency row.

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
A=models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/text_activations/text_l0_b32_s33.pt
bash "$D/run_job.sh" native_pack_stress512_text_oracle_v1 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder \
  --baseline --layer 0 --batch 32 --length 33 --stress 512 --trace --repeats 5 \
  --activation-file "$A" --reference "$D/native_pack_stress512_text_ref_v1.pt" \
  --output "$D/native_pack_stress512_text_oracle_v1.json"
bash "$D/run_job.sh" native_pack_stress512_text_control_v1 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder \
  --layer 0 --batch 32 --length 33 --stress 512 --trace --repeats 5 \
  --policy-file "$D/integrated4_l0_policy.json" --activation-file "$A" \
  --reference "$D/native_pack_stress512_text_ref_v1.pt" --output "$D/native_pack_stress512_text_control_v1.json"
bash "$D/run_job.sh" native_pack_stress512_text_candidate_v1 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_native_pack --full-layer \
  --layer 0 --batch 32 --length 33 --stress 512 --trace --repeats 5 \
  --policy-file "$D/integrated4_l0_policy.json" --activation-file "$A" \
  --reference "$D/native_pack_stress512_text_ref_v1.pt" --output "$D/native_pack_stress512_text_candidate_v1.json"
```

Run under the parent's exclusive serialized hardware schedule with watcher/profiler unset. If this passes, cumulative selection still needs the B1 regression, final chosen-policy interaction and ordinary final validation gates; it does not repair the known bitwise state mismatch. No native-pack run is included among the30 geometry jobs.
