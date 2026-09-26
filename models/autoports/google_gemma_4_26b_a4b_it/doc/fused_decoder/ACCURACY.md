# Accuracy and semantics

Acceptance remains PCC >= 0.995. Both TTNN paths use pinned real model weights and identical seeded headline inputs. Every prefill row and all128 decode outputs are included in the comparisons.

| Layer kind | Functional prefill / min decode PCC | Fused prefill / min decode PCC | Direct TTNN prefill / min decode PCC |
| --- | ---: | ---: | ---: |
| sliding_attention | 0.9985263366 / 0.9990432236 | 0.9985529033 / 0.9989850246 | 0.9996620521 / 0.9996588301 |
| full_attention | 0.9994406151 / 0.9997738820 | 0.9994185025 / 0.9997355920 | 0.9996741208 / 0.9998033554 |

Evidence: verified_equivalence_{sliding,full}_{functional,fused}.json and the paired verified_equivalence_{sliding,full}.json; exact subprocess commands are embedded in the paired files and verified_validation_commands.json. Prefill PCC is the complete output tensor correlation, while decode reports the minimum over128 steps.

Small baseline deltas come from validated native unweighted normalization, selected decode rotary, shared-normalization ordering, and the native mixing/shared-down matmul accumulation order. Native operations can narrow FP32 operands through Src and perturb close expert ranks; external FP32 gamma and precise sliding head normalization prevent the failed variants. Selected outputs remain above the original bar and pass direct TTNN equivalence. Packed gate/up, peer batching and GELU/router merges have bitwise component controls; EXP/RSQRT merges leave headline PCC unchanged. Mixing/shared-down changes have real-input component PCC/replay controls plus complete-layer comparisons, not an unsupported bitwise claim.

The verified_{request_reuse,batched,prefix_continuation}_{sliding,full}.json artifacts cover nine changing requests,32 disjoint cache slots, deterministic replay, and partial-page continuation with prefix/other-slot preservation. Logical lengths include31,32,33,65,1023,1024,1025,2047,2049. No public chunk-alignment restriction is introduced.

verified_pytest.log records the real-weight fused-path regression: FunctionalDecoder._forward is patched to raise, so a functional computation fallback cannot pass. Separate verified_watcher_{sliding,full}.json runs exercise4096/128 with exact program-cache warmup and deterministic repeated output. Watcher inspection files record completed dumps and diagnostic scans.

All maximum-context runs compute every TT query/cache row. Their HF oracle checks291 sampled query rows (subset), including chunk boundaries and final33 rows. Traced decode checks mutable positions at the final two context positions and repeats the last after another cache row has changed. Those repeated positions are not a fixed-state determinism check; headline and batch replay equality provide that control. Headline accuracy is full scope; long-context prefill is not a full-output oracle.

Final selected-policy maximum and non-aligned near-maximum tests all pass. Earlier batch64_long_* checks are retained as intermediate evidence.

| Layer kind | Logical tokens | Prefill PCC (291-row subset) | Minimum end-context decode PCC |
| --- | ---: | ---: | ---: |
| sliding_attention | 262144 | 0.9981859749 | 0.9997277273 |
| sliding_attention | 262143 | 0.9981839549 | 0.9997969071 |
| full_attention | 262144 | 0.9962342265 | 0.9998799242 |
| full_attention | 262143 | 0.9961807402 | 0.9998449925 |

Evidence: verified_long_{sliding,full}_{262144,262143}.json and verified_validation_commands.json; all run the frozen delivered runtime.

## Mixed expert-batch tail and inherited cache sensitivity

Fresh logical S65 pads to96 rows and exercises the64+32 expert batches. Both real-weight layer kinds pass FP32-HF prefill. Full attention also passes all8 traced decode outputs. Sliding fails FP32-HF at position68: fused PCC0.9813712959 and functional0.9809274231; other positions pass. This failed row is retained, not relabeled or excluded from the comparison.

The paired complete outputs remain equivalent: prefill PCC0.9991649499 and all8 decode PCC >=0.9997196429, including0.9998401882 at68. The functional stage probe isolates the discontinuity: attention/residual PCC >0.999997, but CPU routing on the TT residual selects expert70 instead of the FP32-HF oracle's eighth expert10. CPU attention with identical QKV reproduces those routes. A pure CPU HF control on the saved identical inputs reproduces the same sole failed position and expert swap by rounding only cached K/V to BF16 (PCC0.9814915924); BF16 cache plus rotary tables gives0.9815510238. No TT kernel or fused rewrite is needed to reproduce it. This is inherited BF16-cache sensitivity at a close MoE rank boundary; cache dtype/context contract remain unchanged.

The durable test_fused_mixed_expert_tail keeps all8 positions, full-HF prefill checks, unchanged0.995 direct functional-equivalence threshold, finite tensors, deterministic replay, runtime audits, program-cache guard and functional-fallback prohibition. It recognizes only the independently controlled sliding position68 FP32-HF failure. Existing33-token and4096/128 FP32-HF gates retain their original threshold without exceptions.

Artifacts: verified_tail65_*.json, tail65_sliding_{functional,fused}_paired.json, tail65_sliding_equivalence.json, tail65_functional_stages.json, tail65_cpu_precision.json, tail65_diagnostic_commands.json, shared_fidelity_commands.json, AUTODEBUG_tail65.md and verified_mixed_tail_pytest.{json,log}. Exact saved tensors/inputs are local .pt artifacts; compact JSON records the comparisons and CPU fixture hashes.
