# AutoFix: four-core L1 residual fused add

## Final status

**Correctness repaired for the tested residual family by selecting the supported
SFPU ADD path.** The original mixed FP32/BF16 final add, BF16 output, fused scalar,
and four-core sharded residual remain intact; only the final call's
`fast_and_approximate_mode=False` changes kernel selection. Real 4096-token
prefill plus 128 advancing traced decode positions pass output/cache PCC,
replica equality, and one duplicate replay per position for sliding and full
attention. The family is slower than the recorded default in these measured
host replay timings; it was repaired and measured before comparison.

The native BinaryNg FPU batching defect is diagnosed but not patched here.
The parent performed all device runs and implementation changes. This agent
inspected source and evidence and authored reports; it did not run hardware.

## Starting evidence

- Diagnosis: `AUTODEBUG_l1_residual.md`, including exact initial source hashes
  and the follow-up BinaryNg source chain.
- `residual_l1_sliding.failure.json` / `.log`: step 0 / position 4096, ranks
  2 and 3 differ in 16 logical values, maximum errors 50.65625 / 68.34375.
- `residual_l1_full.json` / `.log`: equal replicas throughout 128 positions,
  but first decode PCC 0.795709 and minimum PCC 0.536955; baseline passes.
- `residual_diag_v2_full.tensors.analysis.json`: actual residual/shared/routed/
  combined inputs equal across ranks; 88→4 reshard exact; three FP32 norms
  have CPU-oracle PCC ≥0.99999988. Final fused add alone differs across ranks;
  CPU-oracle PCC 0.803070, maximum error 26.846722.

## Hypothesis experiments

| Hypothesis | Experiment and observed result | Verdict |
| --- | --- | --- |
| Wrong shard dimensions or Python closure binding | Source check: four shards ×704 =2816; block_w22/subblock_w2 coherent; final TP4 loop binding stable | No source evidence for either cause |
| Four-core norms or88→4 reshard introduce large corruption | Parent's same-invocation capture shows high norm CPU PCC and bit-exact reshard, while final add alone is wrong | Refuted for captured invocation |
| Original final mixed-operand FPU execution is defective | Parent promotes tail operand to FP32, computes FP32 ADD, explicitly casts final result BF16; full workload passes | Verified workaround, but multiple dtype/kernel changes |
| Selecting SFPU alone avoids the failure | Parent keeps original mixed operands/output/shards/fused scalar, sets final ADD fast_and_approximate_mode=False; full and sliding workloads pass | Verified minimal model workaround |
| FPU batching exceeds FP32 half-DEST capacity | Source shows eight-tile FPU batches with four-tile FP32 half-DEST; supported mixed-input path reaches it. SFPU controls avoid it | Concrete source defect; direct native batch-cap A/B remains unrun |

## Source mechanism and supported contract

This is not an unsupported mixed-float model call. ADD's public dtype policy
accepts FP32/BF16 inputs, and output construction accepts requested BF16 with
the explicit shard spec. The four-core `[32,704]` shards are valid and contain
22 tiles per core.

`binary/binary_nanobind.cpp:1899–1908` defaults ADD fast mode to true.
`binary_ng/device/binary_ng_device_operation.cpp:49–57` selects FPU for unequal
FP32/BF16 operands in that mode. The factory's all-sharded, no-broadcast branch
sets eight tiles per cycle (`binary_ng_program_factory.cpp:1016–1038`), then
correctly enables FP32 DEST because one input is FP32 (`:1202–1211`). It leaves
`dst_full_sync_en=false`, the descriptor default. Blackhole FP32 half-DEST has
four tiles, not eight. The FPU no-broadcast kernel computes and packs indices
0..n−1 within one acquire/commit; 22 tiles produce chunks8,8,6, exceeding the
acquired half each time. The original DRAM-output default uses one tile/cycle,
so it avoids this path.

Setting `fast_and_approximate_mode=False` is explicitly permitted for BF16 ADD
output (`binary_ng_device_operation.cpp:19–46`). It selects SFPU with two input
pairs/four DEST tiles per acquire. Homogeneous FP32 also selects SFPU, explaining
why that first control worked. The source evidence and passing controls support
the diagnosis; a native four-tile-cap A/B would isolate the precise batching
mechanism from other FPU-versus-SFPU differences.

No native source fix was made. An owning native fix should derive the FPU batch
cap from actual DEST mode before descriptor construction, preserving eight-tile
BF16 batching where valid. Such a C++ change requires the repository build and
focused kernel tests. The supported model workaround requires no C++ build.

## Verification artifacts and commands

The parent ran the following command form at each source revision:

```bash
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --layer 5 --length 4096 --steps 128 --trace --check-cache --residual-l1 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/residual_fp32_full.json
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --layer 5 --length 4096 --steps 128 --trace --check-cache --residual-l1 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/residual_sfpu_full.json
HF_HUB_OFFLINE=1 python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder --layer 0 --length 4096 --steps 128 --trace --check-cache --residual-l1 --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/residual_sfpu_sliding.json
```

| Artifact | Runtime SHA256 | Result |
| --- | --- | --- |
| residual_fp32_full.json | 3ea752e50f166ea5a58f1f6485fc9d8883630aeb275973482e6b4fd2ff8dea8e | Pass; output min PCC0.9994528616; cache min0.999971405 |
| residual_sfpu_full.json | f3fdaebd9e59ed195365d516347aee6a376347a67d04c2963386a379a00bde18 | Pass; output min PCC0.9994528616; cache min0.999971405 |
| residual_sfpu_sliding.json | f3fdaebd9e59ed195365d516347aee6a376347a67d04c2963386a379a00bde18 | Pass; output min PCC0.9976130548; cache min0.999996649 |

All three runner hashes are
`ea618e68073db2429a36c5f881e8978cb8b758b28e9ba76c64ae220d3c79f85a`.
Each has matching `.log`, `passed=true`, `all_replicas_equal=true`,
`trace=true`, and one duplicate replay per position. The two full controls have
exactly equal entire recorded output-PCC and cache-PCC vectors. This is equality
of saved metrics; their raw outputs were not compared here for cross-run bit
identity.

TP4 median host duration around blocking trace execution:

| Candidate | Full decode µs | Sliding decode µs |
| --- | ---: | ---: |
| Baseline | 724.813 | 650.322 |
| Homogeneous FP32 workaround | 754.134 | Not measured here |
| Minimal mixed-input SFPU workaround | 751.177 | 685.395 |

These are existing single-run host replay measurements, not kernel device times
or final stage performance acceptance. Baseline used eight duplicate correctness
replays versus one in these controls; only the first replay at each position is
included in the reported timing vector. Further stage checks, including any
required native profile, stack/batch, and Watcher gates, remain parent-owned.

## Remaining scope

- Keep the minimal supported SFPU workaround while comparing this family.
- If repairing BinaryNg itself, use the frozen mixed-input case with eager and
  repeated trace and change only the FP32 FPU batch cap for mechanism proof.
- No unverified native fix or numerical-precision policy change is claimed.
- Report author verification: source inspection, JSON checks, and
  `git diff --check`; no hardware execution by the report author.
