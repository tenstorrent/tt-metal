# AutoFix: BF16 prefill memory placement

Status: verified repair.

The failing contract was construction and execution of every prepared prefill profile at the requested precision, retaining the advertised 262,144-token context. BF16 normalized activations collided with matmul dataflow buffers before accuracy scoring. The all-BF16 reference subsequently exposed a separate collision in the 512–1,024-row projection path.

## Diagnosis and discriminating experiments

The installed AutoDebug launcher produced [AUTODEBUG.md](AUTODEBUG.md). It reconstructed the 8,192-row activation failure: retained normalized BF16 hidden data used 610,304 bytes per bank, projection input staging 153,600, collective scratch 34,816 and semaphores 896. The resulting allocation frontier was 773,248, while minimal matmul needed a static end of 915,392. BF16 weights further increase the required buffers. No device experiment or source edit ran in the AutoDebug child.

The first repair keeps normalized inputs and projection inputs in DRAM from 4,096 physical rows when a layer uses BF16 normalized activations or BF16 projection weights. Compressed policies retain their original 8,192-row L1 threshold. An independent read-only AutoFix hypothesis audit found the direct-decoder padding boundary: logical rows 4,065–4,095 occupy 4,096 physical rows. The placement predicate therefore rounds to whole tiles. Chunk sizes, context, numerical dtypes and minimal-matmul geometry stay the same. Packed SwiGLU consumes the same DRAM normalization output.

This repair passed the BF16-activation smoke. All-BF16 weights then failed in the separate special 512-row attention-output projection. A second read-only hypothesis audit reproduced its static end exactly: 1,376,256 bytes of CB payload plus 112,576 base bytes equals 1,488,832, beyond the logged frontier of 1,428,608. Conditional K8 instead of K16 for BF16 weights reduces that payload to 786,432. All special-branch siblings have K divisible by 8; the largest predicted static end at 1,024 rows is 1,161,152. BFP8 weights retain K16. Changed accumulation grouping required the measured accuracy checks below; source arithmetic alone was not accepted as a numerical pass.

Changed implementation: `tt/precision.py` and `tt/multichip_decoder.py`. Host boundary coverage is in `tests/full_model_prefill_plan_test.py`. No context reduction, assertion suppression, reset or process termination was used. Previous-stage root reports remain untouched.

## Validation

Exact commands are in `placement_repair_jobs.json`, `activation_repair_full_jobs.json`, `bf16_repair_jobs.json` and `commands.jsonl`, run through `tests/datatype_queue.py` under the pinned lane environment and bounded timeouts.

- All 22 host planner/cache tests pass. Hook logs: `placement_host_hooks.log`, `bf16_geometry_host_hooks.log`.
- `placement_smoke_bfp8_prefill_act_bf16.json` and `placement_smoke_bf16_hifi4.json` pass lengths 511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 8193, 16385 and 32769 on real decoder layers 0 and 3. Trace allocation tracking includes program-cache allocations. After preparation, repeated requests keep program and capture counts fixed with cache misses forbidden.
- `bfp8_prefill_act_bf16.json` and `bf16_hifi4.json` pass full-model prefill and traced teacher forcing on both AIME24 and natural-text references at the unchanged thresholds.
- `capacity_bf16_hifi4.json` passes non-aligned 262,143-token prefill (123.027 seconds), terminal decode at position 262,143, and 262,144-token prefill with live traces, allocation tracking and no new programs.

Original failures remain in `bfp8_prefill_act_bf16_initial_l1_failure.*` and `placement_smoke_bf16_hifi4_initial_k16_failure.*`. The large launcher log has a lossless gzip archive and SHA-256 entry in `large_artifacts.json`; the raw file remains locally available.

Residual scope: these placement decisions serve the normal replicated, unfused full-model path. This stage makes no new claim about optional fused or sharded-residual experimental decoder modes. Precision selection and its final acceptance checks are recorded separately in the stage report.
