# AutoFix: BF16 paged prefill program configuration

## Starting evidence

- Source-only diagnosis: [AUTODEBUG_bf16_cache.md](AUTODEBUG_bf16_cache.md), written
  before implementation edits by a fresh xhigh subagent. The repository AutoDebug
  CLI was attempted first, but its nested sandbox could not execute reads because
  `bwrap` was unavailable; the native fresh-context subagent completed the pass.
- Original command, run by the coordinating agent:

  ```bash
  OMP_NUM_THREADS=8 timeout 600 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_datatype_smoke --config models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/configs/kv_bf16.json --output models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/smoke_kv_bf16.json
  ```

- `smoke_kv_bf16.log` reports CB end 1590848 B above Blackhole L1 limit
  1572864 B. The JSON contains passes through 1025 tokens; the next prompt is 4097.

## Hypothesis experiment

**Hypothesis:** BF16 K/V tiles exceed the L1 budget of the BFP8-oriented
Q64/K256 program when a later full 1024-token prefill chunk needs two Q buffers.

**Experiment:** independently trace the actual cache geometry and factory CB
arithmetic, then execute the unchanged `ConfiguredChunkedPrefillAttention` class
extracted by AST with tensor metadata and operation-recording stubs. This calls
the real Python selection/padding/slicing logic without importing TTNN, executing
C++ or accessing hardware. The allocation calculation is source-derived, rather
than an allocator execution.

**Result:** the 4097-token request allocates 132 pages, with a 576-byte aligned
page-table row. The first failing call has Q `[1,4,1024,512]`, cache
`[132,1,32,512]`, and offset 1024. CBs occupy 1479232 B above L1 base 111616 B,
exactly reproducing end 1590848 B. The shorter 1025-token request already chooses
the existing Q64/K128 boundary program because its 1152-token cache capacity
cannot cover a K256-rounded read end of 1280. Its success does not validate the
larger program.

**Verdict:** allocation cause verified by source arithmetic and a discriminating
host probe; device correctness remains pending.

## Fix and host verification

`tt/optimized_decoder.py` selects the existing `boundary_program` when
`head_dim >= 512 and k_cache.dtype == ttnn.bfloat16`, before computing read padding.
This matches the current full-attention 512-wide head configuration and leaves
the sliding-attention path and every BFP8 program-selection branch intact.
Default BF16 Q64/K128 reduces the reported CB end to 1033792 B, saving 557056 B.
At context 262144, the larger page table yields end 1065984 B, leaving 506880 B
against the hardware L1 limit. This changes neither the supported context nor
logical prompt alignment, cache geometry, compute fidelity or precision.

Saved probe: [verify_bf16_cache_config.py](verify_bf16_cache_config.py).
Before/after evidence: [before JSON](bf16_cache_config_before.json),
[after JSON](bf16_cache_config_after.json), [probe log](bf16_cache_config_probe.log).
The 16 cases cover BF16 and BFP8 short tails, the failing full second chunk,
page-capacity boundaries, the final maximum-context chunk, selecting a row from
a 32-user page table, and internal multi-chunk output assembly. Every modeled
allocation and rounded read fits after the fix. All BFP8 records are identical
before and after, including boundary fallback behavior.

Commands run successfully from the repository root:

```bash
python models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/verify_bf16_cache_config.py models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/bf16_cache_config_after.json models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/bf16_cache_config_before.json
python -m py_compile models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/verify_bf16_cache_config.py
pre-commit run --files models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/verify_bf16_cache_config.py
git diff --check -- models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py
```

No C++ or CMake changed; no build was required. Frozen implementation SHA256:
`2d2e52e029bb840e47e9ea17b7684c5bd6c807f1c56b38f647801a75977d7d55`.

## Final status and pending device evidence

Minimal fix implemented; source and host checks passed. **Device verification is
pending with the coordinating agent.** This subagent did not access hardware and
does not claim numerical correctness, measured performance, or a completed
full-model candidate evaluation.

Required next checks:

1. Rerun the original BF16 smoke with its default lengths
   `31 32 33 1023 1024 1025 4097` and retain a separate post-fix output/log.
2. Add `--lengths 2047 2048 2049 4095 4096 4097` to exercise later full chunks,
   aligned ends, nonaligned tails and cache-capacity rounding for both BF16 and
   the baseline BFP8 configuration. Synchronize/read back through the smoke's
   existing position and token assertions.
3. Evaluate the BF16 full-model candidate against the stage's accuracy and
   capacity requirements. The 262144-context L1 calculation addresses this SDPA
   buffer issue; it does not establish whole-model DRAM capacity.

The main agent was notified before the edit and when the source became frozen.

## Device confirmation by stage owner

The original BF16-cache smoke plus expanded nearby boundaries exited0 on the
final source:31/32/33,1023/1024/1025,2047/2048/2049,4095/4096/4097.
`smoke_kv_bf16_fixed.json` records all12 passing cases, actual cache policy,
final positions and trace counters; `smoke_kv_bf16_fixed.log` records execution.
Command: `OMP_NUM_THREADS=8 timeout 600 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_datatype_smoke --config models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/configs/kv_bf16.json --output models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/smoke_kv_bf16_fixed.json --lengths 31 32 33 1023 1024 1025 2047 2048 2049 4095 4096 4097`.
The L1 overflow is fixed. This reduced two-layer check does not establish
full-model BF16-cache maximum-context capacity or accuracy.
