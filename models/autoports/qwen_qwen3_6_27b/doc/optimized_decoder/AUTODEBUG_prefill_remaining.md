# Remaining prefill audit: profile80 / default81

Source-only AutoFix follow-up, 2026-11-12. No device access, main-source edits, or changes to previously delivered candidates. New candidates are unselected until device correctness and timing pass. Parent measurements cited below are existing artifacts, not new child measurements.

## Concrete config experiment

`perf/profile80_bfp8_0/prefill_perf_report.csv` shows four M512 packed attention matmuls totaling 4.635158 ms, four M512 attention output matmuls totaling 1.776225 ms, and two M512 MLP down matmuls totaling 1.604453 ms. The latter six use **1x1 output subblocks**. All use 88 of 110 cores at grid11x10. Minimal SwiGLU costs 2.070921 ms. This makes subblock geometry a material experiment before further arithmetic changes.

For M512, current grid11x10 implies pm=2. Preserve `out_block_h=pm`, `out_block_w=pn`; the earlier DRAM-sharded partial-width addressing failure is not revisited here.

| Projection | Physical N | Current pn/subblock | First candidate | Second grid candidate |
|---|---:|---|---|---|
| packed_gdn and residual | 17408 | 50 / 1x2 | 2x2, K8 | grid10x8: pn55, 2x1, K4 or K8 |
| out_proj and residual | 5120 | 15 / 1x1 | 2x1, K8 | grid10x8: pn16, 1x4, K16 |
| MLP down | 5120 | 15 / 1x1 | 2x1, K8 | grid10x8: pn16, 1x4, K16 |

Grid8x8 gives packed pn68 and output/down pn20, both allowing 1x4. It uses fewer cores and larger full output blocks; it must earn a measured win. Grid11x10 uses only eight actual M workers for M512, so its nominal 110-core size is not an argument by itself.

Global K16 already has a hard source/runtime failure: `prefill70_k16.log` reports 2,394,048 B static buffers versus 1,572,864 B L1. This does **not** reject K16 for narrow output/down projections. The 2D factory sizes input buffers with `out_block_h*Kblock` and `out_block_w*Kblock` and buffering depth; reducing pn50 to pn15 materially reduces B-buffer storage (`matmul_multicore_reuse_mcast_2d_program_factory.cpp:418-434`). Allocation still needs device validation. L1 input placement should be tried independently, with its transfer included in timing; at M512 down-projection input alone is 17 MiB across the device, so free per-core L1 must be checked with the program buffers.

New `candidates/prefill_role_configs.py`, generated from frozen `default81.py`, adds only prefill 2D overrides. Its default path matches the source exactly. Example policy:

```json
{
  "prefill_subblock_h": true,
  "prefill_projection_configs": {
    "linear_attn.out_proj.weight": {"grid": [10, 8], "in0_block_w": 16},
    "mlp.down_proj.weight": {"grid": [10, 8], "in0_block_w": 16}
  }
}
```

Correction weights inherit the base key's settings automatically. Supported per-key controls: `grid`, `in0_block_w`, `subblock_h`, `subblock_w`, `input_l1`. `subblock_h` is a ceiling, selected down to a legal divisor for short tails; a fixed height2 would otherwise break arbitrary M33 tails with pm1. K blocking must divide K tiles, subblocks must divide per-core dimensions, and FP32 destination subblocks stay at most four tiles. DRAM-sharded decode programs and compute fidelity remain unchanged. For isolation, first run only `prefill_subblock_h:true`; then role-specific grids/K/L1, each against the same correct baseline.

`make_prefill_role_configs.py --source <snapshot> --output <new-path>` refuses to overwrite a candidate. `prefill_role_configs.generation.json` records hashes. `tests/test_prefill_role_configs.py` ran 324 CPU source-config cases across B1/B3/B32, S1/17/33/65/128/511/512/513/1025, four projection roles and three grids. Empty overrides produced identical matmul calls; all tails retained legal full output blocks. These checks establish construction, not kernel accuracy or performance.

## Movement-only corrected projection

`candidates/prefill_chunk_add.py` replaces the flat corrected projection helper. Baseline computes all primary FP32 chunks, concatenates them, computes/concatenates all residual FP32 chunks, then adds to BF16. Candidate runs the **same per-chunk primary/residual matmuls**, adds each pair to BF16, then concatenates BF16 chunks. No weight, fidelity, input row, K blocking, matmul shape, or rounding boundary changes. If primary and residual caps differ, the helper retains the original independent partitioning.

Profile80's primary/residual packed concatenations cost .330521/.331377 ms; output projection concatenations cost .119374/.118160 ms. The candidate replaces those four FP32 concatenations with two BF16 concatenations, while splitting the elementwise add by existing row chunks. It also removes duplicate input slice operations between the primary and correction passes. Net benefit requires measurement: extra add dispatches may offset some movement savings. No speedup is claimed.

`candidates/chunk_add_prefill.py` freezes this change over combo73, with B1 S>32 allowed into the flat helper. Compact decode [1,B<=32,W] dispatch is unchanged. To compose with current defaults:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_prefill_chunk_add.py \
  --source models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/default81.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/<new_name>.py
```

`tests/test_prefill_chunk_add.py` ran 108 CPU arithmetic/partition cases with BF16-exact results, identical recorded matmul row order and geometry, and single BF16 concatenation replacing two FP32 concatenations when applicable. It includes cap256/cap512/cap64, mismatched-cap fallback, tails1 and nonaligned user lengths. `prefill_chunk_add_cpu_checks.json` records results. Device TTNN padding, concat and math still need real-path checks.

## Generalized prefill norm and quality77

Do not select generalized prefill norm from isolated PCC alone. `prefill67_probe.json` passed all seven requested shapes and both real folded gamma tensors, with deterministic stable supplied output. B32/S128 BF16 equality is approximately 99.99936%, so it is **not bit-identical**. Dense SFPU reduction squares five 1024-element tiles, folds those tiles, then reduces columns and rows. The previous explicit-sum policy uses TTNN reduction on the original row layout. Different FP32 reduction/rsqrt rounding can move BF16 rounding boundaries. Both evaluate the same RMSNorm formula; this is an arithmetic-order hypothesis, not evidence that indexing is wrong.

`quality77_jobs.py` changes both `sfpu_prefill_norm=True` **and `mlp_packer_l1_acc=True`** versus passing combo73. The latter is not restricted to prefill; it changes decode MLP accumulation as well. Thus quality77 minimum .99316881 cannot establish that the generalized prefill norm caused the regression. Prefill PCCs are .9999664704 versus .9999660916; a better aggregate prefix PCC does not prove a better recurrent initial state. Isolate (a) norm only, (b) MLP packer only, keeping exact seed/prefix/decode schedule. Retain .995 long-sequence gates. Exact-tree normv2 is being handled separately by the parent team.

Reader/writer source mapping remains sound: logical row r maps to `(r//S)*round_up(S,32)+r%S`, including B1 and nonaligned S. Only the last logical row of each batch zeros that batch's padding. Runtime args are assigned explicitly by core coordinate, so singleton CoreRangeSet sorting cannot permute users here (unlike the earlier native sharded QKV bug). Gamma/zero remain held across rows; per-row input/inverse/output CBs drain; pack format resets before the next row. There is no public S modulo restriction. Long-loop performance and arithmetic are separate outstanding hardware checks; mapping tests do not prove them.

## Canonical integration and final contract checks

1. Keep explicit prefill phase selection for packed full attention; do not infer phase solely from S. S1 prefill and compact decode both exist. Keep phase-packed weight memory accounting (74.375 MiB extra BFP8 storage for the default 14336-wide packed projection) in context-capacity analysis.
2. Keep flat MLP activation until after down projection. For B>1/nonaligned S, use logical TTNN reshape, not a metadata-only reinterpretation of physical padded rows. Current helper uses TTNN reshape correctly. Minimal matmul validates **padded** M alignment (`minimal_matmul_device_operation.cpp:104,135,310-317`), not public logical S alignment.
3. Keep recurrent state FP32 through internal prefill chunks and cast once into the caller's persistent BF16 buffer. Current `prefill_forward` does this. A final internal S1 chunk takes compact SFPU GDN, whose state validator accepts both FP32 and BF16; the dtype switch itself is legal. That chunk does use a different convolution/GDN route and merits a regression test. Existing `default81_l0_b3_s513.json` already passes aggregate prefill .99996923 and decode .99998492 with deterministic repeated runs; this is useful, but not a substitute for per-token contribution and boundary tests.
4. Retain `run_contract --layer 0` and `--layer 3`: B1 S1/31/32/33/127/128/129/511/512/513/1025 plus unaligned continuation and mixed rows. Use final canonical optimized defaults; record provenance. Add/retain B3/S513 and B32/S33/65 real decoder checks and compare output plus recurrent/paged state at unchanged .995 gates. B31 guards compact tail-core work. Full-attention prefix start511/512/513 verifies paged tail writes.
5. Any new config requires the same B32 128-prefix/512-step stress used to select combo73, even if decode code is unchanged: altered prefill matmul ordering can change the recurrent initial state. Run watcher separately from profiling, and keep output/state buffer identities stable under replay. Final profiler and latency must use the selected source, not these unmeasured prototypes.

Validation commands executed here (CPU only):

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_prefill_chunk_add.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_prefill_role_configs.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/prefill_role_configs.py
```
