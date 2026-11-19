# Decode direct coefficients and native cast placement

Source-only integrated candidate, built on the fused-corrected-matmul snapshot `v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot` (SHA256 `8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa`). No device was opened by this builder; integrated correctness, latency, and acceptance remain unmeasured. Parent-owned full-decoder runs decide whether this candidate is retained.

The candidate reads original BF16 TILE activation row-zero bits directly into the pair-flipped coefficient columns consumed by the already-measured streaming FPU multiply/native cumulative scan. This bypasses the ordered projection's DRAM conversion, untilize, reshape/gather, and coefficient tilize sequence without changing the BF16 values or native scan/carry arithmetic. It retains native FP32 `fast_reduce_nc` in ascending block order. Router and attention-output projections place the existing explicit native typecast result directly in DRAM, removing the subsequent placement copy; the integer round-to-nearest-even typecast remains intact. No direct BF16 pack substitution is used: that earlier option failed the strict midpoint gate.

Only `from_state_dict` and `linear` change. Setup adds one static direct-coefficient helper. Three existing singleton ordered branches gain a leading `if terminal_scan` early return; the existing branches remain byte-independent AST copies below that guard. Existing call sites retain `decode and cache_boundary` (three attention/output sites) and the router's `ordered_router`, itself set only by `decode and cache_boundary`. The default flag remains false. Public singleton prefill, internal tail/prefill K1024 processing, non-aligned sequence handling, full-attention V5, normalization, recurrence/cache semantics, and the new fused corrected matmul are unchanged. No context capacity or public contract change is introduced.

The builder proves complete module AST restoration to 8cda after removing precisely the three added early-return guards, one setup assignment, and appended helper/reader literal. All existing static classes remain exact. The embedded direct helper is alpha-equivalent to the measured helper; reader/compute/writer/product-writer strings match the measured device kernel snapshots. It compiles the resulting Python source without importing TTNN or Torch. The runtime helper has no tensor host conversion, runtime AST, test import, untilize/gather/tilize, or input reshard.

## Actual producer layouts

Layout claims are based on source, with source hashes recorded in the provenance JSON:

- Standard `run_decoder.py` and `v5_layer_stream_probe.py` uploads omit `memory_config`. `ttnn/ttnn/operations/core.py` forwards this to `Tensor(mem_config=...)`; `ttnn/cpp/ttnn-nanobind/pytensor.cpp` resolves it with `mem_config.value_or(MemoryConfig())`. The `MemoryConfig` declaration documents that default as **interleaved DRAM**. The decoder's explicit input normalization restores `original_memory = x.memory_config()`. Packed attention and QZ therefore receive **DRAM in these harnesses**, not the optional L1 component-fixture placement.
- Decode `_forward` explicitly moves normalized `mlp_input` to DRAM before `_moe`, so the actual ordered router receives **DRAM**.
- Attention-output projection consumes `nlp_concat_heads_decode(gated, num_heads=32)`. Native `compute_output_specs` specifies **WIDTH_SHARDED L1**, 32 row-major cores, shard shape `(32,128)`; the following reshape retains that memory configuration.
- This integration changes none of those producer placements. L1 attention/router component gains are conditional evidence for those input configurations; they are not substituted for actual-path measurements.

## Existing component evidence

`v6_direct_coefficient_stream_device.json` passed all 40 fixtures, including actual positions 227/738/5799, random and zero inputs, two unchanged prefill cases, interleaved L1, DRAM, and original output width-sharded activation placement. All 38 singleton coefficient/product/partial checks and all final FP32/BF16 outputs were exact. Every timed variant passed 100 exact trace replays. The raw CPU mapping separately covers all 65,536 BF16 bit patterns. These are bounded component results, not a full-decoder claim.

Selected complete-adapter traced medians in milliseconds:

| Input / projection | Streaming baseline | Direct coefficients | Interpretation |
|---|---:|---:|---|
| Actual harness DRAM router | 0.050261151046 | 0.037221657112 | Component improvement |
| Actual output WIDTH_SHARDED L1 | 0.152244232595 | 0.122443656437 | Component improvement |
| Actual harness DRAM attention K512 | 0.179442204535 | 0.181002705358 | **0.87% regression; retained explicitly for full integration decision** |
| Actual harness DRAM QZ K512 | 0.180398486555 | 0.165826641023 | Component improvement |
| DRAM attention K1024 | 0.192631967366 | 0.195371196605 | 1.42% regression; prefill remains unchanged |
| Optional L1 attention K512 | 0.184535956942 | 0.163388554938 | Conditional L1 fixture evidence |
| Optional L1 QZ K512 | 0.185611564666 | 0.146893947385 | Conditional L1 fixture evidence |
| Optional L1 router | 0.055190059356 | 0.035813543946 | Conditional L1 fixture evidence |

`v6_projection_cast_placement_device.json` separately passed all 32 fixtures, 256 signed midpoint controls (below/at/above BF16 ties), and 100 exact replays. Complete-adapter traced medians were router `0.050296215340 -> 0.048975250684 ms` and output `0.148276705295 -> 0.146979698911 ms`. Combining these independent changes is source-proven but still requires integrated device validation. Component savings are not added to predict full-decoder latency. Eager timings and complete sample arrays remain in each measured JSON and are copied into integration provenance.

## Reproduction and frozen artifacts

CPU build command (already run successfully; it refuses to overwrite existing evidence):

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_build_direct_coefficient_snapshot.py
```

For a repeat source-only build, pass a fresh `--output-dir`.

- Builder: `tests/v6_build_direct_coefficient_snapshot.py`, SHA256 `5c174312edbb59323ea983d226c5868379f365877908ac5b7176fdd73f69a3b2`.
- Candidate: `v6_corrected_fma_full_v5_direct_coeff_cast_placement_fused_matmul_integration.py.snapshot`, SHA256 `c823e6f6f8480a9550276856a7aa660b9308ff598c6565a38c2bc54460440ece`.
- Same stem `.provenance.json`: all dependency hashes, exact restoration checks, unchanged guards, producer-layout source hashes, complete component timing records, and limitations.
- Same stem `.builder.py.snapshot`: exact builder bytes used.
- Same stem `.reader.cpp.snapshot`, `.compute.cpp.snapshot`, `.writer.cpp.snapshot`, `.product_writer.cpp.snapshot`: frozen runtime kernel sources. The candidate embeds the reader and exact measured helper statically, reusing the already-frozen matching streaming compute/writer literals and scheduling function from 8cda.

No production decoder, existing helper, native source, earlier measured snapshot, or integration builder owned by another agent was edited. The combined path must pass the original decoder PCC/determinism/cache/long-context gates and beat the best correct traced-decoder candidate before adoption. It does not establish the graph-fusing Stop Rule or stage completion.
