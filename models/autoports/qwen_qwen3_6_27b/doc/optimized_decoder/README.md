# Qwen3.6-27B optimized decoder

This stage adds the single-device `OptimizedDecoder`; it does not enter
multichip, full-model, or serving work.  The selected policy is BFP4 weights,
LoFi compute, and eight-bank width-sharded DRAM weights for decode
(`bfp4_all_dram_w8`). Prefill retains an already-materialized interleaved BFP4
copy so neither runtime path reshards weights.

## Result

All numbers are warmed host medians in milliseconds. The fused decoder is the
correct baseline. Both attention layer kinds passed their existing PCC bars
with exact repeated-run determinism and `throw_exception_on_fallback=True`.

| layer kind | batch | fused traced decode | optimized traced decode | change |
|---|---:|---:|---:|---:|
| full | 1 | 2.364 | 1.007 | -57.4% |
| full | 32 | 2.561 | 1.209 | -52.8% |
| linear | 1 | 3.079 | 1.672 | -45.7% |
| linear | 32 | 21.386 | 19.125 | -10.6% |

Warmed sequence-33 prefill medians:

| layer kind | batch | fused | optimized | change |
|---|---:|---:|---:|---:|
| full | 1 | 2.906 | 2.234 | -23.1% |
| full | 32 | 68.666 | 12.853 | -81.3% |
| linear | 1 | 82.407 | 81.361 | -1.3% |
| linear | 32 | 2569.201 | 2512.844 | -2.2% |

Traced optimized decode PCC was 0.998843 or better for full attention and
0.999953 or better for linear attention. Non-aligned sequence-33 prefill PCC
was 0.999758 or better for full attention and 0.999992 or better for linear
attention at both batches. Public sequence alignment constraints were not
added. Paged KV cache and recurrent-state mutation are exercised by two traced
steps, and restoring identical cache state produces exactly equal outputs.

Final profiler and `tt-perf-report` artifacts are under
`tracy/final_w8_{full,linear}_b{1,32}/`. They confirm the selected DRAM-sharded
BFLOAT4_B/LoFi matmuls at both batches. Timing above comes from the separate
clean warmed logs in `logs/`.

Large-M prefill uses an explicit 8-column by up-to-10-row multicast program
once physical M reaches 10 tiles. The small batch-1 point retains TTNN's
default factory because forcing the large-M config there regressed full
attention from 2.244 to 2.955 ms. At batch 32 the explicit config improves full
attention from 39.425 to 12.858 ms and linear attention from 2538.571 to
2513.046 ms. Candidate and selected logs are named `prefill_config_*` and
`prefill_selected_*`.

## Commands

```bash
pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_synthetic_pcc.py --mode prefill --sequence 33 --batch 1 --decoder optimized
python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --decoder optimized --optimization-policy bfp4_all_dram_w8 --kind full --batch 1
python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --decoder optimized --optimization-policy bfp4_all_dram_w8 --kind linear --batch 32
TT_METAL_WATCHER=10 python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --decoder optimized --kind full --batch 32 --perf-iterations 20
```

The context capability remains unchanged. Cache layout/dtype did not change.
The optimized setup also releases the now-unused unpacked BF16 projection
sources; two BFP4 phase-specific packed copies occupy fewer bytes than the
fused baseline's packed plus unpacked BF16 projections. The proof and artifact
references are recorded in `doc/context_contract.json`.
