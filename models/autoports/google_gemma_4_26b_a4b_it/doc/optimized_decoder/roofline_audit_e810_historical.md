> Historical source-only audit for runtime e810. Its uses of “current” refer to that earlier runtime. This is not final v4 profile evidence; use [the final audit](final_roofline_audit.md).

# Final roofline accounting audit

No numerical accounting correction is needed for the current decoder. This is a
source/artifact review with CPU arithmetic checks; final profiler totals are not
validated by this artifact. Runtime SHA-256: `e81018299b722aa81eae0e4e9ec3ec638520adcb3bdd85d3c2c8b429dfa0e370`.
Source paths, anchors, and hashes are recorded in
[roofline_audit_e810_historical.json](roofline_audit_e810_historical.json).

The [summarizer](../../tests/summarize_perf.py) uses complete first-firmware-start
to last-firmware-end windows, including every native layer operation and internal
gap. Decode uses the mean of 128 successive windows at positions 4096–4223.
[Reconciliation](../../tests/reconcile_perf.py) checks same-run command/CSV
provenance and separates refreshed host-loop timing from fixed-position host
batches; arithmetic timing differences do not isolate individual causes.

## Useful prefill FLOPs

For `S=4096`, `H=2816`, `Q=16`, the formulas count a multiply-add as two FLOPs.
Full attention computes tied K/V once ([TiedQKV](../../tt/fused_decoder.py)).
Shared/expert gated MLPs each contain three projections. Only eight experts per
token contribute useful work.

| Term | Sliding | Full |
| --- | ---: | ---: |
| Q/K/V and output projections | 283,467,841,536 | 401,579,442,176 |
| Shared MLP: `6*S*H*2112` | 146,163,105,792 | 146,163,105,792 |
| Active experts: `6*S*H*704*8` | 389,768,282,112 | 389,768,282,112 |
| Router: `2*S*H*128` | 2,952,790,016 | 2,952,790,016 |
| Causal QK/PV | 60,137,930,752 | 274,945,015,808 |
| **Total** | **882,489,950,208** | **1,215,408,635,904** |

Projection formulas are `2*S*H*(2*Q*256 + 2*8*256)` sliding and
`2*S*H*(2*Q*512 + 2*512)` full. Attention is `4*Q*head_width*pairs`, with
`pairs=S*1024-1024*1023/2` sliding and `S*(S+1)/2` full. Padding, masked or extra
expert work, scalar normalization and transcendental operations are excluded
from useful FLOPs; their time remains included.

## DRAM estimate

[NativePagedAttention](../../tt/optimized_decoder.py) uses batch-one non-MLA GQA,
a DRAM interleaved query and FP32 accumulation. Its dynamic decode K chunk is
128 tokens. Source derivation is in [roofline_native_basis.md](roofline_native_basis.md).
`read_start=floor(logical_start/128)*128`; `read_end=ceil((position+1)/128)*128`.
BFP8 K/V bytes are `2*(read_end-read_start)*kv_heads*head_width*(1088/1024)`.

| Native K/V input reads | Sliding: 8 heads × 256 | Full: 2 heads × 512 |
| --- | ---: | ---: |
| Rounded tokens/head, first 127 positions | 1,152 | 4,224 |
| Rounded tokens/head, last position | 1,024 | 4,224 |
| Mean tokens/head | 1,151 | 4,224 |
| **Mean BFP8 bytes** | **5,009,152** | **9,191,424** |

Other bytes follow CSV operand shapes, dtypes and memory placement. Direct router
BF16 weights cost `2816*128*2 = 720,896` bytes per generic read; the same shape in
FP32 costs 1,441,792 bytes. Native SDPA BF16 output and removed conversions are
accounted for directly through the current CSV. Sparse weights use recorded
`nnz/128` (decode: `8/128`); current sparse activations/outputs are in L1. Cache
updates count one page read/write per cache. Index/page-table operands retain one
read/write. BFP8/BFP4 storage includes exponents (1.0625/0.5625 bytes per element).
Extra worker rereads, NOC/reduction traffic and profiler writes are excluded;
these are operand estimates, not controller counters or complete-layer totals.

## Peak basis and verification

The common peak is `120*4096*1.35e9/4 = 165.888 TFLOP/s`, with `512 GB/s` DRAM per
ASIC. Blackhole FLOP/cycle constants are recorded in `tests/nightly/sdpa_perf_utils.py:26`
and `tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py:807`.
The [official P300c specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html)
states two 120-core ASICs, 1.35GHz and 1,024GB/s card bandwidth; 512GB/s is the
per-ASIC share. This is a common theoretical normalization for mixed fidelity
and SFPU work, not measured FPU utilization. Percentages are not clamped.

The only identified inconsistency was a docstring saying metadata was excluded;
the parent applied the wording correction before this artifact was saved.
No numerical estimator, runtime, or acceptance-gate change was needed.

CPU assertions passed for useful FLOPs, dtype sizes, router weight bytes, sparse
weight selection, all 128 native KV positions and peak arithmetic. Re-run from
the repository root (stdlib only; the complete assertion source is in the JSON):

```bash
python3 - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path('models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/roofline_audit_e810_historical.json').read_text())
exec(compile(audit['cpu_verification']['python_source'], '<roofline_cpu_check>', 'exec'))
PY
```
