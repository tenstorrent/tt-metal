# Advisor challenger result: no change

The frozen batch-32 incumbent remains the shipped decoder. Its weighted best
repeat was **1.338830 ms**, with a **0.018284 ms** repeat spread/noise floor.
No advisor-derived configuration beat it outside that floor, so ties correctly
go to the incumbent and `tt/optimized_decoder.py` remains byte-identical to the
frozen source (`9922fc46...a153`).

The model has 40 sliding-attention and 8 full-attention layers. Both kinds were
captured once, after the incumbent measurement, at the shipped precision:
sliding QKV BFP8, full QKV BF16, O BF16, and MLP BFP8. BF16 DRAM-sharded
eligibility was explicitly enabled. Each capture considered five matmuls and
advised four DS placements.

The advisor agreed with the incumbent on QKV and all three MLP projections.
The material disagreements and weighted decoder results were:

| Chain | Window share | Candidate ms | Result vs 1.338830 ms |
| --- | ---: | ---: | --- |
| Sliding O, non-DS/interleaved | 6.693% | 1.325980 | 0.012850 ms faster, inside noise: rejected tie |
| Full O, non-DS/interleaved | 9.578% | 1.340956 | 0.002126 ms slower |
| Sliding split-layout→norm→reshape→rotary chain, block-sharded | 8.101% | 1.335696 | 0.003134 ms faster, inside noise: rejected tie |
| Full split-layout→norm→reshape→rotary chain, block/width-sharded | 13.528% | 1.333879 | 0.004951 ms faster, inside noise: rejected tie |
| Sliding O + layout/norm/rotary chain | combined | 1.321871 | 0.016959 ms faster, inside noise: rejected tie |
| Full O + layout/norm/rotary chain | combined | 1.336562 | 0.002268 ms faster, inside noise: rejected tie |

The advisor's `compute_config`/math fidelity was treated as traced state, not
advice. No sparse-matmul, SSM, or other terminal tracer boundary exists in this
dense decoder; both layer kinds were captured end-to-end.

Correctness note: this is a no-change result, and the final source is
byte-identical to the frozen incumbent. Therefore the challenger preservation
oracle passes: no candidate correctness change was shipped. This is distinct
from the current environment's absolute-PCC audit. That audit passes full
attention but reports sliding decode PCC 0.988923 against 0.993 on the untouched
incumbent. The real-weight oracle skipped because no local Gemma checkpoint is
present. `measurements/final_incumbent_oracle.json` records the passing
preservation result and the failing current absolute audit separately; the
latter is not presented as a pass or attributed to an advisor candidate.

Only `tt-perf-report` CSVs are retained in `tracy/`; raw Tracy captures are not
part of this deliverable.
