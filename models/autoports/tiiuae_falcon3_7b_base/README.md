# Falcon3-7B-Base TTNN autoport

Full-model status: complete for the `1x4` Blackhole P300c TP4 target. The
optimized path preserves the selected multichip decoder, BFP4/LoFi weights,
BFP8 attention/MLP/CCL/KV policy, persistent two-link ring collectives, and
the BF16 32-core L1 width-sharded inter-layer residual. vLLM is intentionally
outside this stage.

| Batch-1 full-model measurement | Result |
| --- | ---: |
| Cold TTFT, 128-token standard workload | 78.073 ms |
| Warm TTFT, same prompt | 25.083 ms |
| Trace-verified caller-visible token-out, device feedback + split greedy | **75.411 t/s/u** |
| Queued device-only model/sampling trace pair (diagnostic) | 75.687 t/s/u |
| Trace-verified teacher-forcing decode | 75.676 t/s/u |
| Readiness-runner teacher-forcing decode, including callback overhead | 40.850 t/s/u |

The AIME24 correctness gates are prefill top-1/top-5/top-100
`92%/100%/100%` and decode `93%/100%/100%`. See
[`doc/full_model/README.md`](doc/full_model/README.md) for exact commands,
trace evidence, qualitative review, capacity accounting, and limitations.
