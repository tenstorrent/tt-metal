# AutoFix Report: maximum-context full attention

## Starting Evidence

`AUTOTRIAGE_max_context.md` records the initial source-only diagnosis. The
original `full_max_capacity.log` run completed TP1 but printed no TP4 progress
after reservation. Two captures (`full_max_triage.txt` and
`full_max_triage_second.txt`) showed idle model workers and dispatch waiting for
host submissions. The original process was terminated by its hardware owner
after preserving evidence; it did not produce a correctness result.

The runtime under investigation was
`d3dee9c50f549b6a48e657f0f8b2745a42c436a81ebecb8d8b099e1ae34f454b`.
Initial Inspector evidence established 153 distinct full-attention SDPA kernels
and 117.33 seconds of summed kernel compile durations. Each scalar chunk offset
changes SDPA's program key and K-length compile-time arguments. The later
interval without compiler activity remained unexplained.

The root hardware owner executed recovery and both control runs. This reporting
agent inspected source and logs only.

## Hypothesis Experiments

### Deterministic SDPA, fabric, or final-page deadlock

Prediction: the same layer, capacity, cache table, nonaligned final chunk,
reservation, and trace settings reproduce the failure on the unchanged runtime.

Experiment: instrument `_forward` entry/exit every 16 chunks and periodic Python
tracebacks using `tests/diagnose_multichip_progress.py`; run TP4 alone with the
original settings. No production runtime change was made.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.diagnose_multichip_progress \
  --tp 4 --fused-tail --hybrid-experts --optimized-shared \
  --layer 5 --length 262143 --repeat-input --steps 1 --trace --check-cache \
  --prefill-timing-samples 1 --reserve-full-stack \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/full_max_diagnostic.json
```

Result: exit 0 and normal device close at 23:33:18 UTC. Initial and warmed
prefill each completed all 256 chunks, including offset 261120 with 1023 valid
tokens. Trace decode at position 262143, repeat replay, replica equality, cache
readback, and close completed. The first 60-second traceback landed in chunked
SDPA and was followed by progress; the second landed in output readback and was
also followed by progress.

Verdict: a deterministic maximum-context worker/CCL/page-bound defect is refuted
for this runtime and test case. An intermittent original host-side issue remains
possible. The TP4-only control has `passed: null` and empty PCC lists; it is an
execution control, not cross-implementation accuracy evidence.

Evidence: `full_max_diagnostic.log`, `full_max_diagnostic.json`.

Fix: none. No kernel, fabric, precision, capacity, or runtime change was kept.

### Initial specialization/readback cost explains the whole original delay

Prediction: remaining cold SDPA variants compile while progress continues; a
warmed paired run completes without the earlier silent interval.

Experiment: rerun the original paired command with the now-populated JIT cache,
same runtime hash and full reservation.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder \
  --fused-tail --hybrid-experts --optimized-shared \
  --layer 5 --length 262143 --repeat-input --steps 1 --trace --check-cache \
  --prefill-timing-samples 1 --reserve-full-stack \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/full_max_capacity_verified.json
```

Result: passed, with normal device close at 23:35:47 UTC. JIT cache telemetry
reports 3722/3722 hits. Maximum-capacity numerical evidence:

| Check | Result |
| --- | --- |
| Prefill TP1/TP4 PCC | 0.999960157292875 |
| Decode TP1/TP4 PCC | 0.9999247235518492 |
| Minimum shuffled-cache PCC across K/V and four ranks | 0.9999710211573951 |
| Output replicas | Exactly equal |
| Trace replay | Deterministic |
| Additional resident DRAM reservation per device | 21,072,183,296 bytes |
| Context allocation / last decode position | 262144 / 262143 |
| Runtime SHA256 | Unchanged |

Verdict: the initial specialization overhead is verified, and the unchanged
warmed maximum-capacity workload passes. Whether compilation/readback explains
the entire original no-progress interval remains uncertain: no original host
stack established its exact location. Do not label the original delay a fixed
fabric hang or infer a repaired deadlock from cache warming.

Evidence: `full_max_capacity_verified.log`, `full_max_capacity_verified.json`.

Fix: none. No runtime change was needed for the successful verification.

## Final Status

The intended maximum-context correctness and capacity check now passes on the
existing runtime. The original cold-run host delay did not recur in the focused
control or paired verification and has no established root cause. Diagnostics
are isolated in the test wrapper; the implementation retains the 262144-token
capacity and nonaligned-length contract.

The reservation exercises other resident allocations using anonymous DRAM
buffers. It does not execute a complete full-model stack. The paired fixture
uses repeated real layer input and validates against optimized TP1, not an
independent HF maximum-context full-model reference. This report makes no new
performance claim from the single warmed host timing sample.

No build was required for these documentation/diagnostic changes. If the silent
host interval recurs, preserve a live host stack plus current phase/chunk
progress before choosing a code fix; do not reduce the context contract or
modify fabric based on the existing captures.
