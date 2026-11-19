# Serving topology audit before tuning

Independent CPU inspection, plus root source verification. Baseline: accepted
stage 9, selected combined_shared4_mixed policy, C262144/B2/TP4/EP4.
No serving profiler used.

| Boundary | Existing work | Decision / experiment |
| --- | --- | --- |
| Prefill state | Reset fixed slot; copy private state to scratch; run chunks; copy back. Shared attention KV stays in caller pool. Two copies of 239147520 bytes/rank each, including 201326592 bytes/rank compressed index keys. | Candidate 1: trace the existing scratch transfers, preserving exact operands/order. No device math/layout change. |
| Prompt history | Allocate 32 by prompt-width int64 tensor even when common sampler ignores prompt history. At C262144 this is 64MiB host allocation. | Candidate 2: construct only when penalties consume it; preserve seed reset and host compatibility. |
| Decode | B1 active: 98 model traces plus 1 sampling trace per step; B2 active: 147 plus 1. Nonblocking; persistent feedback and positions. | Retain selected path. Prior serving 103.351ms versus resident generator 102.028ms leaves about 1.3% serving overhead. |
| Input refresh | Prior 2207 device decode submissions: 2123 unchanged, 51 page-only, 33 input reloads. | Retain reload contract; rerun stale-input, queued-read and mixed traffic for changed state-copy boundary. |
| LM head / sample | Vocab-sharded BFP8/HiFi2 head, local tile-shaped candidates, candidate collective, canonical greedy split sampling to persistent token tensor. | Retain; prior non-serving split sampling about 0.482ms versus force-argmax 2.766ms. No sampler bottleneck indicated. |
| Dense/sparse decoder | TP4 attention, EP4 routed active experts, complete residual per rank; selected BFP4/BFP8 fidelity policy, BF16 activation/CCL/KV. | Preserve datatype selection and layer exceptions. Broad decoder retuning does not address the small serving gap. |
| Scheduler chunks | Current budget256; internal physical buckets1/2/4/8/16/32. Arbitrary logical lengths and continuation offsets supported. | Measure three budgets against long-prompt interference; retain context262144. |
| Batched prefill | Dense model still executes per row. Prior 128/32/32c32 serial burst exists. | Optional; report serial reference. Do not claim an enabled-arm speedup or batched-equals-solo proof. |

Prior stage9 first128 prefill took 6.347919s and included first prefill32
capture (1.714680s). Later singleton128 prefills took 4.6186/4.6331s without
captures. These generator event intervals explain why new warmed HTTP
benchmark rows are required; they are not substitute HTTP metrics.

Potential further work: runtime-count compressed-index prefix copy; omit LM
head on intermediate prefill chunks; per-slot whole decode trace; traced
reset. Each requires new preparation/lifetime/correctness evidence. Decisions
are recorded against actual experiments and the optimize stop rule.
