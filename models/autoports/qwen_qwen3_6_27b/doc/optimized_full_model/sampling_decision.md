# Greedy sampler decision

The selected complete path uses the common split sampler with physical local
Top32 and semantic greedy k1/p0/temp1. The prepared logits are
`[1,1,32,62080]` per rank, padded/masked to65536 inside the sampler, and its
output is the persistent `tt_out_tok` consumed by the next model replay.

Fresh serial TP4 tests,200 measured iterations, report **0.512781ms** for split
sampling and **2.725168ms** for force-argmax. Both pass three changed-input
unique-maximum tests across all32 physical rows. Exact commands, environment and
source hashes are in `sampler_{split,argmax}.command.json`; outputs and timings
are in the corresponding JSON files. These are isolated sampler measurements;
all64 resident decode is reported separately.

Both implementations return an actual maximum in every tied case. For a
65-way same-rank tie at indices70000..70064, split TopK selects70001 and argmax
selects70000. Split TopK does not promise PyTorch's lowest-index tie ordering;
this is the unchanged stage6 semantic contract, not a newly observed nonmaximum
error. No token-exact HF equivalence for tied maxima is claimed. Cross-rank
ties in the other case select17 on both paths.

The new device profile (`perf/sampling.csv`, `perf/runtime_matmul_policy.csv`)
contains the specialized large-index TopK implementation, two candidate gathers
and device sampling. It contains no generic `TopKDeviceOperation`, force-argmax
or full-vocabulary all-gather. Candidate exchange moves128 values/indices per
row after local selection. The complete sampler is0.453–0.473ms of kernel work
per token, around1.5% of extrapolated full-model device work; the head is
1.55–1.57ms. There is no dominant sampler defect to mask with force-argmax.

Greedy, top-k20/top-p.95 sampled generation, repeated seeded resets, per-slot
parameter changes and permutations are covered by `sampling_contract.json`,
`full_trace_tracking.json` and the final full-path run. Sampling-parameter
caching skips only identical complete padded temperature/k/p uploads. It never
skips requested RNG resets, and slot remapping invalidates its signature.
