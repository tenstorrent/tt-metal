# Initial independent stage review

Date: 2026-07-20 UTC. Verdict: `more-work-needed`.

The first review worker did not return a verdict and was interrupted. A fresh replacement reviewer then audited the full-model implementation and evidence. It found the correctness, context, capacity, trace, sampler, qualitative, Watcher, and runtime-fallback contracts supported, but identified one unresolved performance-accounting issue:

- The inherited optimized-decoder artifact measured one layer at sequence position 17 (`0.318242 ms/layer`), while the full-model trace measured positions 128 through 255. Multiplying that unlike workload by 40 produced only a provisional `12.7297 ms` lower bound and did not explain the measured full-model model trace.
- The then-unrun depth-sweep script also carried a stale 7B reference value and incorrectly described CCL state as BFP8 instead of BF16.

Required remediation was a corrected same-workload depth sweep including all 40 layers at the official 128-prompt/128-replay workload with a full 32768-token cache, plus committed result/log hashes and an attribution of decoder slope, fixed terminal cost, sampling, and orchestration. The reviewer required any demonstrated stack regression to be fixed before rereview.

The remediation is recorded in `results/full_model_depth_sweep.json`, `results/full_model_depth_sweep.log`, `results/perf_summary.json`, the report, and the work log. It found a near-perfect linear depth fit and no growing full-model wrapper boundary; a fresh final review is required for closure.
