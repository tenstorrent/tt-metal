Functional at **80 tasks**: run 42332785769 is valid, with **56/80 correct (70%; Wilson 95% interval 59.23–78.94%)**. Evaluation took **4.281695 h**, device job **4.568056 h**, and workflow **4.573056 h**. Published/GPU reference context is 84.3%/83.33%; scores are reported without a score gate.

All 80 semantic documents match the preregistered cohort. The harness selects membership in ascending dataset order, while its configured unsorted labels are permuted; this audit uses semantic identities and hashes. All nested 40 prompts, visible outputs and retained chain hashes match the accepted C8/40 run. The same80 C4 comparator remains pending, so no same80 speed saving or causal concurrency benefit is credited.

The run used full60 TT `eef534e5cd6bcaf37957647de4d8d4eff4f0e047`, TTI `42dc40754e465cf8c9ec993027ff59d3c936832c`, stock plugin `ca7a18246e4111629ed6cb4a7d1192002749fccf`, and proven image digest `bf1ada4081f7fa338ed8495a02324390d3b6124861cd625a0d2fb2d6a255d425`. Planned clients8, B8, max_num_seqs8 and pool20480 stayed fixed. Context262144, generated-token cap126976 and request timeout14400 s were preserved. The exact generation configuration and pinned reference revision are in RESULT.json. This is the stock HTTP EVALS_COMMON route without task containers or resource overrides.

The phase archive contains 307551 contiguous paired steps, 80 reconstructed chains, 139 remaps, and 2114531 active decode samples. There are no missing/duplicate steps, ambiguous continuity edges, loss flags or pending completions. Raw archive size is 345600831 bytes. Chains are associated through unique prompt lengths, accepted40 hashes, and complete native reply usage/visible hashes; remaining47 and86 follow complete-set elimination after187 and160 are identified. Physical row order is never used as request identity. Prefill token IDs are absent, so physical sample counts are not asserted as exact client-emitted usage.

Thirteen empty answers reached the unchanged cap: semantic documents7,12,15,20,27,35,47,61,86,90,139,147,160. Twelve have long exact periodic suffixes. Document86 has a period57 suffix of50152 samples. Document139 has a nonperiodic degenerate final65536-sample tail using six single-character tokens (four uppercase letters and two punctuation characters, no whitespace). These failures are retained in the score; evidence does not show useful reasoning cut short. Eleven additional answers were incorrect. New loop mechanisms remain eligible for separately bounded diagnosis; this audit changes neither outputs nor limits.

The longest capped physical chain lasted6973.043560 s (1.936957 h); terminal tail after the last noncapped chain was4057.083223 s. Summed overlapping agent durations were29.146892 h, including22.964068 h on capped chains. Eight agents occupied223907 decode steps /10992.087371 host interval seconds, with440.132736 s at one active agent near the end. These are host completion-boundary observations, not profiled device execution time.

Native v1 logging exposes80 received IDs and78 complete unique JSON replies. Only7 of the first8 have fully decodable replies. Native envelope timestamps establish observed initial reply durations of69.27–5971.21 s; outer CI timestamps can be delayed postrun replay and are not live completion timestamps. Complete retained chains and official samples support the scored run, but this run does **not** prove all initial8 native replies or a no-score dry pass.

The device forecast was5.333959 h versus measured4.568056 h, within the7.5 h box. The reconciled workflow forecast was6.203404 h versus4.573056 h. Actual dispatch-to-device queueing was17 s; measured non-evaluation device time was0.286361 h. No device time was spent on this host audit.

| Prospective cohort | Linear evaluation h | Workflow point h | Plus one measured longest tail h | Plus full4 h timeout sensitivity h |
|---|---:|---:|---:|---:|
|160|8.563389|9.699195|11.636151|13.699195|
|198|10.597194|11.733000|13.669956|15.733000|

Workflow projections include measured0.286361 h non-evaluation time and0.849444 h queue allowance. They extrapolate observed work density and are guesses, not bounds. The extra measured tail is1.936957 h; the full4 h request timeout stays unchanged. The160 measured-tail sensitivity fits12 h, while198 does not; both full-timeout sensitivities exceed12 h. Admission also requires remaining stage/dry schedule and preceding valid rung. No larger run is admitted by this report.

Exact per-document hashes, cap evidence, initial request identities, occupancy, source bindings and evidence paths are recorded once in RESULT.json. Private analysis is under bringup/artifacts/gpqa80-c8-final-audit; raw prompts, reasoning and token IDs remain private.
