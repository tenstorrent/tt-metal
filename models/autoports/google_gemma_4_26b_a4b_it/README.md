# Gemma4 26B A4B IT — full model

Full-model TTFT **2115.07 ms** and trace-verified token-out decode
**49.22 tokens/s/user** (4096 input, 128 generated, batch 1,
one concurrent request; four Blackhole ASICs, TP4; warmed request trace).
Teacher-forcing decode: **49.84 tokens/s/user** on the separate
161-input/100-position AIME accuracy workload; it includes host token injection
and is not the autoregressive performance result.

The full model retains the optimized multichip decoder and 262144-token context.
Prefill and decode top-5/top-100 are both 100% over 100 AIME token positions.
See [full-model evidence](doc/full_model/README.md) and [work log](doc/full_model/work_log.md).
Independent [stage review](doc/full_model/stage_review_final.md): **clean-pass**.
Stage06 is complete; local checkpoint provenance is recorded in the work log.

---

## Post-pipeline work

2026-10-03: the user accepts the combined **2/5 SWE** result (all five submitted,
2h14m54s dispatch-to-finish) and requests actual release validation with benchmarks,
GPQA10, TerminalBench5 and original SWE5. [Current release configuration/status](doc/tti_release/RUN_NOTES.md)
records the accepted policy, immutable image, exact subsets and outstanding CI results.
This is not a claim of full accuracy readiness.

The section above is the unchanged pipeline artifact. Subsequent
human-directed work continued from that result:

- Warmed S128/O128/C1 TTFT improved from **419.10 ms to 95.24 ms** (77.3%).
- Remote 4K/C1 throughput improved from **43.04 to 50.68 tokens/s/user**
  (17.74%). Focused local 4K tests also improved C8 by 3.03% and C16 by
  4.44%.
- [QB2 CI completed the Qwen-style 23-row serving benchmark](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36490378756)
  with no request failures.
- [QB2 CI completed all 40 GPQA-Diamond samples](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36501826660/job/109194269630)
  without inference errors and scored 25/40 (62.5%).
- Agentic eval coverage remains partial.
  [Eval job 109283453627](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36530661132/job/109283453627)
  runs the five-task Terminal-Bench 2 subset followed by the five-task
  SWE-Bench Verified subset, with SWE-Bench serialized. It does not run GPQA.

The post-pipeline changes, human prompts, measured gains, and remaining limits
are summarized in [pipeline interventions](doc/PIPELINE_INTERVENTIONS.md).
Detailed evidence is available for
[TTFT optimization](doc/ttft_optimization/README.md),
[remote benchmarks and evals](readiness_vllm/ttft_optimization_remote/README.md),
and [TSU optimization](doc/tsu_optimization/closure.md).

Remaining work:

- Optimize standard and agentic eval execution so a full release run can
  finish in reasonable time.
- Add explicit targets for serving benchmarks and evals.
