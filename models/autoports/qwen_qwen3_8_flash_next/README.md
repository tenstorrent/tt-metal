# Qwen3.8-Flash-Next on four Blackhole chips

Full-model batch-1 performance: **TTFT 7.039 s**, **7.207 decode tokens/s/user**
(138.754 ms/token), prompt **128** / generation **128**, all 48 layers,
supported context allocation 262144. This is serial, warmed public generation
with traced model + common on-device sampling, token feedback and caller token
readback. Allocation tracking, watcher and profiler were disabled for timing.
See [the raw performance result](doc/full_model/acceptance_perf_final_v1.json).

The separate model-trace-only window measured **7.238 steps/s** (138.153 ms/step);
it repeats a resident input token and excludes sampling/readback, so it is not
teacher-forcing accuracy. The installed readiness teacher-forcing check includes
sampling, readback and forced-token uploads: its tracked 203-token AIME prompt /
100-output workload measured **6.716 decode tokens/s/user** and is diagnostic,
not a comparable clean latency benchmark.

Current full-model HF accuracy is **97% / 100% / 100%** prefill and
**96% / 100% / 100%** traced teacher forcing (top-1 / top-5 / top-100, 100
predictions each). Retained-state resumed prefill passes. The blind matched
qualitative suite found zero failures in each backend's ten samples for each
of six prompts; 128-token thinking-mode outputs do not establish final-answer
accuracy. See [full-model evidence](doc/full_model/README.md).

The generator retains the optimized TP4 attention / EP4 expert decoder,
BF16 paged K/V, inherited dtype/fidelity policy and inter-layer residual layout.
The 127 measured steady decode steps perform no host token, position or
page-table refreshes, no cache resets and no trace captures or releases.

Full 48-layer batch-32 state/HF checks pass: both 2704-prediction histories score
97.818% / 100% / 100% top-1 / top-5 / top-100, with exact relocation logits,
tokens and state.
Cross-active-width numerical sensitivity is documented with matched HF controls
in [the disposition](doc/full_model/batch_geometry_disposition.md).

The final reduced-layer profile at context allocation 262144 verifies complete
operation coverage. Split greedy takes 0.429 ms in merged kernel sums versus
2.887 ms for force-argmax; both select the correct greedy token. These are eager
operation measurements, separate from the traced full-model throughput above.
See [terminal attribution](doc/full_model/terminal_final_analysis.md).

The once-only maximum-context acceptance passes: a non-aligned 262143-token
prompt plus two generated tokens consumes 262144 positions, with all 12 QSA
layers coherent on all four ranks and finite terminal logits. This establishes
capacity and execution, not long-context answer quality.

Independent stage review: **clean-pass**, with no required work.
See [the review](doc/full_model/stage_review_round1.md) and
[local checkpoint records](doc/full_model/commits.json).
No vLLM adapter is included.
