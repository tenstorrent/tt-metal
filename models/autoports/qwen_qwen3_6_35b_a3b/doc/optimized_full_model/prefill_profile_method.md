# Prefill kernel attribution method

`report_prefill_kernels.py` is a CPU-only streaming reader for the existing native
`ops.csv` schema. It writes a measured-kernel CSV, a compact setup physical-bucket CSV and one summary JSON in a
fresh output directory. It does not collect hardware, copy/archive captures,
compute artifact hashes, or change model/test source.

```bash
python models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/report_prefill_kernels.py \
  models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/final/profile_reduced \
  --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/final/profile_reduced/prefill_kernels
```

Run after native conversion produced `ops.csv` and `performance.json`. If
`native_program_metadata.json` exists, the script uses its compute-kernel
fidelity inventory by native device/program ID. That inventory may cover only
decode programs; missing entries are marked, never silently called verified.
CSV source names and scalar fidelity remain available, with the scalar's known
incomplete-multi-kernel limitation explicitly labeled.

The script checks reduced real layers0/3, the canonical128-token request and
profile-only/non-headline mode. It recognizes the exact `PERF_TTFT` and
`PERF_TTFT_END` signposts used by `run_full_model_performance.py`, and stops after
that first complete window. Four device ranks are required inside the window.
Missing device duration inside the measured window is an error; missing setup/warmup durations are counted
separately and excluded from timing sums, never converted to zero.
The measured CSV groups operation/program/shape/role/rank; the setup CSV
aggregates only role/physical-axis/rank to avoid reproducing the large warmup
inventory. The measured CSV reports counts, summed and mean device kernel duration and firmware duration in microseconds, with source
CSV line bounds. These are operation-work totals, **not TTFT or additive device
latency**. Four ranks execute in parallel; do not sum them into latency.

There are two explicit regions:

- `measured_TTFT128_diagnostic`: one warmed public128-token prefill, terminal
  norm/head, logits handoff, position upload, traced greedy sampler and first
  token readback. Per-component profiler drains/synchronizations make its host
  TTFT diagnostic. Eager prefill and traced sampler rows remain separate.
- `setup_warmup_capture_unsegmented`: all earlier native rows, including weight
  setup, preparation of physical buckets, warmup, trace capture and possible
  decode warmups. These rows do not have request/chunk boundary signposts and
  must never be presented as warmed measured prefill or headline performance.

Chunk inference is intentionally narrow. Native padded/logical WZYX shapes are
preserved. Embedding input width identifies a **physical token buffer**; its
valid tail length is not encoded. A non-decode SDPA query sequence axis can
identify a logical attention chunk. Other input-Y dimensions remain physical
shapes, not logical chunk sizes: expert/group/head/recurrence dimensions and
matrix M cannot be relabeled as tokens without additional evidence. Roles such
as generic custom kernel or non-terminal matmul remain explicitly unresolved.
The table therefore reports observed physical sizes and verified logical sizes
where available; null means unknown, not zero. The current bounded source plans
linear attention chunks≤64, full attention≤256 with alignment fallbacks, MoE
policy chunks and64-token embedding carriers. Those source limits do not prove
which logical sizes were measured. Do not insert unobserved256-token or tail
rows into the measured128-token table.

This small report complements `profile_full_model_components.py`,
`summarize_multichip_perf.py` and `tt-perf-report`. It does not replace native
capture-loss/execution-coverage validation, advice, complete kernel precision
inspection or separate clean all40 TTFT boundaries. Per-chunk attribution that
cannot be resolved from native shapes remains an explicit evidence limitation;
setup rows alone cannot close that gap. Attach the output paths to the stage's
single evidence manifest alongside the original logs/capture paths.

CPU schema smoke: the existing stage6 capture
`../full_model/performance/profile_reduced_v4_drains` produced
`prefill_schema_smoke_v2/`: 1,556 measured groups from 12,460 untraced and124
traced device rows across four ranks. Its only inferred measured logical SDPA
chunk was128. The setup region contained718,136 untraced and4,236 traced rows,
kept separate. This checks parser/schema compatibility only; these historical
rows are not fresh stage7 performance evidence. Repository hooks passed for the
report script.
