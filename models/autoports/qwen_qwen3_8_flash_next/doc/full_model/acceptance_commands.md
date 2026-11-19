# Acceptance queue after trace-capacity and correctness repairs

These are prepared commands, not executed device evidence. Run serially under
the device owner's lane after the outstanding full48/trace-capacity checks pass.
Use fresh output/log names; archive any earlier installed-runner artifacts before
reusing their fixed paths (`readiness/prefill_resume_check.json`,
`readiness_autoregressive`, and `qualitative/tt_outputs.json`). This queue does
not start vLLM. No reference generation command belongs in it.

CPU preflight `text_preflight_v2.json` passes: exact installed AIME128 greedy
cache, all66 HF qualitative cache entries, pinned tokenizer/chat rendering and
EOS `[248046,248044]`, and AIME203+100/top100 reference geometry. Shared prompt
lengths are60,67,75,61,66,62. The public generator owns their nonaligned chunking.
The six-bucket canonical prefill and direct-DRAM LM head preserve the runner's
public tensor/token interfaces; the runners do not reach into prefill inputs or
LM-head layouts. Source compatibility is not hardware correctness evidence.

```bash
source python_env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_METAL_WATCHER TT_METAL_TRACE_ALLOC_TRACEBACKS
unset TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_PROFILER TT_METAL_STREAMING_PROFILER
export TT_METAL_PROFILE_PERF_COUNTERS=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
acceptance_module=models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_acceptance
acceptance_evidence=models/autoports/qwen_qwen3_8_flash_next/doc/full_model
```

Tracked full48 prefill/teacher-forcing top1/top5/top100 plus the installed resumed
prefill gate, using the advertised capacity. Keep top5≥98%,top100=100% and all
installed resume thresholds unchanged. Its teacher-forcing timing is diagnostic,
not the clean performance result.

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 \
 timeout --signal=INT --kill-after=60 7200 python -u -m "$acceptance_module" \
 --accuracy --resume --capacity 262144 \
 --output "$acceptance_evidence/acceptance_accuracy_resume_final_v1.json" \
 > "$acceptance_evidence/acceptance_accuracy_resume_final_v1.log" 2>&1
```

Clean B1 token-out performance separately disables allocation instrumentation.
It includes TTFT, traced on-device token feedback, and a separate fixed-token
logits-only trace window (the latter is explicitly not teacher-forcing accuracy).

```bash
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 \
 timeout --signal=INT --kill-after=60 3600 python -u -m "$acceptance_module" \
 --perf --capacity 262144 \
 --output "$acceptance_evidence/acceptance_perf_final_v1.json" \
 > "$acceptance_evidence/acceptance_perf_final_v1.log" 2>&1
```

AIME128 autoregression reuses the exact installed HF cache, preserves raw TT
fixed-step output, then slices both logical outputs at their first configured
EOS before automated checks and direct human reading. Read both completions for
coherence, repetition, language drift and early divergence; the automated check
alone is not a qualitative verdict.

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 \
 timeout --signal=INT --kill-after=60 3600 python -u -m "$acceptance_module" \
 --autoregressive --capacity 262144 \
 --output "$acceptance_evidence/acceptance_autoregressive_final_v1.json" \
 > "$acceptance_evidence/acceptance_autoregressive_final_v1.log" 2>&1
```

TT66 qualitative calls use six greedy and60 sampled outputs, exact saved prompt
tokens, seeds0..9, temperature1/top-k20/top-p.95, and128 tokens. The actual device
sampler is used; host compatibility mode is not acceptance evidence. Existing HF
controls suffice; do not run `run_controls.py` without its preparation selectors,
since its default launches HF generation. After TT completion prepare the blind
packet and follow `qualitative/README.md` for hash-bound independent ratings.

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 \
 timeout --signal=INT --kill-after=60 7200 python -u -m "$acceptance_module" \
 --qualitative --capacity 262144 \
 --output "$acceptance_evidence/acceptance_qualitative_final_v1.json" \
 > "$acceptance_evidence/acceptance_qualitative_final_v1.log" 2>&1
python "$acceptance_evidence/qualitative/blind_compare.py" \
 --tt "$acceptance_evidence/qualitative/tt_outputs.json"
```

Max context is last and exactly once. The exclusive attempt file must not exist;
do not remove it to retry. Prompt262143 plus2 outputs consumes262144 positions,
including one decode. The dyadic tail uses16+8+4+2+1 after8191 full32-token chunks,
so there are8196 prefill replays, followed by one decode replay.
The12-hour bound is intentionally loose; previous3–5-hour estimates predate the
new head/trace changes and remain uncertain. A hard capacity limit requires the
separate authorized one pass/fail bracket, not arbitrary repeats of this test.

```bash
test ! -e "$acceptance_evidence/max_context_attempt.json"
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 \
 timeout --signal=INT --kill-after=60 43200 python -u -m "$acceptance_module" \
 --max-context --capacity 262144 \
 --output "$acceptance_evidence/acceptance_max_context_final_v1.json" \
 > "$acceptance_evidence/acceptance_max_context_final_v1.log" 2>&1
```

The harness independently rejects an existing attempt file before opening the
mesh. SIGINT permits its `finally` evidence writes; the later forced kill is a
last-resort bound, not a promise of graceful hardware recovery. Once-only marker
creation occurs immediately before the long context test, after model setup.
No stage completion is claimed by any single driver result.
