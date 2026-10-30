# Functional decoder work log

Stage 01 complete. Independent stage review: clean-pass. Base db97382f (upstream d80d9482).
Model google/gemma-4-31B-it, revision 842da3794eaa0b77d5f08bae87a17459d91ff475.
New implementation derived from transformers 5.12.1 HF Gemma4 reference and generic TTNN ops.
No removed Gemma implementations or Gemma-specific tt_transformers branches used.

Startup: read bringup/run_contract.json and bringup/agentic_evals/inputs.json;
validated installed AutoDebug through scripts/environment.py and enabled skill inventory.
TTNN resolves to this checkout. Four local Blackhole devices enumerated by TTNN;
1x1 mesh open/close passed. Firmware 19.17.0, KMD 2.10.0. No reset performed.
`tt-smi` and `fuser` are absent from PATH; initial health used TTNN open/close.
Machine motherboard B850M-C warning falls back to bus_id as tray_id; single-device
open/close passes. No fabric use in this stage.

Pinned AutoConfig: 262144 context, 60 layers, 5376 hidden, 21504 intermediate,
32 query heads; sliding 16 KV heads x 256, full 4 KV heads x 512.
Sliding window 1024. No shared-KV layers, MoE or per-layer inputs in target config.
Full layers use K projection for V before separate K/V normalization; scale=1.
Both checkpoint shards already exist in HF cache; layer tensors loaded by indexed keys.

Commands:
- `python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py"` passed.
- `python` TTNN get_num_devices/open_mesh_device(MeshShape(1,1), trace_region_size=0)/close passed.
- `TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.google_gemma_4_31b_it.tests.run_decoder`
  initial smoke.log failed at nonexistent BlackholeComputeKernelConfig Python symbol.
  Corrected to shared WormholeComputeKernelConfig used on this architecture; retry smoke2.log.

Installed skill's generic chunked-SDPA warning is older than target source: target
sdpa_nanobind.cpp exposes sliding_window_size on chunked SDPA. Validate it directly.

First real-weight successes (not final stage gates): sliding32 prefill .9999167323,
traced decode .9998826981; full33 .9999895096/.9999846816. Sliding1025
.9999127388/.9998167157; full1025 .9999918938/.9999897480. Sliding batch32
seq33 .9999372363/.9999030828. Each trace ran three identical replays with allocation
tracking enabled and program-cache allocations included. Logs in this directory.

Performance setup: `uv pip install --python python_env/bin/python pip tt-perf-report`
installed pip26.2.2 and tt-perf-report1.5.0 in the active environment.
Hardware sysfs identifies p300c and 1350 MHz. Official specification:
https://docs.tenstorrent.com/aibs/blackhole/p300.html and
https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html
identify 120 Tensix cores and 512 GB/s per ASIC. For BF16/HiFi4 the theoretical
whole-ASIC peak basis is 120 * 4096 * 1.35e9 / 4 = 165.888e12 useful FLOP/s;
4096 operations/cycle/fidelity-phase basis matches installed tt-perf-report's
ArchitectureSpec. Use the full ASIC peak, not the smaller grid used by any op.
A reported core-grid budget is not a replacement denominator.

Long-context full attention found a required failure: `full_context.log` has
prefill262143 PCC .9986207485, exact262144 prefill .9981801510, eager/traced decode
.9902030230 and changed-position replay .9935083389. The stage remains incomplete.
No acceptance threshold or context reduction was made.

AutoDebug launcher was invoked from the autoport directory with personal CODEX_HOME,
ambient OPENAI_API_KEY/CODEX_API_KEY removed, gpt-6-astra/medium, inspection-only scope.
Its sandbox preflight failed because bubblewrap was absent (autodebug_launch.log).
Retried with explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the existing unrestricted
execution/subagent authorization (autodebug_retry.log). No source edits or device
experiments were authorized in the investigator.

Discriminating experiment `python -m models.autoports.google_gemma_4_31b_it.tests.probe_long_attention`
constructs exact HF K/V for262144 tokens and compares TT versus HF queries through
paged decode SDPA, precise exponential and chunk variants, then the layer finish.
Initial probe had a missing required SDPAProgramConfig constructor argument; corrected
only the probe's configuration construction. QUERY_PCC=.9999957681 in that attempt.
The target runtime's default paged decode config is K chunk32 and max cores/head/batch1
(sdpa_decode.cpp), so long reductions warrant direct numerical checks.

AutoDebug completed inspection and wrote ../../AUTODEBUG.md; AutoFix evidence is
../../AUTOFIX.md. Controlled exact-HF-cache probes isolate the default full decode
recurrence: exact exp alone does not pass, while sequence splitting does. Applied
full-only K128/core-cap16/precise-exp configuration; no sliding or prefill arithmetic
changed. `long_reference_control.json` independently compares long SDPA against
HF-style eager and explicit FP32 attention, passing layer PCC >.999998.

Sliding final context run (`sliding_context_final.json/.log`) passed logical
prefill262143, exact262144 prefill and final-position decode with allocation tracking,
nonzero physical page base, sentinel pages, conversion guards and updated positions.
Full repair validation command:
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.google_gemma_4_31b_it.tests.run_decoder --layer 5 --seq 262143 --skip-exact-maximum --result models/autoports/google_gemma_4_31b_it/doc/functional_decoder/full_context_fixed.json`
Output: full_context_fixed.log. The skip option reuses unchanged exact262144 prefill
PCC from full_context.log; only decode dispatch was repaired. It does not reduce
supported context or skip the actual262143-token TTNN cache-producing prefill.

Full decode repair passed with actual TTNN-produced cache: full_context_fixed.json, decode PCC .9999799132 and changed-position replay .9999930859. The context contract now records full262144 support for both layer kinds with no reduction.
Validation queue launched with `python -m models.autoports.google_gemma_4_31b_it.tests.run_validation_queue --skip-context`; long context evidence is reused, and each queued command/environment/status is recorded in validation_queue.json.


Final implementation validation supersedes the older partial-repair evidence above.
The metadata-only logical-tail view removed late FillPad program-cache allocations
for a new logical34 in the already prepared64-row bucket. No tracking exemptions.
Boundary cases change tokens and page mappings on every case, including short reuse.
PCC now accumulates in FP64 (earlier large-batch FP32 values rounded to1).

`python -m models.autoports.google_gemma_4_31b_it.tests.run_validation_queue`
completed all12 serialized jobs, exact command/env/exit records in validation_queue.json;
stdout validation_queue_final.log. Final full-context files are
sliding_context_post_view.json/.log and full_context_post_view.json/.log.
Both exact262144 prefills and position262143 traced decode pass. Minimum maximum-
context prefill PCC is0.9981801950; neither kind reduces advertised capability.
The ten other final runs cover B1/8/16/32, boundary/resume, changed positions/pages,
determinism, runtime guards, six eager-prefill/decode-trace interleavings and watcher.
Watcher logs inspected with `rg --no-ignore -n -i 'fatal|exception|error|overflow|corrupt|invalid|sanitize'`;
no matches in either final watcher directory. Context gate rerun exits0 (context_check.log).

`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m pytest --confcutdir=models/autoports/google_gemma_4_31b_it/tests -q models/autoports/google_gemma_4_31b_it/tests/test_functional_decoder.py`
passes4 synthetic real-shape tests (pytest.log). The two warnings are generic installed
SWIG type deprecations, not device or correctness failures.

`python -m models.autoports.google_gemma_4_31b_it.tests.run_profile_queue`
collects the4096/128 workload for both kinds in B1,32,16,8 order. Exact Tracy/report/
calculation commands and CSV hashes are recorded by profile_provenance.json.


First Tracy capture overflowed profiler DRAM marker storage (collection.log under
tracy/sliding_attention/batch1_overflow). The whole-layer calculation rejected it:
only47 incomplete decode sessions instead of128. No performance number from that
capture is accepted. The harness now drains profiler buffers before/after phases
and after each8replays, outside every whole-layer device window. Recollection has
128complete42-op sliding decode windows. Periodic host drains/refreshes are excluded
from per-layer device time; no end-to-end throughput claim is made.


Independent review classified profiler PROGRAM CACHE HIT=False as stale metadata:
`ttnn/cpp/tools/profiler/op_profiler_json.cpp:175-181` caches short_str including the
first emission's hit bit; `ttnn/api/tools/profiler/op_profiler_serialize.hpp:385-388`
reuses it on later hits, and `tools/tracy/process_ops_logs.py:475-492` parses it.
Trace replays contain the captured device programs; this field does not invalidate
warmup. No kernel/runtime changes or additional profile were needed for this artifact.
The optional WASM trace-copy warning refers to its default generated/.logs path,
while this command uses an explicit output directory. Host/device CSVs, local .tracy
and complete per-replay markers exist under each collection's actual output path.

All eight profiling jobs passed; profile_queue.log and profile_provenance.json
record both kinds at B1,32,16,8. Summary and exact timing/roofline basis are in
../functional_decoder.md; all accepted runs contain 128 complete traced decode windows.


Independent `$stage-review` used a fresh gpt-6-astra/xhigh subagent, read-only against
this live worktree. Verdict: clean-pass; `stage_review.md` records scope, reproduced
metrics, source hashes, anomaly ledger and residual risks. No required findings remain.
All stage work is confined to this autoport; no optimized-decoder, multi-chip,
full-model or serving work began. Only this tt-metal repository was touched.
Local checkpoint SHAs are recorded below; no push is authorized or performed.


Commit hooks passed all checks except their first isort pass, which sorted imports
in six harness files. The decoder and every non-import AST node are unchanged.
formatting_provenance.json maps the measured source hashes to formatted hashes;
evidence_manifest.json retains both. Original collection manifests remain intact.
Independent review rechecks this cosmetic delta before the commit retry.
