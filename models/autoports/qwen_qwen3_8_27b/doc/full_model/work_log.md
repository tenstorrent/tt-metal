# Full-model work log

Stage6 started from clean tt-metal commit `7550299ba23` on the existing branch.
Target: Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Hardware: four local Blackhole p300c chips, `MeshShape(1,4)`, ring fabric,
`TT_MESH_PASS_THROUGH_THREAD_POOL=1`. `device_list.log` and `mesh_smoke.log`
record health and successful open/close. No reset required at startup.

## Inherited contract and topology plan

| Boundary | Implementation | Candidate/decision |
| --- | --- | --- |
| Embedding | BF16 hidden-column TP4 lookup; one hidden all-gather | Preserve decoder replicated residual; vocab-sharded lookup/reduction is a comparison candidate |
| Decoder stack | Selected MultichipDecoder default; shared TT_CCL | No inter-layer conversion/collective; preserve rejection ledger in Stage5 |
| Final norm | BF16 activations; checkpoint gamma + 1 | Local replicated hidden norm; sharded norm comparison belongs only if terminal layout changes |
| LM head | Untied vocabulary-column TP4, BF8/HiFi2, FP32 accumulator | Common LMHead1D chunk/DRAM-sharded programs are tuning candidates |
| Sampling | Common TTSampling, local top32, candidate all-gathers, greedy k1/p0/temp1 | Compare force-argmax and split greedy; never compare generic k32 as greedy |

Carried policy: all decoder projections BFP4/LoFi; BF16 activations, residual,
norm and CCL; FP32 recurrent state and projection accumulation; BFP8 paged K/V.
Prefill fill casts K/V to cache dtype; decode update remains BF16. Page32,
local full KV head1, linear value heads12. Decode residual is L1 width-sharded
on40 cores for B1, public DRAM for B2..32, with no wrapper work between layers.
Stage5 collective/rejection/inter-layer records remain authoritative and unchanged.

## Sampling comparison / AutoFix

`AUTODEBUG_sampling.md` records a fresh independent source diagnosis. Sampling1D
has incompatible method arity and undefined local-index buffers; its low-level
contract also lacks the TTSampling greedy tie handling and request management.
Selected common TTSampling with generator-owned separate sampling trace; no custom
sampler. Native output `[1,1,1,32]` is the persistent embedding token input.
Physical sampling rows32 are independent of decoder logical batch. This is
an API accommodation, not32 model copies. TP4 common logprob calculators return
None; do not advertise logprob support without implementing that missing contract.
Request seeds must advance on device in the sampling trace; seed changes at request
boundaries are explicit. Sampler parameter tensors remain stable through replay.

## Initial smoke commands

Shell environment for TT tests: `PYTHONPATH=.:$PYTHONPATH TT_MESH_PASS_THROUGH_THREAD_POOL=1`,
using `python_env/bin/python` and installed bringup `environment.py` exports.

- `tests/run_full_model.py --output doc/full_model/probe.json`: tokenizer API returned BatchEncoding;
  fixed probe rendering to tokenize the rendered chat string explicitly.
- `tests/run_full_model.py --output doc/full_model/probe2.json`: real layers0/3,
  real terminal weights/cache, S33/G4, split trace replay, repeated output equality passes.
  Output spaces are a reduced-stack smoke, not a qualitative verdict.
- `tests/hf_reference.py --output readiness_aime24_chat.refpt`: fresh exact100-token HF
  AIME24 chat reference. Sequential HF controls use strict memory-mapped checkpoint
  loading to avoid unnecessary full-model CPU copies. Status/results recorded below.
- `tests/run_full_model.py --full --length 128 --generate 128 --output doc/full_model/full_smoke.json`:
  first full-stack smoke, in progress at this entry.

The startup environment lacks tt-smi in this checkout's venv; health used the
existing `/home/mvasiljevic/tt-metal/python_env/bin/tt-smi` from Stage5. No dependencies installed.

## Completed validation and terminal selection

Fresh AIME HF reference completed with exact100-token metadata. `readiness.json`
passes99% top1/100% top5/100% top100 in both modes on the initial terminal.
Its autoregressive phase caught the shared runner stripping the final chat newline
(203 ->202 tokens); preserving text in `run_autoregressive.py` repairs the harness.
`readiness_final.json` uses the selected terminal: prefill99%, decode98%, both
top5/top100100%. It writes both AIME completions and all six shared128-token TT
controls. Warmed S128/G128: TTFT88.561ms,39.0792 token-out t/s/u; S203/G100
teacher forcing: TTFT101.146ms,38.7634t/s/u. This Python job completed and closed
devices, but editing the shell launcher while it was active made the launcher exit2.
The launcher is now frozen; an exit0 repeat is mandatory. No model/runtime error
was hidden by the saved JSON. `readiness_final.launch.log` preserves the wrapper error.

`full_context.json` executes the complete64-layer original terminal at262143
tokens plus last-position decode and262144 prefill. No context reduction.
The selected head adds a separately counted DRAM copy to the memory plan and
will receive a final capacity confirmation.

`contract_b3_final.json` and `contract_b32_final.json` pass fixed slots, mixed31/33
prompts, inactive recurrent/conv state, logits deterministic across slots and
repeated runs, changed/unchanged page tables, greedy/non-greedy alternation, explicit
host compatibility and device-only model/sampler/replay guards. B32 also tests
continuation31+2 inside a page. These are reduced real layers0/3; fullB32 remains
the final all-layer batch gate. A stricter same-input physical-page-permutation
logits oracle was added afterward and will run in the final batch check.

`shapes_final.json` passes1,31,32,33,4095,4096,4097 plus repeats33,31,4097 with
trace allocation tracking. Unknown-prefill signature invalidation fixes the
63 persistent-program-buffer conflicts recorded in `shapes.log`; repeats assert
no additional trace captures. `AUTODEBUG_trace.md` and `AUTOFIX_trace_prefill.md`
describe the evidence. The source-only AutoFix agent additionally verifies23
host control-flow cases (`cache_contract_host.json`, `sampling_params_host.json`),
including inverse temperature, atomic invalid binding, cache budget invalidation,
page-table placement/identity, active masks and compatibility semantics.

Canonical greedy comparisons at reduced S128/G128: split400.296t/s;
force-argmax226.638t/s (rejected). DRAM head block20 failed L1 by3456bytes;
block10 fits and measures415.688t/s. Sharded final norm plus DRAM head measures
433.093/434.091t/s and agrees on all128 greedy tokens with interleaved control;
final-logit PCC .9999766 (`dram_compare.json`). Common TTSampling remains selected.

`profile_split` dropped markers because no intermediate drains were used. It is
invalid performance evidence. `profile_split_drained` and `profile_selected`
complete postprocessing after bounded drains. Per-device advice-enabled reports
show the inherited BFP4/LoFi decoder policy, baselinehead996us -> selectedhead
eight compute/bandwidth-bound chunks, finalnorm95us ->6us, canonical sampling
about490us. `profile_summary.json` records raw microsecond accounting. Fullstack
profiling was never run. Raw captures are moved, not discarded, to the explicit
paths in `profile_archive.json`; compact compressed CSVs remain in the checkout.

Shared128-token controls are coherent and have no mechanical repetition or wrong
language, but TT prompts2/4/5 spend longer reasoning than HF at that cutoff.
Both HF and TT are being extended to256 tokens to review completed answers rather
than waive truncation. AIME degeneracy gate passes (`degenerate_output.json`).

## Watcher failure and AutoFix (required work)

`watcher_final` uses watcher10, O3+NOINLINE, all Ethernet checks and a fresh kernel
cache. It aborts134 with a BRISC missing-read-flush assertion in
`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`. The failure and kernel map
are preserved under `triage/`. Default triage could not use Inspector after the
automatic abort; the explicit `--dev=all` ARC/Ethernet capture succeeds and shows
healthy chips/links. `tt-smi -r` succeeds, the post-reset list has four devices,
and `mesh_smoke_after_reset.log` confirms a clean ring open/close. No further reset
or host reboot was needed.

A fresh xhigh AutoTriage/AutoFix agent identified a17,408-byte BF8 head row being
issued through a16,384-byte single-packet helper. Its native fix, build, focused
watcher regression, original watcher rerun, and subsequent fullmodel confirmation
are required before pass. See `AUTOTRIAGE_dram_head.md` when finalized.

The repair now passes: constexpr row size selects the any-length NoC helper,
which emits16384+1024-byte packets using the existing tags. The preserved ELF
confirms the old over-sized single request; no generic barrier or watcher check
was disabled. Build commands:

```bash
.github/scripts/copilot-build.sh --build-ttnn-tests
/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake --build build_Release --target ttnn --parallel 4
/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake --install build_Release --prefix /home/mvasiljevic/qwen38-full-rerun/tt-metal/build_Release --component ttnn-runtime
```

The prescribed wrapper exits1 because Docker is unavailable; the existing local
build and install both exit0. Logs: `native_copilot_build.log`, `native_build.log`,
`native_install.log`; loaded library and kernel hashes: `native_fixed.sha256`.
No toolchain or dependency installation was needed.

Focused regression command (same watcher environment/cache as original):

```bash
PYTHONPATH=.:$PYTHONPATH TT_MESH_PASS_THROUGH_THREAD_POOL=1 TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_FABRIC_OPT_LEVEL=O3 TT_METAL_CACHE=/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache-full-model-watcher python_env/bin/python -m pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -k 'wide_bfp8_row and tp4' -q
```

`native_wide_watcher.log`:3 passed,105 deselected,16.81s; all reader counts1/2/3,
three changed tensor-address iterations in eager mode. `watcher_fixed` covers
traced replay and repeats
the original generator contract plus the stronger page permutation oracle and
passes exit0. Continuation PCC .9999673; exact page-remapping logits, inactive
state, same-slot/batch determinism, both sampling modes, reset and compatibility
all pass. This directly verifies the source repair at its original trigger.

## Validation queue before final confirmation

Native repair verification; fullB32 and same-input page remapping; selected-path
maximum context and frozen-launcher exit0;256-token qualitative controls; final
performance/source provenance; independent review and local checkpoint commits.
Stage is incomplete until these gates pass.


## Final confirmation and extended quality review

`contract_full_b32` completed with exit 0: all 64 layers, B32 fixed slots, mixed
31/33 prompts, exact physical-page remapping logits, inactive-state preservation,
zero steady host refreshes and explicit host-sampling parity. Continuation PCC
is 0.99921447. The six 256-token shared TT outputs and paired HF controls have
no checker findings, but prompts 0/2/3/5 remain censored by the token limit.
A matched 1024-token extension is required to assess reasoning convergence and
completed answers. CPU HF stops when every selected row reaches EOS:

```bash
PYTHONPATH=.:$PYTHONPATH python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/hf_qualitative.py --output models/autoports/qwen_qwen3_8_27b/doc/full_model/hf_qualitative_extended.json --gen-len 1024 --prompt-ids 0 2 3 5
```

This CPU extension started only after the final unprofiled performance windows
finished; it may overlap the capacity execution, whose duration is not a latency
benchmark. No CPU HF workload overlaps the headline measurements.


`readiness_confirmed` exits 0 with prefill 99/100/100% and decode 98/100/100%
top1/top5/top100. Warmed S128/G128 token-out: 97.2439 ms TTFT and 39.0550 t/s/user;
AIME S203/G100 teacher forcing: 99.5571 ms and 38.7602 t/s/user. The final headline
uses these reproduced numbers, including the higher TTFT than the earlier run.
Both are canonical split traces; only teacher forcing uploads reference feedback.
The same process passes S262143/G2 and S262144/G1 with final position262144
(79.49 s and78.19 s capacity execution, not a latency fixture). The complete DRAM
plan is13,466,968,576 bytes/device versus34,138,688,512 physical bytes; no context
reduction. `full_model_memory.py` now verifies the recorded clean capacity result
before publishing that evidence into `doc/context_contract.json`.

The CPU-only HF extension was temporarily paused with SIGSTOP during
`tests/profile_full_model.sh profile_final` to exclude CPU contention from the
final reduced profile. Its elapsed duration includes that pause and is not an HF
performance claim. No TT process was signalled or reset for this operation.


`profile_final` and per-device report generation finish successfully after the
native repair. CPU HF was resumed immediately after collection/postprocessing.
Device0 windows: model1.8225 ms, sampling0.4897 ms, token-out2.3105 ms. The sampler
is1.91% of measured full-stack token-out latency and does not dominate it.
Actual decoder matmuls retain BFP4/LoFi; head chunks use BFP8/HiFi2 and are
classified BOTH/FLOP. `full_model_profile_summary.py` writes reproducible per-phase
hashes/totals and explicit layer-boundary stack accounting. The kernel-only
48-linear/16-full floor is22.890 ms; instrumented extrapolation including terminal
and sampling is27.010 ms, distinct from the measured25.605 ms/token full stack.
Final profiled prefill has3.099 ms device work +5.819 ms eager dispatch gaps; no
prefill speedup over the earlier baseline is claimed. `performance.md` classifies
this and all material advice. Raw captures are archived by exact path/hash;
per-device compressed input CSVs and advice-enabled tables remain in the repo.


Final profile table reconstruction from only the retained per-device `.csv.gz`
inputs passes (`profile_archive_replay.log`); combined/raw captures are separately
archived with hashes. `full_model_profile_tables.py` supports both original and
archived input layouts. Source checks pass in `precommit_sources_final.log`.
The standard `06-full-model.check.sh` reports clean AIME degeneration and full
262144-token context. These checks do not replace the pending extended quality
reading or independent clean-pass.

The extended TT command is queued after the matched HF control file is complete:

```bash
models/autoports/qwen_qwen3_8_27b/tests/run_full_model_experiment.sh qualitative_extended models.autoports.qwen_qwen3_8_27b.tests.run_readiness --only autoregressive --qualitative --qualitative-reference hf_qualitative_extended.json --qualitative-output tt_qualitative_extended.json
```

The optional qualitative output name changes only test artifact placement; the
standard readiness and generator paths remain unchanged. The original six-prompt
256-token output stays in `tt_qualitative.json`; completed prompts1/4 need no
additional run.


Independent performance review reproduces all48 phase-table hashes/counts/timings
and the representative layer accounting. The inherited Stage5 primary comparison
is48*0.421990 +16*0.306304 =25.156384 ms (isolated layer replay/synchronization
included), compared with25.605 ms/token full model. This is distinct from the
22.890 ms kernel-only accounting floor; the full model amortizes per-layer call
overhead, so the difference is not an isolated terminal-cost measurement.


The matched HF extension exits0 after1024 steps (2079.41 seconds, including the
CPU pause; not a benchmark). Saved first-EOS lengths: p0 haiku284, p3 laws411,
p5 Fibonacci237. The p2 story remains coherent but capped at1024. The queued TT
comparison now runs the same four prompts/cap and writes its distinct extended
artifact. The original six-prompt256 outputs remain intact.

The packaging pre-commit pass succeeds after normalizing71 evidence files' trailing
whitespace/final newlines. Exact original bytes and before/after hashes are in
`format_originals_manifest.json` and its explicit external paths. No model/native
source changed. Active HF/TT logs and the previously checked frozen shell launcher
were excluded from that pass; a final pass follows clean process shutdown.
`final_source.sha256` includes runtime, native repair, both common sampler sources,
and all relevant test/reference/profile helpers.


`qualitative_extended` completes and closes all devices with process/launcher
exit0. Root read all four saved TT outputs: valid5-7-5 haiku418 tokens; complete
coherent brass-key/Memory/hope story531; complete three-law thermodynamics433;
correct iterative Fibonacci function337. HF counterparts are284/1024-capped/
411/237. The old256-token p0 TT prefix differs from its extended run, so the
extended result must not be described as exact continuation of the old branch.
The independent qualitative review records precise prefix differences and the
inherited capacity-dependent SDPA selection; no decoder policy was changed.
The completed matching-budget outputs support an actual quality assessment.


The independent qualitative report now passes the recorded suite; the stage
reviewer independently read and agreed with all selected final answers. All8
extended mechanical-degeneration checks pass. TT p2/p3/p5 preserve their old256
prefixes; p0 changes at token32. Its larger cache crosses the inherited SDPA
selection branch, a plausible numerical cause that was not isolated experimentally.
The report makes no old-branch termination or capacity-invariant token claim.

Final packaging checks pass (`precommit_final.log`, `precommit_quality_report.log`).
The final pass normalizes6 new/re-generated text artifacts, with their original
bytes also preserved by hash. Source and raw per-device CSV evidence stay unchanged.
After restaging, `git -c core.whitespace=cr-at-eol diff --cached --check` passes;
this accepts the profiler tool's preserved CSV CRLF line endings. The source-only
ordinary diff check also passes. The final source manifest matches every file.


## Final acceptance

Independent xhigh `stage-review` returns **clean-pass** in `STAGE_REVIEW.md`;
no required findings remain. The reviewer independently checked source/native
repair, exact artifact identities, all48 profiler tables, current readiness and
capacity results, fullB32 state contracts, and the actual selected qualitative
answers. Final report formatting is checked before the checkpoint. This completes
Stage6 full-model only; no vLLM integration or push was performed.

Final headline: warmed B1 TP4 S128/G128 **97.2439 ms TTFT,39.0550 token-out t/s/user**;
AIME S203/G100 teacher forcing **99.5571 ms,38.7602 t/s/user**. Prefill top1/top5/
top100 **99/100/100%**, decode **98/100/100%** over100 reference tokens. Full context
**262144**, mixed/fixed/inactive B32 contract, canonical separate model/sampler
traces, device token feedback, explicit host compatibility, watcher/native repair
and the recorded six-prompt qualitative suite all pass. Sampling is0.4897 ms,
about1.91% of measured full-stack decode latency.

## Local checkpoints

- Stage implementation, native repair, tests, references and evidence: `177afdc1e94ae4a41149a403b9e018ac98b7a3c6`.
- Parent Stage5 checkpoint: `7550299ba237fa0d578938de1424398046500968`.
- This follow-up commit records the stage SHA and acceptance. The complete local
  commit ledger, including this follow-up's own SHA, is stored at
  `/home/mvasiljevic/qwen38-full-rerun/artifacts/full_model/local_commits.json`
  after commit creation; `git log -2 --oneline` also lists both checkpoints.

All commit-time pre-commit hooks passed. No push was performed.
