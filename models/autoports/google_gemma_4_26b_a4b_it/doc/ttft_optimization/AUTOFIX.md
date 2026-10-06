# TTFT AutoFix experiment ledger

Starting source diagnosis: `AUTODEBUG.md`. Matched serving baseline and raw
artifact locations are in `work_log.md`.

## Bounded prefill trace alone

Hypothesis: generator-owned prefill and first-sample traces remove dominant
eager launch work. Reduced allocation-tracked replay correctness passed18 cases.
Full S128/O16/C1 candidate:486.463ms median versus warmed baseline424.985ms.
S128/O128:494.495ms versus419.105ms. No improvement accepted.

The generator/adapter source still releases all traces when decode sampling
parameter repr differs from prefill's. Actual vLLM prefill parameters have compact
request rows; decode parameters are padded to32. The original reduced probe
passed the same scalar parameters in both phases and missed this integration
cost. The worker repeatedly logs `SPLIT_TRACE_READY` for warmed requests.

## Padded parameter boundary

Hypothesis: preserving an already-compatible sampling/model graph across compact
prefill→padded decode removes recurring capture costs. Added `--wire-sampling`
to the reduced probe to reproduce production's neutral inactive parameter rows.

Before repair, `wire_probe.json` passes token correctness but warm requests have
one new prefill capture each, with first decode35–42ms. After the targeted repair,
`wire_fixed.json` passes identical token checks; warm requests have zero captures
and first decode around3ms (S128:3.148ms). Both runs disable allocation tracking
for timing, use real layers0/5, same inputs, and the same probe.

Repair: `can_reuse_serving_decode` checks the model's captured seed-advance mode
and canonical sampler penalty/logprob graph compatibility. The adapter may then
refresh persistent sampling values without destroying the graphs. Binding shape,
cache and active-slot changes still go through existing decode invalidation.
This does not equate arbitrary parameter repr values or bypass parameter reset.

Full serving verification is pending in `candidate_reuse/`; the reduced timing
does not establish the full-model speedup.

## Environment observation

The first candidate's API-process `/proc/.../environ` includes
`GEMMA4_PREFILL_TRACE=1`, while the EngineCore proc listing does not. This is a
hypothesis about flag propagation, not a conclusion: process-title/environment
rewriting can affect proc visibility. Added a one-time generator policy log to
the next run to attest the actual selected value.

## Commands

Both wire probes use the baseline container and the following pattern, with
`wire_probe` before the graph-reuse repair and `wire_fixed` after it:

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'OMP_NUM_THREADS=8 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/wire_fixed_runtime python models/autoports/google_gemma_4_26b_a4b_it/tests/check_ttft_prefill.py --wire-sampling --skip-long --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/wire_fixed.json > /tmp/gemma4-ttft-wire-fixed.log 2>&1'
```

Status: verified reduced repair, full-server measurement pending; no completion
or accepted speedup claim.

## Actual serving boundary follow-up

`candidate_reuse/` did not establish a win: S128/O16 median446.250ms,
S128/O128 median444.678ms. The policy log confirms the flag is enabled,
refuting the environment-propagation hypothesis. Repeated capture logs persist.
The reduced actual vLLM server in `reduced_diagnostics/` shows the exact-length
key returns None, so no prefill state is prepared at all. Shape-level diagnostics
follow in `reduced_shapes_retry/`. The initial shape diagnostic failed only
because NumPy lengths were not JSON serializable; that raw failure is retained
under `reduced_shapes/diagnostic_serialization_failure.log` and is not performance
evidence. The diagnostic now casts lengths to Python integers.

Host adapter/sampling regression suite after fixture and validation-order repair:
89 passed (`host_tests_fixed.log`). No measured full-serving win accepted yet.

Confirmed shape mismatch: tokens[1,128], prompt_lens[128], hybrid tables[32,8192].
The adapter now slices the leading active prefill rows, retaining every column.
`reduced_compact/` actual vLLM S128/O16 median18.913ms versus45.959ms in
`reduced_shapes_retry/`; O128 median18.042ms versus45.881ms. Exactly one decode
capture across the repeated requests. These are two-layer diagnostic measurements,
not the full model's claimed performance. Full verification in `candidate_compact/`.

## Watcher infrastructure limit

The unmodified Watcher attempt (TT_METAL_WATCHER=10) fails before model execution:
ACTIVE_ETH program28464 bytes exceeds26624 config buffer. Process subsequently
exits139 during teardown. This exactly matches the earlier pipeline's
`doc/optimized_full_model/work_log.md` Watcher failure. Raw log:
`watcher_initial_failure.log`. This is not a model correctness result.

No owned model/server process remained, and host `fuser -v` showed no owners for
/dev/tenstorrent/0..3 before recovery. The serving image lacks the tt-smi entrypoint;
host checkout entrypoint has a non-host Python interpreter. The existing installed
package works through container Python with
`PYTHONPATH=/workspace/tt-metal/python_env/lib/python3.10/site-packages python -m tt_smi`.
Bounded list/reset/list and mesh smoke are required before retry. Initial list
shows all4 p300c chips. Retry will disable only Ethernet Watcher instrumentation
with TT_METAL_WATCHER_DISABLE_ETH=1, retaining worker assertions. Recovery results
remain pending until recorded below.

Recovery completed: bounded list/reset/list each exit0, all4 chips visible, mesh
smoke exit0 and MESH_SMOKE_OK. No foreign processes killed or locks cleared.
Artifacts `recovery_list-before.log`, `recovery_reset.log`,
`recovery_list-after.log`, `recovery_mesh.log`. Worker-only Watcher retry includes
allocation tracking and the batch128/page-refresh candidate; results pending.

Scoped retry passed (exit0):12 eager controls,24 repeated trace/fallback cases,
12 sampling/reset transitions. `watcher_retry.json` and `.log` attest worker
Watcher plus allocation tracking; Ethernet instrumentation remains unavailable
for the measured firmware buffer-size reason. Devices closed cleanly.
