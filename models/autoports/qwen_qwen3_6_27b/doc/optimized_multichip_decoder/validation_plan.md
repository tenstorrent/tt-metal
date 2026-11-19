# Final default-path validation plan

Prepared CPU-only. No device validation was executed by this preparation task. `tests/run_optimized_multichip_validation.py` plans or serially executes the established stage-4 harnesses with stage-5 destinations; `tests/run_multichip_contract.py` adds the missing TP4 version of the HF list/resume contract. No existing harness or decoder was edited. Final performance, profiler/advice, roofline, fallback audit, and independent stage review remain parent-owned.

The runner never passes a policy override, imports a candidate module, selects a replicated residual, or accepts single-chip output as final TP4 evidence. Integrate the selected defaults first. Single-chip jobs in the plan exclusively generate frozen-optimized oracle fixtures; every corresponding acceptance job executes the actual TP4 default. All JSON, PT fixtures, commands, receipts, logs, and watcher captures are written under `doc/optimized_multichip_decoder`. Stage-4 accepted files are never opened for writing.

## Planned phases

| Phase | Default jobs | Coverage |
|---|---:|---|
| correctness |20|Both kinds, B1S31/B2S33/B3S513/B32S33/B1S1025; fresh optimized oracle and TP4; trace allocation tracking, positive control, changed inputs/positions/page mappings via existing decoder harness |
| contract |2|Actual TP4 against HF, each meaningful kind:12 non-aligned/aligned lengths,7 resumed prefixes with dropped-history negative controls, and two independent list-prefill rows with different starts/lengths |
| stack |4|Fresh oracle plus actual TP4 chain of real checkpoint layers0,1,2,3,4 at B2S65/B32S33, recorded text activations; direct native compact decode residual `[1,B,1280]`, unchanged BF16 shape at each boundary, eager/trace equality |
| resources |2|64 independent communication owners,48 GDN and16 full attention; B1/B32 decode and matching prefill batch at S65; watcher, tracking1, program-cache tracking, positive control, no acknowledgments, changed inputs/state and persistent-buffer ownership |
| context |4|Fresh B1 32768-token optimized oracle and actual TP4 for each kind; full decoder on every token, sampled output/state PCC, changed-input traced decode, and full-attention continuation/cache preservation; TP4 holds20GiB additional reservation/device |
| stress |12|Each kind at B1:128 prefill/128 decode steps and B32:33 prefill/512 decode steps; one oracle plus two independent TP4 subprocesses; each harness internally repeats its sequence twice with exact output/state equality |
| watcher |24|Both kinds at B2/B32×S256/S257, B3S513, B17S31; fresh oracle and TP4 watcher runs, row-dispatch observer, trace/positive-control gates; all four devices must detach cleanly |

Total default:68 serial jobs. `--stress-runs 1` yields one TP4 process per stress case, still with two exact internal repeated sequences; default2 also checks independent process reruns. Context may be extended using `--context-lengths 32768 65536`. The smoke does not redefine the advertised262144 maximum. A changed KV dtype/layout or measured capacity reduction requires the separate maximum-context evidence prescribed by the user; this plan does not waive it.

Watchers are enabled only for contract/resources/watcher phases. No device profiler is enabled anywhere in this validation suite. Timings incidentally emitted by existing validation harnesses are not final performance measurements. Every job uses allocation tracking1 and skip-program-cache0; the context/stack harnesses do not implement their own positive-control injection, while decoder and64-owner cases do.

## New TP4 HF contract adapter

`run_multichip_contract.py` copies the logical cases and every numerical assertion from `run_contract.py`:

- Lengths1,31,32,33,127,128,129,511,512,513,1025,17: output PCC>=.995, every contribution-row PCC>=.995, finite outputs.
- Prefix starts1,32,33,128,511,512,544: every positive continuation contribution-row PCC>=.995; dropped-history negative control must fall below.995.
- Independent list rows with starts32/33 and tail lengths33/17: both full-HF output PCC>=.995.

Adaptations are actual1x4 FABRIC_1D mesh, `MultichipDecoder` default construction, `upload_residual` for hidden-sharded public input, and host concatenation of the four rank-local hidden slices only at test readback boundaries. Reset copies replicate local zero state to each rank. The original `from_torch`/`to_torch` forward guards are retained and also applied to list-prefill setup/forward calls. No runtime gather is inserted into the model, no inter-layer chain is synthesized here, and this contract adapter does not claim decode coverage; native compact decode is exercised by the existing stack/decoder/resource runners.

The adapter saves status/pass/provenance, refuses artifact destinations outside stage5, and records failures before propagating them. CPU AST inspection confirmed that all five original assertion expressions are retained unchanged. These CPU checks do not establish that the TP4 HF gates pass; the parent must run the phase and repair failures normally.

## Commands and resumption

After integrating final defaults and finishing all other device drivers:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_optimized_multichip_validation \
  --prefix final_validation_v1
python -m models.autoports.qwen_qwen3_6_27b.tests.run_optimized_multichip_validation \
  --prefix final_validation_v1 --run --resume
```

The first command only writes a manifest and does not import TTNN/torch or touch hardware. The second executes. Add `--expected-default-sha256 <selected-source-hash>` to pin the reviewed/default file explicitly. For named phases, use a distinct prefix for each independently planned group, e.g.:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_optimized_multichip_validation \
  --prefix final_contract_v1 --phases contract --run --resume
python -m models.autoports.qwen_qwen3_6_27b.tests.run_optimized_multichip_validation \
  --prefix final_resources_v1 --phases resources --run --resume
python -m models.autoports.qwen_qwen3_6_27b.tests.run_optimized_multichip_validation \
  --prefix final_context_v1 --phases context --context-lengths 32768 --run --resume
```

Every child is passed through the stage-5 `run_job.sh`; bounds are1200s or1800s, with a second outer bound120s beyond the wrapper's job limit. Subprocesses never overlap. An advisory file lock excludes a second instance of this orchestrator; it does not coordinate unrelated search drivers, so the parent owns exclusive scheduling. The orchestrator never resets hardware or terminates another job.

The manifest fingerprints all model TT Python files, participating harnesses, CCL manager, config, wrapper, and native MMRS C++/header sources. It checks those hashes before and after every device subprocess. Resume requires a passing receipt, identical manifest signature, unchanged artifact/fixture size and modification metadata, and revalidated report gates. A stale default, changed source, altered evidence, preexisting failed output, or missing receipt stops the run. Preserve those files and select a fresh prefix for repaired/repeated runs. Changing the phase list under an existing prefix is intentionally rejected. The preliminary `validation_preselection_plan_manifest.json` is illustrative and must not be reused after source selection; create a final prefix after defaults and harnesses are fixed.

Per-job `.validation.json` receipts and `<prefix>_commands.jsonl` record effective TT environment, duration, exit code, command, artifact size/mtime metadata, and failure details. `<prefix>_summary.json` is written only when every planned job passes. Watcher acceptance additionally requires a captured log with no `Watcher detected`, `ASSERT`, or `TT_FATAL`, and detach records for all four ranks.

## Harness limits to preserve in the final report

The five-real-layer stack proves direct boundary dispatch and final-chain PCC, but its existing JSON only reports final-chain outputs, not per-layer state PCC. The separate per-kind gates and64-owner full-state hashes cover different parts of that risk; do not rename the stack result as a full64-layer numerical model test.

The64-owner harness shares representative weights and inputs while independently allocating CCL/conv resources. It proves ownership/isolation and trace safety, not full-model weight residency or autoregressive composition. If a candidate's default integration retains setup-bound shallow decode views, the existing `copy.copy` resource cloning can preserve an old bound owner; integrate ordinary class methods or adapt owner rebinding before this test. Parent indicated source-level integration is planned, so this orchestrator makes no such mutation.

The context runner's reservation is additional to the real layer/cache/IO, and its memory samples are not a full-stack allocator proof. Reconcile them with the final cumulative context inventory. Maximum context remains262144 unless hard physical evidence requires otherwise.

CPU preparation verification: both new validation modules compile; Black checks pass; the planned manifest has68 jobs and contains no policy override arguments; all writable outputs/references resolve inside stage5; all inherited HF contract assertions remain identical. No device results exist from this preparation task.

Parent adjustment: evidence artifacts and PT fixtures are not individually hashed, following the installed tt-device-usage evidence rule. Source/manifest identities remain hashed; resume checks artifact size/mtime plus all report gates.
