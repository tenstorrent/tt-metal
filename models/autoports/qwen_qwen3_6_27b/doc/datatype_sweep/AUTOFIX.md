# P1 trace-lifecycle repair — resolved with hardware evidence

The isolated [AutoDebug report](AUTODEBUG.md) identified a lifecycle mismatch: the installed P1 runner calls optional `teardown()`, while QwenGenerator owns its traces through `close()`. Four candidates retained **92471296 TRACE bytes** in the approximately 100 MB region; the fifth needed **8060928 bank-footprint bytes** with **7536640 bytes free**. This caused the original `down4_lofi` runtime error, not a numerical rejection. The [primary command](p1_primary.command.json), [log](p1_primary.log) and [verdict](p1_primary_precision_reasoning.json) remain preserved.

The minimal repair is [p1_runner.py](p1_runner.py), a stage-local entry point that calls the unchanged installed readiness runner. Its replacement release hook synchronizes the mesh, records opaque trace IDs and TRACE allocation, invokes the generator's existing `close()`, then requires zero remaining IDs and TRACE bytes. Synchronization/close errors and incomplete cleanup propagate, aborting the shared mesh; the runner's outer `finally` closes it. The adapter records its own and the installed runner's source hashes. [run_p1.py](run_p1.py) uses this entry point and rejects candidate runtime errors even when the selected candidate passes, as well as exit 0 without a newly written verdict.

No model `tt/` code, precision policy, score/text cache, scoring defaults, numerical threshold or trace-region size changed for this repair. Keeping the adapter local to the stage preserves the source/config/text/reference identities of valid cached measurements. The selected precision artifact was written separately after successful final P1; it is not part of the lifecycle fix.

## Host verification receipts

The following commands were run before hardware validation; they used no devices or TTNN imports:

```bash
python_env/bin/python -m black --target-version py312 --line-length 120 \
  models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/test_p1_runner.py \
  models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/check_p1_lifecycle.py \
  models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/run_p1.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/test_p1_runner.py -v
```

Black exited 0, reformatted the test file and left the other two unchanged. All **11 tests passed in 0.025 seconds**. After adding observed KV-shape validation to the lifecycle harness and matching fake fixture, these exact checks passed:

```bash
python_env/bin/python -m black --check --target-version py312 --line-length 120 \
  models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/test_p1_runner.py \
  models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/check_p1_lifecycle.py \
  models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/run_p1.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/test_p1_runner.py
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/check_p1_lifecycle.py --help
```

Black exited 0 with all three files unchanged; all **11 tests passed in 0.016 seconds**; `--help` exited 0 without importing TTNN. Tests cover synchronization-before-close, opaque IDs, orphan allocation despite cleared IDs, remaining IDs despite zero allocation, synchronization/close failures, driver verdict/error handling and five fake builds with collection before each rebuild. These host results alone did not establish hardware cleanup.

## Device ownership stress passed

The bounded [lifecycle command receipt](p1_lifecycle.command.json) exits 0. [p1_lifecycle.json](p1_lifecycle.json) records five sequential builds on one shared mesh, representative layers **0/3**, actual **262144-token KV allocation** for those layers, P128 preparation and all four owned traces. Every cycle measures:

| Before build | After preparation | After close | After GC |
|---:|---:|---:|---:|
| 0 bytes | 1130496 bytes | 0 bytes | 0 bytes |

This demonstrates non-accumulating ownership over five builds. It is diagnostic lifecycle evidence, not full-model numerical, performance or capacity acceptance.

## Actual full-model repaired path passed

The subsequent [final-contender command](p1_final_contenders.command.json) exits 0. Its [archived verdict](p1_final_contenders_precision_reasoning.json) contains **no candidate errors** and all four checks pass. The [log](p1_final_contenders.log) records both complete **64-layer** candidates, `gdn_attn8_head8_mlp4_lofi` and `gdn_attn8_head8_gateup4_lofi`, releasing **22937600 TRACE bytes before close → 0 afterward** on the shared mesh, with no remaining trace IDs. Thus the repair is verified on the actual full-model P1 path as well as the five-build diagnostic. No reset or process termination was needed; no hardware recovery remains pending for this defect.

The selected broad-MLP policy passes P1 at **dNLL 0.00248**, **SE 0.00056**, **KL×1000 3.85138** over **23517 positions**. The strong BF16 reference passes its HF control over **6144 measured positions**, with **KL×1000 0.29102**. The command requests `--hf-positions 2048`; the runner evaluates a complete selected text, producing that larger actual count. The original slower down4 candidate's missing P1 score is carried under the user search cap, never classified as a numerical rejection. It is slower than the now-selected passing winner and does not conceal a faster eligible measured candidate.

## Metadata gate and remaining stage work

The [readiness stage script](stage_gate.command.json) now exits 0 and its [log](stage_gate.log) reports a pass. The [initial exit-2 receipt](stage_gate_initial.command.json) and [log](stage_gate_initial.log) remain preserved: inherited [limitations.json](../eval_readiness/limitations.json) had an invalid `items` type. Correcting it to the required list schema retained the prior observation and fixed this metadata gate. No numerical threshold or model behavior was changed.

The lifecycle AutoFix is **resolved**. The wider datatype stage is **selected, pending final acceptance**: selected-default token-out/trace/context/non-aligned/resumed-prefill and qualitative work, independent review and commit bookkeeping remain in progress. This repair record does not claim those separate jobs are finished.

The AutoDebug launcher initially failed its no-model sandbox preflight because the host disallows unprivileged namespaces. It was rerun with `AUTODEBUG_SKIP_CHILD_SANDBOX=1` under the existing unrestricted, no-approval session authorization. The isolated child used no devices and changed only its report. Its report is preserved here; the pre-existing root report was restored.
