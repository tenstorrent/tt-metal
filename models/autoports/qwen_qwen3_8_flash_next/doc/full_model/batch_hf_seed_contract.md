# Batch HF harness: explicit seed ownership repair

CPU-only preparation, 2026-11-14. No production source, sampler semantics,
accuracy threshold, or existing result was changed. The root device owner runs
the reduced control before a fresh all48 run.

`batch32_hf_factored_v1.json` remains **failed**, SHA256
`eebede318006a16fc7d2d762c9e671f641ba23bc9b3afc61a36c7c583c43240b`.
Both runs contain2704 predictions with2645/2704 top1 and2704/2704 top5/top100
(97.818047%/100%/100%). Token/logit migration controls, dormant state and resident
feedback passed, but combined final-state digests differed for every request.
The original probe SHA256 was
`f673fefa9a2eadb55e1f0c399ba379f09e00b9b6d7a2aa3577c98b8945c1ea4a`.
No inference below changes that recorded verdict.

The actual common formatter explicitly treats `seed` as lane-scoped: scalar0
sets lane0 only; the other31 lanes remain `None`. The generator setter preserves
this rule even when prefill supplies32 slots and ordinary scalar distribution
fields broadcast. `DeviceSeedState.reset_words(None, ...)` draws fresh entropy.
Consequently separate baseline/repeat runs do not have matching full seed state,
and reloading scalar0 after reversing slots assigns seeded identity to a
different request. Greedy output equality does not make these seed states equal.
The former all-state equality comparison therefore compared different declared
sampling histories. This is a concrete caller contract error, not evidence for
changing production scalar seed semantics.

`greedy_params_for_order` now supplies explicit temperature/top-k/top-p lists
and an explicit seed0 for **every** request. Migration rebuilds these host
parameter lists in current request order, preserving each request's identity.
The probe records and asserts actual generator seed assignments after prefill
and migration. The seed reset after migration uses authoritative position+1,
matching the next draw following the already-completed prefill/decode draws.
The complete final-state gate is retained. State capture additionally saves
per-field SHA256 values for each layer/rank, logical K/V history, feedback,
seed state and host position, and reports exact differing fields when it fails.
The old combined digest is still checked; seed state is not dropped.

CPU verification executes the actual AST-extracted generator setter and common
formatter, with the actual decode validation body for migration. It proves:

- old scalar seed with32 explicit slots gives `[0] + [None]*31`;
- new explicit seed list gives32 zeros and normalized greedy k1/p0/temp1;
- distinct per-request seeds remain correctly ordered during reload;
- actual decode migration resets seeds202/101 at counters11/10 for authoritative
  positions10/9, and fills other physical lanes with `None`;
- field-level diagnostics detect both a seed mismatch and a missing field;
- existing trace accounting, resident feedback and mutation-negative controls
  remain intact.

Command: `python_env/bin/python -m pytest --noconftest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_batch_hf_cpu.py models/autoports/qwen_qwen3_8_flash_next/tests/test_batch_hf_evidence_cpu.py`.
Result:12 passed,7 subtests passed; log `batch_hf_seed_contract_cpu_v1.log`.
Pinned isort5.13.2/Black23.10.1 and Python compilation passed for both edited
Python files. No TTNN import or device open occurs in these CPU tests.

Device-owner commands, after the standard lane environment and accepted mesh
descriptor are restored (watcher/profilers off; tracking1, skip-program-cache0):

```bash
timeout --signal=INT --kill-after=60 3600 python -u -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_batch_hf --diagnostic-layers 0 1 3 --batch-size 32 --capacity 512 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch32_seed_state_reduced_v1.json
```

This reduced mode runs the same full32-slot prompts,99-step causal history,
step32 reversed migration, shrinking active rows, dormant/all-inactive controls,
resident sampled feedback and field-level state checks on real layers0/1/3.
It uses one actual QSA pool and labels `acceptance_eligible=false` and
`hf_accuracy_evaluated=false`; the reference supplies causal input history only.
It does **not** score reduced-model tokens against full-HF outputs. A reduced
pass is diagnostic evidence only.

After the reduced control passes:

```bash
timeout --signal=INT --kill-after=60 7200 python -u -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_full_model_batch_hf --batch-size 32 --capacity 512 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch32_hf_factored_v2.json
```

Default execution still constructs all48 layers and12 QSA pools, evaluates the
same HF top-k gates separately per phase/request, and requires complete state,
logit/token, trace and resident-feedback checks. Both device runs were **not run
by this CPU author**; no hardware pass is claimed. Existing cross-geometry B1/B2
failures remain explicitly outside this probe's adjudication.

Prepared probe SHA256: `b0cd9ae133b5cd182f9d89d76381d84ff6f19e050189ca3bd98c5f1d0b528895`.
