# AutoFix: DRAM reader mesh assertion

## Starting evidence

- Diagnosis: `AUTODEBUG_dram_mesh.md` in this directory, rechecked against
  current native source before applying the fix.
- Original command:
  `bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh replicated_fixed_l0 --layer 0`.
- `replicated_fixed_l0.log`: prefill completed; the first eager decode failed
  in the DRAM matmul factory because `get_worker_noc_hop_distance()` requires
  a unit mesh. Inherited projection defaults selected three readers per bank.

## Hypothesis and controlled experiments

Hypothesis: selecting one reader for every projection avoids the unsupported
secondary-reader placement helper while preserving native device execution.
`matmul_utilities.cpp:415-419` returns before the hop helper for one reader;
lines 433-447 enter it for additional readers. The mesh restriction remains
explicit in `tt_metal/impl/device/experimental/device.cpp:19`.

The parent ran the following commands on the target TP4 mesh. This repair
agent inspected both JSON artifacts and their raw logs without using devices.

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh reader1_l0 --layer 0 --policy '{"attention_readers":1,"output_readers":1,"gate_readers":1,"up_readers":1,"down_readers":1}'
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh reader1_l3 --layer 3 --policy '{"attention_readers":1,"output_readers":1,"gate_readers":1,"up_readers":1,"down_readers":1}'
```

Both logs contain completed JSON reports and clean device teardown, without
the original assertion. Both effective policies retain DRAM projection mode,
BFP4/LoFi projection weights/compute, and all five one-reader settings.

| Evidence | Layer 0, linear attention | Layer 3, full attention |
| --- | --- | --- |
| Artifacts | `reader1_l0.json`, `reader1_l0.log` | `reader1_l3.json`, `reader1_l3.log` |
| Prefill PCC | 0.9999952912 | 0.9999977946 |
| Decode PCC | 0.9999997020 | 0.9999992847 |
| Changed decode PCC | 0.9999997616 | 0.9999996424 |
| State/cache PCC | recurrent 0.9999570251; conv 1.0 | key 1.0; value 1.0 |
| Trace/eager bitwise equal | true | true |
| Refreshed-input replay bitwise equal | true | true |
| Measured traced decode | 0.702284 ms | 0.583303 ms |

Verdict: **verified** for the reported assertion and these B1/S128 cases.
Layer 0's earlier harness emitted a warning about allocations with an active
trace. Layer 3's log has no such warning. The parent owns the harness repair
and final broader trace/cache validation; that warning is separate from the
reader-placement assertion and is not dismissed as harmless here.

## Minimal fix and verification

`tt/multichip_decoder.py` now sets `attention_readers`, `output_readers`,
`gate_readers`, `up_readers`, and `down_readers` to 1 in its construction
defaults, with a short comment naming the native mesh restriction. The change
is six added lines. Explicit experiment policies can still override defaults.
The default is resolved before weight shard construction, so weight-bank
padding and matmul reader counts continue to use the same setting.

The frozen `optimized_decoder.py` and native source were not edited. The
runtime experiment reports predate this default-only change; they verify the
identical effective policy supplied as an explicit override. Their recorded
model source SHA is `0b78a9df7d95bd7363095a0c869d3988959e45bf8c4aba35ae3b346eff71f386`.
No new hardware run of the changed default is claimed by this repair agent.

Host validation:

```bash
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py
```

Result: passed. The before/after diff was inspected and contains only the five
defaults and explanatory comment. This Python-only change requires no build.

## Final status

**Current assertion fixed:** the supported one-reader path is the multichip
default, backed by both attention-type controls. Performance alternatives and
broader stage qualification remain ongoing; these two timings do not establish
the fastest available configuration.

Retaining this exact multi-reader DRAM factory on parent-mesh tensors requires
native changes outside the authorized scope. The compiled factory supplies
the parent mesh to a coordinate-free helper, and its Python program config
has no coordinate/reader-assignment override. Coordinate-restricted tensor
views retain the same parent MeshBuffer; `to_device` cannot rebind an existing
device tensor to a unit submesh. Interleaved and minimal native matmul remain
model-local performance candidates described in the diagnosis.
