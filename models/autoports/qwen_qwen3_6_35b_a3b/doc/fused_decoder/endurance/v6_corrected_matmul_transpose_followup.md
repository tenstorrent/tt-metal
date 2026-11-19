# Transpose fusion: matched timing and complete chunk follow-up

The original22fixture device gate passed all strict input/transpose/output FP32 and BF16 code comparisons and closed cleanly. Original e48 component source and evidence remain unchanged. This follow-up investigates a repeatable trace-only regression before selecting any call site.

## Observed timing, not a final selection

| Site | Baseline trace samples (ms) | Candidate trace samples (ms) | Median change |
|---|---|---|---|
| actual_chunk0_gram | 6.496950, 6.497136, 6.496301, 6.497181, 6.498082 | 6.484163, 6.484208, 6.483368, 6.483515, 6.483583 | -0.209% |
| actual_chunk0_state_update | 12.965331, 12.965023, 12.965066, 12.964853, 12.964425 | 12.932986, 12.933531, 12.932942, 12.932171, 12.932145 | -0.247% |
| actual_chunk0_output_attention | 6.496969, 6.497144, 6.496058, 6.495955, 6.495907 | 7.246538, 7.246611, 7.247840, 7.248638, 7.248564 | +11.573% |


Gram and output attention have identical physical shapes `[1,32,64,128]` for both inputs, flags `(False,True)`, M64/N64/K128,128 output tiles on64 cores (2tiles/core), and identical exact compute/config/CB schedule. State update uses flags `(True,False)`, M128/N128/K64,512tiles on64cores (8tiles/core). Both adapters use DRAM inputs/output. The only descriptor differences from the baseline are reader compile arguments and reader source addressing; arithmetic, work partition and K count are unchanged.

The attention slowdown is present in all five samples; it is not an isolated outlier. Eager attention improves6.739247→6.673231ms while traced attention regresses6.496058→7.247840ms. Gram eager regresses6.742324→6.774474ms; state eager improves13.250168→13.136201ms. Therefore neither operator count nor the traced Gram improvement justifies retaining all three sites.

The original harness always completes baseline eager+trace measurements before candidate eager+trace measurements, and executes sites Gram→state→attention. Each trace is allocated and released separately after different temporary allocation history. Original evidence did not record physical addresses. Capture/allocation history and time-dependent device behavior remain hypotheses; the source has no shape/work difference between Gram and attention, but no causal claim is established. Log reports initial AICLK1343MHz against requested1350MHz; it contains no per-sample clock telemetry. That warning alone cannot explain or dismiss the candidate-only regression.

## New frozen probes

- `v6_corrected_matmul_transpose_timing_probe.py` SHA256 `26d81bedfc4df1605f782ae43963f8238b150060dd0ebec366b936c82c91abb9`. Uses frozen e48 helper and pinned22fixture PASS gate. Reverses site order (attention first), runs both candidate→baseline and baseline→candidate capture orders, keeps both traces live, alternates matched eager and trace batches with balanced order, records input/output device addresses, checks exact outputs after each batch, and reports individual samples and paired differences. Defaults6samples×20replays,120 repeated executions per adapter per capture order. Includes all native transposes in the baseline adapter. No TTNN import/device open in source-only setup.
- `v6_chunk64_transpose_output_probe.py` SHA256 `5dbe79f6b70c90c6811c63de6a08c83df617a63ee9411b21769fd9629613273e`. Uses frozen803 complete output/state graph and unchanged canonical HF fixture harness. Patches exactly the three identified `mm(transpose(...))` call expressions into the exact e48 reader adapter, with method-level inverse AST restoration. Other matmuls, triangular recurrence, exp/cumsum/FP32 boundaries, logical padding, and observers remain unchanged. `--sites` can select any nonempty subset; default all three. No selected site is silently omitted based on timing. Baseline and candidate gate complete BF16 output, FP32 output, final state, and observed intermediates; candidate also gates exact HF outputs/state. Optional full-adapter eager/traced timing runs candidate first.

CPU setup evidence passed for both zero and prefix64 nonzero initial state at logical lengths127,65,1:
`v6_chunk64_transpose_output_zero_cpu_setup.json`, `v6_chunk64_transpose_output_nonzero_cpu_setup.json`. These include canonical HF comparisons, saved payloads, executed source snapshots and patched-method snapshots. `v6_corrected_matmul_transpose_timing_cpu_setup.json` records timing source/proof and fixture order. Pinned autoflake2.3.1/isort5.13.2/Black23.10.1 checks pass both new files.

No device allocation capacity change: original40960bytes/core MM CB footprint and64core limit are retained; construction adds only a Python helper object. The fused path removes operand transpose temporaries. Wholechunk device correctness, repeated timing stability, and whole-decoder benefit remain unmeasured at preparation time.

## Parent-owned serial commands

Unset topology/watcher/profiler/allocation-tracking variables; physical chip0 is explicit. Output paths below must be fresh.

```bash
env -u TT_VISIBLE_DEVICES -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE PYTHONPATH=. \
  python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_corrected_matmul_transpose_timing_probe.py --run-device \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_matmul_transpose_matched_timing_device.json \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_matmul_transpose_matched_timing_device.log 2>&1

env -u TT_VISIBLE_DEVICES -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE PYTHONPATH=. \
  python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_chunk64_transpose_output_probe.py --run-device --initial-state zero --lengths 127 65 1 \
  --timing --timing-length 127 --samples 3 --iterations 1 --trace-replays 3 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_transpose_output_zero_device.json \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_transpose_output_zero_device.log 2>&1

env -u TT_VISIBLE_DEVICES -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE PYTHONPATH=. \
  python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_chunk64_transpose_output_probe.py --run-device --initial-state prefix64 --lengths 127 65 1 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_transpose_output_nonzero_device.json \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_transpose_output_nonzero_device.log 2>&1
```

For an evidence-supported subset, add e.g. `--sites gram state_update` and use distinct artifact names; first establish stable matched measurements and compare the complete chunk adapter. This mechanism enables site selection but makes no current profitability claim.
