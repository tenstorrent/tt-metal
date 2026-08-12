# Frozen-snapshot evidence

Runtime source SHA-256:
`c4d3eb3c7df806c67160b20a2d5d22118dc8acae4227b885abcc5adbef85429e`.
Test source SHA-256:
`22b1c18477abc4a6cc2dd6771f221ff7c633edf1a9787ebba5f492a6160e6199`.

All XML files were generated with pytest JUnit stream capture and contain the
test result plus the printed PCC/performance/capacity evidence.

| Artifact | Result |
| --- | --- |
| `default_suite.xml` | 2 passed, 3 explicit opt-in probes skipped |
| `real_pcc.xml` | prefill 0.9997971705; decode 0.9998261581; BFP8 K/V append 0.9930966025 / 0.9931919671 |
| `repeated_decode.xml` | non-aligned prefill lengths 1/17/33/18 and cache-consuming positions 18-22 pass |
| `perf_trace.xml` | eager/trace PCC 1.0; 5.386907 ms prefill; 1.288097 ms traced decode over 100 replays |
| `capacity_4096.xml` | batch 32, sequence 4,096 passes and copies output shape `[1,32,4096,5120]` to host |
| `watcher.xml` | exact final path passes under watcher |
| `watcher_runner.log` / `watcher_device.log` | records `TT_METAL_WATCHER=10`, logical-device attach/detach, passing test, and a zero-match error/assert/hang/NoC signature scan |
| `advisor_1d_pcc.xml` | exact advisor seed passes: prefill 0.9997971705, decode 0.9998337170 |
| `advisor_1d_perf.xml` | exact advisor seed: 5.436437 ms prefill, 1.788112 ms traced decode; rejected versus 1.288097 ms |
| `packed_pcc.xml` | packed gate/up comparison passes: prefill 0.9997971705; decode 0.9998302423; BFP8 K/V append 0.9930640128 / 0.9931633054 |
| `packed_perf.xml` | packed gate/up comparison: 5.377842 ms prefill, 1.857137 ms traced decode; rejected versus 1.288097 ms |
| `mlp_block16_10x32_runner.log` | precision-locked real-weight runner reaches prefill PCC 0.9997971705, then captures the expected 3,977,984 B static-CB failure |
| `mlp_block16_10x64_runner.log` | widest divisibility-legal adaptation reaches the same prefill PCC and captures the same expected failure |
| `runner_evidence.json` | machine-readable environment, config, source hashes, log hashes, exit codes, blocker, watcher result, and post-run health summary |

Dominant MLP block-16 candidates were also run from this source family with
`MISTRAL_SMALL_24B_OPT_POLICY=selected_mlp_10x32` and
`selected_mlp_10x64`. Both reached the real prefill result, then failed at the
first decode gate matmul with the same exact device constraint:

```text
in0_block_w=16
Statically allocated circular buffers on core range [0-0 - 9-9] grow to
3977984 B which is beyond max L1 size of 1572864 B
```

The 32-output-core attempt uses `per_core_N=32`; the widest divisibility-legal
64-output-core adaptation uses `per_core_N=16`. Both runner logs embed the
runtime/test hashes, precision policy, geometry, command, prefill result, exact
device exception, expected nonzero exit, and healthy post-failure inventory.
Widening the output grid does not reduce this program family's block-16 input
circular-buffer allocation.
