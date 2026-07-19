# BFP8 async-CCL AutoFix evidence

## Starting failure

The L1-residual + packed-gate/up decoder failed at its first decode
all-gather when `QWEN2_5_CODER_32B_MULTICHIP_CCL_DTYPE=bfp8`. The standard
persistent AG workspace had regressed to BF16/L1 while `_decode_norm`
requests a BFP8/DRAM async-AG output. This was also a BF16 default-path
regression because the persistent tensor did not match the requested DRAM
memory configuration.

## Repair and focused verification

Standard `_decode_ag_persistent_buffers` now use the selected CCL payload
dtype in DRAM, exactly matching the non-distributed decode AG contract. The
fused AG+matmul candidate keeps its distinct four-shard BF16/L1 buffers.
Reduce-scatter already had the correct pair: a payload-dtype DRAM
intermediate and payload-dtype local L1-sharded output. Both boundaries
explicitly typecast BF16 activations to the CCL dtype and restore BF16 after
the collective.

Focused exact-shape command:

```bash
QWEN2_5_CODER_32B_MULTICHIP_BFP8_CCL_PROBE=1 timeout 300 \
  python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_bfp8_async_ccl_probe.py::test_persistent_bfp8_all_gather_reduce_scatter_trace \
  -q -s
```

Result: PASS. The persistent BFP8/DRAM AG produced canonical rank order
`[1,2,3,4]` on all devices and shape `[1,1,32,5120]`. The persistent
BFP8-DRAM/BFP8-L1 reduce-scatter pair produced `[1,1,32,1280]` per device.
Both outputs were finite and bitwise across two trace replays.

## Full real-weight result

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip \
QWEN2_5_CODER_32B_MULTICHIP_KEEP_RESIDUAL_L1=1 \
QWEN2_5_CODER_32B_MULTICHIP_PACKED_GATE_UP=1 \
QWEN2_5_CODER_32B_MULTICHIP_CCL_DTYPE=bfp8 \
QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS=1 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=3 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=20 \
QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME=autofix_bfp8_ccl_full_probe.json \
timeout 600 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_warmed_prefill_and_traced_decode_perf \
  -q -s
```

| Metric | Repaired BFP8 CCL |
|---|---:|
| Prefill PCC | 0.9932432266 |
| Traced decode PCC | 0.9939028940 |
| Warmed prefill (one sample) | 4.184984 ms |
| Traced decode median (3 trials x 20 replays) | 0.805522 ms |
| Traced decode samples | 0.805461 / 0.805522 / 0.805708 ms |
| Trace replay | bitwise |

Artifact:
`../results/autofix_bfp8_ccl_full_probe.json`, SHA-256
`e73ae2da6e69a6ab22a796c73a99338ef215f257206828dee46e0384ba2c3981`.

This repair run used the earlier attention HiFi2 policy, so BFP8 CCL was also
crossed with the final attention/MLP LoFi policy, 8x4 SDPA, and the complete
seven-trial/100-replay harness. `../results/sweep_ccl_bfp8_final_full.json`
(SHA-256 `c723c0d80ebaa37c7fd1130f697a64a1c3c9779d65febe3fc6dac0eb09540b0f`)
records PCC 0.992493754/0.993622454, prefill 3.708916 ms, and traced decode
0.787956 ms. The current final compatible BF16-CCL control is 0.758047 ms, so
BFP8 is 3.92% slower. It is a functional material rejection; BF16 remains selected.
