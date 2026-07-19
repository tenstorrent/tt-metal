# Fused all-gather + projection AutoFix evidence

## Scope

TP4 Blackhole decode projections only.  The control paths remain unchanged and
the candidate stays behind `use_fused_decode_all_gather_matmul` /
`QWEN2_5_CODER_32B_MULTICHIP_FUSED_AG_MATMUL=1`.

## Isolated findings

- The exact QKV shape (`K=5120`, local `N=2048`, 8x1 matmul) hangs when the
  fused primitive is given the two links used by the standalone collectives.
  Triage could not inspect the stall because the installed `tt-triage` and UMD
  disagree on the `noc_read` buffer ABI.  The retained report is
  `qkv_8x1_hang_triage.txt`.
- The same QKV shape completes with `num_links=1`.  Every device observes the
  canonical gathered rank order `[1,2,3,4]`, with local output shape
  `[1,1,32,2048]`.
- The packed local gate/up shape (`N=14336`) overflows static L1 circular
  buffers on 8x1.  Padding to `N=14400` and using 10x1 removes the static-CB
  error, but the cached kernel stalls after launch.  Evidence is retained in
  `gate_up_padded_10x1_hang_triage.txt`.
- The adapted one-collective decomposition completes: fuse AG+gate at
  `K=5120,N=7168` on 8x1, then feed the returned gathered hidden tensor
  directly to a separate `K=5120,N=7168` up matmul.  No second AG or input
  restore is used.  Both outputs are finite with local shape
  `[1,1,32,7168]`, and the gather order is canonical on all four devices.
- Dynamic fused-AG outputs execute eagerly but are not stable under trace
  replay.  Merely retaining their tensor lifetimes reduces freed-buffer
  corruption but does not make replay bitwise.
- The trace-safe contract uses four persistent AG shards (`[32,1280]` per
  core), satisfying the TP4 ring-size validation, while the fused matmul still
  computes on 8x1.  A focused trace containing both QKV and gate fused ops is
  bitwise across replays, finite, and canonical on all devices.

Focused probe:

```bash
QWEN2_5_CODER_32B_MULTICHIP_FUSED_AG_PROBE=persistent_trace_4shard_ag_8x1_mm \
  timeout 300 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_fused_ag_projection_probe.py::test_persistent_four_shard_ag_eight_core_matmul_trace \
  -q -s
```

Result: PASS; QKV and gate output shapes are respectively `[1,1,32,2048]`
and `[1,1,32,7168]` per device; trace replay is bitwise.

## Full real-weight result

The repaired candidate completes eager decode, trace capture, replay, and the
real-weight correctness gate:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
QWEN2_5_CODER_32B_MULTICHIP_BASELINE_PATH=/tmp/qwen2_5_coder_32b_optimized_baseline.pt \
QWEN2_5_CODER_32B_MULTICHIP_RUN_PERF=multichip \
QWEN2_5_CODER_32B_MULTICHIP_KEEP_RESIDUAL_L1=1 \
QWEN2_5_CODER_32B_MULTICHIP_PACKED_GATE_UP=1 \
QWEN2_5_CODER_32B_MULTICHIP_DISTRIBUTED_NORM=1 \
QWEN2_5_CODER_32B_MULTICHIP_FUSED_AG_MATMUL=1 \
QWEN2_5_CODER_32B_MULTICHIP_PREFILL_TRIALS=1 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_TRIALS=1 \
QWEN2_5_CODER_32B_MULTICHIP_DECODE_REPLAYS=2 \
QWEN2_5_CODER_32B_MULTICHIP_RESULT_NAME=autofix_fused_ag_full_probe.json \
timeout 600 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_multichip_decoder.py::test_real_multichip_warmed_prefill_and_traced_decode_perf \
  -q -s
```

| Metric | Result |
|---|---:|
| Prefill PCC | 0.9933917028 |
| Traced decode PCC | 0.9938940903 |
| Warmed prefill (one sample) | 4.351594 ms |
| Traced decode (two replays, one trial) | 0.960300 ms |
| Trace replay | bitwise |

Artifact:
`../results/autofix_fused_ag_full_probe.json`, SHA-256
`5160bd888b376f41a90f179119b20e053bb09c1e7895c968fc9cad3ef820d9f7`.

The measured numbers are unchanged. After the device run, the artifact's
generic packed-family mesh summary was corrected host-only to describe the
actual candidate: four-core persistent AG storage, 8x1 fused QKV/gate and
direct-up compute, one link per fused hidden gather, and two small
distributed-norm statistics gathers in addition to the two material hidden
gathers.

Because this repair run used the earlier attention HiFi2 policy, the coherent
family was rerun under the final attention/MLP LoFi policy, 8x4 SDPA, and the
full seven-trial/100-replay harness. `../results/sweep_fused_ag_final_full.json`
(SHA-256 `105c2a95bf47a62348236ba1bff5a8a9b217c8c43a470b21505eaf993f8f0502`)
records PCC 0.992527216/0.993663699, warmed prefill 3.657417 ms, and traced
decode 0.930669 ms. The current final compatible control is 0.758047 ms, so this
repaired fused family is 22.7% slower. It is therefore a functional,
trace-safe material rejection rather than the selected default.

The two persistent BF16 AG buffers cost exactly 655,360 bytes (0.625 MiB) per
device and are included in `SharedDecodeCollectiveBuffers`, so a sequential
decoder stack reuses that cost across layers.  The separate interleaved gate
and up weights replace the packed DRAM-sharded decode weight and have the same
aggregate element count and dtype.
