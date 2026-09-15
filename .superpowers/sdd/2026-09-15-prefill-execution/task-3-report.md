# Task 3 report: plain Llama RMSNorm

## Summary

PASS. Plain Llama-3.1-8B RMSNorm is implemented and validated on the assigned 4x8 Blackhole
Galaxy. The final gate ran eight RMSNorm calls over all 32 chips with synthetic and real checkpoint
gamma values, and completed with one passing test and no skips.

Base commit: `89f06e884a070cead69774adf94db65f844552d7`.

Commit: this report is part of the single Task 3 commit named
`Add Llama RMSNorm for Galaxy prefill`; its final SHA is recorded in the controller handoff
because a commit cannot contain its own hash.

## Changes

- Added `models/demos/llama_3p1_8b_d_p/tt/rms_norm.py`.
  - Validates a one-dimensional 4096-wide host gamma and finite positive epsilon.
  - Stores replicated BF16 ROW_MAJOR gamma as `[1, 1, 128, 32]` in DRAM.
  - Requires BF16 TILE interleaved-DRAM input with the full 4096 hidden width on every chip.
  - Calls `ttnn.rms_norm` with HiFi4, `math_approx_mode=False`,
    `fp32_dest_acc_en=True`, `packer_l1_acc=False`, and DRAM output.
  - Does not fuse residuals or modify/deallocate the input.
- Added `models/demos/llama_3p1_8b_d_p/tests/unit/test_rms_norm_vs_ref.py`.
  - Uses Transformers 5.12.1 `LlamaRMSNorm(hidden_size, eps=...)` as the independent float32
    oracle after BF16 rounding the same inputs and gamma.
  - Loads only three requested tensors from the indexed safetensors shards.
  - Explicitly shards the 1024 global rows over SP=4 and replicates each local 256-row shard over
    TP=8.
  - Checks output metadata, finite values, every device shard, exact unchanged inputs, exact zero,
    numerical thresholds, changed input and gamma addresses, program-cache reuse, and validation
    errors through `expect_error`.

No Task 2 module was edited. No MLP or generalized RMSNorm path was added.

## Preflight and live L1 evidence

The required `evidence/rmsnorm-kernel-preflight.md` was read before hardware. Its scoped path is
the interleaved BF16 TILE input plus replicated BF16 ROW_MAJOR gamma path, with named DFB
allocations totaling 1,146,880 bytes per participating core and no cross-core or cross-device
statistics collective.

Raw live value before the final launch:

```text
RMSNorm live L1 before launch: total_bytes_per_bank=1461248,
largest_contiguous_bytes_free_per_bank=1461248,
scoped_named_DFB_bytes_per_core=1146880
```

The largest live free block exceeded the audited requirement by 314,368 bytes per bank. The launch
produced no allocation diagnostic or L1 failure.

## TDD evidence

RED command:

```bash
source /data/divanovic/llama31-8b-disagg/tools/prefill_env.sh
cd /data/divanovic/llama31-8b-disagg/repos/tt-metal
"$PREFILL_PYTHON" -m pytest \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_rms_norm_vs_ref.py \
  --collect-only -q
```

RED result: exit 2 after 102.11 seconds, with the expected missing production module:

```text
ModuleNotFoundError: No module named 'models.demos.llama_3p1_8b_d_p.tt.rms_norm'
no tests collected, 1 error in 102.11s
```

Raw RED log:
`/data/divanovic/llama31-8b-disagg/evidence/task-3-rmsnorm/red.log`.

Intermediate device evidence was retained:

- Attempt 1 stopped before collection because the owned wrapper accidentally passed a literal
  `+` pytest argument. It did not open TTNN devices.
- Attempt 2 executed the kernel and established live L1 capacity, then found that mesh tensors
  expose the local `[1,1,256,4096]` shape rather than the global host
  `[1,1,1024,4096]` shape. The assertion was corrected to the intended per-chip contract.
- Attempt 3 passed normal and exact-zero cases, then exposed float32 metric reduction error on the
  repeated-row constant case: PCC 0.9998398 with NL2 0.0000016, while another case reported PCC
  above one. Only PCC/NL2 scalar accumulation changed to float64; model tensors, oracle, operator,
  and thresholds stayed unchanged.
- Attempt 4 is the final passing run.

Attempt logs:
`green-device.log`, `green-device-attempt2.log`, `green-device-attempt3.log`, and
`green-device-attempt4.log` under
`/data/divanovic/llama31-8b-disagg/evidence/task-3-rmsnorm/`.

## Final Galaxy command and result

The shared flock covered the entire device command:

```bash
flock -x /data/divanovic/llama31-8b-disagg/tmp/prefill-device.lock \
  srun --overlap --jobid 104525 --nodes=1 --ntasks=1 --cpu-bind=none \
  -w bh-glx-110-c07u20 \
  bash /data/divanovic/llama31-8b-disagg/tmp/task3_rmsnorm_run.sh
```

The owned script sourced `tools/prefill_env.sh` and ran:

```bash
timeout 1200 "$PREFILL_PYTHON" -m pytest \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_rms_norm_vs_ref.py \
  --tb=native -p no:cacheprovider -v -s
```

Timing and result:

```text
TASK3_RMSNORM_START_UTC=2026-09-15T19:05:19Z
1 passed in 69.25s (0:01:09)
TASK3_RMSNORM_END_UTC=2026-09-15T19:07:40Z
TASK3_RMSNORM_EXIT=0
```

Per-case worst values across all 32 chips:

| Case | Minimum PCC | Maximum normalized L2 |
| --- | ---: | ---: |
| synthetic normal | 0.9999985 | 0.0018902 |
| synthetic zero | exact zero | 0.0000000 |
| synthetic constant 1.75 | 1.0000000 | 0.0000016 |
| synthetic tiny, scale 1e-4 | 0.9999986 | 0.0022549 |
| model.layers.0.input_layernorm.weight | 0.9999984 | 0.0019026 |
| model.layers.0.post_attention_layernorm.weight | 0.9999985 | 0.0018901 |
| model.norm.weight | 0.9999985 | 0.0019005 |
| synthetic normal, return to first norm | 0.9999985 | 0.0018902 |

The thresholds stayed at PCC >= 0.9999 and normalized L2 <= 0.01.

Raw cache/address output:

```text
RMSNorm gamma addresses: synthetic=0x100080,
model.layers.0.input_layernorm.weight=0x100480,
model.layers.0.post_attention_layernorm.weight=0x100880,
model.norm.weight=0x100c80
RMSNorm cached-program reuse: entries=2 across 8 calls,
2 distinct all-chip input address tuples
JIT cache stats: 18/18 hits (100.0%)
```

The first input used address `0x141080` on each chip. A retained DRAM guard forced every later
input to `0x181080`, proving runtime address updates under the unchanged two-entry program cache.
The final call returned to the first norm instance after using three other gamma allocations.

The test also recorded successful `expect_error` checks for a two-dimensional weight, width 4095,
epsilon zero, and epsilon NaN. Every successful call checked BF16 TILE DRAM output and exact input
preservation on each chip.

Full raw GREEN log:
`/data/divanovic/llama31-8b-disagg/evidence/task-3-rmsnorm/green-device-attempt4.log`.

## Host checks and hooks

Initial owned-file hook command:

```bash
PRE_COMMIT_HOME=/data/divanovic/.cache/pre-commit \
TMPDIR=/data/divanovic/llama31-8b-disagg/tmp \
"$PREFILL_PYTHON" -m pre_commit run --files \
  models/demos/llama_3p1_8b_d_p/tt/rms_norm.py \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_rms_norm_vs_ref.py
```

Result: exit 0. Python formatting/import checks, `expect_error` enforcement, merge-conflict,
large-file, and relevant repository validation hooks passed; unrelated-language hooks skipped.

Before commit, the same hook set is rerun over both Python files and this report, followed by
`git diff --check`.

## Limitations

- Validated only on the required Blackhole Galaxy mesh shape (4,8), SP axis 0 and TP axis 1.
- The module intentionally supports only BF16 TILE interleaved-DRAM activations of hidden width
  4096 and one host gamma vector. Checkpoint caching and generalized layouts remain out of scope.
- This is plain local RMSNorm only. It contains no bias, residual fusion, Gemma weight folding,
  distributed-statistics collective, Q/K norm, activation, MLP, or checkpoint loader.
