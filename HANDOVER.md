# MoE Fused Pipeline Handover

## Branch & Machine
- **Branch**: `sraizada/moe-gpt-fused-integration-cleanup`
- **Machine**: `ubuntu@UF-EV-B4-GWH02` at `/data/handrews/tt-metal`
- **Galaxy**: 4x8 mesh (32 Wormhole devices), cluster_axis=0 for column rings

## Current Status

### What's Working
- **dispatch** (`all_to_all_dispatch_metadata`): PCC pass on all devices
- **dispatch_compute** (`moe_gpt`): PCC ~0.990 on all 32 devices
- **dispatch_compute_combine** (`selective_reduce_combine`): PCC ~0.987 on ALL 32 devices (all ring positions)
- **E2E test** (`test_moe_gpt_e2e.py`): All 5 subtests PASS, ring_pos>0 failures are now blocking (not silently skipped)

### What's Partially Working
- **test_decoder fused_experts** (`test_modules.py`): PCC = 0.732
  - The e2e kernel pipeline is correct (0.987 PCC), but the full module test including
    score weighting, K-dimension sum, and all_reduce gets 0.732
  - **Root cause**: PR #39543 (combine semaphore signaling fix) is NOT in our branch
  - See "Missing PR #39543" section below

### Upstream PRs
| PR | Status | What it fixes |
|---|---|---|
| #38542 (Selective combine fast follow) | **Merged in branch** | K-indexed output shape |
| #39380 (linearized_mesh_coord bug fix) | **Merged in branch** | Device ID mapping |
| #39543 (combine semaphore signaling) | **NOT in branch** | Race conditions in combine writer |

### Missing PR #39543: Combine Semaphore Signaling Fix

PR #39543 fixes three bugs in `selective_reduce_combine/writer.cpp`:

1. **Termination sync wait count** (line 258): Currently waits for
   `(num_token_parallel_cores * num_data_parallel_cores) - 1` signals,
   should be `num_data_parallel_cores - 1`. With COMBINE_H=4, COMBINE_W=3,
   this means waiting for 11 signals instead of 2.

2. **`num_mux_workers` vs `num_mux_workers_per_link`**: The `close_direction_connections`
   template arg uses total workers across all links instead of per-link count.
   Program factory sets `num_mux_workers = num_links * neighbors.size()` but
   should be just `neighbors.size()`.

3. **Missing `noc_async_atomic_barrier()`**: Uses `noc_async_write_barrier()` instead,
   which doesn't synchronize atomic operations properly before the global semaphore wait.

These bugs cause race conditions in the combine writer, producing corrupted output
on some tokens. The isolated e2e test (simpler sync pattern) gets 0.987, but the
full module test with score weighting and all_reduce amplifies the errors to 0.732.

**Next step**: Cherry-pick or rebase onto a commit that includes PR #39543, then
re-run `test_decoder --test-modules=fused_experts` to verify PCC improvement.

## Bugs Fixed in This Branch

1. **Token distribution mismatch** (dm1.cpp floor+remainder fix):
   moe_gpt's combine writer used `div_up` (ceiling division) to distribute tokens
   across height shards, but `selective_reduce_combine` uses floor+remainder.
   Fixed moe_gpt to match. This was the same fix as upstream PR #38542 applied
   to DeepSeek's `moe_compute/dm1.cpp`.

2. **ring_pos>0 masking removed**: E2E test no longer silently skips ring_pos>0
   combine failures. All ring positions must pass.

3. **Global expert IDs**: Test uses global expert IDs (0..127) instead of local.

4. **Bias zeroing**: Reference model biases zeroed since moe_gpt doesn't add bias.

5. **Output gathering**: Proper per-row concat across mesh rows.

6. **moe_gpt_fused cleanup**: Deleted unused 21-file moe_gpt_fused operation.

7. **shard-to-core mismatch**: `corerange_to_cores()` with `row_wise=True` to
   match BLOCK_SHARDED ShardOrientation::ROW_MAJOR.

## How to Run Tests

```bash
# SSH and activate
ssh ubuntu@UF-EV-B4-GWH02
cd /data/handrews/tt-metal && source python_env/bin/activate

# E2E pipeline tests (all 5 subtests)
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py -xvs

# E2E just combine test
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py -xvs -k "dispatch_compute_combine"

# Module test: fused experts (requires 4x8 mesh + decode_high_throughput for use_throughput_experts=True)
HF_MODEL=/data/tt_dnn-models/openai/gpt-oss-120b \
TT_CACHE_PATH=/data/huggingface/tt_cache/openai--gpt-oss-120b \
pytest models/demos/gpt_oss/tests/unit/test_modules.py::test_decoder -xvs \
  -k "layer_1-mesh_4x8-decode_high_throughput" \
  --test-modules=fused_experts

# NOTE: decode_low_latency (batch=1,seq=1) has use_throughput_experts=False
# because batch*seq=1 is not >1. Must use decode_high_throughput (batch=128,seq=1).

# IPMI reset (if devices hang)
sudo ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF  # then wait 90s

# After host-side C++ changes
cp build/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
```

## Key Files
| File | Purpose |
|------|---------|
| `tests/ttnn/nightly/.../test_moe_gpt_e2e.py` | E2E pipeline test (5 subtests) |
| `models/demos/gpt_oss/tests/unit/test_modules.py` | Component test (fused_experts PCC 0.732) |
| `models/demos/gpt_oss/tt/experts_throughput/fused_decode.py` | Fused pipeline forward |
| `models/demos/gpt_oss/tt/experts_throughput/weights.py` | Weight sharding + expert mapping |
| `ttnn/.../moe_gpt/device/kernels/dm1.cpp` | moe_gpt ring kernel (combine output writer) |
| `ttnn/.../selective_reduce_combine/.../reader.cpp` | Combine reader (token_work_split_even) |
| `ttnn/.../selective_reduce_combine/.../writer.cpp` | Combine writer (needs PR #39543 fix) |
| `ttnn/.../selective_reduce_combine/.../*_program_factory.cpp` | Combine program factory (needs PR #39543 fix) |

## Scratch Files to Clean Up
Debug/diagnostic scripts in repo root: `diagnose_*.py`, `fix_*.py`, `*_hang*.txt`, etc.
These are NOT committed and can be deleted.
