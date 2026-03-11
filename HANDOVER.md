# MoE Fused Pipeline Handover

## Branch & Machine
- **Branch**: `sraizada/moe-gpt-fused-integration-cleanup`
- **Machine**: `ubuntu@UF-EV-B4-GWH02` at `/data/handrews/tt-metal`
- **Galaxy**: 4x8 mesh (32 Wormhole devices), cluster_axis=0 for column rings

## Current Status

### What's Working
- **dispatch** (`all_to_all_dispatch_metadata`): PCC pass on all devices
- **dispatch_compute** (`moe_gpt`): PCC ~0.990 on all 32 devices
- **selective_reduce_combine at ring_pos=0**: PCC ~0.987

### What's NOT Working
- **selective_reduce_combine at ring_pos>0**: Produces wrong data
  - E2E tests (`test_moe_gpt_e2e.py`) **mask this as a non-blocking warning** — the tests pass but ring_pos>0 combine results are silently skipped (see lines 676-687 in `run_test_dispatch_compute_combine`)
  - Component test (`test_modules.py`) gets **PCC ~0.665** instead of ~0.99, because only 1/4 of tokens (ring_pos=0) have correct combine results
- **all_reduce**: Not yet wired into pipeline

## Latest Commits
- `58992963b6` — Remove unused moe_gpt_fused operation
- `8b7cda4869` — Fix global expert IDs, per-device weight sharding, bias zeroing, output gathering
- `ec8d57b8e5` — Fix e_t format, dynamic activations_stride, combine verification, fused decode pipeline

## Open Issue: ring_pos>0 Combine Failure

### Root Cause Analysis (in progress)
**Likely cause**: Token distribution mismatch between moe_gpt's combine kernel and selective_reduce_combine's reader.

- **moe_gpt combine** (`matmul_dm1.cpp`, now deleted but was read): Writes W2 output into BLOCK_SHARDED combine buffer. Each expert writes exactly `TOKENS_PER_CHUNK=32` tokens per shard. The combine output has shard shape `[128, 960]` with `COMBINE_H=4, COMBINE_W=3` (12 cores). Each shard contains ALL 4 experts: expert `e` occupies rows `[e*32, (e+1)*32)`.

- **selective_reduce_combine reader** (`reader.cpp`): Uses `token_work_split_even` which distributes tokens with floor division + spread remainder:
  ```cpp
  chunk = count / NumTokenParallelCores;
  rem = count % NumTokenParallelCores;
  // core c gets: (c < rem) ? chunk+1 : chunk
  ```

- **moe_gpt verify_device_output** reveals it uses `div_up` (ceiling division):
  ```python
  max_tph = (count + H - 1) // H  # div_up
  # shard k gets tokens [k*max_tph, min((k+1)*max_tph, count))
  ```

When token counts aren't exact multiples of COMBINE_H, these produce different token-to-core mappings:
- Example: count=5, H=4 -> moe_gpt: [2,2,1,0] vs selective_reduce_combine: [2,1,1,1]

This means selective_reduce_combine reads tokens from wrong offsets for ring_pos>0 devices. Only ring_pos=0 gets correct data (by luck with seed 42), giving ~1/4 correct + 3/4 garbage = PCC ~0.665.

### What Was Confirmed
- Writer routing logic (fabric send, get_route, manhattan_distance) is correct for axis=0
- Fabric connections (NORTH/SOUTH) correctly set up by `get_neighbors`
- BLOCK_SHARDED shard layout: all experts per shard, not one expert per shard
- Worker core ordering with `row_wise=True` matches shard assignment
- DeepSeek's fused test passes with cluster_axis=0 but only verifies final output after reduce_scatter, not per-ring-position combine PCC

### What Was NOT Yet Confirmed
- Need to verify moe_gpt's actual token-to-shard distribution in the combine output
  - `matmul_dm1.cpp` wrote `TOKENS_PER_CHUNK=32` rows per expert (constant), so the mismatch may only matter when per-expert token counts vary
  - The metadata from dispatch tells each ring core how many tokens each expert has
  - The combine writer wrote ALL 32 rows regardless of actual token count (padding with garbage?)
  - selective_reduce_combine's reader uses `token_work_split_even` on the ACTUAL token count, so it may read fewer rows than what was written
- NOTE: `moe_gpt_fused` folder was deleted in latest commit. The relevant combine kernel code is in the **moe_gpt** (non-fused) kernel at `ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/`

### Next Steps
1. **Read moe_gpt (non-fused) combine kernel** to understand how per-expert token counts flow into the combine output layout
2. **Compare** how moe_gpt pads/distributes tokens vs how selective_reduce_combine expects them
3. **Fix the mismatch** — either:
   - Change `token_work_split_even` to use `div_up` to match moe_gpt, OR
   - Change moe_gpt's combine output to match selective_reduce_combine's expected layout
4. **Remove the ring_pos>0 masking** in `test_moe_gpt_e2e.py` (lines 676-687) and verify all ring positions pass
5. **Wire all_reduce** into pipeline (after selective_reduce_combine)
6. **Clean up** remaining scratch files in repo root

## Key Files
| File | Purpose |
|------|---------|
| `tests/ttnn/nightly/.../test_moe_gpt_e2e.py` | E2E pipeline test (5 subtests, ring_pos>0 masked) |
| `models/demos/gpt_oss/tests/unit/test_modules.py` | Component test (PCC ~0.665 issue) |
| `models/demos/gpt_oss/tt/experts_throughput/weights.py` | Weight sharding + expert mapping |
| `models/demos/gpt_oss/tt/experts_throughput/fused_decode.py` | Fused pipeline forward |
| `ttnn/.../moe_gpt/device/kernels/` | moe_gpt ring kernels (combine output writer) |
| `ttnn/.../selective_reduce_combine/device/kernels/dataflow/reader.cpp` | Combine reader (token_work_split_even) |
| `ttnn/.../selective_reduce_combine/device/kernels/dataflow/writer.cpp` | Combine writer (fabric send) |
| `tests/ttnn/nightly/.../test_moe_gpt_single_device.py` | Single-device test with verify_device_output |

## Useful Commands
```bash
# SSH to machine
ssh ubuntu@UF-EV-B4-GWH02

# Activate env
cd /data/handrews/tt-metal && source python_env/bin/activate

# Run e2e tests
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py -xvs -k "dispatch_compute_combine"

# Run component test
pytest models/demos/gpt_oss/tests/unit/test_modules.py -xvs -k "fused_throughput"

# IPMI reset (if devices hang)
sudo ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF  # then wait 90s

# After host-side C++ changes
cp build/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
```

## Scratch Files to Clean Up
Many debug/diagnostic scripts in repo root: `diagnose_*.py`, `fix_*.py`, `*_hang*.txt`, etc.
These are NOT committed and can be deleted.
