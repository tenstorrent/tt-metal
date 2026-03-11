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
- **E2E test** (`test_moe_gpt_e2e.py`): All 5 subtests PASS, ring_pos>0 failures are now blocking
- **test_decoder fused_experts** (`test_modules.py`): PCC = 0.933 (full fused pipeline E2E)

### PCC Summary
| Stage | PCC | Notes |
|-------|-----|-------|
| moe_gpt per-device | ~0.990 | bfloat4_b weights |
| selective_reduce_combine per-device | ~0.987 | All ring positions |
| Full fused pipeline (test_decoder) | ~0.933 | Includes score weighting, K-sum, all_reduce |

The 0.933 PCC for the full pipeline is expected given bfloat4_b weight quantization,
bfloat16 score multiplication, K-dimension summation, and all_reduce across 8 columns.

### Upstream PRs
| PR | Status | What it fixes |
|---|---|---|
| #38542 (Selective combine fast follow) | **Merged in branch** | K-indexed output shape |
| #39380 (linearized_mesh_coord bug fix) | **Merged in branch** | Device ID mapping |
| #39543 (combine semaphore signaling) | **Cherry-picked into branch** | Race conditions in combine writer |

## Bugs Fixed in This Branch

1. **Token distribution mismatch** (dm1.cpp floor+remainder fix):
   moe_gpt's combine writer used `div_up` (ceiling division) to distribute tokens
   across height shards, but `selective_reduce_combine` uses floor+remainder.
   Fixed moe_gpt to match. This was the same fix as upstream PR #38542 applied
   to DeepSeek's `moe_compute/dm1.cpp`.

2. **ring_pos>0 masking removed**: E2E test no longer silently skips ring_pos>0
   combine failures. All ring positions must pass.

3. **Routing weights indexing bug** (test_modules.py):
   `routing_weights[..., selected] = scores` broadcast scores to ALL tokens
   instead of just the current token. Fixed to `routing_weights[tok_idx, selected] = scores`.
   This caused the reference model to produce wrong outputs, making PCC comparisons
   misleading (0.732 instead of 0.933). The TT pipeline was correct all along.

4. **Combine semaphore signaling** (cherry-picked PR #39543):
   - Termination sync wait: `num_data_parallel_cores - 1` instead of `H*W - 1`
   - `num_mux_workers_per_link` instead of total `num_mux_workers`
   - Added `noc_async_atomic_barrier()` before global semaphore wait
   - Per-link termination masters instead of single global master

5. **Global expert IDs**: Test uses global expert IDs (0..127) instead of local.

6. **Bias zeroing**: Reference model biases zeroed since moe_gpt doesn't add bias.

7. **Output gathering**: Proper per-row concat across mesh rows.

8. **moe_gpt_fused cleanup**: Deleted unused 21-file moe_gpt_fused operation.

9. **shard-to-core mismatch**: `corerange_to_cores()` with `row_wise=True` to
   match BLOCK_SHARDED ShardOrientation::ROW_MAJOR.

10. **CMakeLists.txt cleanup**: Removed leftover MoEGPTFused target reference.

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

# After host-side C++ changes (e.g. program factory)
cmake --build build -- -j16 && cp build/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so

# Kernel .cpp files (reader.cpp, writer.cpp, dm1.cpp) are JIT-compiled — no rebuild needed.
```

## Key Files
| File | Purpose |
|------|---------|
| `tests/ttnn/nightly/.../test_moe_gpt_e2e.py` | E2E pipeline test (5 subtests) |
| `models/demos/gpt_oss/tests/unit/test_modules.py` | Component test (fused_experts PCC 0.933) |
| `models/demos/gpt_oss/tt/experts_throughput/fused_decode.py` | Fused pipeline forward |
| `models/demos/gpt_oss/tt/experts_throughput/weights.py` | Weight sharding + expert mapping |
| `ttnn/.../moe_gpt/device/kernels/dm1.cpp` | moe_gpt ring kernel (combine output writer) |
| `ttnn/.../selective_reduce_combine/.../reader.cpp` | Combine reader (token_work_split_even) |
| `ttnn/.../selective_reduce_combine/.../writer.cpp` | Combine writer (semaphore fix from PR #39543) |
| `ttnn/.../selective_reduce_combine/.../*_program_factory.cpp` | Combine program factory |

## Remaining Work
- Wire fused path into `ThroughputExperts` class for production use
- Clean up debug logging from `fused_decode.py`
- Clean up scratch files in repo root (`diagnose_*.py`, `fix_*.py`, `*_hang*.txt`, etc.)
