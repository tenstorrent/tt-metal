# Disaggregated Prefill-Decode PoC

> **Status**: ✅ **SUCCESSFUL** - All core functionality validated

## Overview

This PoC demonstrates **disaggregated prefill-decode** for LLM inference on Tenstorrent hardware, where:
- **Prefill Node (Rank 0)**: Processes the input prompt, generates KV cache, sends to decode node
- **Decode Node (Rank 1)**: Receives KV cache, continues autoregressive token generation

### Hardware Setup
- **Platform**: Tenstorrent QuietBox with 2× N300 devices
- **Topology**: Each N300 = 1×2 mesh (2 Wormhole chips)
- **Model**: Llama 3.1 8B Instruct

```
┌─────────────────────────────────────────────────────────────────┐
│                     QuietBox (2× N300)                          │
│                                                                 │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │   N300 #0       │  ETH    │   N300 #1       │               │
│   │   (Prefill)     │◄───────►│   (Decode)      │               │
│   │   ┌───┬───┐     │         │   ┌───┬───┐     │               │
│   │   │WH0│WH1│     │         │   │WH2│WH3│     │               │
│   │   └───┴───┘     │         │   └───┴───┘     │               │
│   │   mesh_id=0     │         │   mesh_id=1     │               │
│   └─────────────────┘         └─────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Running the PoC

```bash
cd $TT_METAL_HOME
tt-run --rank-binding tests/ttnn/distributed/disaggregated_pd_rank_binding.yaml \
       python3 tests/ttnn/distributed/test_disaggregated_prefill_decode.py
```

Custom prompt via environment variable in rank binding:
```yaml
env_overrides:
  PROMPT: "Your custom prompt here"
```

## Test Checklist

### ✅ Phase 1: Basic Functionality
- [x] Socket communication between N300 devices
- [x] Fix race condition (create sockets before prefill computation)
- [x] KV cache transfer layer-by-layer (32 layers × 2 tensors each)
- [x] Metadata transfer (seq_len, next_token) with UINT32 precision
- [x] Decode continues from correct position

### ✅ Phase 2: Validation
- [x] Generated text is semantically coherent
- [x] Test different prompts

### ✅ Phase 3: Performance
- [x] Measure prefill latency
- [x] Measure KV cache transfer time
- [x] Measure decode throughput (tokens/sec)

### ⬜ Phase 4: Future Work (Stress Testing)
- [ ] Longer prompts (512+ tokens)
- [ ] Multiple decode iterations (100+ tokens)
- [ ] Batch size > 1
- [ ] Compare with single-node baseline

## Performance Results

### Test Run 1: "Quick brown fox"
| Metric | Value |
|--------|-------|
| Prefill compute | 356.3 ms |
| KV cache transfer (send) | 24.1 ms |
| KV cache receive (wait + recv) | 460.2 ms |
| Decode throughput | **12.2 tokens/sec** |
| Generated output | "jumps over the lazy dog" ✅ |

### Test Run 2: "What is the capital of France?"
| Metric | Value |
|--------|-------|
| Decode throughput | **14.0 tokens/sec** |
| Generated output | "Paris" ✅ |

## Issues Found & Fixed

### Issue #1: Socket Timeout Race Condition

**Symptom**: Intermittent `TT_THROW: Timed out trying to establish a socket connection`

**Root Cause**: The decode node created its receive socket immediately after model loading, while the prefill node only created its send socket AFTER completing the expensive prefill computation. By the time prefill finished, the decode node had already timed out waiting for the socket handshake.

**Fix**: Create sockets on BOTH ranks BEFORE starting prefill computation:

```python
# === CREATE SOCKETS BEFORE HEAVY COMPUTATION ===
logger.info(f"Rank {rank}: Creating socket before computation...")
if rank == 0:
    socket = ttnn.MeshSocket(device, socket_config)
else:
    socket = ttnn.MeshSocket(device, socket_config)

# Barrier to ensure both sockets are established
ttnn.distributed_context_barrier()
```

### Issue #2: Metadata Precision Loss with BFLOAT16

**Symptom**: Token ID 35308 sent, but 35328 received (wrong token decoded)

**Root Cause**: BFLOAT16 has limited precision for large integers. Token IDs > 32K lose precision when converted to floating point.

**Fix**: Use UINT32 dtype for metadata transfer:
```python
# Sender
metadata = torch.tensor([actual_seq_len, next_token.item()], dtype=torch.int32)
metadata_tt = ttnn.from_torch(metadata, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

# Receiver
metadata_recv = ttnn.allocate_tensor_on_device(
    ttnn.TensorSpec([1, 1, 32, 32], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT), device
)
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Timeline                                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Rank 0 (Prefill)        │  Rank 1 (Decode)                            │
│  ─────────────────       │  ────────────────                            │
│                          │                                              │
│  1. Load model           │  1. Load model                               │
│  2. Create send_socket   │  2. Create recv_socket                       │
│        └────────────────►├──┘  (socket handshake)                       │
│  3. ═══ BARRIER ═══════  │  ═══ BARRIER ═══════                         │
│  4. Tokenize prompt      │  4. Wait for KV cache                        │
│  5. Run prefill (356ms)  │     │                                        │
│  6. Send KV cache (24ms) │─────┘                                        │
│  7. Send metadata ──────►│  5. Receive KV cache                         │
│  8. Done                 │  6. Receive metadata                         │
│                          │  7. Decode loop (14 tok/s):                  │
│                          │     - Forward pass (1 token)                 │
│                          │     - Argmax → next token                    │
│                          │     - Repeat until EOS or max_tokens         │
│                          │  8. Done                                     │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `test_disaggregated_prefill_decode.py` | Main PoC script |
| `disaggregated_pd_rank_binding.yaml` | Rank binding configuration |
| `n300_dual_mesh_graph_descriptor.textproto` | Mesh topology descriptor |

## Key Learnings

1. **Socket Creation Timing**: Always create sockets BEFORE any heavy computation to avoid timeouts
2. **Data Type Selection**: Use UINT32 for integer metadata (token IDs) to preserve precision
3. **Fabric Configuration**: FABRIC_2D mode enables inter-mesh communication via Ethernet
4. **KV Cache Structure**: 32 layers × 2 tensors (K, V) per layer for Llama 8B

## Future Improvements

1. **Pipelined Transfer**: Overlap KV cache transfer with decode computation
2. **Streaming**: Send KV cache layers as they're computed instead of waiting for all
3. **Multi-user Batching**: Support batch_size > 1 for higher throughput
4. **Continuous Batching**: Dynamic request scheduling across prefill/decode nodes
