# Track 4 — sparse-matmul geometry + explicit L1 I/O: conclusion

**Result: no accessible decode headroom.** Evidence-based, mostly without device time.

## Geometry knobs (now env-exposed for sweeps, defaults unchanged)
`tt/moe_tt.py::create_moe_runtime`:
- `GLM4_MOE_MOE_SPARSE_IN0_BLOCK_W` (default 8) — pre-existing. Prior `decode_matmul_sweep`
  found it a **no-op** for decode.
- `GLM4_MOE_MOE_SPARSE_BLOCK_SIZE` (default 32) — NEW. But **tile-locked**: the sparse
  output tile is `Tile([block, 32])`, so `block` must be 32; other values break tiling.
- `GLM4_MOE_MOE_SPARSE_PER_CORE_M` (default 1) — NEW. Decode has a single token-block, so
  `per_core_M=1` is already minimal; larger only helps prefill (already handled by chunking).

## WIDTH_SHARDED L1 sparse I/O — dead end
Probe: `probe_sparse_width_shard.py` (single device, GLM4 decode gate/up shape).
- Interleaved L1: OK. sparse_matmul output is **6D** `[1,1,1,3,32,1536]`.
- Width-sharded L1: **TT_FATAL** `physical_height == physical_shard_height` — the 6D sparse
  output cannot be width-sharded with a standard config. No model in the repo width-shards
  `sparse_matmul` I/O (gpt_oss + GLM4 use interleaved L1; DeepSeek width-shards only dense
  linears + a2a I/O). GLM4 already gets the L1 benefit via `GLM4_MOE_EP_L1=1` (interleaved).

## Takeaway
Decode geometry is effectively tuned-out. The remaining real decode lever is the
**sharded-residual restructuring** (Track 1 follow-on): carry `hidden/8` TP-sharded
activations through attn O-proj / norms / residual so the validated ring `reduce_scatter`
needs no `all_gather`. That is where the CCL/MoE-output cost can actually drop.
