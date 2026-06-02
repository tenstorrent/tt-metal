# MLA matmul tuning — chunked prefill

Context bridge for optimizing `ttnn.matmul` / `ttnn.linear` via `program_config` +
`memory_config` for the MLA module's matmuls under **chunked prefill**. One-shot prefill
is already tuned (see `test_mla_matmuls.py`); this work re-tunes for the smaller per-chip
chunk and the one relocated matmul. Runs in parallel to the chunked-prefill implementation.

## Reference files
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_mla_matmuls.py` — template test.
  Tunes 6 matmuls (mm0–mm5), each parametrized for `SEQ_LEN_32K=8192` and `SEQ_LEN_25K=6400`
  (per-chip seq = global / 4 galaxies, further SP/8 + TP/4). Mesh 8×4 (SP axis0 = seq dim2,
  TP axis1), `FABRIC_1D`. Runs `ttnn.linear` with hand-tuned
  `MatmulMultiCoreReuseMultiCast[1D]ProgramConfig`, checks PCC ≥ 0.99 vs torch. Grid is
  deliberately **11×10 = 110 cores** (not full 12×10) to dodge di/dt + throttling. HiFi2,
  `fp32_dest_acc_en=False`, `packer_l1_acc=True`.
- `models/demos/deepseek_v3_d_p/utils/parse_ring_joint_perf.py` — theoretical-compute model
  (written for SDPA = QKᵀ + scores@V, but the matmul math is the reusable part). Parses tracy
  `ops_perf_results_*.csv`, column `DEVICE KERNEL DURATION [ns]`.
- `models/demos/deepseek_v3_d_p/tt/mla/mla.py` — the real module. Matmuls go through
  `_get_mm_kwargs(weight_name, seq_len_local)` and `_make_batched_mm_kwargs`; configs live in
  `mla_config.py` (`MLA_MATMUL_CONFIG`, `MLA_SDPA_CONFIG`). The relocated matmul is `wkv_b2` /
  `tt_v_embedding` (around line 672).

## Theoretical compute model (the bar we compare against)
Per matmul:
- `tiles = (M/32)·(K/32)·(N/32)` (× batch if batched)
- `cycles = tiles · FIDELITY_CYCLES`, where `FIDELITY_CYCLES = {HiFi4:64, HiFi3:48, HiFi2:32, LoFi:16}`
- `time_ns = cycles / total_cores / clock_ghz`; clock: blackhole = 1.35 GHz, wormhole_b0 = 1.0
- **No causal `/2` for a plain matmul** (the `/2` in `parse_ring_joint_perf` is SDPA-causal-specific)
- Utilization % = `theoretical_ns / measured_kernel_ns × 100`

Also estimate **data-movement cost**: `bytes(in0 + in1 + out)` (per dtype: bf16 = 2B,
bf8_b ≈ 1B) ÷ DRAM BW. If DM time > compute time ⇒ DM-bound ⇒ move act/out to L1 interleaved.

## Hand-tuning order
1. Know shapes: M, K, N (+ batch if present) **after** SP/TP sharding. Input dtypes stay as in
   the test file (in0 bf16, in1 bf8_b typically; out per case).
2. Compute theoretical compute + predict DM cost; decide if DM-bound.
3. **First** pick `per_core_M` / `per_core_N` → these dictate active core count =
   `ceil(M_tiles/per_core_M) × ceil(N_tiles/per_core_N)` within the 11×10 grid.
   **Maximize cores: target 110, ≥90 fair, ≥80 okayish.**
4. **Then** find the best `out_subblock_h`, `out_subblock_w`, `in0_block_w`:
   - `out_subblock_h | per_core_M`
   - `out_subblock_w | per_core_N`
   - `in0_block_w | K_tiles`
   - **`out_subblock_h × out_subblock_w ≤ 8`** (dst register tiles)
5. Measure perf (tracy), report utilization.
6. Then try moving inputs/output to **L1 interleaved** `memory_config` if DM-bound (the one-shot
   test already mixes `DRAM_MEMORY_CONFIG` / `L1_MEMORY_CONFIG` per case).

ProgramConfig types: `MatmulMultiCoreReuseMultiCastProgramConfig` (2D mcast: `transpose_mcast`,
`fuse_batch`) vs `...MultiCast1DProgramConfig` (1D: `mcast_in0`).

## The one matmul that CHANGES for chunked prefill
`wkv_b2` V-embedding (`tt_v_embedding`, mla.py ≈ line 672) corresponds to the test's **mm4**
(batched, K=512, N=128, head_dim_v per head). Today it is computed **before** SDPA on the
full-kv-seq `tt_kv_nope`. It is being **moved to after SDPA** so it applies to the SDPA output
(fixed query-chunk size) instead of the V cache — because in chunked prefill V's seq grows with
iterations. **Weight shape unchanged (`[…, 512, 128]`); the M dimension shrinks from full-kv-seq
→ query-chunk seq.** All other matmuls: same shapes but smaller per-chip M (chunk instead of
8192 / 6400).

## Open questions (to resolve before writing the chunked param tuples)
1. Chunk size = the new per-chip M (query-chunk seq tokens) replacing 8192 / 6400. Single fixed
   value or a sweep?
2. Post-SDPA matmul exact in0 shape: after `nlp_concat_heads` (flattened) or per-head batched
   `[1, 32, chunk, 512] × [1, 32, 512, 128]` like mm4 today? Need batch / M / K / N exactly.
3. Exact tracy / profile pytest command for this test, and which CSV holds the per-op
   `DEVICE KERNEL DURATION [ns]` to isolate the matmul row.

## Status
Discussion / learning phase. No tuning code written yet. Next: answer the 3 questions, then
build chunked-prefill param tuples (new chunk M, relocated mm4) as a `test_mla_matmuls.py`
variant and iterate per-matmul against tracy.
