# DeepSeek v3 B1 `test_decoder_mlp` — Workload Breakdown

> Code-grounded walkthrough of what the `test_decoder_mlp` pytest case actually runs on a Blackhole 4×2 mesh, with emphasis on the `flash_mla` compute stack and the iteration-parity behaviour that motivates the half-DEST investigation.
>
> Repo root: `/localdev/ncvetkovic/work/tt-metal`. All file paths are absolute. References point at the in-repo source at git HEAD `cb192340ac` on branch `aho/sdpa-ops`.

---

## 1. One-paragraph summary

`test_decoder_mlp` (`models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py`, definition at line 893) exercises one full DeepSeek-V3 decoder block on an 8-device Blackhole mesh **with `enable_routing=False`**. End-to-end that means: (1) the input is RMSNorm'd, mcast'd, projected to Q/KV, RoPE'd and head-split (the "pre-SDPA" stack); (2) the new KV row is written into the on-device KV-cache for `position_id=8190` (slot 0); (3) `flash_mla` decode reads the cache (8190 + 1 = 8191 valid positions) and produces an attention output; (4) the post-SDPA path matmuls/gathers/all-reduces it; (5) MoE runs but **only the shared-expert path** (gate/up/down on the first 2048 of 18432 channels — `DENSE_SHARED_N`); (6) a reduce-to-one collects the dense MLP output onto a single device, where the test reads it back, compares to a torch golden (PCC ≥ 0.975), and additionally validates that the KV-cache NOPE/ROPE writes match the golden (PCC ≥ 0.98 each). The whole device-side pipeline runs inside one unified kernel (`decoder_block_kernel.cpp`) whose `kernel_main()` spins a `while (true)` loop calling `mla_body()` then `moe_body()` — the test sets `num_internal_iterations ∈ {1, 2}` to drive that loop one or two times within a single program launch.

---

## 2. Test configuration

The test ID format follows from the `@pytest.mark.parametrize` decorator stack (lines 850-891). For the specific invocation given, the parameters decode as:

| Param                        | Value                          | Source line | Purpose                                                                |
|------------------------------|--------------------------------|-------------|------------------------------------------------------------------------|
| `arch`                       | `blackhole`                    | fixture     | Target arch (Blackhole P150).                                          |
| `slot_id`                    | `0`                            | 889         | Which KV-cache slot to write/read.                                     |
| `num_slots`                  | `1`                            | 889         | Total batch slots in the on-device cache.                              |
| `num_internal_iterations`    | `1` or `2`                     | 885         | Iterations of the `mla_body→moe_body` loop per program launch.         |
| `noc_mode`                   | `DM_DYNAMIC_NOC`               | 884         | Required by `flash_mla.hpp:24` (`static_assert`).                       |
| `device_params`              | `FABRIC_2D_TORUS_X`, fabric router config 15232, `worker_l1_size=1374544` | 873-882 | Fabric topology + L1 budget. |
| `position_id`                | `8190`                         | 870         | Global token position; `cur_pos+1 = 8191` valid KV rows.               |
| `max_seq_len`                | `32768`                        | 861         | Allocated KV-cache depth per device.                                   |
| `num_iters`                  | `1`                            | 860         | Outer Python loop in the test (launches `DecoderBlock.execute` once).  |
| `mesh_rows, mesh_cols`       | `(4, 2)` ⇒ 8 devices           | 859         | `num_devices = 8`, SP=4 (rows), TP=2 (cols).                            |
| `reduce_cluster_axis`        | `1`                            | 858         | All-reduce axis for post-SDPA / MoE output.                             |
| `use_fp32`                   | `False`                        | 857         | `fp32_dest_acc_en=False`; DEST is in half-sync (32 tiles per bank).    |
| `epsilon`                    | `1e-6`                         | 856         | RMSNorm ε (used by all 3 RMSNorms in the block).                       |
| `sender_row, sender_col`     | `(1, 0)`                       | 850-854     | MoE sender / mcast root mesh coord.                                    |

Derived constants pulled in from elsewhere:

| Derived value                            | Value      | Source                                                       |
|------------------------------------------|------------|--------------------------------------------------------------|
| `DENSE_LAYER_IDX`                        | `0`        | `tests/unit_tests/test_moe_mlp.py:116`                       |
| `DENSE_SHARED_N` (shared-expert width)   | `2048`     | `test_moe_mlp.py:117` (first 2048 of 18432)                  |
| `k_chunk_size` (Flash-MLA)               | `128`      | `micro_ops/flash_mla/op.py:247`                              |
| `NUM_BLOCKS` (S-blocks)                  | `8`        | `micro_ops/flash_mla/op.py:84`                               |
| `CORES_PER_BLOCK` (cores per S-block)    | `8`        | `micro_ops/flash_mla/op.py:85`                               |
| `device_chunk_size` (per-device KV span) | `1024`     | `op.py:259` (`NUM_BLOCKS * k_chunk_size`)                    |
| `NUM_TREE_REDUCTION_STEPS`               | `3`        | `op.py:103` (`log2(8)`)                                      |
| `num_qnope_heads × qnope_head_dim`       | `64 × 128` | `test_decoder_block.py:1075,1098-1101`                       |
| `num_qrope_heads × qrope_head_dim`       | `64 × 64`  | same                                                         |
| `KNOPE_DIM, KROPE_DIM`                   | `512, 64`  | `test_decoder_block.py:1077-1078`                            |
| `HEADS_PER_ROW`                          | `8`        | `test_decoder_block.py:1079`                                 |
| `Q_TILE_HEIGHT`                          | `8`        | `attention_block/op.py:1609`                                 |
| `K_TILE_HEIGHT`                          | `32`       | `attention_block/op.py:1610`                                 |
| `PNHt`                                   | `1`        | `attention_block/op.py:1618,1641` (asserted)                 |
| `Sk_chunk_t`                             | `4`        | `op.py:1620` (`k_chunk_size/K_TILE_HEIGHT = 128/32`)         |
| `DHt`                                    | `(KNOPE+KROPE)/32 = 18` | derived from `(512+64)/32`                          |
| `vDHt` (output head dim in tiles)        | `KNOPE/32 = 16` | `op.py:1617` (`QNOPE_DATA_SIZE/TILE_WIDTH`)             |
| `mla_dst_size`                           | `8`        | `op.py:1646-1650` (half-DEST mode, asserted ≥ 8)             |
| `MathFidelity`                           | `LoFi`     | `decoder_block/op.py:422`                                    |
| `dst_full_sync_en`                       | `False`    | `decoder_block/op.py:425` (= `fp32_dest_acc_en`)              |

Note on `position_id = 8190`: with `device_chunk_size = 1024` and `num_sp = mesh_rows = 4`, the SP-block period is `4 × 1024 = 4096`. `8190 // 4096 = 1` full block, remainder `4094`. The "owning" SP device is `(8190 // 1024) % 4 = 7 % 4 = 3`. So SP rows 0,1,2 each carry 1024 valid K rows, and SP row 3 owns the current write and carries `1024 + (4094-3072) = 1024+1022 = 2046` valid rows for this token. The mask path is engaged on the last chunk because `(8190+1) % 128 = 8191 % 128 = 127`, not zero (see `flash_mla.hpp:740`, `mask_last_chunk` is true).

---

## 3. Workload overview (end-to-end)

```
Host (Python, test_decoder_mlp)
 ├── prepare_dense_layer_weights(submesh, state_dict, DENSE_LAYER_IDX, ...)         # line 930
 ├── create_decoder_block_tensors(...)                                              # 933 — builds Q/K/V buffers, SDPA scratch, RMSNorm gammas, MoE shared weights, KV cache
 ├── create_decoder_golden_tensors(...)                                             # 948 — torch reference
 ├── create_global_semaphore × {4 reduce + 1 persistent_next_iter}                  # 967-968
 ├── AttentionBlock.create_semaphores(num_links_bcast=1, num_links_allreduce=2)     # 973
 ├── MoeOp.create_semaphores                                                        # 976
 ├── DecoderBlock.get_program_context(... enable_routing=False ...)                 # 979 — builds program descriptor
 │     ├── AttentionBlock.get_program_context (CCL bcast + RMS+MM stack, CBs, RTArgs)
 │     ├── MoeOp._build_descriptors (shared-expert-only CBs)
 │     └── Compiles decoder_block_kernel.cpp with #defines:
 │           - (no ENABLE_ROUTING)            ← enable_routing=False
 │           - ENABLE_REDUCE_TO_ONE = 1        ← reduce_root_coord != None
 │           - (no ENABLE_BCAST)
 │           - RECONFIG_MOE_CBS = 1
 ├── for i in range(num_iters=1):
 │     moe_final, attn_out = DecoderBlock.execute(*decoder_program_context)         # 1053 — ttnn.generic_op
 │
 │   ► On device:    while (iteration < num_internal_iterations) {
 │                       MLA_CB_RECONFIG; reset MATH/PACK sync state                 # 3039-3058
 │                       MLA: mla_body()                                             # 3071-3073
 │                       MOE_CB_RECONFIG                                             # 3075-3090
 │                       MOE: moe_body()                                             # 3091-3094
 │                       iteration++ }
 │
 ├── synchronize_device(submesh)                                                     # 1054
 ├── decoder_mlp_output = read back root device tensor                               # 1059-1068
 ├── DecoderBlock.golden(...) → torch ref MLP output                                  # 1081
 ├── Per-device KV cache validation (torch.equal for cold rows; PCC ≥ 0.98 for newly-written NOPE/ROPE on owning SP device)   # 1139-1176
 └── comp_pcc(moe_output.flatten(), decoder_mlp_output_valid.flatten(), 0.975)        # 1181 — main MLP PCC assert
```

The headline assertion is line 1185 (MLP PCC ≥ 0.975 against the torch golden). The KV-cache assertions at 1147 (`torch.equal`) and 1160/1164 (PCC ≥ 0.98 for NOPE / ROPE on the owning SP device) run before it.

---

## 4. Per-iteration kernel timeline

The `while (true)` loop at `decoder_block_kernel.cpp:3037` is the outermost device-side iteration. Each iteration runs the same five DeviceZone-scoped phases:

```
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│  iteration = 0..(num_internal_iterations - 1)                                              │
│                                                                                            │
│  ┌── MLA_CB_RECONFIG ── decoder_block_kernel.cpp:3038-3042 ─────────────────────────────┐  │
│  │  reconfig_cb_interfaces(mla_cb_config);  // remap CB ID-space to attention layout    │  │
│  │  setup_mla_sharded_buffers();            // tag the sharded buffers as ready         │  │
│  │                                                                                      │  │
│  │  TRISC ONLY (3055-3058):                                                             │  │
│  │     MATH(( llk_math_pack_sync_init<false>() ));   // reset MATH/PACK sync FSM        │  │
│  │     PACK(( llk_pack_dest_init<false,false>(0) )); // reset PACK to DEST bank 0       │  │
│  │     → THIS IS THE "WORKAROUND" KNOB. Comment at 3043-3054 explains it forces every   │  │
│  │       iter to start in DEST bank 0, papering over the iter-1 PCC divergence.         │  │
│  │                                                                                      │  │
│  │  NCRISC sender-core only (3060-3068, persistent path or sync sem reset)              │  │
│  └──────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                            │
│  ┌── MLA ── mla_body() lambda, defined 2232-2704, invoked at 3072 ────────────────────────┐│
│  │  See §5 for full breakdown.  In summary:                                              ││
│  │    CCL_BROADCAST  (skipped unless skip_ccl=False; persistent next_iter handshake)      ││
│  │    METADATA_BROADCAST    (mcast_metadata → DeepseekMetadata on all cores)              ││
│  │    cur_pos = metadata_ptr->position_id                                                 ││
│  │    RMSNORM  → MCAST       (RMSNorm input → broadcast to matmul cores)                  ││
│  │    MATMUL → GATHER → RMSNORM2 → MCAST2 → MATMUL2 → MATMUL3 (QNOPE) / QROPE / CREATE_Q_HEADS
│  │    KV-cache branch (skip on devices with no work for this position):                   ││
│  │       DKV_MATMUL → DKV_GATHER → KV_RMSNORM → K_ROPE → KV_CACHE_UPDATE                  ││
│  │       KV_CACHE_SIGNAL_READY (semaphore → MLA reader of the last chunk)                 ││
│  │    FLASH_MLA  (see §5)                                                                  ││
│  │    POST_SDPA  (sdpa_reduce_worker / sdpa_reduce_forwarder over fabric)                 ││
│  │    MATMUL4 → GATHER2 → MCAST3 → MATMUL5 → GATHER3 → CCL all-reduce                     ││
│  └────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                            │
│  ┌── MOE_CB_RECONFIG ── 3074-3090 ────────────────────────────────────────────────────────┐│
│  │  reconfig_cb_interfaces(moe_cb_config);                                                ││
│  │  all-reduce-receiver: drain ccl_sync_semaphore2 (3079-3088)                            ││
│  │  setup_moe_sharded_buffers();                                                          ││
│  └────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                            │
│  ┌── MOE ── moe_body() lambda, defined 2706, invoked at 3093 ──────────────────────────────┐│
│  │  RESIDUAL_MCAST (input → mcast receiver cores)                                          ││
│  │  RMSNORM (third RMSNorm of the block, on sender core)                                   ││
│  │  MCAST  (RMSNorm output broadcast)                                                      ││
│  │  ── ENABLE_ROUTING block is COMPILED OUT — no GateMM/Gather/Gate/MCAST_INDEX  ───       ││
│  │  SHARED_GU_MATMUL  (gate+up KN-sliced matmul, 128 compute cores, DENSE_SHARED_N=2048)   ││
│  │  SHARED_GATE_GATHER / SHARED_UP_GATHER (MoeGather → gated_reduce core)                  ││
│  │  SHARED_GATED_REDUCE   (SiLU(sum(gate)) * sum(up))                                      ││
│  │  GATE_PROJ / UP_PROJ / MUL — ALSO COMPILED OUT (guarded by ENABLE_ROUTING test? no:     ││
│  │  these are unconditional but they're DRAMStreamingMatmul ops whose 'IsActiveCore' is    ││
│  │  Core::Routed::is_gate_proj_core — on the dense-MLP test path the gate_proj weights     ││
│  │  *are* uploaded (op.py:1018-1020), so the gate_proj cores still execute, but with the   ││
│  │  shared-expert path also feeding eltwise_add.  Confirm with op.py paths.)               ││
│  │  SHARED_DOWN_MCAST → SHARED_DOWN_MATMUL → SHARED_RESIDUAL_ADD → SHARED_OUTPUT_GATHER    ││
│  │     → SHARED_OUTPUT_MCAST                                                                ││
│  │  DOWN_PROJ_GATHER → DOWN_PROJ_MCAST → DOWN_PROJ → ELTWISE_ADD                            ││
│  │  REDUCE_TO_ONE   (ENABLE_REDUCE_TO_ONE=1; 4×2 → 1)                                       ││
│  │  Reduce fabric cores → sender core sync sem (3007-3014)                                 ││
│  └─────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                            │
│  iteration++                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

After the loop exits:

```
MCAST_TEARDOWN  (decoder_block_kernel.cpp:3107-3108)
return from kernel_main
```

### 4.1 What changes between iter 0 and iter 1?

There is no host-side `pos_ptr += 1` issued between iterations of the *internal* loop — the test launches `DecoderBlock.execute` exactly once (`num_iters=1`), and the device-side `while(true)` simply re-reads `metadata_ptr->position_id` from L1 at the top of each `mla_body` (`decoder_block_kernel.cpp:2321`). The metadata L1 region is not mutated by the kernel within a launch.

Searching the kernel exhaustively for writes to position_id finds none inside `decoder_block_kernel.cpp`. The fields of `DeepseekMetadata` (`metadata/metadata.hpp`) are output-only (`tok0_pos`, `tok1_pos`, …) plus the input fields, and the only `metadata_ptr` access in the kernel is the read at line 2321.

**Therefore, between iter 0 and iter 1 (within a single host launch), `cur_pos` is the *same*.** What does change is the KV-cache write at the end of iter 0: the kernel does a real `KV_CACHE_UPDATE` (line 2486) into `slot_id=0, position=local_cur_pos`. On iter 1 the K-cache read (`flash_mla.hpp` BRISC, lines 314-426) therefore sees a cache whose `cur_pos`-th row has been re-written by iter 0 — and `cur_pos+1 % k_chunk_size != 0` so the masking row is the same in both iters. The numerical output of iter 1 thus depends on whether iter 0's KV-cache write was bit-exact with the value already there.

> The task brief mentioned an "NCRISC `noc_semaphore_set(...)` + BRISC `*pos_ptr += 1` between iters" — the BRISC pos_ptr increment is **not** present in this version of the kernel (verified via `grep` in §4.1 below). The only between-iter NCRISC `noc_semaphore_set` is the sender-core sync-sem reset at line 3066, guarded by `ENABLE_REDUCE_TO_ONE`. The cross-iter divergence reported empirically (0.99344836 vs 0.99318986) is therefore not driven by `position_id` mutation — it is purely a DEST-bank-parity effect.

---

## 5. Flash MLA in detail (`flash_mla.hpp` TRISC, lines 650-820)

`FlashMLADecode::Op<...>::impl(...)` is invoked once per `mla_body()` from `decoder_block_kernel.cpp:2497-2502`. The TRISC half does flash-attention decode with chunked K and a single `tile_regs_acquire/commit/wait/release` cycle around the entire chunk loop.

### 5.1 Compile-time constants pulled in (TRISC)

From `flash_mla.hpp:652-657` (`get_named_compile_time_arg_val`):

| Name        | Source CT-arg                       | Value (for this test)          |
|-------------|-------------------------------------|--------------------------------|
| `DHt`       | `"DHt"`                             | `18` (= `(512+64)/32`)         |
| `vDHt`      | `"vDHt"`                            | `16` (= `512/32`)              |
| `Sq_chunk_t`| `"PNHt"`                            | `1`                            |
| `Sk_chunk_t`| `"Sk_chunk_t"`                      | `4` (= `128/32`)               |
| `scale_fp32`| `"scale_fp32"`                      | host-supplied IEEE-754 bits    |
| `dst_size`  | `"dst_size"`                        | `8` (half-DEST tiles)          |

Derived:

```
q_chunk_tiles   = Sq_chunk_t * DHt  = 1 * 18 = 18
out_chunk_tiles = Sq_chunk_t * vDHt = 1 * 16 = 16    (asserted even at 672)
num_blocks      = vDHt / dst_size   = 16 / 8 = 2     (795)
block_size      = vDHt / num_blocks = 8              (797)
```

### 5.2 DEST-register offsets (flash_mla.hpp:708-716)

`packed_tile_size = 8 * 2 = 16` (8 rows × 2 cols, half-tile packing in half-DEST mode).

| Constant                  | Value                          | Holds                                                | Read by                    | Written by                    |
|---------------------------|--------------------------------|------------------------------------------------------|----------------------------|-------------------------------|
| `mm2_dst_offset`          | `0`                            | Running `softmax(QK)·V` accumulator (16 tiles)       | FPU (MM2) + PACK at tail   | FPU (`sdpa_custom_mm_reuse_dest_srcb_block`) |
| `mm2_dst_tile_offset`     | `0` (= `0/16`)                 | Same, expressed in tiles                             | PACK (`pack_block_contiguous`) | —                              |
| `max_dst_offset`          | `0 + 16 * 16 = 256`            | Per-row running max(QK)                              | FPU (bcast sub) + SFPU     | SFPU (`sdpa_reduce_max_row`) |
| `max_dst_tile_offset`     | `256 / 16 = 16`                | Same, in tiles                                       | PACK (writes ms tile)       | —                              |
| `sum_dst_offset`          | `256 + 2 = 258`                | Per-row running sum of exp                           | SFPU + FPU (final recip)   | SFPU (`sdpa_reduce_sum_row`) |
| `corr_exp_dst_offset`     | `256 + 16 = 272`               | Correction factor `exp(prev_max − cur_max) − 1`      | FPU (bcast mul) + SFPU     | SFPU (`non_approx_exp_mul_prev` / `recip_sum`) |
| `mm1_dst_offset`          | `272 + 16 = 288`               | QK^T result, `Sk_chunk_t` tiles wide                 | FPU (Q@K) + SFPU exp        | FPU (`sdpa_custom_mm_block`) + SFPU |
| `mm1_dst_tile_offset`     | `288 / 16 = 18`                | Same, in tiles                                       | —                          | —                              |

In half-DEST mode (`fp32_dest_acc_en=False`, `dst_full_sync_en=False` per `op.py:425`), the DEST bank holds 32 tiles × 16 packed-units each = 512 half-units. The above offsets span up to `mm1_dst_offset + Sk_chunk_t*16 = 288 + 64 = 352` units (well within bank capacity).

### 5.3 Setup (flash_mla.hpp:683-688)

```
reconfig_data_format<false,true>(cb_k_in, cb_q_in);   // SRCA=K, SRCB=Q
pack_reconfig_data_format<true>(cb_out_o);
PACK(( llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, Float16_b>() ));
PACK(  SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, true, scale_fp32, true) );
sdpa_custom_mm_block_init_pack_short();
```

Note that `llk_math_sfpu_sdpa_reduce_row_init` is invoked **inside `PACK(())`** — meaning the SFPU runs on TRISC2 (PACK), not TRISC1 (MATH). This pattern is used throughout the SDPA helpers and is core to the iter-parity discussion (§10).

### 5.4 K-chunk loop (flash_mla.hpp:690-774)

```
auto [k_num_chunks, k_chunk_start, k_chunk_end] =
    get_runtime_args(cur_pos, args.cur_batch, args.core_num_in_reduce,
                     args.num_cores_per_head, args.k_chunk_size);
if (k_chunk_start == k_chunk_end) return;

num_active_s_blocks = min(k_num_chunks, args.num_cores_per_head);   // ≤ 8
num_cores_to_wait   = count of tree-reduce steps where this S-block receives;

// Pick output CB based on role
sdpa_output_is_final = do_output && (!do_reduce || num_cores_to_wait == 0);
if (sdpa_output_is_final)        { out=cb_out_final; ms=cb_out_ms; }
else if (num_cores_to_wait > 0)  { out=cb_interm_out; ms=cb_interm_ms; }
else                              { out=cb_out_o;     ms=cb_out_ms; }

pack_block_contiguous_init(sdpa_output_cb);
mask_last_chunk = (k_chunk_end == k_num_chunks) && ((cur_pos+1) % k_chunk_size != 0);
if (mask_last_chunk) cb_wait_front(cb_mask, 1);

cb_wait_front(cb_q_in, q_chunk_tiles);
cb_reserve_back(sdpa_output_cb, vDHt);
cb_reserve_back(sdpa_ms_cb, Sq_chunk_t);

tile_regs_acquire();   //  ◄── ONE acquire wraps the entire chunk loop AND the tail.
for (chunk = 0; chunk < num_chunks; ++chunk) {
    last_chunk = (chunk == num_chunks - 1);
    compute_sdpa_chunk< Sk_chunk_t=4, q_chunk_tiles=18, out_chunk_tiles=16, ... >(
        cb_q_in, cb_k_in, cb_mask, sdpa_output_cb,
        mm1_dst_offset, mm2_dst_offset, max_dst_offset, sum_dst_offset, corr_exp_dst_offset,
        /*first_chunk=*/ chunk == 0,
        /*last_chunk=*/  !sdpa_output_is_final && last_chunk,
        /*mask_chunk=*/  mask_last_chunk && last_chunk);
}
if (!sdpa_output_is_final) {
    PACK( TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU) );
    pack_block_contiguous(max_dst_tile_offset, sdpa_ms_cb, 1);
    cb_push_back(sdpa_ms_cb, Sq_chunk_t);
} else {
    compute_sdpa_recip<out_chunk_tiles=16, exp_approx_mode=false, scale_bf16>(
        cb_q_in, sum_dst_offset, corr_exp_dst_offset, mm2_dst_offset);
}
for (i = 0; i < out_chunk_tiles; i += output_granularity=out_chunk_tiles=16) {
    PACK( t6_semaphore_wait_on_zero<STALL_PACK>(FPU_SFPU) );
    pack_block_contiguous(mm2_dst_tile_offset + i, sdpa_output_cb, output_granularity);
    PACK( t6_semaphore_get<PACK>(FPU_SFPU) );
}
cb_push_back(sdpa_output_cb, out_chunk_tiles);
tile_regs_commit();    //  ◄── flips DEST bank parity for the THREAD that ran the commit
tile_regs_wait();
tile_regs_release();
sdpa_custom_mm_block_uninit();
MATH( t6_semaphore_wait_on_max<STALL_SFPU>(FPU_SFPU) );
```

For the dense-MLP test, `do_reduce` and `do_output` are S-block / batch dependent, but every active core ends up with one full `tile_regs_acquire`→`tile_regs_release` cycle wrapping the chunk loop. **In half-DEST mode that single `tile_regs_commit/release` is exactly what flips MATH and PACK between bank 0 and bank 1 across `flash_mla` invocations.**

For the cores on the tree-reduction receiver path (`do_reduce && num_cores_to_wait > 0`), an additional reduction tail runs (lines 799-813) — a `sdpa_tail` per other partner — which has its **own** `tile_regs_acquire/commit/release` cycles inside `sdpa_tail_ms_reduce` and `sdpa_tail_l_block` (`sdpa.h:476/492/497`, 517/531/544). Each tail flip is independent.

### 5.5 Inside `compute_sdpa_chunk` (`sdpa.h:250-340`)

For one K-chunk (4 K-tiles, packed-tile size 16 each):

| # | Op                                            | Thread driver       | Notes                                                                                  |
|---|-----------------------------------------------|---------------------|----------------------------------------------------------------------------------------|
| 1 | `_init_sdpa_reduce_max_row_8x32_replay_buffers_` | **PACK** (line 265) | TRISC2 prepares SFPU replay buffers.                                                   |
| 2 | `sdpa_custom_mm_block_init_short<transpose_k>` | UNPACK+MATH (266)   | Tile-shape init.                                                                       |
| 3 | `cb_wait_front(cb_k, 4 * 4)`                  | all                 | Wait for 16 K-tiles.                                                                   |
| 4 | `MATH(t6_semaphore_wait_on_max<STALL_MATH>(FPU_SFPU))` (270) | TRISC1 (MATH) | Make sure PACK-SFPU's previous chunk is done.                                          |
| 5 | `sdpa_custom_mm_block<transpose_k>(...)` (271) | FPU (TRISC1)        | **QK^T**, writes `mm1_dst_offset` (4 tiles).                                            |
| 6 | `llk_math_sfpu_sdpa_reduce_max_row<...,chunk_size=4>(mm1,max,!first)` | **PACK** (274) | **SFPU on TRISC2** — reduces `mm1` rows to per-row max at `max_dst_offset`.            |
| 7 | `sdpa_sub_bcast_col_srca_srcb_reuse_tiles_init` | UNPACK+MATH (278)   | Init bcast-sub FPU pipeline.                                                           |
| 8 | `MATH(t6_semaphore_wait_on_max<STALL_MATH>(FPU_SFPU))` (279) | MATH                | Sync with PACK-SFPU.                                                                   |
| 9 | `sdpa_bcast_col_srca_srcb_reuse_preamble(max_dst_offset)` (280) | MATH         | Load max as broadcast operand.                                                         |
|10 | `sdpa_sub_bcast_col_srca_srcb_reuse_tiles<chunk_size=4,false>(mm1)` (281) | FPU         | **mm1 -= max** (per row, broadcast).                                                   |
|11 | *(if not first_chunk)* `non_approx_exp_mul_prev<exp_approx_mode=false, scale_bf16>(sum, corr_exp)` | **PACK** (286)   | **SFPU on TRISC2** — computes correction `exp(prev_max - cur_max) - 1` into corr_exp, mul prev sum. |
|12 | *(if not first)* `PACK(t6_semaphore_post<WAIT_SFPU>(SFPU_FPU))` (287) | PACK         | Signal MATH that correction tile is ready.                                             |
|13 | *(if not first)* `sdpa_mul_bcast_col_srca_srcb_reuse_tiles_init<num_tiles_v=16>` (290) | UNPACK+MATH    | Init bcast-mul.                                                                         |
|14 | *(if not first)* `MATH(t6_semaphore_wait_on_zero<STALL_MATH>(SFPU_FPU))` (291)         | MATH            | Wait for SFPU correction.                                                              |
|15 | *(if not first)* `sdpa_bcast_col_srca_srcb_reuse_preamble(corr_exp_dst_offset)` (292)  | MATH            |                                                                                        |
|16 | *(if not first)* `sdpa_mul_bcast_col_srca_srcb_reuse_tiles<16,true>(mm2)` (293)         | FPU             | **mm2 *= corr_exp** — rescale running accumulator.                                     |
|17 | `init_fast_approx_exp_constants<scale_fp32>` (301) | **PACK**       | Load LREG constants on TRISC2 SFPU.                                                    |
|18 | for `i in 0..chunk_size-1` (302-310): per-tile `fast_approx_exp(mm1 + i*16)` | **PACK** (306)  | **SFPU on TRISC2** — `mm1 := exp((mm1-max)*scale)` per QK tile.                         |
|19 | `sdpa_custom_mm_reuse_dest_srcb_block_init_short(...)` (313) | UNPACK+MATH     | Init MM2 (P@V).                                                                        |
|20 | `sdpa_custom_mm_reuse_dest_srcb_block<output_granularity=16>(...)` (314)             | FPU            | **MM2: mm2 += softmax(QK) · V**, granularity-paced semaphore posts to PACK.            |
|21 | `_init_sdpa_reduce_sum_row_8x32_replay_buffers_` (328) | **PACK**       | Re-init SFPU replay for sum.                                                            |
|22 | `llk_math_sfpu_sdpa_reduce_sum_row<...,chunk_size=4,true>(mm1, sum, !first)` (329)     | **PACK**       | **SFPU on TRISC2** — sum of exp(mm1) into `sum_dst_offset`, with `prev_sum` if not first. |
|23 | *(if not first)* PACK semaphore handshakes (332-338) | PACK            | Synchronise SFPU done.                                                                 |
|24 | `cb_pop_front(cb_k, 16)` (339) | NCRISC               | Free K tiles for next chunk.                                                            |

The "five PACK-frontend SFPU sites" called out in the task brief correspond to **lines 274, 286, 306, 329, 344** (in `sdpa.h`):

- 274: reduce_max (per-chunk)
- 286: correction-exp + mul-prev (per-chunk, skipped on first chunk)
- 306: per-tile fast_approx_exp (per-chunk, looped `chunk_size`=4 times)
- 329: reduce_sum (per-chunk)
- 344: recip_sum (only in the tail `compute_sdpa_recip`)

All five are `PACK((...))` macro-wrapped, i.e. they compile only into TRISC2's image and issue SFPU instructions from TRISC2.

### 5.6 The tail (lines 775-791 of `flash_mla.hpp`)

```
if (!sdpa_output_is_final) {              // tree-reduction worker path
    PACK( TTI_STALLWAIT(STALL_PACK, WAIT_SFPU) );
    pack_block_contiguous(max_dst_tile_offset, sdpa_ms_cb, 1);     // pack [max,sum] tile
    cb_push_back(sdpa_ms_cb, Sq_chunk_t);
} else {                                  // final-output path (no further reduction)
    compute_sdpa_recip<out_chunk_tiles=16, false, scale_bf16>(
        cb_q_in, sum_dst_offset, corr_exp_dst_offset, mm2_dst_offset);
    // ↑ PACK((recip_sum(...))) writes recip into corr_exp
    // ↑ MATH bcast-mul: mm2 *= recip       — at sdpa.h:344-350
}
for (i = 0; i < out_chunk_tiles; i += output_granularity=16) {
    PACK( t6_semaphore_wait_on_zero<STALL_PACK>(FPU_SFPU) );
    pack_block_contiguous(mm2_dst_tile_offset + i, sdpa_output_cb, output_granularity);
    PACK( t6_semaphore_get<PACK>(FPU_SFPU) );
}
cb_push_back(sdpa_output_cb, out_chunk_tiles);
tile_regs_commit(); tile_regs_wait(); tile_regs_release();
```

After release, MATH waits on the global FPU↔SFPU sem to drain (793).

### 5.7 Tree-reduction tails (lines 799-813)

If `do_reduce && num_cores_to_wait > 0`, MATH then runs `sdpa_tail` once per partner. The last partner picks one of two normalize modes depending on `is_sender_after_reduce`:

```
reconfig_data_format_srca<false,true>(cb_ms_in);
exp_tile_init<exp_approx_mode=false, scale_fp32>();
for i in 0..num_cores_to_wait-2:
    sdpa_tail<false, normalize=false, block_size=8, num_blocks=2, scale_fp32, C>(
        cb_ms_in, cb_interm_ms, cb_interm_ms, cb_out_in, cb_interm_out, cb_interm_out);
if (is_sender_after_reduce)
    sdpa_tail<false, normalize=false, 8, 2, scale_fp32, C>(
        cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_o);
else
    sdpa_tail<false, normalize=true , 8, 2, scale_fp32, C>(
        cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_final);
```

Each `sdpa_tail` (`sdpa.h:597-644`) does:

```
sdpa_tail_ms_reduce<...>(cb_worker_ms, cb_prev_ms, cb_cur_ms, cb_l1)
   ├── tile_regs_acquire()
   ├── copy_tile prev_ms→0, worker_ms→1
   ├── MATH(( fused_max_sub_exp_add_tile<approx, mode, normalize>(0, scale_bf16) ))    // SFPI kernel on TRISC1
   ├── sdpa_mul_bcast_col_reuse_tiles_init<block_size=8, dense=false>(cb_l1)
   ├── sdpa_bcast_col_reuse_preamble<normalize>()
   └── if (!normalize) { tile_regs_commit / cb_reserve_back / pack_tile / cb_push_back / tile_regs_release }

for each L-block (i = (normalize?1:0) ... num_blocks=2):
    sdpa_tail_l_block<8, 2, untilize=false, dense=false, manage_cbs=true>(cb_l1, cb_l2, cb_l_out, ...)
       ├── if acquire_regs: tile_regs_acquire()
       ├── cb_wait_front(cb_l2,8); cb_wait_front(cb_l1,8)
       ├── sdpa_mul_bcast_col_reuse_tiles<8>(cb_l2, cb_l1, ...)
       ├── cb_pop_front cb_l1,cb_l2; cb_reserve_back cb_l_out
       ├── tile_regs_commit(); tile_regs_wait()
       ├── pack_block_contiguous(0, cb_l_out, 8)
       └── tile_regs_release()

sdpa_tail_finalize<false>(cb_worker_ms, cb_prev_ms)
   └── sdpa_bcast_col_reuse_postamble()
```

So a single `sdpa_tail` produces between **2 and 3** `tile_regs_acquire/release` cycles depending on `normalize`, each flipping DEST-bank parity.

---

## 6. CB / DEST data-flow diagram (one `compute_sdpa_chunk` call)

```
                       ┌──── TRISC0 (UNPACK) ────┐ ┌──── TRISC1 (MATH/FPU) ───┐ ┌──── TRISC2 (PACK/SFPU) ──┐
                       │                          │ │                           │ │                           │
                       │                          │ │                           │ │ init reduce_max_row buf   │
                       │ unpack K (cb_k)→SRCA     │ │                           │ │                           │
                       │ unpack Q (cb_q)→SRCB     │ │ FPU: QK^T → DEST[mm1...]  │ │                           │
                       │                          │ │ (sdpa_custom_mm_block)    │ │                           │
                       │                          │ │                           │ │ SFPU: reduce_max_row      │
                       │                          │ │                           │ │   DEST[mm1] → DEST[max]   │
                       │ unpack max→SRCB-broadcast│ │ FPU: mm1 -= max bcast     │ │   (TT_SETC16 to mm1 base) │
                       │                          │ │                           │ │                           │
                       │                          │ │       [if !first chunk]   │ │  SFPU: corr_exp=exp((max  │
                       │                          │ │       MATH waits on SFPU  │ │   prev−max)*scale)-1      │
                       │                          │ │       (sem FPU_SFPU=0)    │ │   into DEST[corr_exp]     │
                       │                          │ │ FPU: mm2 *= corr_exp bcast│ │   posts FPU_SFPU=1        │
                       │                          │ │   into DEST[mm2...]       │ │                           │
                       │                          │ │                           │ │ for tile in 0..3:          │
                       │                          │ │                           │ │   SFPU: fast_approx_exp   │
                       │                          │ │                           │ │     DEST[mm1+i*16] in-pl. │
                       │                          │ │                           │ │     (TT_SETC16 each call) │
                       │ unpack V (cb_k strided)  │ │ FPU: mm2 += softmax · V   │ │                           │
                       │                          │ │   (sdpa_custom_mm_reuse_  │ │                           │
                       │                          │ │    dest_srcb_block, packs │ │                           │
                       │                          │ │    via FPU_SFPU sem in    │ │                           │
                       │                          │ │    granularity-16 groups) │ │                           │
                       │                          │ │                           │ │ init reduce_sum_row buf   │
                       │                          │ │                           │ │ SFPU: reduce_sum_row      │
                       │                          │ │                           │ │   DEST[mm1] → DEST[sum]   │
                       └──────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘

  CB usage in this loop body:
    cb_k_in    [pop 16 tiles per chunk after MM2]
    cb_q_in    [reused via SRCB across chunks; popped at end of flash_mla]
    cb_mask    [pop 1 tile at end if mask_last_chunk]

  DEST layout (half-DEST, bank x ∈ {0, 1}):
    [mm2_dst_offset=0 .. 256)            : 16 tiles QKV running output
    [max_dst_offset=256 .. 258)          : per-row max (1 tile worth, but only col 0)
    [sum_dst_offset=258 .. 272)          : per-row sum (col 1 of the same tile)
    [corr_exp_dst_offset=272 .. 288)     : correction factor tile
    [mm1_dst_offset=288 .. 288+64=352)   : 4 QK^T tiles per chunk (mm1)
```

After all chunks, the tail then:

```
  sdpa_output_is_final path:
                  ┌── TRISC2 (PACK/SFPU) ──┐  ┌── TRISC1 (MATH/FPU) ──┐
                  │ SFPU recip_sum         │  │                       │
                  │   DEST[sum] → DEST[corr│  │ FPU: mm2 *= recip      │
                  │       _exp] (TT_SETC16)│  │   bcast (fused_sig.)   │
                  │ post FPU_SFPU=1        │  │                       │
                  │                        │  │                       │
                  │ pack mm2 → cb_out_final│  │                       │
                  │   (block contiguous,   │  │                       │
                  │    output_granularity  │  │                       │
                  │    = 16)               │  │                       │
                  └────────────────────────┘  └───────────────────────┘
```

---

## 7. Compile-time defines / named CT args (decoder_block_kernel.cpp)

The kernel uses `get_named_compile_time_arg_val("…")` for ~100+ named compile-time args. The host wires them up in `decoder_block/op.py:399-438` and the prior `AttentionBlock.get_program_context` / `MoeOp._build_descriptors` calls.

### 7.1 Preprocessor `#define`s (set on the dense-MLP path)

| Define                  | Set when                                                   | Source                                      | Effect                                                                  |
|-------------------------|------------------------------------------------------------|---------------------------------------------|-------------------------------------------------------------------------|
| `ENABLE_ROUTING`        | `enable_routing=True` — **NOT set** for this test          | `moe/op.py:4741-4742`                       | Skips GateMM, Gather, DeepseekMoeGate, MCAST_INDEX, MCAST_EXPERT_SCALE |
| `ENABLE_REDUCE_TO_ONE`  | `reduce_root_coord` is provided — **SET** for this test    | `moe/op.py:4743-4744`                       | Activates the per-iter sender-core sync sem (3026-3034, 3061-3068, 3007-3014) and the ReduceToOne op |
| `ENABLE_BCAST`          | when an upstream socket is attached                        | `moe/op.py:4745-4746`                       | Not active here                                                          |
| `RECONFIG_MOE_CBS`      | `reconfig_moe_cbs=True` — **SET**                          | `moe/op.py:4747-4748`                       | Causes MoE-side `setup_moe_sharded_buffers()` in the MOE_CB_RECONFIG    |
| `COMPILE_FOR_BRISC` / `COMPILE_FOR_NCRISC` / `COMPILE_FOR_TRISC` | always (one of)                                  | toolchain                                   | Selects RISC role                                                       |
| `TRISC_MATH` / `TRISC_UNPACK` / `TRISC_PACK`                  | per TRISC subkernel                                | toolchain                                   | Selects TRISC thread (used heavily in `sdpa.h`)                          |

### 7.2 Named CT-arg categories (decoder_block_kernel.cpp:67-134)

| Category | Examples (all queried via `get_named_compile_time_arg_val(...)` at top of `kernel_main`) |
|----------|-------------------------------------------------------------------------------------------|
| **Core role flags** (one per role)            | `is_input_core`, `is_full_mcast_grid_core`, `is_matmul_core`, `is_matmul2_core`, `is_qnope_core`, `is_qrope_core`, `is_sdpa_input_core`, `is_dkv_matmul_core`, `is_kv_rmsnorm_core`, `is_knope_core`, `is_krope_core`, `is_mla_core`, `is_sdpa_worker_core`, `is_sdpa_forwarder_core`, `is_matmul4_core`, `is_gather_receiver_core`, `is_matmul5_core`, `is_allreduce_sender_core`, `is_allreduce_receiver_core`, `is_sender_core`, `is_mcast_grid_core`, `is_gate_mm_core`, `is_gate_proj_core`, `is_shared_compute_core`, `is_shared_gate_compute_core`, `is_shared_up_compute_core`, `is_shared_gated_reduce_core`, `is_shared_mcast_receiver_core`, `is_reduce_worker_core`, `is_reduce_fabric_core` |
| **Configuration**                              | `skip_ccl`, `persistent_mode`, `num_iterations`, `mesh_row`, `mesh_col`                  |
| **CB IDs**                                     | `mla_k_in_cb`, `mla_q_in_cb`, `mla_mask_cb`, `mla_interm_out_cb`, `mla_interm_ms_cb`, `mla_out_in_cb`, `mla_ms_in_cb`, `mla_out_o_cb`, `mla_out_ms_cb`, `mla_out_final_cb`, `rmsnorm_input_cb`, `mcast_dst_cb`, … |
| **CB tile counts**                             | `rmsnorm_num_tiles`, `mcast_dst_num_pages`, `matmul4_k_num_tiles`, `gather_reduce_src_num_pages`, … |
| **MLA dims**                                   | `DHt=18`, `vDHt=16`, `PNHt=1`, `Sk_chunk_t=4`, `St=1024`, `k_chunk_size=128`, `q_chunk_size_bytes`, `scale_fp32`, `dst_size=8`, `num_cores_per_head`, `q_heads_parallel_factor`, `num_tree_reduction_steps=3`, `k_num_pages`, `k_page_size` |
| **MLA cluster geometry**                       | `mla_sender_noc_x_0..7`, `mla_sender_noc_y_0..7`, `full_grid_mcast_start_x/y`, `full_grid_mcast_end_x/y`, `full_grid_mcast_num_dests`, `num_mcast_dests` |
| **Semaphore addresses**                        | `mla_kv_cache_cur_pos_ready_semaphore_addr` + value, `mla_q_input_mcast_semaphore_addr`, `mla_mcast_semaphore_addr`, `mla_receiver_ready_semaphore_addr`, `mla_ncrisc_brisc_sync_semaphore_addr`, `mla_reducer_semaphore_addr`, `mcast_metadata_receiver_semaphore_addr`, `mcast_data_receiver_semaphore_addr`, `risc_sync_semaphore_addr`, `ccl_sync_semaphore_addr`, `ccl_sync_semaphore2_addr`, `scatter_arrival_semaphore_addr`, `persistent_next_iter_sem_addr`, `reduce_sync_sem_addr` |
| **SP geometry**                                | `kv_cache_device_chunk_size=1024`, `kv_cache_sp_device_idx=0..3`, `kv_cache_num_sp_devices=4` |
| **CB-reconfig payload addresses**              | `mla_reconfig_cb_config_l1_addr`, `moe_reconfig_cb_config_l1_addr` (`decoder_block/op.py:311, 344`) |
| **Gather / fabric NOC coords**                 | `gather_reduce_dest_noc_x/y`, `gather_reduce_grid_start_x/y`, `sdpa_forwarder0/1_noc_x/y`, `ccl_sender_noc_x/y`, `input_noc_coord_x/y`, … |
| **Reduce-to-one**                              | `reduce_device_role` (MESH_ROOT0/MESH_ROOT1/regular), `reduce_sync_num_fabric_cores`, `reduce_brisc_rt_arg_base`, `reduce_brisc_fabric_rt_arg_base`, `reduce_ncrisc_common_rt_arg_base`, `num_ccl_sender_cores`, `sdpa_fwd_num_cores` |

Per-RISC TRISC compute config (`decoder_block/op.py:421-426`):
- `MathFidelity::LoFi`
- `math_approx_mode = False`
- `fp32_dest_acc_en = False`
- `dst_full_sync_en = False`     ← **this is what selects half-DEST mode**

---

## 8. Circular buffers used in the MLA phase

Defined in `fused_ops/attention_block/op.py:962-998` (CB ID allocation) and `:2588-2703` (CB descriptor / sharded buffer wiring). The `flash_mla` TRISC kernel pulls them from `ComputeCTArgs` (`flash_mla.hpp:104-126`).

| Logical name (kernel) | CB ID variable (host) | Tile shape | Format       | Backing | Pages | Purpose |
|-----------------------|-----------------------|------------|--------------|---------|-------|---------|
| `cb_q_in`             | `mla_q_in_cb` (= `create_q_heads_out_cb`) | 8×32 (Q-tiny tile) | `q_df` (bf16) | sharded over `mla_input_output_crs` | `q_tiles = PNHt*DHt = 18` | Q heads after CreateQHeads. Reused via SRCB across all K-chunks. |
| `cb_k_in`             | `mla_k_in_cb`         | 32×32 (KV tile)  | `k_df` (bfp8_b for KV-cache) | `ref_sdpa_kv_cache_buffer`, total `k_tiles*k_tile_size`, `k_tiles = Sk_chunk_t*DHt*2 = 144` (double-buffered) | one k-chunk = 4 * 18 = 72 tiles | K (and strided V) input — read from local KV-cache shard or mcast'd from the S-block sender. |
| `cb_mask`             | `mla_mask_cb`         | 8×32 (q-tiny)    | `q_df`       | `ref_sdpa_forwarder_scratch` (offset, size `q_tile_size`) | 1 | Last-chunk mask, built by NCRISC writer (`mask_last_chunk` in flash_mla.hpp:53-75). |
| `cb_interm_out`       | `mla_interm_out_cb`   | 8×32 (stats)     | `stats_df`   | `ref_sdpa_out_interm_buffer`, total `mla_intermed_output_tiles * stats_tile_size = out0_t * NUM_TREE_REDUCTION_STEPS * stats_size` (= `16*3` tiles) | — | Per-tree-reduction-step intermediate L tile, written by S-block senders. |
| `cb_interm_ms`        | `mla_interm_ms_cb`    | 8×32 (stats)     | `stats_df`   | `ref_sdpa_out_interm_buffer` (offset after `interm_out`) | `mla_intermed_ms_tiles = PNHt * NUM_TREE_REDUCTION_STEPS = 3` | Per-step intermediate max/sum tile.            |
| `cb_out_in`           | `mla_out_in_cb`       | 8×32 (stats)     | `stats_df`   | same as `interm_out` (aliased) | same | Receiver-side L tile input for `sdpa_tail` reductions. |
| `cb_ms_in`            | `mla_ms_in_cb`        | 8×32 (stats)     | `stats_df`   | same as `interm_ms` (aliased) | same | Receiver-side MS tile input. |
| `cb_out_o`            | `mla_out_o_cb`        | 8×32 (stats)     | `stats_df`   | `ref_input_tensor` (offset `input_running_offset`), total `out0_t * stats_tile_size = 16 * stats_size` | 16 | Final O tile written by the S1 (sender_after_reduce) core into the input-tensor scratch. Aliases `mla_interm_out_cb` in the CB descriptor (line 2677). |
| `cb_out_ms`           | `mla_out_ms_cb`       | 8×32 (stats)     | `stats_df`   | `ref_input_tensor` (offset), total `statistics_tiles * stats_tile_size = 1 * stats_size` | 1 | Final MS tile. Aliases `mla_interm_ms_cb` in descriptor (line 2700). |
| `cb_out_final`        | `mla_out_final_cb` = `mla_out_o_cb` | same   | `stats_df`   | same as `cb_out_o` (since unused in fused mode, alias) | same | Final O — used only when `sdpa_output_is_final`. |

Important format note (op.py:2675-2701): `mla_out_o_cb` and `mla_interm_out_cb` are configured with **two `CBFormatDescriptor` entries on the same CB-descriptor**, meaning the two CB IDs share the same L1 footprint but present different views. Likewise `mla_out_ms_cb` / `mla_interm_ms_cb`.

The MOE phase reuses CB-id space — that is the point of `reconfig_cb_interfaces(moe_cb_config)` at line 3076. MoE CBs are configured in `fused_ops/moe/op.py` (not enumerated here; the test uses only the shared-expert subset: residual_mcast, rmsnorm/mcast, shared_gu_matmul, shared_gate/up_gather, shared_gated_reduce, shared_down_mcast/matmul/residual_add/output_gather/output_mcast, gate_proj_gather/mcast, gate_proj/up_proj/mul, eltwise_add, reduce_to_one).

---

## 9. DEST register usage in `flash_mla` TRISC

Half-DEST mode (`fp32_dest_acc_en=False`, `dst_full_sync_en=False`) splits the DEST bank into two 512-half-unit halves. `packed_tile_size = 16` units. The constants in `flash_mla.hpp:708-716` plus `dst_size=8` (op.py:1646-1650):

| Constant (named in code)  | Numeric value | Tile range it spans            | Holds                                      | Read by             | Written by                       |
|---------------------------|---------------|--------------------------------|--------------------------------------------|---------------------|----------------------------------|
| `mm2_dst_offset`          | `0`           | tiles 0…15 (= `vDHt=16`)        | softmax(QK)·V accumulator                  | FPU MM2, PACK tail  | FPU MM2 (`sdpa_custom_mm_reuse_dest_srcb_block`), FPU rescale (`sdpa_mul_bcast_..._reuse_tiles<v,true>(mm2)`)  |
| `max_dst_offset`          | `256`         | tile 16, col 0                  | per-row running max                        | FPU bcast-sub, FPU recip path, MATH copy | SFPU `reduce_max_row` (PACK frontend) |
| `sum_dst_offset`          | `258` (= max+2) | tile 16, col 1                | per-row running sum-of-exp                 | SFPU `recip_sum`     | SFPU `reduce_sum_row` (PACK frontend) |
| `corr_exp_dst_offset`     | `272` (= max+16) | tile 17                       | exp(prev_max−cur_max)−1 correction         | FPU bcast-mul        | SFPU `non_approx_exp_mul_prev` / SFPU `recip_sum` (PACK frontend) |
| `mm1_dst_offset`          | `288` (= corr+16) | tiles 18…21 (= `Sk_chunk_t=4`) | QK^T result                                | FPU bcast-sub, SFPU `fast_approx_exp`, SFPU `reduce_max_row` / `reduce_sum_row` (read), FPU MM2 (read) | FPU `sdpa_custom_mm_block`, SFPU `fast_approx_exp` (in-place) |

The thread that *commits* DEST is the one that wins the parity flip in half-DEST: PACK is the thread issuing the final commit/release via the SFPU tail's `pack_block_contiguous` (`flash_mla.hpp:783-787, 789`). MATH stalls on the FPU_SFPU semaphore until PACK signals completion (793).

---

## 10. Why iter-to-iter parity matters here

### 10.1 Half-DEST bank-flip mechanics

In half-DEST mode the 64-tile DEST bank is partitioned into two 32-tile sub-banks (HALF_SIZE = 32 tiles). On Blackhole, three independent address registers must track the active sub-bank:

| Register                               | Owned by | Where flipped                                                                                 |
|----------------------------------------|----------|------------------------------------------------------------------------------------------------|
| `TRISC1.MATH_Offset` (DEST_TARGET_REG_CFG_MATH_Offset_ADDR32 on MATH thread) | MATH (TRISC1) | At `tile_regs_commit()` / `tile_regs_release()` |
| `TRISC2.SW.dest_offset_id`             | PACK (TRISC2) | At `tile_regs_release()` (PACK side)                                                          |
| `PACK_SEC0..PACK_SEC3`                 | PACK (TRISC2) | At `tile_regs_release()` (PACK side)                                                          |

Within `flash_mla`, **one** `tile_regs_acquire/commit/wait/release` wraps the chunk loop and the tail (lines 747-791). So `flash_mla` itself flips DEST-bank parity exactly **once** per invocation on each thread.

### 10.2 What iter 0 vs iter 1 looks like, *without* the workaround

Removing the iter-top `MATH((llk_math_pack_sync_init<false>()))` / `PACK((llk_pack_dest_init<false,false>(0)))` at `decoder_block_kernel.cpp:3055-3058`:

```
iter 0:
  acquire/commit/release in mla_body/flash_mla   → bank flip: 0 → 1 on MATH, PACK
  (post-SDPA + tail reductions: more flips, evens out by some count N)
  acquire/commit/release in moe_body steps        → more flips

iter 1:
  initial DEST bank: bank-1 (or whichever parity the previous iter's last release left)
  flash_mla writes everything into BANK 1 this iter
  …
```

Empirically the MLP PCC on iter 1 is **0.99344836** vs iter 0's **0.99318986** — a stable, reproducible offset. The two are bit-different even though every named DEST-addressing register has the correct base offset for its bank.

### 10.3 What the workaround does

The 4 lines at 3055-3058 forcibly re-initialize **TRISC1.MATH_Offset = 0** (bank 0) and **PACK's `llk_pack_dest_init<false,false>(0)`** — restoring PACK_SEC0..3 and TRISC2.SW.dest_offset_id to bank 0. So every iter starts with all three registers pointing at bank 0, and `flash_mla` writes to bank 0 every iter. PCC stops alternating.

### 10.4 What's already been ruled out (per the comment at 3043-3054)

> the SDPA SFPU helpers (`ckernel_sfpu_sdpa_reduce_row.h:144,177,202` and `sdpa.h:169,180,200`) already `TT_SETC16` the per-thread MATH_Offset to `src_index + get_dest_buffer_base()` themselves before every PACK-frontend SFPLOAD, so a write to MATH_Offset at iter top is overwritten before any SFPU read.

Concretely:
- `sdpa.h:169` — `fast_approx_exp(dst_index)`: `TT_SETC16(... , dst_index + get_dest_buffer_base())`
- `sdpa.h:180` — `non_approx_exp_mul_prev(..., corr_exp_index)`: same pattern
- `sdpa.h:200` — `non_approx_exp_mul_prev(curr_sum_index, ...)`: same
- `sdpa.h:226` — `recip_sum(..., recip_dst_index)`: same
- `ckernel_sfpu_sdpa_reduce_row.h:144,177,202` — three more `TT_SETC16` calls inside the reduce-row LLK

`get_dest_buffer_base()` returns the bank-correct base offset for whichever sub-bank the **calling thread** currently sees, so TRISC2's PACK-frontend SFPU always reads/writes the right physical DEST cell on its end.

### 10.5 What's still open

The "MATH_Offset is correctly maintained on both threads" diagnosis above does not explain why bank 0 and bank 1 produce numerically different outputs from the same Q, K, mask, and scale inputs. The DEST sub-banks are physically symmetric SRAM cells on the silicon, so any divergence must come from a register whose value depends on which sub-bank is active and that **isn't** routinely re-pointed before each use. Candidates worth pursuing:

- PACK section registers other than PACK_SEC0..3 that participate in DEST address generation (e.g. PACK_DEST_OFFSET fields not touched by `llk_pack_dest_init`).
- UNPACK-side DEST mirroring registers in `dest_srcb_reuse` paths — `sdpa_custom_mm_reuse_dest_srcb` (used in MM2) reads DEST through SRCB, and the SRCB-reuse base pointer is set with its own `TT_SETC16(... , dst_offset)` (e.g. `llk_math_sdpa_custom_mm.h:101`, `llk_math_sdpa_custom_mm_reuse_dest_srcb.h:122,130`).
- Any DEST replay-buffer state that was primed in bank-0 but is consumed in bank-1 (the `_init_sdpa_reduce_*_row_*_replay_buffers_` calls at sdpa.h:265 and 328 happen once per `compute_sdpa_chunk`, but if the replay buffers internally store an offset that's later adjusted by the bank pointer, that could diverge).
- Race between MATH's `t6_semaphore_wait_on_max(FPU_SFPU)` at the bottom of flash_mla (line 793) and PACK's last semaphore_post — possibly leaving a stale `MATH_Offset` from the previous tail's `sdpa_tail_l_block` packing if the order differs between banks. **Not determined from source** which of these is the actual culprit.

What's certain from the code is:
- `flash_mla` does exactly one bank flip per invocation (acquire/commit/release pair at 747/789-791).
- The PACK-frontend SFPU writes in `compute_sdpa_chunk` (sdpa.h:274, 286, 306, 329) and in `compute_sdpa_recip` (sdpa.h:344) all flow through `TT_SETC16(MATH_Offset, … + get_dest_buffer_base())`, so each of them re-points MATH_Offset on its calling thread.
- The workaround at decoder_block_kernel.cpp:3055-3058 forcibly re-init's MATH_Offset and PACK_SEC0..3 to bank 0 at iter top, but does not touch every register a DEST read might consult (in particular, dst_offset_id state inside dest_srcb-reuse MM paths is not explicitly reset).

---

## 11. Appendix — sources cited

| File                                                                                                          | Lines           | What it gives us                                                                |
|---------------------------------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------------|
| `models/demos/deepseek_v3_b1/tests/unit_tests/test_decoder_block.py`                                          | 850-1189        | The test harness (params, golden, PCC checks)                                   |
| `models/demos/deepseek_v3_b1/tests/unit_tests/test_moe_mlp.py`                                                | 116-117         | `DENSE_LAYER_IDX=0`, `DENSE_SHARED_N=2048`                                       |
| `models/demos/deepseek_v3_b1/fused_ops/decoder_block/kernels/decoder_block_kernel.cpp`                        | 67-134, 2188-3109 | Kernel role flags, body lambdas, `while(true)` loop, half-DEST workaround       |
| `models/demos/deepseek_v3_b1/fused_ops/decoder_block/op.py`                                                   | 132-595         | Program-context build; CT-arg/CB wiring                                          |
| `models/demos/deepseek_v3_b1/fused_ops/attention_block/op.py`                                                 | 950-998, 1580-1779, 2577-2703 | MLA CB-id allocation, CT-args, CB descriptors                                    |
| `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`                                                   | 1-820           | The `FlashMLADecode::Op` struct; BRISC/NCRISC/TRISC bodies; TRISC at 650-820   |
| `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h`                      | 144-646         | `compute_sdpa_chunk`, `compute_sdpa_recip`, `sdpa_tail_*`, SFPI fused tile      |
| `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sdpa_reduce_row.h` | 144,177,202     | The three `TT_SETC16(MATH_Offset, … + get_dest_buffer_base())` in the SDPA reduce-row LLK |
| `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`                                                       | 70-265          | `FlashMLAOptimalGridNOC0` (BLOCKS, NUM_BLOCKS=8, NUM_TREE_REDUCTION_STEPS=3, k_chunk_size=128, device_chunk_size=1024) |
| `models/demos/deepseek_v3_b1/metadata/metadata.hpp`                                                           | 12-25           | `DeepseekMetadata` layout (position_id is plain `uint32_t`)                      |
| `models/demos/deepseek_v3_b1/fused_ops/moe/op.py`                                                             | 4734-4749       | Kernel `#define`s (`ENABLE_ROUTING`, `ENABLE_REDUCE_TO_ONE`, `RECONFIG_MOE_CBS`)|

---
