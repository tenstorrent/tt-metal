# `test_decoder_mlp` — Workload Flow Overview

> Briefer companion to [`half-dest-workload.md`](./half-dest-workload.md). Focused on the
> bare structure needed to reason about the half-DEST iter-PCC bug (#43563): where
> `flash_mla` fits, what `moe_output` is, chunks vs iterations, and when DEST bank
> parity flips.

---

## One program launch — the big picture

```
HOST (test_decoder_mlp, Python)
 ├─ Build state dict (torch weights for one decoder layer)
 ├─ create_decoder_block_tensors  → upload Q/KV/cache/weights/sems to device
 ├─ DecoderBlock.execute(..., num_internal_iterations=1 or 2)
 │      └─ ttnn.generic_op → ONE kernel launch of decoder_block_kernel.cpp
 │
 ├─ Read back final tensor from root device (the "decoder MLP output")
 ├─ DecoderBlock.golden(...) → torch reference MLP output
 └─ comp_pcc(device_output, torch_golden)   →  MLP PCC value (>= 0.975 passes)
```

`moe_output` (in test code) = the **torch golden** post-MoE tensor used as reference.
"MLP PCC" compares the device's final output against this:

- Pearson correlation in `[-1, 1]`; 1.0 = perfect linear correlation.
- Threshold 0.975 in the test; the observed values (0.9901 - 0.9935) all pass.
- The bug we are chasing is the **iter-to-iter difference** in PCC, not threshold violation.

---

## Device kernel: `decoder_block_kernel.cpp::kernel_main()`

```
kernel_main():
    initial setup (PACK/MATH init, sems, CBs, fabric)

    uint32_t iteration = 0;
    while (true) {                         ←─ "internal iterations" loop
        MLA_CB_RECONFIG                    ←─ reconfig CB layout for MLA
        (Austin workaround: re-init MATH/PACK sync state)
        ENABLE_REDUCE_TO_ONE sem wait (NCRISC sender only)
        mla_body()                         ←─ everything BEFORE MoE
        MOE_CB_RECONFIG                    ←─ swap CB layout for MoE
        moe_body()                         ←─ MoE/MLP stage
        iteration++;
        if (iteration >= num_iterations) break;
    }
    MCAST_TEARDOWN
```

- `num_internal_iterations ∈ {1, 2}` from the test parametrize → while-loop trip count.
- Each iteration writes a result to a fixed output buffer; the host's read-back
  always gets the LAST iteration's result.
- **All iterations share state**: same kernel, same DEST register file, same CBs.
  Bank parity, semaphore counters, CFG registers all carry across.

---

## Inside `mla_body()` — the attention pipeline

```
mla_body():
    ├─ CCL_BROADCAST           (skipped: skip_ccl=False, but ENABLE_BCAST not set)
    ├─ METADATA_BROADCAST      (broadcast position_id from sender core)
    ├─ cur_pos = metadata_ptr->position_id
    │
    ├─ pre-SDPA stack ("input matmul chain"):
    │     RMSNORM → MCAST → MATMUL(q_a) → GATHER → RMSNORM2 → MCAST2
    │     → MATMUL2(q_b) → MATMUL3(QNOPE) / QROPE → CreateQHeads
    │
    ├─ KV-cache branch (writes new KV row):
    │     DKV_MATMUL → DKV_GATHER → KV_RMSNORM → K_ROPE → KV_CACHE_UPDATE
    │     → signal KV_CACHE_READY semaphore
    │
    ├─ FLASH_MLA  ◀──────────── flash_mla.hpp lives here
    │
    ├─ post-SDPA:
    │     sdpa_reduce_worker / sdpa_reduce_forwarder (tree-reduce across S-blocks)
    │
    └─ post-attention output path:
          MATMUL4 → GATHER2 → MCAST3 → MATMUL5 → GATHER3 → CCL all-reduce
```

`moe_body()` then runs the shared-expert + reduce-to-one to produce what the host
reads back as the "decoder MLP output".

---

## Inside `flash_mla.hpp` (TRISC compute) — the bug zone

```
TRISC compute (lines 651-819):
    ├─ reconfig_data_format, init helpers
    ├─ cur_pos = args.local_cur_pos
    ├─ work assignment → (k_chunk_start, k_chunk_end, num_chunks)
    ├─ if (k_chunk_start == k_chunk_end) return;   ←─ inactive cores skip
    │
    ├─ cb_wait_front(cb_q_in, …)   ;  cb_wait_front(cb_mask) if masked
    ├─ cb_reserve_back(sdpa_output_cb, …)
    │
    ├─ tile_regs_acquire();        ←──── DEST acquire (line 747)
    │
    │   for (chunk = 0; chunk < num_chunks; chunk++) {
    │       compute_sdpa_chunk(...)                ← one k-chunk's compute
    │   }
    │
    │   if (!sdpa_output_is_final) pack max+sum   ← tree-reduce worker tail
    │   else compute_sdpa_recip(...)               ← final-output tail
    │   for (i ...) pack_block_contiguous(mm2 → sdpa_output_cb)  ← final pack
    │   cb_push_back(sdpa_output_cb, …)
    │
    ├─ tile_regs_commit/wait/release  ←─── DEST release (line 791)
    │
    └─ if (tree-reduce-receiver) sdpa_tail × N      ← extra reduce work (own tile_regs cycles)
```

**One `tile_regs_acquire/release` wraps the ENTIRE chunk loop and tail.** So:

- The chunk loop runs at a **single DEST bank** (no flips between chunks within
  one flash_mla call).
- Bank parity flips exactly **once** per flash_mla invocation (at the release).

---

## Inside `compute_sdpa_chunk` (sdpa.h:250) — one k-chunk

```
compute_sdpa_chunk():
    PACK: init reduce_max replay buffer
    UNPACK+MATH: sdpa_custom_mm_block_init
    cb_wait_front(cb_k, …)

    MATH: MM1  (Q @ K^T → mm1_dst_offset)               ← FPU
    PACK/SFPU: reduce_max_row (mm1 → max_dst_offset)    ← SFPU
    MATH: bcast-sub (mm1 -= max, in place)              ← FPU

    if (!first_chunk):                                  ← only chunks 1+
        PACK/SFPU: non_approx_exp_mul_prev              ← corr_exp + rescale prev sum
        MATH: bcast-mul (mm2 *= corr_exp)               ← FPU (rescales running mm2)

    PACK: init_fast_approx_exp_constants
    for (i in 0..chunk_size-1):
        PACK/SFPU: fast_approx_exp (mm1+i in-place)     ← SFPU per tile
        FPU↔SFPU semaphore handshake

    MATH: MM2 (softmax(QK) @ V → mm2_dst_offset)        ← FPU via DEST→SRCB reuse
    PACK: init reduce_sum replay buffer
    PACK/SFPU: reduce_sum_row (mm1 → sum_dst_offset)

    cb_pop_front(cb_k, …)
```

DEST tiles in use within one chunk:

| Tile | Role | Lifetime |
|---|---|---|
| `mm1_dst_offset` | Current chunk's QK^T values (4 tiles) | overwritten each chunk |
| `mm2_dst_offset` | Running P·V accumulator (16 tiles) | **accumulates across chunks** |
| `max_dst_offset` | Cumulative row max (1 tile) | **read+written across chunks** via SFPSTORE/SFPLOAD |
| `sum_dst_offset` | Cumulative row sum (1 tile) | **read+written across chunks** via SFPSTORE/SFPLOAD |
| `corr_exp_dst_offset` | Per-chunk correction factor | overwritten each chunk |

---

## Chunks vs iterations

| Concept | What it is | Set by | Count |
|---|---|---|---|
| **chunk** | A k-axis slice of the attention key matrix, `k_chunk_size=128` positions | host config | `num_chunks` per flash_mla call (depends on `position_id`) |
| **internal iteration** | One full pass of the while-loop body in `kernel_main()` (mla_body + moe_body) | `num_internal_iterations` test param | 1 or 2 |
| **outer iteration** | A full host launch of `DecoderBlock.execute()` | `num_iters` (=1 in our test) | 1 |

- `position_id=127` → 1 chunk total.
- `position_id=511` → 4 chunks total.
- `position_id=8190` → 64 chunks total.

---

## DEST bank flip in half-DEST mode

The test runs with `use_fp32=False`, `dst_full_sync_en=False` → "half-DEST" mode.

```
Half-DEST mode:
    DEST register file split in 2 banks, 32 tiles each.
    Each tile_regs cycle:
        tile_regs_acquire        → MATH waits for free bank
        ...MATH writes DEST...
        tile_regs_commit         → MATH flips its dest_offset_id (bank 0 ↔ 1)
        tile_regs_wait
        ...PACK reads DEST...
        tile_regs_release        → PACK flips its dest_offset_id, ZEROACCs old bank
```

Per-thread `dest_offset_id` (TRISC1 MATH and TRISC2 PACK each maintain their own):

- Starts at 0 (kernel boot).
- TRISC1 toggles at every `tile_regs_commit`.
- TRISC2 toggles at every `tile_regs_release`.
- They stay in sync because commit and release are paired in every `tile_regs` cycle.

### Across iterations

- `flash_mla` flips bank parity exactly **once** per call (the single
  acquire/release pair wrapping the entire chunk loop and tail).
- Other ops (pre-SDPA matmuls, RMSNorms, MM4, MM5, MoE matmuls, etc.) each flip
  once per call too.
- Total flip count per iter — call it `N` — is **odd**. (If it were even, iter-1
  would start in bank 0 too and there'd be no parity dependence; the bug wouldn't
  be observable.)
- So **iter-0** starts at bank 0, ends at bank 1; **iter-1** starts at bank 1,
  ends at bank 0.

---

## Where the bug lives (current understanding)

```
flash_mla.hpp:747-791  (the chunk loop + tail + final pack, inside one tile_regs_acquire/release)
    └─ specifically: some op produces slightly different output when DEST bank parity is 1 vs 0
       even though Q, K, mask inputs are bit-identical (Attempt 18 confirmed).
```

- Per-core SDPA output differs between iter-0 (bank 0) and iter-1 (bank 1) on
  **every** flash_mla call regardless of chunk count (Attempt 19).
- At 1 chunk the downstream stack (post-SDPA → matmul4/5 → CCL all-reduce → MoE)
  cancels the small per-core differences before they reach MLP PCC — MLP PCC
  is bit-identical between iter-0 and iter-1.
- At multi-chunk the cumulative drift inside flash_mla compounds enough that
  downstream stops canceling, surfacing as the ~2.6e-4 MLP PCC alternation
  tracked in [#43563](https://github.com/tenstorrent/tt-metal/issues/43563).

See [`debug_log.md`](./debug_log.md) for the running record of attempts and what
each rules in/out.
