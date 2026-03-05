# Issue #37171: `test_layer_norm_sharded_two_stage` Watcher Failure Investigation

## Summary

When `TT_METAL_WATCHER=1` is enabled, `test_layer_norm_sharded_two_stage` fails with a
`DebugAssertNCriscNOCReadsFlushedTripped` assert on core (0,0) BRISC, indicating the kernel
`reader_mcast_sender_unary_sharded_ln.cpp` completed with pending NOC read transactions.

**Without watcher:** test passes in ~1.08s.
**With watcher:** test aborts due to tripped assert.

## Root Cause: CONFIRMED

**The kernel calls `noc_async_read_one_packet` with a size that exceeds `NOC_MAX_BURST_SIZE`,
causing the NOC hardware to split the response into multiple packets. The software counter
(`noc_reads_num_issued`) only increments by 1 per call, but the hardware counter
(`NIU_MST_RD_RESP_RECEIVED`) increments once per response packet. This is a software bug
in the kernel's `constexpr` size guard, not a hardware quirk.**

### The buggy code

In `reader_mcast_sender_unary_sharded_ln.cpp`, the "gather final results" section:

```cpp
if constexpr (num_tiles_per_worker_bytes <= NOC_MAX_BURST_SIZE) {
    noc_async_read_one_packet(noc_addr_ex, l1_write_addr_ex_global, num_tiles_scaler * num_tiles_bytes);
} else {
    noc_async_read(noc_addr_ex, l1_write_addr_ex_global, num_tiles_scaler * num_tiles_bytes);
}
```

The `constexpr` check tests `num_tiles_per_worker_bytes` against `NOC_MAX_BURST_SIZE`, but
the **actual read size** passed to `noc_async_read_one_packet` is
`num_tiles_scaler * num_tiles_bytes`, which can be up to **2x larger** when welford mode
is active (`num_tiles_scaler = 2`).

### Concrete values from the failing test

```
rdbarrier_A[0] rd_size=8192    ← per-tile reads: exactly NOC_MAX_BURST_SIZE, no issue
SIZES gather_read=16384 MAX_BURST=8192 tile=4096 scaler=2
```

- `single_tile_size_bytes = 4096` (Float32 accumulation tiles)
- `num_tiles_per_worker_bytes = 8192` (2 tiles × 4096 bytes)
- `NOC_MAX_BURST_SIZE = 8192` (Wormhole: 256 words × 32 bytes)
- **Constexpr check:** `8192 <= 8192` → **TRUE** → takes the `noc_async_read_one_packet` path
- **Actual read size:** `num_tiles_scaler(2) × num_tiles_per_worker_bytes(8192) = 16384`
- **16384 is 2× the max burst size**

### What happens at the hardware level

When `noc_async_read_one_packet` sends a single NOC read request for 16384 bytes, the
hardware splits the response into `ceil(16384 / 8192) = 2` response packets. Each packet
increments `NIU_MST_RD_RESP_RECEIVED`. But the software only calls `ncrisc_noc_fast_read`
once, incrementing `noc_reads_num_issued` by 1.

With 2 gather reads (one per `num_all_to_all_workers_first_stage`):
- Software counter: +2 (one per `noc_async_read_one_packet` call)
- Hardware counter: +4 (two response packets per oversized read)
- **Delta: +2 extra hardware responses**

Evidence from `c_tensix_core.h` confirming per-packet ack counting:
```cpp
uint32_t num_acks = size / NOC_MAX_BURST_SIZE + ((size % NOC_MAX_BURST_SIZE) ? 1 : 0);
```

### Why the read barrier exits "successfully"

The barrier spins on `NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[noc]`.
With 194 total issued reads (SW), the hardware responses arrive over time:

1. All normal reads complete: responses 1–192
2. Gather read #1, first half: response 193
3. Gather read #1, second half: response 194 → **HW(194) == SW(194), barrier exits**
4. Gather read #2, first half: response 195 (arrives after barrier exit)
5. Gather read #2, second half: response 196 (arrives after barrier exit)

The barrier exits at step 3 because equality is momentarily achieved, even though
gather read #2's responses haven't arrived yet. The +2 responses land shortly after.

### Why earlier reads (barriers A, B) are unaffected

The earlier reads use `num_tiles_scaler * single_tile_size_bytes = 2 × 4096 = 8192`,
which is exactly `NOC_MAX_BURST_SIZE`. At this size, each read generates exactly 1
response packet, so SW and HW counters match perfectly.

### How we proved this

1. **Logged actual read sizes**: `gather_read=16384 > MAX_BURST=8192` confirmed the
   oversized read. Earlier reads: `rd_size=8192` (exactly at limit, no overflow).

2. **Intentionally issued an oversized read** (> `NOC_MAX_BURST_SIZE`) after barrier C:
   the subsequent `noc_async_read_barrier()` **hung**, because `NIU_MST_RD_RESP_RECEIVED`
   was already ahead of `noc_reads_num_issued` from the gather reads' extra responses,
   and the `==` equality check could never be satisfied.

3. **Earlier experiments** (8+ iterations, originally misattributed to "phantom" responses):
   - Polling after barrier C with no writes showed +2 arriving on its own → actually the
     delayed second response packets from the oversized gather reads
   - Only barrier C (gather reads) showed +2, never barriers A/B → because only the
     gather reads exceed `NOC_MAX_BURST_SIZE`
   - The +2 was always exactly 2 → 2 gather reads × 1 extra response packet each

### Earlier hypotheses (all disproven, explained by this root cause)

All earlier hypotheses (multicast writes, linked flag, cmd_buf identity, VC state,
"phantom hardware responses") were red herrings. The +2 was misattributed to write
operations because they were the first thing executed after the barrier, by which time
the delayed response packets from the oversized reads had arrived. The real cause was
the oversized `noc_async_read_one_packet` call all along.

---

## The Fix

Replace `noc_async_read_one_packet` with `noc_async_read` for the gather reads.
`noc_async_read` internally splits reads larger than `NOC_MAX_BURST_SIZE` into
multiple sub-reads, correctly incrementing `noc_reads_num_issued` for each:

```cpp
// Before (buggy): constexpr check doesn't account for num_tiles_scaler
if constexpr (num_tiles_per_worker_bytes <= NOC_MAX_BURST_SIZE) {
    noc_async_read_one_packet(noc_addr_ex, l1_write_addr_ex_global, num_tiles_scaler * num_tiles_bytes);
} else {
    noc_async_read(noc_addr_ex, l1_write_addr_ex_global, num_tiles_scaler * num_tiles_bytes);
}

// After (fixed): always use the multi-packet version for gather reads
noc_async_read(noc_addr_ex, l1_write_addr_ex_global, num_tiles_scaler * num_tiles_bytes);
```

This is a one-line change. No workarounds, no resync hacks, no firmware changes needed.

---

## Test Command

```bash
export TT_METAL_WATCHER=1  # omit for non-watcher run
pytest tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py::test_layer_norm_sharded_two_stage \
  -k "dtype=torch.bfloat16-tensor_type=ascending_values_repeated_rows-two_stage=True-use_welford=True-h=128-w=256-num_cores_h=4-num_cores_w=2-block_ht=4-block_wt=1-subblock_wt=1"
```

---

## Hardware/Software Context

- **Device:** N150 (single-chip Wormhole)
- **NOC Mode:** `DM_DEDICATED_NOC` (BRISC uses NOC0, NCRISC uses NOC1)
- **Wormhole NOC parameters:**
  - `NOC_WORD_BYTES = 32` (256-bit payload)
  - `NOC_MAX_BURST_WORDS = 256`
  - `NOC_MAX_BURST_SIZE = 8192` bytes (256 × 32)
- **Kernel mapping on each Tensix core:**
  - **BRISC (RISCV_0):** `reader_mcast_sender_unary_sharded_ln.cpp` — uses NOC0
  - **NCRISC (RISCV_1):** `writer_unary_sharded_ln.cpp` — uses NOC1
  - **TRISC0/1/2:** `layernorm_sharded_welford.cpp` (compute)

---

## The Watcher Assert

### Where it triggers

In `brisck.cc`, after `kernel_main()` returns:

```cpp
if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
    WAYPOINT("NKFW");
    ASSERT(ncrisc_noc_reads_flushed(NOC_INDEX), DebugAssertNCriscNOCReadsFlushedTripped);
}
```

### What `ncrisc_noc_reads_flushed` checks

```cpp
inline bool ncrisc_noc_reads_flushed(uint32_t noc) {
    return (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]);
}
```

The assert fires because `NIU_MST_RD_RESP_RECEIVED` (196) != `noc_reads_num_issued` (194).
The hardware received 2 extra response packets from the oversized reads.

---

## Why Adding `noc_async_read_barrier()` at the End Caused a Hang

The barrier spins on `NIU_MST_RD_RESP_RECEIVED == noc_reads_num_issued[noc]`.
Since `HW (196) > SW (194)`, and nothing will ever increment SW or decrement HW,
the `==` check is permanently false → infinite hang.

---

## Key Files

| File | Role |
|------|------|
| `ttnn/.../reader_mcast_sender_unary_sharded_ln.cpp` | BRISC kernel (NOC0) — contains the bug |
| `ttnn/.../sharded_layernorm_factory_helpers.cpp` | Host-side — sets compile-time args including `num_tiles_per_worker_bytes` |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h` | NOC API — `ncrisc_noc_fast_read`, counter logic |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` | `NOC_MAX_BURST_SIZE` definition |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/c_tensix_core.h` | `num_acks = size / NOC_MAX_BURST_SIZE + ...` formula |
| `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | High-level `noc_async_read` / `noc_async_read_one_packet` |
| `tt_metal/hw/firmware/src/tt-1xx/brisck.cc` | BRISC firmware — contains the watcher assert |

---

## Scope: Are Other Kernels Affected?

Any kernel that calls `noc_async_read_one_packet` with a size exceeding
`NOC_MAX_BURST_SIZE` will have the same issue. Search for patterns like:

```cpp
noc_async_read_one_packet(..., size_that_could_exceed_8192);
```

The `_pre_allgather` and `_post_allgather` variants of this kernel have similar
gather-read code and should be audited for the same bug.
