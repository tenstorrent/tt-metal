# MoE Dispatch Kernel

## Overview

Fused MoE token dispatch + expert matmul on a [8,4] TT Galaxy mesh (32 Wormhole chips). Each device holds a shard of tokens sorted by expert ID. The kernel routes each token to the device that owns the target expert, performs `token @ W_up` matmul, and writes results to DRAM.

## Mesh Layout

- **[8,4] mesh**: 8 rows × 4 columns
- **Experts sharded along axis 1 (columns)**: column 0 owns experts 0-7, column 1 owns experts 8-15, etc. (`E_local = E / EP = 32 / 4 = 8`)
- **Each row** of 4 devices exchanges tokens with each other along axis 1
- **Each device** has its own token shard (portion of the sequence), sorted by expert ID, with per-expert counts and offsets

## Data Flow

**Input per device:**
- `sorted_hidden [1, N_padded, D]` — this device's tokens sorted by expert ID
- `w_up [E_local, D, ffn]` — weight matrices for this device's local experts
- `expert_counts[E]` — how many tile-rows this device has for each expert
- `expert_offsets[E]` — where each expert's tokens start in sorted_hidden

**Per device, the sender kernel does:**

```
for each expert e = 0..E-1:
    owner = e / E_local          # which column owns this expert
    n_rows = expert_counts[e]    # tile-rows this device has for expert e

    wait for go_sem >= turn      # EP serialization (device 0 goes first)

    for each tile-row r:
        read tile-row from sorted_hidden DRAM
        if owner == me:
            write to local dispatch_buf DRAM at remapped offset
            increment tiles_ready_sem on receiver
        else:
            fabric unicast write to owner's dispatch_buf DRAM
            + fused atomic increment of owner's tiles_ready_sem

    signal next EP device via go_sem (1 hop forward)
```

**Per device, the receiver reader kernel does:**

```
for each local expert e = 0..E_local-1:
    for each source device d = 0..EP-1:
        for each tile-row r = 0..counts[d][e]-1:
            wait for tiles_ready_sem >= tiles_consumed
            read tile-row from dispatch_buf DRAM into cb_in
            push weight tiles to cb_w
            # compute kernel picks up immediately
```

**Compute kernel:** Standard blocked matmul (`tile @ W_up`) consuming from `cb_in` + `cb_w`, producing to `cb_out`. Unchanged from the proven local-only version (PCC 0.9999).

**Writer kernel:** Writes matmul results from `cb_out` to output DRAM tensor.

## Dispatch Buffer Layout

The dispatch_buf on each owner device aggregates tokens from ALL source devices, laid out expert-major, device-minor:

```
Expert 0: [dev0_rows | dev1_rows | dev2_rows | dev3_rows]
Expert 1: [dev0_rows | dev1_rows | dev2_rows | dev3_rows]
...
Expert E_local-1: [...]
```

Each sender knows its write offset via `expert_dst_row[e]`, computed on the host as:

```
dst_row[e] = expert_base_on_owner[e] + sum(counts[0..my_dev-1][e])
```

## EP Serialization (go_sem)

Within each row of 4 devices, a `go_sem` semaphore serializes senders per expert:
- Device 0 (column 0) has `go_sem` initialized to `E` (all turns pre-granted)
- Other devices start with `go_sem = 0`
- After processing expert `e`, device `d` increments `go_sem` on device `d+1` via fabric atomic inc (1 hop forward)
- Device `d+1` waits for `go_sem >= turn` before sending expert `e`

This ensures only one device writes to the dispatch_buf at a time, preventing write conflicts and fabric congestion.

## Streaming Semaphore (tiles_ready_sem)

Instead of waiting for all senders to finish (full barrier), the receiver processes tile-rows as they arrive:
- Sender increments `tiles_ready_sem` on the receiver after each tile-row write
- Receiver waits for `tiles_ready_sem >= N` before reading the Nth tile-row
- Compute runs continuously, consuming tiles as the receiver pushes them

For local writes: `noc_semaphore_inc` on the receiver core.
For remote writes: `to_noc_fused_unicast_write_atomic_inc` — a single fabric packet that writes the tile data AND increments the semaphore atomically.

## Fabric Routing

- `FabricConnectionManager` with forward/backward connections along `cluster_axis`
- `to_chip_unicast(num_hops)` sets the hop count for multi-hop routing
- `num_hops = |owner - my_device_index|`, forward if owner > me, backward otherwise
- Host calls `append_fabric_connection_rt_args(src_id, dst_id, link=0, program, core, rt_args)`


## Known Issue

Multiple TP rows (8 rows in the [8,4] mesh) share the same physical fabric links along axis 1. When all 8 rows send simultaneously, only ~2 complete — the rest hang on `wait_for_empty_write_slot`. Needs TP row serialization (requires fabric connections along axis 0, which are not set up) or testing on a [1,4] mesh.

## Files

- `device/kernels/dataflow/sender.cpp` — reads sorted_hidden, writes to dispatch_buf (local NoC or fabric), signals receiver
- `device/kernels/dataflow/receiver_reader.cpp` — waits on tiles_ready_sem, reads dispatch_buf into cb_in, feeds weights
- `device/kernels/compute/expert_matmul.cpp` — blocked matmul consuming cb_in + cb_w
- `device/kernels/dataflow/receiver_writer.cpp` — writes matmul output to DRAM
- `device/moe_dispatch_program_factory.cpp` — host-side program creation, offset computation, fabric setup
- `device/moe_dispatch_device_operation.cpp` — output tensor sizing, validation
- `tests/python/test_moe_streaming.py` — end-to-end test with reference computation
