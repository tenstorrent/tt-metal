# AGMM Task 3 — Phase A DRAM-staged implementation blueprint

Guide for implementing the fused program factory + kernels. Scaffold (device op + prim + factory stub) is
committed (`e5e869e690c`); this is the design for `create()` and the kernels. Paths relative to repo root.

## Design (single program per device; regime_a compute engine intact)

**Buffers**
- in0 `[M, K_local]` DRAM interleaved bf16 (per-device shard); in1 `[K_global, N]` DRAM 8-bank width-shard
  (regime_a layout, unchanged); **NEW DRAM gather buffer `[M, K_global]`** interleaved bf16 (regime_a in0
  reader points here); output `[M, N]` (as regime_a).

**Core groups (fabric reserved from grid tail ≥104; regime_a compute in [16,104) — per host plan)**
1. regime_a compute: `8·Pk·Ns·Sm` cores (regime_a placement logic, replicated in the fused factory).
2. mux v2 cores: `num_links·(Ring?2:1)`, grid tail; dual-RISC `tt_fabric_mux_v2.cpp` via `add_fabric_mux_v2_to_program`.
3. fabric injector/worker cores: `num_links·num_workers_per_link`, grid tail (mux clients).

**Kernels**
- `dm_in0_injector.cpp` (NEW, injector cores): read local in0 shard once → write local gather slice (local NoC)
  → fabric-unicast per-tile into each remote device's gather buffer at its global-K offset → atomic-inc each
  remote per-chunk readiness sem → `close()`.
- `in0_ring_reduce_writer.cpp` (COPY of regime_a's into this op's device/kernels, `#ifdef FUSED_AGMM_GATHER`):
  read in0 from the gather buffer; gate remote K-blocks on readiness sem; local K-blocks immediate.
- `in1_reader.cpp`, `compute.cpp`: REUSED UNCHANGED (CreateKernel against regime_a's paths).

**CBs**: regime_a's `c_0..c_7` unchanged; injector has its own L1 payload buf (`C·kb·Mt·2048` B) + packet header.

**Semaphores**: per-chunk readiness sem (injector atomic-incs over fabric; in0 reader `noc_semaphore_wait_min`);
mux v2 client `{flow_control, teardown}` per injector via `append_client_connection_rt_args`; regime_a local
sems unchanged.

## in0-reader hook (the key change), file:line
`regime_a_matmul/device/kernels/in0_ring_reduce_writer.cpp`:
- `:73` accessor `TensorAccessor(in0_args, in0_addr, tile_bytes)`; `:55` `in0_addr`=RT arg0; `:40` `in0_args`=CT `TensorAccessorArgs<14>()`.
- `:228` the single in0 DRAM read `noc_async_read_page((m_start+m)*Kt + (k_start+l), in0, p)`; `:245` `cb_push_back(in0_cb,...)`.
- Override from the FUSED factory (no kernel arithmetic change): CT arg5 `Kt`→`K_global`; CT arg14 `TensorAccessorArgs`→gather buffer; RT arg0 `in0_addr`→gather base; RT arg4 `k_start`→band's GLOBAL K tile offset.
- Add readiness gate before `:228`/`:245` for remote K-blocks (`noc_semaphore_wait_min(ready_ptr, expected)`); local blocks (from `DevicePlan::local_k_blocks`) skip.

## Multi-device hop-distance swap (required or D>1 FATALs)
regime_a factory calls the unit-mesh overload `get_worker_noc_hop_distance(IDevice*,...)` at
`regime_a_matmul_program_factory.cpp:201,:231,:244,:295` → `TT_FATAL num_devices()==1` on a MeshDevice
(`tt_metal/impl/device/experimental/device.cpp:19`). The fused factory MUST use the multi-device overload
`get_worker_noc_hop_distance(MeshDevice*, MeshCoordinate, src, dst, NOC)` (`device.hpp:32-37`, `device.cpp:44-53`),
per participating device coordinate (harvesting varies per chip).

## mux v2 skeletons
Host: `FabricMuxV2Config(num_channels=workers, num_buffers_per_channel, ch_buf_bytes=align(hdr+packet), base_l1)`
→ `add_fabric_mux_v2_to_program(program, cfg, mux_core, src_node, dst_node, link_idx, RISCV_0_default)` →
per client `CreateSemaphore x2` + `cfg.append_client_connection_rt_args(mux_virtual_core, channel_id, {fc,td}, rt)`.
(refs: `test_basic_fabric_mux_v2.cpp:495-532,:599-615`; `fabric_mux_v2_config.cpp:63-77,:119-137,:274-292`.)
Device injector: `FabricMuxV2Sender<>::build_from_args(a)` → `open()` → per tile
`linear::experimental::fabric_unicast_noc_unicast_write(&sender, hdr, src_l1, size, {dst_noc}, num_hops)` →
`noc_async_writes_flushed()` → `fabric_unicast_noc_unicast_atomic_inc(&sender, hdr, {ready_sem_noc,1}, num_hops)`
→ `noc_async_write_barrier()` → `close()`. (`tt_fabric_mux_v2_sender.hpp:60-93,:131-148`; `linear/api.h:81-95,:259-270`.)
Phase A uses direct multi-hop unicast (source→each dest, num_hops=distance); no dedicated receiver kernel
(payload lands in dest DRAM, readiness in dest sem). Ring-relay variant would need a receiver (v2 test
`fabric_mux_v2_receiver.cpp` is the template) — defer.

## Bring-up order & gating
1. **`#ifdef AGMM_FULL_GATHER_BARRIER` diagnostic**: injector incs one barrier sem when its whole shard landed;
   in0 reader waits D-1 incs before first read. == all_gather_async→regime_a. Correctness reference / no-overlap A/B.
   Get PCC≥0.999 + clean watcher at D=2 first.
2. **Streaming (production)**: per-chunk readiness gate (above). Overlap: first matmul before full gather.
   Then D=4/8, linear + bidirectional ring, more chunks than slots, fresh+cached, watcher, overlap evidence.

## Risks / watcher pitfalls
- Payload BEFORE readiness on the SAME mux channel (flush writes, then atomic-inc). Different channel/link ⇒ no order guarantee.
- Readiness ≠ source lifetime; keep flush/barrier before reusing L1/transport slots (test more chunks than slots).
- Per-tile interleaved-DRAM fabric writes under-fill the 4KiB packet (2KiB tile) — measure in Task 4, group tiles if needed.
- Forgetting the multi-device hop-distance overload ⇒ build FATAL for D>1 (most likely first failure).
- Every injector that `open()`ed must `close()` or the v2 mux never tears down (hang).
- Keep fabric duties OFF the in1 BRISC (dedicated injector cores).
- L1: mux v2 reserves `base_l1..get_memory_map_end_address()`; validate against injector bufs + regime_a CBs.
- Program-cache: all fabric knobs feed CT args → in hashed operation_attributes; `override_runtime_arguments`
  must relocate in0/gather/output addresses on replay.

## Open architectural decision (before implementing create())
Reusing regime_a's compute-program build: the blueprint's approach is to **replicate** regime_a's factory
setup (core placement/CBs/compile+runtime args) inside the fused factory and `CreateKernel` against regime_a's
kernel files — self-contained, does NOT touch regime_a (respects "don't alter regime_a"). Alternative: refactor
regime_a's factory into a shared helper (as minimal_matmul did for AGMM) — less duplication but touches regime_a's
host factory. Recommend replicate-first for Task 3; consider extracting a helper later if duplication hurts.
