# fabric_all_gather: proper TTNN op + GLM e2e

Goal: `ttnn.experimental.fabric_all_gather` — a C++ TTNN device operation with the exact Python contract of
`ttnn.experimental.high_bw_all_gather` (drop-in), implemented with the fabric_all_gather algorithm; GLM prefill
e2e runs on it and its CI jobs pass.

## Contract (same as high_bw_all_gather)
input DRAM interleaved (TILE or ROW_MAJOR, any dtype) · preallocated interleaved-DRAM output · `dim` any ·
`cluster_axis` 0 / 1 / None (None = whole mesh, output in row-major chip order) · `num_links` · `subdevice_id` +
`sub_core_grids` · `input_batch_index` / `gathered_dim_size` (runtime, hash-excluded) · trace-safe
`input_batch_index_tensor` (+ `batch_slot_num_layers`, `batch_slot_layer_idx`) and `gathered_prefix_tensor`
(+ `gathered_slab_global`) · caller-owned `ready_semaphore` + `data_valid_semaphore` (left at 0 after every call).

## Algorithm (from ttnn/ttnn/operations/examples/fabric_all_gather)
- One link worker per (ring, ring direction, link): reader (NCRISC, NoC0) + sender (BRISC, NoC1), placed directly
  below its link's Ethernet core (`get_forwarding_eth_core` + physical-column match).
- One copy core per link for the chip's own shard.
- Chunk = run of up to `payload / page` pages consecutive in one DRAM bank *within one stripe*; banks owned per
  (ring, link) with stride `rings x links`.
- Relays wait on an arrival counter, incremented by every 8th packet (fused write + inc); ready fence between calls.
- Rings: axis line/ring, balanced (far shard split in halves); full mesh: two edge-disjoint Hamiltonian cycles on
  a torus with both sides >= 3 (else snake).

## Page model (general dim, slots, extent)
Shard in pages = [A stripes][B pages per stripe] where B = pages from the gather dim inward (active extent B_act
<= B_max) and A = product of the page-dims outside it. Output page of (rank g, stripe a, page b):
`a * G * B_max + g * B_max + b`. Input page: `slot_base + a * B_max + b`. A chunk steps b by the bank count and
never crosses a stripe, so both source and destination runs are contiguous in one bank.
Counts (chunks per shard, per port) are closed-form so the kernels can derive them on device from metadata.

## Steps
1. [ ] C++ op skeleton: types, device op (validation / hash / output spec cloned from high_bw_all_gather), nanobind.
2. [x] Host planner: groups, rings, dual cycles, schedules, links, placement, semaphores, runtime args, cache-hit
       override (addresses, semaphores, slot, extent).
3. [x] Kernels: shared walk header (host + device), reader (slot / extent / metadata), sender (fence, cadence),
       copy.
4. [ ] Tests: port test_fabric_all_gather to the ttnn op; run high_bw_all_gather's accuracy suite through the new
       op (slots, extent, metadata, traced, overlap).
5. [ ] GLM: one switch in the model (`DS_ALL_GATHER_OP`), default to fabric_all_gather; local smoke.
6. [ ] CI: dispatch the GLM e2e prefill jobs on the branch; fix until green; perf vs baseline.

## GLM-5.3 e2e call sites (origin/main, 8x4 Galaxy, FABRIC_2D_TORUS_XY, payload 6144 B, l1_small 1216 B, 2 links)
408 calls per chunk forward (78 layers):
| site | input / device | dim / axis | notes |
|---|---|---|---|
| A,B rms-norm stats (tt_distributed_rms_norm.py:307) | [1,1,640,32] bf16 TILE | 3 / 1 | width gather: A=20 stripes of 1 page |
| C q_a latent (mla.py:1113) | [1,1,640,512] bf16 TILE | 3 / 1 | A=20, B=16 |
| D indexer k (indexer.py:498) | [1,1,640,32] bf16 TILE | 3 / 1 | 21 layers |
| E kv stem (mla.py:1310) | [1,1,640,576] bf16 TILE | 1 / 1 | contiguous |
| F sparse KV prefix (mla.py:2253) | [78,1,S/32,576] bf16 ROW_MAJOR, DRAM **ND-sharded** | 2 / None (32 ranks) | slot + extent (scalar or metadata, traced); 21 layers add subdevice 1 = cores (8,0)-(11,9) + external L1_SMALL semaphores |
Traced jobs capture everything with SubDeviceTraceController; metadata tensors rewritten between replays.

## Status 2026-09-30
- Builds on main. tt-emule (32-chip BH Galaxy, 8x4 torus): 36/36 bit-exact vs torch reference (axis 0/1 width, dim-1,
  row gathers; full-mesh sparse-KV prefix incl. ND-sharded input, slots, growing extent; TORUS_XY dual cycles, FABRIC_2D
  snake; 6 KB and 14 KB payloads). Trace + sub-device tests need hardware (slow dispatch).
- Two deadlocks found by the emulator and fixed (both latent in the Python example too): the reader must push its read
  chunks before blocking on a relay, and every shard's last chunk must carry an increment (else a relay's wait can
  depend on a later shard -> cycle when shards are shorter than 8 chunks).
