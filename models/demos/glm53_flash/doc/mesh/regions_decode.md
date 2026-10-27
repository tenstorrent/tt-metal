# Regions-sourced streaming decode: one expert copy for prefill and decode (2026-10-27, device-free)

Branch `arg/glm53_regions_stream` (from `arg/glm53_mesh` d4c822412c4). Knob `GLM53_REGIONS_DECODE=stream` (default
`off`). Code: `custom_ops/expert_stream/regions.py` (planner), `custom_ops/expert_stream/kernels/*` (`REGIONS_SRC`
variants), `custom_ops/expert_stream/op.py` (`RegionsSource`, `regions_src=True`), `tt/moe.py`, `tt/mesh_model.py`.
Nothing here was run on a device (the Galaxy is owned by the scheduler): the host tests pass, the device tests are
written and byte-compiled, the kernels are unverified under the JIT.

## 1. The finding

8x4 Blackhole Galaxy, 45 layers, 4 users per row, 128-token prompts, traced decode step:

| max_seq_len | step | expert path at decode |
|---|---|---|
| 256 | 70.0 ms | stacked stream copy (`GLM53_REGIONS_KEEP_STACKED=1`) |
| 256 K | 70.9 ms | stacked stream copy |
| 512 K | 151.4 ms | regions kernel (the stacked copy does not fit beside the caches) |
| 1 M | 152.2 ms (139 at 1 user/row) | regions kernel |

Not DRAM fullness, not placement, not the sparse-attention capacity loops (bounding them gained 3.6 ms). It is the
expert path: `tt/mesh_model.py` keeps the second, DRAM-bank-sharded "stacked" expert copy for the streaming decode op
only when `MemoryPlan.fits(max_batch, max_seq_len)` with `keep_stacked=True`; at >= 512 K x 4 users and at 128 K x 32
users it does not fit, `tt/moe.py` `_build_dispatch_path` deallocates `w13` / `w2`, and decode falls back to
`_forward_regions` (the d_p prefill kernel over 72 regions of <= 32 rows plus ~20 glue ops per layer): +80 ms per
step (42 MoE layers x ~1.9 ms) and different numerics. The served 1 M profile (32 users x 1 M, the product target)
decodes at 6.6 tok/s per user instead of ~14.

## 2. Byte model (per chip, 8x4 hybrid layout, bfp8)

| item | bytes |
|---|---|
| one expert slice: gate\|up `[4096, 512]` (128 x 16 tiles x 1088 B) + down `[256, 4096]` (8 x 128 x 1088) | 2.228 + 1.114 = 3.342 MB |
| one MoE layer (72 experts) | 240.6 MB |
| the stacked copy (42 MoE layers) = `regions.one_copy_bytes(72, 4096, 256, 4096, 42)` | 10.11 GB = **9.41 GiB** |
| what a decode step streams per layer at 28 active experts (4 users/row) | 93.6 MB = 312 us at 300 GB/s (341 us measured per block, sweep doc section 9) |

The bytes a step reads are the same under any expert layout (the active slices once). What differs is where they
sit: the stream op needs them bank-local and K-contiguous so that a core next to a bank streams whole columns in
8.7 KB packets; the regions kernel wants one DRAM-interleaved tensor per expert and projection. Today the model
holds both forms when they fit (9.41 GiB extra) and only the regions form otherwise.

## 3. The regions lists already have the stream layout

A DRAM-interleaved buffer places page `p` in bank `p % banks` at offset `(p / banks) * page_bytes` (the allocator's
`Buffer::page_address` and the device address generator `get_bank_offset_index` / `get_bank_index`; the TensorAccessor
and the stream reader's `get_noc_addr_from_bank_id<true>` both add the same `bank_to_dram_offset`). A TILE tensor
numbers its tiles row-major: tile `(k, n)` of a `[K, I_local]` tensor with `nt = I_local / 32` columns is page
`k * nt + n`. With `nt % banks == 0`:

* tile `(k, n)` sits in bank `n % banks` for every `k`: a tile COLUMN lives entirely in one bank;
* its K stride is `ppb = nt / banks` tiles; at `ppb = 1` (the 8x4: `nt = 8` columns on 8 banks) the column is
  contiguous: `kt` tiles at `k * 1088` from the buffer address -- exactly the column `gate_up_stream_layout` builds
  for the stacked copy (bank `b` = gate column `b`, then up column `b`, K-contiguous, 16 packets of 8 tiles).

The down tensor `[I_local, H]` (`ht = H / 32 = 128` columns): tile `(i, h)` is page `i * ht + h`, bank `h % banks`,
offset `(i * C + h / banks) * 1088` with `C = ht / banks = 16` columns per bank. Bank `b` holds the output columns
`h = b + 8 m`; the `cols` columns of one core are contiguous per K tile (8 tiles = 8704 B, one packet), K tiles 16
tiles apart: the same 8 packets of 8704 B per expert and core as the stacked block (`down_stream_layout`: bank `b`
= columns `[16 b, 16 b + 16)`, column-major), only k-major in the CB instead of column-major.

So the stacked copy duplicates bytes that already sit in the right banks in the right order; the only differences
are (1) a base address per expert and projection (216 buffers instead of one slab per projection), (2) the down
block's tile order in L1, (3) which global column a (bank, q) pair maps to (`b + 8 q` instead of `b * ppb + q`).
`tests/test_regions_decode_host.py` checks the kernels' address walk against an independent page map at the three
per-chip widths and both bank counts.

## 4. The three options

| option | verdict |
|---|---|
| (A) a reader variant that walks the regions tensors by (bank, offset) | **chosen**: zero extra DRAM, the same packets at the 8x4 shapes, bit-identical results, 10 kernels gain a preprocessor-gated branch, no new tensors |
| (B) one layout serving both: keep the stacked tensor and build the prefill's per-expert views from it | not available: a ttnn Tensor owns one Buffer (`ttnn.slice` copies; there is no byte-offset view), the regions kernel (`unified_routed_expert_ffn`) takes per-expert tensors (interleaved or DRAM ND-sharded with tile-row shards) and its down tiles are k-major where the stacked down layout is column-major -- a kernel change on one side regardless, plus a view mechanism that does not exist |
| (C) a reduced-precision decode copy (bfp4 gate\|up: 4.7 GiB) or the dispatch/all-to-all path | the headroom at 1 M x 4 users is ~2 GiB after the caches and 32 x 1 M has less; it changes the numerics; (A) costs nothing |

## 5. Variant (A): `GLM53_REGIONS_DECODE=stream`

**Kernels** (`REGIONS_SRC` define; the default programs are unchanged, every variant is `#ifdef`-gated):

| kernel | change |
|---|---|
| `reader_gate_up.cpp`, `reader_gate_up_dispatch.cpp` | per expert `gate_addr = common_arg[e]`, `up_addr = common_arg[E + e]`; column `q` = `kt` tiles from `addr + q * tile_bytes` at `src_step = page_bytes` (ppb = 1: whole-column packets as today) or `ppb * tile_bytes` (single-tile packets); NOC base `get_noc_addr_from_bank_id<true>(bank, 0)`; the pair slot layout (gate tiles then up tiles) and the compute are unchanged |
| `writer_gate_up.cpp`, `writer_gate_up_dispatch.cpp` | the act tile column of pair `q` is `bank + banks * q` (`regions_src.hpp::regions_act_col`, `banks = i_tiles / ppb`) |
| `reader_down.cpp`, `reader_down_dispatch.cpp` | per expert `addr = common_arg[e]`; `seg_tiles` packets of `cols` tiles at `addr + (i * C + c0) * tile_bytes` (`C = bank_expert_stride / seg_bytes`); k-major in the CB |
| `compute_down.cpp`, `compute_down_dispatch.cpp` | `w[k * cols + c]` instead of `w[c * seg_tiles + k]`: the same products per output column in the same k order |
| `writer_down.cpp`, `writer_down_dispatch.cpp` | output column `bank + banks * (r * cols + c)` from today's `col0 = bank * C + r * cols` (`regions_out_col`); the constants `[h_tiles, banks]` / `[banks]` are compile-time args appended AFTER the accessor args (`CT_TAIL`), so no existing index moves |

**Op** (`op.py`): `RegionsSource(gate, up, down, banks)` validates the lists (TILE, DRAM interleaved, shapes, dtypes)
and holds the address lists; `regions_src=True` on `ExpertGateUpStream` / `ExpertDownStream` / `ExpertStreamFused`
/ `ExpertStream` / `ExpertGateUpDispatch` / `ExpertDownDispatch` / `ExpertDispatch` packs the addresses as the
readers' **common runtime args** (2 E = 144 words for gate|up, E = 72 for down; the limit is 341,
`kernel_types.hpp`), appends the define (and the compute define on the down ops), sizes the packets from the
planner and passes the lists in the io tensors (today's order kept, the lists first). On a program-cache hit
`apply_descriptor_runtime_args` (tt_metal `program_descriptors.cpp`) rewrites the common args in place, so every
layer's address list is applied per call like today's per-core `w13` address (the hash covers the arg counts only).
The pair-unit kernels (`GLM53_EXPERTS_32ROW=units`, ppb > 1 only) are stacked-only and refused.

**Model** (`tt/moe.py`): the knob is read per call (`regions_decode_mode`, like the glue knobs: a function of the
env and the static shape, trace-safe). `_stream_serves` lets the stream family serve <= 32 (block) and 33..256
(dispatch) gathered tokens from the regions lists when `stream` is on; `_stream_source` / `_block_dispatch` pick the
regions-sourced twins (`ExpertStream(regions_src=True)`, `ExpertDispatch(..., regions_src=True)`, built on first use,
placement only). `regions` forces the regions kernel for every shape when the lists exist (the A/B reference). The
**one-copy configuration** (`stream` + regions prefill + no keep_stacked, `one_copy_config`) never uploads the
stacked stream tensors (`w13 = w2 = None` from construction; `cache_complete` requires the block tensorbins only)
and refuses a device whose banks are not bank-local for the slice. `tt/mesh_model.py`'s auto rule
(`resolve_keep_stacked`) never keeps a second copy under `stream`. `paths_used` / `last_path` record the path taken
(host bookkeeping for the tests).

**Numerics**: the same BF16 x BFP8 products at HiFi2 with fp32 accumulation, in the same K order per column and the
same expert order, the activations written to the same global act columns, the outputs to the same global output
columns: bit-identical to the stacked ops. The stacked and the regions tensors hold the same bfp8 tiles (block-float
exponents are per 16-element tile row, the tiles are only placed differently), so the inputs are bit-identical too.

**Planner** (`regions.py`): `plan_regions_stream(K, I, H, banks, w13, w2, down_readers, packet_tiles)` ->
`RegionsStreamPlan` (bank locality, ppb, C, cols, packet sizes / counts, src steps, the kernel-mirror address walks,
the (bank, column) maps); `regions_stream_feasible`; `one_copy_bytes`; `active_read_bytes`;
`stacked_packets_per_expert`. Bank locality needs `I_local/32 % banks == 0` and `H/32 % banks == 0`: true on the
8-bank Blackhole at 256 / 1024 / 2048, false on the 12-bank Wormhole at these widths (the stacked layout refuses the
same shapes; the planner returns the reason and the model keeps today's paths).

**Packet model** (NOC packets per active expert and core):

| shape | gate\|up regions / stacked | down regions / stacked | note |
|---|---|---|---|
| 8x4: I_local 256, ppb 1 | 32 / 32 (16 x 8.7 KB per column) | 8 / 8 (8.7 KB) | byte-for-byte the stacked stream |
| (2,4) block: I_local 1024, ppb 4 | 1024 / 128 (single 1088 B tiles: a column's K tiles are 4 tiles apart) | 32 / 16 (4.35 KB vs 8.7 KB) | the test platform only; ~0.1 ms per layer of extra issue cost at ~2.3 experts per core |
| one chip: I_local 2048, ppb 8 | 2048 / 256 | 64 / 32 | idem |

## 6. Expected step on the 8x4

The expert path was the only difference between the 256 K and the >= 512 K points (section 1), so with the
regions-sourced stream the 1 M x 4 users step returns to the 256 K figure: **152 -> ~71 ms** (the stream op's 341 us
per block x 42 layers against ~2.3 ms per layer on the regions kernel; the same bytes from the same banks, the only
new per-launch work is reading 2 E / E common args instead of one address). Served 1 M decode: 6.6 -> ~14 tok/s per
user (the served replay 148 ms -> ~68 ms + the ~45 ms of reads / position uploads the serving profile attributes to
the adapter). At 32 users per row x 1 M (256 gathered tokens, the product target) the regions-sourced DISPATCH ops
replace the regions kernel: the README's 128 K x 32 users envelope point (199 ms on the regions kernel) should land
near the other 256-slot paths (dense one-pass 115.5 ms, EP32 123 ms; the dispatch ops measured 779 us per layer at
58 active experts on one chip) -- an estimate until the 8x4 run, since the regions kernel's time at 256 gathered rows
was never profiled in isolation. The 4 users per row x 1 M step after this change is bounded below by the
256 K step (70.9 ms); the bounded sparse decode branch (3.6 ms) composes with it.

## 7. Tests

| test | what | status |
|---|---|---|
| `tests/test_regions_decode_host.py` (`--noconftest`, no ttnn) | the planner on 8 / 12 banks at 256 / 1024 / 2048, the kernels' address walk vs an independent page map (gate\|up bfp8 / bfp4, down), the (bank, column) maps as permutations, the packet model vs the stacked layout, the byte model (9.41 GiB), the common-arg limit vs `kernel_types.hpp`, the knob, the keep-stacked rule, the kernel ABI (defines, common-arg indexing, the appended tail, the k-major index, exit barriers), the op / model plumbing | 23 passed |
| `tests/test_expert_e1_host.py`, `tests/test_hybrid_glue_host.py`, `tests/test_moe_roofline_host.py`, `tests/test_serving_glm53_host.py` | the existing source pins (updated to the new runtime-arg lines; `_stream_serves` cases for the three modes) | 30 + 20 + 22 + 15 passed |
| `tests/custom_ops/test_expert_stream.py::test_expert_stream_regions_src` (one chip) | block ops at 32 tokens and dispatch ops at 64, I_local 256 and 1024: the regions-sourced ops vs the stacked ops on the same inputs, `torch.equal` on act / out, and the golden bar | written, not run |
| `tests/test_regions_decode_mesh.py` ((2,4) block) | model A = the one-copy configuration (no stacked copy; prefill in 1024-token chunks: the 128-token prompt through the regions-sourced dispatch ops), passes S1 / R1 / S2 / R2 (stream eager, regions-kernel eager, both traced); model B = regions prefill + the stacked copy kept, T1 / T2; asserts S1 == S2, R1 == R2, T1 == T2 and **S1 == T1 == T2 bit for bit** (greedy ids and logits), logs R vs S (PCC, agreement; 0.95 bar) and the replay tok/s of the three decodes | written, not run |

Block command (the rows-4-5 (2,4) block env of `tests/test_decode_packed_inputs_mesh.py`):

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 \
        pytest models/demos/glm53_flash/tests/test_regions_decode_mesh.py -s

(the test sets `GLM53_PREFILL_EXPERTS=regions` and the `GLM53_REGIONS_KEEP_STACKED` / `GLM53_REGIONS_DECODE` knobs per
model itself; `GLM53_REGIONS_STEPS` = decode steps per pass, 12). The op-level A/B on one chip:

    TT_VISIBLE_DEVICES=<chip> pytest models/demos/glm53_flash/tests/custom_ops/test_expert_stream.py -k regions_src -s

The 8x4 point of the finding: the e2e harness at `max_seq_len` 1 M, 4 users per row, with `GLM53_REGIONS_DECODE=stream`
(the auto prefill mode then resolves to regions without the stacked copy, `TtRoutedExperts.one_copy`).

## 8. Open items and risks

* Unverified on device: the kernels compile under `-Wall -Werror` (the unused `w_addr` is voided under the define,
  unused `constexpr` values follow the existing precedent `unit_pairs_unused`), the common-arg indexing, the k-major
  compute index. The op-level A/B is the first thing to run.
* The (2,4) block pays single-tile gate|up packets (ppb = 4); the product shape (8x4, ppb = 1) does not. A k-major
  multi-column compute for ppb > 1 is possible (ppb columns accumulated at once, 2 ppb fp32 DST tiles) but not needed.
* Wormhole (12 banks): the planner refuses at these widths, like the stacked layout; the regions kernel path stays.
* `GLM53_EXPERTS_32ROW=units` is stacked-only (refused under `stream`).
* The one-copy configuration never loads the `_stream` tensorbins (fewer files, ~2 min less build time per 45
  layers); a warm cache with only the stream files and no block files cannot build it (`cache_complete` says so).
