# g6_clone: the op in Tensix terms

This file is evidence, not a design. Work split, block extents, buffers, helper choice and
kernel structure are decided by the planner in op_design.md; where this file and op_design.md
disagree, op_design.md wins. Header citations were read at tt-metal 33bde179040e (branch
llk_helper_library, 2026-11-24); line numbers move, so re-check a signature before relying on it.

## Overview

| Field | Value |
|---|---|
| Classification | data_movement |
| Goal | Copy one DRAM-interleaved tensor into a new DRAM-interleaved tensor of the same shape, dtype and layout, every logical element bit for bit, as one device program. |
| Math | `out[i] = x[i]` bit for bit, new storage (math.md#definition) |
| References | `docs/explorations/g6_clone/DOSSIER.md` and the files it lists |

## Tensors

| Tensor | Logical shape | Dtype | Layout | Memory | Tile rows x tile columns |
|---|---|---|---|---|---|
| x | (d_0, ..., d_{R-1}), R in 1..4 | bfloat16, float32 (TILE); bfloat16, int32, uint32 (ROW_MAJOR) | TILE or ROW_MAJOR, as the production case has it | DRAM interleaved | TILE: prod(d_0..d_{R-3}) * ceil(H/32) x ceil(W/32); ROW_MAJOR: prod(d_0..d_{R-2}) pages of one row each |
| out | the shape of x | the dtype of x | the layout of x | DRAM interleaved | the same grid as x |

Padding: TILE tensors with H or W not a multiple of 32 ([1,1,1,62080], [1,5120], [1,1]) have
partial tiles. Input padding is garbage that must not reach a logical output element; output
padding is unspecified (`sources/local/contract.md:74`). ROW_MAJOR pages have an alignment gap
after each row in DRAM (64 B stride; `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:42`),
which is not data either.

## Geometry and counts

    python3 -B "$EXPLORATOR"/scripts/tile_budget.py tensor p1=16,12,128,128:float32 p2=8,12,128,128:float32 q13=1,12,128,128:float32 p3=1,512,5120:bfloat16 p4=1,1,32,62080:bfloat16 q8=1,1,1,62080:bfloat16 f5=1,2048,128:bfloat16 f6=1,256,3072:bfloat16 f7=1,1:float32 f8=1,1:bfloat16 f9=1,1,2048,128:bfloat16 f10=1,1,256,128:bfloat16 f11=1,1,2304,128:bfloat16 f12=1,256,5120:bfloat16 f13=1,5120:bfloat16 f14=1,8192,128:bfloat16 f16=1,1,8448,128:bfloat16
    python3 -B "$EXPLORATOR"/scripts/tile_budget.py tensor r3=16,3,2560:bfloat16 r7=8,3,2560:bfloat16 r14=1,3,2560:bfloat16 u16=16:uint32 i16=16:int32 u8=8:uint32 u32=32:uint32 u1=1:uint32 i1=1:int32 u4d=1,1,1,32:uint32

Input and output have the same figures; one row per production tensor ([1,1,8192,128] has the
figures of [1,8192,128]).

| Shape (name) | Tensor | Tiles | Bytes | Tile grid (rows x cols) | Note |
|---|---|---|---|---|---|
| p1 [16,12,128,128] float32 TILE | x, out | 3072 | 12582912 | 768 x 4, tall-narrow | tile_aligned; 4096 B per tile |
| p2 [8,12,128,128] float32 TILE | x, out | 1536 | 6291456 | 384 x 4 | tile_aligned |
| [1,12,128,128] float32 TILE | x, out | 192 | 786432 | 48 x 4 | tile_aligned |
| [1,1] float32 TILE | x, out | 1 | 4096 | 1 x 1, single tile | 1 logical element of 1024 |
| p3 [1,512,5120] bfloat16 TILE | x, out | 2560 | 5242880 | 16 x 160, short-wide | tile_aligned; 2048 B per tile |
| p4 [1,1,32,62080] bfloat16 TILE | x, out | 1940 | 3973120 | 1 x 1940, one tile row | tile_aligned |
| [1,1,1,62080] bfloat16 TILE | x, out | 1940 | 3973120 | 1 x 1940, one tile row | h_non_aligned: 124160 logical bytes, 1 row of 32 per tile |
| [1,5120] bfloat16 TILE | x, out | 160 | 327680 | 1 x 160 | h_non_aligned: 10240 logical bytes |
| [1,256,5120] bfloat16 TILE | x, out | 1280 | 2621440 | 8 x 160 | tile_aligned |
| [1,256,3072] bfloat16 TILE | x, out | 768 | 1572864 | 8 x 96 | tile_aligned |
| [1,2048,128] bfloat16 TILE | x, out | 256 | 524288 | 64 x 4 | tile_aligned |
| [1,8192,128] bfloat16 TILE | x, out | 1024 | 2097152 | 256 x 4 | tile_aligned |
| [1,1,2304,128] bfloat16 TILE | x, out | 288 | 589824 | 72 x 4 | tile_aligned |
| [1,1,8448,128] bfloat16 TILE | x, out | 1056 | 2162688 | 264 x 4 | tile_aligned |
| [1,1,256,128] bfloat16 TILE | x, out | 32 | 65536 | 8 x 4 | tile_aligned |
| [1,1] bfloat16 TILE | x, out | 1 | 2048 | 1 x 1 | 1 logical element |
| [16,3,2560] bfloat16 ROW_MAJOR | x, out | - | 245760 stored | 48 pages of 5120 B | page = one row; stride 5120 in DRAM |
| [8,3,2560] bfloat16 ROW_MAJOR | x, out | - | 122880 stored | 24 pages of 5120 B | |
| [1,3,2560] bfloat16 ROW_MAJOR | x, out | - | 15360 stored | 3 pages of 5120 B | |
| [32] uint32, [1,1,1,32] uint32 ROW_MAJOR | x, out | - | 128 stored | 1 page of 128 B | |
| [16] uint32 / int32 ROW_MAJOR | x, out | - | 64 stored | 1 page of 64 B | |
| [8] uint32 / int32 ROW_MAJOR | x, out | - | 64 stored | 1 page of 32 B | DRAM stride 64 B |
| [1] uint32 / int32 ROW_MAJOR | x, out | - | 64 stored | 1 page of 4 B | DRAM stride 64 B |

    python3 -B "$EXPLORATOR"/scripts/tile_budget.py counts --read x=16,12,128,128:float32 --write out=16,12,128,128:float32

| Shape (name) | Multiply-adds | Vector-engine tile passes (per function) | Bytes read | Bytes written | Compute per byte |
|---|---|---|---|---|---|
| p1 [16,12,128,128] float32 | 0 | none | 12582912 | 12582912 | 0 core-cycles per byte: below the machine balance of 0.290 at an assumed 110 cores (no engine work; dtype float32) |
| p2 [8,12,128,128] float32 | 0 | none | 6291456 | 6291456 | 0: below the balance, as p1 |
| p3 [1,512,5120] bfloat16 | 0 | none | 5242880 | 5242880 | 0: below the balance, as p1 |
| p4 [1,1,32,62080] bfloat16 | 0 | none | 3973120 | 3973120 | 0: below the balance, as p1 |
| [16,3,2560] bfloat16 ROW_MAJOR | 0 | none | 245760 | 245760 | 0: below the balance; 48 transfers of 5120 B each way if one per page |
| [1] uint32 ROW_MAJOR (smallest) | 0 | none | 4 logical, 1 page | 4 logical, 1 page | 0; the call is two page transfers in all |

## Axes: evidence for the blocking decision

| Axis | Extent in tiles (production) | Evidence | What a split along it would have to carry |
|---|---|---|---|
| leading dims folded into tile rows (TILE) | 1 to 768 tile rows | no result spans it: out[i] reads x[i] only (math.md, axes table) | nothing |
| W in tile columns (TILE) | 1 to 1940 tile columns | no result spans it | nothing |
| rows (ROW_MAJOR pages) | 1 to 48 pages | no result spans it | nothing |
| elements inside a row (ROW_MAJOR) | 1 to 2560 elements, 4 B to 5120 B per page | no result spans it | nothing; a part of a page is a sub-page transfer, with the DRAM alignment of R5 |

Operands that do not vary along an axis (candidates for reuse): none; x is read once and out
written once, each element by exactly one transfer in an ideal design.

## Compute phases

| # | Phase | Math | Operands and valid regions | Result | Arrives as |
|---|---|---|---|---|---|
| 1 | move | out[i] = x[i], bit for bit (M2) | x: TILE pages (full tiles; partial tiles for the three non-aligned shapes) or ROW_MAJOR pages | out pages of the same layout; needed only until written | tiles or row-major rows, as the case has them |

No arithmetic phase exists. A phase through the unpacker, an engine and the packer would be a
conversion, not a move (risks.md R1).

## Candidate primitives

| Phase | Candidate | Declared at | Restrictions that matter here | Status |
|---|---|---|---|---|
| 1 | `noc_async_read_page` with an address generator | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1243` | reads one page whose size the generator gives; an optional size overrides it | covered |
| 1 | `noc_async_write_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418` | one page per call | covered |
| 1 | `noc_async_read`, `noc_async_write` (address and size) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876` | one transfer of up to `NOC_MAX_BURST_SIZE` without splitting, 256 NoC words (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:289-290`); a transfer cannot span two interleaved pages taken in page order, which lie in different banks | covered |
| 1 | `TensorAccessor`, its `num_contiguous_pages` and `contiguous_page_stride` | `tt_metal/hw/inc/api/dataflow/noc.h:13`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h:181-192` | pages p, p + stride, p + 2 stride ... that lie contiguously in one bank | covered |
| 1 | `noc_async_read_barrier`, `noc_async_write_barrier`, `noc_async_writes_flushed` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980` | every write must have completed before the kernel returns (BB-03 of the spec's rules) | covered |

Tile geometry: 32 x 32 tiles of four 16 x 16 faces (`tt_metal/hw/inc/api/tile_constants.h:18-22`);
tile bytes per format: `tt_metal/api/tt-metalium/tt_backend_api_types.hpp:122`. DRAM banks:
eight on Blackhole (`tt_metal/soc_descriptors/blackhole_140_arch.yaml:11-21`); page p of an
interleaved tensor lies in bank p mod the bank count (INFERENCE from the round-robin placement).

## Gaps

- None. A bit-exact copy of every production dtype (bfloat16, float32, int32, uint32) needs only
  NoC transfers between DRAM and L1; the transfers move bytes and know no dtype. Searched: the
  dataflow scope of the checkout for read, write, page and barrier calls (`primitives.py --scope
  dataflow --grep 'noc_async_read$|noc_async_write$|noc_async_read_page|noc_async_write_page|barrier|flushed'`).

## Mechanism caps implied by the math

| Cap | On which extent | Because |
|---|---|---|
| none from the math | - | no element depends on another; the only limits are of transfer size and alignment (R5), which are mechanism, not math |

## Candidate regimes

C is the number of pieces a split makes (a free choice of the planner); P the pages of x.

| Regime | Predicate | Resident per unit of work | DRAM crossings per tensor | Cross-core traffic |
|---|---|---|---|---|
| page staging: each piece reads its pages into L1 and writes them out | always | a few pages: one TILE page is 2048 B (bfloat16) or 4096 B (float32); one ROW_MAJOR page at most 5120 B in production | x: 1 read; out: 1 write; P read and P write transfers in all, split over C pieces | none |
| bank runs: pages of one bank that lie contiguously moved in fewer, longer transfers | pages taken in bank order and page size equal to its stride (TILE: always; ROW_MAJOR: when the row is a multiple of 64 B, true of [16,3,2560] and [16] uint32, false of [8] and [1]) | the bytes of one run | x: 1 read; out: 1 write; fewer transfers than P | none |
| logical rows only, for TILE tensors with H < 32 ([1,1,1,62080], [1,5120], [1,1]) | H mod 32 != 0 and the content of output padding left unspecified (`sources/local/contract.md:74`) | the logical rows of a tile: 1 row of 32 in these shapes, stored as two 32 B pieces in faces 0 and 1 (`tt_metal/hw/inc/api/tile_constants.h:21-22`) | x: the logical bytes only (124160 B of 3973120 B for [1,1,1,62080]); out: the same | none |

The resident set of the first regime at two float32 tiles is 8192 B (`tile_budget.py resident
x=2:float32 --l1-bytes 1347776`; the fit against the model's free L1 is in risks.md).

## Matching catalog entries

- Not consulted: the examples catalog lies under a tree that this exploration was told not to
  open. No entry is named.

## Hazards specific to this mapping

- An engine path instead of a byte move changes bits (R1).
- Sub-page or odd-sized ROW_MAJOR transfers against the DRAM alignment (R5).
- Tiny tensors are one or two pages: the call is per-call overhead whatever the split (R6).
- A one-tile-row tensor of 1940 tiles split only by tile rows lands on one core (R4).
