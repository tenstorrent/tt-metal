# Shared-MLP DRAM-sharded decode candidate

This document records the CPU/source preparation before hardware handoff. At
that point no runtime or runner patch had been applied and no device was used;
the CPU check did not import TTNN. Subsequent parent-authorized hardware
results are recorded in [shared_dram_grouped_results.md](shared_dram_grouped_results.md).
The candidate addresses the material dense-matmul advice in
[optimization_advice_audit.md](optimization_advice_audit.md), not an accepted
optimization on the basis of source evidence alone.

## Artifacts and scope

- [shared_dram.patch](shared_dram.patch): optional `_DramSharedMLP` and
  `shared_dram=False` factory control.
- [shared_dram_runner.patch](shared_dram_runner.patch): `--shared-dram` flag,
  validation, forwarding, result metadata and capacity-reservation adjustment.
- [shared_dram_provenance.json](shared_dram_provenance.json): exact base and
  candidate SHA256 values for both files.
- [shared_dram_cpu_checks.json](shared_dram_cpu_checks.json): actual-width
  packing, zero-padding, integer geometry and added-weight-byte checks.

Runtime base is `be27d72163e05ac9ff294624e2475412ebc2bf9ba5f5a1afaa022816cac32426`;
candidate is `e9b7dd05c6f1dd3201c869756860dbc9e3050bd7f2eaa2af68c584e68b233d20`.
Runner base is `bb52268f962c9c11ec38782bc9f545a225efb99d5086bb51b115f3a1b8014bdb`.
The base contains the parent's AGMM, shared-geometry and grouped-reduction
experiments plus current runner guards; the patches preserve those interfaces.
The subclass accepts `reduce_output` so grouped reduction can remain under
caller control. Recheck applicability after concurrent edits. Both factory
and runner reject DRAM combined with nonzero `shared_geometry`; compare these
backends as separate options with their own explicit programs.

The option requires `optimized_shared=True`. The existing `_SharedMLP` code
is left intact. The subclass uses the inherited BF16 projections whenever
logical rows exceed one, and replaces only the two single-token shared decode
projections. Active experts, routing, cache, attention, collective placement,
branch norms and the residual contract remain unchanged. Larger logical batch
uses the inherited per-slot decode loop, so each actual projection still has
logical M1 and physical M32.

Weights keep BFP8 for sliding and BFP4 for full. Inputs and outputs stay BF16;
compute stays LoFi, approximate math off, FP32 destination accumulation off,
packer accumulation requested on. This matches the successful shared-policy
control. The new DRAM kernel's accumulation boundaries can still change
rounding: down has one K block and consequently disables effective packer
accumulation in the factory, while gate/up has eight blocks. Correctness needs
measurement despite the preserved requested policy.

## Packing and legal geometry

The original global shared width2112 is padded to2176, then each of four
ranks owns544 columns. The candidate pads **each rank's** `[up544, gate544]`
from1088 to1280; its global packed tensor is `[1,1,2816,5120]`, mesh-sharded
along the last axis. Down stays `[1,1,2176,2816]`, mesh-sharded along K.
The device gate/up output is converted to interleaved L1 and cropped to1088
before the existing544/544 split. No bank padding participates in GELU or down.

| Projection | Local K,N | DRAM bank shard, eight banks | L1 input shard | L1 output shard | Storage cores | K block | Program M,N |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Gate/up | 2816,1280 | 2816,160 | 32,352 | 32,160 | 8 | 11 tiles | 1,5 tiles |
| Down | 544,2816 | 544,352 | 32,544 | 32,2816 | 1 | 17 tiles | 1,88 tiles |

All grids have row-major orientation. Inputs and outputs use explicit L1
WIDTH_SHARDED memory; weight shards use DRAM WIDTH_SHARDED memory. One worker
per bank is requested. The opt-in setup checks the target's eight-bank and
H2816/I544 geometry instead of silently applying it to another configuration.

Source proof, rooted at `ttnn/cpp/ttnn/operations/matmul/device/`:

- `matmul_device_operation.cpp:1302-1375` requires one M tile, width-sharded
  L1 input/output, width-sharded weights, row-major input, and K plus local
  input-shard K divisible by the K block. Here the divisions are88/11,
  11/11 and17/17. `:1216-1230` checks input/output buffer and layout matching;
  their shard widths may differ.
- `factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:124-167`
  chooses eight bank reader/compute workers independently of input/output
  storage grids. Gate workers read five N tiles each; down workers read11.
  The down program's `per_core_N=88` describes output storage on one core,
  not88 compute tiles per bank worker.
- The same factory at `:742-887` divides each worker's output across storage
  shards. With down, eight11-tile segments target successive offsets of the
  one88-tile output shard. A one-core activation/storage grid therefore does
  not reduce the bank reader count to one.
- `utilities/matmul_utilities.cpp:376-393` preserves input storage entries
  while removing shared entries from the separate worker list, so a storage
  core that also serves as a bank reader does not disappear from the input
  multicast count.
- Factory `:202-246` derives effective packer accumulation from K-loop count,
  selects data-format-aware CB sizes and applies DRAM alignment. There is no
  C++ change in this proposal, so no C++ build is required for preparing it.

The primary worker input buffers are modest: gate/up BF16 A double-buffer is
45056 bytes, with B triple-buffer179520 bytes sliding/95040 bytes full. Down
has BF16 A34816 bytes and B203456 bytes sliding/107712 bytes full (one K block).
These figures exclude outputs, multicast storage, runtime/kernel allocations
and overlapping tensors; they are not a live-L1 proof. The down output storage
itself is180224 bytes on one core. Hardware validation must resolve actual
allocation, trace lifetime and reader/storage interactions.

## Capacity and CPU checks

The candidate replaces the interleaved **decode** matrices with bank-sharded
decode matrices, retaining original BF16 prefill weights. Gate/up bank padding
adds528 weight tiles per rank:574464 bytes sliding or304128 bytes full. Across
25 sliding and5 full layers this adds15882240 bytes per device over the
existing optimized-shared accounting. The runner patch adds this padding to
both other-resident reservations and the current layer's weight bound; it
records `shared_dram` in the result. Keeping both interleaved and bank-sharded
decode copies would require a larger bound and is not what this patch does.

Completed checks:

- Both generated candidate Python files parse with `ast.parse`.
- The new class was Black-formatted without reformatting unrelated live code.
- `git apply --check shared_dram.patch` and `git apply --check
  shared_dram_runner.patch` pass against the recorded working sources.
- A Torch CPU sentinel check using the real2112/2176/544/1088/1280 widths
  verifies all four local up/gate pairs survive packing/cropping exactly and
  every bank-padding column is zero. Divisibility and weight-byte calculations
  pass; `ttnn_imported=false` is recorded.

## Parent-owned test ladder

After applying/rebasing the two patches, hold hybrid experts, fused tail,
shared precision, topology and residual layout fixed. Start with layers0 and5,
length65,8 traced steps and cache checks, comparing the existing backend to
`--shared-dram`. Then run paired4096/128 and the stack/batch gates on any
candidate that passes. Compare complete-layer traced host intervals first;
a new native profile must include input/output reshards, gate crop/slices,
GELU and the shared reduction. Do not claim a matmul win if conversion or
collective cost cancels it. Profiler and Watcher runs remain separate.

No geometry/performance verdict exists yet. The imported old MoE DRAM helper's
historical PCC regression is not bypassed by assumption; this candidate needs
the same real-weight gate and cache/trace evidence as other shared backends.

## Optional next roles: setup-only weight reshards

No QKV/WO patch is included. When a current profile justifies it, their already
quantized local weights can be converted at setup with
`ttnn.to_memory_config(existing_weight, bank_width_sharded_memory)`; this
preserves mesh ownership and avoids new Torch packing or quantization.
`core/to_memory_config/to_memory_config_op.cpp` routes this interleaved-to-
sharded transfer, and the interleaved-to-sharded factory explicitly handles
DRAM destinations. WIDTH_SHARDED is supported; legacy DRAM BLOCK_SHARDED is
explicitly rejected.

| Role | Existing local K,N | Suggested input storage cores / K block | Output storage N tiles/core |
| --- | --- | --- | --- |
| QKV sliding | 2816,2048 | 8 /11 | 8 |
| QKV full | 2816,3072 | 8 /11 | 12 |
| WO sliding | 1024,2816 | 8 /4, or4 /8 | 11, or22 |
| WO full | 2048,2816 | 8 /8, or4 /16 | 11, or22 |

All N dimensions already divide8*32, so these conversions need no extra
column padding. QKV can use a decoder-local wrapper that retains the original
`_Projection` for prefill and uses its `decode_compute` with the bank-sharded
copy for logical S1. Preserve FP32 normalized input/output and FP32 destination
accumulation, HiFi2 sliding/LoFi full. Preserve packed local Q/K/V order.

WO requires a decode-only adapter at the existing concatenated-head projection
boundary: preserve the original interleaved `weights.o_proj` for prefill, keep
head concatenation, then use the bank-sharded copy, its working input shard,
and existing HiFi4/FP32-destination compute. **The existing WO input is BF16
SDPA/head-concat output**, whereas its result/accumulation are FP32; changing
WO input to FP32 would be an additional precision change. Restore the current
projection output memory contract before the unchanged reduction.

FP32 support is explicit in the DRAM factory: `:170-174` selects destination
subblocks using `fp32_dest_acc_en`, `:202-209` chooses FP32 intermediate format,
`:386-387` emits `FP32_DEST_ACC_EN`, and `:950-969` derives input/output CB
formats from tensor dtypes. No BF16-only restriction appears in this program's
validator. That establishes source support, not a correctness or performance
result; retain exact Q/K/V/cache and router-sensitive real-weight tests.

The extra setup copies require residency accounting. Converting a weight that
is retained for prefill does not eliminate the original allocation. Keep QKV
and WO separate from the shared experiment and from each other so any outcome
has an identifiable boundary.
