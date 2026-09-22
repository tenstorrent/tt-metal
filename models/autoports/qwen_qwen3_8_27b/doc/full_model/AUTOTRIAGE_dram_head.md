# AUTOTRIAGE

## Diagnosis

The split-bank DRAM reader forces the single-packet read path for a whole K-row
even when that row exceeds `NOC_MAX_BURST_SIZE`. The new BF8 LM-head full chunk
needs 17,408 bytes per reader row, exceeding Blackhole's 16,384-byte limit. This
violates the one-packet primitive's size contract and can leave global NoC read
response accounting inconsistent after all transaction-ID barriers complete.
The source contract violation is verified; its repair still needs hardware A/B
validation. This report was written before changing implementation source.

## Triage Evidence

- `watcher_final.log:171-180`: device 0, logical worker `(0,0)`, virtual `(1,2)`,
  BRISC stopped at waypoint `NKFW`, after
  `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` returned. The exact error
  is pending NoC transactions / missing NoC reads flushed barrier. The sibling
  NCRISC is the in0 multicast sender; compute runs
  `bmm_large_block_zm_fused_bias_activation.cpp`.
- The enabled watcher has no disabled features (`watcher_final.log:21`). The
  environment records `TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_NOINLINE=1`, fabric
  O3, and a separate fresh watcher cache. Exit status is 134.
- `triage/tt-triage.txt` and `triage/triage-summary.txt` did not capture device
  counters or stacks: Inspector data was absent after the host abort, and the
  default `--dev=in_use` selection caused all device checks to skip. Do not infer
  specific counter values or a live LLK/CCL stall from these files. The parent
  captured additional post-abort health evidence with explicit device selection.
- `triage/watcher/kernel_names.txt` and `kernel_elf_paths.txt` preserve kernel
  mappings. The copied watcher log has no flushed terminal failure sample; the
  host log above is the direct terminal evidence.
- Original command from `commands.log:15`:
  `models/autoports/qwen_qwen3_8_27b/tests/run_full_model_experiment.sh watcher_final models.autoports.qwen_qwen3_8_27b.tests.full_model_contract --batch 3`.

## Source Evidence

Paths below are relative to the repository root.

1. `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:126-139`
   computes `read_size = reader_width_tiles * in1_tile_size_bytes`, then calls
   `noc.async_read<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(..., read_size, ...)`.
   The second template argument is an asserted maximum, not a chunk-size limit.
2. `tt_metal/hw/inc/api/dataflow/noc.h:189-212` sets the transaction ID and forwards
   that maximum to `noc_async_read`. In
   `tt_metal/hw/inc/api/dataflow/dataflow_api.h:568-575`, a maximum no greater than
   the burst size selects `noc_async_read_one_packet`; it does not subdivide a
   larger runtime size. `ncrisc_noc_fast_read` in
   `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:480-504`
   submits the supplied length and increments `noc_reads_num_issued` once.
3. The any-length path in that same backend (`:859-879`) explicitly splits at
   `NOC_MAX_BURST_SIZE`, advances both addresses, and calls the counter-accounted
   primitive for each resulting packet. All these calls inherit the transaction
   tag already set by `Noc::async_read`.
4. Blackhole's `noc/noc_parameters.h:287-290` defines 256 words times 64 bytes,
   hence 16,384 bytes. A 32x32 BF8 tile occupies 1,088 bytes, already aligned to
   the 64-byte DRAM boundary. This tile size is also present in the generated
   watcher reader's `chlkc_descriptors.h`.
5. `tt_metal/hw/firmware/src/tt-1xx/brisck.cc:81-91` calls the kernel, then checks
   `ncrisc_noc_reads_flushed` at `NKFW`. That check is the equality between
   `NIU_MST_RD_RESP_RECEIVED` and `noc_reads_num_issued`, whereas transaction-ID
   barriers check `NIU_MST_REQS_OUTSTANDING_ID(trid) == 0`
   (`blackhole/noc_nonblocking_api.h:514-521`). The diagnostic does not by itself
   distinguish in-flight reads from inconsistent packet accounting.
6. Read-only disassembly of the preserved watcher reader ELF
   `reader_bmm_tile_layout_in1_sender_dram_sharded/7100684131117755059/brisc/brisc.elf`
   confirms the compiled full-head path: addresses `0x56a4` and `0x56b8` load
   `s0 = 0x4400` (17,408), `0x57f0` writes that value to the NoC length register,
   and `0x5804-0x5808` increment the software read counter by one. The loop advances
   K by 10 until 160 (`0x588c-0x5894`). This is compiled-artifact evidence, not just
   a reconstruction from model shapes. Inspection command:
   `runtime/sfpi/compiler/bin/riscv-tt-elf-objdump -dC --start-address=0x5680 --stop-address=0x58b0 <preserved-ELF>`.

### Geometry and transaction ledger

The factory is
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`.
It selects two reader workers per bank, requires Blackhole/NOC0, and passes
bank ID, VC, and reader index at runtime argument positions 3, 4, and 5
(`:117-126`, `:789-813`). It requires the bank shard width to equal the reader
count times unpadded reader width (`:161-169`) and passes block dimensions and
row stride at compile-time positions 3, 4, 9, 10, and 11 (`:348-362`).

| Quantity | Full head chunk | Tail head chunk |
| --- | ---: | ---: |
| Logical N per chip | 8192 | 4736 |
| Bank shard width, elements | 1024 | 640 |
| Banks / readers per bank | 8 / 2 | 8 / 2 |
| Reader width, tiles | 16 | 10 |
| BF8 bytes per K-row | 17,408 | 10,880 |
| K tiles / `in0_block_w` | 160 / 10 | 160 / 10 |
| Blocks / rows per block | 16 / 10 | 16 / 10 |
| Baseline software read increments per reader | 160 | 160 |
| Correct split packets per row | 2 (16,384 + 1,024) | 1 |
| Correct software read increments per reader | 320 | 160 |

For block `b` and row `r`, the source tile is
`(b * 10 + r) * bank_row_stride_tiles + reader_index * reader_width_tiles`.
Each block writes exactly 160 BF8 tiles for a full chunk or 100 for the tail.
The CB has three blocks; initially two blocks are reserved. Block 0 is issued
with tag 1; block 1 is issued with tag 2, then tag 1 is drained and block 0 is
pushed. Each later iteration drains and pushes the preceding block, cycling
tags 1/2/3. The final drain pushes the final block. Thus all 16 blocks are
produced and drained once; there is no missing final tag barrier in this loop.
The compute kernel consumes these blocks and produces the output block; the
reader waits for output, writes resharded slices, then performs a write barrier.
A host-only simulation of the exact issue/tag/drain loop for `num_blocks=1..256`
confirmed every block is drained and pushed exactly once and no tag is reused
while its earlier block remains pending. Arithmetic derived from Blackhole's
source constants confirmed the four row sizes and burst counts in the table.

The Stage5 decoder uses BFP4 (576 bytes per 32x32 tile) and generally three
readers per bank, so its rows can stay below the burst threshold. Even an
otherwise identical 16-tile BFP4 row is only 9,216 bytes. Thus passing smaller
decoder projections does not validate the new 17,408-byte BF8 boundary. The
non-split branch uses page sizes computed by `get_max_page_size_and_num_pages`
and is not implicated by this split-row contract violation.

## Downstream Effects

The firmware assert and host abort explain subsequent host teardown and device
recovery problems. There is no evidence here for an Ethernet/fabric root cause,
a compute CB mismatch, a precision-quality issue, or a need to disable watcher
checks. Adding a generic end-of-kernel barrier would not repair incorrect
software read accounting and could instead wait forever on unequal totals.

## Proposed Fix

Make the split-row `read_size` a compile-time constant and use it as the
`Noc::async_read` maximum-page-size template argument. Rows within the burst
limit retain the one-packet path; larger rows select the existing any-length
helper. Preserve transaction tags, row/column offsets, aligned tile stride,
CB reservations, tag barriers, output writes, and VC restoration.

Focused validation proposal, to be run only by the parent hardware owner:

1. A standalone Blackhole matmul with BF16 `[1,1,3,5120]` input stored width
   sharded on 8 cores with physical shard `[32,640]`; BF8 `[5120,8192]` weights
   DRAM sharded across 8 banks with shard `[5120,1024]`; block10, M1, N32,
   two readers, HiFi2, FP32 accumulation, packer accumulation, BF16 output.
   Run all watcher checks with O3/NOINLINE and fresh JIT cache; compare output
   against the same quantized operands in PyTorch. This preserves the failing
   row size and 16-block tag cycle without loading the full model.
2. Nearby controls: BF8 N7680 (15 reader tiles, 16,320 bytes), BF8 N8192
   (16 tiles, 17,408 bytes), BF8 tail N4736 (10,880 bytes including bank padding),
   and BFP4 N8192 (9,216 bytes). Repeat the matmul to exercise kernel handoff.
3. Build the native target under AGENTS.md, then repeat the original batch3
   watcher contract. A host library build may be a no-op for this JIT kernel;
   the fresh-cache watcher run must actually compile it.

## Uncertainty

No hardware was run by this investigator. Hardware counter values were not
captured by the initial triage. The one-packet contract violation and corrective
API selection are source-proven; the exact hardware handling of the oversized
request, focused A/B outcome, numerical output, and original watcher pass still
require the parent validation runs. Do not claim the stage passed from this
source diagnosis alone.

## AutoFix Follow-through

- An independent forked agent verified the single-packet precondition violation
  against `Noc::async_read`, `noc_async_read`, and the Blackhole backend before
  applying the proposed repair. The inherited source did not already choose
  the any-length helper for oversized split rows.
- Applied only the two kernel-line changes proposed above: `read_size` is
  `constexpr`, and the template maximum is `read_size`. No extra barriers,
  precision changes, alternate head implementation, or watcher changes.
- At the parent's request, extended the existing
  `test_matmul_dram_sharded_mesh_readers_cache` parametrization in
  `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py`
  with `(5120, 8192, 8)`, ID `wide_bfp8_row`. With `readers=2` this has the same
  17,408-byte row. It uses block2 (80 blocks), whereas the full head uses block10
  (16 blocks); therefore it is a focused row-size regression, and the original
  model watcher run remains required. The existing test checks numerical output,
  reader placement, cache reuse, and three rounds of distinct tensor addresses.
- Host-only verification passed: `git diff --check`; Python AST compile and
  parameter-to-ID mapping assertions; exact minimal kernel-diff assertions;
  row-size/burst arithmetic; tag ledger simulation for 1 through 256 blocks;
  and read-only disassembly of the original watcher ELF.
- Status: source fix and regression case prepared, native build and hardware
  verification delegated to the parent agent that exclusively owns device use.
  No device, reset, model execution, or native build was run by this investigator
  or its independent verification agent. The parent must record the actual
  build command, fresh JIT watcher regression result, and original watcher
  contract result before marking this repair verified.
