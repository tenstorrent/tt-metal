# Long-context tail address audit

Source-only competing hypothesis audit for AutoFix, 2026-11-12. No device was opened, no target test was executed, and no implementation was edited. Parent supplied the failure: full-attention prefill length 262143, aggregate PCC about 0.999269, aligned row 262110 PCC about 0.99953, final one-token tail row 262142 PCC about 0.94594; short-context tests pass.

## Finding

The inspected address path does **not** contain a 16-bit position or tile-index truncation at 65536. The exact target geometry stays inside every inspected integer/address bound. This refutes the proposed integer-width mechanism in these sources; it does not prove that runtime cache contents are correct.

The stronger source-supported alternative is repeated BF16 rounding of decode's online softmax state. The decoder selects one compute core and a fixed 32-token key chunk. At position 262142 this means 8192 sequential chunks per KV head. Native decode hardcodes both intermediate and statistics circular buffers to BF16, independently of the model's FP32 destination accumulation setting. The aligned prefill path uses a different kernel and 8x8 grid. Changing from aligned prefill to single-token decode therefore changes numerical accumulation behavior exactly where the observed discrepancy occurs.

## Address proof for the target

`tt/functional_decoder.py` uploads `current_pos` as an int32 row-major tensor, and the page table as int32. The cache shape is `[8192, 2, 32, 256]` at advertised context 262144. No modulo capacity is passed, so the native default zero means no position wrapping. `prepare_prefill` dispatches the final 31 tokens of a 262143-token request individually, at absolute positions 262112 through 262142. Those positions have correct RoPE tensors and device current-position values prepared outside forward.

The update-cache reader (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:84-127`) reads a uint32 position, and selects uint32 page entries unless the table element size is uint16. The factory's conditional at `paged_update_cache_program_factory.cpp:302` does not select that uint16 branch for this model. Writer lines 74-115 use the same uint32 position and page mapping.

With page size 32, two KV heads, and eight head-width tiles:

- Logical position 262143 maps to virtual page 8191 and row 31.
- Every permutation entry is in `[0,8191]` and unique; the setup code validates this.
- Physical page stride is `2 * 1 * 8 = 16` tiles.
- Largest physical tile ID is `8191*16 + 1*8 + 7 = 131071`, represented in uint32.
- One BF16 cache is `131072*2048 = 268435456` bytes, with last byte offset 268435455; no 32-bit offset overflow occurs.
- Current-position arithmetic rounds at most 262144 tokens to 8192 tile rows, far below signed-int32 and uint32 limits.
- A page-table row is `8192*4 = 32768` bytes, and the inspected page-table size/index arguments are uint32.

The decode reader (`transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp:115-187`) reads the position through uint32, and its unsharded DRAM table path at lines 271-294 selects a uint32 table pointer. Both K and V map virtual tile rows via `virtual_seq_tile_id_to_physical_tile_id<uint32_t,...>` in `dataflow_common.hpp:687-700,838-845`. The shared mapping definition in `transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:54-68` computes exactly `physical_page*16 + head*8 + width_tile` for this geometry. The native update and decode mappings agree. `rt_args_common.hpp:35-92` uses int32/uint32 for workload endpoints; its uint8 dynamic chunk helper is **not used** because the model explicitly sets a nonzero key chunk size of 32.

The fill-cache path uses a page-table suffix starting at `start_pos//32`; native fill indexes that suffix from the local sequence start. This is consistent for the aligned interiors. The single-token tail uses the original full table and its absolute position, also consistent.

## Precision mechanism and discriminating experiment

In `transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:507-508`, `im_df` and `stats_df` are explicitly `Float16_b`. Lines 695-707 allocate running output, maximum, and sum buffers in those formats. `fp32_dest_acc_en=True` improves arithmetic inside an operation but cannot preserve precision lost when those buffers are packed between chunks.

`sdpa_flash_decode.cpp:539-574` repeatedly rescales the prior sum/output, adds the current chunk, and packs the accumulated values into those BF16 buffers. At 8192 chunks, this is qualitatively different from short tests with only a few chunks. As a simple representability illustration, equal chunk sums of 32 cease incrementing a round-to-nearest BF16 sum at 8192 because the spacing there is 64; real attention has unequal chunks and changing maxima, so this illustration is not a quantitative prediction of the measured output.

The minimal decisive experiment is to hold Q/K/V/page table/current position fixed and compare the native decode result against an HF attention calculation while increasing key-chunk size and/or distributing the reduction across more cores. Check K/V contents separately around positions 65535/65536, the last aligned page, and the tail to retain a direct control on the address hypothesis. More cores require the caller's Q layout/sharding to satisfy the larger program grid; simply changing the program grid while keeping an incompatible shard is not a valid precision test. Improvement without changing cache bytes or index tensors would strongly identify reduction precision. A wrong-cache theory instead predicts mismatched gathered K/V rows independent of compute-grid choice.

These are source findings and proposed experiments, not new device-pass claims.
