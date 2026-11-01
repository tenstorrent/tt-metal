# row_gather

`out[r] = src[index[r]]` for ROW_MAJOR bf16 tensors whose rows are single DRAM pages (`src [S, W]`, `out [N, W]`,
`index [1, N]` uint32 in one page). One data-movement kernel per core over a contiguous range of output rows: the
core's index slice is read once (its range starts at a multiple of 16 entries, a 64-byte offset), then rows stream
through a ring of `GLM53_ROW_GATHER_IN_FLIGHT` (8) in-flight 8 KB reads followed by their writes; indices at or beyond
`S` write a zero row. Built for the regions prefill path's two `ttnn.embedding` row gathers (43-57 GB/s measured
2026-11-01: 27 + 18 ms per MoE layer at 16 K gathered tokens on the 8x4). Golden: `RowGather.golden`.
