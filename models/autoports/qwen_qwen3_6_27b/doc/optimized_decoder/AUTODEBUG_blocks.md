# AutoDebug: DRAM-width-sharded prefill output blocks

Inspection-only diagnosis, 2026-11-12. No TT hardware was opened and no implementation was edited.

## Finding

**Confirmed source-level contract defect: `out_block_w < per_core_N` is unsafe for the current 2D multicast DRAM WIDTH_SHARDED operand-B reader.** The decoder selected that combination whenever `pm > 2` or `pn > 68`. Smaller output **height** blocks are a different case: the current reader explicitly resets its K read offset per block, so the old height-reset hypothesis is refuted by current source. Height blocking with full worker width is the first adaptation to test.

The causal chain is concrete:

1. `matmul_multicore_reuse_mcast_2d_program_factory.cpp:411-425` sets `in1_block_w = out_block_w`, allocates `out_block_w * in0_block_w` B tiles per buffer block, and sets the count of N blocks to `per_core_N / out_block_w`.
2. The same factory at line 922 creates `DramWidthShardWalker(..., per_core_N, ...)`. Its `advance()` method (lines 229-275) supplies bank spans covering the **whole per-core N width**, independently of `out_block_w`.
3. The selected kernel is `reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp` (factory line 1225). In that kernel lines 429-478 reserve only `in1_block_num_tiles`, but the bank loop reads every whole-worker span supplied by the walker. Each K row advances its destination by only `in1_block_w_bytes` (line 473). Thus a full-width read overwrites following logical rows and can overrun the reserved/allocated B buffer. Exact overflow depends on buffering and span sizes; the incorrect row stride exists regardless.
4. For each subsequent N block, this DRAM branch repeats the same bank spans and starts from offset zero (lines 403-409). It never advances the bank-span origin by `bw * out_block_w`. Even absent an allocation overrun, later N blocks therefore read the wrong columns. The nearby comment explicitly assumes one N block.
5. DRAM B validation does not enforce full-width output blocks (`matmul_device_operation.cpp:1985-2051`); the stronger one-output-block restriction there applies only to **non-DRAM** sharded B. Successful API validation does not establish supported reader semantics.

## Evidence consistency

- `dram2_padfix_0.json`: B1/S512, full output blocks, prefill PCC 0.99990765, repeated prefill bit equality true; decode PCC 0.99998888.
- `topology_packed_mlp_0_b1.json`: B1/S512, packed gate/up increases projection N and triggers the width-block branch; prefill PCC NaN, repeated prefill equality false, yet recurrent state and decode remain correct. This is consistent with corruption first entering the MLP after state creation.
- `topology_base_0_b32.json`: B32/S33 means flattened M=1056 and `pm=ceil(1056/320)=4`; the same branch now reduces width for **all** projections. Prefill and downstream recurrent state are corrupt, consistent with first corruption before state creation.

These artifacts corroborate the source defect but do not localize the first bad device tensor; the adaptation still needs a real-weight rerun.

## Decoder-local adaptation and checks

First, always use `out_block_w=pn` for DRAM-width-sharded B. Keep `out_block_h=1` when the full output block does not fit. Height-only blocking has correct K-offset reset in the current kernel. If packed gate/up still exceeds L1, reduce prefill `in0_block_w` from 4 to a legal K divisor (2 or 1) and measure. This shrinks B double-buffer storage without introducing N blocks. Do not reject packed projections before this adaptation is measured.

If height-only blocking exposes another issue, chunk flattened activation rows into at most `32 * gy` rows (`320` for gy=10), perform full-width/full-height (`pm=1`) matmuls, concatenate device outputs, and restore `(batch, logical_seq, N)`. A tail is padded internally to a tile and sliced back; do not impose a logical alignment restriction. Retain explicit logical N cropping for DRAM-bank-padded weights. For B1/S512 the existing successful full-output-block path can remain the control. Row chunking alone cannot fix an oversized full-width B buffer; use smaller K blocks or construction-time packed-weight partitioning if necessary.

Discriminating hardware runs: B1/S512 packed MLP with `out_block_w=pn, out_block_h=1`; B32/S33 base with that same change; then B1/S513 and B32/S31 or S35 to cover tail rows. Require finite prefill/output/state tensors, functional PCC, repeated equality and decode PCC; finally run watcher separately. Compare warmed prefill and traced decode against the existing correct baseline. No performance conclusion is claimed by this source-only report.
