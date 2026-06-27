# reader_bmm_tile_layout_in1_sender_writer_padding — v8 (remigration)

`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
Role: hybrid (sender). Tag: clean. Tier 0a (matmul). Commit `d5ca9f7dbe154aa9b7e1a784f12123055e9ba228`. migrated v8 — PASS.

## Transform (v7 -> v8)
**PURE DELETION** of the 3rd template arg (`in1_mcast_num_dests,`). Nothing added to the ctor (default `consumer_ack_count = ACK_EQUALS_FANOUT`).

Justification (dense): the in1 sender's only live mcast path is the **2D** factory, where in1 is dense — `in1_mcast_num_dests == in1_mcast_num_cores == num_blocks_y - 1` (2D factory lines 480-481), a fully packed column of receivers. So ack == rect-derived fan-out and the default is correct. In the **1D** mcast_in0 path this kernel is compiled with `-DSKIP_MCAST` (confirmed in the JIT compile line), so the SenderPipe is `#ifdef`'d out and never runs there — the in1 divergence concern (which exists in the 1D *factory* numbers) does not apply to a live code path.

Same pipe serves both the in1 and in3/bias `send()`s.

## Validation
- `test_matmul_2d_multiple_output_blocks_per_core` full suite (`--run-all`): **56 passed, 72 skipped, 0 failed** (45s) — this is the suite that actually exercises the in1 SenderPipe.
- `test_matmul_1d_multiple_output_blocks_per_core` full suite: 48 passed, 16 skipped, 0 failed (SKIP_MCAST path here).
- 2D smoke (`--dev`, transpose_mcast=True, in0_sharded=True, grid (8,4), n=1024-k=512-m=512-b=1): PASS.
JIT-built: confirmed (`generated/inspector/programs_log.yaml`).

## Lines
diff_lines_removed: 1 (the 3rd template arg).
