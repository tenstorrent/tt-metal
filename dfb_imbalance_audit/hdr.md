# DFB imbalance audit — family: layout_dm

## EXECUTIVE SUMMARY (exercised-by-tests first)
EXERCISED by tests (will hang under #57646 finish()):
- E1 qsr pad writer_pad_tiled.cpp:53/62 — cb_pad_val reserve(1)+push(1), never popped; writer bound producer+consumer (pad_tile_multicore_program_factory.cpp:148). Hit by llama32_1b_quasar/tests/prototype_ops/test_pad.py (quasar.pad TILE). Fix: Scratchpad (as mainline writer_pad_tiled.cpp:119) or wait/pop(1) at end.
- E2 qsr transpose_wh_rm_sharded.cpp:93/106/110 (+ narrow :53-63) — borrowed output DFB, compute is producer+consumer, no writer (ht<=8); pushes n_blocks*Wt*Ht, wait_front w/o pop, pops=0. Hit by resnet test_fold_transpose.py (b1/b2_wh_4x256x224_aligned, b1_wh_out16_narrow, b1_wh_out8_narrow) / resnet fold transpose(2,3). Fix: pop_front after each push / replace :110 wait with wait+pop.
CONFIRMED but not exercised by the 4 test dirs: s2i RM partial last shard (mainline+qsr), qsr untilize_with_unpadding sharded-output writers (unpad_batch_rows / width_16), untilize wh_multicore writer skipping fully-padded tile rows, mainline pad RM sharded height-only, mainline slice RM sharded reader, mainline concat S2S tiled writer, qsr writer_unary_pad_dims_interleaved dangling reserve, qsr transpose HC sharded generic path, mainline typecast sharded (no consumer), embedding fused/rm sharded output, embedding tilized_indices (unported).
LATENT: many `#ifdef OUT_SHARDED wait_front(no pop)` writers (never defined). SUSPECT: tilize_val_padding reader push_back(0) with multi-TC DFB (likely exercised; depends on whether zero-entry push advances tc_idx), SubCoreGrids untilize, nd-shard untilize ordering, pad width-only shard heights, slice stride.

Per-subfamily details follow (each was traced reader/compute/writer with factory runtime args).
