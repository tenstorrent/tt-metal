## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, shapes (seq 640 -> 20 tile rows, W = 28/32/48/56 tiles), gate.
- $DREAM_HOME/rmsnorm-prefill/history.md — index of all 43+ nodes.
- tests/ttnn/nightly/unit_tests/operations/fused/test_fused_rms_norm_prefill.py — [1,1,640,H] bf16 DRAM interleaved, bcast gamma, no heads/rope/bias -> head_dim_tiles == num_tile_cols, out page = row*W + c.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — trid-pipelined resident input pass (lookahead 4 blocks), pages row*W + c in order: bank c%8 identical on every worker when W%8==0.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — push (ack-free), go wait, stick read, posted drain in slot order to page row*W + c; gamma read already page-rotated by tile_row_start.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE / x*gamma pre-pass / POST all index input_cb and weight_cb by the same slot index and emit output in slot order -> a slot->column remap in the dataflow kernels leaves compute untouched.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — block_size = dst_reg_count, output_cb = 2 rows, writer_reads_weight / streaming / block_major / fuse_rope conditions; compile-arg tails of reader and worker writer.
- tt_metal/jit_build/build_env_manager.cpp — NUM_DRAM_BANKS is a define for every device kernel.

## Nodes consulted
- r01-b03-a03 — bank de-phasing on the 80-core column split: +3.4%, input read hot spot -1.2 µs on h4096, drain tail -0.4..0.6 µs. Never ported to the current 20-core lineage.
- r01-b01-a03, r01-b02-a02/a03, r01-b04-a03 — suggested rotating the input/drain walk; only gamma got rotated.
- r03-b04-a02 — drain per-core-bound; read 355 GB/s aggregate; waves don't pay.
- r03-b03-a03 — h3584 (two bank phases) drains at the same per-core rate as lockstep shapes (evidence against a large drain gain).
- r04-b02-a02 / r04-b02-a03 — synchronized drain starts slow each core's drain 0.13-0.42 µs; suggested rotating each core's first output bank.
- r04-b03-a01 / r04-b03-a02 (parent) / r04-b04-a01 / r04-b04-a02 / r04-b01-a01 / r04-b01-a02 — current writer + HiFi2 wins being stacked by siblings; this node is orthogonal.
- All round 1-4 reflections skimmed for de-phase / rotation / drain findings.

## Docs / external references
- none
