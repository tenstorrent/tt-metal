# Prefill fused all-gather/matmul hang

Stage05 Qwen/Qwen3.8-27B TP4 decoder optimization. Investigation is read-only; do not use devices, rerun, reset, terminate or edit implementation. Other authorized tests may run independently. Write AUTOTRIAGE.md per installed skill.

Evidence root: models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder
Primary capture: triage_prefill_ccl_fused_smoke_ag_h2_l0/tt-triage.txt
Original log: prefill_ccl_fused_smoke_ag_h2_l0.log
Command: docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.run_prefill_ccl_candidate --layer 0 --batch 1 --seq 4096 --steps 128 --prefill-repeats 15 --policy '{"pf_fused":"ag"}' --output models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/prefill_ccl_fused_smoke_ag_h2_l0.json
The sweep watchdog captured triage before termination, then bounded reset/list recovery. This is first eager TP4 prefill, before prefill trace capture. Single-chip baseline and same TP4 separate-op control pass.

Runtime shape: input0 [1,1,4096,1280] BF16 interleaved DRAM per chip, gathered K5120. Linear mix weight [1,1,5120,4608] interleaved BFP4 (explicitly converted from native DRAM sharding). Logical output width4160 after owned slice. Mesh1x4 four Blackhole p300c ASICs, two-link ring. LoFi, no FP32 accumulation, packer L1 accumulation enabled.
Program: 2D multicast grid8x8, allowed_worker_cores rectangle(0,0)-(7,7), K block4, per_core_M16, per_core_N18, out_block_h2, out_block_w18, subblock1x2, fuse_batch false, transpose_mcast false. All-gather worker offset(0,8), dim3, links2, workers/link1, chunks/sync10, buffers/channel2, gather outputDRAM, matmul outputDRAM. Adapter source snapshot prefill_source_provenance/adapter_v3.py matches this trial.

Direct stop site includes NCRISC reader_bmm_tile_layout_in0_sender_padding → MatmulOpReceiver::update_current_block_start_tile_id → semaphore wait_min (worker_sync_utils.hpp:253). Many downstream compute/input1/receiver waiters. Build actual semaphore producer/consumer and M/N/K loop ledgers; determine earliest unmet condition, not broad fanout.

Previous adapted interleaved-weight AG controls at decode M32 pass (column_family32_results.json, tests/multichip_fused_candidate.py). Known native-weight AG descriptor/parser and K-order defects are in AUTODEBUG_fused.md; this failure uses INTERLEAVED weights, so do not merely repeat that diagnosis. Source permits rank4 dims0/1==1 and arbitrary dim2; validation alone did not forbid M4096.

Focus: tests/run_prefill_ccl_candidate.py; experimental all_gather_matmul_async; experimental matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp; reader_bmm_tile_layout_in0_sender_padding.cpp; ccl/kernel_common/worker_sync_utils.hpp and associated AG signalers. Propose smallest caller/config adaptation if sound, or exact source repair. Verify the proposed behavior is absent. Distinguish measured facts from hypotheses; no hardware validation in child. Preserve native decode pipeline, no full-model/vLLM work.
