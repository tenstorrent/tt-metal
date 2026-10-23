# AutoFix — fused TP4 candidate

Failing contract: full-layer sharded residual decoder must execute fused gather/matmul and reduce-scatter without stalled local workers. Baseline and nonfused TP4 outputs pass; fused first mix stalls before eager decode output. See AUTODEBUG.md and triage_fused/tt-triage.txt.

Diagnosis verified against current source: all_gather_async_default_program_factory.cpp builds direction core sets sorted by x while assigning worker indices in allocation order. At2 links×2 workers,12 allocated cores wrap the11-column p300c grid; master/slave synchronization targets differ. Capture identifies first AllGatherMatmulAsyncDeviceOperation, not a later RS failure.

Prioritized repair: use2 physical links with1 worker per direction, keeping the4 CCL cores on one row. This is a model-local candidate change within the authorized scope. Keep all matrix dimensions, local head ownership, BFP4/LoFi precision and RS buffers unchanged. Discriminating check: same full-layer fused-both sharded case must clear eager decode, then pass repeated trace and PCC. If not, capture the next first stall and isolate AG/RS.

Changed file: tests/multichip_fused_candidate.py (`num_workers_per_link=1` for fused AG only).

The original stall is resolved by the successful validation below. AutoDebug also found TP4 hardcoded-transfer assertion risk in fused AG and an RS output-spec inconsistency; neither is claimed fixed by this workaround. Watcher acceptance must investigate any corresponding failure. No shared TTNN source edited.

Launcher preflight initially failed because unprivileged namespace creation is disabled. Retried with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under existing unrestricted execution and skill-authorized isolated investigation. The child remained inspection-only and used no hardware.

Validation: `docker exec qwen38-functional-build timeout 180 bash models/autoports/qwen_qwen3_8_27b/tests/multichip_run.sh fused_both_w1_l0 --layer 0 --layout sharded --fused both` passes. Source was changed only to1 AG worker per direction/link. Both fused boundaries now execute;33-token prefill PCC0.99995815,2 advancing decode PCCs0.99999328/0.99999723,3 restored replays bitwise, trace allocation tracking including program cache clean. Local state reconstruction PCCconv0.99994389/recurrent0.99992455. Host decode819.99us vs replicated nonfused~515us. This proves the workaround for the observed stall; it does not establish a fastest path or watcher compatibility for the fused candidate. Final 4096+128 coherent tests also pass: final_fused_rs_l0/l3 and final_fused_both_l0/l3. Combined 1D AG+RS medians are681.52/599.03us; RS-only690.96/600.34us. Both are slower than selected490.06/408.52us, so fusion is rejected on measured latency. The unselected fused implementation is not claimed watcher-clean; selected unfused implementation passes all watcher gates.
