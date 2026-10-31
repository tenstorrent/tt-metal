# Optimized TP4 decoder

Stage05 optimizes this run's completed `MultichipDecoder` in place on a1×4 Blackhole P300c mesh. The pinned text-only model is google/gemma-4-31B-it, revision842da3794eaa0b77d5f08bae87a17459d91ff475. No full-model or serving implementation is part of this stage.

The accepted baseline is commit bdbdc9e17c3f5dbe0ac12ac1e44bc408c81f8da0. Before/after measurements use real pinned weights and recorded HF inputs,4096 input tokens,128 traced decode replays, batch1. The agent workload also requires batches1,8,16,32; it estimates32 concurrent32768-token conversations. Layer timings are teacher-forced layer measurements, not token-out model generation.

The initial operation-topology audit, commands and experiments are in work_log.md. Candidate JSON/manifests distinguish host timings from device measurements. Final default profiles and acceptance are complete. Independent review is recorded in `stage_review.md`; checkpoint SHAs are recorded in `work_log.md`.

The inherited inter-layer contract is replicated BF16 hidden5376 on TP4, width-sharded across8 local L1 cores in decode and DRAM interleaved in prefill. No collective occurs between layers. Local Q/K/V and intermediate MLP channels are TP-sharded. Row-parallel O/down reduce within the layer. The alternative1344-channel mesh-sharded residual carries through distributed norms and adds, with gathers delayed to consumers; it is measured separately rather than rejected with an immediate restore.

The262144-token logical context, internal padding for arbitrary logical prefill lengths, BFP8 local KV, and2048-token cyclic sliding cache are preserved. No context reduction was made.

## Selected implementation and stack contract

The default keeps packed QKV (sliding) or QK with tied raw V (full), BFP4/LoFi projection weights, BFP8 KV, and split gate/up with fused GELU multiply. Internal1024-token prefill chunks use an8-column block-sharded RMSNorm for total physical rows≤1024; larger batched chunks retain the prior interleaved norm. The norm remains HiFi4/FP32. Each conversion owns private storage before in-place normalization. The height planner uses only occupied shard rows, preserving awkward logical tails.

Decode O/down now slice padded output in L1 and move directly to RS input, removing the intermediate width-reshard. RS staging/output remains persistent in DRAM. AG writes directly into persistent8-core width-sharded L1, ready for the consuming norm. Prefill avoids the duplicate BFP8-to-BFP8 CCL copy.

A stack **must create one `decode_gather_buffers = {}` for its mesh** and pass that same dictionary to each `MultichipDecoder.from_state_dict(..., decode_gather_buffers=decode_gather_buffers)`. It holds two BF16 physical `[1,1,32,5376]` tensors, keyed by `self_attn.o_proj` and `mlp.down_proj`. B1/8/16/32 are logical views of those backings. Per-device pool bytes are688128 total,86016 on each of8 shard cores, independent of layer count. Independent per-layer pools would exceed per-core L1 at60 layers and are not the full-model contract. This stage introduces no full-model code.

Workspace lifetime: all consuming layers/traces retain the dictionary while live. On the one serialized mesh stream, each gather overwrites its output before reading it; the following non-inplace norm consumes it completely before the next same-role gather. Norm and residual outputs do not alias the pool. Concurrent unsynchronized executions on the same pool are unsupported; independent streams need independently budgeted workspaces. Construction and warmup of all intended signatures precede trace capture. RS staging remains independently owned by each layer/signature.

`shared_workspace.json` validates pooled identity across both layer kinds and all supported batch views, four live B1/B32 traces, and32 alternating replays against eager on every device. It acknowledges only exact captured output tensors whose data is consumed immediately after their own replay; no capture/forward/warmup allocation scope or program-cache exclusion is used. The production pool itself is allocated before captures and needs no exemption.

## Rejected alternatives and limitations

`candidate_summary.json` contains whole-layer timings and PCC. The output-sharded fused AG-MM decomposition was adapted to legal padded local weights and BF16 fused inputs, then measured with replicated and carried-sharded residual contracts; both lose. Stage04's distinct fused MM-RS and input-AG-MM/sharded families are retained in `../multichip_decoder/mesh_plan.md` and their numerical artifacts. Geometry, per-role precision, reduced activations and combined reduced-CCL controls are recorded in this stage's work log. The explicit BFP4/LoFi geometry controls include wider sender shards and larger non-power-of-two blocks.

Ethernet watcher instrumentation on this build exceeds the ACTIVE_ETH configuration buffer by2640 bytes. Worker-core watcher acceptance uses `TT_METAL_WATCHER=10` with `TT_METAL_WATCHER_DISABLE_ETH=1`, separately from profiling; numerical, trace, cache and stress checks still exercise Ethernet transfers. The exact capacity evidence is inherited from `../multichip_decoder/watcher_configuration.json`.

## Final measurements and validation

`performance.md` reports before/after whole-layer device windows for batches 1, 8, 16 and 32. At 4096 input tokens and 128 advancing decode replays, batch 1, sliding attention improves from 11422.54 / 484.77 us to 11150.01 / 475.22 us (prefill / decode). Full attention improves from 13748.47 / 534.77 us to 13488.43 / 527.82 us. These are TP4 layer measurements, not end-to-end generation or TTFT. Device windows include all operations and internal gaps; no matmul subtotal is used as the roofline denominator.

All 54 acceptance cases pass the accepted 0.995 PCC gate, including both layer kinds, four batches, awkward logical lengths, cache wrap, 128 advancing stress replays and maximum context 262144. The maximum-context check executes the full prefill and compares its last 32 rows plus decode. `stack.json` confirms two-layer composition without a boundary conversion. `shared_workspace.json` additionally verifies the final source with allocation tracking, program-cache allocations included, four live traces and 32 alternating replays on every device. Runtime fallback guards are enabled.

The final default path reproduces the selected candidate's batch-1/batch-32 traced host latency within 0.06%; final device values come exclusively from that default. The one failed profiler capture and successful identical retry are preserved in `profiler_anomaly.json`; `acceptance_completion.json` reconciles this with the original queue's nonzero exit. No hardware reset was required.

All applicable report advice is dispositioned in `profile_analysis.md`, `optimization_audit.md`, and the candidate tables. Larger-batch input-L1 trials include smaller block adaptations and two complete streamed layout families; capacity evidence distinguishes total capacity from contiguous allocation limits. Reader controls preserve real operands and selected precision. No applicable optimization is deferred. Raw multi-GB captures remain local and excluded from commits; compact tables, CSVs and provenance are retained.
