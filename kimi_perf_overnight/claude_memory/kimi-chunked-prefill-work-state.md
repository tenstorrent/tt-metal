---
name: kimi-chunked-prefill-work-state
description: "State of Kimi K2.6 chunked prefill test work — trace-format adaptation done, ready to run on device"
metadata: 
  node_type: memory
  type: project
  originSessionId: 66f1de8d-b7ca-4fa2-84c2-bbb3793dd8fa
---

Kimi K2.6 chunked-prefill tests on branch ppopovic/chunked_prefill_runner_integration_rebased.
Test functions added earlier (committed 9965d44): test_kimi_prefill_{block,transformer}_chunked + _padded
in models/demos/deepseek_v3_d_p/tests/. Blackhole-gated, GateComputeMode.HOST_ALL, KimiK26Config
fabric payload. Block test uses MoE layer 1 (Kimi NUM_DENSE_LAYERS=1). Env vars: see [[kimi-prefill-env-vars]].

## Trace arrived (2026-06-11) at
/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok

## Trace FORMAT DIFFERS from DeepSeek (the core of this task):
DeepSeek ("single_file"): one safetensors per layer, all tensors as keys.
  hidden_states/layer_N.safetensors keys: decoder_output, post_attn_norm, post_mla_residual.
  kv_cache/layer_N.safetensors keys: compressed_kv, kv_latent_normed, kv_kpe_roped, kv_post_transform.
Kimi ("chunked_group_a_v1"): each tensor is a DIRECTORY of row-sharded files
  rows_<start>_<end>.safetensors (chunk_rows=4096, 14 shards, last 53248-56320). hidden_states/ renamed
  to decoder_io/. decoder_io/decoder_output_layer_N/ and kv_cache/layer_N/. ONLY captures
  decoder_output + kv_post_transform (MLA intermediates + post_attn_norm/post_mla_residual ABSENT).
  metadata.json has token_ids (length 56320) — same as DeepSeek.

## CODE CHANGES MADE (uncommitted, working tree):
- model_variants.py: added TestVariant.prefill_trace_layout field (default "single_file"); KIMI set to
  "chunked_group_a_v1" and prefill_trace_default repointed to the trace path above.
- test_prefill_block_chunked.py + test_prefill_transformer_chunked.py: added _LAYOUT_* consts,
  _read_sharded_rows() (overlap-aware shard concat, VERIFIED correct incl. cross-boundary), _load_layer_rows()
  layout dispatcher. Threaded `layout = variant.prefill_trace_layout` through run functions.
  Block: gated the 5 MLA-intermediate loads+PCCs behind has_intermediates (=_has_mla_intermediates(layout));
  Kimi still checks kv_post_transform + layer_output + cache-sanity. Transformer: both needed tensors exist
  so it validates fully (only the reader changed).

## DeepSeek NON-REGRESSION verified:
- get_slice[:n] == get_tensor[:n] bit-identical on DeepSeek trace (all tensor types).
- Test collection counts unchanged (ds block 10, block_padded 4, transformer 3, transformer_padded 3).
- has_intermediates=True for DeepSeek → all loads+11 PCCs run in identical order.
NOTE: DeepSeek prefill_trace_default still points at the now-removed
.../kimi-26/debug_trace/longbook_qa_eng_prefill_56320_nopad (pre-existing staleness, NOT my change);
real DeepSeek trace is /mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad.

## RESULTS (2026-06-11):
- Kimi BLOCK chunked: ALL 7 PASSED (9m42s) — chunks 1/2/5/10/11 + padded 1k+4k + padded full55k.
  PCCs: kv_post_transform ~0.9999, layer_output 0.9994-0.9995, cache-sanity ~0.9999. Host gate
  (HOST_ALL), layer 1, mesh 8x4, Kimi TTNN cache. Sharded reader + has_intermediates gating
  both confirmed working (skip msg fired 7x). Log: kimi_block_chunked.log.

## HOW TO RUN (from /home/ppopovic/tt-metal, after `source python_env/bin/activate` + the
## [[kimi-prefill-env-vars]] exports). No KIMI_PREFILL_TRACE_DIR needed (default points at the trace).
Block (7 tests, ~10 min):
  python -m pytest -x -rA -s \
    models/demos/deepseek_v3_d_p/tests/test_prefill_block_chunked.py::test_kimi_prefill_block_chunked \
    models/demos/deepseek_v3_d_p/tests/test_prefill_block_chunked.py::test_kimi_prefill_block_chunked_padded
Transformer (6 tests; L61 padded ~40 min):
  python -m pytest -x -rA -s \
    models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked \
    models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_padded
Single case: append param id e.g. [blackhole-kimi-mesh-8x4-moe-gate_host-chunks1].
Tests share the 8x4 mesh -> run sequentially; check mesh free first: pgrep -af "pytest|prefill_runner".

## TRANSFORMER RESULTS (2026-06-11) — all PASS, 0 below-0.88 warnings, record-only:
- NON-PADDED L10 (11 chunks): min decoder PCC 0.995657; per-layer 0.9976(L0)->0.9957. KV-cache
  min 0.993323 (L7 nope); pe(interleaved) >=0.9994 all layers. 7m23s. Log: kimi_transformer_nonpadded_L10.log.
- NON-PADDED L61 (11 chunks, 45m25s): min decoder PCC 0.969316 (layer 60). CLEAN MONOTONIC
  roll-off, NO DeepSeek-style L60 outlier (DS saw ~0.79; Kimi smoothly hits 0.969). Layer-60
  per-chunk: c0 0.9755, c1 0.9734, c2 0.9717, c3 0.9709 (gentle drift w/ context depth).
  KV-cache min 0.967593 (L59 nope); pe(interleaved) stays ~0.9988+ even at deep layers.
  Log: kimi_transformer_nonpadded_L61.log.
- PADDED L10 (full55k, 18 chunks, 8m06s): min decoder PCC 0.995554; ~identical to non-padded
  L10. KV-cache min 0.993323 (L7). Padding path does not perturb accuracy.
  Log: kimi_transformer_padded_L10.log.
- PADDED L61 (full55k, 18 chunks, 50m39s): min decoder PCC 0.967198 (layer 60); clean monotonic
  roll-off, matches non-padded L61 (0.969316). KV-cache min 0.967593 (L59 nope); pe(interleaved)
  >=0.9988 all layers. Log: kimi_transformer_padded_L61.log. ALL 4 TRANSFORMER CASES PASS.
Weight loading: all from prebuilt TTNN cache (.../kimi_k2_6_bh_32dev/8x4/), no on-the-fly convert.
Per layer = 8 MLA weights + ffn_norm + gate(+e_score_bias) + 12 local experts x{gate,up,down} BFLOAT4_B.

## RUNNER + PRODUCER (2026-06-11) — Kimi longbook, slot 1, request-loop PCC: PASSED.
Ran prefill_runner.py (PREFILL_MODEL_VARIANT=kimi_k2_6, PREFILL_REQUEST_LOOP_PCC=1, NCHUNKS=11,
SLOT=1, CHUNK_SIZE=5120, NUM_LAYERS=61, RECORD_ONLY=1, DEEPSEEK_PREFILL_TRACE_DIR=<kimi longbook trace>)
+ prefill_h2d_producer.py (same NCHUNKS/SLOT/PCC, CONNECT_TIMEOUT=120). Producer pushed 11x5120=56320
longbook tokens into slot 1 over the H2D socket; runner consumed 1 chunk/iter (~27s each, 61 layers,
HOST_ALL gate), then KV-cache PCC vs golden. KV cache min PCC 0.967593 (layer 59 nope) — BIT-IDENTICAL
to the non-padded transformer L61 test, confirming socket path == test path. Clean shutdown.
Logs: kimi_runner_slot1.log, kimi_producer_slot1.log.
CODE CHANGE (uncommitted) in prefill_runner.py: added _read_sharded_rows + _load_kv_post_transform
(auto-detect single_file vs chunked_group_a_v1 by checking if kv_cache/layer_N is a dir); swapped the
single-file safe_open in _kv_cache_pcc_check. DeepSeek path unchanged (still hits single_file branch).
Sequencing: runner must export H2D descriptor (after ~6min weight-load+compile) BEFORE producer connects.
NOTE: PCC-check gather materializes full cache [num_users*num_layers, tp, max_seq_len, kvpe] float32
(~69GB host RSS at NUM_USERS=2, MAX_SEQ_LEN=61440) — heavy but OK.

## COMMITTED 2026-06-11: commit b644eec24ff on branch ppopovic/chunked_prefill_runner_integration_rebased
"#0: Kimi K2.6 chunked prefill: support chunked_group_a_v1 sharded trace layout" — the 4 tracked files
(model_variants.py, test_prefill_{block,transformer}_chunked.py, prefill_runner.py). NOT pushed.
Untracked .ds_pcc_env.sh / .h2d_pcc_env.sh deliberately left out (personal env scratch).
Pre-commit black reformatted test_prefill_transformer_chunked.py on first attempt (re-add + recommit worked).

## RUNNER PER-CHUNK TIMING (request-loop longbook run, slot 1, 5120 tok, 61 layers, HOST_ALL):
~27.3s/chunk steady (iter0 26.8s -> iter10 28.0s, gentle rise as KV cache fills). NO warmup spike
(compile() runs before the loop). 11 chunks ~= 5 min compute.

## RUNNER RUN MODES (prefill_runner.py main() dispatch) + MINIMAL ENV:
- The ONLY env var that MUST be set for Kimi is PREFILL_MODEL_VARIANT=kimi_k2_6. Everything else has a
  correct in-code/variant default for the 8x4 BH setup: cache paths (variant ttnn_cache_default =
  /mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill resolves to .../kimi_k2_6_bh_32dev/8x4;
  TT_DS_PREFILL_* are setdefault'd), HF config (repo-local models/.../reference/kimi_k2_6/config.json),
  PREFILL_SP=8 TP=4 NUM_LAYERS=61 CHUNK_SIZE=5120 MAX_SEQ_LEN=61440 NUM_USERS=2, gate=HOST_ALL,
  is_balanced=False. So tests still need the [[kimi-prefill-env-vars]] exports, but the RUNNER does not.
- Mode select: PREFILL_STANDALONE_CHUNKED=1 -> golden chunked loop + KV PCC (no producer);
  PREFILL_STANDALONE=1 -> file-input loop from standalone_input.json, NO PCC, NO socket (single proc);
  else -> request loop (builds H2DStreamService, exports descriptor /dev/shm/tt_h2d_stream_service_<id>.bin,
  blocks at "Waiting for request to arrive..." until a producer/scheduler pushes; runs until SIGTERM).
  Add PREFILL_REQUEST_LOOP_PCC=1 to make the request loop run N chunks then KV-PCC.
- Slot in standalone/non-PCC paths comes from standalone_input.json "slot_id" (%NUM_USERS), NOT an env var.
  In PCC modes slot = PREFILL_STANDALONE_CHUNKED_SLOT.
- Producer caveat: PREFILL_H2D_CONNECT_TIMEOUT default 60s < runner ~6min load+compile, so bump it (e.g.
  900) OR start producer only after the runner logs "exported descriptor". Producer needs no model env.

## DEVICE GATE SWITCH (2026-06-11): Kimi chunked tests + runner now default to GateComputeMode.DEVICE.
PR #46761 (Danilo Djekic) adds _device_topk_gate() in tt_moe_gate_prefill.py: when n_expert_groups==1
(Kimi) the grouped routing collapses to a plain ttnn.topk (sigmoid->+bias-for-selection->topk->gather
unbiased scores->normalize*route_scale); also adds a Kimi mm_config keyed by n_routed_experts (per_core_N=12)
and switches the NON-chunked block/transformer tests to DEVICE. PR was NOT on my branch or origin/main, so
I CHERRY-PICKED its 3 commits onto ppopovic/chunked_prefill_runner_integration_rebased (8dde2cfd241,
81c610f206d, dc882e71e20 -> local 60d587e154e/09602205e1b/06a02a7d8ed; clean, tt_moe_gate_prefill.py was
identical to PR base). These dedupe on rebase once the PR lands on main.
MY EDITS (uncommitted working tree): runner_utils.py default_gate_mode "HOST_ALL"->"DEVICE";
test_prefill_block_chunked.py both Kimi parametrize blocks (1,HOST_ALL)/moe-gate_host ->
(1,DEVICE)/moe-gate_device + section comment; test_prefill_transformer_chunked.py both hardcoded
GateComputeMode.HOST_ALL args (run_chunked_transformer + _padded) -> DEVICE + section comment.
No gate-mode-dependent PCC thresholds exist in the chunked tests, so nothing else changed.
VALIDATED on device: block chunks1 PASSED (44s) — kv_post_transform 0.999899, layer_output 0.999271,
cache 0.999899; bit-identical to old HOST_ALL PCCs. Remaining block/transformer cases not re-run yet.

## RUNNER+PRODUCER on DEVICE gate (2026-06-11) — slot 1, 56320 longbook tokens (11x5120), 61 layers, PASSED.
Same run as the HOST_ALL baseline but runner now defaults to DEVICE gate. Per-chunk pipeline.prefill()
~3.3-3.6s/chunk (iter0 3314ms -> iter10 3573ms, gentle KV-fill drift) vs ~27.3s/chunk on HOST_ALL =
~8x speedup. Timing wraps ONLY pipeline.prefill() (excludes idle socket wait AND the post-loop KV PCC).
KV-cache min PCC 0.966684 (layer 59 nope), threshold 0.88, PASSED — essentially identical to HOST_ALL
baseline (0.967593, also layer 59). pe(interleaved) >=0.9986 all layers. DEVICE gate matches HOST_ALL
accuracy at ~8x throughput. Logs: /tmp/kimi_runner_device.log, /tmp/kimi_producer_device.log.

## PR #46761 REWORK + DEVICE_FP32 (2026-06-12) — branch ppopovic/kimi_chunked_prefill_device_gate
PR #46761 ("Adding a device MoE gate for Kimi 2.6") CHANGED APPROACH since the earlier cherry-pick: it
dropped the Python _device_topk_gate() (plain ttnn.topk collapse) and instead EXTENDED the C++ kernel
moe_grouped_topk to handle n_groups==1 (4th commit da58ec2d4e4 "extending the deepseek group topk to
support kimi single-group setup"). Now Kimi single-group routes through _device_grouped_gate_fp32() for
BOTH GateComputeMode.DEVICE and DEVICE_FP32 (identical for Kimi); the PR's canonical Kimi label is
DEVICE_FP32 (it switched the non-chunked test_prefill_block.py moe case DEVICE->DEVICE_FP32, id
moe-gate_device->moe_gate_device). C++ validate now early-returns for n_groups==1 (experts%32==0,
0<n_activated<=64) at moe_grouped_topk_device_operation.cpp:29-45, skipping the summed_experts_per_group==2
assert (which is the multi-group/DeepSeek path).
- Cherry-picked da58ec2d4e4 onto my branch = local f02e7f0dfa8 (CLEAN; all 8 touched files matched the
  commit's parent since I had the same 3 predecessor commits). Then committed b6bcae28665: switched Kimi
  chunked block+transformer tests AND runner_utils Kimi default_gate_mode DEVICE->DEVICE_FP32 + refreshed
  the stale "collapses to a plain top-k" comments. DeepSeek chunked tests left as DEVICE (bf16 grouped).
- *** BUILD GOTCHA (cost an hour) ***: the C++ kernel change needs a REBUILD, and there are TWO copies of
  _ttnncpp.so. `cmake --build build_Release --target ttnncpp` (or full build) writes the FRESH lib to
  build_Release/ttnn/_ttnncpp.so, but the python package ttnn/ttnn/_ttnn.so (a standalone copy, NOT a
  symlink) has RUNPATH build_Release/lib FIRST, so it loads the STALE build_Release/lib/_ttnncpp.so —
  which is NOT in the ninja graph (a leftover from an older build layout) and never gets updated by the
  build. FIX after every ttnncpp rebuild: `cp -f build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so`.
  Symptom if you forget: TT_FATAL "summed_experts_per_group must be 2. Got 1" at device_operation.cpp:30
  (old binary line number; source now has the n_groups==1 early-return). Verify with
  `ldd ttnn/ttnn/_ttnn.so | grep ttnncpp` + mtime.
- DEVICE_FP32 VALIDATED on device (2026-06-12): block chunks1/10/11 ALL PASS (186s) — kv_post_transform
  0.999896, layer_output 0.999171, cache 0.999896 (bit-identical to old DEVICE bf16). Logs:
  /tmp/kimi_block_devicefp32_chunks1_v3.log.
- RUNNER+PRODUCER DEVICE_FP32 (2026-06-12), slot 1, 61 layers, 11x5120 longbook, request-PCC: PASSED.
  KV cache min PCC 0.965424 (layer 59 nope; L60 0.969, L57 0.969) vs old DEVICE bf16 0.966684 — same
  accuracy. Per-iter pipeline.prefill(): 3159.79 -> 3409.35 ms (gentle KV-fill rise), mean ~3.30s/chunk,
  11 chunks ~36s compute. Runner banner confirms PREFILL_GATE_FALLBACK_MODE=DEVICE_FP32, variant kimi_k2_6.
  NOTE: PREFILL_H2D_SERVICE_ID defaults to "ds_prefill" — that's the SOCKET CHANNEL NAME, NOT the model
  (the model is kimi_k2_6); the "ds" in /dev/shm/tt_h2d_stream_service_ds_prefill.bin is just the channel.
  Logs: /tmp/kimi_runner_devicefp32.log, /tmp/kimi_producer_devicefp32.log.

## RUNNER PERF INVESTIGATION (2026-06-12) — PUSHED, cross-machine reachable.
Branch ppopovic/investigation on origin (github.com/tenstorrent/tt-metal). Commits: bfafc0a3a12
"Runner investigation" (env-gated experiment toggles) + 8061e7b54fd (full notes md). Full writeup:
models/demos/deepseek_v3_d_p/tt/runners/RUNNER_PERF_INVESTIGATION.md.
QUESTION: runner ~3.3s/chunk vs no-PCC transformer test ~1.94s/chunk (same 5120 tok, 61 layers,
DEVICE_FP32, mesh 8x4). Gap = ~constant ~1.4s additive per-chunk, prefix-independent.
ALL measurement hypotheses RULED OUT (each env-gated, all NO-CHANGE ~3.3s): PREFILL_PREFILL_SYNC=1
(post-sync), PREFILL_DISABLE_LAYER_ACK=1, PREFILL_NUM_USERS=1, 2-pass cold/warm (not compile),
PREFILL_PRESYNC=1 (pre-timer sync). => gap is GENUINE forward_chunk compute. Only remaining construction
diff: mla_seq_len 61440 (runner=pipeline.config.max_seq_len) vs 56320 (test SEQ_CACHE=55*1024). Attention
is logical_n-bounded (mla.py:649 ring_mla; sdpa factory 346/379) so shouldn't cost 1.4s — suspect the
ring_mla all-gather into _chunked_kv_buf touches full mla_seq_len. NEXT (not done): (1) rerun runner with
PREFILL_MAX_SEQ_LEN=56320 (=11*5120, valid) — 1 env var, decisive for mla_seq_len; (2) per-section
instrument forward_chunk (embed/per-layer/MLA-vs-MoE); (3) tracy per-op diff. STUCK-PROC HAZARD: killing
a runner mid-init races chip lock CHIP_IN_USE_*_PCIe and crashes a concurrently-starting runner in
Cluster::start_driver — kill+confirm-zombie BEFORE launching next; tt-smi -s checks health; never delete
TT_UMD_LOCK.* shm. Logs /tmp/kimi_runner_{nopcc,exp,2pass,presync}.log are LOCAL to bh-glx-d04u02 only
(all numbers durable in the pushed md).

## NO-PCC REQUEST-LOOP RUNNER+PRODUCER TIMING (2026-06-12) — runner in request-loop mode, PCC SKIPPED.
How: runner launched WITHOUT PREFILL_REQUEST_LOOP_PCC (-> pcc_mode=False, expected_chunks=None, no
_kv_cache_pcc_check, runs until SIGTERM); the request loop logs per-iter pipeline.prefill() ms regardless
of pcc_mode (prefill_runner.py:439). Producer launched WITH PREFILL_REQUEST_LOOP_PCC=1 (that var on the
PRODUCER only selects the longbook trace token source, _load_tokens():103 — independent of runner PCC).
KEY: same env var, two roles; split it across the two processes. Producer pushed 11x5120 longbook into
slot 1, runner consumed all 11, then idle-blocked on the 12th h2d_socket_sync -> SIGTERM (exit 144,
signal-killed mid-blocking-call; leaves a defunct zombie that holds no device — mesh still free).
Env: runner = PREFILL_MODEL_VARIANT=kimi_k2_6 + SP8/TP4/NUM_LAYERS61/MAX_SEQ_LEN61440/CHUNK5120/NUM_USERS2,
gate DEVICE_FP32 (runner_utils default). Producer = +PREFILL_REQUEST_LOOP_PCC=1, NCHUNKS=11, SLOT=1,
DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
(the debug_trace default in .h2d_pcc_env.sh does NOT exist — use the kimi_longbook_56320 dir, token_ids=56320).
RESULT per-iter pipeline.prefill() ms (kv_actual_isl 0->51200): 3262.45, 3245.69, 3273.44, 3379.16,
3383.08, 3392.40, 3380.65, 3452.35, 3429.80, 3416.24, 3412.26. Min 3245.69 (iter1), max 3452.35 (iter7),
MEAN ~3366 ms/chunk, TOTAL ~37.0s for 11 chunks. Roughly FLAT (only +150ms over 51k ctx, gentle KV-fill).
== IDENTICAL to the earlier WITH-PCC runner run (3.16-3.41s, mean ~3.30s, ~36s) -> confirms PCC overhead
is NOT inside the timed region (timing wraps only pipeline.prefill(), PCC gather is post-loop). Logs:
/tmp/kimi_runner_nopcc.log, /tmp/kimi_producer_nopcc.log.
CONTRAST vs no-PCC TRANSFORMER TEST (~1.94s/chunk ramped): runner is ~1.7x SLOWER/chunk and FLATTER ramp
(+0.15s vs test's +0.60s over same 0->51200 kv range). The delta is the PATH (socket H2D + runner
pipeline vs test's host-fed forward), confirmed NOT PCC since this run has none.

## NO-PCC TRANSFORMER TEST TIMING (2026-06-12) — test_kimi_prefill_transformer_chunked_no_pcc,
## -k "L61 and chunks11 and iters20 and kimi and mesh-8x4", DEVICE_FP32 gate, total_len=56320, chunk=5120.
20 iters x 11 chunks. Per-chunk RAMPS WITH CHUNK INDEX within an iter then resets each iter (KV cache fills
within the iter; chunk N attends to 0..N): chunk0 ~1.70-1.84s -> chunk10 ~2.28s, mean ~1.94s/chunk (iter1
chunks sum 21.37s/11). Per iter (11 chunks): iter0 35.4s (chunk0 alone 15.9s = one-time program-cache
WARMUP spike), iters 1-19 steady ~21.4-21.7s; inter-chunk overhead negligible (iter1: 21.37s chunk time
vs 21.381s wall). 20-iter run ~7min compute (14:41:27->14:48:35).
CONTRAST w/ RUNNER+PRODUCER DEVICE_FP32 (same 11x5120 longbook/61L): runner was ~3.16-3.41s/chunk FLAT
(~36s/iter); this test ~1.94s/chunk RAMPED (~21.5s/iter) = ~1.7x faster. Cause UNCONFIRMED — likely the
timed region differs (runner times pipeline.prefill() AFTER per-chunk H2D socket pull; test feeds golden
trace host-side) and/or runner doesn't reset cache between chunks the way the test resets between iters.
Worth a timed-region diff if perf attribution matters.

## NEXT:
- ALL Kimi transformer + block tests AND runner+producer path PASS; changes committed (not pushed).
  Remaining: push when ready; optional DeepSeek on-device non-regression (static proof done).
- Optional DeepSeek non-regression on device (not yet run; static proof already done): run
  test_ds_prefill_block_chunked with DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad.
- Nothing committed/pushed yet. Also uncommitted: prefill_runner.py 1-line "Waiting for request
  to arrive..." log (unrelated).

## SOURCE CONTEXT (this node): transcripts in
/data/ppopovic/.claude/projects/-home-ppopovic-tt-metal/ — main work session b9b7310c (Jun 10,
DeepSeek runs + rebase onto jjovicic/layerack-ack-request-loop + initial Kimi test scaffolding),
this session 66f1de8d (Jun 11, trace-format adaptation + block test run).
