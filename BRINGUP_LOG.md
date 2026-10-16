# Molmo2-8B Bringup Log

## Current Status

**Status**: COMPLETE — 97/105 accuracy, S=36864 (384 frames + 4992 text) works. Rebased on main 2026-10-14: see Session 11.
**Branch**: `ssinghal/molmo2_new` — commit `5aca298376d`
**Docker image**: `molmo2-pr-dev:latest` (overlay on `0.12.0-735b65d-7a07a97`)

## Session 13 — 2026-10-14 (docker flow)

**Status**: tt-inference-server `run.py --docker-server --impl molmo2` verified end to end on the Galaxy (spec `id_molmo2_Molmo2-8B_galaxy`, one container, DP=4).
Image built from the local rebased branches via `models/demos/molmo2/serve/Dockerfile.local`. Request types pass (text, base64 image, https video, GCS image
URL, `file://` video inside the container); 105-suite 97/100, 0 errors, 393 s wall; heavy mix 12/12 (34k-token videos and 20-image prompts, 4 concurrent);
0 device timeouts. First heavy-mix run showed 6 failures: only the host `file://` path (not mounted in the container); rerun with the file copied in: 12/12.
Wikimedia image URL gives 422 (HTTP 403 to vLLM's User-Agent), not a model bug. Open: decode ~19-20 tok/s vs ~35 on the March T3K reference (optimization not started); branches not pushed.

## Session 15 — 2026-10-15 (performance)

**Status**: decode 48.8 -> 26.5 ms/step (20.5 -> 37.8 tok/s); 30-frame video prefill 1.62 -> 1.07 s; 105-suite (local files, sequential) mean 2.31 -> 1.96 s
with the di/dt throttle on, 98/100, 0 errors. 4 servers: suite 54 s; one DP=4 instance: 138 s; heavy requests (34k-token video 27-31 s, 20 images 17-21 s) all OK, 0 device timeouts.
Decode profile (tracy, eager step = traced step ops): lm_head replicated matmul 7.6 ms, 1-core RMSNorm 82 us x2/layer, 5 matmuls/layer at ~110 GB/s, readback of a tile-padded 9.7 MB logits tensor 9 ms.
Fixes: vocab-parallel lm_head (padded to 152064, sliced on device); logits row-major in the trace; decode RMSNorm sharded over 16 cores;
wqkv/wo bf8_b (HF teacher-forced agreement unchanged, PCC tests pass); ViT features all-gathered on device and pooled from the on-device table
(PCC 1.000000 vs host round trip; a single concat over 48 chunks hung the device, now grouped by 4); pooling chunk buckets (..2048, 2560, 3072, 4096);
FINE_PREFILL_BUCKETS for the eager path (S=2701 pads to 3072, not 4096). Remaining decode time (23.5 ms kernel): matmuls 14 ms (DRAM-sharded matmuls would roughly
halve it), CCL ~4.5 ms, SDPA 0.7 ms. Next: DRAM-sharded matmul programs, row-parallel wo, on-device sampling. Tests: tests/bench_decode.py, profile_decode_ops.py (tracy),
profile_prefill_wall.py, profile_vision_stages.py, test_vision_on_device.py. The Docker image was built before these changes (rebuild with serve/Dockerfile.local).

## Session 14 — 2026-10-14 (context limits over HTTP)

vllm-tt-plugin server on one 1x8 slice (max-model-len 36864, TT_MM_THROTTLE_PERF=3), raw token-id prompts via /v1/completions
(`tests/limits_http.py`, `limits_http2.py`): 36,848-token prompt + 16 new = 36,864: OK (9 s, answer " Marit"); one token more: clean HTTP 400
(context length 36864); chat request, 2048 new tokens (ignore_eos): OK, 105 s, distinct-4gram 0.98; 34,000-token prompt + 2048 new
(36,048 total): OK, 119 s, 1,604 words, tail distinct-4gram 0.45 (raw completion, no chat template; opens with a short repeated phrase, ends coherent);
server healthy, 0 device timeouts. (A first S=34k run with ignore_eos was invalid: the model had already emitted EOS.)

## Session 11 — 2026-10-14 (rebase onto main + hang triage)

**Status**: Rebased `ssinghal/molmo2_new_glx` onto origin/main (branch `ssinghal/molmo2_glx_main`). All tests pass on Galaxy
slices 1, 2, 3; **slice 0 (tray 3 chip 4) hangs under the full model workload — hardware suspect.**
**PCC**: 8 sub-block tests >= 0.998 (ViT 0.998176), TP MLP 0.999955, chunked pooling 0.999973, ViT DP 0.998637.
**Block Hash**: 55fa79e7042

### Work Done
- transformers 5.12 (pinned on main) cannot run the Molmo2 HF remote code (model card: transformers==4.57.1). Weights are
  now read straight from safetensors and the processor loads through a shim (`tt/hf_loading.py`, bit-exact vs 4.57.1).
  HF reference/golden scripts need a separate transformers==4.57.1 CPU env.
- Multi-image bug (also on the old branch): processor pooling indices are per-image-relative; HF adds per-image patch
  offsets (`build_batched_images`), TT did not -> images 2..N pooled image 1's patches. Fixed in `tt/vision_inputs.py`,
  used by demo + `generator_vllm`.
- Demo fixes: pytest options never registered (`demo/conftest.py`); batch>1 decoded all users from KV slot 0.
- Server adapter (`tests/test_generator_vllm_adapter.py`): prefill_forward/decode_forward over 5 back-to-back requests
  (decode trace reused) vs HF 4.57.1 greedy tokens: teacher-forced top-1 agreement 97-100%.
- Model-card lengths (`tests/test_long_context.py`): 384-frame video S=34455 + free generation, 97% teacher-forced vs HF;
  36848-token text prompt + 16 tokens OK (max_position_embeddings=36864).
- Full suite (4 PCC modules + 6 demo cases, 19 tests): 19/19 on slices 1, 2, 3.

### Prefill hang (slice 0 only)
- Symptom: host blocks in `to_torch` at end of `forward_prefill` (S~25k, 20-image prompt). 8 chips: 7 sit in an
  `AllGatherDeviceOperation`, ONE chip never finishes a compute op on a single/few column-0 cores.
- `tools/tt-triage.py` (5 clean captures): always mesh node N5 = physical Device 4 (tray 3, PCI 0000:05:00.0), whether
  the stuck op was `LayerNormDeviceOperation [1,4,25344,128]` (core (0,4), identical PCs across runs) or
  `MatmulDeviceOperation` (column (0,0)-(0,8)). Telemetry (1000 MHz, ~33 C, DRAM/ETH/ARC) all healthy.
- Not the cause: async CCL race (sync before q_norm: still hangs), the op itself (`tests/rmsnorm_loop.py`: 400k
  isolated q_norm ops pass on slice 0 and slice 1), thermal, links. Reordering `TT_VISIBLE_DEVICES` keeps the stall on Device 4.
- Counts: slice 0 hung in 7/9 full-model runs; slices 1/2/3: 0 hangs in ~60 prefills + 3 full suites (one concurrent
  4-process run also stalled slice 2 once, triage failed; solo reruns clean).
- Action: treat slice 0 as unavailable for Molmo2 until tray 3 chip 4 is checked by hardware; run 3 instances (slices 1-3).
  Repro: `python models/demos/molmo2/tests/stress_prefill_hang.py 12` on slice 0 (`TT_VISIBLE_DEVICES=0..7`, t3k mesh descriptor).

## Session 12 — 2026-10-14 (after Galaxy power cycle)

**Status**: all checks below pass, including on slice 0 once TT_MM_THROTTLE_PERF=3 is set (the hang was di/dt, see below).
**Block Hash**: fc756c8fe30

- Demo fix verified on device: 6/6 cases at ~20 tok/s; 384-frame video (S=34455) + 2048-token budget gives a coherent 562-token answer
  (HF also coherent); traced decode default + server warm-up before trace capture (first-occurrence video shape after capture hung).
- Clean latency pass (1 server, sequential): 3.88 s https / 2.31 s local-file per request, 97/100; HF baseline 3.61 s, March T3K 10.27 s.
  Decode 19-20 tok/s (March T3K ~35). See verification/latency_comparison.md.
- 30-video comparison vs HF 4.57.1: 6 identical, ~18 phrasing-only, ~6 detail differences; spot checks (bear video: TT right on 3 cubs).
  Teacher-forced 95.8% mean (diagnostic only).
- tt-inference-server: Molmo2 specs + launchers; run.py end to end: per-slice 97/100; single-instance DP=4: 97/100 after fixing slot
  routing, slot_remap and per-row multimodal kwargs in generator_vllm.
- **Hang resolved (revises Session 11's "hardware suspect")**: it is the Wormhole di/dt problem, not a bad chip. Evidence: the same
  single-core stall signature (LayerNorm [1,4,S,128] on one core / Matmul column, rest of the mesh waiting in all-gather) later appeared
  on Device 23 of the whole-Galaxy DP=4 instance (device timeout in a 23k-token 20-image prefill, ~10 heavy prefills in), and ttnn's own
  compute_throttle_utils documents di/dt hangs for matmuls on >48 cores on WH_B0 (other Galaxy specs set TT_MM_THROTTLE_PERF).
  Slice 0 stress (25k prefill x12): unthrottled hangs in 0-5 iterations (7/9 runs), level 1 hung on the first prefill, level 3 12/12,
  level 5 12/12. DP=4 instance with level 3: suite 97/100 (245 s vs 219 s unthrottled, +12%), 24/24 heavy requests, 0 timeouts.
  Default is now TT_MM_THROTTLE_PERF=3 in the specs and serve/launch_4x_galaxy_t3k.sh.

## Session 1 — 2026-04-29

**Status**: Architecture phase complete (image + video)
**PCC**: N/A (no TTNN blocks implemented yet)
**Block Hash**: N/A

### Work Done
- Examined all config/processing/modeling files for `allenai/Molmo2-8B`
- Read full `modeling_molmo2.py` (1764 lines), `image_processing_molmo2.py`, `processing_molmo2.py`, `video_processing_molmo2.py`
- Analyzed full weight map (706 tensors): model.transformer (291), model.vision_backbone (414), lm_head (1)
- Confirmed bidirectional masking mechanism for image tokens at prefill (token_type_ids or_mask)
- Documented complete video inference path (384 frames, pooling_size=[3,3], no multi-crop)
- Updated `models/demos/molmo2/ARCHITECTURE.md` with all implementation details

### Key Findings
1. **Masking CONFIRMED**: image+video tokens get bidirectional attention at prefill via `token_type_ids` or_mask. All 8 IMAGE_TOKENS (not just patches) are marked as type_id=1. Different from Qwen3-VL.
2. **Fused weights**: att_proj (Q+K+V, split 4096|1024|1024); ff_proj (gate+up but REVERSED: first half=value, second half=gate, unlike Llama)
3. **QK-norm after head reshape**: q_norm/k_norm shape=[128], applied after `view(B,S,n_heads,head_dim)`
4. **ViT only runs 25 of 27 layers**: Features from layers 18 and 24 concatenated → 2304-dim pooling input
5. **image_pooling_2d wq/wk/wv take 2304-dim input** (2× ViT hidden), not 1152
6. **Pooling query = masked mean of patches** (dynamic, not learned)
7. **Image features ADDED to embeddings** at image_patch_id positions (not replacing)
8. **Video: no multi-crop**, pooling_size=[3,3] → 81 tokens/frame, max 384 frames
9. **Preprocessing**: Use HF processors as-is for both image and video

### T3K Parallelization Decisions
- Text decoder: full tensor-parallel (8-way), ShardTensor2dMesh(dims=(None,-1))
- ViT: replicated weights + data-parallel input (ShardTensorToMesh dim=0); AllGather after ViT
- Pooling + projector: replicated (Phase 1); projector can move to tensor-parallel later
- Embedding + LM head: replicated
- Total weights/device: ~5.2GB; KV cache at max seq: ~679MB; headroom ~5.9GB

### Bottleneck Summary (video, S=36864)
- #1: Text SDPA ~1,061ms (compute, O(S²)) — use flash-attn, practical ISL lower
- #2: ViT fp32 attention — verify HiFi4 TTNN kernel
- #3: Pooling gather (irregular indices) — precompute flat gather index in processor
- #4: image feature scatter-add — reformulate as padded-dense add
- #5: Text MLP ~425ms, CCL ~238ms, ViT prefill ~313ms
- Decode: ~4ms/token, ~250 tok/s at full KV cache

## Session 2 — 2026-04-29

**Status**: Reference phase complete (all blocks verified PCC=1.0)
**PCC**: 1.000000 on all blocks (RMSNorm, dual embedding, text attention, text MLP, full decoder block, prefill mask invariants, ViT encode, vision adapter, full prefill forward, decoder block 0 with image embeddings)
**Block Hash**: N/A (reference PyTorch, not TTNN)

### Work Done
- Created `models/demos/molmo2/reference/functional.py` — standalone PyTorch reference for all blocks
- Created `models/demos/molmo2/reference/test_functional.py` — 6-step verification against HF
- Saved goldens to `models/demos/molmo2/reference/golden/` (6 files)
- Installed `transformers==4.57.1` (required for TransformersKwargs, make_batched_metadata)

### Key Bugs Found and Fixed During Reference Phase
1. **wo bias missing**: `ViTMultiHeadDotProductAttention.wo` uses `nn.Linear` default `bias=True`, but `load_vit_resblock_weights` and `vit_attention` were missing `wo_bias`. Fixed in both `functional.py` and the test.
2. **ViT capture layer order reversed**: HF `Molmo2VisionBackbone.encode_image` uses `vit_layers = [-3, -9]` → `[24, 18]` and concatenates in that order (24 first). Reference was using `sorted([18, 24]) = [18, 24]` — wrong order. Fixed `capture_layers` default to `(24, 18)`.
3. **Processor chat template format**: HF processor requires `apply_chat_template` with `{"role": "user", "content": [{"type": "image"}, ...]}` format, not raw `USER: <image>` strings.
4. **Text attention test bug**: `text_attention` takes pre-normed input but test was passing unnormed `hidden` — fixed to pass `rmsnorm(hidden, attn_norm_weight)`.

### Architecture Confirmations
- `ff_proj` gate layout confirmed: `x, gate = ff_proj_out.chunk(2)` → first half=value, second half=gate. `silu(gate) * value`.
- QK-norm applied after `[B, S, n_heads, head_dim]` reshape — per-head normalization over head_dim.
- Token count: 9 crops for logo image → 1316 image patch tokens, 1347 total sequence length.
- End-to-end generation produces coherent text ("The image features a logo for Molmo 2...").

### Golden Files Saved
- `reference/golden/e2e_generation.pt` — input_ids, output_ids, generated text
- `reference/golden/decoder_block0.pt` — text block 0 input/output (text-only)
- `reference/golden/prefill_mask.pt` — token_type_ids, combined causal+bidir mask
- `reference/golden/vit_encode.pt` — pixel_values, ref + HF ViT features [2 crops, 729, 2304]
- `reference/golden/vision_adapter.pt` — image_features_4d, ref + HF projector output [N_valid, 4096]
- `reference/golden/full_prefill.pt` — full forward inputs/outputs (last_hidden, image_features, logits)
- `reference/golden/block0_with_image.pt` — decoder block 0 with image embeddings injected

### Next Steps
- **ALL 5 PCC TESTS PASS** on T3K (8 devices, 9.05s total)
- Next: run full demo pipeline (text/image/video) end-to-end on T3K
- Phase 2: add tensor-parallel column-row MLP + Ring CCL

## Session 4 — 2026-04-29

**Status**: TTNN tests running — 3/5 PASS, 2 in progress after KV-dtype + SDPA-dtype fixes
**PCC**: prefill_mask=PASS, MLP=0.999993 ✓, ViT=0.997602 ✓, attention=0.9777 (bfloat8_b fix pending)
**Block Hash**: N/A

### Work Done
- Fixed pytest fixture scope mismatch: moved `mesh_device` to module-scoped with `ttnn.set_fabric_config(FABRIC_1D)`
- Fixed KV cache dtype: initialized as `bfloat8_b` to match typecast fill tensors
- Fixed SDPA dtype: removed bfloat8_b cast of q/k/v before SDPA (use bfloat16 for Phase 1 accuracy)
- Created qwen3_vl-style pytest demo: `models/demos/molmo2/demo/demo.py`
  - text-only, image+text, video+text, parametrized test cases
  - sample prompt JSONs in `demo/sample_inputs/`
  - warm-up prefill on first batch, greedy decode loop, per-iteration throughput logging
- Created verification script: `models/demos/molmo2/verification/test_video_verification.py`
  - 10/10 inference PASS against HF CPU reference
  - 8/10 exact frame-count match, 2/10 WARN±1 (known append-last-frame version delta)

### Key TTNN Bugs Found
1. **Fixture scope mismatch**: module-scoped fixtures (state_dict, molmo2_cfg) depended on function-scoped mesh_device. Fixed: define module-scoped mesh_device in test file.
2. **KV cache dtype mismatch**: cache initialized as bfloat16 but fill tensors were bfloat8_b. Fixed: initialize cache as bfloat8_b.
3. **SDPA precision loss**: casting q/k/v to bfloat8_b before SDPA gave PCC=0.977 on short sequences. Fixed: use bfloat16 for SDPA, only cast to bfloat8_b for KV cache storage.
4. **Wrong RoPE op**: used `rotary_embedding_llama` (interleaved-pair LLaMA-style) but Molmo2 uses `rotate_half` (HF-style, concatenated halves). PCC was 0.886 for RoPE alone. Fixed: use `ttnn.experimental.rotary_embedding` (HF-style) with cos/sin in [c0,...,c63,c0,...,c63] concatenated-halves format produced by `get_rot_mats_hf`.
5. **rotary_embedding pads seq dim to tile=32**: output is [1,n_heads,32,head_dim] even for S=6. Fixed: slice output to `[:,:,:seq_len,:]` after RoPE. Pass full [1,1,max_seq,head_dim] cos/sin matrix (not sliced to S) so rotary_embedding can apply correct positions.

## Session 3 — 2026-04-29

**Status**: TTNN implementation complete (all blocks, Phase 1 correctness mode)
**PCC**: Not yet run (requires device)
**Block Hash**: N/A

### Work Done
- Created `models/demos/molmo2/tt/` directory with 9 implementation files:
  - `model_config.py` — Molmo2Config dataclass with T3K defaults (num_devices=8, cluster_shape=[1,8])
  - `attention.py` — TtMolmo2TextAttention: fused att_proj + QK-norm after head split + column-parallel QKV + AllGather + replicated wo
  - `mlp.py` — TtMolmo2TextMLP: fused ff_proj (gate=second half, value=first half) + SwiGLU + replicated weights
  - `vision_block.py` — TtMolmo2ViTBlock: reuses LayerNorm from qwen3_vl; adapted VisionAttention and MLP for Molmo2 ViT weight keys
  - `vision_encoder.py` — TtMolmo2ViTEncoder: 25 ViT blocks, captures layers 18+24, bicubic pos emb interpolation
  - `image_pooling.py` — TtMolmo2ImagePooling2D: cross-attention with 2304-dim input, masked-mean query
  - `image_projector.py` — TtMolmo2ImageProjector: SwiGLU no-bias projector (w1/w3/w2)
  - `prefill_mask.py` — build_molmo2_prefill_mask: causal + image-bidirectional combined mask
  - `model.py` — TtMolmo2Model: full assembly with dual embedding, 36 decoder blocks, vision backbone, image injection
- Created `models/demos/molmo2/tests/test_tt_text_decoder.py` with T3K-parameterized tests for: prefill mask, attention, MLP, full decoder block, ViT encoder

### Key Design Decisions
1. **Phase 1 T3K layout**: column-parallel QKV (ShardTensor2dMesh dims=(2,3)) + AllGather + replicated wo. This avoids tt_all_reduce which is a no-op for T3K cluster_axis=1.
2. **QK-norm placement**: applied AFTER nlp_create_qkv_heads (Qwen3-style), reusing RMSNorm from models.common.rmsnorm.
3. **MLP gate/value**: ff_proj[:12288]=value→w3, ff_proj[12288:]=gate→w1. Standard SwiGLU formula then works correctly.
4. **ViT**: directly reuses LayerNorm from qwen3_vl/tt/vision_layernorm.py; adapts VisionAttention pattern for Molmo2 weight keys.
5. **Image pooling**: cross-attention with 2304-dim input (two-layer ViT concat), head_dim=72 padded to 96.
6. **AllGather**: uses ttnn.all_gather (not tt_ccl) for T3K — works without fabric semaphore setup.

### Next Steps (Session 3)
- Run PCC tests on T3K
- Run full demo pipeline end-to-end

## Session 5 — 2026-04-29

**Status**: End-to-end TTNN video inference PASS — test.jsonl test 1 matches HF reference
**PCC**: 8/8 unit tests PASS — attention=0.999832, MLP=0.999974, decoder=0.999907, ViT=0.999270, projector=0.999775, pooling+proj=0.999807, decoder+image=0.999999
**Block Hash**: COMPLETE

### Work Done
- Fixed TTNN text attention decode mode: HEIGHT_SHARDED `nlp_concat_heads_decode` gave global [1,1,1,128] shape (distributes heads over cores). Solution: full CPU reference decode using block weights pulled from device.
- Fixed CPU decode weight reconstruction (3 bugs):
  1. **wqkv de-interleaving**: shards are in [Q₀,K₀,V₀,Q₁,K₁,V₁,...] order; split naively as [Q_all||K_all||V_all] was wrong. Fixed: re-extract Q/K/V columns per device, then cat → [6144,4096] att_proj.
  2. **wo orientation**: TTNN stores `attn_out.weight.T`; F.linear needs `attn_out.weight`. Fixed: `wo_cpu.T` in block_weights dict.
  3. **MLP replicated concat**: cat 8 device copies along dim=-1 gives [4096,98304] (8× features). Fixed: use `ttnn.get_device_tensors(ffn.w1)[0]` only.
- Weight cache at `/tmp/molmo2_weight_cache/` (634 `.tensorbin` files): model loads in 18-20s vs ~145s cold.
- End-to-end test: `test.jsonl` test 1 (video QA, 30 frames, 2701 tokens).
  - TTNN prefill + CPU decode: `'B.'` ✓ matches HF reference `'B.'`
  - Total generate time: 88.95s (includes KV cache + block weight CPU pull)
- Demo: `models/demos/molmo2/demo/demo.py` — pytest-based, text/image/video, module-scoped fixtures

### Key Bugs Fixed This Session
- HEIGHT_SHARDED decode: full CPU decode path bypasses TTNN decode mode entirely
- wqkv de-interleaving: reconstruct [Q||K||V] from device-interleaved shards
- wo orientation: `wo_cpu.T` for F.linear
- MLP replicated: device 0 only for REPLICATED weights

### Final Status
- All 8 TTNN unit tests: PASS
- Video inference test 1: ✓ MATCH (`B.` == `B.`)
- Weight cache working (634 files, 18s load)

### Next Steps

## Session 6 — 2026-04-30

**Status**: 105-video back-to-back suite complete — 98/100 decodable tests PASS
**PCC**: All 8 unit tests PASS; TP MLP PCC=0.999946 RMS=0.993
**Block Hash**: COMPLETE

### Work Done
- Decode trace for fast decode (~8s/test from ~88s)
- Prefill bucketing (power-of-2 padding, SDPA partial-tile hang fix)
- Fixed 3 DRAM OOM: ViT 8-crop batching, eager LM head slice, adaptive SDPA
- Ran all 105 video tests back-to-back (single warm-up, trace reused)
- Implemented TP MLP with trace-safe AllReduce:
  - Column-parallel w1/w3: ShardTensor2dMesh(dims=(-2,-1))
  - Row-parallel w2: ShardTensor2dMesh(dims=(-1,-2))
  - AllReduce = reduce_scatter_minimal_async + all_gather_async (CCL async, trace-safe)
  - Root cause of broken AllGather: ring-ordering mismatch on T3K
  - Root cause of broken all_reduce: ttnn.all_reduce not replay-safe in decode traces

### Final Results (105-video suite, T3K)
- **98/100 PASS** (5 corrupt .webm excluded, same on GPU ref)
- Replicated MLP baseline: 91/100 = 91%
- **TP MLP reduce_scatter+all_gather: 98/100 = 98%** (+7%)

| MLP variant | PASS/100 | Notes |
|-------------|----------|-------|
| Replicated bf8b | 91% | Phase 1 baseline |
| TP AllGather | 87-84% | Ring-ordering mismatch |
| **TP reduce_scatter+all_gather** | **98%** | Correct, trace-safe |

## Session 7 — 2026-05-01

**Status**: tt-inference-server integration complete — 98/100 accuracy via OpenAI API
**PCC**: All 8 unit tests PASS
- test_prefill_mask: PASS (structural)
- test_text_attention_pcc: 0.999832
- test_text_mlp_pcc: 0.999970
- test_decoder_block_pcc: 0.999905
- test_vit_encoder_pcc: 0.999270
- test_image_projector_pcc: 0.999775
- test_vision_adapter_pcc: 0.999807
- test_decoder_block_with_image_pcc: 0.999999
**Block Hash**: COMPLETE

### Work Done

#### tt-inference-server vLLM Plugin Integration
- `generator_vllm.py`: TTMolmo2ForConditionalGeneration vLLM plugin
  - `initialize_vllm_model`: loads HF weights + creates TtMolmo2Model
  - `prefill_forward` / `decode_forward`: server inference API
  - `decode_forward` uses `forward_decode_step` (TTNN, no trace)
  - `allocate_molmo2_kv_cache`: bfloat16 KV cache
- `tt_vllm_plugin/__init__.py`: Registers TTMolmo2ForConditionalGeneration
- `tt_platform.py`: Renamed from platform.py (stdlib conflict)
- `tt_worker.py`, `tt_model_runner.py`, `ascend_scheduler.py`: vLLM V1 API compat
- `workflows/model_spec.py`: Molmo2-8B DeviceModelSpec with Molmo2VideoBackend

#### Key Fix: Frame Markers via video_input_ids
**Root cause of 68% → 95% improvement**: vLLM's PromptReplacement by default inserts
only N_pooled image_patch_id tokens (no frame markers `<im_start>`/`<im_end>`), giving
S=2481 vs demo's S=2701. Frame markers provide temporal structure for the SDPA attention
mask; without them, accuracy drops to 68%.

**Fix**: In `molmo2.py` (_call_hf_processor), store the full HF video token sequence
(including frame markers) as `video_input_ids` in the BatchFeature. The PromptReplacement
uses this sequence as the replacement content, giving S=2701 matching the demo exactly.

#### token_type_ids: Frame Markers Fix
Include <im_start> (151936) and <im_end> (151937) in token_type_ids reconstruction.
The HF processor marks all three as type=1 (N_frames × 83 positions). Without markers,
the 2 frame boundary tokens per frame had causal-only attention — each marker could not
see the patches of the same frame ahead of it, breaking the bidirectional image attention
the model was trained with. Adding them: 98/100 (+3 from 95/100).

#### KV Cache Precision
Changed attention.py KV cache dtype from bfloat8_b → bfloat16 for better SDPA precision.

#### Decode Trace
Disabled decode trace in server (trace captured at first request's S doesn't scale to
other S values with different SDPA program config). Using `forward_decode_step` instead.

### Final Results (105-video suite, T3K, via tt-inference-server)

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Patches-only (no frame markers) | 68% | Default PromptReplacement path |
| **Full sequence (with frame markers)** | **98%** | video_input_ids fix |
| Session 6 demo (direct) | 98% | HF processor + CPU float32 decode |

**98/100 PASS** via tt-inference-server OpenAI API
- Average latency: 6.4s per test (5.3s prefill + 1.1s decode @ 16 tokens)
- 5 remaining failures at indices 22, 26, 55, 56, 76

## Session 8 — 2026-05-01

**Status**: Optimization — 99/100 accuracy, 11% latency improvement
**PCC**: All 8 unit tests PASS (vit_encoder 0.999167)
**Block Hash**: COMPLETE

### Optimization Attempted

**ViT QKV matmul: in0_block_w 1→4**
Load 4 input tiles (128 elements) per step instead of 1 (32 elements) for
better DRAM register reuse in the column-parallel QKV projection.
- ViT matmul time: 5806→5097µs (-12%)
- Total latency: 6.4s → 5.67s per test (-11%)
- Accuracy: 98/100 → **99/100** (+1 test)

**Reverted (documented):**
- SDPA q_chunk_size 128→384: L1 overflow for n_local_heads=2, SDPA time doubled
- all_reduce num_links 1→2: T3K device deadlock, not supported for this pattern

**Replicated-weights optimization (not attempted):**
Would eliminate 50 all_reduce calls (56ms) but OOMs for 384-frame input
([384, 16, 729, 96] activations per device exceed L1+DRAM budget).
TP layout is required for large video inputs.

### Final Results (105-video suite, T3K, via tt-inference-server)

| | Accuracy | Latency |
|---|---|---|
| Before optimization | 98/100 | 6.4s/test |
| **After optimization** | **99/100** | **5.67s/test** |

Remaining failure: idx=22 (C vs B, logit margin=0.17 — borderline)

## Session 9 — 2026-05-03/04

**Status**: Docker server integration complete
**Accuracy**: 97/100 (local server, 105-video suite, local files)

### Work Done

#### Docker Server Fixes
1. **VLLM_TARGET_DEVICE=empty** at install time — vLLM `setup.py` has no "tt" device case; `empty` reads `common.txt` deps only (no torch reinstall)
2. **tt-vllm-plugin** copied and installed in Docker image — TT platform not registered without it
3. **transformers==4.57.1** pinned — vLLM allowed 5.7.0 which removed `ROPE_INIT_FUNCTIONS['default']` used by `modeling_molmo2.py`
4. **HF_HUB_OFFLINE=1** in `run_vllm_api_server.py` — prevents downloading newer incompatible model code files
5. **HF_MODEL env var** in `generator_vllm.py` — uses local symlink instead of HF ID, works with offline mode
6. **TT_CACHE_PATH** used for TTNN weight cache — persists across container restarts

#### Server Warmup (no JIT/trace during inference)
All JIT compilation done at server bringup:
1. `warmup_all_buckets(PREFILL_BUCKETS)` — 9 buckets [128..32768] JIT'd (~13s)
2. `forward_decode_step` — decode kernel JIT'd (~1s)
3. `warmup_vision_compile()` — ViT+pooling+projector JIT'd (~15s)

#### Performance (105-video, local files, T3K)
| Metric | Value |
|---|---|
| Server latency avg | **4.57s** |
| Server latency median | **3.90s** |
| Direct TTNN inference avg | **3.87s** |
| Server overhead | **0.42s** (HTTP + vLLM) |
| JIT during inference | **0 events** |
| Accuracy | **97/100 = 97%** |
| Total 105-test time | **479s (8.0 min)** |

Timing breakdown (direct TTNN):
- Vision prep (ViT+pool+proj): 0.65s
- Text gen (prefill+decode): 3.30s

Server overhead vs direct TTNN: +0.42s only (HTTP + vLLM scheduler)

## Session 10 — 2026-05-04

**Status**: Docker server COMPLETE — 105/105 success, 97.1% accuracy
**Accuracy**: 97/105 vs HF (102/105 = 97.1% excluding 3 empty HF responses)

### Work Done

#### Critical Bug Fix: tti_padded Shape Mismatch in Vision Prefill Mask
Commit: `0f36262bf08 molmo2: fix tti_padded shape mismatch in vision prefill mask`

**Root Cause**: In `forward_prefill`, the `_S_pad_early` fix (session 9) set `pad_len = 0` for
vision inputs (since x_ttnn is already padded to `_S_pad_early` in the vision block). But the
`token_type_ids` for the prefill mask still had shape `[B, S]` (actual sequence length), while
the causal mask cache had shape `[1, 1, S_pad, S_pad]`. `ttnn.maximum(causal[8192,8192],
img_mm[4233,4233])` failed with "Invalid subtile broadcast type".

**Fix**: Use `tti_pad_len = S_pad - S` (always correct) instead of `pad_len` when building
the padded token_type_ids for mask construction. This correctly pads tti to [B, S_pad] so
img_mm and causal have matching shapes.

#### Docker 105-Video Test Results (T3K, Docker, GCS videos)
| Metric | Value |
|---|---|
| Total tests | **105/105 success** |
| Accuracy vs HF | **97/105 = 92.4% (97.1% excl. empty HF)** |
| End-to-end latency (avg) | **10.4s** (includes GCS download ~5-8s) |
| End-to-end latency (median) | **10.6s** |
| Server prefill avg (excl. test 0) | **8.4s** |
| Server prefill median | **7.3s** |
| Decode step 1 avg | **0.225s** |
| JIT during inference | **0 events** |

#### Warmup Sequence (Docker, all JIT at bringup)
1. `warmup_all_buckets(PREFILL_BUCKETS)` — 9 text-only buckets [128..32768] JIT'd
2. `forward_decode_step(0, PREFILL_BUCKETS[-1])` — decode kernel JIT'd
3. `warmup_vision_compile()` — ViT+pooling+projector JIT'd
4. Vision-integrated prefill for all 9 buckets — vision path + ttnn.add JIT'd for each bucket

Total warmup: ~15 minutes (dominated by weight loading from volume)

#### Additional Fixes (eliminating test-0 JIT overhead)
Three more fixes applied after initial Docker verification to eliminate ~22s test-0 overhead:

1. **`tti_padded` mask warmup** (`9f60265cb3d`): pass `token_type_ids` in vision-integrated
   warmup so `build_molmo2_prefill_mask` JIT (ttnn.mul/maximum/where) is compiled at bringup
   not on first real inference. Capped at S≤8192 to avoid [S,S] DRAM OOM at S=32768.

2. **CPU input_ids padding** (`b655be4ec6e`): pad `input_ids` to `_S_pad_early` on CPU before
   `ttnn.embedding` so the embedding output is already bucket-sized. Eliminates all on-device
   `ttnn.concat` for padding — the main source of per-S JIT stalls. Also changes
   `_S_pad_early = get_padded_prefill_len(S)` for S≤8192 so it always equals `padded_S`
   (bucket boundary), fixing the S≤128 edge case.

#### Final Docker 105-Video Results (zero JIT, T3K)
| Metric | Before fixes | After fixes |
|---|---|---|
| Test-0 prefill | 24s | **8.4s** |
| Avg prefill (all) | 8.4s | **4.0s** |
| E2E latency avg | 10.4s | **5.7s** |
| E2E latency median | 10.6s | **4.3s** |
| Accuracy vs HF | 97/105 | **97/105 = 97.1%** |
| JIT during inference | 0 | **0** |

### Notes
- Prefill times vary 2-10s depending on sequence length (S ranges ~1K to ~4.2K for 51-frame videos)
- E2E latency improved from 10.4s to 5.7s avg after eliminating on-device concat JIT stalls
- No JIT stalls during inference: all shapes pre-compiled at bringup
- Remaining test-0 overhead (~6s vs ~2s for test-1 at same S) is vLLM first-call initialization, not JIT

## Optimization plan (written 2026-10-15, after Session 15)

Status: decode 48.8 -> 26.5 ms/step; suite mean 1.96 s vs GPU 3.61 s (local files, TT_MM_THROTTLE_PERF=3).
Dominant bottleneck: decode = matmul DRAM bandwidth (~14 of 23.5 ms kernel time at ~110 GB/s, default interleaved
configs); prefill = ViT + pooling first, then the text decoder.

Done: vocab-parallel lm_head + row-major logits; 16-core decode RMSNorm; bf8 wqkv/wo; on-device ViT features and
bucketed pooling; fine prefill buckets.

Remaining, in order:
1. DONE (Session 16) DRAM-sharded decode matmuls (wqkv, wo, w1/w3/w2) + decode built without the di/dt throttle.
2. Row-parallel wo (currently replicated 4096x4096 on every chip).
3. CCL tuning (~4.5 ms/step) and on-device argmax/sampling for greedy decoding.
4. Text prefill (~0.49 s left): on-device causal mask (~55 ms), drop embedding-injection copy (~42 ms).
5. Chunked prefill and batch>1 paged KV (concurrency in the DP=4 server).
Outstanding checklist items: Tracy profile of every sub-block, profile xlsx (run_block_profiles.sh), formal bottleneck write-up.
Each step must keep PCC > 0.99, HF teacher-forced agreement, and the 105-suite at >= 98/100 with 0 timeouts.

## Session 16 — 2026-10-15 (optimization plan step 1)

- `tt/dram_sharded.py` (`DramShardedLinear`): decode-only copies of wqkv, wo, w1, w3, w2 stored DRAM-width-sharded
  (bf8_b), L1 width-sharded activations, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`; prefill keeps the
  interleaved weights. w1/w3 outputs and the w2 input share one sharding (no reshard). Disable: `MOLMO2_DRAM_SHARDED=0`.
- The matmul throttle `TT_MM_THROTTLE_PERF=3` (needed only for long all-core prefills) cost ~3 ms/step in decode; it is
  read at op-creation time, so the decode warm-up + trace capture run with it removed (`_decode_matmul_throttle_off`
  in `tt/model.py`; `MOLMO2_DECODE_THROTTLE=1` keeps it). Prefill is still throttled.
- Decode step (bench_decode, one 1x8 slice): 26.5 -> 24.9 ms (MLP sharded) -> 22.9 ms (+wqkv/wo; per-op kernel total
  21.8 ms, matmuls 11.8 ms, wo still ~129 us/layer = ~138 GB/s) -> 21.0 ms (throttle off in decode, 16-core norm; see
  the determinism bug below) -> **22.3 ms = 44.8 tok/s** with the final 8-core decode norm (no sharded norm: 26.5 ms).
- Validation: PCC tests (adapter, text decoder, tp_mlp) 11 passed; HF teacher-forced unchanged (text 100%, image 97%,
  video 100%, two-images 97%; 30-video suite 95.9% (was 95.8%); 34k-token long video per-100-token [0.94,0.95,0.86,0.95]
  traced).
- Server (4 slices): 105-suite 98/100, 0 errors, 54 s wall; single server sequential: mean **1.82 s** / median 1.23 /
  p90 3.83 (GPU baseline 3.61 s mean); 8/8 concurrent heavy requests (34k video 56 s, 20 images 25 s) OK, 0 timeouts.
- Next: step 2, row-parallel `wo` (wo is now the largest single matmul: 16 MB/chip replicated).

### Session 16 bug: prefill not repeatable in the server (found while testing the Docker image)
- Symptom: the same 34k-token video request gave different output on the 2nd+ request (prefill logits differed, max|diff| up
  to 3.1), on single-slice servers and on the DP=4 instance; the 1st request was always the same on every server. Short
  prompts (<5k tokens) and text were unaffected, so the 105-suite never showed it. `logprobs=true` requests also crash the
  engine (vLLM `gather_logprobs` index error; unrelated, still open).
- Bisect: pre-perf code (3626d79f6c1) repeats exactly; 7bf9523ba70 (vocab-parallel lm_head + 16-core decode RMSNorm) does
  not. New `tests/test_repeat_determinism.py` reproduces it in-process when the decode trace is captured BEFORE the first long
  prefill (the server's order; the old adapter test captured after it, so it hid the bug).
- Cause: the 16-core width-sharded decode RMSNorm (grid (4,4) or (8,2)). With the trace-first order the decode replays
  corrupt state that later prefills use; 8 cores ((4,2)) repeat exactly (6 repeats, S=34455, max|diff| 0.0), no sharded
  norm also repeats. Mechanism not pinned down further. Fix: `_NORM_GRID = (4, 2)`; env `MOLMO2_NORM_GRID=x,y` and
  `MOLMO2_DECODE_NORM_SHARDED=0` for experiments. Earlier numbers measured with the 16-core norm (1.82 s suite mean,
  21.0 ms/step) are superseded by the final ones below.
- Verified after the fix: test_repeat_determinism identical x6; 4 slice servers x4 repeats of the 34k video (300 tokens)
  identical on every server and across servers; suite 98/100 (4 servers, 55 s); HF teacher-forced unchanged (text 100%,
  image 97%, video 100%, 30-video suite 95.3%, long video [0.92, 0.97, 0.88, 0.95]).
