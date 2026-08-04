# Bring-up run report — `tencent/HunyuanVideo-1.5`

_Generated: 2026-07-08 05:20:52 UTC_

## Outcome

**Converged** after ? iteration(s).
- Run ended: bring-up complete — gate can_stop (all components graduated or fell back)

## Backend & template match

- **Backend picked:** `hf_eager universal (Video)`
- **Closest template:** `models/demos/hf_eager/demo.py`

## Placement summary

- **ON_DEVICE** (18): graduated, native ttnn, PCC verified
  - `ada_layer_norm_continuous`, `ada_layer_norm_zero`, `combined_timestep_text_proj_embeddings`, `feed_forward`, `hunyuan_video15_ada_norm`, `hunyuan_video15_by_t5_text_projection`, `hunyuan_video15_image_projection`, `hunyuan_video15_individual_token_refiner`, `hunyuan_video15_individual_token_refiner_block`, `hunyuan_video15_patch_embed`, `hunyuan_video15_rotary_pos_embed`, `hunyuan_video15_time_embedding`, `hunyuan_video15_token_refiner`, `hunyuan_video15_transformer_block`, `linear_activation`, `pix_art_alpha_text_projection`, `timestep_embedding`, `timesteps`
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run

## Module placement (all components)

| module | on device? | why | per-module pytest |
|---|---|---|---|
| `ada_layer_norm_continuous` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous` |
| `ada_layer_norm_zero` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero` |
| `combined_timestep_text_proj_embeddings` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_combined_timestep_text_proj_embeddings.py::test_combined_timestep_text_proj_embeddings` |
| `feed_forward` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_feed_forward.py::test_feed_forward` |
| `hunyuan_video15_ada_norm` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_ada_norm.py::test_hunyuan_video15_ada_norm` |
| `hunyuan_video15_by_t5_text_projection` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_by_t5_text_projection.py::test_hunyuan_video15_by_t5_text_projection` |
| `hunyuan_video15_image_projection` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_image_projection.py::test_hunyuan_video15_image_projection` |
| `hunyuan_video15_individual_token_refiner` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner.py::test_hunyuan_video15_individual_token_refiner` |
| `hunyuan_video15_individual_token_refiner_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner_block.py::test_hunyuan_video15_individual_token_refiner_block` |
| `hunyuan_video15_patch_embed` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_patch_embed.py::test_hunyuan_video15_patch_embed` |
| `hunyuan_video15_rotary_pos_embed` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_rotary_pos_embed.py::test_hunyuan_video15_rotary_pos_embed` |
| `hunyuan_video15_time_embedding` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_time_embedding.py::test_hunyuan_video15_time_embedding` |
| `hunyuan_video15_token_refiner` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_token_refiner.py::test_hunyuan_video15_token_refiner` |
| `hunyuan_video15_transformer_block` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_transformer_block.py::test_hunyuan_video15_transformer_block` |
| `linear_activation` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_linear_activation.py::test_linear_activation` |
| `pix_art_alpha_text_projection` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_pix_art_alpha_text_projection.py::test_pix_art_alpha_text_projection` |
| `timestep_embedding` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timestep_embedding.py::test_timestep_embedding` |
| `timesteps` | ✅ yes | graduated — native ttnn, PCC-verified | `models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timesteps.py::test_timesteps` |

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_continuous.py::test_ada_layer_norm_continuous -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_ada_layer_norm_zero.py::test_ada_layer_norm_zero -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_combined_timestep_text_proj_embeddings.py::test_combined_timestep_text_proj_embeddings -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_feed_forward.py::test_feed_forward -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_ada_norm.py::test_hunyuan_video15_ada_norm -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_by_t5_text_projection.py::test_hunyuan_video15_by_t5_text_projection -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_image_projection.py::test_hunyuan_video15_image_projection -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner.py::test_hunyuan_video15_individual_token_refiner -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_individual_token_refiner_block.py::test_hunyuan_video15_individual_token_refiner_block -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_patch_embed.py::test_hunyuan_video15_patch_embed -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_rotary_pos_embed.py::test_hunyuan_video15_rotary_pos_embed -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_time_embedding.py::test_hunyuan_video15_time_embedding -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_token_refiner.py::test_hunyuan_video15_token_refiner -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_hunyuan_video15_transformer_block.py::test_hunyuan_video15_transformer_block -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_linear_activation.py::test_linear_activation -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_pix_art_alpha_text_projection.py::test_pix_art_alpha_text_projection -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timestep_embedding.py::test_timestep_embedding -svv
python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/pcc/test_timesteps.py::test_timesteps -svv
```

## Next steps

- **All components graduated** — wire the end-to-end pipeline:
  - `python -m scripts.tt_hw_planner emit-e2e tencent/HunyuanVideo-1.5`

<!-- BEGIN optimize -->
# Optimize (perf) — `hunyuanvideo_1_5`

_Updated live: 2026-07-23 09:29:47 UTC · 57 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — hunyuanvideo_1_5 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)

Op breakdown — device time by op class (latest profile · what to target, ranked):
op class         device_ms      %   count  bound  dominant op (shape)
---------------------------------------------------------------------------------------------------
matmul                3.71  72.1%      56   dram  MatmulDeviceOperation 32 x 8192 x 2048
host_overhead         1.26  24.6%       0   host  
datamove              0.67  13.0%     158   slow  ReshapeViewDeviceOperation
reduction             0.40   7.7%      27   slow  LayerNormDeviceOperation
eltwise               0.18   3.4%      38   slow  BinaryNgDeviceOperation
other                 0.15   2.9%      25   slow  TernaryDeviceOperation
attention             0.04   0.8%       2   slow  JointSDPADeviceOperation

Block-level timing (per-stage trace) — latest lever on ReshapeViewDeviceOperation:
  matmul (FF/QKV/out, M=32 dispatch-bound)      3.71 ms  ###################### · True  <- hottest
  datamove (ReshapeView, silu/rope launches trimmed)      0.67 ms  ####..................
  reduction (LayerNorm, width-sharded)      0.40 ms  ##....................
  eltwise (BinaryNg, bias/addcmul-fused)      0.18 ms  #.....................
  other (Ternary addcmul)      0.15 ms  #.....................
  attention (JointSDPA)      0.04 ms  ......................

op                                 grid      fidelity  dtype     shard     tt-lang   cpp       host        best ms
------------------------------------------------------------------------------------------------------------------
BinaryNgDeviceOperation            —         —         —         —         ✓win      ·try      —              5.44
BinaryNgDeviceOperation            —         —         —         —         ✓win      —         —              5.33
LayerNormDeviceOperation           ✓win      —         —         ✓win      —         —         ✓win           5.17
MatmulDeviceOperation              —         —         —         —         ·try      —         —              5.48
MatmulDeviceOperation              —         —         —         —         ·try      —         —              5.39
MatmulDeviceOperation              —         —         —         ·try      —         —         —              7.55
MatmulDeviceOperation              ·try      —         ·try      ·try      —         —         —              3.12
MatmulDeviceOperation              —         —         —         —         —         —         ·try           5.15
MatmulDeviceOperation              ·try      ·try      ·try      ·try      ·try      ·try      ✓win           5.11
ReshapeViewDeviceOperation         ·try      —         —         —         —         —         ✓win           5.14
ReshapeViewDeviceOperation         —         —         —         —         ·try      —         —              5.39
TernaryDeviceOperation             —         —         —         —         —         —         ✓win           5.15
generation_loop                    —         —         —         —         —         —         ·try           5.32
host_overhead                      —         —         —         —         —         —         ·try           5.32


Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):
op                                        lever        ms  gain vs base  result     why tried / why it won or failed
--------------------------------------------------------------------------------------------------------------------
MatmulDeviceOperation                      grid      6.44      -1.30 ms  · no gain  Hypothesis: dominant matmul profiled as grid=partial, so full core_grid should saturate idle DRAM bandwidth. Outcome: no gain (6.4316->6.4426, slightly slower) — M=32 is only 1 tile high, so widening 
MatmulDeviceOperation                     dtype      6.44      -1.29 ms  · no gain  Hypothesis: op tagged bound_by=memory (weight reads dominate at M=32), so bf8_b weights quarter the DRAM read bytes vs fp32. Outcome: NO gain (bf16 also flat, bf8_b 6.4316->6.4376). Weight-dtype is in
MatmulDeviceOperation                     dtype      6.43      -1.29 ms  · no gain  Enabled the stub's purpose-built bf16 fast path (HY_DIT_BF16 default off->on): bf16 weights+activations, HiFi2, fp32 accumulate, its own comment claims '2-4x faster matmuls + 2x less DRAM'. This is th
BinaryNgDeviceOperation                 tt-lang      6.34      -1.20 ms  ✓ win      Bias-fusion (ttnn.linear) in the profiled transformer_block stub removes ~38 standalone BinaryNg add launches from the 2-layer forward. WIN: 6.4316->6.344ms (1.36%, is_real_gain). Root cause: path is 
BinaryNgDeviceOperation                     cpp      5.44      -0.29 ms  · no gain  Authored a real C++ Metalium eltwise-add via ttnn.generic_op (in-repo binary reader/compute + unary writer kernels by FILE_PATH, single Tensix core, fp32 tiles; plain ProgramDescriptor -> SPMD replica
MatmulDeviceOperation                       cpp      6.03      -0.88 ms  · no gain  Authored a real C++ Metalium matmul via ttnn.generic_op (official multi-core output-tiles-partitioned reader + mm compute + unary writer, full core grid, fp32, SPMD across mesh) on the replicated toke
generation_loop                      structural      5.37      -0.23 ms  · no gain  none (already handled): investigated the decode/repeat_prefill signal. This is a DIFFUSION transformer, not autoregressive — there is no KV-cache. The analogous cross-step lever (cache the step-INVARI
MatmulDeviceOperation                structural      5.37      -0.23 ms  · no gain  none reducible (stable): the 8192x2048 op is the FF up-projection — DENSE (no MoE/sparsity to gather), and within a single denoise forward it is not recomputed (conditioning is cached; see generation_
host_overhead                        structural      5.37      -0.23 ms  · no gain  already implemented: the trace-capture + 2-CQ structural lever for the generation loop EXISTS in the model — denoise_trace_setup captures resident buffers, denoise_trace_step is a host-op-free fixed-s
generation_loop                    structural-decode      5.37      -0.23 ms  · no gain  none applicable (architectural mismatch + already handled): this is a DIFFUSION transformer (MMDiT), NOT autoregressive generation — there is no token-by-token decode and no KV-cache to add (every den
MatmulDeviceOperation                     dtype      5.37      -0.23 ms  · no gain  Hypothesis: dominant FF matmul tagged bound_by=memory at M=32, so bf8_b WEIGHTS (activations/accum stay fp32) should quarter the KxN weight-read bytes vs fp32 and cut DRAM traffic. Applied to the STUB
generation_loop                            host      5.37      -0.23 ms  · no gain  structural-decode rung, re-verified this session. Signal 'repeat_prefill: add KV-cache + single-token decode_step' is an ARCHITECTURAL MISCLASSIFICATION: HunyuanVideo-1.5 is a DIFFUSION MMDiT, not aut
MatmulDeviceOperation                   tt-lang      5.39      -0.24 ms  · no gain  Fusion lever (not kernel): merged the two dual-stream AdaLayerNormZero modulation matmuls into ONE C->12C matmul sharing a single silu(temb), and produced all 12 params directly as (B,1,C) to drop ~11
BinaryNgDeviceOperation                 tt-lang      5.33      -0.18 ms  ✓ win      Eltwise fusion (not a kernel): the block's AdaLN modulation (norm*(1+scale)+shift) and 4 gated residuals (h + x*gate) each ran as a multiply+add PAIR of dispatch-bound BinaryNg launches. Hypothesis: o
ReshapeViewDeviceOperation              tt-lang      5.39      -0.25 ms  · no gain  Datamove fusion (not a kernel): replaced per-QKV 3 slice + 3 head-split reshape + 3 permute with one ttnn.experimental.nlp_create_qkv_heads, output permute+reshape with nlp_concat_heads, and the RoPE 
MatmulDeviceOperation                   tt-lang      5.48      -0.33 ms  · no gain  Two fusions batched (both reverted): (A) fold FF GELU-tanh into up-proj matmul epilogue via ttnn.linear(activation='gelu_tanh'); (B) RoPE rot as a batched 4D (B,S,H,D)@(D,D) matmul to drop the flatten
MatmulDeviceOperation                   tt-lang      5.32      -0.18 ms  · no gain  Fidelity knob (reverted): dropped the fp32 path's math_fidelity HiFi4->HiFi2 (half the MAC passes) on the shared compute_config. Hypothesis: if the 69%% matmul bucket were compute-pass-bound, HiFi2 wo
BinaryNgDeviceOperation                 tt-lang      5.31      -0.17 ms  ✓ win      Reused the distilled addcmul fusion knob on the remaining composite-path glue: token_refiner block gating (h+x*gate, x2 sites) and ada_layer_norm_continuous norm_out modulation (norm*(1+scale)+shift).
host_overhead                        structural      5.32      -0.18 ms  · no gain  host_overhead structural rung: pipeline ALREADY implements the full trace-capture + 2-CQ lever (begin/end_trace_capture, execute_trace on cq0, and denoise_write_inputs overlaps the NEXT step's H2D lat
generation_loop                    structural-decode      5.32      -0.18 ms  · no gain  generation_loop 'repeat_prefill' rung is a diffusion-mis-flagged-as-AR-decode false positive: HunyuanVideo is a diffusion sampler, not an autoregressive LM. There is no KV-cache / single-token decode_
MatmulDeviceOperation                     dtype      5.32      -0.17 ms  · no gain  Hypothesis: FSM tags this FF down-proj memory-bound; weights are fp32, so bf16 (HY_DIT_BF16=1 fast path, shared block def -> reaches all layers) should halve DRAM reads. Outcome: NO GAIN 5.3244->5.318
MatmulDeviceOperation                  fidelity      5.32      -0.17 ms  · no gain  Same HY_DIT_BF16=1 flag also drops math fidelity HiFi4->HiFi2 (fp32 accumulate kept). No gain (5.3244->5.3185, noise), PCC 0.99999. Lower fidelity buys nothing on a dispatch-bound M=32 matmul — the MA
MatmulDeviceOperation                      grid      5.45      -0.30 ms  · no gain  Hypothesis: FSM tags grid=partial, so forcing the full 8x8 compute grid (core_grid on the _row_linear matmul, shared by FF down-proj + attn out-proj) should raise occupancy. Outcome: SLOWER 5.3244->5.
ReshapeViewDeviceOperation           structural      5.32      -0.18 ms  · no gain  Hypothesis: reshape is dispatch-bound, so cut its launch count via fusion — reshape the AdaLN modulation (B,6C)->(B,1,6C) ONCE and slice params directly to (B,1,C), removing ~10 (B,C)->(B,1,C) reshape
MatmulDeviceOperation                      grid      5.45      -0.30 ms  · no gain  This 32x2048x2048 op is the attention out-proj (to_out), which shares the _row_linear path I forced to the full 8x8 grid in the down-proj grid test. Same measured result: SLOWER 5.3244->5.4486. Single
generation_loop                    structural-decode         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: E RuntimeError: Invalid sharding core_grid
LayerNormDeviceOperation                  shard      5.32      -0.18 ms  · no gain  Hypothesis: memory-bound tiny-grid layernorm might win by width-sharding activations into L1 across the core row. Outcome: NON-VIABLE. Naive width-shard + default layer_norm broke PCC (22.09 — per-sha
MatmulDeviceOperation                      grid      5.32      -0.17 ms  · no gain  Hypothesis: FF net[2] matmul tagged memory-bound, so full-grid core_grid should fan N-tiles across all cores and raise aggregate DRAM read BW. Outcome: no gain 5.324->5.318 (within noise) — op is disp
MatmulDeviceOperation                     dtype      5.32      -0.18 ms  · no gain  Hypothesis: op tagged memory-bound + weights stored float32, so bf8_b weights (4x fewer DRAM bytes) should cut the dominant traffic. Outcome: no gain 5.324->5.320 (within noise) — at M=32 (1 tile) the
MatmulDeviceOperation                     shard      5.32      -0.18 ms  · no gain  Hypothesis: shard activations into L1 to cut DRAM reads on this memory-bound op. Outcome: no gain 5.324->5.321 (within noise). Structural reason: M=32 is a single tile row (no height to shard) and the
MatmulDeviceOperation                     shard      5.32      -0.18 ms  · no gain  Re-record after profile_model reset the ladder. Shard/L1-place activations on this memory-tagged FF net[2] matmul: no gain 5.324->5.321 (within noise). M=32 is one tile row (no height to shard) and th
MatmulDeviceOperation                      grid      5.60      -0.46 ms  · no gain  Hypothesis: roofline flags grid=partial as a gap, so forcing full core_grid on the block matmuls (wo/ao out-proj + FF) should occupy idle cores. Outcome: SLOWER 5.32->5.60 (-5.2%). At M=32 (1 tile row
MatmulDeviceOperation                   tt-lang      5.32      -0.17 ms  · no gain  Op-count fusion (not a kernel): concatenated the 3 QKV projection weights (to_q|to_k|to_v, and add_q|add_k|add_v, and self-attn qkv) into ONE fused matmul per stream + tile-aligned slices, collapsing 
MatmulDeviceOperation                     dtype      5.32      -0.17 ms  · no gain  knob:dtype on the 32x2048x2048 out-proj/FF matmuls: lowered ALL pipeline _lin_w projection weights fp32->bf8_b (shared def -> reaches every block instance) to quarter DRAM weight-read bytes on this me
BinaryNgDeviceOperation                 tt-lang      5.32      -0.18 ms  · no gain  Op-count fusion (reused distilled addcmul knob): the from-parts block path (pipeline.py _transformer_block_from_parts + _refiner_block1_from_parts) still ran the 4 gated residuals + 2 norm2 modulation
MatmulDeviceOperation                     dtype      5.56      -0.42 ms  · no gain  Hypothesis: to_out/to_add_out (2048x2048) tagged memory-bound, so bf8_b weights should cut DRAM reads ~4x. Outcome: reverted, SLOWER 5.32->5.56ms — at M=32 (1 tile row) the matmul is dispatch/launch-b
generation_loop                    structural-decode         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: subblock_wt=8, but subblock width must less than 4 tiles in fp32 mode when dst_full_sync_en is false (assert.hpp:104)
generation_loop                    structural-decode         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: Physical shard shape (12, 256) must be tile {32, 32} sized! (assert.hpp:104)
LayerNormDeviceOperation                   grid      5.32      -0.18 ms  · no gain  Hypothesis: norm2/norm2_context LNs run on a tiny grid; widen via a sharded program_config to occupy more cores. Outcome: NOT APPLICABLE / reverted. The LN input is a single ragged tile-row (B*L=12 ro
LayerNormDeviceOperation                   grid      5.25      -0.11 ms  ✓ win      Hypothesis: norm2/norm2_context LNs sit on a tiny default grid; widen participation. Fix: the LN input is a ragged single tile-row (B*L=12), so create_sharded_memory_config fails tile-alignment — buil
LayerNormDeviceOperation                  shard      5.19      -0.05 ms  ✓ win      Hypothesis: the _adazero modulated-norm LN is the remaining tiny-grid single-tile-row LN; apply the same manual tile-padded WIDTH_SHARDED L1 spec (_wln) as the norm2 win. Outcome: KEPT 5.252->5.1937ms
LayerNormDeviceOperation             structural      5.17      -0.03 ms  ✓ win      Structural coverage: an LN instance still ran on the tiny default grid — the once-per-forward non-affine norm_out (ada_layer_norm_continuous, s_norm_out) which my block-stub width-shard didn't reach. 
generation_loop                    structural-decode         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_THROW: Statically allocated circular buffers in program 37 clash with L1 buffers on core range [0-0 - 0-0]. L1 buffer allocated at 1024000 a
MatmulDeviceOperation                     shard      5.17      -0.03 ms  · no gain  Hypothesis: to_out/to_add_out (2048x2048) memory-bound; width-shard the weight across the full 8x8 grid into L1 so the matmul reads it from L1 not DRAM. Outcome: reverted, L1 CLASH — 'Statically alloc
ReshapeViewDeviceOperation           structural      5.32      -0.18 ms  · no gain  grid rung inapplicable (ttnn.reshape takes no program_config), so tried the STRUCTURAL lever: eliminate the two rope reshapes in _apply_rope by doing the rot(x) fixed-matrix multiply directly in 4D (t
LayerNormDeviceOperation                  shard      5.17      -0.03 ms  · no gain  Extended the affine width-shard lever to the context-path refiner LNs (individual_token_refiner._layer_norm, weight/bias). PCC 0.999991. Outcome: reverted, NO GAIN (5.1697->5.1705, within noise). The 
MatmulDeviceOperation                     dtype      5.17      -0.03 ms  · no gain  Hypothesis: prior dtype no-gain used a RUNTIME typecast; test BUILD-TIME bf8_b weight storage on the FF matmuls (largest 3.71ms bucket, tagged memory-bound) so they read ~1/4 bytes with ZERO runtime t
MatmulDeviceOperation                structural      5.15      -0.01 ms  ✓ win      Dispatch-fusion (bound_by=dispatch, so fuse launches): _row_linear did matmul then a STANDALONE add(bias) after the (single-device no-op) all_reduce. On tp=1 fold the bias into the matmul epilogue (tt
TernaryDeviceOperation               structural      5.15      -0.01 ms  ✓ win      Dispatch-fusion in _apply_rope: out = x*cos + rot*sin was mul+mul+add (3 eltwise launches); rewrite as mul + addcmul(t, rot, sin) = 2 launches. rope runs 4x/block (q,k of both streams) so ~8 launches/
ReshapeViewDeviceOperation           structural      5.14      +0.00 ms  ✓ win      datamove bucket is dispatch/launch-bound (158 reshape ops). Hypothesis: modulation params were sliced 2D then reshaped to (B,1,C) 12x/fwd; reshape fused output to 3D once and slice (B,1,C) directly. O
MatmulDeviceOperation                structural      5.15      -0.01 ms  · no gain  Hypothesis: FF applies gelu as a standalone launch after the up-proj; fuse activation into the ttnn.linear matmul epilogue (activation='gelu') to drop 2 gelu launches/fwd on the dispatch-bound path. O
MatmulDeviceOperation                     shard      3.12      +2.02 ms  · no gain  Hypothesis: memory-bound skinny matmuls (to_out/to_add_out/ff-down, M=32) are DRAM-bw bound; DRAM-width-shard the weight across all banks + L1-width-shard the activation (tt_transformers decode idiom)
generation_loop                    structural-decode         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: Physical shard shape (1, 32) must be tile {32, 32} sized! (assert.hpp:104)
MatmulDeviceOperation                     shard      7.55      -2.40 ms  · no gain  Hypothesis: AdaLN proj (C->6C) is always-small-M (batch), so DRAM-width-shard its weight single-copy at build (no OOM, no large-M) for max read bw — the one DRAM-shard target safe for real inference. 
ReshapeViewDeviceOperation                 grid      5.14      +0.00 ms  · no gain  knob:grid on a ReshapeViewDeviceOperation is not actionable: reshape is a metadata VIEW op that takes no program_config/compute-grid, so there is no full-grid knob to set (unlike a matmul/LN). The rea
ReshapeViewDeviceOperation           structural      5.10      +0.04 ms  ✓ win      Op-count fusion (dispatch-bound regime, the only lever that wins here): (a) silu(temb) was computed twice per block — once in each of the two dual-stream _adazero calls on the identical temb — hoisted
MatmulDeviceOperation                   tt-lang      5.11      +0.03 ms  · no gain  Authored a real fused-FFN tt-lang (ttl 1.0.1) kernel via generic_op: up-proj + gelu + down-proj in ONE launch, keeping the wide [32,8192] intermediate L1-resident (the ONE fusion ttnn can't express — 

Code changes — every attempt (win or fail):
===========================================

[#1] MatmulDeviceOperation · grid · no gain  -1.30 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    index 41b47828b22..7086127ee9a 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    @@ -132,8 +132,25 @@ def _norm_w(device, norm):
         return w, b, float(getattr(norm, "eps", 1e-6))
     
     
    +_FULL_CORE_GRID = {}
    +
    +
    +def _full_grid(device):
    +    """Per-device full compute grid, cached. The DiT linear matmuls have tiny M
    +    (32 = 1 tile) so ttnn's default heuristic picks a PARTIAL grid (few cores),
    +    leaving DRAM bandwidth idle. Forcing the full grid parallelizes the N-tile
    +    fan-out across every core -> the memory-bound weight reads saturate."""
    +    key = id(device)
    +    g = _FULL_CORE_GRID.get(key)
    +    if g is None:
    +        cg = device.compute_with_storage_grid_size()
    +        g = ttnn.CoreGrid(y=cg.y, x=cg.x)
    +        _FULL_CORE_GRID[key] = g
    +    return g
    +
    +
     def _linear(x, w, b, cc):
    -    y = ttnn.matmul(x, w, compute_kernel_config=cc)
    +    y = ttnn.matmul(x, w, compute_kernel_config=cc, core_grid=_full_grid(x.device()))
         if b is not None:
             y = ttnn.add(y, b)
         return y

[#2] MatmulDeviceOperation · dtype · no gain  -1.29 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    index 41b47828b22..4d27939e637 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    @@ -120,8 +120,19 @@ def _f32(device, t):
         return ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
     
     
    +# Weight dtype for the memory-bound DiT linear projections. Weights dominate DRAM
    +# traffic (tiny M=32, big KxN weights), so bf8_b quarters the weight-read bytes
    +# vs fp32. Activations/accumulation stay fp32 (fp32-dest-acc in the compute cfg).
    +_LIN_W_DTYPE = ttnn.bfloat8_b
    +
    +
     def _lin_w(device, linear):
    -    w = _f32(device, linear.weight.detach().t())
    +    w = ttnn.from_torch(
    +        linear.weight.detach().t().contiguous().float(),
    +        dtype=_LIN_W_DTYPE,
    +        layout=ttnn.TILE_LAYOUT,
    +        device=device,
    +    )
         b = _f32(device, linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
         return w, b

[#3] MatmulDeviceOperation · dtype · no gain  -1.29 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index b1550b31cb9..409d47b2a1e 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -71,7 +71,7 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         # bf16 on entry and back to fp32 on exit, so it stays fp32-in/fp32-out (drop-in
         # for the fp32 glue/sub-stubs). ~2-4x faster matmuls + ~2x less weight/activation
         # DRAM. Default OFF = original fp32 behavior. The joint SDPA is bf16 either way.
    -    _bf16 = os.environ.get("HY_DIT_BF16", "0") == "1"
    +    _bf16 = os.environ.get("HY_DIT_BF16", "1") == "1"
         wdt = ttnn.bfloat16 if _bf16 else ttnn.float32
     
         blk = torch_module

[#4] BinaryNgDeviceOperation · tt-lang · win  -1.20 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index b1550b31cb9..4d79e9de61c 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -253,10 +253,13 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             return ttnn.from_torch(t, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=device)
     
         def _linear(x, w, b):
    -        y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
    +        # Fuse bias into the matmul epilogue (ttnn.linear) instead of a separate
    +        # ttnn.add: the standalone add is its own dispatch-bound op launch, and
    +        # the block issues one per QKV/FF projection, so folding them removes
    +        # that many launches from the profiled 2-layer forward.
             if b is not None:
    -            y = ttnn.add(y, b)
    -        return y
    +            return ttnn.linear(x, w, bias=b, compute_kernel_config=compute_config)
    +        return ttnn.matmul(x, w, compute_kernel_config=compute_config)
     
         def _all_reduce(x, mesh_axis=None):
             """Megatron all-reduce for a row-parallel output: reduce_scatter + all_gather,
    @@ -294,9 +297,10 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             s = ttnn.silu(temb)
             parts = []
             for w, b in zip(ws, bs):
    -            p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
                 if b is not None:
    -                p = ttnn.add(p, b)
    +                p = ttnn.linear(s, w, bias=b, compute_kernel_config=compute_config)
    +            else:
    +                p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
                 parts.append(p)  # each (B, C)
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
             B = int(x.shape[0])
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py b/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    new file mode 100644
    index 00000000000..c57c6830670
    --- /dev/null
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    @@ -0,0 +1,198 @@
    ... (truncated, 657 more lines)

[#5] BinaryNgDeviceOperation · cpp · no gain  -0.29 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    index 1c0da6f75cc..e152e3dcd91 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    @@ -36,6 +36,65 @@ HF_MODEL_ID = "tencent/HunyuanVideo-1.5"
     
     _MAX_PERIOD = 10000
     
    +# --------------------------------------------------------------------------- #
    +# C++ Metalium kernel (ttnn.generic_op) — the terminal cpp rung for the eltwise
    +# BinaryNgDeviceOperation. Single-core interleaved fp32 tile add adapting the
    +# in-repo binary reader/compute + unary writer kernels by FILE_PATH (compute
    +# hardcodes out CB c_2). Single core keeps the NoC/CB choreography trivial ->
    +# wedge-safe. A plain ProgramDescriptor is replicated SPMD across the mesh.
    +# --------------------------------------------------------------------------- #
    +_RD_BIN = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp"
    +_CP_BIN = "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp"
    +_WR_UN = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
    +
    +
    +def _generic_add(a, b):
    +    """a + b for identical-shape fp32 TILE_LAYOUT DRAM tensors via a C++ Metalium
    +    kernel (ttnn.generic_op). One Tensix core streams all tiles."""
    +    dev = a.device()
    +    shape = [int(v) for v in a.shape]
    +    h = ((shape[-2] + 31) // 32) * 32
    +    w = ((shape[-1] + 31) // 32) * 32
    +    num_tiles = (h // 32) * (w // 32)
    +    for d in shape[:-2]:
    +        num_tiles *= int(d)
    +    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    +
    +    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    +    page = 4096  # fp32 tile = 32*32*4
    +
    +    def _cb(idx):
    +        return ttnn.CBDescriptor(
    +            total_size=2 * page,
    +            core_ranges=core,
    +            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)],
    ... (truncated, 39 more lines)

[#6] MatmulDeviceOperation · cpp · no gain  -0.88 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    index 1c0da6f75cc..24500a1c720 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    @@ -36,6 +36,74 @@ HF_MODEL_ID = "tencent/HunyuanVideo-1.5"
     
     _MAX_PERIOD = 10000
     
    +# --------------------------------------------------------------------------- #
    +# C++ Metalium kernel (ttnn.generic_op) — cpp rung for MatmulDeviceOperation
    +# (the 32x8192x2048 FF projection). Adapts the official multi-core matmul
    +# programming example (output-tiles-partitioned reader + mm compute + unary
    +# writer, out CB c_16). Embarrassingly parallel (no mcast/semaphores) so low
    +# wedge risk; plain ProgramDescriptor is SPMD-replicated across the mesh.
    +# --------------------------------------------------------------------------- #
    +_MM_PFX = "tt_metal/programming_examples/matmul/matmul_multi_core/kernels/"
    +
    +
    +def _generic_matmul(x, w, compute_config):
    +    """x @ w for fp32 TILE_LAYOUT DRAM tensors via a C++ Metalium kernel."""
    +    dev = x.device()
    +    xs, ws = [int(v) for v in x.shape], [int(v) for v in w.shape]
    +    M = 1
    +    for d in xs[:-1]:
    +        M *= d
    +    K, N = xs[-1], ws[-1]
    +    Mt, Kt, Nt = (M + 31) // 32, (K + 31) // 32, (N + 31) // 32  # tile counts (padded)
    +    num_out = Mt * Nt
    +    out = ttnn.allocate_tensor_on_device(
    +        ttnn.Shape(xs[:-1] + [N]), ttnn.float32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    +    )
    +    g = dev.compute_with_storage_grid_size()
    +    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    +    (_, cores, cg1, cg2, wpc1, wpc2) = ttnn.split_work_to_cores(all_cores, num_out)
    +    page = 4096
    +
    +    def _cb(idx):
    +        return ttnn.CBDescriptor(
    +            total_size=2 * page, core_ranges=cores,
    +            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)],
    ... (truncated, 50 more lines)

[#7] generation_loop · structural · no gain  -0.23 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 951dc2ca456..555149f9866 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -166,6 +166,32 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
             return w, b
     
    +    def lin_col_qkv(linears):
    +        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
    +        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
    +        parallelism correct, interleave the output columns per tp-device group as
    +        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
    +        own local heads of q, k and v contiguously; the forward then slices the
    +        fused output at the LOCAL inner dim."""
    +        g = tp if sharded else 1
    +        hloc = heads_total // g
    +
    +        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
    +            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)
    +
    +        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
    +        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
    +        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
    +        if all(m.bias is not None for m in linears):
    +            bcat = torch.cat(
    +                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
    +                dim=1,
    +            ).reshape(1, -1)
    +            b = f32(bcat, mesh_mapper=_mapper(-1))
    +        else:
    +            b = None
    +        return w, b
    +
         def lin_row(linear):
             """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
             replicated and is added once, by the caller, after the all-reduce."""
    @@ -214,12 +240,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
         adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)
    ... (truncated, 42 more lines)

[#8] MatmulDeviceOperation · structural · no gain  -0.23 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 951dc2ca456..555149f9866 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -166,6 +166,32 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
             return w, b
     
    +    def lin_col_qkv(linears):
    +        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
    +        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
    +        parallelism correct, interleave the output columns per tp-device group as
    +        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
    +        own local heads of q, k and v contiguously; the forward then slices the
    +        fused output at the LOCAL inner dim."""
    +        g = tp if sharded else 1
    +        hloc = heads_total // g
    +
    +        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
    +            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)
    +
    +        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
    +        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
    +        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
    +        if all(m.bias is not None for m in linears):
    +            bcat = torch.cat(
    +                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
    +                dim=1,
    +            ).reshape(1, -1)
    +            b = f32(bcat, mesh_mapper=_mapper(-1))
    +        else:
    +            b = None
    +        return w, b
    +
         def lin_row(linear):
             """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
             replicated and is added once, by the caller, after the all-reduce."""
    @@ -214,12 +240,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
         adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)
    ... (truncated, 42 more lines)

[#9] host_overhead · structural · no gain  -0.23 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 951dc2ca456..555149f9866 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -166,6 +166,32 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
             return w, b
     
    +    def lin_col_qkv(linears):
    +        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
    +        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
    +        parallelism correct, interleave the output columns per tp-device group as
    +        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
    +        own local heads of q, k and v contiguously; the forward then slices the
    +        fused output at the LOCAL inner dim."""
    +        g = tp if sharded else 1
    +        hloc = heads_total // g
    +
    +        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
    +            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)
    +
    +        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
    +        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
    +        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
    +        if all(m.bias is not None for m in linears):
    +            bcat = torch.cat(
    +                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
    +                dim=1,
    +            ).reshape(1, -1)
    +            b = f32(bcat, mesh_mapper=_mapper(-1))
    +        else:
    +            b = None
    +        return w, b
    +
         def lin_row(linear):
             """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
             replicated and is added once, by the caller, after the all-reduce."""
    @@ -214,12 +240,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
         adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)
    ... (truncated, 42 more lines)

[#10] generation_loop · structural-decode · no gain  -0.23 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 951dc2ca456..555149f9866 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -166,6 +166,32 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
             return w, b
     
    +    def lin_col_qkv(linears):
    +        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
    +        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
    +        parallelism correct, interleave the output columns per tp-device group as
    +        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
    +        own local heads of q, k and v contiguously; the forward then slices the
    +        fused output at the LOCAL inner dim."""
    +        g = tp if sharded else 1
    +        hloc = heads_total // g
    +
    +        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
    +            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)
    +
    +        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
    +        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
    +        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
    +        if all(m.bias is not None for m in linears):
    +            bcat = torch.cat(
    +                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
    +                dim=1,
    +            ).reshape(1, -1)
    +            b = f32(bcat, mesh_mapper=_mapper(-1))
    +        else:
    +            b = None
    +        return w, b
    +
         def lin_row(linear):
             """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
             replicated and is added once, by the caller, after the all-reduce."""
    @@ -214,12 +240,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
         adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)
    ... (truncated, 42 more lines)

[#11] MatmulDeviceOperation · dtype · no gain  -0.23 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 951dc2ca456..555149f9866 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -166,6 +166,32 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
             return w, b
     
    +    def lin_col_qkv(linears):
    +        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
    +        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
    +        parallelism correct, interleave the output columns per tp-device group as
    +        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
    +        own local heads of q, k and v contiguously; the forward then slices the
    +        fused output at the LOCAL inner dim."""
    +        g = tp if sharded else 1
    +        hloc = heads_total // g
    +
    +        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
    +            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)
    +
    +        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
    +        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
    +        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
    +        if all(m.bias is not None for m in linears):
    +            bcat = torch.cat(
    +                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
    +                dim=1,
    +            ).reshape(1, -1)
    +            b = f32(bcat, mesh_mapper=_mapper(-1))
    +        else:
    +            b = None
    +        return w, b
    +
         def lin_row(linear):
             """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
             replicated and is added once, by the caller, after the all-reduce."""
    @@ -214,12 +240,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
         adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)
    ... (truncated, 42 more lines)

[#12] generation_loop · host · no gain  -0.23 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 951dc2ca456..555149f9866 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -166,6 +166,32 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
             return w, b
     
    +    def lin_col_qkv(linears):
    +        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
    +        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
    +        parallelism correct, interleave the output columns per tp-device group as
    +        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
    +        own local heads of q, k and v contiguously; the forward then slices the
    +        fused output at the LOCAL inner dim."""
    +        g = tp if sharded else 1
    +        hloc = heads_total // g
    +
    +        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
    +            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)
    +
    +        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
    +        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
    +        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
    +        if all(m.bias is not None for m in linears):
    +            bcat = torch.cat(
    +                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
    +                dim=1,
    +            ).reshape(1, -1)
    +            b = f32(bcat, mesh_mapper=_mapper(-1))
    +        else:
    +            b = None
    +        return w, b
    +
         def lin_row(linear):
             """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
             replicated and is added once, by the caller, after the all-reduce."""
    @@ -214,12 +240,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
         adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)
    ... (truncated, 42 more lines)

[#13] MatmulDeviceOperation · tt-lang · no gain  -0.24 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 951dc2ca456..555149f9866 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -166,6 +166,32 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             b = f32(linear.bias.detach().reshape(1, -1), mesh_mapper=mapper) if linear.bias is not None else None
             return w, b
     
    +    def lin_col_qkv(linears):
    +        """Column-parallel FUSED q/k/v: concatenate the three projections into ONE
    +        matmul (1 launch instead of 3 on the launch-bound path). To keep column-
    +        parallelism correct, interleave the output columns per tp-device group as
    +        [q_local | k_local | v_local] so ordinary -1 sharding hands each device its
    +        own local heads of q, k and v contiguously; the forward then slices the
    +        fused output at the LOCAL inner dim."""
    +        g = tp if sharded else 1
    +        hloc = heads_total // g
    +
    +        def _grouped(t2d):  # (out=heads_total*dim_head, C) -> (g, hloc*dim_head, C)
    +            return t2d.reshape(heads_total, dim_head, -1).reshape(g, hloc * dim_head, -1)
    +
    +        wcat = torch.cat([_grouped(m.weight.detach()) for m in linears], dim=1)  # (g, 3*hloc*dim_head, C)
    +        wcat = wcat.reshape(g * 3 * hloc * dim_head, -1)  # (3*inner_total, C)
    +        w = f32(wcat.t(), mesh_mapper=_mapper(-1))
    +        if all(m.bias is not None for m in linears):
    +            bcat = torch.cat(
    +                [m.bias.detach().reshape(heads_total, dim_head).reshape(g, hloc * dim_head) for m in linears],
    +                dim=1,
    +            ).reshape(1, -1)
    +            b = f32(bcat, mesh_mapper=_mapper(-1))
    +        else:
    +            b = None
    +        return w, b
    +
         def lin_row(linear):
             """Row-parallel: shard the INPUT (contraction) dim on tp_axis; bias stays
             replicated and is added once, by the caller, after the all-reduce."""
    @@ -214,12 +240,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
         ada1_w, ada1_b, ada1_eps, C = ada_chunks(blk.norm1)
         adac_w, adac_b, adac_eps, _ = ada_chunks(blk.norm1_context)
    ... (truncated, 42 more lines)

[#14] BinaryNgDeviceOperation · tt-lang · win  -0.18 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 555149f9866..e12d24b2ed8 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -208,7 +208,16 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             L = adazero.linear
             C = int(L.out_features) // 6
             w = f32(L.weight.detach().t())  # (Cin, 6C)
    -        b = f32(L.bias.detach().reshape(1, -1)) if L.bias is not None else None  # (1, 6C)
    +        # Bake the modulation "+1" into the bias of the two SCALE params (order:
    +        # shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp -> idx 1 & 4), so
    +        # the matmul emits (1+scale) directly and the runtime add(scale,1.0) is gone
    +        # (correct for any batch; the bias is per-feature). Downstream modulation then
    +        # collapses to a single fused addcmul(shift, norm, scale).
    +        bias = L.bias.detach().clone() if L.bias is not None else torch.zeros(6 * C)
    +        bias = bias.reshape(6, C)
    +        bias[1] += 1.0
    +        bias[4] += 1.0
    +        b = f32(bias.reshape(1, -1))  # (1, 6C)
             eps = float(getattr(adazero.norm, "eps", 1e-6))
             return w, b, eps, C
     
    @@ -326,9 +335,10 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
             B = int(x.shape[0])
             nx = ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)  # no affine
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))
    +        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
             shift_r = ttnn.reshape(shift_msa, (B, 1, C))
    -        nx = ttnn.add(ttnn.multiply(nx, ttnn.add(scale_r, 1.0)), shift_r)
    +        # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    +        nx = ttnn.addcmul(shift_r, nx, scale_r)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
         def _unsq(g):
    @@ -509,16 +519,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
                 nh, ne, freqs_cis=freqs_cis, attn_bias=attn_bias, logical_n=kwargs.get("logical_n")
             )
     
    -        h = ttnn.add(h, ttnn.multiply(attn_out, _unsq(gate_msa)))
    ... (truncated, 21 more lines)

[#15] ReshapeViewDeviceOperation · tt-lang · no gain  -0.25 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 555149f9866..e12d24b2ed8 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -208,7 +208,16 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             L = adazero.linear
             C = int(L.out_features) // 6
             w = f32(L.weight.detach().t())  # (Cin, 6C)
    -        b = f32(L.bias.detach().reshape(1, -1)) if L.bias is not None else None  # (1, 6C)
    +        # Bake the modulation "+1" into the bias of the two SCALE params (order:
    +        # shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp -> idx 1 & 4), so
    +        # the matmul emits (1+scale) directly and the runtime add(scale,1.0) is gone
    +        # (correct for any batch; the bias is per-feature). Downstream modulation then
    +        # collapses to a single fused addcmul(shift, norm, scale).
    +        bias = L.bias.detach().clone() if L.bias is not None else torch.zeros(6 * C)
    +        bias = bias.reshape(6, C)
    +        bias[1] += 1.0
    +        bias[4] += 1.0
    +        b = f32(bias.reshape(1, -1))  # (1, 6C)
             eps = float(getattr(adazero.norm, "eps", 1e-6))
             return w, b, eps, C
     
    @@ -326,9 +335,10 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
             B = int(x.shape[0])
             nx = ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)  # no affine
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))
    +        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
             shift_r = ttnn.reshape(shift_msa, (B, 1, C))
    -        nx = ttnn.add(ttnn.multiply(nx, ttnn.add(scale_r, 1.0)), shift_r)
    +        # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    +        nx = ttnn.addcmul(shift_r, nx, scale_r)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
         def _unsq(g):
    @@ -509,16 +519,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
                 nh, ne, freqs_cis=freqs_cis, attn_bias=attn_bias, logical_n=kwargs.get("logical_n")
             )
     
    -        h = ttnn.add(h, ttnn.multiply(attn_out, _unsq(gate_msa)))
    ... (truncated, 21 more lines)

[#16] MatmulDeviceOperation · tt-lang · no gain  -0.33 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 555149f9866..e12d24b2ed8 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -208,7 +208,16 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             L = adazero.linear
             C = int(L.out_features) // 6
             w = f32(L.weight.detach().t())  # (Cin, 6C)
    -        b = f32(L.bias.detach().reshape(1, -1)) if L.bias is not None else None  # (1, 6C)
    +        # Bake the modulation "+1" into the bias of the two SCALE params (order:
    +        # shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp -> idx 1 & 4), so
    +        # the matmul emits (1+scale) directly and the runtime add(scale,1.0) is gone
    +        # (correct for any batch; the bias is per-feature). Downstream modulation then
    +        # collapses to a single fused addcmul(shift, norm, scale).
    +        bias = L.bias.detach().clone() if L.bias is not None else torch.zeros(6 * C)
    +        bias = bias.reshape(6, C)
    +        bias[1] += 1.0
    +        bias[4] += 1.0
    +        b = f32(bias.reshape(1, -1))  # (1, 6C)
             eps = float(getattr(adazero.norm, "eps", 1e-6))
             return w, b, eps, C
     
    @@ -326,9 +335,10 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
             B = int(x.shape[0])
             nx = ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)  # no affine
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))
    +        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
             shift_r = ttnn.reshape(shift_msa, (B, 1, C))
    -        nx = ttnn.add(ttnn.multiply(nx, ttnn.add(scale_r, 1.0)), shift_r)
    +        # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    +        nx = ttnn.addcmul(shift_r, nx, scale_r)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
         def _unsq(g):
    @@ -509,16 +519,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
                 nh, ne, freqs_cis=freqs_cis, attn_bias=attn_bias, logical_n=kwargs.get("logical_n")
             )
     
    -        h = ttnn.add(h, ttnn.multiply(attn_out, _unsq(gate_msa)))
    ... (truncated, 21 more lines)

[#17] MatmulDeviceOperation · tt-lang · no gain  -0.18 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 555149f9866..e12d24b2ed8 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -208,7 +208,16 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             L = adazero.linear
             C = int(L.out_features) // 6
             w = f32(L.weight.detach().t())  # (Cin, 6C)
    -        b = f32(L.bias.detach().reshape(1, -1)) if L.bias is not None else None  # (1, 6C)
    +        # Bake the modulation "+1" into the bias of the two SCALE params (order:
    +        # shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp -> idx 1 & 4), so
    +        # the matmul emits (1+scale) directly and the runtime add(scale,1.0) is gone
    +        # (correct for any batch; the bias is per-feature). Downstream modulation then
    +        # collapses to a single fused addcmul(shift, norm, scale).
    +        bias = L.bias.detach().clone() if L.bias is not None else torch.zeros(6 * C)
    +        bias = bias.reshape(6, C)
    +        bias[1] += 1.0
    +        bias[4] += 1.0
    +        b = f32(bias.reshape(1, -1))  # (1, 6C)
             eps = float(getattr(adazero.norm, "eps", 1e-6))
             return w, b, eps, C
     
    @@ -326,9 +335,10 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
             B = int(x.shape[0])
             nx = ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)  # no affine
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))
    +        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
             shift_r = ttnn.reshape(shift_msa, (B, 1, C))
    -        nx = ttnn.add(ttnn.multiply(nx, ttnn.add(scale_r, 1.0)), shift_r)
    +        # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    +        nx = ttnn.addcmul(shift_r, nx, scale_r)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
         def _unsq(g):
    @@ -509,16 +519,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
                 nh, ne, freqs_cis=freqs_cis, attn_bias=attn_bias, logical_n=kwargs.get("logical_n")
             )
     
    -        h = ttnn.add(h, ttnn.multiply(attn_out, _unsq(gate_msa)))
    ... (truncated, 21 more lines)

[#18] BinaryNgDeviceOperation · tt-lang · win  -0.17 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    index 9482d8f1e60..9211cd2776f 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    @@ -118,9 +118,8 @@ def build(device, torch_module):
                 x, epsilon=eps, weight=ttnn_norm_w, bias=ttnn_norm_b, compute_kernel_config=compute_config
             )
     
    -        one_plus_scale = ttnn.add(scale, 1.0)
    -        out = ttnn.multiply(norm, one_plus_scale)
    -        out = ttnn.add(out, shift)
    +        # norm*(1+scale)+shift fused into one ternary launch (was add(mul(norm,·),shift)).
    +        out = ttnn.addcmul(shift, norm, ttnn.add(scale, 1.0))
             return out
     
         return forward
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    index 1c0da6f75cc..97d94e36965 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_token_refiner.py
    @@ -249,9 +249,9 @@ def build(device, torch_module):
                 norm_h = _ln(h, blk["n1"])
                 attn_out = _attention(norm_h, blk)
                 gate_msa, gate_mlp = _ada_gate(temb, blk)
    -            h = ttnn.add(h, ttnn.multiply(attn_out, gate_msa))
    +            h = ttnn.addcmul(h, attn_out, gate_msa)  # h + attn_out*gate in one ternary launch
                 ff_out = _ff(_ln(h, blk["n2"]), blk)
    -            h = ttnn.add(h, ttnn.multiply(ff_out, gate_mlp))
    +            h = ttnn.addcmul(h, ff_out, gate_mlp)
             return h
     
         return forward

[#19] host_overhead · structural · no gain  -0.18 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py b/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    index c57c6830670..c2ae0a8f18f 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    @@ -1,125 +1,110 @@
    -# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
    -#
    -# SPDX-License-Identifier: Apache-2.0
    -
    -"""Performance (profiler) test for the ``i2v`` HunyuanVideo-1.5 TTNN pipeline.
    -
    -Runs the SAME chained TTNN i2v denoise forward as ``demo/demo_i2v.py`` (via the
    -shared ``run_demo`` entrypoint), but BOUNDED and profiler-safe: the heavy video
    -axes are trimmed, the DiT depth is capped via ``TT_PERF_LAYERS``, and every
    -dispatched ttnn op drains the device profiler so tracy's marker buffer never
    -overflows. The device forward runs IN-PROCESS (never shelled out) so tracy can
    -see every op. No PCC / correctness assertion — perf only.
    -"""
    -
    -from __future__ import annotations
    -
     import os
     import time
    -
    +import pytest
     import ttnn
     
    -# Lift the demo's build + run entrypoint straight from demo/demo_i2v.py.
    -from models.demos.hf_eager.hunyuanvideo_1_5.demo._common import build_argparser, run_demo
    +from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import build_pipeline
     
     PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
     PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
    -
     # perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
     # overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
     # is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
     os.environ.setdefault("TT_PERF_LAYERS", "2")
     
    -# Trim the HEAVY axes for a bounded, dispatch-representative pass. This is a diffusion VIDEO model, so
    ... (truncated, 829 more lines)

[#20] generation_loop · structural-decode · no gain  -0.18 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py b/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    index c57c6830670..c2ae0a8f18f 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_i2v_perf.py
    @@ -1,125 +1,110 @@
    -# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
    -#
    -# SPDX-License-Identifier: Apache-2.0
    -
    -"""Performance (profiler) test for the ``i2v`` HunyuanVideo-1.5 TTNN pipeline.
    -
    -Runs the SAME chained TTNN i2v denoise forward as ``demo/demo_i2v.py`` (via the
    -shared ``run_demo`` entrypoint), but BOUNDED and profiler-safe: the heavy video
    -axes are trimmed, the DiT depth is capped via ``TT_PERF_LAYERS``, and every
    -dispatched ttnn op drains the device profiler so tracy's marker buffer never
    -overflows. The device forward runs IN-PROCESS (never shelled out) so tracy can
    -see every op. No PCC / correctness assertion — perf only.
    -"""
    -
    -from __future__ import annotations
    -
     import os
     import time
    -
    +import pytest
     import ttnn
     
    -# Lift the demo's build + run entrypoint straight from demo/demo_i2v.py.
    -from models.demos.hf_eager.hunyuanvideo_1_5.demo._common import build_argparser, run_demo
    +from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import build_pipeline
     
     PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
     PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
    -
     # perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
     # overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
     # is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
     os.environ.setdefault("TT_PERF_LAYERS", "2")
     
    -# Trim the HEAVY axes for a bounded, dispatch-representative pass. This is a diffusion VIDEO model, so
    ... (truncated, 829 more lines)

[#33] MatmulDeviceOperation · tt-lang · no gain  -0.17 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    index 3edd50b840d..6d2fa100c24 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    @@ -126,6 +126,22 @@ def _lin_w(device, linear):
         return w, b
     
     
    +def _lin_w_cat(device, linears):
    +    """Concatenate several Linear layers that share the SAME input into ONE
    +    (in, sum_out) weight (+ bias). The QKV projections all read the same
    +    activation, so in the dispatch-bound M=32 regime running them as 3 separate
    +    matmul launches is 3x the (dominant) launch overhead; a single fused matmul
    +    + cheap tile-aligned slices collapses that to one launch."""
    +    import torch
    +
    +    w = _f32(device, torch.cat([lin.weight.detach().t() for lin in linears], dim=-1))
    +    if all(lin.bias is not None for lin in linears):
    +        b = _f32(device, torch.cat([lin.bias.detach().reshape(1, -1) for lin in linears], dim=-1))
    +    else:
    +        b = None
    +    return w, b
    +
    +
     def _norm_w(device, norm):
         w = _f32(device, norm.weight.detach().reshape(1, 1, -1)) if getattr(norm, "weight", None) is not None else None
         b = _f32(device, norm.bias.detach().reshape(1, 1, -1)) if getattr(norm, "bias", None) is not None else None
    @@ -305,12 +321,8 @@ class HunyuanVideo15Pipeline:
                 inner=int(attn.to_q.out_features),
                 dim_head=int(attn.to_q.out_features) // int(attn.heads),
                 scale=float(getattr(attn, "scale", (int(attn.to_q.out_features) // int(attn.heads)) ** -0.5)),
    -            wq=_lin_w(d, attn.to_q),
    -            wk=_lin_w(d, attn.to_k),
    -            wv=_lin_w(d, attn.to_v),
    -            awq=_lin_w(d, attn.add_q_proj),
    -            awk=_lin_w(d, attn.add_k_proj),
    -            awv=_lin_w(d, attn.add_v_proj),
    +            wqkv=_lin_w_cat(d, [attn.to_q, attn.to_k, attn.to_v]),
    +            awqkv=_lin_w_cat(d, [attn.add_q_proj, attn.add_k_proj, attn.add_v_proj]),
                 wo=_lin_w(d, attn.to_out[0]),
    ... (truncated, 63 more lines)

[#34] MatmulDeviceOperation · dtype · no gain  -0.17 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    index 3edd50b840d..010f34c4f15 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    @@ -121,7 +121,15 @@ def _f32(device, t):
     
     
     def _lin_w(device, linear):
    -    w = _f32(device, linear.weight.detach().t())
    +    # knob:dtype — store the memory-bound projection weights as bf8_b (4x fewer
    +    # DRAM bytes than fp32); activations/accumulation stay fp32 via the compute
    +    # config. Targets the memory-tagged 32x2048x2048 out-proj / FF matmuls.
    +    w = ttnn.from_torch(
    +        linear.weight.detach().t().contiguous().float(),
    +        dtype=ttnn.bfloat8_b,
    +        layout=ttnn.TILE_LAYOUT,
    +        device=device,
    +    )
         b = _f32(device, linear.bias.detach().reshape(1, -1)) if linear.bias is not None else None
         return w, b

[#35] BinaryNgDeviceOperation · tt-lang · no gain  -0.18 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    index 3edd50b840d..415b9b9e7e3 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/tt/pipeline.py
    @@ -412,14 +412,18 @@ class HunyuanVideo15Pipeline:
             nh, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.s_adazero[i](h, temb)
             ne, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.s_adazero_ctx[i](e, temb)
             attn_out, ctx_out = self._joint_attention_inline(self.block_attn[i], nh, ne, freqs_cis, attn_bias)
    -        h = ttnn.add(h, ttnn.multiply(attn_out, self._unsq(gate_msa)))
    -        e = ttnn.add(e, ttnn.multiply(ctx_out, self._unsq(c_gate_msa)))
    +        # Gated residuals + norm2 modulation as fused ternary (addcmul) launches
    +        # instead of separate multiply+add pairs — this from-parts path was still
    +        # unfused while the composite stub already uses addcmul; each fusion drops
    +        # one dispatch-bound BinaryNg launch on the hot path. addcmul(a,b,c)=a+b*c.
    +        h = ttnn.addcmul(h, attn_out, self._unsq(gate_msa))
    +        e = ttnn.addcmul(e, ctx_out, self._unsq(c_gate_msa))
             nh2 = ttnn.layer_norm(h, epsilon=self.norm2_eps[i], compute_kernel_config=cc)
    -        nh2 = ttnn.add(ttnn.multiply(nh2, ttnn.add(self._unsq(scale_mlp), 1.0)), self._unsq(shift_mlp))
    +        nh2 = ttnn.addcmul(self._unsq(shift_mlp), nh2, ttnn.add(self._unsq(scale_mlp), 1.0))
             ne2 = ttnn.layer_norm(e, epsilon=self.norm2c_eps[i], compute_kernel_config=cc)
    -        ne2 = ttnn.add(ttnn.multiply(ne2, ttnn.add(self._unsq(c_scale_mlp), 1.0)), self._unsq(c_shift_mlp))
    -        h = ttnn.add(h, ttnn.multiply(self._unsq(gate_mlp), self.s_ff[i](nh2)))
    -        e = ttnn.add(e, ttnn.multiply(self._unsq(c_gate_mlp), self.s_ff_ctx[i](ne2)))
    +        ne2 = ttnn.addcmul(self._unsq(c_shift_mlp), ne2, ttnn.add(self._unsq(c_scale_mlp), 1.0))
    +        h = ttnn.addcmul(h, self._unsq(gate_mlp), self.s_ff[i](nh2))
    +        e = ttnn.addcmul(e, self._unsq(c_gate_mlp), self.s_ff_ctx[i](ne2))
             return h, e
     
         def _refiner_block1_from_parts(self, h, temb):
    @@ -432,11 +436,11 @@ class HunyuanVideo15Pipeline:
             norm_h = ttnn.layer_norm(h, epsilon=eps1, weight=w1, bias=b1, compute_kernel_config=cc)
             attn_out = self._self_attention_inline(self.rb1_attn, norm_h)
             gate_msa, gate_mlp = self.s_ada_norm1(temb)
    -        h = ttnn.add(h, ttnn.multiply(attn_out, gate_msa))
    +        h = ttnn.addcmul(h, attn_out, gate_msa)  # fused gated residual (was mul+add)
             norm2 = ttnn.layer_norm(h, epsilon=eps2, weight=w2, bias=b2, compute_kernel_config=cc)
             ff = self.s_linact1(norm2)  # LinearActivation: proj + SiLU
             ff = _linear(ff, self.rb1_ff2_w, self.rb1_ff2_b, cc)  # net[2]: Linear
    -        h = ttnn.add(h, ttnn.multiply(ff, gate_mlp))
    +        h = ttnn.addcmul(h, ff, gate_mlp)  # fused gated residual (was mul+add)
    ... (truncated, 3 more lines)

[#40] LayerNormDeviceOperation · grid · win  -0.11 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index e12d24b2ed8..a9381fc2acc 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -496,6 +496,37 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             y = _row_linear(y, w2, b2)
             return y
     
    +    def _wln(x, eps):
    +        # grid knob: LN input is a single ragged tile-row (M_tiles=1). Build a
    +        # tile-PADDED width-sharded L1 spec by hand (create_sharded_memory_config
    +        # derives height from the ragged logical L and fails tile-alignment) so the
    +        # width dim spreads over a row of gx cores instead of the default tiny grid.
    +        B, L, Cx = (int(d) for d in x.shape)
    +        Mt = (B * L + 31) // 32
    +        Nt = Cx // 32
    +        gx = 8
    +        while gx > 1 and Nt % gx != 0:
    +            gx -= 1
    +        padded_m = Mt * 32
    +        shard_shape = [padded_m, (Nt // gx) * 32]
    +        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, 0))})
    +        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    +        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    +        xs = ttnn.to_memory_config(x, mem)
    +        bw = Nt // gx
    +        sw = min(bw, 3)  # fp32 mode requires subblock_w < 4 tiles
    +        while sw > 1 and bw % sw != 0:
    +            sw -= 1
    +        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
    +            compute_with_storage_grid_size=(gx, 1),
    +            subblock_w=sw,
    +            block_h=Mt,
    +            block_w=bw,
    +            inplace=False,
    +        )
    +        y = ttnn.layer_norm(xs, epsilon=eps, program_config=pc, compute_kernel_config=compute_config)
    +        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
    +
         def forward(
    ... (truncated, 14 more lines)

[#41] LayerNormDeviceOperation · shard · win  -0.05 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index a9381fc2acc..9ad0558d0ba 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -334,7 +334,7 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
                 ttnn.slice(p, (0, i * C), (Bp, (i + 1) * C)) for i in range(6)
             )
             B = int(x.shape[0])
    -        nx = ttnn.layer_norm(x, epsilon=eps, compute_kernel_config=compute_config)  # no affine
    +        nx = _wln(x, eps)  # no affine; width-sharded (shard knob) — same lever as norm2
             scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
             shift_r = ttnn.reshape(shift_msa, (B, 1, C))
             # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).

[#42] LayerNormDeviceOperation · structural · win  -0.03 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    index 9211cd2776f..9a107512add 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    @@ -89,6 +89,36 @@ def build(device, torch_module):
                 return t
             return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
     
    +    def _wln(x, eps):
    +        # grid/shard knob: this norm_out LN runs on a tiny default grid (single
    +        # ragged tile-row). Spread it over a width-sharded row of cores with a
    +        # tile-PADDED manual L1 spec (create_sharded_memory_config fails alignment
    +        # on the ragged logical row). Same lever proven on the transformer block.
    +        B, L, Cx = (int(d) for d in x.shape)
    +        Mt = (B * L + 31) // 32
    +        Nt = Cx // 32
    +        gx = 8
    +        while gx > 1 and Nt % gx != 0:
    +            gx -= 1
    +        shard_shape = [Mt * 32, (Nt // gx) * 32]
    +        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, 0))})
    +        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    +        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    +        xs = ttnn.to_memory_config(x, mem)
    +        bw = Nt // gx
    +        sw = min(bw, 3)  # fp32 mode requires subblock_w < 4 tiles
    +        while sw > 1 and bw % sw != 0:
    +            sw -= 1
    +        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
    +            compute_with_storage_grid_size=(gx, 1),
    +            subblock_w=sw,
    +            block_h=Mt,
    +            block_w=bw,
    +            inplace=False,
    +        )
    +        y = ttnn.layer_norm(xs, epsilon=eps, program_config=pc, compute_kernel_config=compute_config)
    +        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
    +
         def forward(x, conditioning_embedding=None, *args, **kwargs):
             if conditioning_embedding is None:
    ... (truncated, 17 more lines)

[#44] MatmulDeviceOperation · shard · no gain  -0.03 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    index 9211cd2776f..9a107512add 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    @@ -89,6 +89,36 @@ def build(device, torch_module):
                 return t
             return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
     
    +    def _wln(x, eps):
    +        # grid/shard knob: this norm_out LN runs on a tiny default grid (single
    +        # ragged tile-row). Spread it over a width-sharded row of cores with a
    +        # tile-PADDED manual L1 spec (create_sharded_memory_config fails alignment
    +        # on the ragged logical row). Same lever proven on the transformer block.
    +        B, L, Cx = (int(d) for d in x.shape)
    +        Mt = (B * L + 31) // 32
    +        Nt = Cx // 32
    +        gx = 8
    +        while gx > 1 and Nt % gx != 0:
    +            gx -= 1
    +        shard_shape = [Mt * 32, (Nt // gx) * 32]
    +        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, 0))})
    +        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    +        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    +        xs = ttnn.to_memory_config(x, mem)
    +        bw = Nt // gx
    +        sw = min(bw, 3)  # fp32 mode requires subblock_w < 4 tiles
    +        while sw > 1 and bw % sw != 0:
    +            sw -= 1
    +        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
    +            compute_with_storage_grid_size=(gx, 1),
    +            subblock_w=sw,
    +            block_h=Mt,
    +            block_w=bw,
    +            inplace=False,
    +        )
    +        y = ttnn.layer_norm(xs, epsilon=eps, program_config=pc, compute_kernel_config=compute_config)
    +        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
    +
         def forward(x, conditioning_embedding=None, *args, **kwargs):
             if conditioning_embedding is None:
    ... (truncated, 17 more lines)

[#45] ReshapeViewDeviceOperation · structural · no gain  -0.18 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    index 9211cd2776f..9a107512add 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    @@ -89,6 +89,36 @@ def build(device, torch_module):
                 return t
             return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
     
    +    def _wln(x, eps):
    +        # grid/shard knob: this norm_out LN runs on a tiny default grid (single
    +        # ragged tile-row). Spread it over a width-sharded row of cores with a
    +        # tile-PADDED manual L1 spec (create_sharded_memory_config fails alignment
    +        # on the ragged logical row). Same lever proven on the transformer block.
    +        B, L, Cx = (int(d) for d in x.shape)
    +        Mt = (B * L + 31) // 32
    +        Nt = Cx // 32
    +        gx = 8
    +        while gx > 1 and Nt % gx != 0:
    +            gx -= 1
    +        shard_shape = [Mt * 32, (Nt // gx) * 32]
    +        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, 0))})
    +        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    +        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    +        xs = ttnn.to_memory_config(x, mem)
    +        bw = Nt // gx
    +        sw = min(bw, 3)  # fp32 mode requires subblock_w < 4 tiles
    +        while sw > 1 and bw % sw != 0:
    +            sw -= 1
    +        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
    +            compute_with_storage_grid_size=(gx, 1),
    +            subblock_w=sw,
    +            block_h=Mt,
    +            block_w=bw,
    +            inplace=False,
    +        )
    +        y = ttnn.layer_norm(xs, epsilon=eps, program_config=pc, compute_kernel_config=compute_config)
    +        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
    +
         def forward(x, conditioning_embedding=None, *args, **kwargs):
             if conditioning_embedding is None:
    ... (truncated, 17 more lines)

[#46] LayerNormDeviceOperation · shard · no gain  -0.03 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    index 9211cd2776f..9a107512add 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    @@ -89,6 +89,36 @@ def build(device, torch_module):
                 return t
             return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
     
    +    def _wln(x, eps):
    +        # grid/shard knob: this norm_out LN runs on a tiny default grid (single
    +        # ragged tile-row). Spread it over a width-sharded row of cores with a
    +        # tile-PADDED manual L1 spec (create_sharded_memory_config fails alignment
    +        # on the ragged logical row). Same lever proven on the transformer block.
    +        B, L, Cx = (int(d) for d in x.shape)
    +        Mt = (B * L + 31) // 32
    +        Nt = Cx // 32
    +        gx = 8
    +        while gx > 1 and Nt % gx != 0:
    +            gx -= 1
    +        shard_shape = [Mt * 32, (Nt // gx) * 32]
    +        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, 0))})
    +        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    +        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    +        xs = ttnn.to_memory_config(x, mem)
    +        bw = Nt // gx
    +        sw = min(bw, 3)  # fp32 mode requires subblock_w < 4 tiles
    +        while sw > 1 and bw % sw != 0:
    +            sw -= 1
    +        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
    +            compute_with_storage_grid_size=(gx, 1),
    +            subblock_w=sw,
    +            block_h=Mt,
    +            block_w=bw,
    +            inplace=False,
    +        )
    +        y = ttnn.layer_norm(xs, epsilon=eps, program_config=pc, compute_kernel_config=compute_config)
    +        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
    +
         def forward(x, conditioning_embedding=None, *args, **kwargs):
             if conditioning_embedding is None:
    ... (truncated, 17 more lines)

[#47] MatmulDeviceOperation · dtype · no gain  -0.03 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    index 9211cd2776f..9a107512add 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/ada_layer_norm_continuous.py
    @@ -89,6 +89,36 @@ def build(device, torch_module):
                 return t
             return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
     
    +    def _wln(x, eps):
    +        # grid/shard knob: this norm_out LN runs on a tiny default grid (single
    +        # ragged tile-row). Spread it over a width-sharded row of cores with a
    +        # tile-PADDED manual L1 spec (create_sharded_memory_config fails alignment
    +        # on the ragged logical row). Same lever proven on the transformer block.
    +        B, L, Cx = (int(d) for d in x.shape)
    +        Mt = (B * L + 31) // 32
    +        Nt = Cx // 32
    +        gx = 8
    +        while gx > 1 and Nt % gx != 0:
    +            gx -= 1
    +        shard_shape = [Mt * 32, (Nt // gx) * 32]
    +        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, 0))})
    +        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    +        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    +        xs = ttnn.to_memory_config(x, mem)
    +        bw = Nt // gx
    +        sw = min(bw, 3)  # fp32 mode requires subblock_w < 4 tiles
    +        while sw > 1 and bw % sw != 0:
    +            sw -= 1
    +        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
    +            compute_with_storage_grid_size=(gx, 1),
    +            subblock_w=sw,
    +            block_h=Mt,
    +            block_w=bw,
    +            inplace=False,
    +        )
    +        y = ttnn.layer_norm(xs, epsilon=eps, program_config=pc, compute_kernel_config=compute_config)
    +        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
    +
         def forward(x, conditioning_embedding=None, *args, **kwargs):
             if conditioning_embedding is None:
    ... (truncated, 17 more lines)

[#48] MatmulDeviceOperation · structural · win  -0.01 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 9ad0558d0ba..377abd48289 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -311,6 +311,13 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             return ttnn.reshape(x4, (B, L, Cx))
     
         def _row_linear(x, w, b):
    +        # Single-device (tp=1): the all-reduce is a no-op, so fold the bias into the
    +        # matmul epilogue (ttnn.linear) instead of a standalone dispatch-bound add.
    +        # Removes one add launch per to_out / to_add_out / FF-down call (8/forward).
    +        if not sharded:
    +            if b is not None:
    +                return ttnn.linear(x, w, bias=b, compute_kernel_config=compute_config)
    +            return ttnn.matmul(x, w, compute_kernel_config=compute_config)
             y = ttnn.matmul(x, w, compute_kernel_config=compute_config)
             y = _all_reduce(y)
             if b is not None:

[#49] TernaryDeviceOperation · structural · win  -0.01 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 377abd48289..efd288553a8 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -362,7 +362,8 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             if cos_b.dtype != x4.dtype:  # bf16 fast path: match the fp32 freqs to activations
                 cos_b = ttnn.typecast(cos_b, x4.dtype)
                 sin_b = ttnn.typecast(sin_b, x4.dtype)
    -        return ttnn.add(ttnn.multiply(x4, cos_b), ttnn.multiply(rot4, sin_b))
    +        # x*cos + rot*sin fused: mul + addcmul (2 ops) instead of mul + mul + add (3).
    +        return ttnn.addcmul(ttnn.multiply(x4, cos_b), rot4, sin_b)
     
         def _joint_attention(nh, ne, freqs_cis=None, attn_bias=None, logical_n=None):
             """Joint (dual-stream) attention via the fused flash-attention-style

[#50] ReshapeViewDeviceOperation · structural · win  +0.00 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index efd288553a8..82680bdcb42 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -337,20 +337,22 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             else:
                 p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
             Bp = int(p.shape[0])
    +        # Reshape the fused output to 3D ONCE, so every sliced param is already
    +        # (Bp, 1, C) — the broadcast shape the downstream addcmuls need. This
    +        # removes the 6 per-param (B,C)->(B,1,C) reshapes (scale/shift here + the
    +        # _unsq calls in forward), a dispatch-bound datamove win (12 reshapes/
    +        # forward -> 2). Slicing along the last dim of the 3D tensor is the same
    +        # single op as the 2D slice was.
    +        p = ttnn.reshape(p, (Bp, 1, 6 * C))
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    -            ttnn.slice(p, (0, i * C), (Bp, (i + 1) * C)) for i in range(6)
    +            ttnn.slice(p, (0, 0, i * C), (Bp, 1, (i + 1) * C)) for i in range(6)
             )
    -        B = int(x.shape[0])
             nx = _wln(x, eps)  # no affine; width-sharded (shard knob) — same lever as norm2
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
    -        shift_r = ttnn.reshape(shift_msa, (B, 1, C))
             # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    -        nx = ttnn.addcmul(shift_r, nx, scale_r)
    +        # scale_msa already carries (1+scale): +1 baked into the AdaLN bias.
    +        nx = ttnn.addcmul(shift_msa, nx, scale_msa)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
    -    def _unsq(g):
    -        return ttnn.reshape(g, (int(g.shape[0]), 1, C))
    -
         def _apply_rope(x4, cos, sin):
             # x4: (B, S, H, D); cos/sin: (S, D). out = x*cos + rot(x)*sin, all on device.
             Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
    @@ -559,18 +561,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
     
             # Gated residual in ONE ternary launch: h + attn_out*gate (was mul+add).
    -        h = ttnn.addcmul(h, attn_out, _unsq(gate_msa))
    ... (truncated, 21 more lines)

[#51] MatmulDeviceOperation · structural · no gain  -0.01 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index efd288553a8..82680bdcb42 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -337,20 +337,22 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             else:
                 p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
             Bp = int(p.shape[0])
    +        # Reshape the fused output to 3D ONCE, so every sliced param is already
    +        # (Bp, 1, C) — the broadcast shape the downstream addcmuls need. This
    +        # removes the 6 per-param (B,C)->(B,1,C) reshapes (scale/shift here + the
    +        # _unsq calls in forward), a dispatch-bound datamove win (12 reshapes/
    +        # forward -> 2). Slicing along the last dim of the 3D tensor is the same
    +        # single op as the 2D slice was.
    +        p = ttnn.reshape(p, (Bp, 1, 6 * C))
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    -            ttnn.slice(p, (0, i * C), (Bp, (i + 1) * C)) for i in range(6)
    +            ttnn.slice(p, (0, 0, i * C), (Bp, 1, (i + 1) * C)) for i in range(6)
             )
    -        B = int(x.shape[0])
             nx = _wln(x, eps)  # no affine; width-sharded (shard knob) — same lever as norm2
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
    -        shift_r = ttnn.reshape(shift_msa, (B, 1, C))
             # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    -        nx = ttnn.addcmul(shift_r, nx, scale_r)
    +        # scale_msa already carries (1+scale): +1 baked into the AdaLN bias.
    +        nx = ttnn.addcmul(shift_msa, nx, scale_msa)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
    -    def _unsq(g):
    -        return ttnn.reshape(g, (int(g.shape[0]), 1, C))
    -
         def _apply_rope(x4, cos, sin):
             # x4: (B, S, H, D); cos/sin: (S, D). out = x*cos + rot(x)*sin, all on device.
             Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
    @@ -559,18 +561,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
     
             # Gated residual in ONE ternary launch: h + attn_out*gate (was mul+add).
    -        h = ttnn.addcmul(h, attn_out, _unsq(gate_msa))
    ... (truncated, 21 more lines)

[#52] MatmulDeviceOperation · shard · no gain  +2.02 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index efd288553a8..82680bdcb42 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -337,20 +337,22 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             else:
                 p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
             Bp = int(p.shape[0])
    +        # Reshape the fused output to 3D ONCE, so every sliced param is already
    +        # (Bp, 1, C) — the broadcast shape the downstream addcmuls need. This
    +        # removes the 6 per-param (B,C)->(B,1,C) reshapes (scale/shift here + the
    +        # _unsq calls in forward), a dispatch-bound datamove win (12 reshapes/
    +        # forward -> 2). Slicing along the last dim of the 3D tensor is the same
    +        # single op as the 2D slice was.
    +        p = ttnn.reshape(p, (Bp, 1, 6 * C))
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    -            ttnn.slice(p, (0, i * C), (Bp, (i + 1) * C)) for i in range(6)
    +            ttnn.slice(p, (0, 0, i * C), (Bp, 1, (i + 1) * C)) for i in range(6)
             )
    -        B = int(x.shape[0])
             nx = _wln(x, eps)  # no affine; width-sharded (shard knob) — same lever as norm2
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
    -        shift_r = ttnn.reshape(shift_msa, (B, 1, C))
             # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    -        nx = ttnn.addcmul(shift_r, nx, scale_r)
    +        # scale_msa already carries (1+scale): +1 baked into the AdaLN bias.
    +        nx = ttnn.addcmul(shift_msa, nx, scale_msa)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
    -    def _unsq(g):
    -        return ttnn.reshape(g, (int(g.shape[0]), 1, C))
    -
         def _apply_rope(x4, cos, sin):
             # x4: (B, S, H, D); cos/sin: (S, D). out = x*cos + rot(x)*sin, all on device.
             Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
    @@ -559,18 +561,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
     
             # Gated residual in ONE ternary launch: h + attn_out*gate (was mul+add).
    -        h = ttnn.addcmul(h, attn_out, _unsq(gate_msa))
    ... (truncated, 21 more lines)

[#54] MatmulDeviceOperation · shard · no gain  -2.40 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index efd288553a8..82680bdcb42 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -337,20 +337,22 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             else:
                 p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
             Bp = int(p.shape[0])
    +        # Reshape the fused output to 3D ONCE, so every sliced param is already
    +        # (Bp, 1, C) — the broadcast shape the downstream addcmuls need. This
    +        # removes the 6 per-param (B,C)->(B,1,C) reshapes (scale/shift here + the
    +        # _unsq calls in forward), a dispatch-bound datamove win (12 reshapes/
    +        # forward -> 2). Slicing along the last dim of the 3D tensor is the same
    +        # single op as the 2D slice was.
    +        p = ttnn.reshape(p, (Bp, 1, 6 * C))
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    -            ttnn.slice(p, (0, i * C), (Bp, (i + 1) * C)) for i in range(6)
    +            ttnn.slice(p, (0, 0, i * C), (Bp, 1, (i + 1) * C)) for i in range(6)
             )
    -        B = int(x.shape[0])
             nx = _wln(x, eps)  # no affine; width-sharded (shard knob) — same lever as norm2
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
    -        shift_r = ttnn.reshape(shift_msa, (B, 1, C))
             # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    -        nx = ttnn.addcmul(shift_r, nx, scale_r)
    +        # scale_msa already carries (1+scale): +1 baked into the AdaLN bias.
    +        nx = ttnn.addcmul(shift_msa, nx, scale_msa)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
    -    def _unsq(g):
    -        return ttnn.reshape(g, (int(g.shape[0]), 1, C))
    -
         def _apply_rope(x4, cos, sin):
             # x4: (B, S, H, D); cos/sin: (S, D). out = x*cos + rot(x)*sin, all on device.
             Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
    @@ -559,18 +561,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
     
             # Gated residual in ONE ternary launch: h + attn_out*gate (was mul+add).
    -        h = ttnn.addcmul(h, attn_out, _unsq(gate_msa))
    ... (truncated, 21 more lines)

[#55] ReshapeViewDeviceOperation · grid · no gain  +0.00 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index efd288553a8..82680bdcb42 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -337,20 +337,22 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             else:
                 p = ttnn.matmul(s, w, compute_kernel_config=compute_config)
             Bp = int(p.shape[0])
    +        # Reshape the fused output to 3D ONCE, so every sliced param is already
    +        # (Bp, 1, C) — the broadcast shape the downstream addcmuls need. This
    +        # removes the 6 per-param (B,C)->(B,1,C) reshapes (scale/shift here + the
    +        # _unsq calls in forward), a dispatch-bound datamove win (12 reshapes/
    +        # forward -> 2). Slicing along the last dim of the 3D tensor is the same
    +        # single op as the 2D slice was.
    +        p = ttnn.reshape(p, (Bp, 1, 6 * C))
             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
    -            ttnn.slice(p, (0, i * C), (Bp, (i + 1) * C)) for i in range(6)
    +            ttnn.slice(p, (0, 0, i * C), (Bp, 1, (i + 1) * C)) for i in range(6)
             )
    -        B = int(x.shape[0])
             nx = _wln(x, eps)  # no affine; width-sharded (shard knob) — same lever as norm2
    -        scale_r = ttnn.reshape(scale_msa, (B, 1, C))  # already (1+scale): +1 baked into bias
    -        shift_r = ttnn.reshape(shift_msa, (B, 1, C))
             # Fused shift + norm*scale in ONE ternary launch (was add(mul(norm,scale),shift)).
    -        nx = ttnn.addcmul(shift_r, nx, scale_r)
    +        # scale_msa already carries (1+scale): +1 baked into the AdaLN bias.
    +        nx = ttnn.addcmul(shift_msa, nx, scale_msa)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
    -    def _unsq(g):
    -        return ttnn.reshape(g, (int(g.shape[0]), 1, C))
    -
         def _apply_rope(x4, cos, sin):
             # x4: (B, S, H, D); cos/sin: (S, D). out = x*cos + rot(x)*sin, all on device.
             Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
    @@ -559,18 +561,19 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             )
     
             # Gated residual in ONE ternary launch: h + attn_out*gate (was mul+add).
    -        h = ttnn.addcmul(h, attn_out, _unsq(gate_msa))
    ... (truncated, 21 more lines)

[#56] ReshapeViewDeviceOperation · structural · win  +0.04 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 82680bdcb42..258a17748fd 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -329,8 +329,11 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             # the prior manual mean/multiply/rsqrt/multiply sequence it replaces).
             return ttnn.rms_norm(x, epsilon=rms_eps, weight=w, compute_kernel_config=compute_config)
     
    -    def _adazero(x, temb, w, b, eps):
    -        s = ttnn.silu(temb)
    +    def _adazero(x, s, w, b, eps):
    +        # `s` is the PRE-computed silu(temb): temb is identical for the hidden and
    +        # context streams (both _adazero calls share it), so silu is hoisted to the
    +        # caller and computed ONCE per block instead of twice (one fewer dispatch-
    +        # bound launch on the launch-bound path).
             # ONE fused (C -> 6C) matmul (bias in epilogue), then slice the 6 params.
             if b is not None:
                 p = ttnn.linear(s, w, bias=b, compute_kernel_config=compute_config)
    @@ -353,17 +356,24 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
             nx = ttnn.addcmul(shift_msa, nx, scale_msa)
             return nx, gate_msa, shift_mlp, scale_mlp, gate_mlp
     
    -    def _apply_rope(x4, cos, sin):
    -        # x4: (B, S, H, D); cos/sin: (S, D). out = x*cos + rot(x)*sin, all on device.
    +    def _rope_bcast(cos, sin, Sx, Dx, dtype):
    +        # Broadcast-reshape (and dtype-match) the (S,D) freqs to (1,S,1,D) ONCE per
    +        # block: rope runs for BOTH q and k with identical cos/sin, so hoisting this
    +        # out of _apply_rope removes 2 reshapes (+ up to 2 typecasts) per block on the
    +        # launch-bound path.
    +        cos_b = ttnn.reshape(cos, (1, Sx, 1, Dx))
    +        sin_b = ttnn.reshape(sin, (1, Sx, 1, Dx))
    +        if cos_b.dtype != dtype:  # bf16 fast path: match the fp32 freqs to activations
    +            cos_b = ttnn.typecast(cos_b, dtype)
    +            sin_b = ttnn.typecast(sin_b, dtype)
    +        return cos_b, sin_b
    +
    +    def _apply_rope(x4, cos_b, sin_b):
    +        # x4: (B, S, H, D); cos_b/sin_b: pre-broadcast (1,S,1,D). out = x*cos + rot(x)*sin.
             Bx, Sx, Hx, Dx = (int(d) for d in x4.shape)
             x2 = ttnn.reshape(x4, (Bx * Sx * Hx, Dx))
    ... (truncated, 34 more lines)

[#57] MatmulDeviceOperation · tt-lang · no gain  +0.03 ms
    diff --git a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    index 258a17748fd..b67dd78aeb7 100644
    --- a/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    +++ b/models/demos/hf_eager/hunyuanvideo_1_5/_stubs/hunyuan_video15_transformer_block.py
    @@ -512,6 +512,21 @@ def build(device, torch_module, ccl_manager=None, tp=1, sp=1, tp_axis=1, sp_axis
     
         def _ff(x, parts):
             w1, b1, act, w2, b2 = parts
    +        # Fused-FFN tt-lang kernel (up-proj + gelu + down-proj in ONE launch, wide
    +        # intermediate kept L1-resident): the FF down-proj is the dominant
    +        # 32x8192x2048 memory-tagged matmul, and this is the one fusion ttnn cannot
    +        # express (ttnn.linear folds only the activation into ONE matmul, not across
    +        # both). Only on the fp32 / tp=1 path and when M=B*L is tile-aligned (the
    +        # kernel needs whole 32-row m-tiles — true for the latent stream, Limg=32;
    +        # the ragged context stream falls back). Default ON; set HY15_FFN_KERNEL=0
    +        # to force the stock 3-op path.
    +        if os.environ.get("HY15_FFN_KERNEL", "1") == "1" and not sharded and not _bf16:
    +            B, L, Cx = (int(d) for d in x.shape)
    +            if (B * L) % 32 == 0:
    +                from models.demos.hf_eager.hunyuanvideo_1_5.tt.ffn_kernel import fused_ffn
    +
    +                y2 = fused_ffn(ttnn.reshape(x, (B * L, Cx)), w1, b1, w2, b2, compute_config=compute_config)
    +                return ttnn.reshape(y2, (B, L, int(w2.shape[1])))
             y = _linear(x, w1, b1)
             y = act(y)
             y = _row_linear(y, w2, b2)

Limitations / suggested manual next steps:
- 8 op(s) tried but no lever beat baseline: MatmulDeviceOperation, MatmulDeviceOperation, MatmulDeviceOperation, MatmulDeviceOperation, MatmulDeviceOperation, ReshapeViewDeviceOperation, generation_loop, host_overhead
  -> inspect the per-op device report and consider a hand-written kernel or a structural change.

Reproduce:
  trace+2CQ perf:  python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_main_perf.py::test_main_perf -svv
  demo (real input→output):  python models/demos/hf_eager/hunyuanvideo_1_5/demo/demo_i2v.py
  full-model e2e PCC:  python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_e2e_pipeline.py -svv

levels: grid -> fidelity -> dtype -> shard -> tt-lang -> cpp -> host   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->
