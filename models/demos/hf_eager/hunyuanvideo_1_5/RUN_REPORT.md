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

_Updated live: 2026-07-21 03:07:48 UTC · 17 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — hunyuanvideo_1_5 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)

Op breakdown — device time by op class (latest profile · what to target, ranked):
op class         device_ms      %   count  bound  dominant op (shape)
---------------------------------------------------------------------------------------------------
matmul                3.71  69.7%      56   dram  MatmulDeviceOperation 32 x 8192 x 2048
host_overhead         3.00  56.3%       0   host  
datamove              0.65  12.3%     140   slow  ReshapeViewDeviceOperation
reduction             0.56  10.5%      27   slow  LayerNormDeviceOperation
eltwise               0.31   5.9%      64   slow  BinaryNgDeviceOperation
other                 0.05   0.9%      16   slow  TernaryDeviceOperation
attention             0.04   0.8%       2   slow  JointSDPADeviceOperation

Block-level timing (per-stage trace) — latest lever on generation_loop:
  matmul (FF/QKV/out, M=32 dispatch-bound)      3.72 ms  ###################### · True  <- hottest
  host_overhead (denoise glue, overlapped under trace+2CQ)      2.38 ms  ##############........
  datamove (ReshapeView x140)      0.65 ms  ####..................
  reduction (LayerNorm x27, fused)      0.56 ms  ###...................
  eltwise (BinaryNg x104, bias-fused)      0.40 ms  ##....................
  attention (JointSDPA)      0.04 ms  ......................

op                                    grid   dtype tt-lang     cpp    host   best ms
------------------------------------------------------------------------------------
BinaryNgDeviceOperation                  —       —    ✓win    ·try       —      5.44
BinaryNgDeviceOperation                  —       —    ✓win       —       —      5.33
MatmulDeviceOperation                    —       —    ·try       —       —      5.48
MatmulDeviceOperation                    —       —    ·try       —       —      5.39
MatmulDeviceOperation                 ·try    ·try       —       —       —      6.43
MatmulDeviceOperation                    —    ·try    ·try    ·try    ·try      5.32
ReshapeViewDeviceOperation               —       —    ·try       —       —      5.39
generation_loop                          —       —       —       —    ·try      5.37
host_overhead                            —       —       —       —    ·try      5.37


Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):
op                                      lever        ms  gain vs base  result     why tried / why it won or failed
------------------------------------------------------------------------------------------------------------------
MatmulDeviceOperation                    grid      6.44      -1.12 ms  · no gain  Hypothesis: dominant matmul profiled as grid=partial, so full core_grid should saturate idle DRAM bandwidth. Outcome: no gain (6.4316->6.4426, slightly slower) — M=32 is only 1 tile high, so widening 
MatmulDeviceOperation                   dtype      6.44      -1.11 ms  · no gain  Hypothesis: op tagged bound_by=memory (weight reads dominate at M=32), so bf8_b weights quarter the DRAM read bytes vs fp32. Outcome: NO gain (bf16 also flat, bf8_b 6.4316->6.4376). Weight-dtype is in
MatmulDeviceOperation                   dtype      6.43      -1.10 ms  · no gain  Enabled the stub's purpose-built bf16 fast path (HY_DIT_BF16 default off->on): bf16 weights+activations, HiFi2, fp32 accumulate, its own comment claims '2-4x faster matmuls + 2x less DRAM'. This is th
BinaryNgDeviceOperation               tt-lang      6.34      -1.02 ms  ✓ win      Bias-fusion (ttnn.linear) in the profiled transformer_block stub removes ~38 standalone BinaryNg add launches from the 2-layer forward. WIN: 6.4316->6.344ms (1.36%, is_real_gain). Root cause: path is 
BinaryNgDeviceOperation                   cpp      5.44      -0.11 ms  · no gain  Authored a real C++ Metalium eltwise-add via ttnn.generic_op (in-repo binary reader/compute + unary writer kernels by FILE_PATH, single Tensix core, fp32 tiles; plain ProgramDescriptor -> SPMD replica
MatmulDeviceOperation                     cpp      6.03      -0.70 ms  · no gain  Authored a real C++ Metalium matmul via ttnn.generic_op (official multi-core output-tiles-partitioned reader + mm compute + unary writer, full core grid, fp32, SPMD across mesh) on the replicated toke
generation_loop                          host      5.37      -0.05 ms  · no gain  none (already handled): investigated the decode/repeat_prefill signal. This is a DIFFUSION transformer, not autoregressive — there is no KV-cache. The analogous cross-step lever (cache the step-INVARI
MatmulDeviceOperation                    host      5.37      -0.05 ms  · no gain  none reducible (stable): the 8192x2048 op is the FF up-projection — DENSE (no MoE/sparsity to gather), and within a single denoise forward it is not recomputed (conditioning is cached; see generation_
host_overhead                            host      5.37      -0.05 ms  · no gain  already implemented: the trace-capture + 2-CQ structural lever for the generation loop EXISTS in the model — denoise_trace_setup captures resident buffers, denoise_trace_step is a host-op-free fixed-s
generation_loop                          host      5.37      -0.05 ms  · no gain  none applicable (architectural mismatch + already handled): this is a DIFFUSION transformer (MMDiT), NOT autoregressive generation — there is no token-by-token decode and no KV-cache to add (every den
MatmulDeviceOperation                   dtype      5.37      -0.05 ms  · no gain  Hypothesis: dominant FF matmul tagged bound_by=memory at M=32, so bf8_b WEIGHTS (activations/accum stay fp32) should quarter the KxN weight-read bytes vs fp32 and cut DRAM traffic. Applied to the STUB
generation_loop                          host      5.37      -0.05 ms  · no gain  structural-decode rung, re-verified this session. Signal 'repeat_prefill: add KV-cache + single-token decode_step' is an ARCHITECTURAL MISCLASSIFICATION: HunyuanVideo-1.5 is a DIFFUSION MMDiT, not aut
MatmulDeviceOperation                 tt-lang      5.39      -0.06 ms  · no gain  Fusion lever (not kernel): merged the two dual-stream AdaLayerNormZero modulation matmuls into ONE C->12C matmul sharing a single silu(temb), and produced all 12 params directly as (B,1,C) to drop ~11
BinaryNgDeviceOperation               tt-lang      5.33      +0.00 ms  ✓ win      Eltwise fusion (not a kernel): the block's AdaLN modulation (norm*(1+scale)+shift) and 4 gated residuals (h + x*gate) each ran as a multiply+add PAIR of dispatch-bound BinaryNg launches. Hypothesis: o
ReshapeViewDeviceOperation            tt-lang      5.39      -0.07 ms  · no gain  Datamove fusion (not a kernel): replaced per-QKV 3 slice + 3 head-split reshape + 3 permute with one ttnn.experimental.nlp_create_qkv_heads, output permute+reshape with nlp_concat_heads, and the RoPE 
MatmulDeviceOperation                 tt-lang      5.48      -0.15 ms  · no gain  Two fusions batched (both reverted): (A) fold FF GELU-tanh into up-proj matmul epilogue via ttnn.linear(activation='gelu_tanh'); (B) RoPE rot as a batched 4D (B,S,H,D)@(D,D) matmul to drop the flatten
MatmulDeviceOperation                 tt-lang      5.32      +0.00 ms  · no gain  Fidelity knob (reverted): dropped the fp32 path's math_fidelity HiFi4->HiFi2 (half the MAC passes) on the shared compute_config. Hypothesis: if the 69%% matmul bucket were compute-pass-bound, HiFi2 wo

Code changes — every attempt (win or fail):
===========================================

[#1] MatmulDeviceOperation · grid · no gain  -1.12 ms
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

[#2] MatmulDeviceOperation · dtype · no gain  -1.11 ms
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

[#3] MatmulDeviceOperation · dtype · no gain  -1.10 ms
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

[#4] BinaryNgDeviceOperation · tt-lang · win  -1.02 ms
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

[#5] BinaryNgDeviceOperation · cpp · no gain  -0.11 ms
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

[#6] MatmulDeviceOperation · cpp · no gain  -0.70 ms
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

[#7] generation_loop · host · no gain  -0.05 ms
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

[#8] MatmulDeviceOperation · host · no gain  -0.05 ms
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

[#9] host_overhead · host · no gain  -0.05 ms
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

[#10] generation_loop · host · no gain  -0.05 ms
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

[#11] MatmulDeviceOperation · dtype · no gain  -0.05 ms
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

[#12] generation_loop · host · no gain  -0.05 ms
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

[#13] MatmulDeviceOperation · tt-lang · no gain  -0.06 ms
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

[#14] BinaryNgDeviceOperation · tt-lang · win  +0.00 ms
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

[#15] ReshapeViewDeviceOperation · tt-lang · no gain  -0.07 ms
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

[#16] MatmulDeviceOperation · tt-lang · no gain  -0.15 ms
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

[#17] MatmulDeviceOperation · tt-lang · no gain  +0.00 ms
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

Limitations / suggested manual next steps:
- 7 op(s) tried but no lever beat baseline: MatmulDeviceOperation, MatmulDeviceOperation, MatmulDeviceOperation, MatmulDeviceOperation, ReshapeViewDeviceOperation, generation_loop, host_overhead
  -> inspect the per-op device report and consider a hand-written kernel or a structural change.
- No net speedup recorded — the model may already be at its ttnn floor, or the dominant op needs a custom kernel.

Reproduce:
  trace+2CQ perf:  python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_main_perf.py::test_main_perf -svv
  demo (real input→output):  python models/demos/hf_eager/hunyuanvideo_1_5/demo/demo_i2v.py
  full-model e2e PCC:  python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_e2e_pipeline.py -svv

levels: grid -> dtype -> tt-lang -> cpp -> host   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->
