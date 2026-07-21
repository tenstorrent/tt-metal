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

_Updated live: 2026-07-21 01:47:51 UTC · 4 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — hunyuanvideo_1_5 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)

Op breakdown — device time by op class (latest profile · what to target, ranked):
op class         device_ms      %   count  bound  dominant op (shape)
---------------------------------------------------------------------------------------------------
host_overhead        38.20 593.9%       0   host  
matmul                4.40  68.4%      90   slow  MatmulDeviceOperation 32 x 2048 x 2048
eltwise               0.84  13.1%     175   slow  BinaryNgDeviceOperation
datamove              0.59   9.2%      94   slow  ReshapeViewDeviceOperation
reduction             0.56   8.7%      27   slow  LayerNormDeviceOperation
attention             0.04   0.7%       2   slow  JointSDPADeviceOperation

op                                    grid   dtype tt-lang     cpp    host   best ms
------------------------------------------------------------------------------------
BinaryNgDeviceOperation                  —       —    ✓win       —       —      6.34
MatmulDeviceOperation                 ·try    ·try       —       —       —      6.43


Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):
op                                      lever        ms  gain vs base  result     why tried / why it won or failed
------------------------------------------------------------------------------------------------------------------
MatmulDeviceOperation                    grid      6.44      -0.01 ms  · no gain  Hypothesis: dominant matmul profiled as grid=partial, so full core_grid should saturate idle DRAM bandwidth. Outcome: no gain (6.4316->6.4426, slightly slower) — M=32 is only 1 tile high, so widening 
MatmulDeviceOperation                   dtype      6.44      -0.01 ms  · no gain  Hypothesis: op tagged bound_by=memory (weight reads dominate at M=32), so bf8_b weights quarter the DRAM read bytes vs fp32. Outcome: NO gain (bf16 also flat, bf8_b 6.4316->6.4376). Weight-dtype is in
MatmulDeviceOperation                   dtype      6.43      +0.00 ms  · no gain  Enabled the stub's purpose-built bf16 fast path (HY_DIT_BF16 default off->on): bf16 weights+activations, HiFi2, fp32 accumulate, its own comment claims '2-4x faster matmuls + 2x less DRAM'. This is th
BinaryNgDeviceOperation               tt-lang      6.34      +0.09 ms  ✓ win      Bias-fusion (ttnn.linear) in the profiled transformer_block stub removes ~38 standalone BinaryNg add launches from the 2-layer forward. WIN: 6.4316->6.344ms (1.36%, is_real_gain). Root cause: path is 

Code changes — every attempt (win or fail):
===========================================

[#1] MatmulDeviceOperation · grid · no gain  -0.01 ms
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

[#2] MatmulDeviceOperation · dtype · no gain  -0.01 ms
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

[#3] MatmulDeviceOperation · dtype · no gain  +0.00 ms
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

[#4] BinaryNgDeviceOperation · tt-lang · win  +0.09 ms
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

Limitations / suggested manual next steps:
- 1 op(s) tried but no lever beat baseline: MatmulDeviceOperation
  -> inspect the per-op device report and consider a hand-written kernel or a structural change.

Reproduce:
  trace+2CQ perf:  python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_main_perf.py::test_main_perf -svv
  demo (real input→output):  python models/demos/hf_eager/hunyuanvideo_1_5/demo/demo_i2v.py
  full-model e2e PCC:  python -m pytest models/demos/hf_eager/hunyuanvideo_1_5/tests/e2e/test_e2e_pipeline.py -svv

levels: grid -> dtype -> tt-lang -> cpp -> host   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->
