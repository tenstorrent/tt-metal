# Qwen-Image-Edit on T3K (TTNN)

End-to-end image editing with [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) on a
T3K (8 x Wormhole, opened as a 2x4 mesh). The pipeline chains the TTNN ports graduated by the three
component bring-ups:

| component | bring-up folder | graduated modules (status NEW + last_good snapshot) |
|---|---|---|
| text encoder (Qwen2.5-VL-7B) | `models/demos/qwen_image_edit_text_encoder` | `vision_patch_embed`, `v_l_vision_block`, `v_l_patch_merger`, `vision_transformer_pretrained_model`, `v_l_decoder_layer`, `language_model_layers_0_mlp`, `v_l_text_model` |
| VAE (AutoencoderKLQwenImage) | `models/tt_dit/pipelines/qwen_image_edit_vae` | `qwen_image_encoder3d`, `qwen_image_decoder3d`, `qwen_image_causal_conv3d`, `qwen_image_residual_block`, `qwen_image_resample`, `zero_pad2d`, `qwen_image_mid_block`, `qwen_image_attention_block`, `qwen_image_r_m_s`, `qwen_image_up_block`, `qwen_image_upsample` |
| MMDiT (QwenImageTransformer2DModel, 20B) | `models/tt_dit/pipelines/qwen_image_edit_transformer` | `timesteps`, `timestep_embedding`, `qwen_timestep_proj_embeddings`, `qwen_embed_rope`, `qwen_image_transformer_block`, `feed_forward`, `ada_layer_norm_continuous` |

All 25 are on the real data path (see "Gate 2" below). The plan behind this package is in
[`e2e_plan.json`](e2e_plan.json).

## Call 1: `image_edit` (image + instruction -> edited image)

```
image, instruction --HF Qwen2VLProcessor / VaeImageProcessor / FlowMatch scheduler (host)--> encoded inputs
  vision_encode   vision tower (patch embed -> 32 blocks -> merger) on the condition image
  text_encode     token embeddings with the image embeddings spliced in -> 28-layer LM -> drop the 64 template tokens
                  (prompts and the negative prompt " " run as one 64-sequence batch)
  vae_encode      VAE encoder -> quant_conv -> mode -> normalise -> 2x2 pack
  denoise x 50    cat[latents, image latents] -> 60 transformer blocks (cond + uncond), true-CFG 4.0 with norm
                  rescale, FlowMatch Euler step
  vae_decode      unpack -> denormalise -> post_quant_conv -> VAE decoder -> image in [0, 1]
```

`tt/pipeline.py:run_image_edit` is the ONE chained forward. The demo and the e2e test both call it.

### Run

```bash
# demo: the 32 bundled samples (crops of models/sample_data photos, 32 instructions, seeds 1000..1031)
python -m models.demos.qwen_image_edit.demo.demo_image_edit --compare-golden
# demo: your own image(s)
python -m models.demos.qwen_image_edit.demo.demo_image_edit --image path/to/photo.jpg --prompt "Turn it into a watercolor painting."

# e2e gate (Gates 1/2/3, B=4, full 50 steps, ~15 min). The HF golden is built on CPU once (~1.5 h) and cached in _golden/
./python_env/bin/python -m pytest models/demos/qwen_image_edit/tests/e2e/test_e2e_image_edit.py -s
# perf: every stage trace-captured and replayed (trace+1cq) via the generic PipelineStageAdapter
./python_env/bin/python -m pytest models/demos/qwen_image_edit/tests/e2e/test_image_edit_perf.py -s
# contract: per-stage trace capture at full depth, and the depth knob
./python_env/bin/python -m pytest models/demos/qwen_image_edit/tests/test_pipeline_contract.py -s
# build the golden explicitly
python -m models.demos.qwen_image_edit.reference.golden --batch 4 --steps 50
```

### Input size

The HF pipeline hard-codes `calculate_dimensions(1024 * 1024)` for the condition image. The gate
runs at 256 x 256 (512 image tokens + ~100 text tokens), and the golden runs the same HF pipeline with
that one target area overridden. The reason is the golden's cost: fp32 on CPU is ~10 s per
sample-forward at 612 tokens, so 32 samples x 2 (CFG) x 50 steps is already ~9 h at 256^2. The demo
takes `--area`.

## Layout on the 2x4 mesh (measured, per chip)

| part | placement | DRAM per chip |
|---|---|---|
| transformer (60 blocks) | TP=8 over all 8 chips; collectives run on axis 1 then axis 0 | 5.36 GB |
| text encoder | TP=4 over the 4 columns. LM row-staged: layers 0-13 on row 0, 14-27 on row 1. Vision tower replicated over the rows. `embed_tokens` sharded by hidden dim | 2.26 GB |
| VAE | W-parallel over the 4 columns, batch-parallel over the 2 rows | 0.41 GB |
| total after build / after prepare (B=32) | | 8.04 / 8.21 GB |

The ceiling used is the registered 10.5 GB usable per chip with a CCL axis in play (12 GB DRAM per
chip). The graduated text-encoder layout (TP=4 x DP=2, whole LM on both rows) measured 4.72 GB per
chip, which with the transformer is past that ceiling. That is why the LM is row-staged.

The trace region is 896 MB, sized from the largest stage trace (measured): denoise with the precise
transformer 760 MB, vision_encode 548 MB, vae_decode 267 MB.

Per-stage batch ceilings (measured, re-tested after the last memory change):
* `vae_decode` runs 32 images as 2 programs of 16. 32 in one program fails in the allocator even with
  the halo buffers released; 16 peaks at 9.27 GB live plus the trace region.
* `text_encode` runs the 64 sequences (32 prompts + 32 negative prompts) as 2 programs of 32. With the
  896 MB trace region reserved, one 64-sequence program ran out of DRAM (a 704 MB allocation).

Every other stage runs all 32 samples in one program.

## Numerics (why the ports run in their precise modes)

The VAE ports reach HF parity as graduated (image latents 0.99998). The transformer does too per
forward (eps 0.99999), but a denoising run chains 100 forwards. Its precise mode
(`_stubs/_precise.py`) takes the CFG-combined noise from 0.99993 to 0.9999956 PCC per step, using
2-limb activations and an exact-lane QK^T; see Results for what that does over 50 steps. The Qwen2.5-VL
text encoder did not reach parity as graduated: at B=32 its prompt embeddings reached only
0.973 PCC on the worst sample. The vision tower grows massive activations (|x| up to 2.6e4 at blocks
17 and 31) that amplify small per-block errors ~1000x on a few tokens. HF fp32 vs fp64 differs by
<0.4% there, so the port has to be accurate to about fp32. What was measured on this T3K and fixed
(the ports' `precise` mode, which the pipeline switches on):

| trap | measured | fix |
|---|---|---|
| `ttnn.add(fp32, bf16)` rounds the SUM to bf16 | ~4e-4 on every biased projection | fp32 biases |
| `packer_l1_acc=True` with some auto matmul configs | 1.4e-3..1.7e-3 vs 3.1e-4 | `packer_l1_acc=False` |
| fused `ttnn.softmax` on masked window rows | one token 6.7% off from exact inputs | explicit fp32 max/exp/sum/divide |
| `ttnn.all_reduce` / `reduce_scatter` on fp32 | 7e-3..1.1e-2 abs | all_gather + fp32 add (exact) |
| fused `ttnn.rms_norm` | ~2x the manual error | fp32 mean / rsqrt / scale |
| tile matmul accumulation | 3.1e-4 relative, K-independent | vision: fp32 input as 3 bf16 limbs x 8 lanes (<= 4 nonzeros per 32 terms, 8 apart: exact to 1.2e-6), median of 3 rotated lane partitions (rare power-of-two tile glitches, ~1 per 6e5 outputs, are outvoted); LM: 2 bf16 limbs, dense |

Result at B=32: the text encoder's prompt / negative embeddings are >= 0.99998 PCC vs HF for every
sample. Merged vision embeddings are PCC 1.000000 on the samples that were worst before. The
text-encode stage takes 80 s per 32-sample call.

## Results (T3K, 2x4 mesh, B=32 distinct samples, 256x256, 50 steps, true-CFG 4.0)

Status: **Gate 1 PASS, Gate 2 PASS, Gate 3 NOT MET** (20 of 32 samples >= 0.99; min 0.720).

| gate | result |
|---|---|
| 1 native | static torch-compute scan of the 25 graduated stubs + glue + chain: clean. host_op_observer over the full forward: 0 host aten ops |
| 2 invoked | all 25 graduated modules invoked by the real forward (e.g. v_l_vision_block 32, v_l_decoder_layer 28, qwen_image_transformer_block 6000 = 60 blocks x 2 (CFG) x 50 steps, feed_forward 12000, zero_pad2d 3, qwen_image_upsample 6) |
| 3 image PCC >= 0.99 per sample | 20/32 pass. Min 0.720 (sample 30), then 0.941 (26), 0.947 (14), 0.948 (5), 0.970 (15), 0.977 (21), 0.978 (25), 0.979 (24), 0.982 (20), 0.987 (7), 0.989 (8, 31) |
| independence | 32 distinct outputs; every output matches its own golden best |
| horizon | the full 50-step schedule ran on both sides (no cap) |

Per stage, against the HF fp32 reference (same inputs):

| stage | PCC |
|---|---|
| text encoder (vision + LM), prompt / negative embeds, B=32 | min 0.99994 / 0.99993 |
| VAE encode (image latents) | 0.99998 |
| transformer, one forward (precise mode), B=2, full depth | eps 0.9999997; CFG-combined 0.9999956 |
| VAE decode of the golden's own final latents, B=32 | min 0.9963, mean 0.9996 |
| trace replay vs eager, every stage (final configuration) | 1.000000 |

Why Gate 3 is not met: the 50-step true-CFG 4.0 trajectory is chaotic for some inputs. The HF
reference does not reproduce itself on them. The same HF fp32 pipeline run at B=2 instead of B=32
(identical math; only the text padding length differs) gives these image PCCs against the B=32 golden:

| sample | HF(B=2) vs HF(B=32) | TT vs HF(B=32) |
|---|---|---|
| 30 | 0.953 | 0.720 |
| 26 | 0.966 | 0.941 |
| 25 | 0.9992 | 0.978 |
| 5 | 0.9984 | 0.948 |
| 14 | 0.99999 | 0.947 |
| 20 | 0.99992 | 0.982 |

On samples 30 and 26 no implementation that is not bit-identical to that one HF run can reach 0.99.
On the others (14, 20, 5, 25) the reference is stable. There the TT per-forward error, ~3e-3 relative
on the CFG noise, is still ~100x fp32 rounding, and the trajectory amplifies it. Getting there would
mean exact-accumulation matmuls in every transformer projection. By the text-encoder measurements
that is ~8x the matmul work: ~10 min per step, ~8 h per 32-image call. It was not done. The demo's
edits are visually correct for all 32 prompts (see `demo/output/edit_*.png`).

Timing (B=32, after build): build 200 s; vision + text encode 80 s; VAE encode 1.4 s; denoise 120 s
per step (precise transformer; 55 s without it); VAE decode 20 s. About 102 min per 32-image call.

## Trace / perf contract

`PIPELINE_STAGES = ["vision_encode", "text_encode", "vae_encode", "denoise", "vae_decode"]`. Each stage
exposes `<stage>_trace_inputs / _trace_setup / _trace_step / _trace_items` on the object returned by
`tt.pipeline.build_pipeline(device, model=None, layers=None, **kw)`. `layers` caps every repeated
stack (vision blocks, LM layers, transformer blocks); the per-stack overrides are
`vision_encode_layers`, `text_encode_layers` and `denoise_layers`. The HF reference stays reachable as
`pipe.hf`. `pipe.trace_capture_selftest()` captures, replays and releases one step per stage.
`pipe.host_op_selftest(p)` runs the forward under `scripts.tt_hw_planner.host_op_observer`. The
module-level `tt.pipeline.host_op_selftest()` / `tt.pipeline.trace_capture_selftest()` are the zero-arg
entry points the harness probes call: they open the mesh themselves (through `mesh.py`, outside `tt/`),
build 2 blocks per stack at B=2 and run the same checks. The pipeline proper never opens a device; it
runs on the device handed to `build_pipeline`.

The denoise loop in `run_image_edit` is itself traced: step 0 runs eagerly (compiles), then one
scheduler step is captured over persistent (latents, t, dt) buffers and replayed for steps 1..N-1, fed
by device-to-device copies of the pre-uploaded per-step t / dt. Measured at B=4, full depth: 16.06 s per
step both eager and replayed (the step is device-compute bound), replay PCC vs eager 1.0.

## Layout

```
demo/demo_image_edit.py     runnable demo (argparse, __main__)
tt/inputs.py                HF processors -> encoded inputs (shared by TT and golden)
tt/pipeline.py              the chained forward, build_pipeline, trace contract, selftests
tt/text_encoder.py, tt/transformer.py, tt/vae.py   stage wiring over the graduated ports
tt/tracker.py, tt/gates.py  Gate 2 counters, Gate 1 scan
reference/golden.py         HF QwenImageEditPipeline golden (fp32 CPU, cached in _golden/)
mesh.py                     mesh open/close for the standalone entry points (demo, perf test, selftests)
tests/e2e/test_e2e_image_edit.py      the correctness gate (B=4)
tests/e2e/test_image_edit_perf.py     trace+1cq per stage
tests/test_pipeline_contract.py       full-depth trace capture per stage, depth knob
```
