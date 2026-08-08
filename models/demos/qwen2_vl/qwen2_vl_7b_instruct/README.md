# Qwen2-VL-7B-Instruct (TTNN)

End-to-end TTNN bring-up of [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
(`Qwen2VLForConditionalGeneration`), a vision-language model, on a single
Tenstorrent **Blackhole p150**. One generative task head: **image-text-to-text**
(image + text prompt → generated text).

The whole model runs on device: the vision tower (patch embed → 32 vision
blocks → 2×2 patch merge) and the text tower (28 decoder layers with GQA + mRoPE
+ SwiGLU → RMSNorm → LM head) are native `ttnn`, with **zero host aten ops in
the hot path**.

## Prerequisites

- A working [tt-metal / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  build (this bring-up is validated on a single **Blackhole p150**, `device_id=0`).
- The commands below use the repo's `./python_env/bin/python`. Substitute your
  own environment if you build tt-metal differently.

## Install Python dependencies

The TT model is self-contained `ttnn`; these packages are only for the
**reference/golden** side (HF model + image processor):

```bash
pip install -r models/demos/qwen2_vl/qwen2_vl_7b_instruct/requirements.txt
```

Validated pins: `transformers==5.10.2`, `accelerate==1.7.0`, `pillow`.

## Download the weights

The demo and tests load `Qwen/Qwen2-VL-7B-Instruct` from the HuggingFace cache
(~16 GB, safetensors). Pre-fetch it once:

```bash
./python_env/bin/hf download Qwen/Qwen2-VL-7B-Instruct --exclude "*.bin"
```

(If you skip this, the first run downloads the weights automatically.)

## Run the demo

Prints the generated answer for an image + prompt. The demo uses the
**KV-cache decode** path (`generate_kv`) by default — the fast, context-flat
path (see [PERF.md](PERF.md)). With no `--image`, a deterministic built-in
gradient image is used (the same one the golden was captured from):

```bash
./python_env/bin/python -m models.demos.qwen2_vl.qwen2_vl_7b_instruct.demo.demo_image_text_to_text \
    --prompt "Describe the colors in this image." --max-new-tokens 24
```

Pass `--image /path/to/img.png` for your own image. Example output:

```
PROMPT: Describe the colors in this image.
ANSWER: The image features a gradient of colors transitioning from a dark purple at the top to ...
```

## Reproduce the accuracy (PCC) results

### 1. Generate the HF golden (one time)

The e2e / KV tests compare against a captured HF golden
(`_captured/e2e_golden.pt`: real input, HF `generate()` tokens + per-step
logits + vision `image_embeds`). A copy is included; regenerate it any time with:

```bash
./python_env/bin/python models/demos/qwen2_vl/qwen2_vl_7b_instruct/_captured/capture_e2e_golden.py
```

### 2. Per-component PCC (7 graduated stubs, on device)

Each stub is compared to its HF submodule (captured real input shapes):

```bash
./python_env/bin/python -m pytest models/demos/qwen2_vl/qwen2_vl_7b_instruct/tests/pcc/ -svv
```

### 3. End-to-end PCC gate (chained pipeline vs HF golden)

Real input → the shared chained TTNN pipeline over all 7 stubs → next-token
logits, compared to the HF golden. Asserts **PCC ≥ 0.95** and **exact token
match** over the greedy horizon `N=16`:

```bash
./python_env/bin/python -m pytest models/demos/qwen2_vl/qwen2_vl_7b_instruct/tests/e2e/test_e2e_image_text_to_text.py -svv
```

### 4. Host-free + trace-capture contract

The full forward fires **zero** host aten ops, and every pipeline stage captures
host-free into a replayable trace:

```bash
./python_env/bin/python -m pytest models/demos/qwen2_vl/qwen2_vl_7b_instruct/tests/e2e/test_trace_2cq.py -svv
```

### 5. KV-cache decode correctness

Tier-2 fixed-capacity KV-cache decode (prefill once + `seq=1` steps) vs the HF
golden (token match + PCC ≥ 0.95, cached attention runs fp32):

```bash
./python_env/bin/python -m models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._kv_check
```

## Measure performance

Times a full N-token greedy decode three ways (best-of-3, warm): full-seq eager,
KV-cache eager, and KV traced+2CQ. Capacity `C` and horizon `N` are env-tunable
to show the KV payoff scaling with context length:

```bash
# validated demo point
QV_CAP=64  QV_NTOK=16 ./python_env/bin/python -m \
    models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._bench_kv_trace2cq

# longer context (shows KV-cache scaling)
QV_CAP=512 QV_NTOK=32 ./python_env/bin/python -m \
    models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._bench_kv_trace2cq
```

Measured numbers are in [PERF.md](PERF.md).

## Results

- **Accuracy:** e2e next-token logits PCC = **0.970** vs HF golden at `N=16`,
  with **16/16** greedy tokens matching HF exactly. Vision `image_embeds`
  PCC = 0.992. The Tier-2 KV-cache decode scores PCC **0.9987** (fp32 cached
  attention).
- **On device:** all 7 graduated stubs run real `ttnn` (float32 vision tower,
  bf16 + HiFi4/fp32-acc text); `host_op_selftest` → 0 host aten ops in the
  forward; every stage captures host-free into a trace.

### Per-component PCC (measured on device, target ≥ 0.99)

| component | PCC |
|-----------|-----|
| `patch_embed` | 0.99999 |
| `patch_merger` | 0.99993 |
| `vision_mlp` | 0.99999 |
| `qwen2_v_l_vision_block` | 0.99995 |
| `qwen2_v_l_decoder_layer` | 0.99942 |
| `qwen2_v_l_text_model` | 0.99802 |
| `qwen2_vision_transformer_pretrained_model` | covered by e2e (see note) |

> The top-level vision tower's per-component test is **skipped**: its windowed
> attention needs a real `(pixel_values, grid_thw)` pairing that the synthetic
> per-component harness can't construct. It is validated on real input by the
> e2e test (`image_embeds` PCC 0.992, 16/16 tokens).

> Note on horizon: beyond ~step 18, a genuine bf16 near-tie flips a single token
> and (as in any AR greedy loop) cascades. The validated horizon `N=16` is where
> the TT and HF greedy sequences agree token-for-token.

## Graduated components (7)

| stub | role |
|------|------|
| `patch_embed` | flattened-patch conv3d-as-matmul projection (vision) |
| `vision_mlp` | vision block MLP `fc2(quick_gelu(fc1))` |
| `qwen2_v_l_vision_block` | 1 of 32 vision blocks (LN + windowed attn + MLP) |
| `patch_merger` | 2×2 patch merge → hidden 3584 (the `image_embeds`) |
| `qwen2_vision_transformer_pretrained_model` | top-level vision tower |
| `qwen2_v_l_decoder_layer` | 1 of 28 text decoder layers (RMSNorm + GQA + mRoPE + SwiGLU); also holds the fixed-capacity KV-cache decode path |
| `qwen2_v_l_text_model` | top-level text model (inputs_embeds → 28 layers → RMSNorm) |

## Layout

```
tt/pipeline.py                        ONE shared chained forward (build_pipeline,
                                      generate / generate_kv, trace hooks, selftests)
demo/demo_image_text_to_text.py       runnable demo (argparse + __main__)
_stubs/                               native ttnn component impls (the 7 stubs above)
_stubs/kv_cache_select_op.py          fused generic_op for the traceable KV-cache write
_stubs/kv_cache_select_kernels/       its reader/compute/writer Metalium kernels
_captured/capture_e2e_golden.py       regenerates the HF golden
_captured/e2e_golden.pt               the captured HF golden (input + tokens + logits)
_captured/<component>/manifest.json   captured input shapes for the per-component PCC tests
tests/pcc/                            7 per-component PCC tests
tests/e2e/test_e2e_image_text_to_text.py   the e2e Gate 1/2/3 test
tests/e2e/test_trace_2cq.py           host-free + trace-capture contract
tests/e2e/_kv_check.py                KV-cache decode PCC check
tests/e2e/_bench_kv_trace2cq.py       KV-cache decode perf bench (Tier-2)
tests/e2e/_bench_trace2cq.py          full-seq trace+2CQ perf bench (Tier-1)
```
