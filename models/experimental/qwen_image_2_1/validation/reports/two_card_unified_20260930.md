# Qwen-Image 2.1: unified two-card execution report

Draft, 2026-09-30. All generation measurements below completed successfully.

## Findings

The model now runs one native text-to-image request across **two Blackhole P150 cards** using tensor parallelism. The warm-cache repeat completed the full 20-step pipeline in **53.449 s**, with **3.990 s** spent in denoising. The same workload on one card with resident weights took **58.340 s** overall and **4.852 s** denoising. Observed two-card gains are **1.216× for denoising** and **1.092× overall** in these single-run measurements.

The large improvement over the old 568-second run mainly comes from **keeping prepared weights on device**, not from doubling card count. The single-card resident implementation preserves all 20 latent tensors and the decoder output **bit-for-bit**. Tensor parallelism changes floating-point arithmetic: its final native latent has PCC **0.997788949** and **6.643%** relative RMS difference versus one card despite identical starting inputs.

## Common workload and environment

| Parameter | Configuration |
|---|---|
| Model | Qwen/Qwen-Image-2.1 |
| Checkpoint revision | `790c92633540aa0cb11d9abf19eb46d861714758` |
| Diffusers commit | `5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2` |
| Raw prompt | `the quick brown fox jumps over the lazy dog` |
| Image size / batch | 384 × 256 / one image |
| Schedule / seed | 20 complete denoising steps / 42 |
| Prompt expansion | Off |
| Precision | BF16 tensors; existing compute settings retained |
| TT host / firmware / runtime | f02cs02 / 19.11.0 / TTNN 0.65.1rc17.dev6333+h3.3 |
| Two-card allocation | UMD 0 and 1: `0000:01:00.0`, `0000:21:00.0` |
| Single-card control | UMD 0; earlier streamed native baseline used UMD 4 of the same SKU |
| CUDA reference | One RTX A6000, Torch 2.8.0+cu128, model CPU offload |

Every native generation computes its own prompt embeddings, initial noise, denoising trajectory, and VAE decode from raw text and the seed. Captured activations are not injected into those generations. CPU work includes tokenization, deterministic schedule/rotary metadata, checkpoint loading and image writing. The matched-input accuracy diagnostic is separate and deliberately uses identical CUDA inputs.

## Two-card implementation

The DiT's QKV projections and attention heads are split into **16 heads per card**. Each card computes **6,144 gated MLP channels**. Output projections produce partial 4,096-channel results, combined by two BF16 sum collectives per transformer block over the 1D fabric. The residual stream, prompt encoder, input/output projections, scheduler, and VAE are replicated across both cards. This is tensor parallel execution of one request, not two independent image generations. Encoder/VAE replication currently adds no useful model parallelism to those stages.

Prepared DiT block weights are retained in DRAM for the entire request through `--resident-block-weights`. They are still loaded and prepared once for every fresh process/request. No prefix KV cache or persistent inference service is implemented.

## Runtime results

| Run | Cards | Complete steps | Recorded total | Denoising total | Mean step |
|---|---:|---:|---:|---:|---:|
| CUDA native reference | 1 | 20 | 36.152 s | Not separately captured in original run | — |
| TT1, weights streamed each step, previous native run | 1 | 20 | 568.455 s | 538.633 s | 26.931634 s |
| TT1, resident weights, existing JIT cache | 1 | 20 | 58.340 s | 4.852 s | 0.242622 s |
| TT2, resident weights, first full distributed VAE run | 2 | 20 | 100.195 s | 4.048 s | 0.202378 s |
| TT2, resident weights, warm JIT repeat | 2 | 20 | 53.449 s | 3.990 s | 0.199525 s |

These are host wall times. Resident cases synchronize module boundaries and include host work, file I/O and compilation inside each boundary. TT totals begin before device open, after CPU tokenization/metadata construction; the CUDA total begins before pipeline loading. The two-card first full run includes first-use distributed VAE compilation. The repeat uses existing JIT caches, with a new process and new weight loading. There is one generation per listed case, on a shared host; no confidence interval or service latency distribution is claimed.

### Module breakdown with resident weights

| Module | One card | Two cards, first full run | Two cards, warm repeat |
|---|---:|---:|---:|
| prompt encoder | 24.314758 s | 31.363418 s | 17.582273 s |
| resident weight preparation | 24.444620 s | 26.400759 s | 26.623713 s |
| initial noise | 0.044661 s | 0.402984 s | 0.052488 s |
| text projection | 0.027326 s | 0.030562 s | 0.031688 s |
| input and conditioning | 0.120083 s | 0.147372 s | 0.154411 s |
| dit block | 3.236666 s | 2.534019 s | 2.536071 s |
| output head | 0.061350 s | 0.068886 s | 0.068810 s |
| euler update | 0.028424 s | 0.033651 s | 0.031922 s |
| vae decode | 3.285410 s | 36.373158 s | 3.494737 s |

Step-level module entries are summed across all 20 steps. DiT block times exclude resident weight preparation, which is listed separately. Module sums do not cover every initialization, collection and serialization operation. Weight preparation and prompt encoding dominate the warm two-card request; multiplying denoising performance does not proportionally reduce full-pipeline time.

## Accuracy and reproducibility

| Comparison | Input relationship | PCC | Relative RMS |
|---|---|---:|---:|
| TP2 first-step velocity vs CUDA | Identical captured inputs | 0.999617692 | 2.7769% |
| TP2 first-step updated latent vs CUDA | Identical captured inputs | 0.999997646 | 0.2211% |
| TP2 first-step velocity vs TP1 | Identical captured inputs | 0.999805313 | 2.0091% |
| TP2 final latent vs TP1 | Native runs; starting noise and prompt embeddings bitwise identical | 0.997788949 | 6.6430% |
| TP2 final decoder tensor vs TP1 | Native runs; identical starting inputs, accumulated trajectory difference | 0.994342309 | 8.5032% |
| TP2 VAE vs CUDA VAE | Identical final TP2 latent; separate subsequent CUDA decode | 0.999947954 | 0.8019% |

The single-card caching change preserved all 20 updated latents and the final decoder tensor bitwise versus the original streamed implementation. The two-card repeat also reproduced its first full run's final decoder tensor bitwise. The final TP2/TP1 RGB PNG mean difference was **4.486/255**, maximum **210/255**. Decoder-tensor PCC above includes all four output channels.

High PCC is not an all-element error guarantee. For the matched-input TP2/TP1 first-step velocity comparison, relative RMS is **2.0091%**, maximum absolute difference is **0.164062**, and `torch.allclose(atol=0.01, rtol=0.01)` is **false**. Partitioned matrix multiplication and BF16 partial-output reductions alter rounding; this is not a bitwise distributed port. The final-latent trajectory difference is accumulated error, not an isolated module metric.

### Native TP2 versus TP1 trajectory

| Completed steps | Latent PCC | Relative RMS difference |
|---:|---:|---:|
| 1 | 0.999997135 | 0.2400% |
| 2 | 0.999993095 | 0.3733% |
| 4 | 0.999978019 | 0.6680% |
| 8 | 0.999878429 | 1.5670% |
| 16 | 0.998511291 | 5.4514% |
| 20 | 0.997788949 | 6.6430% |

Native CUDA uses a different random generator. Its seed-42 noise has PCC approximately 0.00669 against native TT noise; therefore final native CUDA/TT image PCC is not reported as correctness. Both TT layouts use bitwise-identical initial noise and prompt embeddings, allowing the TP1/TP2 comparisons above.

## Image observations and limits

The one-card and two-card images have the same broad scene: a fox on snow with two patterned balls. They do not depict the requested fox jumping over a dog. This raw-prompt, small-resolution, 20-step diagnostic has not implemented the official prompt enhancer and does not establish semantic parity with the official demo. Numerical agreement is assessed separately from prompt adherence.

The implementation remains experimental: batch one, text-only image generation, no conditioning images, classifier-free guidance, VAE encoding, prefix KV cache, or service API. Larger resolutions, peak DRAM, isolated device kernel timings, and multi-request latency distributions were not measured here. No strict all-element 0.01 accuracy qualification is claimed for TP2.

## Reproduction

Use the model's pinned environment and matching TTNN wheel. Reserve the listed cards before launching. Reuse the same pinned DiT, text encoder and VAE checkpoint exports. From the project directory:

```bash
CUDA_VISIBLE_DEVICES='' TT_METAL_HOME=/path/to/matching/tt-metal \
  uv run --no-sync python -m benchmarks.tt_full_denoise \
  --native --prompt 'the quick brown fox jumps over the lazy dog' \
  --height 256 --width 384 --steps 20 --seed 42 \
  --checkpoint "$DIT_CHECKPOINT" \
  --tt-encoder-checkpoint "$ENCODER_CHECKPOINT" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --tensor-parallel 2 --resident-block-weights --timing-breakdown \
  --device-bdf 0000:01:00.0,0000:21:00.0 \
  --output-dir "$OUTPUT_DIR"
```

For the resident single-card control, set `--tensor-parallel 1` and supply one BDF. To reproduce the paired diagnostic, replace native prompt/encoder options with `--cuda-dir "$CUDA_CAPTURE" --max-steps 1`; it is a separate captured-input accuracy test.

In the TT-Metal fork, invoke `models.experimental.qwen_image_2_1.validation.tt_full_denoise` from the repository root instead of the workspace benchmark module. The fork's native and captured-input pytest cases accept `QWEN_IMAGE21_TEST_TENSOR_PARALLEL=2`, two distinct BDFs, and `QWEN_IMAGE21_TEST_RESIDENT_BLOCK_WEIGHTS=1`.

The companion [JSON report](two_card_unified_20260930.json) includes per-step timings, per-block timing summaries, and all comparison metrics. Complete raw module records remain with the run artifacts outside Git. Raw tensors, generated images and checkpoints remain outside Git on the data disk. The local two-card viewer is at port 8894; it is not a publicly hosted report asset. Earlier detailed component coverage remains in the [module PCC report](module_pcc_20260930.md), with earlier timing scope in the [runtime report](runtime_20260930.md).

## Fork-path hardware verification

The migrated fork package passed **12 tests**, with one optional oracle case skipped, in **163.27 s**. These cover native TP2 encoding/noise/one-step/TT-VAE execution with CUDA unavailable, the paired TP2 first-step denoiser PCC regression, and CPU metadata checks. The test suite duration includes compilation and is not an inference benchmark. Full 20-step native generation was verified separately in the workspace runner.

## Average DiT block wall times with resident weights

Each average covers the 20 steps within one image generation; it is not a multi-request latency statistic. These block boundaries include both sum collectives on TP2 and synchronize device completion.

| Block | One card | Two cards, warm repeat |
|---:|---:|---:|
| 0 | 0.015102 s | 0.016098 s |
| 1 | 0.004730 s | 0.003672 s |
| 2 | 0.004729 s | 0.003662 s |
| 3 | 0.004739 s | 0.003672 s |
| 4 | 0.004744 s | 0.003660 s |
| 5 | 0.004731 s | 0.003673 s |
| 6 | 0.004730 s | 0.003667 s |
| 7 | 0.004735 s | 0.003663 s |
| 8 | 0.004732 s | 0.003640 s |
| 9 | 0.004730 s | 0.003566 s |
| 10 | 0.004734 s | 0.003553 s |
| 11 | 0.004730 s | 0.003565 s |
| 12 | 0.004734 s | 0.003550 s |
| 13 | 0.004736 s | 0.003550 s |
| 14 | 0.004730 s | 0.003554 s |
| 15 | 0.004730 s | 0.003554 s |
| 16 | 0.004728 s | 0.003554 s |
| 17 | 0.004730 s | 0.003533 s |
| 18 | 0.004733 s | 0.003529 s |
| 19 | 0.004735 s | 0.003542 s |
| 20 | 0.004735 s | 0.003527 s |
| 21 | 0.004736 s | 0.003530 s |
| 22 | 0.004735 s | 0.003527 s |
| 23 | 0.004739 s | 0.003529 s |
| 24 | 0.004733 s | 0.003528 s |
| 25 | 0.004731 s | 0.003527 s |
| 26 | 0.004733 s | 0.003529 s |
| 27 | 0.004732 s | 0.003530 s |
| 28 | 0.004734 s | 0.003527 s |
| 29 | 0.004734 s | 0.003531 s |
| 30 | 0.004735 s | 0.003528 s |
| 31 | 0.004734 s | 0.003532 s |
