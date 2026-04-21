# X-VLA on Tenstorrent p150a (Blackhole)

Port and optimize the [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base)
Vision-Language-Action model on a single Blackhole p150a chip.

X-VLA = Florence-2 (DaViT vision + BART encoder) → SoftPromptedTransformer
(24-layer, 1024-dim) → flow-matching action decoder. One inference produces a
30-step action chunk; the SoftPromptedTransformer is rolled out 10× per chunk
(`num_denoising_steps`).

## Layout

```
x_vla/
  weights/xvla_base/         # downloaded from HF (3.3 GB)
  benchmark/
    run_benchmark.py         # FIXED metric harness (do not modify)
    lerobot_bootstrap.py     # workarounds for two unrelated lerobot import bugs
  reference/                 # cached reference action chunk (PCC oracle)
  tt/                        # TT-NN port lives here (grows over iterations)
  tests/                     # PCC tests for individual TT-NN components
```

## Run

```bash
# Baseline (torch CPU, what the HF model does out of the box)
python3 models/experimental/x_vla/benchmark/run_benchmark.py --backend torch_cpu

# TT-NN backend (falls back to torch_cpu until tt/policy.py exists)
python3 models/experimental/x_vla/benchmark/run_benchmark.py --backend ttnn
```

The harness prints three greppable lines:

```
inference_speed=<frames per second; one frame = one action step>
accuracy=<PCC% vs cached reference action chunk; 100.0 on first ever run>
peak_dram=<peak on-device DRAM MB; 0 on torch_cpu>
```

## Iteration loop

`results.tsv` (workspace root, not committed) tracks every experiment:

```
commit  gen_speed  accuracy  status  description
```

Per PROGRAM.md: keep iff `gen_speed` improved AND `accuracy ≥ 99`; else revert.
