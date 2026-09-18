# TT BFP weight quantization

Offline preprocessing tools for model experiments in tt-metal. Start with this
README; use the [experiment guide](docs/EXPERIMENTS.md) for a complete workflow
and [agent instructions](AGENTS.md) when handing the work to a coding agent.

Two offline weight transforms from the Llama/Qwen quantization experiments:

| Method | Formats | Calibration | What it does |
|---|---|---|---|
| **Max/max−1 exponent search** (Matt's method) | `bfloat4_b`, `bfloat8_b` | None | For each 16-value group, choose Emax or Emax−1 by squared weight error. |
| **GPTQ + exponent search** | `bfloat4_b` | Layer-input second moments | Choose Emax/Emax−1/Emax−2 inside GPTQ, then compensate the error in remaining input columns. |

Outputs are ordinary tensors containing prepared values. Load them through the
existing TTNN BFP conversion: no inference-time scales, permutations or new kernels.
This package targets TT's **B-format**, not INT4, NF4, or `bfloat4`/`bfloat8` A-format.

## Install and try

Python 3.10+, PyTorch and NumPy; preprocessing runs on the CPU. Install PyTorch
appropriate for your machine, then from the tt-metal repository root:

```sh
cd tools/bfp_quantization
python -m pip install '.[checkpoint]'  # includes safetensors checkpoint export
tt-bfp-quant build             # optional fast C++17 backend; recommended for large matrices
python examples/toy_experiment.py
python -m pip install '.[test]'  # optional test dependencies
python -m pytest --confcutdir=. -q
```

The commands below run from `tools/bfp_quantization`. This subpackage installs
independently of TTNN and can preprocess on a CPU machine without a TT build.
`--confcutdir=.` isolates its CPU tests from tt-metal's device fixtures.
NumPy is constrained below 2 to agree with this repository's TTNN dependency.

Linux builds try OpenMP; macOS defaults to serial C++. Set `CXX` to choose a
compiler, or use `--openmp on` with an OpenMP-capable compiler. Without a build,
the NumPy/PyTorch implementation works but GPTQ's column loop is slower.
`backend="native"` requires a built backend; `"auto"` uses it when available.

## Export a complete model; keep your inference script

The checkpoint-to-checkpoint command starts with a **local,
original FP32/BF16/FP16 Hugging Face safetensors checkpoint**, single or sharded:

```sh
tt-bfp-quant export --model /models/Llama-3.1-8B-Instruct \
  --output /models/max-minus-one/Llama-3.1-8B-Instruct \
  --recipe examples/mlp-max-minus-one.json --dry-run

# Run the same command without --dry-run to write the new checkpoint.
tt-bfp-quant export --model /models/Llama-3.1-8B-Instruct \
  --output /models/max-minus-one/Llama-3.1-8B-Instruct \
  --recipe examples/mlp-max-minus-one.json
```

The example selects MLP gate/up for BFP4, using Emax/Emax−1. The GPTQ example
selects the same weights so the comparison changes only the method. To also
search existing BFP8 down weights, use `examples/mlp-all-max-minus-one.json`.
**Edit these rules to match your model's names, deployed dtypes and output shards.**
Unselected weights are unchanged. Tensor names/shapes/dtypes, config and tokenizer
assets are preserved. The export is an ordinary model directory, so existing
`from_pretrained(...)` calls can load it by changing just the path.

For a TT model script that already accepts these variables:

```sh
export HF_MODEL=/models/max-minus-one/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/models/tt-cache/llama-max-minus-one
# Run your usual model command with its existing precision configuration.
```

The output contains prepared floating-point values; **the checkpoint does not
change runtime BFP dtypes or packing**. No kernel/model edits are needed when the
existing loader's transpose, output ordering and shard boundaries match the recipe.
Arbitrary fusions/reorders need an adapter. Use a fresh TT weight cache per variant.

For GPTQ + search, supply the saved calibration statistics too. This example
changes gate/up only; down and other weights stay as in the source checkpoint:

```sh
tt-bfp-quant export --model /models/Llama-3.1-8B-Instruct \
  --output /models/gptq-search/Llama-3.1-8B-Instruct \
  --recipe examples/mlp-gptq-search.json --hessians stats/hessians.json
```

See [checkpoint recipes and Hessian mapping](docs/CHECKPOINTS.md). No model is
loaded for execution during export; weights are processed one shard at a time.
For the ordinary-rounding control through the same export path, use
`examples/mlp-round.json`. All recipes are examples, not automatic model policies.

## Your model: weights only

The Linear APIs take **`[out_features, in_features]`**, as in `torch.nn.Linear`.
TT packs the transpose `[K,N]`, so exponent groups run over **16 outputs for
each input channel**. The API handles this transpose.

```python
import torch
from tt_bfp_quant import search_linear, to_bf16_exact

W = model.get_submodule("your.layer.projection").weight.detach().cpu()
ordinary, _ = search_linear(W, bits=4, exponent_deltas=(0,))
improved, info = search_linear(W, bits=4)  # default: Emax and Emax-1
torch.save(to_bf16_exact(improved), "projection.max-minus-one.pt")
```

Use `bits=8` for existing BFP8 weights. Smaller exponents provide finer steps
but clip large magnitudes; each group keeps the option with the lowest error.

## Your model: GPTQ + search

Collect representative **training/calibration** prefixes with the original model
first. Keep evaluation data separate. No backward pass or labels are needed.

```python
from tt_bfp_quant import capture_linear_inputs, factor_hessian, gptq_search

names = ["your.layer.projection"]  # exact names from model.named_modules()
model.eval()
with capture_linear_inputs(model, names, device="cpu") as stats:
    with torch.inference_mode():
        for batch in calibration_batches:  # unpadded batches, on the model's device
            model(**batch)
H = stats[names[0]].value()          # mean(X.T @ X), shape [in_features, in_features]
torch.save(H, "projection.hessian.pt")  # reuse for subsequent experiments

factor = factor_hessian(H, damping=0.01)
prepared, info = gptq_search(W, factor=factor)
torch.save(to_bf16_exact(prepared), "projection.gptq-search.pt")
```

Collect all intended statistics before changing weights. Reuse a factor for
projections with **identical inputs and channel ordering**, such as gate/up.
For padded inputs or a TTNN model, capture the real layer inputs yourself and
call `HessianAccumulator.add(x, mask=valid_tokens)`; see [integration notes](docs/INTEGRATION.md).
The convenience hooks support `torch.nn.Linear`, not arbitrary fused modules.

One dense FP32 Hessian costs **4 × input_width² bytes**, plus factorization
temporaries. At width 16,384 the Hessian alone is 1 GiB. Process subsets of layers
if needed. Saved Hessians avoid recapturing activations for each quantization recipe.

## Load on TT hardware correctly

Use a **fresh TT weight-cache location for each variant** and keep inference
dtypes/configuration fixed for a comparison at the same precision. The saved
BF16 tensor is a lossless carrier of prepared values, not an already packed BFP file.

For a standard Linear weight, pass `prepared.T.contiguous()` to your existing
`ttnn.from_torch(..., dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)` path.
If the model loader already transposes checkpoint weights, supply `[out,in]`
there and let it perform that transpose.

**Sharding/fusion matters:** use `output_splits=[local_N, ...]` on both Linear APIs
when output shards have their own exponent-group boundaries. For fused, reordered,
or otherwise packed tensors, apply `search_packed` to **each physical shard after
layout transformations**. See the [TTNN example](examples/ttnn_loading.py).
Check `validate_repacking(prepared, bits=4, native=True)` against your installed
TTNN packer; pass the same `output_splits` or `layout="packed"` as appropriate.

The APIs also apply to attention projections and output heads. Accuracy gains on
one model or layer do not guarantee gains elsewhere; compare ordinary rounding,
max/max−1 and GPTQ + search on identical held-out inputs. Measure end-to-end
perplexity/reference agreement and throughput on your actual TT implementation.

## CLI for exported matrices

Each input file contains a single `torch.save(tensor, path)` tensor:

```sh
tt-bfp-quant quantize --weight W.pt --output W.search.pt --method max-minus-one --bits 8
tt-bfp-quant quantize --weight W.pt --hessian H.pt --output W.gptq.pt --method gptq-search --bits 4
```

The CLI writes a BF16 tensor and JSON metadata and checks repacking stability.
Add `--validate-ttnn` for native host-packer validation; it does not open a device.
Use `--layout packed` only for a single tensor already arranged as a physical shard.

[Method details, provenance and validation](docs/METHODS.md) ·
[GPTQ paper](https://arxiv.org/abs/2210.17323).
