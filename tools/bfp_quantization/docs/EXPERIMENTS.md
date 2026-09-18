# Running an experiment on your model

This guide is for engineers and agents using the offline BFP quantizers. It covers
the steps needed to turn a supported floating checkpoint into ordinary model
files that an existing inference script can load. The model-specific parts are
the packing/dtype recipe, activation capture for GPTQ, and the evaluation runner.

## 1. Establish the loader contract

Record the checkpoint identifier and revision, tt-metal commit, model script,
hardware/mesh, context length and deployed weight dtypes. Inspect the script's
weight preparation code before selecting weights:

- What are the **raw checkpoint** keys? For example, `model.layers.0.mlp...`
  and `model.language_model.layers.0.mlp...` are different namespaces. The
  safetensors index lists keys without loading weights.
- Does each selected matrix remain a Linear weight `[out,in]` that is transposed
  to `[in,out]` for TTNN conversion? Which weights are BFP4 and which are BFP8?
- Are output channels sharded, fused, permuted or padded? Standard groups contain
  16 contiguous outputs per input channel. If a shard boundary is not aligned to
  16 outputs, specify `output_splits` with the actual physical widths.

MLP gate/up/down projections are a useful starting point. Attention projections
can be quantized too, but QKV fusion, RoPE head permutations and GDN packing can
change the groups. Use the lower-level APIs with a layout adapter for such cases;
do not assume raw checkpoint order equals final TT packing order.

The exporter accepts original FP32/BF16/FP16 **safetensors** checkpoints, single
or sharded. It does not import community INT4/GPTQ/AWQ or FP8 checkpoints. It never
changes the inference script's selected BFP dtypes. A drop from BFP8 to BFP4 needs
an existing configuration option or a separate model change; it is not caused by
writing a recipe with `bits: 4`.

## 2. Install and smoke-test

From the tt-metal repository root, on a CPU machine or your normal development
host with an appropriate PyTorch installation:

```sh
cd tools/bfp_quantization
python -m pip install -e '.[checkpoint,test]'
tt-bfp-quant build
TT_BFP_TEST_NATIVE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest --confcutdir=. -q
python examples/toy_experiment.py
```

This installs only the tool and its dependencies; it does not build tt-metal.
The optional C++17 build accelerates preprocessing. No device kernels change.
Use `--backend native` to require that backend, or `--backend numpy` for the
portable implementation. `auto` uses the native backend if already built.

Commands in the remaining sections run from `tools/bfp_quantization`. Store your
models, statistics, recipes and results outside the checkout.

## 3. Define comparable variants

Copy the example recipes and adapt names, bits and shard boundaries to the
contract established above. Keep the selected matrices and runtime settings
identical when comparing algorithms.

| Example recipe | Selected weights | Method | Calibration |
|---|---|---|---|
| `mlp-round.json` | Gate/up, BFP4 | Ordinary Emax rounding | None |
| `mlp-max-minus-one.json` | Same gate/up, BFP4 | Emax/Emax−1 weight-error search | None |
| `mlp-gptq-search.json` | Same gate/up, BFP4 | GPTQ + Emax/Emax−1/Emax−2 search | Input second moments |
| `mlp-all-max-minus-one.json` | Gate/up BFP4 **and down BFP8** | Emax/Emax−1 search | None |

The last row changes the scope as well as the weights; label it accordingly.
Unselected weights remain original and go through the model's usual conversion.
The ordinary-rounding export is a useful control for testing the same checkpoint
path and verifying the loader preserves the prepared values. The original source
checkpoint with the same BFP policy is the deployment baseline.

Preview a recipe without writing or loading tensor data:

```sh
tt-bfp-quant export --model /models/Llama-3.1-8B-Instruct \
  --output /experiments/max-minus-one/Llama-3.1-8B-Instruct \
  --recipe examples/mlp-max-minus-one.json --dry-run
```

Every pattern must match; overlapping patterns are rejected. A GPTQ dry run also
lists every weight requiring a Hessian. Check those names before collecting data.

## 4. Collect statistics once, only for GPTQ

Exponent search needs only weights. GPTQ additionally uses `H = mean(X.T @ X)`
for the inputs to each targeted Linear. Gather it with the original model on
representative calibration/training prefixes, with evaluation mode and no
gradients. Labels are unnecessary. Exclude padding and keep evaluation data
separate. Save the tokenizer revision, exact token IDs or corpus revision,
selection seed, context length and number of valid calibration tokens.

Use [`capture_linear_inputs`](../src/tt_bfp_quant/calibration.py) for ordinary
PyTorch Linear modules, or `HessianAccumulator.add(x, mask=...)` in your temporary
TT capture adapter. The [calibration example](CHECKPOINTS.md#calibration-files-for-gptq)
saves a JSON map from checkpoint weight names to Hessian files. Existing TT
statistics can be reused when their input channel order and model revision match.
For TT inputs, remove replicas/padding and concatenate input-channel shards
correctly; see [integration notes](INTEGRATION.md#capture-activations-in-any-implementation).

Gate/up projections often have exactly the same input. In that case map both to
one saved Hessian file to reuse factorization. Matching widths alone do not prove
the inputs are interchangeable. A Hessian at input width 16,384 takes 1 GiB in
FP32, before factorization temporaries; capture layer subsets when needed. Save
statistics rather than raw activations and reuse them across recipes.

The earlier Qwen experiment used 65,536 calibration tokens and separate evaluation
data. That is a reproducible starting size, not a guarantee of adequate coverage
for other models, domains or routed experts.

## 5. Export the weights and run your normal model script

```sh
# No inference pass or Hessians needed for this method.
tt-bfp-quant export --model /models/Llama-3.1-8B-Instruct \
  --output /experiments/max-minus-one/Llama-3.1-8B-Instruct \
  --recipe examples/mlp-max-minus-one.json --backend native

# Reuses previously captured statistics; does not run calibration again.
tt-bfp-quant export --model /models/Llama-3.1-8B-Instruct \
  --output /experiments/gptq-search/Llama-3.1-8B-Instruct \
  --recipe examples/mlp-gptq-search.json \
  --hessians /experiments/stats/hessians.json --backend native
```

The output preserves names/shapes/dtypes and ordinary config/tokenizer assets.
It is a self-contained floating checkpoint, not a native packed BFP file.
Conversion to BFP remains in the existing model loader. The exporter processes
one shard at a time, caches one Hessian factor and never loads the model for
execution. Plan for checkpoint-sized output disk space, shard/working-matrix RAM
and dense Hessian/factorization memory. Input files are not modified.

Every prepared tensor must survive ordinary numerical BFP repacking. On a host
with TTNN configured, add `--validate-ttnn` to check native host conversion for
each tensor. This validates the recipe layout, not any later loader transformation.
Check the loader's actual final packing boundary before assuming those coincide.

For model scripts that already accept `HF_MODEL` and `TT_CACHE_PATH`:

```sh
export HF_MODEL=/experiments/max-minus-one/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/experiments/tt-cache/max-minus-one
# Run the usual model/evaluation command, with the original dtype configuration.
```

Other implementations may expose different path/cache flags. Retain the original
model-family basename under a variant directory, since some TT scripts infer
configuration from it. Use a new cache per variant to avoid silently loading
previously packed weights. The inference environment does not need this package.

## 6. Evaluate on held-out prefixes

Compare every variant on the **same tokenized prefixes, positions and masks**.
Use teacher forcing with identical input tokens for reference comparisons;
independently generated continuations quickly diverge and measure something else.
Exclude padding and use consistent context boundaries, BOS/EOS handling and token
shift. The calibration corpus must not overlap these evaluation samples.

| Metric | Definition | Better |
|---|---|---|
| Perplexity | `exp(mean(-log p(actual next token)))` on held-out text | Lower |
| Ground-truth next-token accuracy | Quantized top-1 equals the actual next token | Higher |
| Reference top-1 agreement | Quantized top-1 equals reference top-1 at the same position | Higher |
| KL divergence | Mean `KL(p_reference || p_quantized)` over the full vocabulary, in nats | Lower |
| Reference top-1 recall in top-5 | Reference top-1 appears in quantized top-5 | Higher |
| Top-5 set overlap | Mean `size(reference top-5 intersect quantized top-5) / 5` | Higher |

Top-5 overlap ignores rank. Optional Jaccard (`intersection / union`) measures the
same sets on another scale; neither establishes rank agreement. Top-1 recall in
top-5 may saturate, so report it alongside the more discriminating metrics.
Compute KL with stable full-vocabulary log-softmax; truncating to the top five
tokens does not give the same quantity. Chunk logits/accumulate statistics so
evaluating a large vocabulary does not require retaining all logits in memory.

Published reference scores cannot supply per-prefix logits for agreement or KL.
Run/cache the reference outputs for these inputs, recording reference precision,
runtime and model revision. A BF16 or FP32 reference is distinct from ordinary
BFP8; do not label the latter unquantized. If you use a published perplexity or
task score, compare it only under a matching documented evaluation protocol.

Use enough held-out tokens/examples to resolve the observed differences; report
the counts and, where possible, paired uncertainty across documents/examples.
For downstream task accuracy, keep the dataset split/version, prompt template,
few-shot examples, generation/scoring settings and harness version identical.
The package does not supply a generic language-model/task benchmark runner.

## 7. Record results and cost

Keep `tt_bfp_quantization.json` from each export, the recipe, checkpoint/model
revision, Hessian mapping/provenance, exact evaluation command and environment
with the results. Report absolute scores as well as baseline differences; avoid
an ambiguous single "accuracy %" that mixes task accuracy and reference agreement.

Separate time spent loading weights, collecting statistics, factoring Hessians,
quantizing, validating/writing exports and evaluating. The export report includes
total export time, factorization time, per-weight quantization and validation
times; calibration/evaluation require their own timers. A reused factor's metadata
contains its original factorization time, so use report-level `factor_seconds`
instead of summing that field once per weight. Reading/writing overhead is included
in export total, not individually instrumented.

At unchanged dtypes, offline quantization improves candidate weight values without
changing the kernel workload. Do not infer inference speedups from faster CPU
preprocessing or from reconstruction error. Measure TT throughput/latency and
memory separately when experimenting with lower runtime precision.
