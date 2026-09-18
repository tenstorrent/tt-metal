# Working with offline BFP quantization

Read [README.md](README.md) for commands and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for the experiment protocol. This directory is an independently installable CPU
tool inside tt-metal. The Python package is `tt_bfp_quant` and its CLI is
`tt-bfp-quant`; no device or TTNN build is needed for preprocessing.

## Running an experiment

- Record the model/revision, TT implementation/commit, weight dtypes and physical
  packing layout before selecting tensors. Recipes operate on raw checkpoint
  names, not the model loader's renamed state dictionary.
- Keep dtype and target-matrix scope identical when comparing quantizers.
  The `mlp-round`, `mlp-max-minus-one` and `mlp-gptq-search` recipes target the
  same gate/up weights. The `mlp-all-max-minus-one` recipe additionally changes
  BFP8 down weights and must be labeled as a separate scope.
- Reuse compatible saved Hessians. Never substitute identity matrices or collect
  statistics from evaluation examples to get a GPTQ export to run.
- Use `export --dry-run` first. Export into a new directory, retain the model
  family basename, and choose a fresh TT weight cache per variant. The exporter
  does not change runtime dtypes, kernels, or model configuration.
- Standard Linear weights are `[out,in]`, with BFP groups over 16 outputs after
  transposing to `[in,out]`. Check output-shard boundaries and loader permutations.
  Do not claim arbitrary fused/reordered matrices work without checking packing.
- Report calibration, factorization, quantization, validation/export and evaluation
  costs separately where measured. Keep perplexity, reference agreement and
  ground-truth next-token accuracy distinct. Record anything not measured.

## Editing and testing

Core code is in `src/tt_bfp_quant`; the optional C++ backend is
`src/tt_bfp_quant/csrc/quantize.cpp`. Keep its float32 arithmetic consistent with
the NumPy/PyTorch oracle and retain `-ffp-contract=off`. GPTQ currently supports
BFP4 only; weight-only exponent search supports BFP4 and BFP8.

From the repository root:

```sh
cd tools/bfp_quantization
python -m pip install -e '.[checkpoint,test]'
tt-bfp-quant build
TT_BFP_TEST_NATIVE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest --confcutdir=. -q
python examples/toy_experiment.py
```

The cut directory prevents loading tt-metal's device fixtures. CPU-only exception
tests explicitly document the repository hook's permitted `allow-pytest.raises`
exception. Optional TTNN tests require an installed/configured runtime; a skip
does not establish native packing correctness. Do not claim TT performance or
end-to-end model accuracy from the synthetic reconstruction tests.

Keep weights, saved activations/Hessians, model downloads, native binaries and
generated export directories outside the repository. The small frozen test fixture
contains synthetic data. Its provenance is recorded in `tests/frozen-experiment.json`;
do not regenerate expected values from the implementation under test.
