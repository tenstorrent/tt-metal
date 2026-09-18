# Complete checkpoint export

`tt-bfp-quant export` writes an ordinary HF safetensors model directory. It does
not require Transformers, construct a model, or run inference. Changing the
checkpoint path is sufficient for an existing loader that accepts that format
and uses the packing/precision described in your recipe. The quantizer itself
is model independent; choosing the correct recipe is an integration step.

## Recipe

The included `examples/mlp-gptq-search.json` selects gate/up only. You can also
combine methods explicitly, as in this recipe that adds max/max−1 on down:

```json
{
  "schema_version": 1,
  "rules": [
    {
      "match": ["model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"],
      "method": "gptq-search",
      "bits": 4
    },
    {
      "match": "model.layers.*.mlp.down_proj.weight",
      "method": "max-minus-one",
      "bits": 8
    }
  ]
}
```

- `match`: case-sensitive shell-style glob, or list of globs, against **raw
  checkpoint tensor names**. Every pattern must match and each tensor can match
  only once. A Qwen checkpoint may use `model.language_model.layers.*...`;
  inspect its index and adjust the pattern. There is no automatic dtype policy.
- `method`: `round` (Emax baseline), `max-minus-one` (Emax/Emax−1), or
  `gptq-search` (GPTQ with Emax/Emax−1/Emax−2).
- `bits`: 4 or 8. GPTQ + search currently supports **BFP4 only**.
- Optional `output_splits`: widths of the actual output shards, e.g.
  `[40,40]` for an 80-output Linear. Must sum to the number of output rows.
  Without this field, groups span 16 outputs across the full matrix. This is
  also equivalent to aligned contiguous output shards whose widths are
  multiples of 16. It does not describe arbitrary interleaved sharding.

All selected tensors must be standard Linear weights **[out,in]**, packed on TT
as **[in,out]**. Attention head permutations, fused QKV/MLP, GDN reorders,
expert stacks, embeddings, convolutions and custom tiles are not inferred.
Use the lower-level APIs with an appropriate layout adapter for these cases.
Known tied embedding/output matrices are rejected when the config declares tying;
other architecture-specific parameter sharing must be checked in the loader.

Use `--dry-run` to read headers and print selected names/shapes/dtypes without
loading tensor data or writing files. A GPTQ dry run may omit `--hessians`; it
lists the weight names needing statistics. A real export requires them.

## Calibration files for GPTQ

`--hessians stats/hessians.json` takes an exact **checkpoint-weight-name → file**
map. Each file contains one `torch.save(H, ...)` tensor, where
`H = mean(X.T @ X)` over representative, unpadded calibration inputs. Relative
paths are resolved against the mapping JSON's directory, not the working directory:

```json
{
  "model.layers.0.mlp.gate_proj.weight": "layer0-mlp-input.pt",
  "model.layers.0.mlp.up_proj.weight": "layer0-mlp-input.pt"
}
```

Include **every** weight selected for GPTQ. Additional entries are allowed so the
same statistics directory can serve several recipes. Mapping two weights to
the same file reuses its factor when processed consecutively; do this only for
identical input activations and channel ordering. The exporter groups matching
Hessian paths within a shard and keeps one factor cached, avoiding an unbounded
factor cache. A shared input split across checkpoint shards may be factored again.

For an ordinary PyTorch model, use the existing capture helper:

```python
import json
from pathlib import Path
import torch
from tt_bfp_quant import capture_linear_inputs

stats_dir = Path("stats")
stats_dir.mkdir(exist_ok=True)
names = ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]
model.eval()  # original model, before applying any quantized weights
with capture_linear_inputs(model, names, device="cpu") as captured:
    with torch.inference_mode():
        for batch in calibration_batches:  # representative unpadded train prefixes
            model(**batch)
mapping = {}
for i, (name, accumulator) in enumerate(captured.items()):
    filename = f"hessian-{i}.pt"
    torch.save(accumulator.value(), stats_dir / filename)
    mapping[name + ".weight"] = filename  # verify this is the raw checkpoint key
(stats_dir / "hessians.json").write_text(json.dumps(mapping, indent=2))
```

That snippet captures just two modules; extend or batch the selected modules
according to RAM. A dense Hessian requires `4 * input_width**2` bytes. For a
large model, avoid keeping every layer's dense statistics in memory at once.
The exporter consumes saved Hessians; it does **not** perform calibration.
Reuse already collected TT statistics when available, provided their input
ordering matches these checkpoint matrices. See [integration notes](INTEGRATION.md)
for padding, tensor parallel reconstruction and calibration/evaluation separation.

## Output, memory and checks

The exporter supports `model.safetensors` or `model.safetensors.index.json` with
root-level shards. Names, shapes, dtypes, shard membership and index are preserved.
Selected values must fit the source floating dtype exactly, and ordinary BFP
repacking must preserve them. `--validate-ttnn` additionally checks the installed
TTNN host packer using the declared transpose and output splits.

Unchanged shards are copied directly. Changed shards are opened with safetensors,
transformed and written one at a time. Expect RAM for one shard, one working
matrix, one cached factor and quantization/factorization temporaries. This is not
a layerwise low-memory calibration engine. The full output needs roughly the
same disk space as the original floating-point checkpoint; BFP compression still
happens during the normal TTNN conversion. The source is never modified.

Standard top-level JSON/config/tokenizer/model/code/text assets and named
`chat_templates/*.jinja` are copied as independent files. Arbitrary nested custom
loader assets need manual copying. No checkpoint Python code is executed during
export. A new output path is required; it becomes visible only when all shards
succeed. Failed exports clean up their staging directory. The exporter does not
resume partial runs.

`tt_bfp_quantization.json` records the resolved recipe, changed tensors, source
paths, exponent choices, factor reuse, timing and validation. This is informational;
the inference loader does not need it. Source model config stays unchanged and
no runtime `quantization_config` is inserted.

Already quantized FP8/INT4/GPTQ/AWQ checkpoints are **not input formats** for this
command. Start with the original floating checkpoint. This package's GPTQ produces
TT BFP values, not the packed INT4/scales expected by community GPTQ loaders.

## Loading and TT constraints

The ordinary loader works with no package imports or hooks:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/models/search/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/models/search/Llama-3.1-8B-Instruct")
```

For TT implementations accepting `HF_MODEL`, point it at that directory and
use a new `TT_CACHE_PATH`. Retain the original model-family directory basename
under a variant parent, as in the example: some TT implementations derive
configuration from that name. Other scripts may expose different path/cache flags.

The output alone cannot change an inference script's BFP8 policy to BFP4. Use a
recipe matching the current precision settings, or select the desired policy via
an existing configuration flag. Do not assume a recipe changes deployment dtypes.
Layouts that regroup values can invalidate prepared exponents; validate at the
loader's actual packing boundary. Lower-level `search_packed` is available for
that integration, but such an adapter is outside this generic exporter.

Local integration tests create tiny Llama models, calibrate when needed, export
single/sharded checkpoints for both methods, reload with unmodified
`AutoModelForCausalLM`/`AutoTokenizer`, verify every parameter against the saved
checkpoint, and run forward inference. This proves HF checkpoint compatibility,
not accuracy or correct packing in every TT model. Run the optional TTNN checks
and your own held-out evaluation before drawing those conclusions.
