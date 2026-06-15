There are referent input and output values for MLA layer, indexer and kv cache in bit_sculpt repository next to this one.
Available values are for layers 0, 30, 60.


You can load and use reference values like this:

```
from safetensors.torch import load_file
import glob, torch

def load_stream(stream_dir):
    """Load one (stream, layer) → full tensor. Single-file or chunked, both work."""
    files = sorted(glob.glob(f"{stream_dir}/rows_*.safetensors"))
    parts = []
    for f in files:
        d = load_file(f)
        (key,) = d.keys()          # exactly one tensor per file
        parts.append(d[key])
    return torch.cat(parts, dim=0)

run = "results/deepseek-v32"
L = 0

mla_in  = load_stream(f"{run}/module_io/mla_input_layer_{L}")     # (5120, 7168) bf16
mla_out = load_stream(f"{run}/module_io/mla_output_layer_{L}")    # (5120, 7168) bf16
idx_in  = load_stream(f"{run}/module_io/indexer_input_layer_{L}") # (5120, 7168) bf16  (== mla_in)
logits  = load_stream(f"{run}/dsa/indexer_logits_layer_{L}")      # (5120, 5120) fp32; logits.to(torch.bfloat16)
topk    = load_stream(f"{run}/dsa/dsa_topk_indices_layer_{L}")    # (5120, 2048) int32
kv      = load_stream(f"{run}/kv_cache/layer_{L}")                # (5120, 576)  bf16
```
