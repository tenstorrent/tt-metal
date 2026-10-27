# Ternary-Bonsai-2-27B on Blackhole

PrismML Ternary-Bonsai-2-27B is Qwen3.8-27B with ternary weights
(`PQ2_0`: 128-element groups of {-1, 0, +1} with an fp16 scale) and a blockwise Hadamard rotation folded into
the weights (block 1024, explicit ±1 signs). Its text config is identical to Qwen3.8-27B, so it runs on the
existing [`models/demos/blackhole/qwen36`](../qwen36/README.md) code path with **no model-code changes**: the
only Bonsai-specific step is converting the GGUF to a dense bf16 Hugging Face checkpoint.

## 1. Convert the GGUF to a dense HF checkpoint

```bash
python models/demos/blackhole/bonsai2/convert_gguf_to_hf.py \
    Ternary-Bonsai-2-27B-PQ2_0.gguf  <Qwen3.8-27B snapshot dir>  <out dir>
```

- Ships its own GGUF reader: upstream `gguf-py` does not know types 142 (`PQ2_0`) / 143 (`PTQ1_0`).
- Unfolds the rotation: `W_dense = W' @ blockdiag(H) @ diag(signs)` (the embedding too), so `y = W_dense x`
  reproduces the GGUF runtime's `y = W' (H D x)` exactly.
- Applies the GGUF -> HF layout fixups (GDN v-head regrouping of `in_proj_qkv/z/a/b`, `A_log`, `dt_bias`,
  `conv1d` rows; norms stored as `1 + w` except `ssm_norm`; `ssm_a = -exp(A_log)`). Each fixup is chosen by
  cosine similarity against the base Qwen3.8-27B tensor, which is why the base snapshot is an input (it also
  supplies config and tokenizer). `--validate-only` prints those checks without writing.
- Output is ~51 GB of bf16 safetensors. Ternary tensors land at ~0.88 cosine vs the base weights, as expected.

## 2. Run

Exactly as the qwen36 demo, pointing `HF_MODEL` at the converted directory:

```bash
export HF_MODEL=<out dir>
export TT_CACHE_PATH=<somewhere outside the HF dir>   # see qwen36 README
export MESH_DEVICE=P150          # one P150; or P150x4 for a (1, 4) mesh
pytest models/demos/blackhole/qwen36/demo/text_demo.py -s --timeout=7200 -k "traced_128 and not traced_128k"
```

`--timeout` matters on the first run (weight-cache build); `-k traced_128` alone also matches `traced_128k`.

## 3. Check against the CPU reference

```bash
python models/demos/blackhole/bonsai2/cpu_reference.py <out dir>
```

Prints the top-5 first-token distribution and 8 greedy tokens for the same 128-token prompt `traced_128` uses.

## Measured

| Mesh | ISL | TTFT | decode (batch 1) |
| --- | --- | --- | --- |
| 1 x P150 (`MESH_DEVICE=P150`) | 128 | 0.83 s | 8.1-8.3 tok/s |
| 1 x P150 | 4k (2642 tok) | 12.5 s (212 tok/s prefill) | 8.1 tok/s |
| 2 x P300 = (1, 4) mesh (`MESH_DEVICE=P150x4`) | 128 | 0.35 s | 26.4 tok/s |

Weights are bfp4 for the MLP gate/up and bfp8 elsewhere (~24 GB). On a host with several P300 boards, pick the four
chips that are Ethernet-cabled into one ring with `TT_VISIBLE_DEVICES` (otherwise the mesh downgrades to 2x1).

The first generated tokens match the CPU reference exactly.

Ternary weights do not buy speed here: after conversion the weights are stored as bfp4/bfp8 like any Qwen3.8-27B
checkpoint, so DRAM traffic per token (~23 GB) is the same as for the base model. A speedup needs a 2-bit /
ternary weight format in the matmuls.
