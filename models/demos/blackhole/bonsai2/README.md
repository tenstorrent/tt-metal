# Ternary-Bonsai-2-27B on Blackhole

## Summary

Ternary-Bonsai-2-27B is Qwen3.8-27B with ternary weights
(`PQ2_0`: 128-element groups of {-1, 0, +1} with an fp16 scale) and a blockwise Hadamard rotation folded into
the weights (block 1024, explicit ±1 signs). Its text config is identical to Qwen3.8-27B.

**Status:** the model runs end to end on Blackhole through the existing
[`models/demos/blackhole/qwen36`](../qwen36/README.md) code path, with **no model-code changes**. The first
generated tokens match a CPU Hugging Face reference exactly. The only Bonsai-specific step is converting the GGUF
to a dense bf16 Hugging Face checkpoint.

**Limitation:** the weights are not executed as ternary. The converter expands them to dense bf16, and the
qwen36 path stores them as bfp4 (MLP gate/up) and bfp8 (everything else). DRAM traffic per token, and therefore
decode speed, is the same as for the base Qwen3.8-27B. The ternary speedup needs the work listed in
[Next steps](#next-steps).

## Current performance

Batch 1, `traced_128` and `traced_4k` demo prompts.

| Mesh | ISL | TTFT | Decode |
| --- | --- | --- | --- |
| 1 x P150 (`MESH_DEVICE=P150`) | 128 | 0.83 s | 8.1-8.3 tok/s |
| 1 x P150 | 4k (2642 tok) | 12.5 s (212 tok/s prefill) | 8.1 tok/s |
| 2 x P300 = 4 chips, (1, 4) mesh (`MESH_DEVICE=P150x4`) | 128 | 0.35 s | 26.4 tok/s |

A separate decode-optimized stack (not in this branch) runs the same converted weights at ~77 tok/s/user at
batch 1 on the same 4 chips. That shows how much headroom the demo path has even before ternary weights.

## Projected ceiling

Batch-1 decode is bound by DRAM bandwidth (512 GB/s per Blackhole chip, 2 TB/s across 4 chips).

| Weights | Bytes read per token | 4 chips at 100% bandwidth | 4 chips, realistic | 1 x P150, realistic |
| --- | --- | --- | --- | --- |
| Today: dense (bfp4 gate/up, bfp8 rest) | ~21.5 GB | ~95 tok/s | 26 (this demo, measured); 77 (decode-optimized stack, measured) | 8 (measured) |
| True ternary (2 bits + fp16 scale per 128 weights ≈ 2.13 bits/weight) | ~7.0 GB | ~290 tok/s | ~135-170 tok/s | ~50-55 tok/s |

Speculative decoding (MTP or a draft model such as DFlash) on top of ternary weights could take per-user decode
toward the 200-290 tok/s range on 4 chips. That figure is a projection, not a measurement.

Quality: in a teacher-forced check, Bonsai scored 91.6 / 99.8 (top-1 / top-5 %) against 99.4 / 100 for the base
Qwen3.6 control on the same setup. It trades some accuracy for the smaller weight format.

## Next steps

1. **Ternary weight format in the matmuls.** Store the 2-bit codes and the per-128 fp16 scale in DRAM and
   unpack them in the matmul reader or compute kernels. The existing `bfp2_b` format cannot carry an fp16 scale
   per 128-element group, so this needs a new format or a custom matmul. This is the main lever: about 3x fewer
   bytes per token.
2. **Runtime Hadamard rotation.** Ternary execution needs the rotation kept out of the weights, so the
   activations have to be rotated on device (`y = W' (H D x)`, block 1024). This can be a fast Walsh-Hadamard op or
   a small block-diagonal matmul. It costs little next to the weight reads.
3. **Converter mode that keeps ternary.** Emit packed ternary weights plus the rotation signs instead of
   dense bf16.
4. **Decode-path optimizations.** Close the gap between the demo (26 tok/s) and the decode-optimized stack
   (77 tok/s) on 4 chips, independently of the weight format.
5. **Speculative decoding** (MTP / DFlash) once ternary decode runs.
6. **Accuracy evaluation** on standard benchmarks against the base Qwen3.8-27B.

Target hardware: 4 Blackhole chips (2 x P300, or 4 x P150).

## How to run

### 1. Convert the GGUF to a dense HF checkpoint

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

### 2. Run

Exactly as the qwen36 demo, pointing `HF_MODEL` at the converted directory:

```bash
export HF_MODEL=<out dir>
export TT_CACHE_PATH=<somewhere outside the HF dir>   # see qwen36 README
export MESH_DEVICE=P150          # one P150; or P150x4 for a (1, 4) mesh
pytest models/demos/blackhole/qwen36/demo/text_demo.py -s --timeout=7200 -k "traced_128 and not traced_128k"
```

`--timeout` matters on the first run (weight-cache build); `-k traced_128` alone also matches `traced_128k`.
On a host with several P300 boards, pick the four chips that are Ethernet-cabled into one ring with
`TT_VISIBLE_DEVICES` (otherwise the mesh downgrades to 2x1).

### 3. Check against the CPU reference

```bash
python models/demos/blackhole/bonsai2/cpu_reference.py <out dir>
```

Prints the top-5 first-token distribution and 8 greedy tokens for the same 128-token prompt `traced_128` uses.
