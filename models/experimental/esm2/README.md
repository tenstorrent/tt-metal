# ESM-2 650M (esm2_t33_650M_UR50D) — TTNN implementation for Tenstorrent Blackhole

SPDX-License-Identifier: MIT

Production-style TTNN implementation of [facebook/esm2_t33_650M_UR50D
](https://huggingface.co/facebook/esm2_t33_650M_UR50D): a 33-layer,
651M-parameter masked-LM protein language encoder (hidden 1280, 20 heads,
FFN 5120, rotary position embeddings, tied decoder head, vocab 33).
Semantics verified against the pinned checkpoint and the `transformers`
EsmForMaskedLM FP32 reference.

## Package layout

| Path | Contents |
|---|---|
| `tt/esm2/` | device implementation: `config.py`, `loader.py` (safetensors → canonical names), `ttnn_backend.py` (`TtnnEsm2`), `reference_layers.py` (CPU FP32 twin, same op map) |
| `demo/` | standalone CLI: FASTA → per-residue embeddings/logits |
| `reference/` | strict FP32 reference forward (host) |
| `tests/` | portable tests (CPU-only + on-device) |
| `benchmarks/` | PyTorch FP32 baseline script |
| `docs/` | `esm2.md` (implementation notes), `benchmark.md` (measured evidence) |

## Precision policy (BF16 anchor; every exception is measured)

BF16 weights/activations with fp32 accumulation, plus measured fp32
exceptions: fp32 residual stream (bf16 residual rounding dominates hidden
error otherwise), fp32-compute softmax composite (the runtime's
`ttnn.softmax` rounds internally), rotary tables and additive mask on
host, and `out_fp32` at selected matmul sites ({qkv, pv, ao, ffn2}).
Full rationale + measurements: `tt/esm2/ttnn_backend.py` and
`docs/esm2.md`.

## Measured results (see `docs/benchmark.md` for the full table)

- **smoke PASS** (bf16): max NRMSE 0.030 vs FP32 oracle, argmax exact.
- **bringup 4/8 failures — both documented as measured limits** (gates
  NOT widened):
  - 3 argmax flips on bf16 near-tie logits (top-2 margin 0.014; bf16
    rounding error ~7× the margin; the NVIDIA A100 in bf16 flips the
    same fraction of its own cases — this is intrinsic bf16 precision,
    not a port bug).
  - 1 long-sequence hidden NRMSE 0.042 (gate 0.04) at the runtime's
    implementation floor (every lever measured and closed; see
    `docs/benchmark.md` for the full rejection table).
- **Latency** vs same-host PyTorch-FP32-CPU baseline:
  - L=1026: 181.5 ms @ 5658 res/s vs 3270 ms CPU (**18.0×**)
  - L=144: 79.9 ms vs 324 ms CPU (**4.1×**)
  - Traced short-sequence replay: **14.9 ms** (dispatch-bound fast path)
  - A100 FP32 p50 32 ms recorded as context.
- **Memory**: 50% weight reduction via bf16 (1.3 GB → 0.65 GB).

## Known limitations (honest, measured)

1. BF16 argmax near-ties at 3 bringup positions: two residues have
   near-identical logits (margin < bf16 precision); the port matches
   the FP32 reference within its NRMSE gate but flips the argmax on
   these intrinsic ties.
2. Long-sequence hidden NRMSE at 0.042 vs 0.04 gate: attributed to the
   runtime's bf16 dense-matmul rounding (measured 2–3.5× over exact
   bf16 rounding across 33 layers); all in-graph levers measured and
   closed (see `docs/benchmark.md` "sized-and-closed optimizations").
3. `ttnn.transformer.scaled_dot_product_attention` is not used: its
   `attn_mask` input produces incorrect results on this runtime
   (measured 0.53 additive / 0.35 multiplicative NRMSE); the port keeps
   manual attention (matmul + mask + composite softmax).

## Running

```bash
# CPU demo (host only, no device needed)
python demo/demo.py --fasta demo/example.fasta --checkpoint /weights \
    --device cpu --precision fp32 --out demo/out.npz

# Portable tests (no device required)
python tests/test_portable_model.py

# On-device tests (requires TT hardware + checkpoint)
python tests/test_ttnn_bringup.py
```

Out of scope: >1024-residue inputs (model limit), ESMFold structure
heads, fine-tuning, biology-leaderboard claims.

## License

- Code: MIT (SPDX headers on all source files)
- Model weights: [MIT](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
- This port is an independent implementation; no Meta code was copied.
