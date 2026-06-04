# 05 · MLP / FFN — FF1, FF2, Fused Activation, `minimal_matmul`

The MLP is two matmuls per block: **FF1** (`H → 4H`, with fused activation) and **FF2**
(`4H → H`, with residual add). Plus the **attention-output** projection (`H → H`). This
file combines BGE-M3 (where MLP at batch 32 forced `minimal_matmul`), ViT (block-sharded
FFN with fused GELU), and the general matmul-tuning method.

---

## 1. The matmul families and their shapes

| Family | (M, K, N) shape pattern | CB pressure |
|---|---|---|
| Attention output | (B·S, H, H) | balanced |
| FF1 (+ activation) | (B·S, H, 4H) | wide N → big output CB |
| FF2 | (B·S, 4H, H) | wide K → big input CB |

The K vs N asymmetry drives the tuning: FF1's output CB is the constraint, FF2's input CB is.

**Gated MLP (SwiGLU / GeGLU — Llama, DeepSeek):** three matmuls instead of two —
`w2_in = SiLU(FF1(x)) * FF3(x)`, `y = FF2(w2_in)`. FF1 and FF3 share the input `x`, so
they share program configs and can be fused (double the FF1 N and split, or use the op's
`activation_fn="swiglu"` path). FF2 is the down-projection. All the tuning below applies
per-matmul identically.

---

## 2. Fuse the activation into FF1 — never split it

`fused_activation=(ttnn.UnaryOpType.GELU, True)` applies GELU as the matmul packs each
output tile. The SFPU's LUT hides inside the packer schedule:

```python
ff1 = ttnn.linear(
    x, w1, bias=b1,
    program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        ...,
        fused_activation=(ttnn.UnaryOpType.GELU, True),   # approx GELU, fused
    ),
    ...
)
```

BGE-M3 sweep (per-forward total): fused-approx GELU **49.3 µs** vs matmul→separate-GELU
**56.1 µs** vs separate-accurate **91.6 µs**. Fused approx adds only +0.8 µs over a bare
matmul. **Never run GELU as a separate op after FF1.** ViT does the same
(`fused_activation=(GELU, True)` in `ff1_matmul_program_config`).

For SwiGLU/GeGLU gated MLPs, the gate can be fused similarly where the op supports it
(some campaigns double the FF1 N and split — check the op's `activation_fn="swiglu"` path).

---

## 3. Program config — block-sharded 2D-mcast (ViT, small/mid batch)

When the activation fits L1, keep FF1/FF2 block-sharded so they chain with the
surrounding LN/residual without reshards (the ViT pattern):

```python
# FF1: H → 4H, fused GELU
ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(gx, gy),
    in0_block_w=H_tiles // gx,
    out_subblock_h=1,
    out_subblock_w=min(4, (4·H_tiles//gx)),   # cap by DST + L1
    per_core_M=seqL_t,
    per_core_N=4 · (H_tiles // gx),            # 4× expansion
    transpose_mcast=False,
    fused_activation=(ttnn.UnaryOpType.GELU, True),
)
# FF2: 4H → H, residual stays block-sharded
ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(gx, gy),
    in0_block_w=4 · (H_tiles // gx),           # wide K, chunked
    out_subblock_h=1, out_subblock_w=H_tiles // gx,
    per_core_M=seqL_t, per_core_N=H_tiles // gx,
    transpose_mcast=False, fused_activation=None,
)
```

For high-resolution (long seq), ViT **halves `in0_block_w`** on FF1 and chunks FF2's wide
K (`in0_block_w = 2·dim_t__x`) to fit the larger intermediate in L1. **The block-sharded
FF2 output and residual add stay in the same L1 config** so there's no reshard between FF2
and the residual.

---

## 3b. Matmul variant by regime (LLM prefill vs decode)

For generative LLMs the MLP matmul *variant* changes by phase (full detail in 08 section 2):

| Phase | Bottleneck | Variant | Activation |
|---|---|---|---|
| **Prefill** | compute | **Matmul 2D** (`MultiCastProgramConfig`) | DRAM interleaved |
| **Decode** | DRAM bandwidth (weights) | **DRAM-sharded** (`MultiCastDRAMShardedProgramConfig`) | L1 width-sharded |

Decode reads weights at ~240 GB/s (DRAM-sharded) vs ~190 GB/s (interleaved) on Wormhole —
that bandwidth *is* the decode MLP perf. Prefill is compute-bound so 2D mcast with
maximized subblock/`in0_block_w` wins. **Same weight, two configs — build both.**

This is distinct from the encoder/large-batch case below, where `minimal_matmul` is the
escape hatch for an L1 CB clash rather than a decode-bandwidth play.

---

## 4. `minimal_matmul` — mandatory when 2D-mcast clashes L1

At large batch the FF1 intermediate (`B·S × 4H`) is huge. 2D-mcast's in0 CB can exceed
the 1.57 MB L1 budget — BGE-M3 batch-32 FF1 via `ttnn.linear` produced **0 compiling
candidates**. `ttnn.experimental.minimal_matmul` streams K in `K_block_size` chunks,
reusing one in0 CB, collapsing the requirement from ~1.66 MB to ~256 KB:

```python
ff1 = ttnn.experimental.minimal_matmul(
    x, w1, bias=b1,
    program_config=ttnn.MinimalMatmulConfig(
        M_block_size=8, K_block_size=8, N_block_size=8,
        subblock_h=8, subblock_w=1,          # tall subblock unlocked by fp32_dest=False
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    ),
    compute_kernel_config=mlp_compute,        # LoFi, fp32_dest=False
    # GELU fused via the minimal_matmul builtin
)
```

**When to reach for `minimal_matmul`:**
- 2D-mcast `ttnn.linear` crashes with CB clash (1335/1326) at your shape — it's the fix.
- An upstream L1 handoff occupies L1, so the downstream matmul can't fit a 2D-mcast CB
  (BGE-M3 FF2 reads FF1's L1 output → must use `minimal_matmul`).

**When NOT to:** for shapes that *do* fit 2D-mcast (QKV, attn-out, any short-seq matmul),
`minimal_matmul` is slower (BGE-M3 QKV 510 vs 424 µs; Swin-L QKV 109 vs 88 µs). It is a
**CB-budget escape hatch, not a faster matmul.**

---

## 5. The subblock unlock — `fp32_dest_acc_en=False` + tall subblocks

```python
mlp_compute = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,        # cap h·w: 4 → 8
    packer_l1_acc=True,
)
```

With the cap at 8, BGE-M3 batch-32 swept subblock and found **`8×1` wins for both FF1 and
FF2** (pack tall rows, `subblock_w=1`):
- FF1: `8×1` = 61.10 ms vs `4×1` 61.32, `1×8` 61.12.
- FF2: `8×1` = 60.81 ms vs `1×8` 61.09.

The trend at large batch: **tall subblock (`subblock_h` large), `subblock_w=1`.** For
attention-output (wide N), the opposite — `1×8` won (sub-width tall): BGE-M3 attn-out
`1×4`(fp32_dest=True) → `1×8`(False) saved −34 µs/op × 24 = −816 µs.

**Always re-sweep subblock after flipping `fp32_dest_acc_en` — the new optimum is a
different point, not the old one with a doubled dimension.**

---

## 6. Fidelity walk for MLP

The MLP matmuls are the largest device-time buckets, so the fidelity walk pays most here:

| Step | BGE-M3 B32 win |
|---|---|
| FF1 HiFi4 → HiFi2 | −10 ms |
| FF2 HiFi4 → HiFi2 | −11.4 ms |
| FF1 HiFi2 → LoFi | −6 ms |
| FF2 HiFi2 → LoFi | −2 ms |

bf8b MLP matmuls tolerate LoFi (PCC holds). Walk down per family, full-model-PCC gated.

---

## 7. L1 handoff — FF1 → FF2 island (large batch)

At large batch where everything else is DRAM, a **local** L1 island still wins: write FF1
output to L1 so FF2 reads from L1:

```python
ff1 = ttnn.experimental.minimal_matmul(..., output_memory_config=ttnn.L1_MEMORY_CONFIG)
ff2 = ttnn.experimental.minimal_matmul(ff1, ..., output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

BGE-M3 B32: FF1 L1 → FF2 L1 saved −17 µs/layer = −408 µs/forward. **Gate it to the exact
shape** — at any other shape the L1 budget fails. Same pattern for attention-output → L1 →
post-attention LN.

**Critical caveat:** when an L1 handoff occupies L1, the downstream matmul often can't use
`ttnn.linear` anymore (CB clash) — switch it to `minimal_matmul` (BGE-M3 FF2 did exactly
this).

---

## 8. Attention-output projection (`H → H`)

Smaller than the MLP but same rules. At small batch, `ttnn.linear` 1D-mcast or auto is
fine. At large batch, the `fp32_dest=False` + `1×8` subblock unlock applies:

```python
ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(gx, gy),
    in0_block_w=4, out_subblock_h=1, out_subblock_w=8,   # 1×8 via fp32_dest=False
    per_core_M=..., per_core_N=...,
    fuse_batch=True, mcast_in0=False, fused_activation=None,
)
```

ViT keeps the self-output block-sharded on the full grid so it chains into the residual +
LN without reshards.

---

## 9. The matmul sweep method (any family)

1. **Lock the compute kernel**: bf8b in/weights + LoFi + `fp32_dest_acc_en=False` +
   `packer_l1_acc=True` (the last is mandatory — see 01 §4).
2. **`in0_block_w`**: divisors of `K_tiles`, prefer 4 or 8.
3. **Subblock**: for each ibw, try `(1,8)`, `(2,4)`, `(4,2)`, `(8,1)` with `h·w ≤ 8`.
4. **Verify** `out_block_h | per_core_M` and `out_block_w | per_core_N`.
5. **Reject** compile/PCC failures.
6. **In-model validate the top 2–3** — most standalone winners clash in-model. The one
   that survives ships.
7. **Use the real device grid** in the harness (never hard-code 8×8 on BH).

---

## 10. What didn't work (with data)

| Attempt | Result |
|---|---|
| FF1 2D-mcast `ttnn.linear` at large batch | 0 candidates compile (L1 blowup) → use `minimal_matmul` |
| FF separate GELU op | +6.7 µs vs fused; never split |
| FF2 `ttnn.linear` + fp32_dest=False with upstream L1 input | CB clash → `minimal_matmul` |
| FF subblock `4×2`/`2×4`/`1×8` at large batch | all worse than `8×1` |
| `minimal_matmul` for shapes that fit 2D-mcast | slower (escape hatch only) |
| FF1 → DRAM at large batch (instead of L1 island) | +1.95 ms regression |
| `fuse_batch=True` with rank-3 reshape | regressed |
| `M_block=16`/`N_block=16` (minimal_matmul) | CB clash |

---

## 11. Quick reference

| Question | Answer |
|---|---|
| FF1 activation | fuse via `fused_activation=(GELU, True)` — never separate |
| FF program config (fits L1) | 2D-mcast block-sharded, chain with LN/residual |
| FF program config (large batch, clashes) | `minimal_matmul` (streaming-K) |
| `minimal_matmul` rule | CB-budget escape hatch, NOT a faster matmul |
| Compute | LoFi + `fp32_dest_acc_en=False` |
| Subblock (wide-K / post-headsplit) | tall `8×1` |
| Subblock (wide-N: FF1, attn-out) | `1×8` |
| FF1→FF2 handoff (large batch) | FF1 output → L1, FF2 reads L1 (gate to shape) |
| Downstream matmul after L1 handoff | often must switch to `minimal_matmul` |
| Re-sweep after fp32_dest flip | always — new optimum |
