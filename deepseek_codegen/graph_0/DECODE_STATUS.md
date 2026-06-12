# Update — decode ~158.5 ms device perf (~6.7× since E47)

**0.98 → ~6.5 samp/s** · E47 was 1.02 s/token (0.98 samp/s).
**Correctness:** next-token argmax = 100% throughout; each effort ≈-numerically-equal to the prior step. Golden-logits PCC fell ~0.92 → **0.90** at the bf4 `moe_compute` swap (effort 1) and held there (bf4 floor).

**Device perf** (per decode step; 61 layers = 3 dense + 58 MoE):
```
prologue + attn×61 + dense×3 + moe×58 + lm_head
0.019    + 0.890×61 + 0.619×3 + 1.739×58 + 2.439  ≈ 158.5 ms
```

| effort | what | commits | speedup |
|---|---|---|---|
| **Fused MoE experts** | sparse_matmul path → `ttnn.experimental.moe_compute` (bf4) | `dcec039`,`6bca510` | **+314%** |
| **Attention layout** | MLA chains→L1, `permute→ttnn.transpose`, reshape elim, RoPE op | `bf666df` | +25% |
| **MoE tail precision** | BF16 weighted-k-sum → `FastReduceNC` fast path (170→15 µs) | `012ba4c`,`70d3616` | +15% |
| **Reshard + fuse collectives** | local `mesh_partition` reshard → qkv-down 3 reductions→1 `all_gather+sum`; router too | `16b3ab2`,`f8cebd8`,`b3e0f41` | +7% |

**Pure CCL share** (explicit `all_gather`/`reduce_scatter`/`all_to_all`):

| part | attn | dense | moe | lm_head | **full model** |
|---|---|---|---|---|---|
| CCL % | 23% | 57% | 32% | 24% | **~29%** |

Excludes the MoE combine fused inside `MoECompute` (~35–40% incl. it). **We don't run trace here** — the full model is served with trace, which CCLs are probably the most affected by, so real CCL cost is likely lower.

**UNBLOCKED (2026-06-12):** MoE-phase collective restructuring (was blocked by the `moe_compute selective_reduce_combine` L1 fragility, #46208) now works — vendored **PR #46544** (`e4bc86b`, dynamic mux-buffer sizing) gives the combine adaptive L1 headroom, so the previously-deterministic concurrent-combine hang is gone. First win on top: **shared-FFN w1+w3 CCL fusion** (`reduce_scatter+all_gather → all_gather+FastReduceNC`), measured **−87 µs/MoE-layer ≈ −5.0 ms e2e** (~158.5 → ~153.5 ms), golden PCC held at 0.8989. Reopens dense-MLP fusion + MoE L1-sharding (the "wide all-reduce is bandwidth-negative" projection was disproven by measurement — the HiFi4 reduce_scatter dominates).

**Latest (E_idxnorm, commit `55d3fe8`):** indexer k_norm runs in bf16 (dropped the fp32 up/down-cast round-trip, both layers; −4 typecasts/decode-step). Full-graph live-outs **bit-identical** to the ropecse golden (the indexer is a top-k *selection* signal → bf16-tolerant). ~−0.27 ms. After this, the **safe typecast/CSE vein is exhausted**: remaining fp32 casts are precision/integer-critical (embedding+RMS residual, dense-MLP cross-device reduction accumulation, the UINT32 KV-cache-index `where`, pre-lm_head norm); the cache-index/`where` is already computed once and shared across layers. Bigger levers stay blocked (MoE combine #46208) or structural+hang-risky (head-relayout permutes).

**Compiler could auto-emit** (each generalizes a manual win above):
- lower transposing `permute` → `ttnn.transpose` (~10× faster) — *Attention layout*
- fuse N reductions of one matmul's output slices → 1 reduction — *Reshard + fuse collectives*
- tile-aware gather dim: pack partials into a tile dim, not the leading dim — *stat-gather repack*
- emit combine/reduce in the dtype that hits `FastReduceNC` (BF16) — *MoE tail precision*
- CSE loop-invariant RoPE cos/sin broadcasts — *already covered* (hand-added code, outside the compiled graph)

**Detail:** [MoE-compute journal](https://github.com/tenstorrent/tt-metal/blob/mvasiljevic/deepseek-decode-sharding/deepseek_codegen/graph_0/MOE_COMPUTE_JOURNAL.md) · [Decode-sharding journal](https://github.com/tenstorrent/tt-metal/blob/mvasiljevic/deepseek-decode-sharding/deepseek_codegen/graph_0/DECODE_SHARDING_JOURNAL.md) · [Latest perf profile](https://github.com/tenstorrent/tt-metal/tree/mvasiljevic/deepseek-decode-sharding/deepseek_codegen/graph_0/perf_reports/ropecse_2026_06_12_07_08_23)

<sub>speedup = per-effort gain vs the prior step. Full-model device perf projected to s/token (anchored to E47). +5% RoPE-CSE / +stat-gather tile-repack carry it the last bit to ~6.5 samp/s.</sub>
