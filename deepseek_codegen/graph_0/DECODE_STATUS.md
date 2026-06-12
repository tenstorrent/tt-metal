## Update — decode now ~158.5 ms device perf (~6.7× since E47, 0.98 → ~6.5 samp/s)

**Now: ~0.15 s/token · ~6.5 samples/sec — ~6.7× since E47** (E47 = 1.02 s/token, 0.98 samp/s).
Hand-tuned the EmitPy `main.py`; PCC held throughout (argmax 100%, logits cos ≈ 1.0 vs baseline).

**Full-model device perf** (per decode step, ms; 61 layers = 3 dense + 58 MoE):
```
device perf = prologue + attn×61 + dense×3 + moe×58 + lm_head
            = 0.019    + 0.890×61 + 0.619×3 + 1.739×58 + 2.439
            ≈ 158.5 ms
```

| effort | what | commit(s) | speedup |
|---|-----|---|---|
| **Fused MoE experts** | swap sparse_matmul path → `ttnn.experimental.moe_compute` (+ tail `reduce_scatter`; freeing dead weights fixed a combine hang) | `dcec039`,`6bca510` | **+314%** |
| **Attention layout** | MLA chains→L1, `permute→ttnn.transpose` (~10×), reshape elimination, RoPE op (attn 1.97→1.10 ms/layer) | `bf666df` | +25% |
| **MoE tail precision** | drop redundant FP32 upcasts, then BF16 weighted-k-sum → hits `FastReduceNC` fast path (Reduce 170→15 µs) + BF16 `reduce_scatter` | `012ba4c`,`70d3616` | +15% |
| **Reshard + fuse collectives** | re-shard q_a locally (`mesh_partition` replaces a `reduce_scatter`) so qkv-down's 3 reductions → 1 `all_gather+sum`; same for router all-reduce | `16b3ab2`,`f8cebd8`,`b3e0f41` | +7% |

**Biggest lever, still blocked:** MoE = ~59% of decode device perf, but expert/combine collective restructuring trips the `moe_compute selective_reduce_combine` L1-overlap fragility — **#46208** (term-barrier **#43444** fixed by **#45764**; residual worked around with `num_buffers 14→13`).

**Patterns worth automating in the compiler:** (a) fuse N reductions of slices of one matmul output → 1 reduction; (b) tile-aware gather dim (pack partials into a tile dim, not the leading dim); (c) CSE loop-invariant RoPE broadcasts across layers; (d) emit combine/reduce in the dtype that hits `FastReduceNC`.

*speedup = how much faster that effort made it (relative to just before it). Full-model device perf projected to s/token (anchored to E47 = 1.02 s/token). Smaller follow-on wins (tile-packing, RoPE CSE) carry it the rest of the way to ~6.5 samp/s.*
