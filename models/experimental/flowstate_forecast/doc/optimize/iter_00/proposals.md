# Round 1 proposals (G2 automatic; screened with the reduced-layer bench)

The estimates are for paper. Their basis is the gap to the bandwidth floor, assuming about half of it is recovered.

1. **Matmul program configs per role** (§5.1): B, C, gate, w1, w2, w3 and embed, swept in one bench job. About −90 ms. No precision change.
2. **Fused SwiGLU** (`minimal_matmul fuse_swiglu`, keeping the b1/b2 biases) (§5.1). About −40 ms. If it is not available, fold silu into the program config's fused activation (about −18 ms).
3. **Time-segmented route-K scan** over all cores (an in-kernel carry fix-up; fp32). About −20 ms at paper and about −100 ms at long_context. Micro-benchmark it first.
4. **Elementwise fusion**:
   - sigmoid as an input activation of the gate multiply;
   - D·u + y as one op;
   - the z + y add in the w3 epilogue (§5.1).
   About −28 ms.
5. **Padding** for B < 32 (single, single_long, long_context): fold time segments into rows (builds on #3). Guard shapes only.
6. **Deferred:**
   - bf16 x into C, and bf16 B/C weights. These are precision changes: they need a per-row table, and X if they change the policy.
   - Having the last layer's scan write only rows ≥ r0.

The realistic total for #1-#4 is 375 → 200-230 ms. The performance class is likely to be retired unless #1 makes the dense matmuls compute-bound.
