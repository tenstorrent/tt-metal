# KDA distribution plan — Galaxy 8×4, prefill (Phase 8)

> Phase 8 artifact per `~/.claude/references/model-bringup.md`. Scope: the **KDA layer's** mapping onto
> the production **Galaxy 8×4 torus**, optimized for **prefill**, plus a full-model sketch so the layer
> choice stays consistent with the hybrid. **Ends at a judgment-call gate** — options + estimates +
> recommendation below; no implementation until a mapping is chosen. All estimates are *first-order*
> (order-of-magnitude, to be firmed up by a profiler in the perf phase). Citations are `file:line` in
> the tt-metal main tree.

## TL;DR

Map KDA exactly onto the production MLA mesh — **SP=8 (axis 0, sequence) × TP=4 (axis 1, heads)** — and
honor the same block-boundary contract MLA/MoE use. KDA's *only* layer-specific collective is a
**fixed-size state-scan on the SP axis** (~9 µs/layer), which *replaces* MLA's expensive KV all-gather.
Prefill is **compute-bound** on this mapping (~8–18:1 compute:data-movement), and it slots into the
hybrid with zero change to the attention↔MoE boundary. Recommendation: **Option B**.

---

## 1. Why KDA is easy to distribute

- **The recurrence is per-head independent** — no cross-device communication *inside* the delta rule
  (the in-tree GDN TP impl confirms this: heads partitioned across TP, recurrence local, collectives
  only at in/out projections). So head-parallelism is nearly free.
- **Projections dominate compute (~95%);** the novel recurrence is ~5% (per-token: 39.5 M MACs of
  q/k/v/o/gate projections vs 2.1 M MACs of recurrence). So for the heavy part KDA distributes like a
  standard TP dense layer; the recurrence rides along cheaply.
- **KDA state is fixed-size** (`[heads, K, V]`, independent of sequence length) — this is what makes its
  sequence-parallelism cheap (see §4).

## 2. The mesh & the contract KDA must honor (from recon)

Production sparse-MLA (DeepSeek V3.2 / GLM, the family Kimi-Linear belongs to) runs on **SP=8 × TP=4**,
mesh `(8,4) = (sp_size, tp_size)`, **axis 0 = SP (sequence), axis 1 = TP (heads/hidden)** — hard-asserted
(`tests/sparse_mla/test_sparse_mla_perf.py:278-279,308-310`; `tt/mla/mla.py:1386`;
`tests/sparse_mla/sparse_mla_mesh.py:7-8`). Fabric = **FABRIC_2D, 2 links, `Topology.Linear`**, ~200
GB/s/dir/link → ~400 GB/s effective (`tt/mla/mla.py:259,360`; `test_sparse_mla_ccl_perf.py:34-41`).

The hybrid runs on **one fixed mesh**; attention and MoE use *different* parallelism on the *same*
devices, reconciled by collectives at each block boundary (`deepseek_v3` tree):
- Attention: TP shards heads (axis 1), SP shards sequence (axis 0). SP prefill = all_gather then
  reduce_scatter on axis 0 (`tt/mla/mla2d.py:85-95,188-211`).
- MoE: expert-parallel — 256 experts sharded across the mesh; token routing = `all_to_all_dispatch` /
  `all_to_all_combine` on **axis 0** (`tt/moe.py:213-214,450-487`); return to TP via `reduce_scatter`
  on axis 1 (`tt/moe.py:216-230`).
- Boundary: residual stream stays **TP-sharded on hidden**; entering MoE, `all_gather(axis1)` rebuilds
  hidden → gate → `all_to_all(axis0)` → experts → `reduce_scatter(axis1)`
  (`moe_decoder_block_2d.py:138-272`).

**Contract for KDA:** consume a TP-sharded hidden and produce one, exactly like MLA — then the
attention↔MoE boundary is unchanged. KDA replaces the attention op in the 3/4 linear-attention layers;
everything around it (norms, residual, MoE) is untouched.

> **Inconsistency flagged:** the older `deepseek_v3` MoE tree puts TP on axis 1 with `cols==8` (a `(4,8)`
> convention; `experts.py:286-297`), whereas the active sparse-MLA `deepseek_v3_d_p` tree uses `(8,4)`
> SP=8×TP=4. Kimi-Linear is sparse-MLA-family, so this plan anchors on **SP=8×TP=4 / axis0=SP**. Which
> tree Kimi-Linear integrates into is an open question for the full-model phase.

## 3. Candidate mappings (prefill, batch=1, per-KDA-layer, first-order)

Fabric BW = 400 GB/s eff; compute bracketed at 50–100 TFLOP/s bf16 (assumption — verify with profiler).
Total compute is fixed once all 32 chips are used; the lever is how you split them between SP and TP.

| Option | Split | compute/dev | proj collectives | state-scan | C:DM | Verdict |
|---|---|---|---|---|---|---|
| **A** | TP=32, SP=1 | 1.7 / 0.85 ms | **0.76 ms** (32-way, full-T) | — | ~1–2:1 | comms-bound; also breaks the TP=4 weight/boundary contract |
| **B** | **SP=8 × TP=4** | 1.7 / 0.85 ms | **0.094 ms** (4-way, T/8) | 0.009 ms | **~8–18:1** | **compute-bound; = production mesh** |
| **C** | TP=4, DP=8 | **13.6 / 6.8 ms** | 0.76 ms | — | ~9–18:1 | 8× worse latency for one long prompt (no SP compute-sharding); throughput-only |

(numbers at T=32K; they scale ~linearly in T — at T=128K, B: compute 3.4–6.8 ms, collectives 0.39 ms.)

**Reading it:** A and B do identical *total* compute (both use 32 chips), but B shards the projection-
collective *volume* by SP=8 **and** shrinks the TP group 32→4, so B's data-movement is ~8× smaller. C
keeps compute on the critical path un-sharded by sequence, so a single long prompt is 8× slower. **B
dominates for prefill latency and matches production.**

## 4. KDA-specific SP mechanics (the one genuinely-new piece)

MLA's SP axis **all-gathers the KV cache** to re-materialize the sequence — the flagged prefill
bottleneck (`indexer.py:420-425`). KDA needs no such thing: the recurrence carries a **fixed-size state**
across sequence spans, so SP over KDA is a **scan of that state along axis 0**, not a sequence gather.

Per chunked-linear-attention SP (Mamba/GLA-style), each SP rank owns a contiguous span of the sequence:
1. **Local, fully parallel:** each rank runs the chunked recurrence over its span → local output + a
   local state-transition `(Ā_span, correction)` (cumulative decay × span + accumulated update).
2. **Scan across axis 0 (8 ranks, line):** combine state-transitions to get each span's *incoming*
   prefix state. Exchanged object = `[heads/TP=8, K=128, V=128]` fp32 = **512 KB/boundary**; 7 line hops
   / 400 GB/s ≈ **9 µs** (negligible vs ~ms compute).
3. **Apply prefix correction** locally: add the `q·S_prefix` term to each span's output.

So KDA's sequence-parallelism is **structurally cheaper than MLA's** — it exchanges fixed-size state, not
the growing cache. Open sub-choice: sequential line-scan (simple, 7 hops) vs parallel/associative scan
(log-depth) — at SP=8 the linear scan's 9 µs is already negligible, so start simple.

## 5. Full-model sketch (consistency check)

On the shared `(8,4)` mesh, per hybrid layer:
- **KDA layers (3/4):** TP=4 shards the 32 heads → 8 heads/chip; in-proj col-parallel, out-proj
  row-parallel (`reduce_scatter`+`all_gather` on axis 1) — same as MLA; SP=8 state-scan on axis 0.
- **MLA layers (1/4):** unchanged (TP heads + SP KV all-gather).
- **MoE (every layer):** expert-parallel, `all_to_all` on axis 0, `reduce_scatter` on axis 1 — unchanged.
- **Boundary:** unchanged — KDA hands the MoE a TP-sharded hidden exactly as MLA does.

KDA introduces **no new boundary collective** and reuses the exact mesh + fabric the rest of the model
already uses. The only new primitive anywhere is the SP-axis state-scan inside the KDA op.

## 6. Tradeoffs

- **B vs A:** B wins on comms (8×) at no compute cost, and TP=4 is *required* anyway to match the MLA/MoE
  weight-sharding and boundary contract (TP=32 would need a bespoke reshard into the MoE). A is only
  interesting if KDA ran standalone (it doesn't).
- **B vs C:** C (DP over axis 0) only helps multi-sequence *throughput*; for a single long prefill it
  leaves 8× compute on the critical path. B is strictly better for prefill latency; C's throughput mode
  is a batching decision orthogonal to the layer mapping.
- **Cost of B:** implementing the SP state-scan (new vs the in-tree GDN, which does TP-only, full-seq
  per chip). Cheap in bytes, but it's real new code and must interoperate with the chunked kernel and the
  block-cyclic SP seq layout the MLA stack uses.

## 7. Recommendation

**Option B — SP=8 (axis 0) × TP=4 (axis 1)**, mirroring the production MLA mesh contract, with a
fixed-size SP-axis state-scan replacing MLA's KV all-gather. Prefill is compute-bound on this mapping;
data-movement (incl. the state-scan) is <10% of compute in the first-order model. It is the only option
consistent with the full hybrid without a bespoke reshard.

## 8. Open questions / what needs measurement (before/at the perf phase)

1. **Compute FLOP/s assumption** (50–100 TFLOP/s bf16) — the C:DM ratio depends on it; confirm with a
   Tracy dump of the TP=4 projection matmuls at 8 heads/chip.
2. **TP=4 matmul efficiency** at 8 heads/chip (head_dim 128) — is the per-chip projection matmul large
   enough to be efficient, or dispatch-bound?
3. **State-scan implementation** — sequential line-scan vs associative parallel scan; whether to fuse it
   with the chunk kernel (mirror the "fuse the SP all-gather into the score op" TODO at `indexer.py:420`).
4. **Chunk size × SP interplay** — the per-span chunked scan must tile cleanly into T/SP; interaction
   with the MLA stack's block-cyclic SP seq layout (`sparse_mla_mesh.py:51-55`).
5. **Which tree** Kimi-Linear integrates into (the `(8,4)` vs `(4,8)` convention inconsistency, §2).
6. **Decode** (out of scope here) will invert the balance to latency/collective-bound — a separate
   mapping analysis when decode is prioritized.
