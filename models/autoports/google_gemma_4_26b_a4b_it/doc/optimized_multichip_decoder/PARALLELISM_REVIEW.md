# Parallelism review: is TP=4 the right decomposition for this model?

Read on 2026-08-17 from a second machine, from the committed source and this
stage's own `profile_accounting.md` / `work_log.md`. No new hardware runs; every
number below is either quoted from those artifacts or derived arithmetically
from the config, and each is tagged.

## What the implementation does

Uniform Megatron-style tensor parallelism over the 1×4 P300C ring, applied
identically to every projection in every layer:

| Family | Shard | Per-rank width |
|---|---|---|
| QKV | column | 4 of 16 Q heads; 2 of 8 KV heads (sliding), 1 KV head (full) |
| O | row → all-reduce | K = 1024 (sliding) |
| dense gate/up | column, **packed into one matmul** | N = 544 each, 1088 packed |
| dense down | row → all-reduce | K = 544 |
| expert gate, expert up | column, **two separate sparse matmuls** | N = 192 each |
| expert down | row → all-reduce | K = 192 |
| router | replicated FP32, no collective | N = 128 |

Residual stays replicated BF16 in DRAM; three hidden-width ring all-reduces per
layer (O, dense-down, expert-down) — 90 per token across 30 layers.

## Already swept by this stage — do not re-run

Recorded with numbers, so these are closed:

- **Sharded/L1-resident residual chain (R11/R22).** `optimized_decoder/work_log.md:89`
  and `README.md:106`: sliding PCC 0.994795/0.994694 (under the 0.995 bar), batch-1
  *slower* at 1.378/1.347 ms, batch-32 hits an L1 circular-buffer clash. Rejected.
  This is why the profile still shows replicated residual norms at 34.7–44.5 µs
  each rather than the 5–6 µs L1-sharded head norms.
- **Persistent async all-reduce.** Measured whole-layer 1.08369 (sliding) /
  1.11153 (full); kept for full layers, rejected for sliding
  (`optimized_multichip_decoder/work_log.md:61`). The `persistent_default` in
  `multichip_decoder.py` matches that result — it is a measured choice, not an
  oversight.
- **Expert matmul geometry.** Block 22/44/88 × N=2/3/4 swept; block 44, N=2 kept
  (`work_log.md:57-58`). BFP4 experts rejected on prefill PCC 0.994712 (`:64`).
- **Fusing the dense-down and expert-down collectives.** Explored and recorded as
  illegal for a stated op-contract reason (`work_log.md:92-97`).

Worth noting the last one is doubly blocked: in `optimized_decoder.py:942-961`
the dense and MoE branches are *parallel* reads of the same residual, but each
applies its own RMSNorm (`post_ff_ln_1`, `post_ff_ln_2`) **before** the add. Norm
is nonlinear, so the two row-parallel partials cannot be summed and reduced once
even if the op contract allowed it.

## The one clear untried lever: pack the expert gate/up

The expert gate and up projections are the most expensive family in the layer,
and they are the only place in this model where a packable gate/up pair was left
unpacked.

Measured, from `profile_accounting.md`:

| | op count | device time |
|---|---:|---:|
| dense gate/up (**packed**) | 1 | 10.783 µs |
| expert gate/up (**separate**) | 2 | 75.613 + 75.550 µs |
| expert down | 1 | 19.088 µs |

Derived: with top-8 of 128 experts at BFP8, one expert gate matmul moves about
8 × 2816 × 192 ≈ 4.3 MB of weights, so 75.6 µs is ≈ 57 GB/s against this stage's
own 512 GB/s floor — about 9× off bandwidth-bound. The stage's byte accounting
says the whole sliding layer needs 25.1 MB/token/device ≈ 49 µs at that floor,
while expert gate/up/down alone spend ≈ 170 µs. So these ops are limited by
per-op overhead and grid utilisation at N = 192 (6 tiles), not by weight traffic.

Packing gate|up into one sparse matmul gives one op at N = 384 instead of two at
N = 192: same bytes, half the launches, twice the width to fill the grid with.
The precedent is in the same model — `packed_mlp_gate_up` in
`optimized_decoder.py:1197-1220`, which is why the dense pair costs 10.8 µs.

Implementation sketch, following how the dense path already does it:

1. At load, concatenate `expert_gate` and `expert_up` per expert into
   `[1, 128, 2816, 384]` — a host-side concat of tensors that are already stored
   adjacently, no extra bytes. `_packed_gate_up_mesh_source` in
   `multichip_decoder.py:55` is the existing helper for getting the per-rank
   interleave right; the expert case needs the same treatment one dimension in.
2. One `ttnn.sparse_matmul` with the packed weight and `in0_block_w` re-swept
   (the retained block-44 geometry was tuned for N = 192).
3. Slice the result into gate/up halves for `apply_geglu`, exactly as
   `_dense_mlp` slices its packed output.

Caveat that decides whether this is worth doing: the expert path lives in
**shared** code (`models/demos/gemma4/tt/experts/decode.py`), which the
`models/demos/gemma4` model also uses. Either add a packed variant behind a flag
there, or override it in the autoport. Do not silently change the shared op for
another model's benefit.

Measure with the existing harness before and after — one number decides it:

```bash
GEMMA4_FUNCTIONAL_DECODER_PERF=1 pytest \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
  -k "perf_profile and batch1 and sliding"
```

## Structural observation: the MoE axis is never used

This is a 128-expert, top-8 model with a 704-wide expert intermediate. The
pipeline sharded that intermediate 4 ways (192 per rank) and left the expert
dimension untouched, which is the decomposition a dense model wants. The
consequence is the narrow-N problem above: every rank touches all 8 selected
experts at 1/4 width, rather than each rank owning whole experts.

Expert parallelism is the decomposition the architecture invites — 32 experts per
rank at the full N = 704 (22 tiles), with an all-to-all replacing the
expert-down all-reduce. The stage got close to this: `work_log.md:82-92` records
that a fixed selected-expert projection at N = 768 was feasible but that "that
feasibility does not complete a coherent active-expert decoder path", citing
device routing and the rank-5 `[1,128,K,N]` weight layout. So it was assessed as
incomplete rather than measured as worse.

The honest trade-off, unmeasured either way:

- At **batch 1** EP is badly imbalanced — 8 experts over 4 ranks, worst case all
  8 on one rank — so it may well lose to TP at the latency the benchmark reports.
- At the **served batch 32** there are 256 expert-token assignments per layer,
  which spread far more evenly, and this is the regime the throughput target
  actually covers.
- A hybrid EP=2 × TP=2 keeps N = 352 with ~4 experts per rank and bounds the
  imbalance.

Nothing here justifies re-architecting on a hunch. It justifies measuring EP at
batch 32 before treating uniform TP=4 as settled for this model, because the
pipeline chose it by template rather than by comparison.

**Converging analysis, with evidence this review lacked.**
`doc/IMPROVEMENT_EXPERT_PARALLEL.md` (pushed independently while this was being
written) reaches the same diagnosis — the 192-wide per-device expert matmul is the
bottleneck and expert parallelism is the lever — and supplies two things this
review could not:

- `doc/multichip_decoder/mesh_plan.md` **rejected EP=4 analytically, not by
  measurement**, on batch-1 dispatch/combine and imbalance grounds — the same doubt
  raised above.
- The fleet corpus already ran that experiment on **gpt-oss-20b**, same 1×4
  Blackhole mesh: EP4 with whole experts per rank beat TP-fracture decisively
  (decode 0.599 vs 0.656 ms, prefill 26.7 vs 39.8), and an analytic EP drop was
  recorded there as the inferior call.

That resolves the batch-1 doubt in EP's favour on prior evidence, so EP is the
larger lever and should be measured first. The expert gate/up packing proposed
above stays worth doing and is complementary: it is a contained change that keeps
the current TP decomposition, whereas EP changes the weight layout and the
collective structure.

## Memory: the full-attention KV cache is stored twice

Derived from the config and `model.py:273`. The model has
`num_global_key_value_heads = 2` for full-attention layers, and the mesh is 4
ranks with `local_kv_heads = 1`, so 4 KV heads are stored where the model has 2.
Rank r holds Q heads [4r, 4r+3], which map to KV head 0 for ranks 0–1 and KV
head 1 for ranks 2–3: each pair keeps its own copy.

At the advertised 262,144-token context that is 2048 blocks × 1 head × 128 ×
512 × 2 B = 268 MB per full layer per rank, ×5 full layers = **1.34 GiB/rank, of
which 0.67 GiB is pure duplication**. Two global KV heads simply do not divide a
4-rank mesh, so this is the ordinary cost of TP=4 here rather than a defect — but
it is worth naming, because it is the same DRAM budget that
`doc/tti_release/AUTOFIX.md` reports as making BF16 expert weights infeasible at
the unchanged context.

## Secondary hypothesis, needs one experiment

The dense-branch all-reduce is consumed by `post_ff_ln_1` immediately, before any
expert work is issued. Ring collectives run on fabric/ethernet resources rather
than the compute grid, and `all_reduce_async` exists to let a collective be
issued and consumed later. If issuing the dense-branch reduce *before* the
expert matmuls and consuming it after genuinely overlaps on this command-queue
setup, it hides ≈17 µs per layer (≈0.5 ms/token) behind ≈170 µs of expert
compute. If TTNN serialises them on one queue it hides nothing. That is a
single-experiment question, and this stage's own plan already listed "fused
matmul+RS/AG+matmul" in the same area (`work_log.md:26`).
