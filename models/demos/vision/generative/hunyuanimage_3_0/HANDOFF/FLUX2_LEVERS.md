# FLUX2 → HunyuanImage-3.0 optimization backlog

Mined from `sadesoye/flux2_1024` (92 non-merge commits, merge-base `721579be15`,
tip `91fca838f3`; branch reached ~7.9 s/image on Blackhole Galaxy). Every commit
is accounted for below (see the full coverage table). Target model: our T2I MoE
LLM on `hunyuan-image3-unified` (box `bh-glx-exp-b04u14`, MeshShape(8,4), 32 chips,
FABRIC_1D). Baseline ~84 s/image warm @1024²/50-step. Latest Tracy: CCL-bound —
ReduceScatter+AllGather ~30% device-kernel time (> matmul 23%), ~55% wall-clock
with cross-chip sync.

Discipline: measure on TRACED WALL-CLOCK (ONDEVICE_E2E_TOTAL_LATENCY_S / ms/step),
PCC-gate every change, one lever per A/B.

---

## KEY STRUCTURAL FINDING (drives the cherry-pick plan)

**flux2 is a DIFFERENT framework** (dense DiT on `models/tt_dit/`), NOT our MoE
`_stubs/`. But the ttnn C++ kernels flux2's speedups ride on **already exist in
our tree** (apande-TT fork, merge-base with flux2 = `28e2841a2b`). flux2 only
*modifies* them — it does NOT introduce them:

| op dir (present in OUR tree) | flux2 delta vs our HEAD |
|---|---|
| `experimental/minimal_matmul/` | +610/-99 (19 files): `fuse_swiglu`, virtual-concat, `minimal_matmul_split`, `dit_minimal_matmul_addcmul_fused`, skip-x1 |
| `experimental/ccl/all_gather_minimal_matmul_async/` | +704/-367 (11 files): addcmul fusion, swiglu, chunks, skip-x1, Ring-only guard |
| `experimental/ccl/minimal_matmul_strided_reduce_scatter_async/` | +64/-19 (7 files): addcmul, virtual-concat, skip-x1 |
| `experimental/transformer/fused_distributed_rmsnorm/` | +304/-135 (20 files): per-head mode (no stats all-gather) |
| `transformer/sdpa/` | +3714/-780 (60 files): ring-joint sharded set, split-forward both-links |
| `experimental/ccl/strided_reduce_scatter_async/`, `ring_attention_all_gather_async/` | small: skip-x1, split-forward |

**Only ONE op dir is brand-new on flux2:**
`experimental/ccl/dit_fused_distributed_groupnorm/` (commit `0d4dde7478`) — the
on-device-VAE unblock (deferred).

**Consequence:** a full rebase onto flux2 is the WRONG move (flux2's model/VAE/
encoder tree is entirely different -> enormous conflict surface we don't want).
The right move is **targeted op-dir updates** (`git checkout refs/tmp/flux2 -- <dir>`
+ reconcile CMake/nanobind registration + rebuild) for the specific fused ops,
and one clean cherry-pick of the isolated new groupnorm dir. **FSDP itself needs
NO C++ change** — see Lever 1.

---

## PRIORITIZED BACKLOG (actionable levers)

Legend — CLASS: `kernel` = portable ttnn C++ op (update/cherry-pick + rebuild);
`config` = flux2-shape-specific pattern, adapt to OUR shapes; `model` = tt_dit
technique to re-implement in `_stubs/`. HAVE?: do we already have it.

| # | Lever | flux2 sha(s) / files | CLASS | HAVE? | Expected impact | Effort | Rebuild? | Prio |
|---|---|---|---|---|---|---|---|---|
| 1 | **FSDP on the 4-axis** — shard the 4x-replicated attention qkv/o + shared-MLP weights across the length-4 DP axis; all-gather weight at forward (cluster_axis=dp_axis). | `c394320254`, `0c131c83f3` (linear.py `mesh_axes=[fsdp,tp]`, `all_gather_persistent_buffer(weight,dim=2/3)`) | model | **NO** | Teja's top lever. Cuts per-chip weight DRAM footprint 4x on the memory-bound QKV/O + shared-MLP matmuls; the DP axis is currently dead replication. Directly shrinks weight-read/CCL cost. Faithful to flux2's own separate-weight-AG FSDP. | S (pure python) | **NO** | **P0** |
| 2 | **Update fused-op dirs to flux2 versions** — one rebuild unlocks: `fuse_swiglu`, skip-x1 addcmul, virtual-concat, `dit_minimal_matmul_addcmul_fused`, `minimal_matmul_split`, per-head rmsnorm, split-forward. | `cfb612f7ff`,`81bcb828c1`,`cc0a905aae`,`6140b0de10`,`359ef1d2d9`,`801e84f5dc` (op dirs listed above) | kernel | partial (older) | Enables levers 3-6. Attacks the #1 CCL + weight-read bottleneck. | M | **YES ~26min** | **P0** |
| 3 | **Fused SwiGLU** on MoE routed-expert + shared-MLP gate/up->silu->mul->down. Packed `[gate|up]` weight, `minimal_matmul(fuse_swiglu=True)`; kills chunk+silu+multiply chain + K-doubled read. | `cfb612f7ff`; `minimal_matmul` device op; `prepare_for_fused_swiglu` | kernel+model | **NO** | Direct hit on our memory-bound M=32 expert matmuls (biggest device-time bucket). N_block must be even (TT_FATAL). | S (after #2) | via #2 | **P1** |
| 4 | **Fused MM+ReduceScatter** for O-proj + expert down-proj — replace `all_reduce` (all_gather+sum) with `minimal_matmul_strided_reduce_scatter_async` (grid split into MM zone + RS zone; optional addcmul residual+gate at final write). | `e042e47258`,`080cb63a72`,`7bd7a349de`,`08d90ece49`; `RowParallelLinear.forward_fused_addcmul` | kernel+model | **NO** | Attacks the #1 bottleneck (RS+AG ~30%). Collapses matmul->RS->residual into one streaming kernel. Requires restructuring `_mesh_reduce` to reduce-scatter (output fractured). | M | via #2 | **P1** |
| 5 | **Even-ring split-forward** — diametric ring-AG/RS slice relays its two halves on BOTH fabric links so no link idles on the last hop. | `6140b0de10` (ring_attention_all_gather_async, sdpa/ring_fusion) | kernel | **NO** | Pure CCL bandwidth: "up to ~10% device-time on transport-exposed shapes", Galaxy ring-8. Applies to our exposed AG/RS. Note: our fabric is Linear-topology; verify ring applicability / gate on ring. | S (after #2) | via #2 | **P1** |
| 6 | **skip-x1 in fused addcmul** — when fused ternary scalar==1.0, host define `ADDCMUL_SCALAR_IS_ONE` `#ifndef`-guards out the per-tile `mul_unary`. | `81bcb828c1` (all_gather_minimal_matmul_async, strided_reduce_scatter_async, minimal_matmul) | kernel | **NO** | Free per-tile save on every fused residual-gate epilogue (our residual gate is scalar 1.0). | XS (in #2) | via #2 | **P2** |
| 7 | **Fused AG+MM to HIDE the FSDP weight all-gather** — use `all_gather_minimal_matmul_async` so the weight-shard gather streams into the matmul (avoids materializing full W + re-read). | `ab0ccc190b`,`c7c03d3963`,`c394320254` (ColParallelLinear fused path) | kernel+model | **NO** | Makes FSDP (#1) a stronger win by not paying a separate weight-AG. NOTE: flux2's fused-AGMM gathers the *activation* over the TP axis; verify whether it can gather the weight over dp_axis in our layout, else keep FSDP weight-AG separate (still measured net-positive in flux2). Ring-topology only. | M | via #2 | **P2** |
| 8 | **Matmul block-size heuristic** `get_matmul_config` — TP-K-shard-aware auto-blocking + log-ratio nearest-neighbor fallback, no hand-sweep needed. | `3e369fbc47`,`533eb7cce8`,`01823d065b`; `utils/matmul.py` | config | **NO** (we use `HUNYUAN_MM_FULLGRID` only) | Teja's block-size-sweep lever. Gives valid blockings for OUR QKV/O + gate/up/down shapes; `num_k_shards=tp` aware. Port the ALGORITHM, sweep OUR shapes overnight. | M | NO | **P2** |
| 9 | **num_workers_per_link supplied** + RS-zone/MM-zone grid split formula (`(grid.y-mm.y)*grid.x/(2*links)-1`). | `4ac91383e8`,`e042e47258` (FusedMMRSConfig) | config | partial (`HUNYUAN_CCL_LINKS`) | Extra CCL tuning knob on the fused-RS path once #4 lands. | S | NO | **P3** |
| 10 | **Per-head QK-norm** (no stats all-gather) — divisor=head_dim, stats stay local. | `3c8daa54a6`,`49f8531e8c`,`359ef1d2d9` (fused_distributed_rmsnorm per-head) | kernel+model | **NO** | Removes a cross-chip stats AG inside QK-norm. We already do qk-norm per-head-local via `ttnn.rms_norm` on split heads — verify we're not paying a stats AG; low incremental value. | S | maybe via #2 | **P3** |
| 11 | **Fused QKV as one matmul** (`ColParallelLinear(chunks=3)` / `minimal_matmul_split`) — one projection, per-device Q\|K\|V interleave. | `facf7938b2`,`adbd7267cf`,`9e68ef877c` | model | **YES** (we already fuse qkv into one `_shard_linear` + `nlp_create_qkv_heads`) | Already have the fused-QKV idea; `minimal_matmul_split` variant is marginal. | — | — | **DONE-ish** |
| 12 | **Skip unused last-block output projection** — don't project a stream whose output is discarded (keep it only as attention K/V context). | `f88677eb0d`,`7e64bcd638`,`de66987db1` | model | **NO** | Analogous to skipping output/LM-head projection for non-final positions. Small for image (all positions used); more relevant to decode. | S | NO | **P3** |
| 13 | **Virtual concat over K** — feed `[prefix,suffix]` into matmul with per-segment tile-padded weight; no real `concat` op. | `cc0a905aae` (minimal_matmul variant input) | kernel+model | **NO** | Drops real concat ops before shared linears (e.g. our `ttnn.repeat`/concat in the MoE batching, or spatial+prompt merges if added). Situational. | M | via #2 | **P3** |
| 14 | **Keep tensors sharded, let the fused CCL op gather internally** with a persistent scratch buffer (no explicit AG + re-partition). | `dac68d52b9`,`3bddf8e84e`,`25470e0aeb` (ring-joint sharded set) | kernel+model | **NO** | SDPA-path; relevant if we sequence-shard a second stream. Lower priority for our attention shape. | M | via #2 | **P3** |
| 15 | **In-place activations + hoist per-call allocs/reshapes out of the hot loop** — in-place silu (`output_tensor=`), pre-shaped rope (removes "2800 ops/inference"), precompute modulation params once & share across blocks, pre-alloc dummy tensors in `__init__`, no host round-trips between stages, cache prepared conv weights, arange+le causal mask (no `tril`) for traceable masks. | `f4fca3a600`,`c70ffe7d6d`,`1ff178f825`,`5879d4696a`,`9f0a84398d`,`0b09bb9971`,`ce48a41d57`,`68c51225e5`,`cfaa85bf00`,`28a6d0ad45`,`90320a2d1e` | model | partial | Op-count / host-dispatch reduction; aligns with our "layout-bound, cut op-COUNT" finding. Each is small; batch them. | S each | NO | **P3** |
| 16 | **Fused distributed GroupNorm on mesh** (PRE->fabric-AG-of-*stats*->POST). | `0d4dde7478`,`5ce1ec0c32` (NEW dir `dit_fused_distributed_groupnorm/`) | kernel (NEW) | **NO** | Unblocks the deferred on-device VAE (GroupNorm was the blocker). Clean self-contained cherry-pick (good "test the cherry-pick mechanism" candidate). Not transformer-perf. | M | **YES** | **P4 (VAE)** |
| 17 | **Ring-SDPA chunk sizes keyed by (is_bh, sp, tp, seq-len)** + CCL-worker-grid row-vs-column orientation. | `e8ab44c6a4`,`65bb74f4be`,`118f1f54e9`,`a48325396b`,`00519c9538`,`adbd7267cf`,`a5a6936034` | config | **NO** | SDPA-op tuning; memoized per-seq-len program config. Our SDPA is a smaller device-time bucket; low priority. | S | NO | **P4** |
| 18 | **Matmul config sweep tables** (per (M,K,N,grid) blockings + subblock). | `44cf8242fd`,`6a085f868c`,`a09713fef7`,`07deada8be`,`39b1834f55`,`4d1c62661c`,`5d05a16998`,`603fdd35ff`,`a375eb1989`,`655e3722a6`,`ce1a5b559a` | config | **NO** | flux2-shape-specific VALUES (do not copy). Use the METHODOLOGY (#8) + sweep OUR shapes. | overnight | NO | **P4** |

---

## COMPLETE 92-COMMIT COVERAGE (auditable)

Grouped; every sha appears. `class`: K=kernel, C=config, M=model, T=trivia.

**Kernel (portable C++) commits** — `3bddf8e84e`(K sdpa sharded-joint), `801e84f5dc`(K AGMM Ring guard + M linear gather), `359ef1d2d9`(K per-head rmsnorm + M attn), `25470e0aeb`(K ring-AG sharded joint), `81bcb828c1`(K skip-x1), `cc0a905aae`(K virtual-concat), `6140b0de10`(K split-forward), `0d4dde7478`(K NEW groupnorm).

**FSDP / parallelism** — `c394320254`(M fsdp wiring & dynamic loading), `0c131c83f3`(M hybrid TP+FSDP config), `7bd7a349de`(M colParallel fused method), `08d90ece49`(M col_parallel fused non-TP branch).

**Fused-op model wiring** — `9e68ef877c`(M new transformer AG+MM/MM+RS fusions), `ab0ccc190b`(M AGMM homing), `c7c03d3963`(M cleanup fusion), `e042e47258`(M+C fused MM/RS), `080cb63a72`(T fused-MMRS log), `c614a51e6f`(M isolate attention_opt_flux2), `cfb612f7ff`(M+K fused swiglu wiring), `c635aaea54`(M shared spatial+prompt attn fusion), `dac68d52b9`(M keep-sharded joint sdpa + single-block fusion), `e7e93c1544`(M attention seq1/seq2 refactor), `49f8531e8c`(M distributed rms norm), `3c8daa54a6`(M per-head norm flag).

**Block-size / conv3d heuristics** — `3e369fbc47`(C matmul heuristic), `533eb7cce8`(C conv3d nonsweep), `01823d065b`(C fix heuristics+sp1).

**Matmul/SDPA sweep VALUES (shape-specific)** — `44cf8242fd`,`6a085f868c`,`a09713fef7`,`07deada8be`,`39b1834f55`,`4d1c62661c`,`5d05a16998`,`603fdd35ff`,`a375eb1989`,`655e3722a6`,`ce1a5b559a`(all C, matmul tables); `e8ab44c6a4`,`118f1f54e9`,`a48325396b`,`00519c9538`,`adbd7267cf`(C SDPA chunk maps); `65bb74f4be`(T fix chunk keys); `a5a6936034`(C CCL grid row-vs-col).

**CCL tuning** — `4ac91383e8`(C num_workers_per_link).

**Sequence-parallel / VAE config** — `47d0228a2f`(M shard prompts: sequence-shard the prompt stream across the SP axis, all-gather q/k/v only for the joint SDPA, mesh_partition the output back — the SP-shard-a-second-stream pattern), `fe59336ffd`(C VAE TP off, 8.57s).

**Skip-work / compute-elision** — `f88677eb0d`(M skip last block output), `7e64bcd638`(M override to recompute), `de66987db1`(M breakout branches + merged-stream matmul), `1199bf6ca4`(T zero-length stream guard).

**1D-matmul fallback** — `ce596d6b4f`(C add control), `00c1225503`(T remove it — `minimal_matmul` wins even small-M).

**Op-count / host-dispatch / layout** — `1ff178f825`(M hoist rope), `c70ffe7d6d`(M pre-shaped rope, -2800 ops), `5879d4696a`(M pre-cast modulation bf16), `9f0a84398d`(M share (1+scale) across blocks), `0b09bb9971`(M+C addcmul fold + FFN cc), `ce48a41d57`(M pre-alloc dummy joint input), `68c51225e5`(M drop VAE host round-trip), `cfaa85bf00`(M cache prepared conv weights), `28a6d0ad45`(M traceable encoder mask), `90320a2d1e`(M defer AG past reductions + in-place silu, 4096 OOM fix), `facf7938b2`(M fused-QKV chunks=3 VAE).

**Model/base add + VAE hybrid parallel** — `782ab363aa`(M add Flux.2 base), `701d463520`(M vae_opt 9.3s), `5ce1ec0c32`(M groupnorm padding).

**Tracing / state infra** — `e12e441b27`,`748379ccdd`,`f19ff75bbf`,`54123cbfbc`,`97b8bb13ad`,`049d2d1a29`(T/M: Tracer, StateTensor, on-device sigma, traced step).

**Config placement / glx-bh** — `a0d72cf918`(C glx submesh), `40ea9279ad`(C bh 2-link), `50d6994ba8`(M/T trace+dynload plumbing + head-pad proj_out), `5bf74f1db0`(T trace perf test), `297b364f54`(T all-transformers test), `0033868995`(T transformer test + fsdp perf case).

**Pure trivia / cleanup / fixes** — `15bb740ffb`(rebase fix), `8466a06043`(checkpoint arg), `7dc6b3ad18`(merge conflict), `02193ad164`(comments), `672d565570`(remove files), `de82531995`(remove pce), `c9e57c4c22`(cleanup files), `068e2683cc`(spdx), `e92d0c2d8f`(prompt-encoder layout fix), `0891255a23`(profiling layer count), `91fca838f3`(mmrs buffer getter fix — real bug fix on Linear-topology RS intermediate shape).

> `f19ff75bbf`, `00c1225503`, `f4fca3a600` each bundle a real change + trivia and
> may be cited under two themes; all 92 shas are represented.

---

## TOP-3 TO LAND FIRST + EXECUTION PLAN

### #1 — FSDP on the 4-axis (NO REBUILD, do this first)

**Why first:** it is Teja's flagged top lever AND it needs zero C++ change — pure
python weight-sharding + our existing `ttnn.all_gather`. It de-risks the top
hypothesis in one build-free A/B before we spend a 26-min rebuild on anything.

**Where (box `hunyuan-image3-unified`, `_stubs/image3_decoder_layer.py` + `_stubs/mo_e.py`):**
- `__init__`: mesh is `(8,4)`; `self.tp_axis` = length-8 axis; **add**
  `self.dp_axis = 1 - self.tp_axis`, `self.dp = mesh_shape[self.dp_axis]`, and an
  env gate `self.is_fsdp = os.environ.get("HUNYUAN_FSDP","0")=="1" and self.dp>1`
  (follow the existing `HUNYUAN_EP_FULLMESH` pattern).
- **New helper** next to `_shard`:
  ```python
  def _shard_fsdp(self, t, tp_dim, fsdp_dim, *, dtype=ttnn.bfloat16):
      # 2D shard: tp_dim across tp_axis, fsdp_dim across dp_axis
      dims = [None, None]; dims[self.tp_axis] = tp_dim; dims[self.dp_axis] = fsdp_dim
      return ttnn.from_torch(t.to(_host_of(dtype)), dtype=dtype, layout=ttnn.TILE_LAYOUT,
          device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
          mesh_mapper=ttnn.ShardTensor2dMesh(self.device, dims=tuple(dims), mesh_shape=self.mesh_shape))
  ```
- `_build_sharded`: when `is_fsdp`, upload the two attention linears 2D-sharded
  (transposed weight is `[in, out]`):
  - `qkv_w` (column-parallel, TP shards out): FSDP shards **in** ->
    `self.qkv_w = self._shard_fsdp(qkv_w.t().contiguous(), tp_dim=-1, fsdp_dim=0)`
  - `o_w` (row-parallel, TP shards in): FSDP shards **out** ->
    `self.o_w = self._shard_fsdp(cfg.o_proj.weight.t().contiguous(), tp_dim=0, fsdp_dim=-1)`
  - Do the same for the shared-MLP `shared_gu` (fsdp_dim=-1, col-parallel) /
    `shared_down` (fsdp_dim=0, row-parallel) in `mo_e.py::_build_sharded` (they
    are fully replicated today). Router `wg` is tiny — leave replicated.
- **Forward** — gather the FSDP dim back just before each matmul (mirrors flux2
  `all_gather_persistent_buffer(weight, dim=2/3, mesh_axis=fsdp)`):
  ```python
  def _fsdp_gather(self, w, dim):
      if not self.is_fsdp: return w
      return ttnn.all_gather(w, dim=dim, cluster_axis=self.dp_axis,
                             num_links=_mo_e._ccl_links(), topology=ttnn.Topology.Linear)
  ```
  QKV: `qkv = ttnn.linear(h, self._fsdp_gather(self.qkv_w, dim=0))`
  O:   `o  = ttnn.linear(attn, self._fsdp_gather(self.o_w, dim=-1))`; then existing `_mesh_reduce`.
  (`dim` is the weight's FSDP-sharded axis: 0=in for qkv, -1=out for o.)

**PCC gate:** `tests/pcc/test_image3_decoder_layer_sharded.py` and
`test_mo_e_sharded.py` must match EP-off PCC. FSDP is a pure reshard — output
must be numerically equivalent to replicated.

**Perf gate:** `HUNYUAN_FSDP=1 HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2
./python_env/bin/python -m pytest -o timeout=0 .../tests/e2e/test_host_glue_stage3_perf.py -s`
-> compare ONDEVICE_E2E_TOTAL_LATENCY_S vs `HUNYUAN_FSDP=0`. One lever, one A/B.
Keep if it wins; revert if not.

**Risk / open question to verify on-device:** plain (non-fused) weight all-gather
materializes full W and the matmul re-reads it, so the DRAM-read win is not
guaranteed — the win is footprint + fabric-vs-DRAM offload + overlap. flux2 uses
a *separate* materialized weight-AG too and still measures net-positive, so this
is a faithful first cut. If neutral, Lever 7 (fused AG+MM) is the follow-up.

### #2 — Update the fused-op dirs to flux2 versions (the one rebuild)

**Cherry-pick plan (targeted, NOT full rebase):**
1. Per op dir we want, bring flux2's version:
   `git checkout refs/tmp/flux2 -- ttnn/cpp/ttnn/operations/experimental/minimal_matmul`
   (repeat for `ccl/all_gather_minimal_matmul_async`,
   `ccl/minimal_matmul_strided_reduce_scatter_async`,
   `ccl/strided_reduce_scatter_async`, `ccl/ring_attention_all_gather_async`,
   `transformer/sdpa`, `transformer/fused_distributed_rmsnorm`).
2. Reconcile shared registration/headers these dirs reference (CMake
   `sources.cmake`, `*_nanobind.cpp` aggregators, any shared `ccl_common`
   headers). Diff each dir's `#include`s against what our base provides —
   flux2's op versions may reference helpers added on flux2's base; if so, pull
   those helper files too. **Verify per dir before the rebuild.**
3. Rebuild (`build_Release`, ~26 min) on the box.
4. Smoke-test: import ttnn, run the component PCC tests
   (`tests/pcc/test_mo_e_sharded.py`, `test_image3_decoder_layer_sharded.py`)
   BEFORE wiring any new op call — confirms we didn't regress the existing ops.
5. **The isolated new op** `dit_fused_distributed_groupnorm` (`0d4dde7478`) is a
   clean `git cherry-pick` of a self-contained new dir + cmake + nanobind — good
   candidate to validate the cherry-pick mechanism first, but VAE-only (defer).

> **GATE:** state this plan in the status report and get a sanity-check BEFORE
> the first rebuild (device is shared; Teja runs overnight sweeps — keep the box
> free when asked, run the rebuild + heavy A/Bs off-peak).

### #3 — Fused SwiGLU on the MoE expert + shared-MLP matmuls (after #2)

Pack `[gate|up]` (already how our `gate_and_up_proj` is laid out), call
`minimal_matmul(..., fuse_swiglu=True)` (N_block even), drop the
`chunk`+`silu`+`multiply` chain. Highest-value use of the #2 rebuild — hits our
biggest device-time bucket (M=32 memory-bound expert matmuls). PCC >= 0.999.

Then #4 (MM+RS on O-proj/down-proj) and #5 (split-forward) reuse the same build.

---

## WHAT WE ALREADY HAVE vs TOP NEW OPPORTUNITIES

**Have (do not redo):** EP=32 full-mesh expert shard (`HUNYUAN_EP_FULLMESH`),
CCL_LINKS=2, MM_FULLGRID, fused attention all-reduce, bf8/bf4 experts, MoE
batched into 2 merged 2D matmuls, fused QKV projection, on-device host-glue
render. The core fused ttnn ops (`minimal_matmul`, `all_gather_minimal_matmul_async`,
`minimal_matmul_strided_reduce_scatter_async`, `fused_distributed_rmsnorm`)
already exist in-tree (older versions).

**Top NEW opportunities (ranked):** (1) **FSDP** on the dead 4-axis for
attention/shared-MLP — no rebuild; (2) **fused SwiGLU** on expert matmuls; (3)
**fused MM+ReduceScatter** on O-proj/down-proj to collapse the #1 CCL bucket; (4)
**even-ring split-forward** (~10% on transport-exposed CCL); (5) **matmul
block-size heuristic** + overnight sweep on OUR shapes; (6) **skip-x1** free
addcmul save. (VAE: fused distributed GroupNorm unblocks the deferred on-device
VAE.)
