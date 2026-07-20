<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ACE-Step v1.5 — Tensor-Parallel (TP=4) implementation plan

Branch: `ign/ACE_tp4` (worktree at `../tt-metal-tp4`, forked from `ign/ACE_demo_modified…`).

## Goal

Run ACE-Step v1.5 on the **BH_QB 2×2 mesh (4 Blackhole chips)** with true **tensor
parallelism** instead of today's *replicate-everything* scheme, so each chip does ~1/N of the
DiT/LM/VAE work and holds ~1/N of the weights.

## Current state (why this is needed)

On BH_QB today every module **replicates** across all 4 chips and CFG/Euler run on the host:

- DiT weight upload mapper returns `None` → implicit replicate
  (`utils/tt_device.py: ace_step_dit_weight_mesh_mapper`).
- The "replicate" mapper is `ShardTensor2dMesh(dims=(None,None))`
  (`utils/tt_device.py: ace_step_replicate_mesh_mapper`).
- Host-forced multi-chip paths: `ace_step_mesh_use_host_temb_precompute`,
  `ace_step_mesh_use_host_cfg_euler`, `ace_step_mesh_use_sequential_cfg` (two full B=1 forwards).
- Readback (`ace_step_ttnn_to_torch`) uses `to_torch_auto_compose` — **replicate-only**; a
  sharded tensor read this way is silently corrupted.

So multi-chip currently buys memory headroom (via host offload), **not** compute speedup.

## Convention (mirror `tt_transformers`)

The 5 Hz LM already runs tensor-parallel because it is built from stock
`models/tt_transformers` via `create_tt_model(mesh_device=…)`. To make the DiT/VAE collectives
line up with that stack we adopt the **same 2-D pod convention** (`cluster_shape=(rows,cols)`,
see `models/tt_transformers/tt/model_config.py:688-694, 836-856`):

- Attention heads / `w1`,`w3` output → sharded across **cols** (`cluster_shape[1]`).
- `k`-dim of the row-parallel projection → sharded across **rows** (`cluster_shape[0]`), output all-reduced.

On a 2×2 that is **2-way per axis** → combined 4-chip utilisation. Column-parallel weights are
head/feature-local (no reduce); row-parallel weights need an **all-reduce**.

## Foundation (DONE — this commit)

`ttnn_impl/tp_config.py` — default **OFF**; enabled only via `ACE_STEP_TP=on|auto`:

- `resolve_tp_config(mesh_device) -> TPConfig{enabled, axis, rows, cols, degree}`
- `tp_weight_mesh_mapper(mesh_device, shard_dim=…)` — shard mapper, else replicate mapper.
- `tp_all_reduce(t, mesh_device)` / `tp_all_gather(t, mesh_device, dim=…)` — no-op when off.
- `tp_read_sharded_to_torch(t, mesh_device, shard_dim=…)` — gather-then-readback (avoids the
  silent-corruption trap).

Verified: OFF path is a strict pass-through even on a 2×2 mesh (no behaviour change until a
call site opts in AND `ACE_STEP_TP` is set), covered host-only by `tests/test_tp_config.py`.
Collective **keyword signatures** are written to the ttnn API present in this build but still
need on-device confirmation — that is gate **G0**.

## Phased plan (each phase PCC-gated against the replicate baseline)

- **G0 — collective bring-up. (DONE — validated on BH_QB 2×2.)** Confirmed signatures:
  `ttnn.all_gather(input_tensor, dim, *, cluster_axis=…)` and
  `ttnn.all_reduce(input_tensor, *, cluster_axis=…)` — **no `mesh_device=` kwarg** (mesh is
  implicit in the tensor); `tp_config.py` updated accordingly. Results: shard→all_gather→readback
  **PCC=1.00000**; `all_reduce(ones)` → `degree` (=2) along the cluster axis. **Critical finding:**
  CCL requires the fabric context up *before* the mesh opens — call
  `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` before `open_dit_device` and reset to
  `DISABLED` after close (TT_FATAL `fabric_context_ != nullptr` otherwise). The current replicate
  DiT path does **not** enable fabric, so Phase 2 must add this to the mesh lifecycle.
  Repro: `perf/tp_g0_collective_bringup.py`.
  NOTE: on a 2×2, single-axis TP `degree == 2`. True 4-chip TP needs 2-D sharding across BOTH
  axes (open question #1) — G0 proves the per-axis primitives; Phase 2 extends to 2-D.
- **Phase 1 — LM head narrow fix. (DONE — this commit.)** `ace_step_lm_head_narrow.py` selects
  vocab `split_sizes` (which tile a single device's `padded_vocab_size // num_devices` shard)
  against **global** band indices — correct only at `num_devices == 1`. Fix: **bypass narrowing
  when `num_devices > 1`** (LMHead vocab-sharded) and defer to the stock forward, which already
  does the correct sharded matmul + all-reduce. No-op today (LM runs on a 1×1 preprocess chip).
  Covered by `tests/test_lm_head_narrow_tp.py` (guard + helper unit tests, host-only).
  A future optimisation could make narrowing itself shard-aware (global→local column mapping).
- **Phase 2 — DiT MLP. (DONE — validated on BH_QB 2×2.)** `TtQwen3MLP` sharded: fused gate/up
  **column-parallel** (host chunks interleaved `[g0,u0,g1,u1,…]` so a contiguous dim-0 shard gives
  each chip its slice of *both*), `down` **row-parallel** + `tp_all_reduce`; program configs use the
  local intermediate; OFF path byte-identical (legacy `mapper=None`). Gate result: **TP-on vs
  replicate PCC=0.999988, TP-on vs torch=0.999592** (`perf/tp_phase2_mlp_pcc.py`).
  Readback finding: an all-reduced (replicated) output must be read via **device-0's shard**
  (`tp_read_replicated_to_torch`); `to_torch_auto_compose` mis-infers post-CCL topology
  (`dims must be unique`). Activations upload **replicated** (`ace_step_replicate_mesh_mapper`).
- **Phase 2b — DiT attention. (DONE — validated on BH_QB 2×2.)** `TtAceStepAttentionSDPA` sharded:
  Q/K/V **column-parallel by head** (fused `w_qwkv` interleaved `[q_d,k_d,v_d]` per chip; `wq`/`wo`
  heads already contiguous so plain dim-0/dim-1 shard), `o_proj` **row-parallel** + `tp_all_reduce`
  with the bias deferred past the reduce; head-split/GQA/RoPE/SDPA use **local** head counts.
  Gate (`perf/tp_phase2b_attn_pcc.py`): self on-vs-off **0.999992** / on-vs-torch 0.990591;
  cross on-vs-off **0.999992** / on-vs-torch 0.990019. Requires `n_heads`,`n_kv` divisible by degree.
- **Phase 2c — full DiT block. (DONE — validated on BH_QB 2×2.)** `TtAceStepDiTLayer` (self-attn +
  cross-attn + modulated MLP) under TP with **no new sharding code** — the TP-aware sub-blocks
  compose; modulation/norms/residuals run on the always-replicated hidden. Gate
  (`perf/tp_phase2c_layer_pcc.py`): **on-vs-off 0.999325**, on-vs-torch 0.989585 (== off-vs-torch,
  i.e. only bf16 drift). Confirms the replicated-hidden invariant between sharded sub-blocks holds.
- **Phase 3 — end-to-end integration.** **KEY SIMPLIFICATION (confirmed):** because each block
  all-reduces back to **replicated** hidden, inter-block activations, patch-embed, output-head, `xt`
  and velocity all stay replicated — so **no** patch-embed/output-head sharding and **no** per-step
  `xt` all-reduce are needed; TP is entirely inside the blocks. Done so far:
  - **Fabric in mesh open (DONE):** `open_dit_device` enables `FABRIC_1D` when `ACE_STEP_TP` is set
    (multi-chip only); `close_ace_step_device` resets. Legacy replicate path untouched.
  - **Readback (DONE):** `ace_step_ttnn_to_torch` reads device-0's shard under TP (auto_compose
    asserts on post-CCL tensors); replicate path unchanged.
  - **Eager e2e (DONE — runs clean):** `ACE_STEP_TP=on --no-use-trace` turbo 15s on BH_QB produced
    audio, no errors (Wall 101.2s, DiT 0.88s, RTF 0.15×). First full TP inference.
  - **TODO:** controlled TP-vs-replicate eager A/B (audio parity + speedup); base variant; then the
    **trace path** — CCL now lives inside the traced denoise step; validate `--use-trace` replay or
    keep collectives eager. Perf A/B via `perf/run_benchmarks.py`.
- **Phase 4 — trace vs collectives.** Collectives inside `begin_trace_capture` may not
  replay on a multi-device mesh. Start with collectives **eager** (outside the traced body);
  measure; then attempt in-trace CCL. Gate: trace-on RTF ≥ replicate baseline, audio parity.
- **Phase 5 — drop host offload.** Once DiT is genuinely sharded, retire
  host temb / host CFG-Euler / sequential-CFG for the mesh path.
- **Phase 6 — VAE (optional).** Channel-parallel: shard k=1 convs / detokenizer projections
  (all-reduce at output); keep k>1 conv + conv_transpose2d weights replicated, shard activations.
  Lower priority (VAE decode is ~14–24 s of wall).

## Open questions

1. **2×2 = 2-way-per-axis vs flat 4-way.** Foundation defaults to the tt_transformers per-axis
   convention (degree = size of one axis). Full 4-chip DiT sharding needs the 2-D scheme
   (heads across cols, k across rows) — reconcile precisely during Phase 2.
2. **Phase-A stays single-chip.** LM/Qwen/condition preprocess runs on 1×1 today; making the LM
   TP-parallel only helps if it also moves onto the mesh — separate decision, not required for
   DiT TP.
3. **Collectives in trace** — the main perf risk (Phase 4).

## Validation harness

Reuse `perf/run_benchmarks.py` for RTF A/B (replicate vs TP). Add per-module PCC tests under
`tests/` comparing TP output to the replicate baseline for the same seed/inputs. **Do not** land
any phase whose PCC gate is unmet.
