# HunyuanImage-3.0 — Unified manual stack (branch `hunyuan-image3-unified`)
Assembled 2026-07-31 from the fragmented manual branches. Base = `hunyuan-image3-t2i-demo` (1c4585ccb6).

## Commit stack (on top of t2i-demo)
- `1bab756fcf` prune dead `_perf` duplicate tree (23 files)
- `ee61b0ce3c` EP=32 full-mesh expert-parallel (63cfd0eb26) re-expressed onto t2i merged-2D-matmul MoE — **opt-in `HUNYUAN_EP_FULLMESH`**
- `abe7a0ba5a` shard-shared expert + fold into 2-axis all_reduce (025dbff313) — gated with the same flag

## What the base (t2i-demo) already contains (verified in its own development)
Galaxy 6 MoE wins (fuse per-expert loop → 2 merged 2D matmuls, host pre-cast bf16, **bf4_b experts**, full-grid down matmul), Build A (router all_gather drop + skip l_aux + fuse SwiGLU silu), incremental-KV decode + trace, host-glue s1/2/3 (on-device head-glue), T2I diffusion demo, richest test suite (11 e2e files), **sparse-MoE opt-in `HUNYUAN_SPARSE_MOE`** (dense default; sparse is a measured 47x regression — never default).

## MoE decisions (already correct on the t2i base)
- **bf4_b experts = default** (4.88ms, PCC 0.99963). bf8_b was decode-line only; not reintroduced.
- **no-permute SwiGLU (391b3db31f) DROPPED as superseded** — the galaxy 2-merged-matmul fold already removed the permute it targeted.
- **sparse-MoE stays opt-in** (`HUNYUAN_SPARSE_MOE`), never default.
- superseded forms NOT ported: batched-grouped (730b22e66e), wide gate/up (dda1e8dbd0), bf4-savepoint gate/up-only (04bbcfbfab), bringup device_ms line (3bf0e74b59 — orphan, re-done natively elsewhere).

## GATED opt-ins (OFF by default — UNVERIFIED, mesh-only)
`HUNYUAN_EP_FULLMESH=1` turns on: (a) EP=32 (shard 64 experts across all 32 chips, 2/chip, + 2-axis all_reduce) and (b) shard-shared expert. Default OFF = behavior-identical to t2i (n_shard=tp, TP-axis shard, single-axis reduce). These re-expressions are **UNVERIFIED** — the re-expression onto the merged-matmul MoE has not been run on a mesh.

## DEFERRED follow-ups (documented, not applied — too risky/unverifiable to blind-merge)
1. **On-device lm_head + ROW_MAJOR argmax** (335e9da3aa + 8b65a1183f, decode/text path): a 78-line restructure of t2i's *deliberately host-based* decode head (`_decode_head_argmax`), with mesh-sharded lm_head (column-parallel + all_gather). Splice as a gated opt-in once the fabric is back; verify token-match vs the host head. Files: `tt/pipeline.py`.
2. **`test_image3_gen_perf.py` s/image harness** (90af667ea2): imports decode-line `tt/image_gen.py`, but t2i ships `tt/gen_image.py`. Adapt the harness import/API to `gen_image`, then bring it.

## VERIFICATION STATUS — BLOCKED by the wedged Galaxy fabric
The manual model calls `ttnn.all_reduce`/`all_gather` even at TP=1 (no `tp==1` short-circuit), so it needs the inter-chip fabric *even single-chip*. With `bh-glx-exp-b04u14` fabric wedged (eth e0-4/e0-5 stuck at STARTED), **NO runtime test passes** (single-chip test_mo_e fails at all_reduce: "un-initialized fabric context"; multi-chip blocked). What IS confirmed: file parses (ast), module imports, and the default (EP-off) path is byte-behavior-preserved (only additive gated code).

### When the fabric is fixed — verify checklist
1. `pytest tests/pcc/test_mo_e_sharded.py test_image3_decoder_layer_sharded.py test_top_k_gate_sharded.py` on the 4x8 mesh (default EP-off) — confirm the merge didn't regress.
2. `HUNYUAN_EP_FULLMESH=1 pytest tests/pcc/test_mo_e_sharded.py` — verify EP=32 + shard-shared PCC (expect ~0.999) + `tests/e2e/test_image3_prefill_perf.py` for the +70% t/s/u. If good, flip the default ON.
3. Apply + verify deferred follow-up #1 (on-device head), then #2 (gen-perf harness).

---

## 2026-07-31 — VERIFICATION UNBLOCKED (fabric recovered)

The inter-chip fabric on `bh-glx-exp-b04u14` RECOVERED (was wedged: Device 0/1 eth
ch4/5). Confirmed by probe: full `MeshShape(8,4)` opens under FABRIC_1D/2D/2D_TORUS_XY
and `all_reduce` moves data correctly on BOTH axes, including cluster_axis=1 which
crosses the historically-dead Device 0<->1 link (got 4.0/expect 4). Sub-meshes still
never bring fabric up here — only the full 32-chip mesh does.

**Tier-1 single-chip fix landed (commit `09f00b6f00`)** — 3 guards make the mesh model
run fabric-free at 1 device (`_is_mesh_device` requires >1 dev; `_mesh_reduce` no-ops
off-mesh; `HY3_SINGLE_CHIP=1` opens a plain device). Behavior-identical for real
multi-chip.

**All PCC gates GREEN (TT_HY3_PCC=0.95, one 6U Blackhole Galaxy):**

| path | test | PCC |
|---|---|---|
| single-chip (fabric-free) | test_mo_e | 0.9996 |
| single-chip | test_image3_decoder_layer | 0.99999 |
| single-chip | test_top_k_gate | 1.0 |
| multi-chip TP=8 (EP off, default) | test_mo_e_sharded | 0.9940 |
| multi-chip TP=8 | test_image3_decoder_layer_sharded | 0.99999 |
| multi-chip TP=8 | test_top_k_gate_sharded | 1.0 |
| multi-chip EP=32+shard-shared (HUNYUAN_EP_FULLMESH=1) | test_mo_e_sharded | 0.9940 |
| multi-chip EP=32+shard-shared | test_image3_decoder_layer_sharded | 0.99999 |

The EP=32/shard-shared opt-ins are now PCC-verified (were UNVERIFIED). Still gated OFF
by default — flipping the default should be gated on a perf (device_ms) comparison, not
just PCC. Remaining deferred: the pipeline lm_head/argmax splice + gen-perf harness.

---

## 2026-08-01 — Lever 1 LANDED: EP=32 + shard-shared DEFAULT ON

Measured on the full (8,4) mesh via `tests/e2e/test_image3_t2i_perf.py` (32 layers, 12 steps,
1024², CCL_LINKS=1, trace on); metric = trimmed steady-state ms/step (drop step-1 compile).

| config | steady ms/step | loop_s | E2E s/image | vae_s |
|---|---|---|---|---|
| EP off (`HUNYUAN_EP_FULLMESH=0`) | 7770 | 98.1 | 157.8 | 59.2 |
| **EP on (new default)** | **6368 (-18.0%)** | 85.2 | **143.1 (-9.3%)** | 57.4 |

EP-on steps 2-12 rock-steady 6282-6522ms (zero overlap with EP-off 7658-8078); both rendered
correctly. Flipped `HUNYUAN_EP_FULLMESH` default ON at `_stubs/mo_e.py:158` (`!= "0"` + a
num_experts-%-num_devices divisibility guard; `=0` forces OFF, non-divisible meshes fall back
to the TP-axis shard). PCC re-verified: sharded default 0.9940/0.99999/1.0, escape-hatch EP-off
0.9940, single-chip 0.9996/0.99999.

Profile insight: per-step device compute (attn+moe) ~3s but traced-replay wall-clock ~6.4s
=> ~55% CCL/sync-bound. The ~58s host VAE tail is now the single largest chunk of the 143s E2E
=> Lever 2 (on-device VAE) next, then CCL/CFG levers.

## 2026-08-01 — Lever 1b: CCL_LINKS default 1 -> 2

EP=32 on. `test_image3_t2i_perf` (32L/12step/1024², trace) steady ms/step:

| CCL_LINKS | steady ms/step | E2E s/image |
|---|---|---|
| 1 | 6368 | 143.1 |
| **2 (new default)** | **6093 (-4.3%)** | **135.8** |

`_ccl_links()` default 1->2 (`_stubs/mo_e.py:122`); `HUNYUAN_CCL_LINKS=1` restores single-link.
PCC unchanged (num_links = transport, not math): sharded 0.9940 / 0.99999.
Cumulative vs original EP-off/links=1 baseline: 7770 -> 6093 ms/step (-21.6%), E2E 157.8 -> 135.8s.

## 2026-08-01 — Lever 1c: MM_FULLGRID default OFF -> ON

EP=32 + CCL2 on. `test_image3_t2i_perf` (32L/12step/1024², trace) steady ms/step:

| MM_FULLGRID | steady ms/step | E2E s/image |
|---|---|---|
| off | 6093 | 135.8 |
| **on (new default)** | **5736 (-5.9%)** | **131.6** |

`_mm_grid()` default OFF->ON (`_stubs/mo_e.py:113`); `HUNYUAN_MM_FULLGRID=0` restores op-default
grid. Grid-only (no math change): PCC 0.9936 / 0.99999. Cumulative vs original EP-off baseline:
7770 -> 5736 ms/step (-26.2%), E2E 157.8 -> 131.6s (-16.6%).

## 2026-08-02 — Lever 1d: MM_FIDELITY=lofi = NO-OP (not flipped)

PCC-safe (lofi 0.9936 == default), but perf 5759.6 vs 5736.0 ms/step = +0.4% (jitter). The ttnn
default is already LoFi-equivalent for the bf4_b MoE matmuls, so explicit lofi buys nothing. Left
at default "". Cheap-gated-knob vein exhausted (EP/CCL2/FULLGRID landed, -26.2% cumulative).
