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
