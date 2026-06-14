<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax-M3 TTNN Bring-up — Completion Status

**Model:** `MiniMaxAI/MiniMax-M3` — multimodal (vision-language) sparse-MoE, ~428B params, 60-layer text decoder + CLIP vision tower, 1M context.
**Target:** Blackhole Galaxy, 32 chips, tensor-parallel. **Scope (current cut):** text-first, **prefill**; vision best-effort.
**Branch:** `bringup/minimax-m3`. **State of truth:** `.bringup_state.json` (+ `BRINGUP_LOG.md` rendered from it).
**Driver:** agentic bring-up framework (`tt-bringup-framework/`, installed as `skills/`).

Last updated: 2026-06-14.

---

## Pipeline status

| Phase | Status | Notes |
|---|---|---|
| Architecture | ✅ done | 24 components (topo DAG) + 3 use cases; TP=32 placement plan computed |
| Reference | ✅ **23/24** | All PCC ≥ 0.9999 vs HF (transformers 5.12.0 oracle); `mtp_head` blocked (weights absent from release) |
| Missing-op discovery | ✅ done | 1 true ttnn gap (`sparse_lightning_attention`); `vision_rope_3d` composes; 21 components `ttnn-compose` |
| tt-lang authoring | ✅ done | `sparse_lightning_attention` indexer block-scoring sim-validated PCC 1.0 (honest sim gap: top-k/scatter host-side) |
| ttnn (text-first) | 🟡 partial | foundation + 5 leaf ops written & validated single-chip (PCC > 0.9999); mesh validation blocked on fabric |
| ttnn composites / decoder layers / lm_head | ⬜ pending | unblocked design-wise; awaits mesh |
| Full text-prefill integration | ⬜ pending | awaits decoder layers + mesh |
| Vision path | ⬜ deferred | all `ttnn-compose`, goldens ready; second pass |
| Generation / perf | ⬜ pending | |

---

## What's completed (artifacts, all committed)

### Reference (`reference/`)
- `functional.py` — 29 pure-PyTorch reference functions, HF-verified, for all 23 buildable components.
- `golden/*.pt` (gitignored, on disk) — input/weight/output goldens per block; `*.meta.json` tracked.
- Key model facts pinned: gemma `+1` RMSNorm; partial RoPE (rotary_dim 64, half-split); per-head QK-norm pre-rope; SwiGLU-OAI `down((up+1)·gate·σ(1.702·gate))`, `gate.clamp(max=7)`/`up.clamp(±7)`; MoE 128-expert top-4, sigmoid+bias-for-selection, normalize-to-1 then ×2.0, no grouped routing; vision Conv3d patch-embed (=linear), MHA no qk-norm, 3D rope (78 rot / 2 passthrough).

### Missing-op discovery
- `MISSING_OPS.md`, `missing_ops.json` — per-component ttnn-op needs + the single authoring target.

### tt-lang (`ttlang/`)
- `sparse_lightning_attention.py` — lightning-indexer block-scoring kernel; sim-validated bit-exact (block-index 0/2560 mismatch, e2e PCC 1.0).
- `test_sparse_lightning_attention_sim.py`, `NOTES.md` (sim/compiler gaps; ttnn drop-in = `block_scores → ttnn.topk → SDPA-with-mask`).

### ttnn (`tt/`)  — TP foundation + leaf ops
- `model_config.py` — TP config, dtype discipline, sharding-recipe mappers, text dims.
- `weight_loader.py` — lazy single-tensor load from the 59 safetensors shards via the index (never loads the full 869GB model).
- `rms_norm.py`, `final_norm.py`, `qk_norm.py`, `embedding.py`, `rope.py` — LightweightModules, lint-clean (no torch in forward), PCC > 0.9999 validated **single-chip** (TP-invariant ops).
- `test_leaf_ops.py` — mesh PCC harness.

---

## Active blocker: TP=32 fabric (hardware)

System-health (`test_system_health`) confirms: **32 ASICs alive, but no inter-tray ethernet links** (cross-tray channels `DOWN/unconnected`, retrain count 0; `Cluster does not support intermesh links`). Auto-discovery forms only a **4×4 = 16-chip** mesh. A full 8×4=32 single mesh needs the inter-tray links cabled/trained — **not fixable by `tt-smi` reset** (two `-glx_reset_auto` cycles did not restore them). The 16-chip mesh opens and runs cleanly.

Open decision (text-first first cut): **TP=16 now** (needs bf8_b weights, ~435GB ≤ 16×34GB) vs **fix inter-tray cabling for TP=32** (bf16) vs **2×16** (DP over two tray-meshes).

### Critical op note — mesh open/close lifecycle
Correct lifecycle (else fabric corrupts): `set_fabric_config(FABRIC_1D, STRICT_INIT, None, TensixConfig.DISABLED, UDMMode.DISABLED, ManagerMode.DEFAULT)` → `open_mesh_device(...)` → close submeshes → `close_mesh_device` → **`set_fabric_config(DISABLED)`** → `del`. Canonical source: repo-root `conftest.py` `mesh_device` fixture.

### Critical infra note — galaxy reset ↔ /data NVMe
`tt-smi -glx_reset_auto` (RESET_PCIE_LINK + IPMI) **re-enumerates the host NVMe** and can shut down the `/data` XFS (which holds the model weights + venvs). Protocol: `sync && umount /data` before reset; `vgchange -an/-ay data-vg && mount /data` after. Root `/` (boot NVMe) is unaffected — all committed work is safe there.

---

## Environment

- **Weights:** `/data/hf` (HF_HOME), 59 shards. **Reference oracle:** `/data/oracle-venv` (Python 3.12, transformers **5.12.0** — has M3; tt-metal's pinned 4.53.0 does not). **tt-lang:** `/data/ttlang-venv` (tt-lang-sim 1.1.3).
- **Device:** `bh_galaxy` added to the framework registry; `tt-smi` at `/usr/local/bin`.
- **Storage:** `/` = 877G (boot NVMe); `/data` = 28T striped XFS over 4×7T NVMe (no redundancy).

## How to resume
1. Decide TP topology (16 / fix-for-32 / 2×16) + dtype.
2. Set `TT_MESH_GRAPH_DESC_PATH` if forcing a non-default mesh; open via the lifecycle above.
3. Re-validate the 5 leaf ops on the mesh, then bring up composites bottom-up: `gqa_attention` → MLPs → MoE (via `models/demos/gpt_oss`) → `dense_decoder_layer` → `moe_decoder_layer` (sparse attn = `block_scores`+`ttnn.topk`+SDPA-mask; dense-GQA fallback exact for seq ≤ 2048) → `final_norm`/`lm_head` → full text-prefill → PCC vs chained reference.
