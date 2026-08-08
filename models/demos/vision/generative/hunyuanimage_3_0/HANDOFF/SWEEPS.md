# Block-size sweeps — results (Teja's 4-family plan)

All on BH Galaxy `bh-glx-exp-b04u14`, HunyuanImage-3.0 image path.

| family | status | result |
|---|---|---|
| Other matmuls (gate/up, down, QKV, O-proj, shared-down; M≈4096) | DONE | winners `~/nkira/mm_sweep_winners.csv`; wired behind `HUNYUAN_MMCFG` → render A/B ~WASH (+2.4%). Within-op best-vs-worst spread 1.85–4.30× but tuned `minimal_matmul` ≈ default `ttnn.matmul`. Matmul ≈23% of device-kernel → small render ceiling. |
| AG+MM (`all_gather_minimal_matmul_async`) | DONE | winners collected; needed a 12×9 grid (12×10 hit an "Illegal NOC" worker-zone conflict); EP=32 shapes; spreads 1.5–3.8×. |
| RS+MM (fused Ring `minimal_matmul_strided_reduce_scatter_async`) | PARKED | full sweep IMPRACTICAL — the fused Ring op DEADLOCKS certain block configs, which WEDGES the fabric (`glx_reset` per hang → hours). Harness + chunked runner built (`sweep_mm_block_sizes_hunyuan.py` `bh_8x4_ring`/`is_mmrs`; `~/nkira/run_rsmm_chunked.sh`). 0 winners. RESUME via a COARSE safe-config sweep. Op proven to RUN + PCC-correct (0.99999). |
| 1D-decode matmuls (M=32, `MatmulMultiCoreReuseMultiCast1DProgramConfig`, `mcast_in0`) | DONE | winners `~/nkira/decode1d_winners.csv`; bf4 expert shapes 34–36× best-vs-worst spread; NO perf gain (decode path parked; spread is best-vs-worst, not best-vs-default). |
| ring-SDPA-chunk, conv3d | NOT STARTED | pending. |

**Infra lessons.** Hugepage leak across repeated per-shape mesh-opens (mitigated by a chunked one-process-per-shape runner). The box's multi-chip fabric is RESET-RECOVERABLE now (CPLD updated) — `tt-smi -glx_reset` clears a wedged (8,4) mesh; prefer `-glx_reset` over `-r`.
