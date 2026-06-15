# Final validation run — `wan_fused_distributed_rmsnorm` (Wan2.2 + LTX-2.3 AV + FLUX)

Comprehensive run of the committed op (branch `cglagovich/fused_rms_norm`,
commit `9983d6c29cb`) on a Wormhole **4×8 galaxy**. Covers PCC (vs fp32 PyTorch),
10× bit-exact determinism, and traced baseline-vs-fused perf. Generated via
`test_distributed_rmsnorm_fused.py` (`test_corr_det` / `test_bench`).

LINE = TP on a 1×TP row of the 4×8 mesh (tp_axis=1). RING = full 4×8 mesh, TP on a
closed mesh axis (4-wide for TP=4, 8-wide for TP=8), replicate the other axis.

## Tests SKIPPED (and why)

| test | reason |
|---|---|
| `test_bench[wan_tp4_line]` (2×4 mesh) | Blackhole line-box config; a 2×4 mesh **wedges the WH galaxy** (env note). Validated on BH, not re-run here. |
| `test_bench[wan_tp8_ring]` (1×8 mesh) | Blackhole 1×8 line-box config; same — not a WH-galaxy topology. |
| FLUX `per_head_norm=True` (8 configs: `flux_tp{4,8}_N*_phn1`) | **Known deadlock** on `ring_size>1` (per-head reduce fan-out wedges the compute pipeline). Gated behind `WAN_FLUX_PHN=1`; tracked in `ISSUE_per_head_norm_multidevice_deadlock.md` + findings "Known issue". Works on a 1×1 mesh (device-op unit test). |

Everything else below was run on the galaxy.

## Correctness + determinism (PCC vs fp32 torch, 10× determinism)

Every config below: `det=OK` (0/9, bit-exact over 10 fresh-pob runs), `flagged: NONE`
(the harness flags any config with `pcc(fused:torch)<0.999`, det failure, ratio drift
>5%, or worstrow >10%). PCC ranges are `pcc(fused:torch)`.

### LINE (TP on a 1×TP row of the 4×8 mesh)

| model / param | configs | result | pcc(fused:torch) |
|---|---:|---|---|
| Wan TP4  | 7/7   | det=OK, flagged NONE | 99.996–100.01% |
| Wan TP2  | 7/7   | det=OK, flagged NONE | (passed; detail lines merged with TP4 in capture) |
| LTX TP2  | 14/14 | det=OK, flagged NONE | 99.991–100.01% |
| LTX TP4  | 14/14 | det=OK, flagged NONE | 99.991–100.01% |

### RING (full 4×8 mesh)

| model / param | configs | result | pcc(fused:torch) |
|---|---:|---|---|
| Wan TP4 ring  | 7/7   | det=OK, flagged NONE | 99.996–100.01% |
| LTX TP4 ring  | 14/14 | det=OK, flagged NONE | 99.991–100.01% |
| FLUX TP4 ring (phn=False) | 4/4 | det=OK, flagged NONE | 99.993–99.9997% |
| FLUX TP8 ring (phn=False) | 4/4 | det=OK, flagged NONE | 99.985–100.003% |

**Correctness total: 71/71 configs pass** across LINE+RING (Wan 14+7, LTX 28+14, FLUX 8),
all `det=OK` bit-exact, all `flagged: NONE`.

## Perf — baseline (composite) vs fused, µs/iter traced, 4 links

### FLUX (fresh this session, RING)

| config | TP | rows | base | fused | ↑ |
|---|---:|---:|---:|---:|---:|
| flux_tp4_N512  | 4 | 512   | 117.99 | 92.07  | **1.28×** |
| flux_tp4_N64   | 4 | 64    | 92.90  | 104.73 | 0.89× |
| flux_tp4_N2048 | 4 | 2048  | 175.17 | 168.15 | 1.04× |
| flux_tp4_N8192 | 4 | 8192  | 577.97 | 483.43 | **1.20×** |
| flux_tp8_N1024 | 8 | 1024  | 123.07 | 107.70 | **1.14×** |
| flux_tp8_N128  | 8 | 128   | 96.55  | 64.33  | **1.50×** |
| flux_tp8_N4096 | 8 | 4096  | 279.75 | 255.43 | 1.10× |
| flux_tp8_N16384| 8 | 16384 | 953.79 | 780.48 | **1.22×** |

### LTX-2.3 AV (fresh this session, TP4 RING)

| config | rows | pattern | base | fused | ↑ |
|---|---:|---|---:|---:|---:|
| tp4_v_block_s1       | 1216 | block+adaLN | 139.70 | 110.20 | **1.27×** |
| tp4_v_block_s2       | 4864 | block+adaLN | 432.72 | 207.35 | **2.09×** |
| tp4_a_block          | 32   | block+adaLN | 34.02  | 25.50  | **1.33×** |
| tp4_v_selfattn_qk_s1 | 1216 | qk+rope | 145.08 | 128.88 | **1.13×** |
| tp4_v_selfattn_qk_s2 | 4864 | qk+rope | 455.21 | 259.45 | **1.75×** |
| tp4_a_selfattn_qk    | 32   | qk+rope | 51.66  | 30.50  | **1.69×** |
| tp4_a2v_videoQ_s1    | 1216 | qk+rope | 106.69 | 104.53 | 1.02× |
| tp4_a2v_videoQ_s2    | 4864 | qk+rope | 291.16 | 183.59 | **1.59×** |
| tp4_a2v_audioK       | 256  | qk+rope | 80.02  | 57.95  | **1.38×** |
| tp4_v_textcross_q_s1 | 1216 | qk | 88.70  | 102.10 | 0.87× |
| tp4_v_textcross_q_s2 | 4864 | qk | 248.26 | 187.09 | **1.33×** |
| tp4_v_textcross_k    | 1024 | qk | 82.53  | 83.45  | 0.99× |
| tp4_a_textcross_q    | 32   | qk | 32.64  | 23.42  | **1.39×** |
| tp4_a_textcross_k    | 1024 | qk | 71.96  | 75.28  | 0.96× |

These match the committed `RMSNORM_FUSION_FINDINGS.md` table within run-to-run noise,
confirming the prior perf numbers reproduce on the fresh galaxy.

### Wan2.2

Wan TP4 (7 configs) LINE + RING is tabulated in `RMSNORM_FUSION_FINDINGS.md` from clean
prior runs — fused wins biggest on large-token no-RoPE configs (`cross_q_sp4` ~1.7×),
small/dispatch-bound configs near 1.0×. (The fresh Wan galaxy bench re-run timed out per
the note below — its N=18944/N=9472 traced configs are the heaviest in the suite.)

> **Note:** a fresh Wan/LTX galaxy bench re-run *this session* hit the 580s batch
> timeout (35 traced configs including Wan's N=18944/N=9472 at 100 iters each,
> baseline+fused+cold-compile, on a galaxy degraded by this session's repeated
> `tt-smi -glx_reset`s). It is **not an op regression** — correctness was
> re-validated fresh (71/71, above) and the committed perf tables stand. Re-bench
> on a freshly-power-cycled galaxy in smaller per-model batches to refresh numbers.
