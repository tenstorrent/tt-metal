# t375 — conv VAE fused RMSNorm+SiLU speedup (PLAN t372 item 3)

Base: origin/ttp/t48-ltx25-integrated @ 90ed8257bac. Scripts: notes/t375 (copies of tt-project/t375, live on blx01 /var/tmp/fasth3/t375).

## Diagnosis (profile job 452, t334)
Norm ops already use the full 120-core grid at ~150 GB/s (DRAM peak 512) with the same ~3.3 us/tile at every
channel count, so they are compute-bound in the kernel (tilize -> square -> reduce -> rsqrt -> mul+SiLU -> untilize,
HiFi4 + fp32 dest acc, block 4). A better grid/shard config is unlikely to help; compute-config levers are the candidates.
SiLU SFPU always uses the exact sigmoid (math_approx_mode does not change it).

## Job b result: blx01 broker job 527 (completed 2026-10-10 21:53Z, exit 0), log notes/t375/results/run_b_job527.log
Weighted decoder total (res128 x9, res256 x12, res512b x8; min of 5 reps, 900 MHz clamp, relative only):

| variant | weighted ms | saved | quality vs V0 (current) |
|---|---|---|---|
| V0 HiFi4 fp32acc SiLU RM (current) | 41.66 | - | - |
| V2 HiFi2 fp32acc | 40.18 | 1.5 | PCC 0.99983..1.0 |
| V3 LoFi fp32acc | 39.88 | 1.8 | PCC 0.9985 (res512b), max abs 0.25: degrading |
| V4 HiFi4 bf16acc | 35.12 | 6.5 | PCC 0.9995..1.0, max abs 0.078 |
| V5 LoFi bf16acc | 33.00 | 8.7 | PCC 0.9981 (res512b): degrading |
| V6 TILE input (else V0) | 36.10 | 5.6 | bit-identical, but the decoder's conv3d gives RM: a tilize costs more |
| V1/V7 no SiLU (reference only) | 38.78/32.71 | - | - |

Verdict for (a): no config lever reaches the 10 ms keep line. The only one near it without quality loss
(V4, 6.5 ms) is not bit-identical and below the bar; V6's gain needs a TILE conv3d output that does not exist.
The op is the generic interleaved layer_norm (RMSNORM) program with a default program config; a sharded config
would need TILE + sharded input, i.e. extra reshard/tilize ops around every norm. No code change made.

(b) fold into conv3d: not started (high-effort C++ in conv3d; needs a stats pass plus scale+SiLU on the vol2col
patches, which repeat each input stick 27x). Estimated net 10-20 ms. Recommended only as part of PLAN item 5
(vol2col reuse), where the window is read once and norm+SiLU could be applied once per stick.

Cleanup: removed /var/tmp/fasth3/t375/jit (333 MB) and out_b/generated on blx01; scripts + log kept (small).

## Drops seen (not ours)
2026-10-10 21:13Z blx01: chips 24/25 left PCIe during ltx-host job 512 / fabric-check 513; recovered by 21:17Z (hold 520 ended).
