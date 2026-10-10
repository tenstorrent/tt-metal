# t375 — conv VAE fused RMSNorm+SiLU speedup (PLAN t372 item 3)

Base: origin/ttp/t48-ltx25-integrated @ 90ed8257bac. Scripts: notes/t375 (copies of tt-project/t375, live on blx01 /var/tmp/fasth3/t375).

## Diagnosis (profile job 452, t334)
Norm ops already use the full 120-core grid at ~150 GB/s (DRAM peak 512) with the same ~3.3 us/tile at every
channel count, so they are compute-bound in the kernel (tilize -> square -> reduce -> rsqrt -> mul+SiLU -> untilize,
HiFi4 + fp32 dest acc, block 4). A better grid/shard config is unlikely to help; compute-config levers are the candidates.
SiLU SFPU always uses the exact sigmoid (math_approx_mode does not change it).

## Job b (running): micro-benchmark bench_norm.py on blx01, t48 build bf7db12a149 (same norm kernels), full 4x8
Variants V0 (current HiFi4/fp32acc/SiLU/RM) .. V7; see bench_norm.py header. Output: BENCH lines + BENCH_TOTAL weighted_ms.
Driver: drv375b (ttp detach --remote g15blx01), marker /var/tmp/fasth3/t375/drv375b.done, log run_b_job<J>.log.
Check: ttp detach --check --host g15blx01 /home/smarton/.ttp-detach/1439/drv375b

## Next
If a variant saves >=10 ms weighted with PCC vs fp32 ref ~unchanged: add opt-in LTX_VAE_NORM_FAST in vae_ltx.py for
norm1/norm2/norm_out, A/B test (real conv VAE weights, 5 seeds, PCC>=0.9999, PSNR>=45 dB incl. luma, 3 timed replays),
build t48 head + change on blx01 (/var/tmp/fasth3/t375/b), A/B job, then standard e2e. Else consider conv3d fold or fail.

## Drops seen (not ours)
2026-10-10 21:13Z blx01: chips 24/25 left PCIe during ltx-host job 512 / fabric-check 513; recovered by 21:17Z (hold 520 ended).
