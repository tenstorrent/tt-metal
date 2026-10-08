# t221: LTX-2.3 bf16 e2e, BH 4x8 (blx01), 1088x1920 145f, seeds 0-4 (user #156)

Counterpart of #220/#251 (8-bit, broker job 909, mean 5.756 s, steady 5.37 s). Same tree (blx01
/var/tmp/fasth3/t220/src, commit d791f4e9497), same caches/JIT dir, same env (LTX_QUALITY=medium schedule:
6 S1 + 1 S2 steps, LTX_TRACED=1, LTX_ATTN_FABRIC_AGMM=0, BWE/VOC trace off), same prompt and test. Only change:
`LTX_QUANT=` (empty), which apply_quality_env's setdefault keeps, so the DiT runs bf16/HiFi2 baseline and its cache
goes to cache/dit-ltx23/ltx-2.3-22b-distilled-1.1/transformer (~45 GB new, on blx01 root fs, 181 GB free at start).

Files on blx01 /var/tmp/fasth3/t221: run221.sh, env.yaml, drv221.sh, cmp221.py (copies here).
Driver (started 2026-10-08 ~03:07 UTC, setsid): fill1..3 (seed 0, until one completes) -> time (seeds 0-4,
out_time/) -> cmp221.py bf8 vs bf16 seed 0 (out_time/cmp.txt, still_bf16_seed0_f72.png, still_bf8_vs_bf16_seed0_f72.png).
Marker: /var/tmp/fasth3/t221/drv221.done; log drv221.done.log (job ids, statuses).

Next: when the marker exists, read it; if TIME_DONE completed, parse out_time/run.log (E2E_WALL_S lines + stage
tables) and cmp.txt and report the table vs #251. If FILL_NOT_OK, read out_fill*/run.log (OOM? timeout? drop?).
