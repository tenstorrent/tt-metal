# t225: DiffVAE DIFFVAE_S5_2D=1 quality check, production path (device stage-5 noise), 5 seeds

Code under test: t48 036247a8eb6 python (4 files changed since a40d78b8bae) as an overlay on blx01
/var/tmp/fasth3/t225/ov (hardlinked t212/b/models + those files written via .new + mv), on the t212 build
/var/tmp/fasth3/t212/b (a40d78b8bae C++). Overlay md5s match `git show 036247a8eb6:<file>`.

blx01 (/var/tmp/fasth3/t225): driver.sh started 2026-10-07 22:01 UTC (pid 150626, setsid). It waits for broker
health and for #227's job (863) to clear, then ONE broker job (run.sh, -t 300; job 855 measured 144 s for a similar
load): arm 1d then arm 2d, each its own process, warm-up + seeds 0-4, writes out/{1d,2d}_seed{N}.yuv.
Then post.py (cmp.json: 2d vs 1d, each vs #214 host-noise ref; still_seed*_f72.jpg top 1d / bottom 2d;
crop_seed*_f*.png 1d|2d|8x diff) and mp4s (crf 12). Marker drv/driver.marker "T225_DRIVER_DONE stage=done rc=0 ...".

g15blx02: local.sh (ttp detach t225local, run 917) waits for that marker, fetches mp4s + post outputs into
tt-project/data/g15/t225/{1d,2d,post}, then ltx_eval batch (cand 2d, ref 1d, --vbench-ref, 5 dims) ->
data/g15/t225/eval, eval.log. Marker data/g15/t225/LOCAL.done "<rc> <stage>".

Next on wake: read LOCAL.done, post/driver.log (drops), post/run.log (DECODE_MEAN, brick), post/post.log (T225_CMP),
eval.log (BATCH line, per-dim vbench vs vbench_ref); view stills/crops; verdict; copy small results to t225/results;
then clean up: blx01 out/*.yuv (4.5 GB), g15 data/g15/t225 eval PNGs + mp4s once results are recorded.
