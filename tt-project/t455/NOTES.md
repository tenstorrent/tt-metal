# t455 notes

## Step 1: SLA device isolation (blx01 broker job 027, submitted 2026-10-12 12:02 PDT)
Spec tt-project/t455/s-iso.txt -> /var/tmp/fasth3/t455/run455.sh, which runs
test_sla_device_minimax_h3.py (-k 4x8sp1tp0nl2_ring, seq 75449, SP=8) in overlay /var/tmp/fasth3/t455/w.
Logs: broker /var/log/tt-device-broker/2026-10-12_190219_027.log; pytest /var/tmp/fasth3/t455/run.log (T455_EXIT=<rc>).
Status: ssh g15blx01 tt-device-mcp status -j 027

Read the T455 log lines (grep T455 run.log):
- "topk_large_indices rows with the exact top-k set, by k": any value < 1.0 means topk_large_indices is the bug (check the k=176 = k_pad value).
- "selection: ids in range, duplicates, overlap mean/min, rows under 0.5": out-of-range ids, duplicates or low overlap with LightX2V
  means the stage is the bug (pooling, scores, cand_mask, global-id mapping b - (b>>shift)*(slots-tiles)).
- "output PCC, worst sampled q tile: vs device ids, vs LightX2V": vs device ids < 0.999 means vsa_sdpa (n_exempt=0, index rows) is the bug.
- All pass: the bug is in the pipeline (rung 75776 vs warmed 76800, rope/AdaLN tables, permute_metadata).

## Job 027 result (failed, exit 1, 49 s, no drops)
- topk "exact set" fractions < 1 at every k ({16: 0.871, 64: 0.75, 128: 0.639, ...}): bf16 ties at the k-th value,
  not a kernel bug (the device beats the CPU no-tie fraction at every k). Test now compares selected values.
- Stage selection vs LightX2V overlap 0.994; vsa_sdpa vs its own ids PCC 0.9996. With 2 heads/device the ops are fine.
- Still open: the real model runs 14 heads/device (56 heads, TP=4): pool_t folds heads into M (H*d=1792),
  topk/vsa_sdpa at H=14. Real Q/K/V statistics are the other suspect.

## Step 1b: same test at 14 heads/device (blx01 broker job 030, submitted 2026-10-12 12:10 PDT)
Cmd: T455_PY_TIMEOUT=400 bash /var/tmp/fasth3/t455/run455.sh (-k "4x8sp1tp0nl2_ring and h14"), -t 450.
Status: ssh g15blx01 tt-device-mcp status -j 030; log /var/tmp/fasth3/t455/run.log (grep T455).
Any failing T455 line at h14 (with h2 passing in 027) pins the bug on head count. All pass: dump real-layer
Q/K/V/ids (VSA_DUMP_INDICES) from one e2e step and replay them through the test.

## Next
1. Fix it, with a CPU test that fails without the fix
   (run: PYTHONPATH=tt-project/t454/stub:. python_env/bin/python tt-project/t454/runtests.py).
2. One blx01 A/B job: v1.0 dense vs SLA LoRA v0.1 + sla 0.85, seed 0 (templates t452/run452.sh, chain452.sh, psnr452.py;
   one arm is about 316 s warm, so one arm per job with -t <= 540).
3. ttp push --own --detach onto ttp/t454-fix-fasth3-turbo-sla-attention-mode-agai.
