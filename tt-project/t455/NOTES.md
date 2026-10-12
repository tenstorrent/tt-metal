# t455 notes

## Step 1: SLA device isolation (blx01 broker job 027, submitted 2026-10-12 19:02 PDT)
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

## Next
1. Fix it, with a CPU test that fails without the fix
   (run: PYTHONPATH=tt-project/t454/stub:. python_env/bin/python tt-project/t454/runtests.py).
2. One blx01 A/B job: v1.0 dense vs SLA LoRA v0.1 + sla 0.85, seed 0 (templates t452/run452.sh, chain452.sh, psnr452.py;
   one arm is about 316 s warm, so one arm per job with -t <= 540).
3. ttp push --own --detach onto ttp/t454-fix-fasth3-turbo-sla-attention-mode-agai.
