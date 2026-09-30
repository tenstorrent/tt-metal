# t36 notes (blx03 setup)

Disk before: /home 229 GB free (97%), ~/fasth3 did not exist.
Clone: blob-less clone of origin, branch ttp/t36-blx03-ltx25 = 9287c953eb8 (t10/t26 line, NA fix d079ee7cd11) + merge of t22 (cc51d4a21bc, S2 prompt reuse). Pushed.
Setup: ~/fasth3/setup36.sh (log setup36.log; build.log ends BUILD_EXIT=, venv.log ends VENV_EXIT=, then IMPORT_OK, SETUP_DONE).
Driver: ~/fasth3/drive36.sh (log drive36.log) waits for setup, then queues ONE cold-cache smoke job smoke_dv145
(1080p/145f, seed 0, broker cap 1800s, pytest 1740s) via tmp/blx03/submit.sh, waits, prints disk use, ends DRIVE36_DONE.
Smoke output: ~/fasth3/out/ltx25_1080p_6s/smoke_dv145/run.log; broker log in ~/fasth3/smoke_job.log.
Next: when DRIVE36_DONE, read LTX_TIME / stage lines from run.log, fill the timing into READY_blx03.md,
remove ~/fasth3/uvboot and .uv-cache, report disk after.
If blx03 reboots: check `tt-device-mcp status` for our job; rerun setup36.sh only if build.log lacks BUILD_EXIT=0.

Attempt 2 (18:39): the first smoke never ran. drive36 was detached with no venv, so the broker defaulted to
<ws>/tt-metal/python_env and refused it; drive36 then parsed "3" from "fasth3" as the job id. Fixed: submit.sh
passes -e tmp/blx03/env.yaml (c11bf19dc4); drive36 id parse fixed. Smoke resubmitted by hand as broker job 867
(queued 18:39, cold cache: "Cache does not exist. Loading PyTorch state dict").
Waiter: ~/fasth3/wait36.sh 867 (log ~/fasth3/wait36.log, ends SMOKE36_DONE RUN_EXIT[...]).
Next: read timings from ~/fasth3/out/ltx25_1080p_6s/smoke_dv145/run.log, record DiT cache size, fill READY_blx03.md,
remove ~/fasth3/uvboot and any .uv-cache, report disk after.

Attempt 3 (18:57): job 867 ran 945s and failed in the warmup decode (not the model): DIFFVAE_MEM_LOG
ping_pong_buffer_report crashed on reduce-scatter cache keys (no kind label). Fixed in 49ccc0ebb4 (CPU-tested).
867 did write all 10 DiT caches (72G, ~/fasth3/cache/dit-ltx25; cold convert: transformer 143s, text encoder 72s,
vocoder 39s, others <25s) and the kernel cache (5.6G). Warmup gen got through S1/S2 and into decode (warmup 159s).
Disk after 867: /home 141 GB free (99%), ~/fasth3 90G (cache 79G, tree 11G). Removed ~/fasth3/uvboot (venv uses ~/fasth3/.uv-python).
Rerun: ~/fasth3/drive36b.sh (log ~/fasth3/drive36b.log, ends DRIVE36B_DONE) waits until no other project job
holds blx03 (869 from another task was running), then queues smoke2_dv145 (cap 1200s, pytest 1140s), waits, prints disk.
Next: when DRIVE36B_DONE, read LTX_TIME / stage lines from ~/fasth3/out/ltx25_1080p_6s/smoke2_dv145/run.log
(expect all load-cache hits), grab a still, fill timing into READY_blx03.md.
If blx03 reboots and drive36b is gone: check `tt-device-mcp status` for a smarton smoke2 job; resubmit via the driver if none.
