# Supervision: DeepSeek-V4.1-Flash, galaxy_8x4 full (seeded from the quietbox_2x2 run)

## 2026-10-26 10:41:59 UTC start (overseer: Claude session on the owner Mac, driving bh-glx-120-b06u08 over ssh)
- Host: bh-glx-120-b06u08 (Exabox 120-B67, Prefill slot, Slurm job 153825, 32 chips, TDP_LIMIT_MAX 0x82).
- Spec: seed spec unchanged except paths (owner-approved intake): art /scratch/$USER/bringup (node-local), hf hf_full
  (HF rev 5254448 text-only + the 4 vision/MTP shards, re-downloaded here), golden on /data (shared NFS).
- Goldens imported from bh-qb-12 via a private HF relay (minimal sets: s10240 52.9 GB, 32 non-seed chunk_00 layers left
  out; s56320 37 GB, 400 files left out); verify_copy intact on /data. Dry checks before intake: B.1 PASS, G.s10240 PASS,
  G.s56320 PASS, CK.1 all checkpoint/prompt metrics ok (intake_approved was 0 before approval).
- Note to W.1/M.1 (overseer note): chunk 5120 -> 160 query rows per chip, never gated (plan checked 8x4 at chunk 16384);
  first run of 4-column TP (loudbox_2x4 never ran). Plan deploy chunk 8192 vs seed spec deploy.chunk 4096: inconsistent,
  not blocking.
- Known infra: real-time profiler D2H socket pin failed on devices 7 and 10 during B.1 (F155 perf measurement may be
  affected at X.1); a 2x2 submesh of a galaxy fails its fabric handshake (no TT_VISIBLE_DEVICES 2x2 subset exists).
