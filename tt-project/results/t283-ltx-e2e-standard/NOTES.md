# t283: standard LTX e2e test, 16-bit vs 8-bit, BH 4x8 (blx01)

Test: `models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k bh_4x8sp1tp0_ring`,
unmodified. It prints `print_ltx_timing_table` after each gen: gen #0 captures traces, gen #1 is pure replay.

## Attempt 1: ltx-rt HEAD b9f8587ce6c (failed, test broken unmodified)
blx01 broker job 053 (bf8, 14:53 UTC): `TypeError: LTXPipeline.__init__() got an unexpected keyword argument
'image_conditioning'` (the ltx-rt test passes it, ltx-rt's pipeline lacks it). Log: logs/ltxrt_bf8_job053_FAILED.log.
The t220 test patch on blx01 was swapped back afterwards (md5 dbd2b250...).

## Attempt 2: t48 (ttp/t48-ltx25-integrated), same test, unmodified
Python: blx01 /var/tmp/fasth3/t208/tree (t48 5e4e0cd643a overlay; test md5 d9a26aaf... = t48 HEAD 9e20d905481,
later t48 commits touch only DiffVAE). Build/kernels/JIT cache: /var/tmp/fasth3/t48 (bf7db12a149).
Test defaults (LTX-2.3, 1088x1920, 145 f, seed 10, 8+3 steps, traced). 16-bit = LTX_QUANT unset (bf16);
8-bit = LTX_QUANT=all_bf8_lofi. RUN_VBENCH=0 RUN_CLIP=0. Scripts: run283b.sh, drv283b.sh, env48.yaml.
- jobs 056 (bf16) / 058 (bf8), 15:00 UTC: failed in 21 s, HF_HOME lacked the LTX-2.3 upscaler (log in logs/).
  Fixed: HF_HOME=/home/sulphur/hf (as t220/t221).
- 15:03:36 UTC: bf16 job 060 submitted by drv283b.sh; bf8 follows. Marker /var/tmp/fasth3/t283/drv283b.done,
  run logs /var/tmp/fasth3/t283/out48_{bf16,bf8}/run.log.

Next: when the marker exists, copy out48_*/run.log into logs/, quote the gen #0/#1 tables, delete DiT cache
dirs this task created under /var/tmp/fasth3/t220/cache/dit-ltx23 (find -newermt "2026-10-08 15:03").
No drops so far.
