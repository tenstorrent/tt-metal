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

Result: both failed. bf16 job 060 (169 s): `RuntimeError: parameter has no data` in the cold-cache save, after
`post_load_hook` -> `fold_gates_on_device` (t48-only, a613d669eef, on by default via LTX_FUSE_GATE_ON_DEVICE=1).
bf8 job 062 (128 s): `TT_FATAL concat ... All Tensors should have same dtypes` inside `fold_gate_on_device`
(attention_ltx.py:683; bf8 Q/QKV concat with bf16 gate). Both are t48 bugs in the default gate fold, not the test.
The DiT transformer cache was empty (t221 deleted the bf16 one), so both were cold loads.

## Attempt 3: pure ltx-rt HEAD b9f8587ce6c Python (git archive -> blx01 t283/ltxrt_tree), t48 build
blx01 job 066 (15:14 UTC), bf16: same `image_conditioning` TypeError in 15 s. So ltx-rt HEAD's own test is broken
unmodified: neither ltx-rt nor t48 LTXPipeline.__init__ takes `image_conditioning`; 4e7d0cefcdb (09-04) removed it
and the 09-29 consolidation merge 9ac584d0266 brought the stale test line back. t48's test no longer passes it.
Scripts run283c.sh / drv283c.sh / envrt.yaml (driver stopped by pgid before bf8 was submitted).
Broker event after 066 (not a drop of a running job): post-job health gate did a bridge reset of chip 27 ~15:15 UTC.

## Attempt 4: t48 Python (t208 tree), gate fold off by env only
run283d.sh / drv283d.sh (pgid 700512 on blx01, started ~15:22 UTC). Jobs in order:
bf16nf (LTX_FUSE_GATE_ON_DEVICE=0, fills the unfused bf16 cache), bf16 (t48 default, fold on, warm cache),
bf8nf (LTX_QUANT=all_bf8_lofi + LTX_FUSE_GATE_ON_DEVICE=0). One retry per config only on a pytest Timeout (cold load).
Marker /var/tmp/fasth3/t283/drv283d.done, log drv283d.done.log, run logs out48d_<P>/run.log.

Result (drv283d.done: `DONE bf16nf:072:completed bf16:077:completed bf8nf:078:failed`):
- bf16nf job 072 (15:27 UTC, gate fold off, cold bf16 cache fill): PASSED, gen #1 replay Total 4.78 s, gen #0 29.19 s.
- bf16 job 077 (15:33 UTC, t48 default fold on, warm cache): PASSED, gen #1 Total 4.67 s, gen #0 30.37 s.
- bf8nf job 078 (15:35 UTC): FAILED in warmup: the default bf8 activation cast (LTX_QUANT_ACTIVATIONS=1) hands a
  BFLOAT8_B tensor to dit_fused_distributed_rmsnorm (transformer_ltx.py:90 _norm_adaln), which takes bf16/fp32 only.
Logs: logs/drv283d_{bf16nf,bf16,bf8nf}.run.log, logs/drv283d.done.log.

## Attempt 5: 8-bit = bf8 weights, bf16 activations
run283e.sh (exports LTX_QUANT_ACTIVATIONS=0, then run283d.sh bf8wnf) / drv283e.sh (pgid 854093 on blx01, ~15:45 UTC).
Waits behind another project job on blx01 (080, t286). Marker /var/tmp/fasth3/t283/drv283e.done, run log
/var/tmp/fasth3/t283/out48d_bf8wnf/run.log. Reuses the existing q-all_bf8_lofi DiT cache (45 GB) if its key matches.

Old next step (attempt 4): when the marker exists, copy out48d_*/run.log into logs/, quote the gen #0/#1 tables, delete DiT cache
dirs created after 2026-10-08 15:00 UTC under /var/tmp/fasth3/t220/cache/dit-ltx23 (find -newermt), delete
t283/ltxrt_tree. Follow-up: t48 gate fold breaks cold-cache bf16 save and bf8.
