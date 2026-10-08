# t283: standard LTX e2e test, 16-bit vs 8-bit, BH 4x8 (blx01)

Test (unmodified, ltx-rt b9f8587ce6c, md5 bb2d68369bbbf2c7af8f15f22fd02ec8):
`models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k bh_4x8sp1tp0_ring`.
It prints `print_ltx_timing_table` (utils/ltx.py) after each gen: gen #0 captures traces, gen #1 is pure replay.

Tree: blx01 /var/tmp/fasth3/t220/src (ltx-rt b9f8587ce6c build; only diff was the t220 test patch, so the
pristine test file was copied over it for this task; original saved as /var/tmp/fasth3/t283/test_patched_t220.py,
restore after both jobs). Precision: 16-bit = test default (LTX_QUANT unset -> bf16/HiFi2); 8-bit =
LTX_QUANT=all_bf8_lofi. Both on the test's default 8+3 step schedule. RUN_VBENCH=0 RUN_CLIP=0 (perf-only).

Driver on blx01: /var/tmp/fasth3/t283/drv283.sh (bf8, then bf16; bf16 cold-fills its DiT cache ~37 GB).
Marker /var/tmp/fasth3/t283/drv283.done, log drv283.done.log, run logs out_{bf8,bf16}/run.log.
- 2026-10-08 14:53:22 UTC: bf8 broker job 053 submitted.

Next: when the marker exists, copy out_*/run.log here, quote the gen #0/#1 tables, restore the t220 test
file, delete the bf16 DiT cache (t220/cache/dit-ltx23/ltx-2.3-22b-distilled-1.1/transformer).
