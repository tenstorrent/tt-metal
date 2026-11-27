# WA.3: wo_a bank-local reads in the model, DS41_WOA_BANK off vs on (2026-11-27)

**off == on bit-identical on every case (every layer output and every served record), the on side's repeats
bit-identical, real-input 160-row calls bit-identical. wo_a a chunk 3.06 -> 2.65-2.74 ms (-0.33 to -0.41 ms), host a wo_a
call unchanged within noise. Load-time delta +1.39 s for layers 0-20 (+2.6 s for 40, estimate): no cache.**

Switch: `DS41_WOA_BANK` = off | on, **call-time** (bringup/hooks.py batch_call_settings, refreshed in batch_call_apply),
default **off** until the owner accepts. Copies (OV-77): `DS41_WOA_BANK_COPIES` = selected (default: a load uploads only
the copy of the layout WOA_BANK selects: one copy, as today) | both (the A/B and tests). The model never falls back: a
call to a layout its load did not upload raises; a held model switched between runs gets the missing copy then
(batch_call_apply -> tt/attention.woa_ensure, re-read from the checkpoint), never on the request path. test_woa_offon
adds the bank copy to the held model itself (one load) and drops it at the end.

## off vs on in the model (tests/woa/test_woa_offon.py, the gate unit; one load, layers 0-20, DECODER_SKIP=decode)

History H = 139264 (AgentX p50), agentic_cc text, 5 requests, 10 slots; off in slot set A, on in set B, set A
re-prefilled and on again (the repeat).

| case | off == on (outputs + served) | on repeat | device ms off / on | wo_a ms a chunk off / on (20 calls) | host us a wo_a call off / on |
|---|---|---|---|---|---|
| single 8192 [H, H + 8192) | 0.0 (20 layers, 0 served records differ) | 0.0 | 85.53 / 84.82 | 3.064 / 2.655 (-0.409) | 105.5 / 110.4 |
| packed 4 x 2k | 0.0 | 0.0 | 94.94 / 94.28 | 3.064 / 2.737 (-0.327) | 131.3 / 127.1 |
| single 5120 (last partial chunk) | 0.0 | 0.0 | 98.41 / 97.97 | 3.065 / 2.653 (-0.412) | 105.7 / 104.9 |
| real160 (op level: the 20 real wo_a inputs of the single 8192 off chunk, chip 0, first 160 rows, 160-row knobs) | 0.0 (20 calls, every chip) | | | | |

Device ms: real-time profiler, slowest chip, the median of 3 runs a side, alternating; wo_a ms = the programs inside a
"woa" signpost around each call (each call's slowest chip, summed). Host us: perf_counter around
ttnn.bringup.grouped_matmul, warm, no profiler, the better of 2 runs (20 calls a chunk). The single 5120 chunk pads to
256 rows a chip in the 8192 load (the 160-row shape of a 5120-chunk model is covered by the real160 row and by WA.4's
5120 ladder). Metrics: results/WA.3.json `woa_*` (suffix `_tagentic_cc`).

## OV-79: today's core set vs the band clusters (tests/woa/test_woa_layout_probe.py, op level, no model)

Device us a call (rt profiler, slowest chip, 20 calls, the min of 2 rounds), random fixed-seed tensors of wo_a's shapes,
every config bit-identical to today's call and repeatable. All bank configs on the model's matmul-capped 11 x 10 grid:

| layout | 256 rows | 160 rows |
|---|---|---|
| today (interleaved, 88 cores) | 151.1 | 131.7 |
| rect 11 (today's core set: 8 x 11 from (0, 0), row g = bank g; best knobs of 6) | 147.7 | 121.2 |
| band 10, one kernel group (wired) | **130.3** | **114.2** |
| band 10, two kernel groups (WA.2's program) | 130.0 | 113.6 |
| band 10 on the full 12-wide grid (WA.2's measurement; not legal in the model, below) | 124.9 | 104.1 |

Today's rectangle keeps only 3 / 10 us of the 21 / 17 us: the clusters next to their banks are what pays, so the band
layout is wired (WOA_BANK_SHAPE=band; rect / row stay selectable). The cores must stay in the 11 x 10 matmul grid: a
packed chunk's wo_a runs inside the gather-hide region (tt/gather_hide.Split.region, compute sub-device 11 x 10 next to
the gather column), where the full-grid layout's column-11 cores failed "Programs must be executed on a single
sub-device" (first gate run, DS41-382); 11 wide costs 5-10 us a call of WA.2's number. Knob sweep at 11 wide (35 configs
a row count, WOA_LP_SWEEP=1): (depth 4, w_inflight 1, x_ahead 1) at 256 rows and (4, 2, 1) at 160 stay within 0.7 us of
the best.

## OV-78: the host delta and its fix

Split (median of 5 alternating rounds of 50 warm calls, output buffer passed as the model does):

| | 256 rows: whole / Python wrapper / run_prepared | 160 rows |
|---|---|---|
| today | 80.2 us run_prepared, 18.3 Python | 51.2, 10.9 |
| bank, two kernel groups (24 core ranges) | 113.1, 18.8 | 73.7, 11.3 |
| bank, one kernel group (4 core ranges; wired) | 84.3, 19.0 | 53.9, 11.4 |

- Python (checks + prepared-program key): equal on both paths after WA.3's trims (the key no longer carries str() of
  three memory configs, ~1-2 us each; dtypes keyed as values; the ckc key cached a config object; the bank weight's
  shard check once a tensor object). It was never the cause: str() of the bank memory config costs 1.7 us.
- Enqueue (run_prepared): the cause. The bank program had one kernel group per core width (3- and 4-column cores) over
  8 bands: 24 core ranges vs today's 2, +23-45 us a launch on 32 chips. Fix: grouped_matmul `one_group=True` (the core's
  column tiles a runtime arg, define GMM_PN_RT): 4 core ranges, host a launch = today's within noise (+4 / +3 us,
  run-to-run spread ~+-8 us), device unchanged (+0.3 / +0.6 us). Same patch count rule (3 per kernel group; one group
  now), no per-call layout, CoreSlot or runtime-arg rebuild (the layout's key is a digest computed once).
- In the model (table above): host a wo_a call off / on 105.5 / 110.4, 131.3 / 127.1, 105.7 / 104.9 us: equal within
  the runs' spread (104.9-159.4 us).

## Load time and DRAM

- Load delta (constraint 4), layers 0-20, each copy built from the checkpoint read (host prep 1.29 s, common to both):
  interleaved conversion + upload 1.38 s, bank pack + conversion + upload 2.77 s: **+1.39 s** (+66 ms a layer);
  40 layers ~+2.6 s (estimate x 40 / 21). Not cached (WOA_BANK_DESIGN.md section 4's line: > ~10 s).
- Weight DRAM a chip for wo_a: shipped load (WOA_BANK_COPIES=selected) one copy, 35.65 MB a layer = 0.70 GiB for 21
  layers, 1.33 GiB for 40, as today; A/B / test load (both) twice that (+0.70 / +1.33 GiB; fits the 1M packed budget,
  WOA_BANK_DESIGN.md section 4). WA.4 states it for its loads.

## Rule 20 (serving reality)

No per-chunk value touched: the bank call's x / w / y addresses are prepared-program patches; the layout, knobs and core
set are compile-time / load-time; WOA_BANK is read per call from the settings (a host branch, no program key change
within a side); both sides' programs are built at warm-up (the cold no-rebuild unit runs with WOA_BANK=on). No extra
launches (one grouped_matmul program a wo_a call either way).

## Re-run

    PYTHONPATH=$PWD python -m models.demos.common.bringup gate WA.3 --spec $BRINGUP_SPEC --check
    # the op probe alone (resident runner, mesh only):
    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh \
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_layout_probe.py"
