## Verdict data (2026-11-27, attempt 1)

- **Accuracy (switch on):** off == on bit-identical over **all 40 layers** (DECODER_SKIP=off, one load, both copies)
  on single 8192, packed 4x2k and single 5120 (every layer output and every served record: 0.0), on-repeats 0.0,
  real160 (40 real wo_a inputs at the 160-row knobs) 0.0 (test_woa_offon, WOA_OFFON_LAYERS=all). The fresh 40-layer
  reference (test_chunk8192_reference) and the fresh-process determinism run in the accept gate.
- **Device win, same load (test_woa_offon timing, layers 0-20, median of 3 profiled runs a side, alternating):** wo_a
  3.058 -> 2.656 ms a single-8192 chunk (-0.40), 3.067 -> 2.662 (4x2k, -0.41), 3.062 -> 2.651 (single 5120 chunk in the
  8192 load, -0.41); whole chunk 85.41 -> 84.82 (-0.60), 94.72 -> 94.20 (-0.52), 98.41 -> 97.73 ms (-0.68).
- **Host a wo_a call** (warm, best of 2): 97.4 / 97.8 us (single 8192), 109.0 / 116.2 (4x2k), 98.8 / 106.7 (5120);
  WA.3's runs spread 104.9-159.4 us, so +7-8 us is within the spread (at most +0.16 ms a chunk on 20 calls).
- **Ladders (on side, this session):** the tables below. The single-8192 attention section is 0.25-0.53 ms lower at
  every rung than the last accepted ladder (PB.MM256); packed device deltas sit within the sessions' drift.
- **Load time:** the bank copy costs +1.97 s over the interleaved copy for all 40 layers (bank 4.37 s, interleaved
  2.40 s, host prep 3.34 s common to both); WA.3's 21-layer number was +1.39 s. Not cached.
- **Weight DRAM a chip, wo_a:** 35.65 MB a layer. Shipped load (DS41_WOA_BANK_COPIES=selected, OV-77): one copy,
  1.33 GiB for 40 layers (0.70 GiB for the 21-layer perf load): the same as today, whichever layout is selected. A/B /
  test load (both): 2.66 GiB for 40 (+1.33 GiB), only while an off / on test holds both (test_woa_offon adds the copy and
  drops it after). The ladders' own `dram_load_gib_max` (perf load, held by the resident runner): 10.75 GiB (8192),
  9.16 GiB (5120).
- **Cold no-rebuild (strict, fresh process, switch on):** PASS on every path, 0 program builds after warm-up.

**The off side of the ladders is missing from this A/B:** `orchestrator measure --task WA.4` stops in
orchestrator.ab_batches -> metrics.reset with `OSError: [Errno 36] File name too long`: ab_test_names names the
single-8192 test_skip_positions unit after every variable it does not share with the 5120 unit (BRINGUP_CHUNK and 11
DS41_PACKED* / POS_DYNAMIC switches: 316 characters, a file name allows 255). This is framework code (outside this step's paths), and the
orchestrator's own A/B after this step goes through the same code. The on side was measured per part
(test_woa_offon, test_packed_walls, test_skip_positions__BRINGUP_CHUNK-5120, test_no_rebuild) and the single-8192
ladder through the gate unit WA.4#test_skip_positions. The off columns below are therefore a labelled PROXY (the last
accepted ladder, PB.MM256's on side: the off path's code before WA.3). Once the naming is fixed and the A/B has run
both sides, `python3 -m models.demos.deepseek_v41_flash_d_p.tests.woa.woa_ladder_report` renders the real off vs on
tables (this section: tests/woa/wa4_notes.md).
