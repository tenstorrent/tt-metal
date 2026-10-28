# Engram host path: hand-off from the 2x2 session to the Galaxy session (2026-10-28)

Owner decisions of 2026-10-28, done on the 2x2 QuietBox, rebased on 5b275cb7622 (FIX.7 + the Galaxy's MADV_RANDOM).
Details and every number: `ENGRAM_HOST_PATH.md` section 3e. Coordination rules unchanged (section 4a): the 2x2 session
owns `tt/engram.py`, `tt/engram_kv.py`, `tt/engram_bfp8.py`, `reference/weights.py` and the engram block of
`tt/settings.py`.

## What changes for you

1. **The server builds both engram tables in RAM at start, already as bfp8.** `ENGRAM_TABLE=resident` (default) with
   `ENGRAM_ROWS=bfp8` (default) reads each table once from the checkpoint and keeps it in anonymous memory as
   bfloat8_b rows (272 B a row: 16 shared exponents next to 256 magnitudes; 104.4 GB a layer, ~209 GB for layers 1
   and 14). Nothing is written to disk. Build time on the QB: 84 s a layer cold (1.24 GB/s), ~45 s with the files in
   the page cache. Then a 5120-token chunk takes **7.6 ms a layer** from token ids to TILE bytes (8 threads; 5.0 ms on
   16), vs ~0.6 s cold from disk with read-ahead off and ~4.5-7 s with read-ahead.
2. **Tests do not build it (F222).** Under pytest `ENGRAM_TABLE` is `random` (the checkpoint mapping, read-ahead off):
   the same rows bit for bit (`tests/test_engram_table_modes.py`), no build at each test's start. So gate, ladder and
   contract test times do not change. A process outside pytest that builds the model (a profiling or serving script)
   pays the build once. To force a mode anywhere: `DS41_ENGRAM_TABLE=resident|random|mmap`.
3. **The rows go to the device as bfp8 (`ENGRAM_ROWS=bfp8`), packed on the host with no float.** Integer operations
   on the table's e4m3 + e8m0 bytes; byte-equal to ttnn's own packer of the exact rows (`tests/test_engram_bfp8.py`,
   device, bit-exact on every chip). Since FIX.7 (`GEMM_INPUTS=bfp8`) the wkv matmul reads bfloat8_b anyway:
   `engram_kv` now takes bfp8 rows as they arrive (`gemm_input` is a no-op on them), so the rounding just moves to the
   host. `ENGRAM_ROWS=bf16` remains as the option.
4. **engram_rows carries an owner waiver** (`spec.yaml tests.waivers.engram_rows`): its exact-match check (limit 0)
   fails by the bfp8 rounding (rel 3.5e-3, 10 % of values rounded, 1 % to 0). The engram_rows gates in `tasks.yaml`
   also run `tests/test_engram_bfp8.py`, so a lookup bug still fails. The galaxy_8x4-full ledger has its own spec and no
   engram_rows tasks, so nothing to copy there: its ladder and K.1 judge the rows end to end.
5. **Three framework changes come with it** (`models/demos/common/bringup/dev/BREADCRUMBS.md`):
   - F220: a model switch set in the environment (`DS41_*`) is part of a unit's stamp. Before, an A/B run with a
     `DS41_` switch reused the other side's PASS.
   - F221: a waived swapped step also covers the vs-golden check of the later steps it feeds (its vs-CPU checks still
     gate). Without it, `swap_engram_kv_vs_golden_err` failed on layer 1 (0.00350 vs 0.00335) from the waived rows.
   - F222: `Setting(test=...)`, a value pytest processes get instead of the default, only for a choice bit-identical
     to the default.

## Checked on the 2x2 (after the rebase on FIX.7)

- CPU: the engram table tests (every mode, the packed table against the per-chunk pack over the whole table) and the
  framework selftests (659 pass; the one failure is the known xing40 fixed threshold).
- Device: `C.swa_engram.engram_rows`, `C.enc_source_engram.engram_rows` (waived, rel 0.0035), `test_engram_bfp8.py`
  (bit-exact), `S.swa_engram.18` and `S.enc_source_engram.20` (every step on the device; each covers the shorter swap
  tests of its block type) pass. Logs: `/localdev/dnijemcevic/bringup/engram_2026-10-28/post_rebase/` on the QB.
- Before the rebase, the 2x2 ladder bf16 vs bfp8 rows (3 rungs) PASS with every metric within 1.3 %.
- Layer 14's block out in `S.enc_source_engram.20` is over its limit (0.058 vs 0.033) and waived. It is the same with
  bf16 rows (0.0583 vs 0.0580 with bfp8) and was the same before the rebase (0.057): it comes from the steps waived
  before (bfp4 experts, indexer), not from the rows. The WAIVED line quotes the engram_rows waiver only because F158
  names the first waived step of the block.

## What the Galaxy should run

- The 40-layer ladder at 56k and K.1 (with the "Paris" smoke) on the pushed code: the end-to-end check of bfp8 rows,
  which the 2x2 subset cannot give.
- Optional: the time of `begin_chunk`'s engram part in a served run (expect ~8 ms a layer a chunk after the build).
- Host RAM: ~209 GB for the two tables beside everything else the server holds (the owner confirmed the Galaxy host
  has it).

## Not done, open for the owner

- Reading the next chunk's rows while the device runs the current one (the async stage): not needed with the tables
  in RAM (~15 ms a chunk for both layers), kept for a host without the RAM.
- A row cache instead of the whole table: measured, not built (ENGRAM_HOST_PATH.md 3e: 38-80 % hits; text stays far
  from resident's time).
- The decode side (tt-blaze) keeps its tables on the chips as bfp8 (272 B rows), so prefill and decode now round the
  engram rows the same way; whether their conversion is bit-identical to ours was not checked.
