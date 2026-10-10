# `noc_async_read_barrier_with_trid` returns before the read has landed (repro notes)

Found 2026-10-10 in combine_fabric2d's reader (flat_combine_overlap work, `ce47cef24d5`). Written down so it can be
tested in isolation later; nothing in this directory has been run yet.

## What we saw

combine_fabric2d's reader fills ring slots with DRAM reads and hands them to the sender on the same core, which sends
them over the fabric. We changed its per-batch `noc_async_read_barrier()` to
`noc_async_read_set_trid(t)` + reads + `noc_async_read_barrier_with_trid(t)` (to keep two batches in flight).
Result: 1-3 wrong output slots out of 10240 per failing case (max |diff| 6-9, i.e. partly stale token rows), in standalone
combine and in the overlap, with or without the pipelining; reproduced on every run (LoudBox 8 x p150, 8 x 1 ring,
`test_flat_combine_overlap.py`). Two independent changes each removed it:

- global barriers again (`CMBF2D_NO_TRID=1` probe: `noc_async_read_barrier()` instead of the trid barrier), or
- waiting for the read command buffer before polling the id:
  `while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {}` then `noc_async_read_barrier_with_trid(t)` (the fix kept).

## Theory (unconfirmed)

`noc_async_read_barrier_with_trid(t)` spins on `ncrisc_noc_read_with_transaction_id_flushed(noc, t)`, i.e.
`NOC_STATUS(NIU_MST_REQS_OUTSTANDING_ID(t)) == 0` (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h`).
That counter counts requests the NIU has accepted. `noc_async_read` returns as soon as it has programmed the read
command buffer and kicked it; the NIU takes the request some cycles later. If the barrier polls in that window, the
last read is not yet counted, the count is 0 and the barrier returns with that read's data still to come. The global
barrier instead compares `NIU_MST_RD_RESP_RECEIVED` against the software count of issued reads, which already includes
the queued read, so it has no such window.

Same family as the write-side `noc_async_writes_flushed` misunderstanding reported by Sofija Jovic (the status says
"request accounted", not "data moved"), but a different signal and the read direction.

Open questions the repro should answer:
1. Does the window exist with one read per batch, or only when the command buffer is still busy with an earlier read
   of the same batch (back-to-back issues)?
2. Is it only the LAST read of a batch (as the theory says) or any?
3. Size dependence: one packet (<= NOC_MAX_BURST_SIZE) vs reads split into several packets by
   `noc_async_read` (which waits for the command buffer between packets, so only the final packet could be exposed).
4. Wormhole too? (same code path in `tt-1xx/wormhole/noc_nonblocking_api.h`.)
5. Does `noc_async_read_one_packet_with_state_with_trid` (the "proper" trid read API) have the same exposure?

## Standalone repro (`trid_read_race.cpp`, `test_trid_read_race.py`)

One Tensix core, one data-movement kernel (NCRISC, NoC 0), no fabric:

1. Host fills an interleaved DRAM buffer of `NPAGES` pages of `READ_BYTES`; every uint32 of page p is `p + 1`
   (never the sentinel).
2. Per iteration: write a sentinel (`0xDEADBEEF`) into the last 16 B of each of the batch's `K` L1 destinations; set a
   trid; issue `K` reads (consecutive pages, so consecutive banks); barrier as selected; `invalidate_l1_cache()`;
   then check each destination's last word. Still the sentinel = the barrier returned before that read landed;
   another wrong value = partial / misplaced data. Count failures and the batch index of the first one.
3. Defines select the barrier: none = `noc_async_read_barrier_with_trid` (expected to fail),
   `DRAIN_CMDBUF` = drain the command buffer first (expected to pass), `GLOBAL_BARRIER` = `noc_async_read_barrier`
   (expected to pass). TODO for question 5: a variant on `noc_async_read_one_packet_set_state` /
   `noc_async_read_one_packet_with_state_with_trid` instead of set_trid + `noc_async_read`.
4. Results go to a small DRAM output: [failures, iterations, first failing iteration, first failing j, failures on the
   last read of a batch, failures on other reads].

Sweep: K in {1, 2, 4, 8}, READ_BYTES in {64, 2048, 14336 (combine's Kimi token), 32768 (two packets)}, ITERS 100000.
Prediction from the theory: failures with K >= 2 (and maybe 1), only on j = K - 1, none with `DRAIN_CMDBUF` or
`GLOBAL_BARRIER`. If nothing fails at all, the window needs NoC load (combine had the fabric and the sender's NoC 1
traffic on the same core and the flat expert's traffic on the chip); add a second kernel on the BRISC (NoC 1) that
streams writes to a neighbour, and a few other cores hammering the same DRAM banks.

Run (once written up as a real test):
```
cd /localdev/mstaletovic/tt-metal-pr58093
source /localdev/mstaletovic/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD:$PWD/tools
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/flat_combine_tools/trid_read_barrier_race/test_trid_read_race.py
```

## In-situ repro (the case that showed it)

On `mstaletovic/flat-combine-overlap` at `ce47cef24d5` (or later), in
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/reader_combine_fabric2d.cpp`
delete the two `while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {}` loops (in `Reader::finish` and
`Reader::prefetch_metadata`), then run the accuracy test with the 64 gu + 19 down layout:
```
TT_MESH_GRAPH_DESC_PATH=models/demos/deepseek_v3_d_p/tests/op_unit_tests/flat_combine_tools/p150_x8_ring_8x1.textproto \
TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 FLAT_CMB_PIN=1 MIMO_FL_ROWS=1,9 MIMO_FL_RD_SAMECOL=1 \
MIMO_FL_XDOWN="1,0;4,0;9,0;10,0" CMBF2D_PIPE=0 scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_flat_combine_overlap.py -k "test_flat_combine_overlap and not perf"
```
Expected (as observed): one of the four cases fails with "flat then combine: 10237-10239/10240" (standalone combine)
and the overlap not bit-identical; `CMBF2D_NO_TRID=1` (global barriers) makes all four pass again.

## Results (2026-10-10)

**Standalone (`test_trid_read_race.py`): does NOT reproduce.** 0 failures in every case: K 1-8 x 64 B .. 32 KiB
(100 000 batches each), with 10 other row-0 cores reading the same DRAM pages back to back
(`test_trid_read_race_loaded`), and combine's metadata-prefetch shape (16 / 32 / 64 back-to-back 64 B reads on one id,
quiet and loaded, `test_trid_read_race_meta`); plain trid barrier, drained, and global alike.

**In situ: reproduces, and it is the metadata-prefetch barrier.** Removing the command-buffer drain from:
| barrier | `CMBF2D_PIPE=0` | `CMBF2D_PIPE=1` |
|---|---|---|
| both | fails (standalone combine 10238 / 10239 of 10240 in two cases) | passes |
| token batches only | passes | - |
| metadata prefetch only | fails (10238 / 10240) | - |

With pipelining the token-batch barrier runs one batch late, so the drain is long done; the metadata barrier is
polled right after its last read either way. Ruled out: DRAM alignment (12 B metadata rows are padded to 64 B pages
on Blackhole, so the reads are aligned, as in the repro) and counter wrap (the per-id outstanding count holds 255).
What in combine's context opens the window is still open: candidates are the sender on the same core (BRISC, NoC 1,
fabric writes), the read VC reprogramming, and the reads interleaved with metadata use. Next step for whoever picks it
up: grow the standalone kernel toward the reader (same TensorAccessor, a BRISC kernel pushing fabric-sized NoC 1
writes, VC 0 reads) until it fails.
