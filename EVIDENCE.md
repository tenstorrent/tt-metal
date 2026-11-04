# UNPACK_SYNC release after unpack_to_dest_tile_done: evidence

This branch is PR #62121 plus the race probe Anil Mahmud wrote in his review of it. The probe isn't part of the PR;
it is here with the runs that back the change.

## The race

`unpack_to_dest_tile_done` restores unpacker 0's channel-1 Z-stride with a `WRCFG` that follows the `UNPACK_TO_DEST`
semaphore post, whose `STALLWAIT` waits for the unpack to Dest to finish. On Wormhole it also restores the two format
words the TEN-3868 dummy unpack changes, after a `STALLWAIT` for the dummy unpack. Callers released `UNPACK_SYNC`
before calling it, so the `SEMGET` ran as soon as the MOP issued. A `configure_unpack_AB` right after (e.g. a
`compute_kernel_hw_startup` mid-kernel) then passes `wait_for_idle`, stores the new values from the RISC, and the
queued `WRCFG`s land on top of them. The PR moves the release after `unpack_to_dest_tile_done` in every caller:
Wormhole `_llk_unpack_A_` and tilize to Dest; Blackhole `_llk_unpack_A_`, tilize to Dest, the topk_xl copy and
top32_rm.

## The probe

`tt_metal/tt-llk/tests/python_tests/test_unpack_to_dest_reconfig_race.py` and
`tt_metal/tt-llk/tests/sources/unpack_to_dest_reconfig_race_test.cpp`: 256 times, a Float32 unpack to Dest through
`_llk_unpack_A_`, then `_llk_unpack_hw_configure_` for Float16_b, `tensix_sync()`, and a check that the Z-stride and
unpacker 0's input and output format words hold the Float16_b values. The counts go to L1 0xA0000 and the test fails
on any stale value. Changes from Anil's version: the format words are checked too, the words are read back with
`ttexalens.tt_exalens_lib.read_from_device`, and it isn't limited to Blackhole.

On hardware, from `tt_metal/tt-llk/tests` with the usual LLK test setup:

    pytest python_tests/test_unpack_to_dest_reconfig_race.py

## Results

Blackhole p100 (Anil Mahmud), stale Z-stride out of 256 reconfigures, three runs each:

| Blackhole `_llk_unpack_A_` | run 1 | run 2 | run 3 |
|---|---|---|---|
| main (`SEMGET`, then `unpack_to_dest_tile_done`) | 255 | 255 | 255 |
| `SEMGET` after `unpack_to_dest_tile_done` | 0 | 0 | 0 |

Not run on Wormhole silicon. Functional ttsim passes the probe on main and on this branch, on both chips: it finishes
an unpack when the UNPACR issues, so the `STALLWAIT` never holds the `WRCFG`s back and they land before the RISC
gets to its stores.

LLK tests on ttsim main (806e1c65), `LLK_TEST_SEED=42`, tt-metal main 5555d35d1d9 against main plus this PR, same
collected cases, passed / failed:

| Selection | Wormhole main | Wormhole PR | Blackhole main | Blackhole PR |
|---|---|---|---|---|
| `test_fused.py -k 'sfpu_binary or sfpu_unary'` | 6 / 6 | 12 / 0 | 12 / 0 | 12 / 0 |
| tilize, transpose_dest, typecast (int and Float32 cases) | 16 / 25 | 41 / 0 | 41 / 0 | 41 / 0 |
| unpack_A, datacopy, bcast, pack_rows, unpack_tilize_sweep, `-k 'Float32 or Int32'` | 387 / 301 | 688 / 0 | 494 / 0 | 494 / 0 |
| pack_rows_to_addr, pack_untilize, `-k 'Float32 or Int32'` | 16 / 140 | 156 / 0 | 168 / 0 | 168 / 0 |
| top32_rm, topk_xl, topk_xl_split, sort_headers_coexist, set_dst_write_addr_offset | | | 24 / 230 | 24 / 230 |

The Wormhole failures on main are all ttsim refusing the old dummy unpack (an FP32 or INT32 unpack to SrcA). The
Blackhole topk failures are ttsim refusals unrelated to this change (`SFPCONFIG` lane masks, a non-last UNPACR, a
PACR read-interface select), the same cases before and after. No case goes from pass to fail.
