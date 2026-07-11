# Quasar perf suite progress summary

Updated: 2026-07-11 ~15:35 UTC  
Host: tensix-l-01-special-ndivnic-for-reservation-109931  
Suite map: 13 tests (matmul excluded), suite ids 02–14.

## Overall status

| PerfRunType | Suite status | CSV reports (`*.csv` / `*.post.csv`) | Notes |
|-------------|--------------|--------------------------------------|-------|
| **L1_TO_L1** | **DONE — all tests PASSED** | *(counted done per request; treat as full success)* | User-confirmed: every test worked |
| **UNPACK_ISOLATE** | Mostly done | 11× `.csv` + matching `.post.csv` (22 files) | See gaps below |
| **MATH_ISOLATE** | Partial | 4× `.csv` (+ `.post.csv` = 8 files) | Several failures / hangs / missing |
| **PACK_ISOLATE** | **In progress** (restarted) | 0 | Prior run invalid (`-k` bug); re-running with fixed filter |
| **L1_CONGESTION** | Not started | 0 | Queued after PACK |

**Currently running:** `PACK_ISOLATE` / `perf_eltwise_unary_datacopy_quasar` with `-k PerfRunType.PACK_ISOLATE` (attempt after exalens timeout retry).  
**Queued after PACK:** `L1_CONGESTION` → incomplete-rerun pass (skip PASSED / HANG / COMPILE_FAIL).

---

## Suite test map

| ID | Test |
|----|------|
| 02 | `perf_eltwise_unary_datacopy_quasar` |
| 03 | `perf_eltwise_binary_broadcast_quasar` |
| 04 | `perf_eltwise_binary_quasar` |
| 05 | `perf_unpack_tilize_quasar` |
| 06 | `perf_unpack_unary_operand_quasar` |
| 07 | `perf_transpose_dest_quasar` |
| 08 | `perf_pack_quasar` |
| 09 | `perf_pack_untilize_quasar` |
| 10 | `perf_unary_broadcast_quasar` |
| 11 | `perf_pack_l1_acc_quasar` |
| 12 | `perf_reduce_quasar` |
| 13 | `perf_eltwise_binary_reuse_dest_quasar` |
| 14 | `perf_unpack_reduce_col_tilizeA_strided_quasar` |

---

## L1_TO_L1 (done — all passed)

Per user direction: treat as **complete success** for all 13 suite tests.

| ID | Test | Status |
|----|------|--------|
| 02–14 | all suite tests above | **PASSED** |

---

## UNPACK_ISOLATE

| ID | Test | Status | Duration |
|----|------|--------|----------|
| 02 | `perf_eltwise_unary_datacopy_quasar` | **PASSED** | 24m04s |
| 03 | `perf_eltwise_binary_broadcast_quasar` | **PASSED** | 13m13s |
| 04 | `perf_eltwise_binary_quasar` | **PASSED** | 4m03s |
| 05 | `perf_unpack_tilize_quasar` | **PASSED** | 2m32s |
| 06 | `perf_unpack_unary_operand_quasar` | **PASSED** | 2m32s |
| 07 | `perf_transpose_dest_quasar` | **PASSED** | 4m43s |
| 08 | `perf_pack_quasar` | **PASSED** | 16m23s |
| 09 | `perf_pack_untilize_quasar` | **PASSED** | 15m16s |
| 10 | `perf_unary_broadcast_quasar` | **HANG** | 21m35s |
| 11 | `perf_pack_l1_acc_quasar` | **FAILED** | 0m12s |
| 12 | `perf_reduce_quasar` | *not finished in suite report* | — |
| 13 | `perf_eltwise_binary_reuse_dest_quasar` | *not finished in suite report* | — |
| 14 | `perf_unpack_reduce_col_tilizeA_strided_quasar` | *not finished in suite report* | — |

**CSVs present:** datacopy, binary_broadcast, binary, unpack_tilize, unpack_unary, transpose, pack, pack_untilize, reduce, reuse_dest, unpack_reduce_col.  
**No CSV / open:** unary_broadcast (hang), pack_l1_acc (failed). Ids 12–14 have CSVs on disk (may be from earlier tagging); suite report never recorded them as PASSED.

**Incomplete-rerun plan:** skip PASSED + HANG; rerun FAILED / missing (e.g. 11, and any unusable 12–14).

---

## MATH_ISOLATE

| ID | Test | Status | Duration |
|----|------|--------|----------|
| 02 | `perf_eltwise_unary_datacopy_quasar` | **FAILED** | 20m14s |
| 03 | `perf_eltwise_binary_broadcast_quasar` | **EXALENS_TIMEOUT** | 55m10s |
| 04 | `perf_eltwise_binary_quasar` | *missing from report* | — |
| 05 | `perf_unpack_tilize_quasar` | *missing from report* | — |
| 06 | `perf_unpack_unary_operand_quasar` | *missing from report* | — |
| 07 | `perf_transpose_dest_quasar` | **PASSED** | 7m13s |
| 08 | `perf_pack_quasar` | **PASSED** | 20m34s |
| 09 | `perf_pack_untilize_quasar` | **PASSED** | 5m53s |
| 10 | `perf_unary_broadcast_quasar` | **PASSED** | 1m32s |
| 11 | `perf_pack_l1_acc_quasar` | **HANG** | 43m08s |
| 12 | `perf_reduce_quasar` | *missing from report* (output file exists) | — |
| 13 | `perf_eltwise_binary_reuse_dest_quasar` | *missing from report* | — |
| 14 | `perf_unpack_reduce_col_tilizeA_strided_quasar` | **HANG** | 8m34s |

**CSVs present:** transpose, pack, pack_untilize, unary_broadcast.  
Suite exited early after a mid-run script edit (parse error); incomplete-rerun will pick up FAILED / EXALENS_TIMEOUT / missing (not HANG).

---

## PACK_ISOLATE

### First attempt (invalid — do not use)

`-k PACK_ISOLATE` also matched `UNPACK_ISOLATE` (substring). Mixed schemas → CSV combine refused to write. Many HANG/FAILED; **0 PACK CSVs**.

### Restart (current)

- Filter fixed to `-k PerfRunType.PACK_ISOLATE`
- Restarted ~15:23 UTC; order: PACK → L1_CONGESTION → L1_TO_L1
- In progress on id 02 (`perf_eltwise_unary_datacopy_quasar`), retrying after exalens timeout

---

## L1_CONGESTION

Not started. Queued after PACK completes.

---

## Issues / lessons

1. **`-k PACK_ISOLATE` substring bug** — also selected UNPACK; fixed to `PerfRunType.<name>`.
2. **Hang policy** — single test >5 min after exalens ready → kill; no retry. Exalens 600s timeout → retry up to 5×.
3. **Incomplete rerun** — after main queue: skip PASSED / HANG / COMPILE_FAIL; fill gaps only.
4. **Editing the suite script while it was running** contributed to an early MATH exit.

---

## Remaining work

1. Finish **PACK_ISOLATE** (all 13 tests) with fixed `-k`
2. Run **L1_CONGESTION** (full)
3. Run **incomplete-rerun** for UNPACK / MATH / PACK / L1_CONGESTION gaps  
   (**L1_TO_L1** treated as already done — do not rerun)
4. Rename any new CSVs with `_<PerfRunType>` suffix

---

## CSV inventory (tagged on disk)

| Run type | `.csv` bases present |
|----------|----------------------|
| UNPACK_ISOLATE | 11 tests (see list above) |
| MATH_ISOLATE | transpose, pack, pack_untilize, unary_broadcast |
| PACK_ISOLATE | *(none yet — restart in progress)* |
| L1_CONGESTION | *(none)* |
| L1_TO_L1 | *(counted done; not re-collected in this pass)* |
