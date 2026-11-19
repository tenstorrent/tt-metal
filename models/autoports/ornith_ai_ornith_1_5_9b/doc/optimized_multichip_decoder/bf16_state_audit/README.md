# BF16 persistent state staged-kernel source audit

Source-only AutoFix review. No device calls, resets, queued-source edits, or hardware claims. Source hashes and exact scan comparison are in `source_audit.json`.

**Verdict:** no concrete CB ownership, backpressure, page-size, or pack-format defect found in the staged design. This supports testing it; it does not establish numerical acceptance. BF16 state is intentionally rounded after every token and may accumulate unacceptable drift. Existing quality gates must remain unchanged.

## Buffer ledger and progress

| CB | Producer → consumer | Pages per head | Bytes/page | Purpose |
|---|---|---:|---:|---|
| 28 | NCRISC → TRISC0 | push/pop 16 | 2048 | Persistent BF16 state input, then unqueued NCRISC scratch |
| 5 | TRISC2 → TRISC0 | push/pop 16 | 4096 | Expanded FP32 current state |
| 7 | TRISC2 → BRISC | push/pop 16 | 2048 | BF16 updated state |

`tests/kernels/bf16_state_staged/reader.cpp:18–23` reads all 16 BF16 state tiles, waits for NoC, publishes them, then reserves all 16 pages again for scratch. `compute.cpp:134–143` consumes that input and publishes the FP32 expansion **before** waiting for Q/K/V. Thus the reader's second reserve cannot form a QKV dependency cycle. The scratch reserve does not push or advance the CB write pointer; NCRISC remains the sole producer. Every later scratch read is barrier-protected; its maximum transfer is 4096 bytes (`reader.cpp:60`), within the 32768-byte allocation. All state transactions span the complete 16-page capacity, so ring pointers return to phase zero after each head.

The UNPACK-side pop can occur before the final PACK completes, but each incoming tile has already been copied to DST before that pop; scratch reuse does not overwrite a still-needed input tile. The expansion's push/wait protects consumption of CB5. Reader, compute, and writer use the same first/stride/iteration partition (`reader.cpp:114–116`, `compute.cpp:194–197`, `writer.cpp:25–27`, helper:94–108). Multiple heads on one core have disjoint persistent slabs. The writer drains output before state, but compute has room for a complete state publication before producing output, so this ordering introduces no cycle.

## Formats and recurrence rounding

`compute.cpp:136–142` explicitly restores FP32 packing to CB5 and copies BF16 input exactly into FP32 storage. `compute.cpp:147–148` restores FP32 scan source/pack formats. `scan_state_bf16.hpp:108` changes only the final persistent-state pack to BF16; all scan inputs, intermediate packs, and HiFi4 FP32 DST accumulation remain the accepted ordered scan. Automated comparison confirms that the local scan differs from the shared `chunk_gdn_math.hpp:1046` scan only in function name and that single pack reconfiguration.

Rounding order remains: FP32-packed residual `-(k*beta*decay) @ S + I @ (v*beta)`; FP32-packed inverse product; output sum `q_decay @ S + intra @ v_new` in one FP32 DST accumulation; final state sum `(decay*I) @ S + k_transpose @ v_new` in one FP32 DST accumulation, now packed to BF16. Output is formed before the new state rounding and uses the expanded previous state. On the next token the stored BF16 state is widened exactly. `api/compute/matmul.h:107–120` configures UNPACK/MATH, not PACK, so matmul initialization does not undo the final BF16 pack selection. Postnorm explicitly packs its BF16 outputs; the next head explicitly restores FP32 CB5 packing.

Comments claiming every scan CB is FP32 are stale (`scan_state_bf16.hpp:11`, `compute.cpp:188`): the final destination is now BF16. This is a documentation defect, not evidence of a runtime defect. No queued file was changed.

## Writer and memory accounting

Helper lines69–82 give CB28 16×2048 bytes, CB5 16×4096, and CB7 16×2048. Reader/writer state accessors both use 2048-byte tile pages, and writer lines20–22 transfer exactly 16 such pages to `head*16 + tile`; there is no FP32-page stride left in the persistent-state transfer. Output is allocated in L1 by helper lines29–35, so its existing 32-byte face-row writes do not violate the DRAM alignment constraint. B1–32 row/face indexing stays within the single padded batch tile; only logical rows are written.

Persistent recurrent storage per chip at B32 changes from 32×8×128×128×4 = 16 MiB to 8 MiB. These three CB allocations together change from 64+64+4 = 132 KiB to 64+32+32 = 128 KiB per active core. These are source-derived allocation sizes, not measured memory/context feasibility. Update `doc/context_contract.json` if selecting this change alters the maximum supported context.

## Dispatch scope and required follow-up

`multichip_bf16_state_staged_candidate.py:9–15` converts only pooled batches larger than one. Private states and batch 1 remain FP32 and therefore do not validate BF16 storage. Reordered/subset/mixed pooled requests use the production per-request extraction path (`tt/multichip_decoder.py:706–714,1246–1283`), widened by the candidate's read override and rounded by its write override. This is a reasonable source-level contract but needs isolation/reset/trace coverage if retained.

Queued B6/B32 short and 1024-token tests are necessary numerical evidence. A minimal discriminating direct test should initialize the BF16 candidate and accepted FP32 post-loop control from the **same BF16 state**, then after each control step round its resulting state to BF16 and widen to FP32 for the next step. Compare candidate output against that control, and candidate state against the BF16-rounded control state. This isolates implementation errors from the intended persistent-state quantization; it supplements, never replaces, unchanged comparisons with the full FP32 baseline.

If retained, add direct B1/B2/B13/B14/B16/B17/B31/B32 coverage, small waves with repeated heads, DRAM/L1, zero/tiny QK and near-zero beta, signed nonconstant state/gamma/gates, watcher runs, and repeated trace replay. Re-run pooled subset/reordered/single-nonzero-row/mixed-private-state cases, per-request reset with untouched-neighbor bitwise checks, and input/state refresh after separate full/subset trace captures. Candidate read/write dtype conversions must be warmed and covered by allocation tracking. No hardware tests were launched by this audit.
