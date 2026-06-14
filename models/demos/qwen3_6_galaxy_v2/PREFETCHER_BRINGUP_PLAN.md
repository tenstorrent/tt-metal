# Plan: Enable the DRAM prefetcher on qwen3.6 BH (TP=32) decode — the path to 60+ tok/s

## Why (the whole motivation)
- qwen3-32B at **TP=32 + prefetcher = >60 tok/s**; TP=8 only 34. qwen3.6 on TP=32 is stuck at **~27**
  ONLY because it runs `use_prefetcher=False`.
- Device-kernel analysis this session proved decode on BH-without-prefetcher is a **no-free-lunch
  tradeoff**: sharding minimizes weight-DRAM-read but pays CCL; replication kills CCL but balloons
  the read (col-replicated MLP measured ~335µs vs sharded ~290µs — replication LOSES at device-kernel).
  Neither fusion (+8.1% committed, device-kernel ~flat) nor col-replication beats it.
- The **prefetcher is the only lever that attacks BOTH**: it streams the DRAM weight into L1 overlapped
  with compute/CCL, hiding the read while KEEPING sharding's low CCL. That's how qwen3-32B TP=32 hits 60+.

## Feasibility — PROVEN (not a question anymore)
`test_prefetcher_BH.py` on this BH box (MESH_DEVICE=P150x8): **Qwen3-32B PASSES at PCC 0.99999**
(dram_prefetcher + global_cb + ring matmul + trace, num_receiver_cores=3) across qkv/wo/ff1/ff3/ff2.
The "NO-GO bank x=8" note was STALE. 8/10-receiver fails are a custom-core-mapping config (legal list),
NOT a bank limit.

## Root cause of qwen3.6's prefetcher being off
`tt/model_config.py:179 get_core_ranges` (copied from llama70b) hardcodes the **WH** layout:
`all_dram_cores = [CoreCoord(idx,0) for idx in range(12)]` (12 banks) + senders on cols 0/4. On BH there
are **8 banks at X=[1,3,2,0,5,7,6,4]**, senders on cols **0/7**, receivers cols **1-7 / 8-11**. `range(12)`
indexes bank 8+ → "bank x=8". And `tt/qwen36_model_config.py:111 self.use_prefetcher = False`.

## The BH-correct implementation ALREADY EXISTS
`models/tt_transformers/tt/prefetcher.py` (arch-aware) + `prefetcher/prefetcher_config.yaml`:
```
blackhole: dram_banks:[1,3,2,0,5,7,6,4]  legal_receiver_cores:[1,2,3,8,10]
  bank_ordered_y_coords: left:[9,1,7,3] right:[0,2,6,4]
  sender_cols: left:0 right:7   receiver_cols: left:[1,7] right:[8,11]
  sender_rows.active: left:[0,3,7,9] right:[1,4,6,9]
```
`PrefetcherCoreConfig(num_receiver_cores=2or3)` builds the senders/receivers/dram_banks for BH.

## Integration steps (the work)
1. **BH `get_core_ranges`** (tt/model_config.py:179): arch-gate. On `is_blackhole()`, build the 8 return
   values (active_sender_cores, dram_cores, all_sender_cores, active_receiver_cores_list,
   all_receiver_cores, worker_cores_range_set, mm_optimised_ring_cores, hop_grid) from the BH config —
   reuse `tt_transformers.tt.prefetcher.PrefetcherCoreConfig` (don't hand-rewrite). num_reader_cores=8 (not 12).
2. **Enable** `use_prefetcher=True` for decode (qwen36_model_config.py:111) — arch-gated to BH.
3. **PrefetcherSetup (prefetcher_common.py)**: its global_cb sender/receiver mapping + sub-device must use
   the BH cores from step 1; `num_reader_cores=12`→`len(dram_banks)`; the "WH cols 0/4 senders" comment
   (line 94) → cols 0/7. Validate `create_global_cb()` + weight insertion (`insert_tensor`) on BH.
4. **Decode matmuls through global_cb**: the ring matmul progcfgs already take `prefetch=use_prefetcher`;
   the MLP `double_matmul_line_reduce_scatter` (llama_mlp.py:130) passes
   `global_cb=prefetcher_setup.global_circular_buffer`. With the prefetcher on, the FF12 weight-reshard
   AND the FF2 w2-L1-reshard problems vanish (prefetcher IS the L1 streaming).
   ⚠️ The RS-fused `llama_rs_matmul` on BH is still unproven (its ring-40 cousin deadlocked) — validate it
   first in isolation with the prefetcher global_cb before the full decode.

## Measurement (device-kernel, NOT wall-clock; NOT via test_prefetcher_BH)
The standalone PoC is trace-based + runs matmuls on a sub-device → Tracy did NOT capture the prefetcher
matmul per-op (only verification ops; 5 op-codes, no Matmul). So measure via the qwen36 EAGER decode
harness `tests/profile_decode_eager.py` (which DID cleanly capture matmuls+CCL device-kernel), with the
prefetcher enabled: compare prefetcher decode-MLP device-kernel vs the current (~290µs sharded). Then the
full 64L generator demo tok/s vs 27.35 baseline.

## Validation gates (unit-test-driven, per CLAUDE.md)
- G1: BH `get_core_ranges` returns valid disjoint cores (CPU shape test) + `create_global_cb()` succeeds
  on BH (no "bank x=8").
- G2: a single FF12 `double_matmul_line_reduce_scatter` with the prefetcher global_cb on BH: PCC>0.99,
  no deadlock (isolation, decode-mode tt_ccl). [the RS-on-BH risk gate]
- G3: profile_decode_eager 1L with prefetcher: next_tok matches baseline + device-kernel measured.
- G4: 64L generator demo: coherent + tok/s vs 27.35 (target → toward 60+).

## Risks
- The RS-fused `llama_rs_matmul` on BH (G2) is the key unknown (ring-40 deadlocked; standard-config +
  prefetcher untested). If it deadlocks, fall back to plain prefetcher matmuls (FF1/FF3) + separate
  line_reduce_scatter — still gets the matmul weight-streaming win.
- Board is wedge-prone; keep runs serial, to-file, never `| head`, never kill a device process, set
  MESH_DEVICE. See [[qwen36-fused-ff2-task3-resume]] for the device-safety rules + full context.
