# h46k final increment: paged decode attention (merge notes)

Deliverable: `/mnt/tt-data/ssinghal/wt/h46k/changes.diff` (224 lines, `git apply --check -p1` clean against main 9733d47f2f8): chunked `PagedKVPool.stage_commit`
(`tt/paged_ops.py`), `tests/test_paged_ops_device.py`, `tests/test_paged_decode_steps.py`. Everything else I built (paged ops/kernels, attention classes, step state, indexer
per-user valid length + `write_keys`, chain plumbing, dumps, tests) is already in main. API / handoff contract: `/mnt/tt-data/ssinghal/wt/h46k/paged_api.md`.

## What changed in this increment
* `stage_commit` uploads the staging pool in 16384-row chunks through `paged_scatter_rows` (in place; all-zero chunks skipped; no host fp32 copy for an fp8 pool). The old version allocated a
  SECOND pool-sized device buffer (that was the batch-128 bf16 OOM: 5.56 GB twice) -- bf16 path validated (`test_stage_commit_chunked[bf16]` passes).
* `test_indexer_users_of_every_mesh_row` (new): users of every mesh row, different keys / queries / valid lengths, 2 columns per row checked: PASSED for matmul (worst selected score mass vs
  host top-512: 0.9999) and fused (0.9982) -> no per-device causal-offset problem for the DECODE indexer (matmul, the default <= 65536 entries, does not call `indexer_score_dsa` at all; the fused
  call uses kv_len = n_alloc, chunk_start = n_alloc - 32, i.e. everything visible).
* `test_paged_decode_steps.py`: synthetic filled-cache mode (`DSV41_SYN_CTX`), one-user dumps tiled to the batch, Engram history sized to the prompt, a PAGED_MEM2 DRAM line after trace.

## Verified on device (4x8 mesh)
* Ops vs host model (`paged_kv_step`, `paged_scatter_rows` incl. `base_offset`, `write_keys` bf16/bfp8, ragged per-user valid lengths on both indexer backends, chunked commit bf16).
* Attention vs the checkpoint reference: real data, short contexts 0.9996 (= the existing attention); 64k synthetic oracle ids 0.997-0.9995; real 2048-token prompt: layer 20 0.9991 (device indexer 0.9989),
  layer 2 0.987; device top-512 covers as much attention mass as the reference's own (0.6587 vs 0.6583, 0.5333 vs 0.5321).
* 40-layer decode, teacher-forced, indexer ACTIVE (compressed entries 1024 / 2048 > 512), real 2048-token prompt, batch 16, bf16 pool, matmul backend: logits PCC vs reference step 0 / 1 / 2 =
  0.99729 / 0.99635 / 0.99242, tokens match 16/16 at every step, 48.2 ms/token steady (20.8 tok/s/user). ISL 128 (real dump, indexer off): 49.4 ms/token, PCC 0.989-0.991.

## ISL 64k decode rows (40 layers, paged, indexer ON, synthetic filled caches + random Engram rows, TEACHER-FORCED timing: no sampling feedback, no reference; wall incl. host inputs + readback; 6 steps, steady = steps 1-5)
| batch (users/row) | pool | ms/token | tok/s/user | aggregate tok/s | pool per chip | free GiB/chip after pool | free GiB/chip after embedding/head/Engram/trace |
|---|---|---|---|---|---|---|---|
| 16 (4) | bf16 | 61.5 (device 57.2) | 16.3 | 260 | 662 MiB | 8.42 | not logged (run predates the line) |
| 32 (8) | bf16 | 73.7 | 13.6 | 435 | 1325 MiB | 7.69 | 6.07 |
| 64 (16) | bf16 | 89.7 | 11.1 | 712 | 2650 MiB | 6.23 | 4.61 |
| 128 (32) | fp8 | 124.8 | 8.0 | 1026 | 2650 MiB (the log's "5300 MiB" assumes 1 KiB rows; fp8 rows are 512 B) | 5.89 | 4.27 (largest free block 543 MiB/bank) |
Indexer backend = matmul (every layer <= 65536 entries). Short-context reference point: 48.6 ms/token at positions <= 14 (batch 16), 49.4 ms at ISL 128. The 64k premium at batch 16 is 13.6 ms device time
(8 indexer layers + 64k sparse attention). Batch 128 bf16 (old commit) failed: OOM in `stage_commit` (double pool allocation); NOT re-run with the chunked commit.
Largest ISL at batch 128 (ESTIMATES from the measured free DRAM, not run; rule: no runs > 64k): fp8 fits 64k with 4.27 GiB/chip left; each extra token costs ~51 KiB/chip at 32 users/row
(pool 1.25 KiB + key slab 0.34 KiB per user) -> roughly 140k+ tokens in total; bf16 pool (2.5 KiB/token/user) costs 2.65 GiB more at 64k, leaving ~1.6 GiB -> roughly 80k tokens.
Per-layer indexer cost at batch > 16: NOT measured separately in the full model (no profiling run). Single-layer measurements (matmul, U = users/row): U=4: 435-460 us at 2k-4k, 594 us (16k, layer 20),
770 / 1132 us (64k, layers 2 / 20); U=32 at 4k: 686 / 782 us. The matmul backend is batched over users (ONE call per layer, not per user); the fused op (`indexer_score_dsa`) takes ONE user per call (U calls).

## NOT verified / open
1. Ratio-2 layers (2, 8, 14), group-completing step on real data: layer 2 PCC 0.987 at 2048 tokens (0.92 at 256 tokens); layer 20 (ratio 1) is fine. Cause not isolated (new latent matches the reference entry at PCC 0.995).
2. fp8 pool: kernel conversion == torch's e4m3 cast in a CPU model, device readback matches EXCEPT subnormal values (|x| < 2^-6 read back as 0: 3879 of 358400 = the fraction of randn below 2^-6); whether the
   flush is in the readback typecast or in the write is unknown. fp8 accuracy on REAL data not measured (synthetic oracle PCC 0.9969-0.9983 vs bf16 0.9976-0.9995). The updated unit test excludes the subnormal range (edited after its last run: UNRUN). Keep fp8 opt-in (`DSV41_PAGED_KV=fp8`).
3. Layers 24/28/32/36 score densely against layer 20's keys (the candidate-block hierarchy of layer 20 is NOT implemented); the 2k accuracy above includes that deviation.
4. Contexts > 64k, 4k real indexer-active accuracy (the 4096-token dump was started: `/mnt/tt-data/ssinghal/dsv4-prefill-s4096b1f`, not used), batch > 16 accuracy (timing only), spec decode (`nq > 1`) on the model.
5. `DSV41_PAGED_RAGGED=1` per-user valid lengths: op-level tested only (not in a full-model run). The indexer's valid length is otherwise user 0's position.
6. bench scripts `bench_launch.sh` / `bench_seq2.sh` (overlay root) are mine, not part of the diff; an early version of `bench_launch.sh` attached the hang watcher to a lock waiter and caused two spurious device resets on .44 (fixed).
