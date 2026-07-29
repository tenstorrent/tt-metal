# b07u08 — PASS (Blackhole Galaxy, 2026-07-29)

**b07u08 is clean. 27/27 bit-exact, 18:18.** It shows none of the non-determinism that
b06u02 shows, on the same repo, the same commit and the same mesh mapping.

| | b07u08 | b06u02 |
|---|---|---|
| Verdict | PASS 27/27 | FAIL — one bad Tensix core |
| Collectives (18 tests) | bit-exact | bit-exact |
| Local compute, no CCL | bit-exact, all 32 chips | shard 21 diverges, 1.7% of output |
| readback / eltwise | bit-exact | bit-exact |
| matmul1 / matmul2 | bit-exact | shard 21 diverges |
| matmul core locality | bit-exact, all 3 shapes | block idx (M 8 of 10, N 6 of 12), invariant |
| Culprit | none | device 14, PCI `0000:47:00.0` |

## What was run

```
cd /data/nmilicevic/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=/data/nmilicevic/tt-metal PYTHONPATH=/data/nmilicevic/tt-metal && mpirun --bind-to none --pernode --tag-output python3 -u -m pytest models/demos/deepseek_v3_d_p/tests/test_det_ccl_micro.py -p no:randomly -s -q --durations=0
```

Host `bh-glx-b07u08`, driver `tenstorrent 2.10.0`, 32 Blackhole boards. Log kept at
`/data/nmilicevic/b07u08_det_full.log`. Needs no weights, no model, no trace and no DeepSeek
env vars.

Faster subset when only the matmul question matters — 8 tests, about 3 min of test work:

```
mpirun --bind-to none --pernode --tag-output python3 -u -m pytest models/demos/deepseek_v3_d_p/tests/test_det_ccl_micro.py -p no:randomly -s -q -k "local_op or matmul_core or device_mapping"
```

The 18 CCL tests are the slow half and they pass on the bad box too, so they say nothing
about the defect.

## Result

```
27 passed, 3 warnings in 1098.09s (0:18:18)
```

Every local test reports `chips=[]` at every iteration — no chip disagrees with the other 31,
and none disagrees with itself between runs:

```
local matmul chain, no CCL: 10 iterations, seq_local=3200
  iter 1..9: BIT-EXACT ndiff=0/734003200 maxabs=0.000e+00 chips=[]
local readback bit-exact across 8 iterations and all 32 chips
local eltwise  bit-exact across 8 iterations and all 32 chips
local matmul1  bit-exact across 8 iterations and all 32 chips
local matmul2  bit-exact across 8 iterations and all 32 chips
matmul 3200x7168 @ 7168x4608 -> output 100x144 tiles: bit-exact, 6 iterations, all 32 chips
matmul 3200x7168 @ 7168x2304 -> output 100x72  tiles: bit-exact, 6 iterations, all 32 chips
matmul 1600x7168 @ 7168x4608 -> output 50x144  tiles: bit-exact, 6 iterations, all 32 chips
```

The mesh mapping is identical to b06u02, so the comparison is apples to apples:

```
row 0: shard idx [0, 1, 2, 3]     -> device ids [0, 4, 28, 24]
row 1: shard idx [4, 5, 6, 7]     -> device ids [1, 5, 29, 25]
row 2: shard idx [8, 9, 10, 11]   -> device ids [2, 6, 30, 26]
row 3: shard idx [12, 13, 14, 15] -> device ids [3, 7, 31, 27]
row 4: shard idx [16, 17, 18, 19] -> device ids [11, 15, 23, 19]
row 5: shard idx [20, 21, 22, 23] -> device ids [10, 14, 22, 18]
row 6: shard idx [24, 25, 26, 27] -> device ids [9, 13, 21, 17]
row 7: shard idx [28, 29, 30, 31] -> device ids [8, 12, 20, 16]
```

Row 5 is the row that fails on b06u02 and passes here, which rules out anything mapping- or
topology-shaped: shard 21 is not a fixed weak spot in the mesh, it is one physical chip on
one box.

## One thing to expect before the tests run

b07u08 first errored all 27 tests in 22 s, and `tt-smi -ls` itself failed:

```
Read 0xffffffff over PCIe ID 15: the board should be reset.
RuntimeError: MMIO per-op timeout: 4B load took 55632 us (budget=2 ms)   # distributed.py:671
```

That is a wedged board at `open_mesh_device`, before any kernel runs — a no-result, not a
failing result, and unrelated to determinism. `tt-smi -glx_reset_auto` cleared it and the box
then passed 27/27. Use `tt-smi -glx_reset_auto` on these Galaxy boxes, not `tt-smi -r`.

## Resuming on b06u02

The bad-box evidence and the decision ladder are in `DETERMINISM_BISECTION.md`. Nothing here
needs redoing on b06u02; the open item there is not a software question:

- Re-confirm with the 8-test subset that device 14 (PCI `0000:47:00.0`) still fails, and that
  the block index is still `(M 8 of 10, N 6 of 12)`.
- Done 2026-07-29, same failure set — see `DETERMINISM_RESULT_b06u02.md`.
- Then the next variable is the firmware/KMD pair, not an RMA: this box is on fw `19.12.0` /
  KMD `2.10.0`, b06u02 on `19.8.1.0` / `2.8.0`, and both reported-failing boxes are on the
  older pair. Bring b06u02 to the newer pair, rerun the subset, and only then call it a die.
  b07u02 is reported to have the same symptom and is worth running the subset on, recording
  *which* shard fails there.
- The two side issues found while bisecting are independent of the box and still open:
  `conftest.py` `weight_cache_path` has no `use_pretrained` gate, so a random-weight test on
  an unstaged box starts a 181 GB HF download; and several CI `-k` selectors over-select by
  substring (`mesh-8x4` matches `fabric2d-mesh-8x4`, `balanced` matches `non_balanced`,
  `iter2` matches `iter25`).
