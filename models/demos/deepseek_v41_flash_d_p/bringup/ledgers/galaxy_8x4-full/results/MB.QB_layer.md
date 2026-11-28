# MB.QB layer check: wq_b bank layout (WQB_BANK) and the single-request fusions (NATIVE_FUSIONS), 5120 chunk

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_mb_qb.py` (op level on the held mesh, no model load; fixed-seed values in the model's io shapes, through the model's own calls). on = the switch on (the shipped default), today = off (model_bringup before MB.QB).

| case | change | mismatch (on vs today) | repeat | cache growth (on) | host us today | host us on | device us today | device us on | programs a call | calls a chunk |
|---|---|---|---|---|---|---|---|---|---|---|
| wqb160 | wq_b: stock 1D matmul -> G = 1 bank grouped_matmul (WQB_BANK) | 0.0 (rel vs fp64 1.6796e-03 -> 1.6796e-03) | 0.0 | 0 | 86.2 | 51.8 | 177.7 | 151.71 | 1 -> 1 | 21 (layers 0-20) |
| emb5120 | ids reorder + embedding -> embedding (reorder as its reader; NATIVE_FUSIONS) | 0.0 | 0.0 | 0 | 162.1 | 133.3 | 42.4 | 34.43 | 3 -> 2 | 1 |
| rot_ckv5120 | replay rotation: 2 page copies -> 1 (NATIVE_FUSIONS) | 0.0 | 0.0 | 0 | 68.0 | 49.7 | 45.24 | 31.55 | 2 -> 1 | 2 a replay layer (ckv + index keys) |
| rot_idx5120 | replay rotation: 2 page copies -> 1 (NATIVE_FUSIONS) | 0.0 | 0.0 | 0 | 68.7 | 52.5 | 37.0 | 21.9 | 2 -> 1 | 2 a replay layer (ckv + index keys) |
| rot_ckv5120_r128 | replay rotation: 2 page copies -> 1 (NATIVE_FUSIONS) | 0.0 | 0.0 | 0 | 70.3 | 52.0 | 3.28 | 2.44 | 2 -> 1 | 2 a replay layer (ckv + index keys) |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes. Device us: real-time profiler, slowest chip, 20 calls a section. emb programs: the SP all-gather + (today) the IdsReorder kernel and the embedding, (on) the embedding with the reorder as its reader.
wqb160: rel vs fp64 of the device's bfp8 inputs (chip 0): today 1.679603e-03, bank 1.679603e-03; max |bank - today| 0.0 (bf16 values); every chip equal True; bank cores a chip 88 (<= the 110-core matmul cap, rule 18).
