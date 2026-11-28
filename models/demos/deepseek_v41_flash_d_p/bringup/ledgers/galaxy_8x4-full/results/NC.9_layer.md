# NC.9 layer check: the remaining tiny-op fusions (NATIVE_FUSIONS on = fused, off = today's sequence)

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_fusions.py` (op level on the held mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). (2) the history tail's page copy + slice_write is not on the shipped path (BREADCRUMBS NC.9) and was not fused.

| case | fusion | mismatch | repeat | cache growth (fused) | host us today | host us fused | device us today | device us fused | programs a call (today -> fused) | calls a chunk |
|---|---|---|---|---|---|---|---|---|---|---|
| emb8192 | (1) ids reorder + embedding -> embedding (reorder as its reader) | 0.0 | 0.0 | 0 | 162.5 | 132.3 | 47.96 | 36.03 | 3 -> 2 | 1 / 0 |
| emb5120 | (1) ids reorder + embedding -> embedding (reorder as its reader) | 0.0 | 0.0 | 0 | 161.8 | 132.8 | 42.27 | 34.73 | 3 -> 2 | 1 / 0 |
| rot_ckv8192 | (3) replay rotation: 2 page copies -> 1 | 0.0 | 0.0 | 0 | 68.0 | 50.1 | 72.35 | 50.36 | 2 -> 1 | 2 a replay chunk (ckv + index keys, a layer) |
| rot_idx8192 | (3) replay rotation: 2 page copies -> 1 | 0.0 | 0.0 | 0 | 76.2 | 55.3 | 57.7 | 33.75 | 2 -> 1 | 2 a replay chunk (ckv + index keys, a layer) |
| rot_ckv5120 | (3) replay rotation: 2 page copies -> 1 | 0.0 | 0.0 | 0 | 67.4 | 50.4 | 45.23 | 31.56 | 2 -> 1 | 2 a replay chunk (ckv + index keys, a layer) |
| rot_ckv8192_r128 | (3) replay rotation: 2 page copies -> 1 | 0.0 | 0.0 | 0 | 69.1 | 51.1 | 3.87 | 2.53 | 2 -> 1 | 2 a replay chunk (ckv + index keys, a layer) |
| carry | (4) window carry: zeros + rows -> 1 | 0.0 | 0.0 | 0 | 68.3 | 50.8 | 6.04 | 4.13 | 2 -> 1 | a carry at 0 < start < 128, a window layer |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes. Device us: real-time profiler, slowest chip, 20 calls a section. emb programs: the SP all-gather + (today) the IdsReorder kernel and the embedding, (fused) the embedding with the reorder as its reader.
