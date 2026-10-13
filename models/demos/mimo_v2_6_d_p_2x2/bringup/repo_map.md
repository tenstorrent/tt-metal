# Repo map: mimo_v2_6_d_p_2x2

| Need | Where it is |
|---|---|
| Serving KV contract on a mesh with no size-1 axis and no SP (2x2, TP over the flattened mesh) | `models/demos/mimo_v2_6_d_p_2x2/tt/runners/kv_contract.py` (own full-length NdShard cache + table walk, one-chip device groups at `MeshCoordinate(d // 2, d % 2)`; reuses the 1x4 prior's `_slab` and read-back) + `tt/runners/adapter.py` (engine ids gathered over axis 0 on the device) |
