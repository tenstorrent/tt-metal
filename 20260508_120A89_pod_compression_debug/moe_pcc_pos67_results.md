# MoE fused test — position 67 PCC sweep

Setup:
- Test: `test_moe_fused_with_reduce` (DeepSeek V3 B1 fused MoE + reduce-to-one on 4x2 mesh)
- Compression: `compressed_tp8=True` (TP8 BSPM)
- Weights: deterministic-random (`USE_RANDOM_WEIGHTS=True`, default)
- For each MoE layer (3-12), loaded ONLY the 8 experts that were actually selected at position 67 in the compressed pod run (see `accepted_experts.json`), remapped into gate slots 0-7 via `remap_experts_to_slots_0_7`.

| Layer | Stage | Selected experts (8) | Reduce-output PCC | Result |
|---|---|---|---|---|
| 3  | 4  | 101, 197, 215, 109, 119, 199, 126, 66 | 0.9914 | PASS |
| 4  | 5  | 2, 64, 77, 10, 14, 95, 7, 3 | 0.9918 | PASS |
| 5  | 6  | 16, 59, 42, 48, 78, 92, 134, 81 | 0.9915 | PASS |
| 6  | 7  | 9, 194, 3, 38, 151, 35, 140, 143 | 0.9917 | PASS |
| 7  | 8  | 151, 147, 30, 5, 26, 148, 64, 231 | 0.9916 | PASS |
| 8  | 9  | 12, 8, 26, 240, 243, 226, 245, 87 | 0.9917 | PASS |
| 9  | 10 | 61, 40, 11, 2, 204, 205, 221, 103 | 0.9916 | PASS |
| 10 | 11 | 177, 149, 135, 174, 126, 121, 31, 14 | 0.9917 | PASS |
| 11 | 12 | 5, 126, 0, 7, 77, 162, 72, 179 | 0.9918 | PASS |
| 12 | 13 | 25, 193, 195, 201, 153, 237, 228, 148 | 0.9917 | PASS |

All 10 layers pass the 0.99 threshold; PCCs cluster tightly in 0.9914-0.9918.

## How to reproduce

```bash
TT_METAL_CLEAR_L1=1 TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 \
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_moe_mlp.py::test_moe_fused_with_reduce \
  -k 'pos67-NOC_MODE.DM_DYNAMIC_NOC-True-fabric_2d' -v
```

To run against real DeepSeek weights instead of deterministic random:
```
USE_RANDOM_WEIGHTS=False DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized ...
```

## Data sources

- `accepted_experts.json` / `accepted_experts.txt` — selected experts per (layer, position) extracted from the compressed-pod kernel DPRINT log (`token position: N selected expert indices: ...`), de-duplicated by taking the LAST occurrence of each `(layer, pos)` so spec-decode re-corrections win over rejected attempts.
