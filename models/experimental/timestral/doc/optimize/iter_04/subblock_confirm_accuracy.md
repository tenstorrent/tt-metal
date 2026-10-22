# Subblock sweep (accuracy, rows 65536, 7 x 10 calls per candidate)

## wo_residual

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[16,8,4] sb 4x1` | 0.802 | -9.3 % | 0.99998 |
| `[16,8,4] sb 1x4` | 0.804 | -9.0 % | 0.99998 |
| `[8,8,8] sb 1x8 (shipped)` | 0.884 | +0.0 % | 0.99998 |

## w_down_residual

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[16,8,4] sb 4x1` | 1.349 | -1.8 % | 0.99998 |
| `[16,8,4] sb 1x4` | 1.353 | -1.6 % | 0.99998 |
| `[8,8,8] sb 1x8 (shipped)` | 1.374 | +0.0 % | 0.99998 |

## wv

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[16,8,4] sb 1x4` | 0.724 | -2.6 % | 0.99996 |
| `[16,8,4] sb 4x1` | 0.725 | -2.5 % | 0.99996 |
| `[8,8,8] sb 1x8 (shipped)` | 0.743 | +0.0 % | 0.99996 |

## patch_out

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[16,8,4] sb 4x1` | 1.647 | -1.7 % | 0.99996 |
| `[16,8,4] sb 1x4` | 1.649 | -1.6 % | 0.99996 |
| `[8,8,8] sb 1x8 (shipped)` | 1.676 | +0.0 % | 0.99996 |
