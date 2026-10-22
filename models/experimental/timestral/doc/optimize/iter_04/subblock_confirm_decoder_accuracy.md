# Subblock sweep (accuracy, rows 2048, 7 x 10 calls per candidate)

## wo_residual

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[16,8,4] sb 1x4` | 0.054 | -15.2 % | 0.99998 |
| `[16,8,4] sb 4x1` | 0.056 | -11.6 % | 0.99998 |
| `[8,8,8] sb 1x8 (shipped)` | 0.063 | +0.0 % | 0.99998 |

## w_down_residual

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[8,8,8] sb 1x8 (shipped)` | 0.075 | +0.0 % | 0.99998 |
| `[16,8,4] sb 1x4` | 0.078 | +4.4 % | 0.99998 |
| `[16,8,4] sb 4x1` | 0.085 | +14.4 % | 0.99998 |

## wv

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[8,8,8] sb 1x8 (shipped)` | 0.054 | +0.0 % | 0.99996 |
| `[16,8,4] sb 4x1` | 0.056 | +2.8 % | 0.99996 |
| `[16,8,4] sb 1x4` | 0.057 | +5.1 % | 0.99996 |
