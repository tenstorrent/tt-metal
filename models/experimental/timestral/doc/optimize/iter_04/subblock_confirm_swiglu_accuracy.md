# Subblock sweep (accuracy, rows 65536, 7 x 10 calls per candidate)

## swiglu

| candidate | ms | vs shipped | PCC |
|---|---|---|---|
| `[8,12,2] sb 2x2` | 2.511 | -1.6 % | 0.99988 |
| `[8,12,4] sb 1x4` | 2.538 | -0.5 % | 0.99988 |
| `[8,12,2] sb 4x2 (shipped)` | 2.552 | +0.0 % | 0.99988 |
