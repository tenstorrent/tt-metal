# r02-b01-a01: ok, score 1.1058

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 15.527 | 16.401 | 16.992 | 1.0944 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 16.573 | 17.851 | 18.256 | 1.1016 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 21.134 | 21.786 | 23.478 | 1.1109 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 23.311 | 23.709 | 26.026 | 1.1165 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
