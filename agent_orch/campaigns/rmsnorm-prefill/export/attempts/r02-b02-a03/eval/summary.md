# r02-b02-a03: ok, score 1.2396

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 13.953 | 14.691 | 16.992 | 1.2178 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 15.116 | 16.016 | 18.256 | 1.2077 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 18.672 | 19.169 | 23.478 | 1.2574 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 20.384 | 20.670 | 26.026 | 1.2768 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
