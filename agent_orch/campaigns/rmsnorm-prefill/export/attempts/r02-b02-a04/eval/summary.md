# r02-b02-a04: ok, score 1.2346

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 14.014 | 14.751 | 16.992 | 1.2125 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 15.116 | 15.726 | 18.256 | 1.2077 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 18.866 | 19.685 | 23.478 | 1.2445 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 20.412 | 20.790 | 26.026 | 1.2750 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
