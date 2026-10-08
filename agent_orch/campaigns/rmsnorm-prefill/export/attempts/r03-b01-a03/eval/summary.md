# r03-b01-a03: ok, score 1.3255

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.538 | 13.413 | 16.992 | 1.3552 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.169 | 14.587 | 18.256 | 1.2884 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.765 | 18.174 | 23.478 | 1.3216 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.455 | 19.754 | 26.026 | 1.3378 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
