# r03-b03-a01: ok, score 1.3175

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.864 | 13.628 | 16.992 | 1.3209 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.070 | 16.035 | 18.256 | 1.2975 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.859 | 18.549 | 23.478 | 1.3146 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.458 | 19.757 | 26.026 | 1.3375 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
