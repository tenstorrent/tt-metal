# r03-b03-a03: ok, score 1.3023

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.704 | 13.566 | 16.992 | 1.3375 | 0.9999985 | 0.0223 | PASSED |
| deepseek-v4-flash-h4096 | 14.568 | 15.781 | 18.256 | 1.2532 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 18.250 | 19.640 | 23.478 | 1.2865 | 0.9999985 | 0.0245 | PASSED |
| kimi-k2-7-h7168 | 19.512 | 19.942 | 26.026 | 1.3338 | 0.9999985 | 0.0242 | PASSED |

Noise band: ±1.0% (from baseline.json).
