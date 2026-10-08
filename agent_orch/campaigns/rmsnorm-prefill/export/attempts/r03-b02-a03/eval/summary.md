# r03-b02-a03: ok, score 1.3000

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.853 | 13.532 | 16.992 | 1.3220 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.382 | 14.896 | 18.256 | 1.2694 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 18.198 | 18.766 | 23.478 | 1.2901 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.726 | 20.114 | 26.026 | 1.3194 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
