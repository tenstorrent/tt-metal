# r02-b02-a02: ok, score 1.2292

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 13.877 | 14.381 | 16.992 | 1.2245 | 0.9999985 | 0.0239 | PASSED |
| deepseek-v4-flash-h4096 | 15.093 | 15.439 | 18.256 | 1.2096 | 0.9999985 | 0.0251 | PASSED |
| glm-5-3-h6144 | 19.085 | 19.442 | 23.478 | 1.2302 | 0.9999985 | 0.0251 | PASSED |
| kimi-k2-7-h7168 | 20.772 | 21.056 | 26.026 | 1.2529 | 0.9999985 | 0.0242 | PASSED |

Noise band: ±1.0% (from baseline.json).
