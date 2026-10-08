# r03-b02-a02: ok, score 1.3333

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.591 | 13.529 | 16.992 | 1.3495 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.051 | 14.622 | 18.256 | 1.2993 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.550 | 19.056 | 23.478 | 1.3378 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.320 | 19.725 | 26.026 | 1.3471 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
