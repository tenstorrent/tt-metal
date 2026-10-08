# r01-b02-a02: ok, score 1.0338

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 16.404 | 17.225 | 16.992 | 1.0358 | 0.9999985 | 0.0213 | PASSED |
| deepseek-v4-flash-h4096 | 17.887 | 18.750 | 18.256 | 1.0206 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 22.563 | 23.329 | 23.478 | 1.0406 | 0.9999985 | 0.0223 | PASSED |
| kimi-k2-7-h7168 | 25.062 | 25.714 | 26.026 | 1.0385 | 0.9999985 | 0.0221 | PASSED |

Noise band: ±1.0% (from baseline.json).
