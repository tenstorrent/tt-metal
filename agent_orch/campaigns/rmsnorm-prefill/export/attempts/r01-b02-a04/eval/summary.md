# r01-b02-a04: ok, score 0.9853

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 15.086 | 16.284 | 16.992 | 1.1263 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 19.073 | 19.923 | 18.256 | 0.9572 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 24.941 | 25.386 | 23.478 | 0.9413 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 28.025 | 28.339 | 26.026 | 0.9287 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
