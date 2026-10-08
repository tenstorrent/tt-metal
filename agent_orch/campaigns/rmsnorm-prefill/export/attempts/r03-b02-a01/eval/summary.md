# r03-b02-a01: ok, score 1.3231

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.725 | 13.577 | 16.992 | 1.3353 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.127 | 14.665 | 18.256 | 1.2923 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.722 | 18.167 | 23.478 | 1.3248 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.415 | 19.701 | 26.026 | 1.3405 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
