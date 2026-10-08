# r01-b04-a04: ok, score 1.2132

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 14.173 | 16.023 | 16.992 | 1.1989 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 15.428 | 16.148 | 18.256 | 1.1833 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 19.079 | 19.914 | 23.478 | 1.2306 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 20.975 | 21.620 | 26.026 | 1.2408 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
