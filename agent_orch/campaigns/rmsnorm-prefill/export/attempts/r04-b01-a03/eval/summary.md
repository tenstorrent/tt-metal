# r04-b01-a03: ok, score 1.4475

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.668 | 12.886 | 16.992 | 1.4563 | 0.9999985 | 0.0224 | PASSED |
| deepseek-v4-flash-h4096 | 13.048 | 13.835 | 18.256 | 1.3991 | 0.9999985 | 0.0220 | PASSED |
| glm-5-3-h6144 | 16.345 | 16.973 | 23.478 | 1.4364 | 0.9999985 | 0.0235 | PASSED |
| kimi-k2-7-h7168 | 17.348 | 17.834 | 26.026 | 1.5002 | 0.9999985 | 0.0239 | PASSED |

Noise band: ±1.0% (from baseline.json).
