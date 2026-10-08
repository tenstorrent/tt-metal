# r04-b01-a02: ok, score 1.3911

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.040 | 13.277 | 16.992 | 1.4113 | 0.9999985 | 0.0224 | PASSED |
| deepseek-v4-flash-h4096 | 13.510 | 14.088 | 18.256 | 1.3513 | 0.9999985 | 0.0220 | PASSED |
| glm-5-3-h6144 | 17.134 | 18.077 | 23.478 | 1.3703 | 0.9999985 | 0.0235 | PASSED |
| kimi-k2-7-h7168 | 18.165 | 18.883 | 26.026 | 1.4328 | 0.9999985 | 0.0239 | PASSED |

Noise band: ±1.0% (from baseline.json).
