# r04-b02-a03: ok, score 1.4216

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.738 | 13.239 | 16.992 | 1.4476 | 0.9999985 | 0.0224 | PASSED |
| deepseek-v4-flash-h4096 | 13.349 | 14.464 | 18.256 | 1.3676 | 0.9999985 | 0.0220 | PASSED |
| glm-5-3-h6144 | 16.782 | 17.449 | 23.478 | 1.3990 | 0.9999985 | 0.0235 | PASSED |
| kimi-k2-7-h7168 | 17.647 | 18.153 | 26.026 | 1.4748 | 0.9999985 | 0.0239 | PASSED |

Noise band: ±1.0% (from baseline.json).
