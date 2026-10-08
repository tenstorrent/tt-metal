# r05-b01-a03: ok, score 1.4661

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.143 | 14.260 | 16.992 | 1.3993 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 12.937 | 15.138 | 18.256 | 1.4111 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 15.124 | 16.345 | 23.478 | 1.5524 | 0.9999985 | 0.0231 | PASSED |
| kimi-k2-7-h7168 | 17.265 | 18.942 | 26.026 | 1.5074 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
