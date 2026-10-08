# r04-b02-a01: ok, score 1.2988

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.848 | 14.065 | 16.992 | 1.3225 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.518 | 16.479 | 18.256 | 1.2575 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 18.433 | 19.683 | 23.478 | 1.2737 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.376 | 19.833 | 26.026 | 1.3432 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
