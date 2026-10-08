# r04-b03-a03: ok, score 1.3894

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.991 | 13.134 | 16.992 | 1.4171 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.306 | 14.105 | 18.256 | 1.3720 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.048 | 17.637 | 23.478 | 1.3772 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 18.702 | 19.190 | 26.026 | 1.3916 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
