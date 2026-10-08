# r04-b04-a01: ok, score 1.3666

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.315 | 13.358 | 16.992 | 1.3798 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.641 | 14.226 | 18.256 | 1.3383 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.349 | 18.010 | 23.478 | 1.3533 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 18.645 | 19.057 | 26.026 | 1.3959 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
