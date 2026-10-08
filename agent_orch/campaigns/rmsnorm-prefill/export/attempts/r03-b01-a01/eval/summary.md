# r03-b01-a01: ok, score 1.3129

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.973 | 14.942 | 16.992 | 1.3098 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.238 | 14.907 | 18.256 | 1.2822 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.730 | 18.361 | 23.478 | 1.3242 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.479 | 19.912 | 26.026 | 1.3361 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
