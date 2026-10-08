# r03-b04-a03: ok, score 1.3318

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.766 | 13.569 | 16.992 | 1.3310 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.201 | 14.671 | 18.256 | 1.2855 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.751 | 18.368 | 23.478 | 1.3226 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 18.720 | 19.520 | 26.026 | 1.3903 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
