# r04-b04-a02: ok, score 1.3997

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.961 | 13.137 | 16.992 | 1.4206 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.615 | 14.300 | 18.256 | 1.3409 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 16.911 | 17.781 | 23.478 | 1.3883 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 17.931 | 19.177 | 26.026 | 1.4515 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
