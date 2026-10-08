# r01-b04-a03: ok, score 1.2075

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 14.462 | 15.157 | 16.992 | 1.1749 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 15.267 | 17.060 | 18.256 | 1.1958 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 19.145 | 19.697 | 23.478 | 1.2263 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 21.095 | 21.817 | 26.026 | 1.2338 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
