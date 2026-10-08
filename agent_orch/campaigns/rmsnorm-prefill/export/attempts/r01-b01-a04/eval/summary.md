# r01-b01-a04: ok, score 1.2042

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 14.147 | 16.011 | 16.992 | 1.2011 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 15.348 | 16.122 | 18.256 | 1.1895 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 19.316 | 19.994 | 23.478 | 1.2155 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 21.494 | 22.054 | 26.026 | 1.2108 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
