# r01-b04-a02: ok, score 0.8397

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 20.081 | 20.997 | 16.992 | 0.8462 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 21.647 | 22.767 | 18.256 | 0.8434 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 28.074 | 28.941 | 23.478 | 0.8363 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 31.247 | 31.674 | 26.026 | 0.8329 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
