# r01-b03-a02: ok, score 1.0942

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 15.378 | 16.055 | 16.992 | 1.1050 | 0.9999985 | 0.0213 | PASSED |
| deepseek-v4-flash-h4096 | 18.051 | 18.676 | 18.256 | 1.0114 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 20.571 | 21.358 | 23.478 | 1.1413 | 0.9999985 | 0.0223 | PASSED |
| kimi-k2-7-h7168 | 23.154 | 23.887 | 26.026 | 1.1240 | 0.9999985 | 0.0217 | PASSED |

Noise band: ±1.0% (from baseline.json).
