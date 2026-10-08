# r01-b03-a04: ok, score 1.1653

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 14.587 | 16.531 | 16.992 | 1.1649 | 0.9999985 | 0.0213 | PASSED |
| deepseek-v4-flash-h4096 | 16.059 | 16.666 | 18.256 | 1.1368 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 20.292 | 21.132 | 23.478 | 1.1570 | 0.9999985 | 0.0223 | PASSED |
| kimi-k2-7-h7168 | 21.627 | 22.337 | 26.026 | 1.2034 | 0.9999985 | 0.0217 | PASSED |

Noise band: ±1.0% (from baseline.json).
