# r01-b02-a01: ok, score 1.0025

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 16.708 | 18.132 | 16.992 | 1.0170 | 0.9999985 | 0.0213 | PASSED |
| deepseek-v4-flash-h4096 | 18.308 | 19.051 | 18.256 | 0.9972 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 23.504 | 24.191 | 23.478 | 0.9989 | 0.9999985 | 0.0223 | PASSED |
| kimi-k2-7-h7168 | 26.107 | 26.945 | 26.026 | 0.9969 | 0.9999985 | 0.0221 | PASSED |

Noise band: ±1.0% (from baseline.json).
