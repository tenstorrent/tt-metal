# r01-b02-a03: ok, score 1.1721

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 14.560 | 14.999 | 16.992 | 1.1670 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 15.751 | 16.177 | 18.256 | 1.1590 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 19.726 | 20.113 | 23.478 | 1.1902 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 22.199 | 22.551 | 26.026 | 1.1724 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
