# r02-b03-a01: ok, score 1.0312

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 16.614 | 17.559 | 16.992 | 1.0228 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 17.684 | 18.399 | 18.256 | 1.0323 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 22.727 | 23.261 | 23.478 | 1.0330 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 25.100 | 25.590 | 26.026 | 1.0369 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
