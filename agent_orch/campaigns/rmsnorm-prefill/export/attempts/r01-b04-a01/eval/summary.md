# r01-b04-a01: ok, score 1.0761

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 15.233 | 16.258 | 16.992 | 1.1155 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 16.220 | 16.524 | 18.256 | 1.1255 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 22.325 | 22.574 | 23.478 | 1.0516 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 25.621 | 25.891 | 26.026 | 1.0158 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
