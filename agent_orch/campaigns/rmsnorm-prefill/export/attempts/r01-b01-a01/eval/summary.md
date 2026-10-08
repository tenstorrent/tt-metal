# r01-b01-a01: ok, score 1.1089

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 15.130 | 15.951 | 16.992 | 1.1231 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 16.127 | 16.765 | 18.256 | 1.1320 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 21.252 | 21.605 | 23.478 | 1.1047 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 24.170 | 24.457 | 26.026 | 1.0768 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
