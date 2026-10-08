# r02-b04-a01: ok, score 0.8470

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 19.856 | 20.744 | 16.992 | 0.8558 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 21.333 | 22.277 | 18.256 | 0.8558 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 27.956 | 28.794 | 23.478 | 0.8398 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 31.098 | 31.793 | 26.026 | 0.8369 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
