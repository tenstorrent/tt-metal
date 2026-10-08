# r01-b01-a02: ok, score 0.8994

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 18.437 | 19.267 | 16.992 | 0.9216 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 20.041 | 21.140 | 18.256 | 0.9109 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 26.442 | 27.077 | 23.478 | 0.8879 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 29.651 | 30.009 | 26.026 | 0.8777 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
