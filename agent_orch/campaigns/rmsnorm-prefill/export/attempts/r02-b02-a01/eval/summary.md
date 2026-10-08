# r02-b02-a01: ok, score 1.2385

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 13.870 | 15.943 | 16.992 | 1.2251 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 15.196 | 15.822 | 18.256 | 1.2014 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 18.725 | 19.197 | 23.478 | 1.2538 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 20.416 | 20.752 | 26.026 | 1.2748 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
