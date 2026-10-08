# r06-b03-a01: ok, score 1.5512

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.502 | 13.319 | 16.992 | 1.4773 | 0.9999985 | 0.0205 | PASSED |
| deepseek-v4-flash-h4096 | 12.336 | 13.781 | 18.256 | 1.4799 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 14.628 | 15.294 | 23.478 | 1.6050 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 15.771 | 16.782 | 26.026 | 1.6502 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
