# r06-b02-a01: ok, score 1.5419

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.429 | 13.249 | 16.992 | 1.4867 | 0.9999985 | 0.0205 | PASSED |
| deepseek-v4-flash-h4096 | 12.386 | 14.009 | 18.256 | 1.4739 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 14.997 | 16.272 | 23.478 | 1.5655 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 15.793 | 16.409 | 26.026 | 1.6479 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
