# r04-b02-a02: ok, score 1.3175

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.692 | 14.681 | 16.992 | 1.3388 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.288 | 14.990 | 18.256 | 1.2777 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 18.017 | 18.575 | 23.478 | 1.3031 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.255 | 19.675 | 26.026 | 1.3516 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
