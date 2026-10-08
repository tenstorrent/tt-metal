# r03-b01-a02: ok, score 1.3220

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.664 | 13.516 | 16.992 | 1.3418 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.088 | 14.618 | 18.256 | 1.2959 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.923 | 18.567 | 23.478 | 1.3099 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.410 | 19.764 | 26.026 | 1.3409 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
