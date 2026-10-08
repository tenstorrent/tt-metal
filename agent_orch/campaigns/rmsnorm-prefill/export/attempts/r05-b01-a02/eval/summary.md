# r05-b01-a02: ok, score 1.4320

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 13.022 | 14.256 | 16.992 | 1.3049 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.278 | 14.892 | 18.256 | 1.3749 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 15.300 | 16.240 | 23.478 | 1.5345 | 0.9999985 | 0.0231 | PASSED |
| kimi-k2-7-h7168 | 17.040 | 17.610 | 26.026 | 1.5273 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
