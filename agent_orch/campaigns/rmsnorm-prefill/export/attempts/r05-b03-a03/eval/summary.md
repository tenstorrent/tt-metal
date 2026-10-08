# r05-b03-a03: ok, score 1.4881

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.277 | 13.900 | 16.992 | 1.3841 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 12.622 | 14.385 | 18.256 | 1.4464 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 15.122 | 15.972 | 23.478 | 1.5526 | 0.9999985 | 0.0231 | PASSED |
| kimi-k2-7-h7168 | 16.498 | 17.397 | 26.026 | 1.5775 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
