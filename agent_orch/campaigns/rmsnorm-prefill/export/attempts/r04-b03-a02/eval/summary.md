# r04-b03-a02: ok, score 1.3906

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.249 | 14.201 | 16.992 | 1.3872 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.424 | 14.137 | 18.256 | 1.3600 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 16.984 | 17.482 | 23.478 | 1.3824 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 18.150 | 18.577 | 26.026 | 1.4339 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
