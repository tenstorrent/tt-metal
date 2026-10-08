# r05-b02-a01: ok, score 1.4078

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.913 | 13.108 | 16.992 | 1.4263 | 0.9999985 | 0.0239 | PASSED |
| deepseek-v4-flash-h4096 | 13.384 | 14.275 | 18.256 | 1.3640 | 0.9999985 | 0.0251 | PASSED |
| glm-5-3-h6144 | 16.994 | 18.948 | 23.478 | 1.3815 | 0.9999985 | 0.0251 | PASSED |
| kimi-k2-7-h7168 | 17.806 | 18.320 | 26.026 | 1.4616 | 0.9999985 | 0.0242 | PASSED |

Noise band: ±1.0% (from baseline.json).
