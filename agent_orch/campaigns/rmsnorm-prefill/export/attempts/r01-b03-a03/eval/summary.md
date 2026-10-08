# r01-b03-a03: ok, score 1.1313

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 15.285 | 15.982 | 16.992 | 1.1117 | 0.9999985 | 0.0213 | PASSED |
| deepseek-v4-flash-h4096 | 16.385 | 17.273 | 18.256 | 1.1142 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 20.438 | 21.230 | 23.478 | 1.1487 | 0.9999985 | 0.0223 | PASSED |
| kimi-k2-7-h7168 | 22.604 | 23.887 | 26.026 | 1.1514 | 0.9999985 | 0.0217 | PASSED |

Noise band: ±1.0% (from baseline.json).
