# r01-b03-a01: ok, score 0.9867

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 18.684 | 23.668 | 16.992 | 0.9094 | 0.9999985 | 0.0213 | PASSED |
| deepseek-v4-flash-h4096 | 19.103 | 24.156 | 18.256 | 0.9557 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 21.821 | 22.687 | 23.478 | 1.0759 | 0.9999985 | 0.0223 | PASSED |
| kimi-k2-7-h7168 | 25.676 | 31.028 | 26.026 | 1.0136 | 0.9999985 | 0.0217 | PASSED |

Noise band: ±1.0% (from baseline.json).
