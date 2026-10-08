# r03-b03-a02: ok, score 1.3120

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.900 | 13.704 | 16.992 | 1.3172 | 0.9999985 | 0.0223 | PASSED |
| deepseek-v4-flash-h4096 | 14.256 | 14.908 | 18.256 | 1.2806 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 17.826 | 18.275 | 23.478 | 1.3171 | 0.9999985 | 0.0245 | PASSED |
| kimi-k2-7-h7168 | 19.514 | 19.781 | 26.026 | 1.3337 | 0.9999985 | 0.0242 | PASSED |

Noise band: ±1.0% (from baseline.json).
