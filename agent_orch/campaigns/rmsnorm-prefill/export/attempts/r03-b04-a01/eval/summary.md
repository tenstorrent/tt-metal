# r03-b04-a01: ok, score 1.3131

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.895 | 13.665 | 16.992 | 1.3177 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.195 | 14.715 | 18.256 | 1.2861 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.832 | 18.324 | 23.478 | 1.3166 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.532 | 19.911 | 26.026 | 1.3325 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
