# r05-b03-a02: ok, score 1.4219

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 13.053 | 14.368 | 16.992 | 1.3018 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.592 | 15.652 | 18.256 | 1.3431 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 15.516 | 17.767 | 23.478 | 1.5131 | 0.9999985 | 0.0231 | PASSED |
| kimi-k2-7-h7168 | 16.843 | 18.178 | 26.026 | 1.5452 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
