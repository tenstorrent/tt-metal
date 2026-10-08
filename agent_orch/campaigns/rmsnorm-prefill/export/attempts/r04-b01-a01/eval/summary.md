# r04-b01-a01: ok, score 1.3436

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.432 | 13.372 | 16.992 | 1.3668 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.120 | 14.788 | 18.256 | 1.2929 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.731 | 18.331 | 23.478 | 1.3241 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 18.683 | 19.394 | 26.026 | 1.3930 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
