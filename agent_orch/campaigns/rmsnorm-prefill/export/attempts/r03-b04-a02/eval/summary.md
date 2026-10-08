# r03-b04-a02: ok, score 1.2440

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 13.759 | 14.138 | 16.992 | 1.2350 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 14.779 | 15.221 | 18.256 | 1.2353 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 18.780 | 19.685 | 23.478 | 1.2502 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 20.726 | 21.112 | 26.026 | 1.2557 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
