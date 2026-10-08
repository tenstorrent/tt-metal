# r04-b04-a03: ok, score 1.3949

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.066 | 13.080 | 16.992 | 1.4083 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.455 | 15.351 | 18.256 | 1.3568 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.058 | 18.828 | 23.478 | 1.3764 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 18.081 | 18.541 | 26.026 | 1.4394 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
