# r05-b03-a01: ok, score 1.5174

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.468 | 13.205 | 16.992 | 1.4817 | 0.9999985 | 0.0205 | PASSED |
| deepseek-v4-flash-h4096 | 12.762 | 14.150 | 18.256 | 1.4305 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 15.251 | 16.293 | 23.478 | 1.5394 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 16.017 | 17.228 | 26.026 | 1.6249 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
