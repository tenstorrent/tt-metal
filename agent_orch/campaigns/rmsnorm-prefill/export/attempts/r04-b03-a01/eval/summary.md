# r04-b03-a01: ok, score 1.3543

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.328 | 13.094 | 16.992 | 1.3783 | 0.9999985 | 0.0204 | PASSED |
| deepseek-v4-flash-h4096 | 13.761 | 15.656 | 18.256 | 1.3266 | 0.9999985 | 0.0239 | PASSED |
| glm-5-3-h6144 | 17.535 | 18.263 | 23.478 | 1.3389 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 18.937 | 19.214 | 26.026 | 1.3743 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
