# r01-b01-a03: ok, score 1.1868

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 14.510 | 15.048 | 16.992 | 1.1711 | 0.9999985 | 0.0217 | PASSED |
| deepseek-v4-flash-h4096 | 15.522 | 16.274 | 18.256 | 1.1761 | 0.9999985 | 0.0243 | PASSED |
| glm-5-3-h6144 | 19.466 | 19.965 | 23.478 | 1.2061 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 21.796 | 22.343 | 26.026 | 1.1941 | 0.9999985 | 0.0241 | PASSED |

Noise band: ±1.0% (from baseline.json).
