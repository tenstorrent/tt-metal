# r05-b01-a01: ok, score 1.5696

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 11.191 | 13.064 | 16.992 | 1.5184 | 0.9999985 | 0.0205 | PASSED |
| deepseek-v4-flash-h4096 | 12.077 | 13.599 | 18.256 | 1.5116 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 14.585 | 15.805 | 23.478 | 1.6097 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 15.843 | 16.673 | 26.026 | 1.6427 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
