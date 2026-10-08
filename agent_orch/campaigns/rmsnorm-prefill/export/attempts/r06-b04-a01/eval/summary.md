# r06-b04-a01: ok, score 1.4264

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 12.465 | 13.673 | 16.992 | 1.3632 | 0.9999985 | 0.0205 | PASSED |
| deepseek-v4-flash-h4096 | 13.247 | 14.230 | 18.256 | 1.3781 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 15.950 | 16.669 | 23.478 | 1.4720 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 17.385 | 19.321 | 26.026 | 1.4970 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
