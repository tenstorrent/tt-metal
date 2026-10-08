# r05-b04-a01: ok, score 1.1939

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | pytest |
|---|---|---|---|---|---|---|---|
| kimi-k3-latent-moe-h3584 | 15.721 | 15.915 | 16.992 | 1.0808 | 0.9999985 | 0.0205 | PASSED |
| deepseek-v4-flash-h4096 | 16.659 | 16.883 | 18.256 | 1.0959 | 0.9999985 | 0.0237 | PASSED |
| glm-5-3-h6144 | 17.959 | 18.283 | 23.478 | 1.3073 | 0.9999985 | 0.0240 | PASSED |
| kimi-k2-7-h7168 | 19.832 | 20.199 | 26.026 | 1.3123 | 0.9999985 | 0.0231 | PASSED |

Noise band: ±1.0% (from baseline.json).
