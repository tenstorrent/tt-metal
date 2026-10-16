# r03-b01-a01: runtime_error

| shape | µs (chip mean) | µs (max over chips) | baseline µs | speedup | PCC | max_abs | mean_abs | pytest |
|---|---|---|---|---|---|---|---|---|
| deepseek-v3-h7168-i2048 | 1386.331 | 1386.331 | - | - | - | - | - | FAILED |
| kimi-k2-7-h7168-i2048 | 454.776 | 454.776 | - | - | - | - | - | FAILED |
| kimi-k3-latent-h3584-i3072 | 189.609 | 189.609 | - | - | 0.9804434 | 0.9934 | 0.146626 | PASSED |
| glm-5-3-h6144-i2048 | 326.175 | 326.175 | - | - | 0.9802855 | 1.6094 | 0.232011 | PASSED |
| mistral-small-4-h4096-i2048 | 365.095 | 365.095 | - | - | 0.9802601 | 1.0138 | 0.149542 | PASSED |
| deepseek-v4-flash-h4096-i2048 | 264.327 | 264.328 | - | - | 0.9804868 | 0.9978 | 0.151036 | PASSED |
| deepseek-v4-pro-h7168-i3072 | - | - | - | - | - | - | - | FAILED |
| minimax-m3-h6144-i3072 | - | - | - | - | 0.9824444 | 2.3511 | 0.346576 | PASSED |
| gpt-oss-120b-h2880-i2880 | - | - | - | - | 0.9835044 | 1.1097 | 0.168411 | PASSED |

Noise band: ±1.0% (from baseline.json).
