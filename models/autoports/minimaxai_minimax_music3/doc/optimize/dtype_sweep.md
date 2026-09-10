| run | LLM policy | DiT | frame-hidden PCC (min) | latent PCC w0 / w1 | log-mel RMS dB | wav PCC | AR frames/s (teacher-forced) | DiT chunk 0 s | total 10 s clip s | DRAM GB | bars |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_mlp-bfp4_kv-bf16_dit-bf16 | opt_mlp-bfp4_kv-bf16 | bf16 | 0.99062 (0.91407) | 0.99351 / 0.99567 | 1.182 | 0.99271 | 22.76 | 2.98 | 28.7 | 16.2 | pass |
| sweep_mlp-bfp4_kv-bf16_dit-bfp8 | opt_mlp-bfp4_kv-bf16 | bfp8 | 0.99062 (0.91407) | 0.99370 / 0.99528 | 1.183 | 0.99266 | 22.78 | 2.93 | 28.3 | 14.0 | pass |
| sweep_mlp-bfp4_kv-bfp8_dit-bf16 | opt_mlp-bfp4_kv-bfp8 | bf16 | 0.99050 (0.91800) | 0.99340 / 0.98609 | 1.209 | 0.99092 | 22.76 | 2.98 | 28.5 | 14.7 | pass |
| sweep_mlp-bfp4_kv-bfp8_dit-bfp8 | opt_mlp-bfp4_kv-bfp8 | bfp8 | 0.99050 (0.91800) | 0.99244 / 0.98555 | 1.234 | 0.99020 | 22.76 | 2.93 | 28.4 | 12.4 | pass |
| sweep_mlp-bfp8_kv-bf16_dit-bf16 | opt_mlp-bfp8_kv-bf16 | bf16 | 0.99914 (0.98436) | 0.99927 / 0.99898 | 0.974 | 0.99869 | 22.45 | 2.94 | 28.9 | 19.0 | pass (within bf16 control) |
| sweep_mlp-bfp8_kv-bf16_dit-bfp8 | opt_mlp-bfp8_kv-bf16 | bfp8 | 0.99914 (0.98436) | 0.99922 / 0.99889 | 0.975 | 0.99862 | 22.47 | 2.93 | 28.5 | 16.7 | pass (within bf16 control) |
| sweep_mlp-bfp8_kv-bfp8_dit-bf16 | opt_mlp-bfp8_kv-bfp8 | bf16 | 0.99903 (0.97895) | 0.99890 / 0.99883 | 0.991 | 0.99837 | 22.46 | 2.94 | 28.6 | 17.4 | pass (within bf16 control) |
| sweep_mlp-bfp8_kv-bfp8_dit-bfp8 | opt_mlp-bfp8_kv-bfp8 | bfp8 | 0.99903 (0.97895) | 0.99887 / 0.99878 | 0.982 | 0.99827 | 22.53 | 2.91 | 28.4 | 15.1 | pass (within bf16 control) |
| before | functional | bf16 | 0.99941 (0.99512) | 0.99951 / 0.99923 | 0.754 | 0.99875 | 14.29 | 3.50 | 35.7 | 21.7 | pass (within bf16 control) |
