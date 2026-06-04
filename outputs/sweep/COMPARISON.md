# Wan2.1 image-gen perf sweep — config comparison

All runs: traced, 20 inference steps, num_frames=1, on WH galaxy 4x8.
Total seconds = encoder + denoising + VAE (full pipeline, e2e).

Configs:
- **plain 4x4** (`wh_4x4_sp1tp0`): single 4x4 submesh, cond+uncond run sequentially. Folder `4x4_sp4tp4/`.
- **CFG 4x4** (`cfg2_4x4_sp1tp0`): two 4x4 submeshes carved from the 4x8 parent, cond/uncond concurrent. Folder `cfg2_4x4_sp4tp4/`.
- **CFG 2x4** (`cfg2_2x4_sp0tp1`): two 2x4 submeshes, cond/uncond concurrent (only 8 devices/submesh). Folder `cfg2_2x4_sp2tp4/`.

## Total e2e seconds (sorted by token count, desc)

| Resolution   | Tokens | plain 4x4 | CFG 4x4 | CFG 2x4 |
|--------------|-------:|----------:|--------:|--------:|
| 1x2048x2048  | 16384  | 46.42     | 24.09   | OOM     |
| 1x1536x2048  | 12288  | 36.63     | 19.03   | OOM     |
| 1x2048x1536  | 12288  | 36.60     | 18.99   | OOM     |
| 1x1344x2048  | 10752  | 34.95     | 18.14   | 27.22   |
| 1x2048x1344  | 10752  | 34.94     | 18.17   | 27.15   |
| 1x1344x2016  | 10584  | 34.77     | 18.12   | 27.02   |
| 1x2016x1344  | 10584  | 34.74     | 18.04   | 26.92   |
| 1x1152x2048  | 9216   | 29.40     | 15.30   | 23.28   |
| 1x2048x1152  | 9216   | 29.49     | 15.30   | 23.25   |
| 1x1536x1536  | 9216   | 29.53     | 15.28   | 23.35   |
| 1x2048x880   | 7040   | 24.78     | 12.99   | 18.25   |
| 1x1088x1632  | 6936   | 24.74     | 12.93   | 18.21   |
| 1x1632x1088  | 6936   | 24.76     | 12.89   | 18.23   |
| 1x1152x1536  | 6912   | 23.60     | 12.30   | 17.85   |
| 1x1536x1152  | 6912   | 23.63     | 12.34   | 17.84   |
| 1x960x1696   | 6360   | 23.02     | 12.09   | 17.20   |
| 1x1696x960   | 6360   | 22.93     | 12.14   | 17.14   |
| 1x1024x1280  | 5120   | 20.17     | 10.57   | 15.19   |
| 1x1680x720   | 4725   | 19.92     | 10.41   | 14.77   |
| 1x1024x1024  | 4096   | 17.44     | 9.20    | 11.68   |
| 1x1344x768   | 4032   | 17.47     | 9.46    | 11.55   |
| 1x768x1344   | 4032   | 17.46     | 9.45    | 11.51   |
| 1x1152x896   | 4032   | 17.51     | 9.22    | 11.52   |
| 1x832x1216   | 3952   | 17.40     | 9.14    | 11.49   |
| 1x1280x720   | 3600   | 17.21     | 9.12    | 11.16   |
| 1x768x1024   | 3072   | 16.31     | 8.61    | 9.57    |
| 1x576x1024   | 2304   | 15.08     | 8.00    | 8.12    |
| 1x1080x1920  | 8100   | invalid (H,W must be divisible by 16) |||

## Takeaways

- **CFG 4x4 is the clear winner**: fastest at every resolution and the only config with no
  OOMs (covers the full set incl. 2048x2048). It roughly halves plain-4x4 e2e
  (e.g. 2048x2048 46.4s -> 24.1s; 1024x1024 17.4s -> 9.2s) — the cond/uncond concurrency
  across the two submeshes pays off directly.
- **CFG 2x4 DRAM ceiling ~= 10752 tokens**: the three >=12288-token shapes
  (2048x2048, 1536x2048, 2048x1536) OOM on the 8-device submesh. Everything <=10752 fits.
  Where it fits it still beats plain 4x4 (CFG concurrency outweighs the smaller submesh),
  but it's slower than CFG 4x4 (e.g. 1024x1024 11.7s vs 9.2s).
- Encoder is ~0.09s (CFG) / ~0.14s (plain); VAE is sub-0.6s everywhere. Per-step time
  scales ~linearly with token count in all configs.

## Artifacts
- `4x4_sp4tp4/`        — 27 PNGs + `wan2_1_4x4_sp4tp4_shape_sweep_perf.csv`
- `cfg2_4x4_sp4tp4/`   — 27 PNGs + `wan2_1_cfg2_4x4_sp4tp4_shape_sweep_perf.csv`
- `cfg2_2x4_sp2tp4/`   — 24 PNGs (4 missing: 3 OOM + 1 invalid) + `wan2_1_cfg2_2x4_sp2tp4_shape_sweep_perf.csv`
