# TT-DiT Experimental Models

In-progress model variants built on top of the Wan2.2 I2V pipeline.

## Models

- **[AniSora V3.2](models/AniSora.md)** — Anime-domain I2V fine-tune of Wan2.2-I2V-A14B
- **[Wan2.2-Distill](models/Wan2_2_Distill.md)** — 4-step distilled I2V via lightx2v (7× speedup)
- **[Wan2.2 LoRA](models/Wan2_2_LoRA.md)** — LoRA adapter pipeline (camera control, style transfer, multi-LoRA stacking)
- **[Wan2.2 SVI 2.0 Pro](models/Wan2_2_SVI.md)** — Stable-Video-Infinity long-video generation by autoregressive clip chaining

## Directory Structure

```
experimental/
├── models/              # Model documentation
│   ├── AniSora.md
│   ├── Wan2_2_Distill.md
│   ├── Wan2_2_LoRA.md
│   └── Wan2_2_SVI.md
├── pipelines/           # Pipeline implementations
│   ├── pipeline_anisora.py
│   ├── pipeline_wan_distill.py
│   ├── pipeline_wan_lora.py
│   └── pipeline_wan_svi.py
├── utils/               # Shared utilities
│   ├── lightx2v_loader.py
│   └── lora.py
└── tests/               # End-to-end tests
    ├── test_pipeline_anisora.py
    ├── test_pipeline_lora.py
    ├── test_pipeline_wan_distill_i2v.py
    └── test_pipeline_wan_svi.py
```

All pipelines extend `WanPipelineI2V` from the main `tt_dit` tree and reuse the
existing transformer, VAE, text encoder, and parallel infrastructure.
