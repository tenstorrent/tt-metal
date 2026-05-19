# TT-DiT Experimental Models

In-progress model variants built on top of the Wan2.2 I2V pipeline.

## Models

- **[AniSora V3.2](models/AniSora.md)** — Anime-domain I2V fine-tune of Wan2.2-I2V-A14B
- **[Wan2.2-Distill](models/Wan2_2_Distill.md)** — 4-step distilled I2V via lightx2v (7× speedup)

LoRA adapter support is now first-class — see
[../pipelines/wan/Wan2_2_LoRA.md](../pipelines/wan/Wan2_2_LoRA.md).

## Directory Structure

```
experimental/
├── models/              # Model documentation
│   ├── AniSora.md
│   └── Wan2_2_Distill.md
├── pipelines/           # Pipeline implementations
│   ├── pipeline_anisora.py
│   └── pipeline_wan_distill.py
├── utils/               # Shared utilities
│   └── lightx2v_loader.py
└── tests/               # End-to-end tests
    ├── test_pipeline_anisora.py
    └── test_pipeline_wan_distill_i2v.py
```

All pipelines extend `WanPipelineI2V` from the main `tt_dit` tree and reuse the
existing transformer, VAE, text encoder, and parallel infrastructure.
