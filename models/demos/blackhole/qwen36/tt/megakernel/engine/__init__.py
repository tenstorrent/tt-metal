"""Megakernel engine package (spec v0.9.3). build.py (integration track) assembles the per-track pieces:
layout_mlp.py (v12), layout_attn.py (attention RQKV / wo / KV), ar_phase.py (AR)."""
from models.demos.blackhole.qwen36.tt.megakernel.engine.build import EngineProgram, build_engine, weight_layouts  # noqa: F401
