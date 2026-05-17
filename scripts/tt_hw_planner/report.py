# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Output formatters.

Three backends, all built from the same data:

    table     — terminal-rendered, dense, the default
    json      — for CI / dashboards / scripting
    markdown  — for PR comments and design-doc inclusion
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from .hardware import Overhead
from .probe import ModelProbe
from .verdict import FitRow, FitVerdict, Tightness


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fmt_gb(x: float, prec: int = 2) -> str:
    return f"{x:.{prec}f}"


def _confidence_for(probe: ModelProbe) -> str:
    if probe.category in ("Image", "Video", "TTS"):
        return "LOW"
    if probe.category == "Unknown":
        return "LOW"
    if probe.config_status == "failed" and probe.category in ("LLM", "VLM", "STT", "Embed"):
        return "MEDIUM"
    if probe.config_status is False and probe.category in ("LLM", "VLM", "STT", "Embed"):
        return "MEDIUM"
    return "HIGH"


def _category_guidance(category: str) -> str:
    return {
        "LLM": "Standard transformer LLM. Closest templates: models/tt_transformers/, "
        "models/demos/llama3_subdevices/, models/demos/qwen3_tts/.",
        "VLM": "Vision-language model. Closest templates: models/demos/qwen3_vl/, "
        "models/demos/qwen25_vl/, models/demos/multimodal/gemma3/.",
        "STT": "Speech-to-text. Closest template: models/demos/audio/whisper/.",
        "TTS": "Text-to-speech. Closest template: models/demos/qwen3_tts/.",
        "Embed": "Embedding model. Closest templates: models/demos/bge_large_en/, "
        "models/demos/sentence_bert/, models/demos/squeezebert/.",
        "CNN": "Convolutional / vision model. Closest templates: models/demos/vision/.",
        "Image": "Image generation (diffusion). Closest template: "
        "models/demos/stable_diffusion_xl_base/. Memory estimate rough.",
        "Video": "Video generation. No mature TT template; rough estimate only.",
        "Unknown": "Pipeline tag not recognized; verdict is best-effort.",
    }.get(category, "")


# ---------------------------------------------------------------------------
# Table backend
# ---------------------------------------------------------------------------


def _format_overhead(o: Overhead, hbm_per_chip: float) -> str:
    return (
        f"dispatch={o.dispatch_gb:.1f} GB + ccl={o.ccl_gb:.1f} GB + "
        f"frag={o.fragmentation_frac*100:.0f}% of {hbm_per_chip:.0f} GB"
    )


def render_table(
    probe: ModelProbe,
    verdict: FitVerdict,
    batch: int,
    seq: int,
    kv_dtype: str,
    dtypes: list,
    show_overhead: bool = True,
) -> str:
    out = []
    p = lambda s="": out.append(s)
    rule = "=" * 78
    sub = "-" * 78

    p(rule)
    p(f"  Model:        {probe.model_id}")
    p(f"  Category:     {probe.category:<7} (pipeline_tag={probe.pipeline_tag}, " f"library={probe.library})")
    if probe.arch_family:
        p(f"  Architecture: {probe.arch_family}")
    p(rule)
    p()

    # On-disk footprint
    disk_gb = probe.weight_bytes_total / 1e9
    p(f"  On-disk weights:  {disk_gb:.2f} GB  (saved as {probe.saved_dtype_pretty})")
    if probe.total_params:
        suffix = "exact" if probe.weight_bytes_safetensors > 0 else "inferred from file size"
        p(f"  Total parameters: {probe.total_params / 1e9:.2f} B ({suffix})")

    # Transformer fields
    if probe.arch_spec:
        a = probe.arch_spec
        p(
            f"  Transformer:      {a.num_layers} layers, hidden={a.hidden_size}, "
            f"Q/KV heads={a.num_attention_heads}/{a.num_key_value_heads}, "
            f"head_dim={a.head_dim}"
        )
        if a.max_position_embeddings:
            p(f"  Native context:   {a.max_position_embeddings:,}")
        if a.kv_lora_rank:
            p(f"  MLA:              kv_lora_rank={a.kv_lora_rank}, " f"qk_rope_head_dim={a.qk_rope_head_dim}")
        if a.num_experts:
            p(f"  MoE:              {a.num_experts} experts, top-{a.experts_per_token} routing")

    if probe.flags:
        p()
        p("  Flags:")
        for f in probe.flags:
            p(f"    * {f}")

    p()
    p(f"  Probe knobs:  batch={batch}  seq/horizon={seq}  KV dtype={kv_dtype}")

    # Memory budget (untouched by parallelism — model-level)
    if probe.memory_model:
        m = probe.memory_model
        w = m.weights_bytes(dtypes[0]) / 1e9
        kv_bpe = 2.0 if kv_dtype == "bf16" else 1.0
        k = m.kv_cache_bytes(batch, seq, kv_bpe) / 1e9
        a = m.activation_bytes(batch, seq) / 1e9
        p()
        p(f"  Memory budget @ {dtypes[0]} (model-level, before sharding):")
        p(f"    weights                       {w:>8.2f} GB")
        p(f"    KV cache                      {k:>8.2f} GB")
        p(f"    activations                   {a:>8.2f} GB")
        p(f"    TOTAL                         {w+k+a:>8.2f} GB")

    # Overhead breakdown (replaces the flat-0.80 from the old script).
    if show_overhead:
        p()
        p("  Per-chip overhead (subtracted from HBM before fit-check):")
        seen = set()
        for r in verdict.rows:
            key = r.box.arch
            if key in seen:
                continue
            seen.add(key)
            o = r.box.overhead
            p(f"    {key:<10} {_format_overhead(o, r.box.hbm_per_chip_gb)}")
            p(f"               source: {o.source}")

    # Detect if any row needs PP info; that widens the parallel column.
    show_parallel = any(r.parallel.pp != 1 or r.parallel.ep != 1 for r in verdict.rows)

    p()
    p("  " + "-" * (90 if show_parallel else 76))
    if show_parallel:
        hdr = (
            f"  {'BOX':<8}  {'DTYPE':<8}  {'MESH':<8}  {'PARALLEL':<15}  "
            f"{'PER-CHIP':>9}  {'USABLE':>9}  {'HEADROOM':>10}  VERDICT"
        )
    else:
        hdr = f"  {'BOX':<8}  {'DTYPE':<8}  {'MESH':<8}  " f"{'PER-CHIP':>9}  {'USABLE':>9}  {'HEADROOM':>10}  VERDICT"
    p(hdr)
    p("  " + "-" * (90 if show_parallel else 76))
    last_box: Optional[str] = None
    for r in verdict.rows:
        if last_box is not None and last_box != r.box.name:
            p("  " + " " * 6 + "-" * (84 if show_parallel else 70))
        last_box = r.box.name
        mesh = f"[{r.mesh_shape[0]},{r.mesh_shape[1]}]"
        prefix = f"  {r.box.name:<8}  {r.dtype:<8}  {mesh:<8}"
        if show_parallel:
            prefix += f"  {r.parallel.label:<15}"
        p(
            f"{prefix}  "
            f"{_fmt_gb(r.per_chip_gb):>7}G  "
            f"{_fmt_gb(r.usable_per_chip_gb):>7}G  "
            f"{_fmt_gb(r.headroom_gb):>8}G  "
            f"{r.tightness.value}"
        )
    p("  " + "-" * (90 if show_parallel else 76))

    # Recommendation
    p()
    if verdict.best:
        r = verdict.best
        mesh = f"[{r.mesh_shape[0]},{r.mesh_shape[1]}]"
        parallel_note = f" ({r.parallel.label})" if r.parallel.pp != 1 or r.parallel.ep != 1 else ""
        p(f"  RECOMMENDATION: {r.box.name} mesh {mesh}{parallel_note} with {r.dtype} weights")
        p(f"                  per-chip={r.per_chip_gb:.1f} GB, headroom={r.headroom_gb:.1f} GB")
        p(f"                  verdict: {r.tightness.value}")
        p(f"                  {r.box.notes}")
        if r.tightness == Tightness.TIGHT:
            p()
            p("  WARNING: this is a tight fit (< 10% headroom). Run a hardware")
            p("           smoke test before committing — the overhead model may")
            p("           be off by 1-2 GB/chip.")
    else:
        p("  RECOMMENDATION: nothing fits.")
        if probe.weight_bytes_total / 1e9 > 384:
            p("                  Even Galaxy (384 GB) is not enough; this model")
            p("                  requires multi-host deployment or aggressive quantization.")

    p()
    p(f"  Category guidance ({probe.category}):")
    p(f"    {_category_guidance(probe.category)}")
    p()
    confidence = _confidence_for(probe)
    p(f"  CONFIDENCE: {confidence}")
    if confidence == "LOW":
        p("              Category-level estimates only. Smoke-test before deciding.")
    elif confidence == "MEDIUM" and probe.config_status == "failed":
        p("              Could not load config.json; KV cache not included.")
    elif confidence == "MEDIUM" and probe.config_status is False:
        p("              Config loaded but transformer fields not at standard paths.")
    elif probe.memory_model is None:
        p("              Weights-only estimate (no transformer config); no KV math applied.")
    else:
        p("              Exact weight footprint + architecture-aware KV math.")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# JSON backend
# ---------------------------------------------------------------------------


def render_json(probe: ModelProbe, verdict: FitVerdict, batch: int, seq: int, kv_dtype: str, dtypes: list) -> str:
    payload = {
        "schema_version": "1.0",
        "model": {
            "id": probe.model_id,
            "category": probe.category,
            "pipeline_tag": probe.pipeline_tag,
            "library": probe.library,
            "weight_bytes_total": probe.weight_bytes_total,
            "weight_bytes_safetensors": probe.weight_bytes_safetensors,
            "weight_bytes_legacy": probe.weight_bytes_legacy,
            "saved_dtype": probe.saved_dtype,
            "saved_dtype_pretty": probe.saved_dtype_pretty,
            "total_params": probe.total_params,
            "bytes_per_param_on_disk": probe.bytes_per_param_on_disk,
            "architecture": probe.arch_family,
            "transformer_config": (asdict(probe.arch_spec) if probe.arch_spec else None),
            "flags": probe.flags,
            "confidence": _confidence_for(probe),
        },
        "probe_knobs": {
            "batch": batch,
            "seq": seq,
            "kv_dtype": kv_dtype,
            "dtypes_tested": dtypes,
        },
        "rows": [
            {
                "box": r.box.name,
                "arch": r.box.arch,
                "dtype": r.dtype,
                "mesh_shape": list(r.mesh_shape),
                "parallel": {
                    "tp": r.parallel.tp,
                    "pp": r.parallel.pp,
                    "ep": r.parallel.ep,
                    "dp": r.parallel.dp,
                },
                "per_chip_gb": round(r.per_chip_gb, 3),
                "usable_per_chip_gb": round(r.usable_per_chip_gb, 3),
                "headroom_gb": round(r.headroom_gb, 3),
                "verdict": r.tightness.name,
                "fits": r.fits,
                "components": {
                    "weights_gb": round(r.sharded.weights_bytes / 1e9, 3),
                    "kv_cache_gb": round(r.sharded.kv_cache_bytes / 1e9, 3),
                    "activations_gb": round(r.sharded.activation_bytes / 1e9, 3),
                },
            }
            for r in verdict.rows
        ],
        "recommendation": (
            None
            if verdict.best is None
            else {
                "box": verdict.best.box.name,
                "dtype": verdict.best.dtype,
                "mesh_shape": list(verdict.best.mesh_shape),
                "per_chip_gb": round(verdict.best.per_chip_gb, 3),
                "headroom_gb": round(verdict.best.headroom_gb, 3),
                "verdict": verdict.best.tightness.name,
            }
        ),
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Markdown backend
# ---------------------------------------------------------------------------


def render_markdown(probe: ModelProbe, verdict: FitVerdict, batch: int, seq: int, kv_dtype: str, dtypes: list) -> str:
    out = []
    p = out.append

    p(f"## `tt_hw_planner` — {probe.model_id}")
    p("")
    p(f"- **Category**: {probe.category} ({probe.pipeline_tag})")
    if probe.arch_family:
        p(f"- **Architecture**: {probe.arch_family}")
    p(f"- **On-disk**: {probe.weight_bytes_total/1e9:.1f} GB  " f"(saved as `{probe.saved_dtype_pretty}`)")
    if probe.total_params:
        p(f"- **Parameters**: {probe.total_params/1e9:.2f} B")
    if probe.arch_spec:
        a = probe.arch_spec
        p(
            f"- **Transformer**: {a.num_layers} layers, hidden={a.hidden_size}, "
            f"Q/KV heads={a.num_attention_heads}/{a.num_key_value_heads}"
        )
    if probe.flags:
        p("- **Flags**:")
        for f in probe.flags:
            p(f"  - {f}")
    p(f"- **Probe knobs**: batch={batch}, seq={seq}, kv_dtype={kv_dtype}")
    p("")
    p("### Per-box fit table")
    p("")
    p("| Box | DTYPE | Mesh | Per-chip | Usable | Headroom | Verdict |")
    p("|---|---|---|---|---|---|---|")
    for r in verdict.rows:
        mesh = f"[{r.mesh_shape[0]},{r.mesh_shape[1]}]"
        p(
            f"| {r.box.name} | {r.dtype} | {mesh} | "
            f"{r.per_chip_gb:.1f} GB | {r.usable_per_chip_gb:.1f} GB | "
            f"{r.headroom_gb:+.1f} GB | {r.tightness.value} |"
        )
    p("")
    if verdict.best:
        r = verdict.best
        mesh = f"[{r.mesh_shape[0]},{r.mesh_shape[1]}]"
        p(
            f"**Recommendation**: `{r.box.name}` mesh `{mesh}` with `{r.dtype}` "
            f"weights — {r.tightness.value}, headroom {r.headroom_gb:.1f} GB."
        )
    else:
        p("**Recommendation**: nothing fits.")
    p("")
    p(f"**Confidence**: {_confidence_for(probe)}")

    return "\n".join(out)
