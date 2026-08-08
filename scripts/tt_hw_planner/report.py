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

from .compatibility import CompatReport, Status
from .hardware import Overhead
from .kernel_constraints import KernelReport, Severity
from .probe import ModelProbe
from .verdict import FitRow, FitVerdict, Tightness


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

    disk_gb = probe.weight_bytes_total / 1e9
    p(f"  On-disk weights:  {disk_gb:.2f} GB  (saved as {probe.saved_dtype_pretty})")
    if probe.total_params:
        suffix = "exact" if probe.weight_bytes_safetensors > 0 else "inferred from file size"
        p(f"  Total parameters: {probe.total_params / 1e9:.2f} B ({suffix})")

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

    show_parallel = any(r.parallel.pp != 1 or r.parallel.ep != 1 for r in verdict.rows)
    display_rows: list[FitRow] = []
    seen = set()
    for r in verdict.rows:
        key = (
            r.box.name,
            r.dtype,
            r.parallel.label if show_parallel else "",
            round(r.per_chip_gb, 4),
            round(r.usable_per_chip_gb, 4),
            round(r.headroom_gb, 4),
            r.tightness.value,
        )
        if key in seen:
            continue
        seen.add(key)
        display_rows.append(r)

    p()
    p("  " + "-" * (86 if show_parallel else 72))
    if show_parallel:
        hdr = f"  {'BOX':<8}  {'DTYPE':<8}  {'PARALLEL':<15}  {'PER-CHIP':>9}  {'USABLE':>9}  {'HEADROOM':>10}  VERDICT"
    else:
        hdr = f"  {'BOX':<8}  {'DTYPE':<8}  {'PER-CHIP':>9}  {'USABLE':>9}  {'HEADROOM':>10}  VERDICT"
    p(hdr)
    p("  " + "-" * (86 if show_parallel else 72))
    last_box: Optional[str] = None
    for r in display_rows:
        if last_box is not None and last_box != r.box.name:
            p("  " + " " * 6 + "-" * (80 if show_parallel else 66))
        last_box = r.box.name
        prefix = f"  {r.box.name:<8}  {r.dtype:<8}"
        if show_parallel:
            prefix += f"  {r.parallel.label:<15}  "
        else:
            prefix += "  "
        p(
            f"{prefix}  "
            f"{_fmt_gb(r.per_chip_gb):>7}G  "
            f"{_fmt_gb(r.usable_per_chip_gb):>7}G  "
            f"{_fmt_gb(r.headroom_gb):>8}G  "
            f"{r.tightness.value}"
        )
    p("  " + "-" * (86 if show_parallel else 72))

    p()
    if verdict.best:
        r = verdict.best
        parallel_note = f" ({r.parallel.label})" if r.parallel.pp != 1 or r.parallel.ep != 1 else ""
        p(f"  RECOMMENDATION: {r.box.name}{parallel_note} with {r.dtype} weights")
        p(f"                  per-chip={r.per_chip_gb:.1f} GB, headroom={r.headroom_gb:.1f} GB")
        p(f"                  verdict: {r.tightness.value}")
        p(f"                  {r.box.notes}")
        for note in verdict.notes:
            if note.startswith("recommendation bumped"):
                p(f"                  NOTE: {note}")
        p("                  Runtime mesh is selected in `prepare` after TP/divisibility and demo constraints.")
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
    p("| Box | DTYPE | Per-chip | Usable | Headroom | Verdict |")
    p("|---|---|---|---|---|---|")
    seen_md = set()
    for r in verdict.rows:
        key = (
            r.box.name,
            r.dtype,
            round(r.per_chip_gb, 4),
            round(r.usable_per_chip_gb, 4),
            round(r.headroom_gb, 4),
            r.tightness.value,
        )
        if key in seen_md:
            continue
        seen_md.add(key)
        p(
            f"| {r.box.name} | {r.dtype} | "
            f"{r.per_chip_gb:.1f} GB | {r.usable_per_chip_gb:.1f} GB | "
            f"{r.headroom_gb:+.1f} GB | {r.tightness.value} |"
        )
    p("")
    if verdict.best:
        r = verdict.best
        p(
            f"**Recommendation**: `{r.box.name}` with `{r.dtype}` weights — {r.tightness.value}, headroom {r.headroom_gb:.1f} GB."
        )
        p("Runtime mesh is selected in `prepare` after TP/divisibility and demo constraints.")
    else:
        p("**Recommendation**: nothing fits.")
    p("")
    p(f"**Confidence**: {_confidence_for(probe)}")

    return "\n".join(out)


_STATUS_GLYPH = {
    Status.SUPPORTED: "[ ok ]",
    Status.PARTIAL: "[part]",
    Status.MISSING: "[MISS]",
    Status.UNNEEDED: "[ -- ]",
}


def render_compat_table(
    report: CompatReport,
    kernel_report: Optional[KernelReport] = None,
    *,
    verbose: bool = False,
    chips: Optional[int] = None,
) -> str:
    """Human-readable compatibility table."""
    out = []
    p = out.append

    p("=" * 92)
    p(f"HF model:          {report.model_id}")
    p(f"Architecture:      {report.architecture_family}")
    if report.similar_supported_model and report.similar_supported_model != report.model_id:
        p(f"Closest TT port:   {report.similar_supported_model}  (use as a starting point)")
    elif report.similar_supported_model == report.model_id:
        p(f"Already supported: yes -- listed in tt_transformers prefill/perf tables")
    p(f"Overall verdict:   {report.overall}")
    p(f"                   {report.effort_summary}")
    p("=" * 92)
    p("")

    disc = getattr(report, "discovery", None)
    if disc is not None:
        from .discovery import format_inline as _format_discovery

        p("REPO DISCOVERY -- where the model lives in tt-metal source:")
        for line in _format_discovery(disc, indent="  "):
            p(line)
        if getattr(disc, "in_external_demo", False):
            p("  --> `scaffold` does not apply here (its tables only affect tt_transformers).")
            p("  --> `prepare` will route to the discovered pytest entry automatically.")
        elif not getattr(disc, "is_supported", False) and getattr(disc, "target_entry", None) is not None:
            p("  --> Listed as a future CI target, but not yet wired in. The")
            p("      architectural breakdown below tells you what would be needed.")
        elif not getattr(disc, "is_supported", False):
            p("  --> This model is not yet referenced anywhere in models/. See the")
            p("      block-by-block analysis below for what a port would entail.")
        p("")
    p("SECTION 1 -- Building-block availability (does TT have a module for each HF concept?)")
    p("-" * 92)
    p(f"{'STATUS':<7} {'BLOCK':<32} {'EFFORT':<24} TT IMPLEMENTATION")
    p("-" * 92)

    needed = [r for r in report.results if r.needed]
    not_needed = [r for r in report.results if not r.needed]

    for r in needed:
        glyph = _STATUS_GLYPH[r.status]
        tt = r.block.tt_path or "(none)"
        effort = r.effort.value if r.status != Status.SUPPORTED else "drop-in"
        p(f"{glyph:<7} {r.block.name:<32} {effort:<24} {tt}")
        if r.notes and (verbose or r.status != Status.SUPPORTED):
            for line in _wrap_notes(r.notes, indent=8):
                p(line)

    if verbose and not_needed:
        p("")
        p("Not required by this architecture:")
        for r in not_needed:
            p(f"  {_STATUS_GLYPH[Status.UNNEEDED]} {r.block.name}")

    p("")
    p("-" * 92)
    missing = report.by_status(Status.MISSING)
    partial = report.by_status(Status.PARTIAL)
    supported = [r for r in report.results if r.needed and r.status == Status.SUPPORTED]
    p(f"Summary: {len(supported)} ready  /  {len(partial)} partial  /  {len(missing)} missing")

    if kernel_report is not None:
        p("")
        p("")
        p("SECTION 2 -- Kernel-level constraints (do the model's shapes/dtypes satisfy TTNN ops?)")
        p("-" * 92)
        _render_kernel_section(kernel_report, p, verbose=verbose, chips=chips)

    if missing or partial:
        p("")
    if missing:
        p("BLOCKERS -- these require new TT building blocks before this model can run:")
        for r in missing:
            p(f"  * {r.block.name}: {r.block.description}")
    if partial:
        p("WORK ITEMS -- these exist but require adaptation:")
        for r in partial:
            p(f"  * {r.block.name}  ({r.block.tt_path})")

    return "\n".join(out)


def _render_kernel_section(kr: KernelReport, p, *, verbose: bool, chips: Optional[int] = None) -> None:
    """Render the kernel constraints table inside a compat report."""
    shape = kr.shape_findings
    blockers_shape = [f for f in shape if f.passes is False and f.severity == Severity.BLOCKER]
    warns_shape = [f for f in shape if f.passes is False and f.severity == Severity.WARN]
    info_shape = [f for f in shape if f.passes is not True and f.severity == Severity.INFO]

    p(f"{'STATUS':<7} {'OP':<48} {'FIELD':<24} CONSTRAINT")
    p("-" * 92)
    show_set = shape if verbose else (blockers_shape + warns_shape + info_shape)
    if not show_set:
        p("[ ok ]  all shape / dtype constraints pass.")
    else:
        for f in show_set:
            field_val = f"{f.field}={f.value}"
            p(f"{f.status_glyph():<7} {f.op:<48} {field_val:<24} {f.constraint}")
            if not f.passes and f.fix:
                for line in _wrap_notes(f"fix: {f.fix}", indent=8):
                    p(line)
            if verbose and f.source:
                p(f"        source: {f.source}")

    tp_fail = kr.tp_dependent_findings
    p("")
    p("Per-TP divisibility:")
    for tp in kr.tp_grid:
        fails = tp_fail.get(tp, [])
        if not fails:
            p(f"  TP={tp:<3} [ ok ]  all divisibility constraints satisfied")
        else:
            short = ", ".join(f"{f.field}={f.value}" for f in fails)
            p(f"  TP={tp:<3} [FAIL]  {short}")
    if tp_fail:
        p("")
        p("  Note: TP failures rule out that mesh shape, not the model overall.")
        p("        Pick a TP from the rows marked [ ok ] above.")
    if chips and chips > 1:
        from .parallelism import select_parallelism

        _pc = select_parallelism(chips, kr)
        p("")
        p(
            f"  Selected split on {chips} chips: TP={_pc.tp} x DP={_pc.dp}  "
            f"(largest kernel-viable TP that divides {chips}; DP fills the rest)"
        )


def _wrap_notes(notes: str, indent: int = 8) -> list:
    """Naive word-wrap so notes don't sprawl past ~88 cols."""
    width = 80
    pad = " " * indent
    words = notes.replace("\n", " ").split()
    lines, line = [], pad
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line.rstrip())
            line = pad + w
        else:
            line += (" " + w) if line.strip() else w
    if line.strip():
        lines.append(line.rstrip())
    return lines


def render_compat_json(report: CompatReport, kernel_report: Optional[KernelReport] = None) -> str:
    """Machine-readable compatibility report."""
    payload = {
        "schema_version": "compat-1.1",
        "model_id": report.model_id,
        "architecture_family": report.architecture_family,
        "closest_supported_model": report.similar_supported_model,
        "overall": report.overall,
        "effort_summary": report.effort_summary,
        "blocks": [
            {
                "name": r.block.name,
                "description": r.block.description,
                "needed": r.needed,
                "status": r.status.value,
                "effort": r.effort.value,
                "tt_path": r.block.tt_path,
                "notes": r.notes if r.needed else "",
            }
            for r in report.results
        ],
    }
    if kernel_report is not None:
        payload["kernel_constraints"] = {
            "tp_grid": kernel_report.tp_grid,
            "findings_by_tp": {
                str(tp): [
                    {
                        "op": f.op,
                        "field": f.field,
                        "value": f.value,
                        "constraint": f.constraint,
                        "passes": f.passes,
                        "severity": f.severity.value,
                        "fix": f.fix,
                        "source": f.source,
                    }
                    for f in findings
                ]
                for tp, findings in kernel_report.findings_by_tp.items()
            },
        }
    return json.dumps(payload, indent=2)
