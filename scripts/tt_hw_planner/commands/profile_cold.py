"""`tt_hw_planner profile-cold <model>` — evidence-based COLD/HOT
classification.

Walks the model's components, measures four signals on the natural
workload:

  1.  Frequency        — fires per pass over N forward passes
  1b. Op affinity      — ttnn-op-level device-favorability score
  2.  CPU latency      — per-component %% of total inference time
  3.  Compute density  — ops / I/O bytes (bandwidth-bound check)

Persists the enriched evidence record to hot_cold.json. The
categorizer reads it via load_hot_cold_evidence() and makes
evidence-based COLD verdicts instead of defaulting to COLD when
signals are missing.

Modality-agnostic: the input factory detects the model's primary
modality (vision / text / audio / multimodal) from the forward
signature + model.config and constructs the appropriate primary
input. The driver framework's introspected fan-out then routes to
the right invocation path.
"""

from __future__ import annotations

import inspect as _inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Tuple


def cmd_profile_cold(args) -> int:
    model_id = args.model_id
    n_passes = int(getattr(args, "n_passes", 3) or 3)
    n_iters = int(getattr(args, "n_iters", 5) or 5)

    from ..bringup_loop import _safe_id, find_demo_dir
    from ..cold_evidence import build_evidence_report, evidence_report_to_hot_cold_dict
    from ..overlay_manager import persist_hot_cold

    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        print(f"error: no scaffolded demo dir for `{model_id}`; run `up` first.", file=sys.stderr)
        return 2

    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        print(f"error: no bringup_status.json at {status_path}", file=sys.stderr)
        return 2

    try:
        status = json.loads(status_path.read_text())
    except Exception as exc:
        print(f"error: failed to read bringup_status.json: {exc}", file=sys.stderr)
        return 2

    # Build {component_name: submodule_path} for every NEW component that
    # has a captured manifest (submodule_path is recorded there).
    component_paths: Dict[str, str] = {}
    for c in status.get("components", []) or []:
        if c.get("status") != "NEW":
            continue
        name = c.get("name") or ""
        if not name:
            continue
        submodule_path = (c.get("submodule_path") or "").strip()
        if not submodule_path:
            # Fall back to capture manifest
            safe = _safe_id(name)
            mp = demo_dir / "_captured" / safe / "manifest.json"
            if mp.is_file():
                try:
                    submodule_path = (json.loads(mp.read_text()).get("submodule_path") or "").strip()
                except Exception:
                    pass
        if submodule_path:
            component_paths[name] = submodule_path

    if not component_paths:
        print(f"error: no components have submodule_path recorded; nothing to probe.", file=sys.stderr)
        return 2

    print(f"profile-cold: loading HF model `{model_id}`…")
    try:
        import transformers

        model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
    except Exception as exc:
        print(f"error: HF load failed: {exc}", file=sys.stderr)
        return 2

    pixel_values_fn, modality = _build_input_factory(model)
    print(
        f"profile-cold: probing {len(component_paths)} component(s) — "
        f"modality={modality}, n_passes={n_passes}, n_iters={n_iters}"
    )

    report = build_evidence_report(
        model=model,
        demo_dir=demo_dir,
        pixel_values_fn=pixel_values_fn,
        component_paths=component_paths,
        n_passes_freq=n_passes,
        n_iters_latency=n_iters,
    )

    classification = evidence_report_to_hot_cold_dict(report)
    persist_hot_cold(model_id, classification)

    # Print per-component summary.
    print()
    print(
        f"  {'component':<35} {'kind':<10} {'freq':>6}  {'cpu_ms':>8} {'pct':>7}  "
        f"{'density':>10} {'aff':>5}  {'evidence'}"
    )
    print(f"  {'-'*35} {'-'*10} {'-'*6}  {'-'*8} {'-'*7}  {'-'*10} {'-'*5}  {'-'*40}")
    for name in sorted(report.keys()):
        ev = report[name]
        freq = "-" if ev.frequency is None else f"{ev.frequency:.2f}"
        lat_ms = "-" if ev.cpu_latency_ms is None else f"{ev.cpu_latency_ms:.2f}"
        pct = "-" if ev.cpu_latency_pct is None else f"{ev.cpu_latency_pct:.2f}%"
        dens = "-" if (ev.compute_density is None or ev.compute_density == 0) else f"{ev.compute_density:.2e}"
        aff = "-" if ev.affinity_score is None else f"{ev.affinity_score:+d}"
        why = "; ".join(ev.evidence)[:60]
        print(f"  {name:<35} {ev.kind:<10} {freq:>6}  {lat_ms:>8} {pct:>7}  {dens:>10} {aff:>5}  {why}")

    hot_count = sum(1 for e in report.values() if e.kind == "HOT")
    cold_count = sum(1 for e in report.values() if e.kind == "COLD")
    unknown_count = sum(1 for e in report.values() if e.kind == "UNKNOWN")
    print()
    print(f"  HOT: {hot_count}, COLD: {cold_count}, UNKNOWN: {unknown_count}")
    print(f"  persisted to: {Path('scripts/tt_hw_planner/overlays') / model_id.replace('/', '_') / 'hot_cold.json'}")
    return 0


def _infer_image_size(model) -> int:
    """Pull image_size from model.config (HF convention). Falls back to
    1024 if not present (vision-default)."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        # Try common attribute names
        for attr in ("image_size", "input_size"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int):
                return val
        # Nested vision config (CLIP-style)
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            val = getattr(vision_cfg, "image_size", None)
            if isinstance(val, int):
                return val
    return 1024


def _detect_modality(model) -> str:
    """Return one of: ``vision`` | ``text`` | ``audio`` | ``multimodal``.

    Heuristic, in order of preference:

      1. Forward signature's first non-self parameter NAME.
         ``pixel_values``      → vision
         ``input_ids`` / ``input_tokens``       → text
         ``input_features`` / ``input_values``  → audio
      2. model.config.model_type lookup against the curated sets in
         scripts.tt_hw_planner.probe (VISION_ONLY_MODEL_TYPES,
         AUDIO_ONLY_MODEL_TYPES).
      3. Default: ``vision`` (the original tool default; the input
         factory's vision-shape tensor passes harmlessly to text/audio
         drivers since they use their own input construction).
    """
    cls = type(model)
    try:
        sig = _inspect.signature(cls.forward)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        first_param = params[0].name if params else ""
    except (TypeError, ValueError):
        first_param = ""

    if first_param in ("pixel_values", "image", "images"):
        return "vision"
    if first_param in ("input_ids", "input_tokens", "token_ids"):
        return "text"
    if first_param in ("input_features", "input_values", "audio_values"):
        return "audio"

    # Fall back to model_type-based detection
    try:
        from ..probe import AUDIO_ONLY_MODEL_TYPES, VISION_ONLY_MODEL_TYPES

        cfg = getattr(model, "config", None)
        mt = getattr(cfg, "model_type", "") if cfg else ""
        mt = (mt or "").lower()
        if mt in VISION_ONLY_MODEL_TYPES:
            return "vision"
        if mt in AUDIO_ONLY_MODEL_TYPES:
            return "audio"
    except Exception:
        pass

    # Has vision_config nested in config → likely multimodal/VLM
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "vision_config", None) is not None:
        return "multimodal"

    return "vision"  # safe-default; driver framework adapts


def _build_input_factory(model) -> Tuple[Callable[[int], Any], str]:
    """Construct a (seed → input-tensor) factory appropriate for the
    model's primary modality.

    Returns ``(factory, modality_label)``. The factory is called by
    cold_evidence's probe with varied seeds across multi-pass runs.

    Modality-aware so the probe works on any HF model class (vision,
    text LLM, audio STT, multimodal VLM) — not just vision. For
    text/audio models the produced tensor flows to the driver framework
    whose introspected_forward picks the correct entry point (input_ids
    for text, input_features for audio); the tensor we synthesize is
    treated as a seed by those drivers.
    """
    import torch

    modality = _detect_modality(model)
    cfg = getattr(model, "config", None)

    if modality == "vision" or modality == "multimodal":
        image_size = _infer_image_size(model)

        def vision_factory(seed: int) -> Any:
            gen = torch.Generator().manual_seed(seed)
            return torch.randn(1, 3, image_size, image_size, generator=gen)

        return vision_factory, ("multimodal" if modality == "multimodal" else f"vision[{image_size}x{image_size}]")

    if modality == "text":
        vocab = getattr(cfg, "vocab_size", None) or 32000
        # Cap vocab range to keep tokens within actual model vocab. Cap seq_len
        # so probe stays fast (we just need to fire the forward).
        max_pos = getattr(cfg, "max_position_embeddings", None) or 2048
        seq_len = min(64, max_pos)

        def text_factory(seed: int) -> Any:
            gen = torch.Generator().manual_seed(seed)
            return torch.randint(1, min(vocab, 1000), (1, seq_len), dtype=torch.long, generator=gen)

        return text_factory, f"text[seq_len={seq_len}, vocab={vocab}]"

    if modality == "audio":
        n_mels = getattr(cfg, "num_mel_bins", None) or 80
        # Cap audio frames to keep probe fast.
        max_pos = getattr(cfg, "max_source_positions", None) or 1500
        n_frames = min(max_pos * 2, 3000)

        def audio_factory(seed: int) -> Any:
            gen = torch.Generator().manual_seed(seed)
            return torch.randn(1, n_mels, n_frames, generator=gen)

        return audio_factory, f"audio[mels={n_mels}, frames={n_frames}]"

    # Fallback (shouldn't reach here given _detect_modality's safe-default)
    def fallback_factory(seed: int) -> Any:
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(1, 3, 224, 224, generator=gen)

    return fallback_factory, "fallback-vision"
