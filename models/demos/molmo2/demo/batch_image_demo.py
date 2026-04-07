# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run **separate** image+text inferences from a JSON manifest (not a single batched forward).

Each entry is one full ``Molmo2Generator.run_inference`` (prefill + decode). This matches
**32 independent requests** by listing 32 objects in the manifest — throughput is sequential
on one mesh unless you run multiple processes.

**In-tree preference:** Use the **CLI** as the default entry point — ``python -m ...`` with
``--manifest`` (alias: ``--batch-manifest``) — so runs are reproducible from shell/CI and
match the example manifest under ``verification/``. For notebooks, tests, or a custom driver
that already holds a ``Molmo2Generator``, call ``run_batch_image_inference()`` directly
instead of shelling out; the module ``main()`` is a thin wrapper around that helper.

Manifest format: JSON array of objects::

    [
      {"image": "path/to/a.jpg", "prompt": "<|image|> Your question."},
      ...
    ]

Paths may be absolute or relative to the **current working directory** (run from ``tt-metal``).
The checked-in ``batch_manifest.example.json`` has **32** rows for a sequential 32-request run
(same image, varied prompts). That is **32× text batch 1**, not one LM batch of 32: the mesh is
**TP=8** on ``MeshShape(1, 8)`` (same as the video demo). **Vision DP=8** applies to **multi-frame**
ViT batches in ``demo.py`` (e.g. 32 frames ⇒ four 8-wide frame-DP rounds), not to this sequential
manifest driver.

Example::

    cd /path/to/tt-metal
    python -m models.demos.molmo2.demo.batch_image_demo \\
      --manifest models/demos/molmo2/verification/batch_manifest.example.json \\
      --max-tokens 32

Tracing is **off** by default so per-request shapes and KV state do not reuse wrong captured graphs.
With trace flags on, use the same options as ``demo.py``; each manifest row calls
``release_all_traces()`` first so captures are not replayed with stale programs across different images.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

import ttnn


def _configure_molmo2_stdlib_logging_to_stderr() -> None:
    """Match demo visibility for stdlib loggers under ``models.demos.molmo2``."""
    molmo_log = logging.getLogger("models.demos.molmo2")
    molmo_log.setLevel(logging.INFO)
    if molmo_log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"))
    molmo_log.addHandler(handler)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON array")
    out = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"entry {i} must be an object")
        if "image" not in row:
            raise ValueError(f"entry {i} missing 'image'")
        if "prompt" not in row:
            raise ValueError(f"entry {i} missing 'prompt'")
        out.append(row)
    return out


def print_text_tensor_parallel_banner(mesh_device, model, mesh_shape_rc: Tuple[int, int]) -> None:
    """
    Explain decoder TP on the mesh: Q/KV heads are column-sharded; attention/MLP use
    ``all_reduce(..., cluster_axis=1)`` to combine partial hidden states across devices.
    """
    attn = model.text_model.blocks[0].self_attn
    n_dev = mesh_device.get_num_devices()
    rows, cols = int(mesh_shape_rc[0]), int(mesh_shape_rc[1])
    total_q = attn.num_heads_per_device * n_dev
    total_kv = attn.num_kv_heads_per_device * n_dev
    lines = [
        "",
        "=== Molmo2 text decoder: tensor parallelism (TP) on mesh ===",
        f"  MeshShape({rows}, {cols})  ->  {n_dev} devices; text ops use cluster_axis=1 all-reduce (column mesh).",
        f"  Attention: {total_q} Q heads total -> {attn.num_heads_per_device} Q heads per device.",
        f"  Attention: {total_kv} KV heads total -> {attn.num_kv_heads_per_device} KV heads per device.",
        "  Each manifest entry runs with text batch_size=1; the same text TP=8 layout is used on every request.",
        "  32 manifest rows = 32 sequential requests (not one text batch of 32). Multi-frame video TP=8 + vision DP width 8 is in demo.py (--video / --max-video-frames).",
        "",
    ]
    for line in lines:
        print(line, flush=True)
    logger.info(
        "text TP: mesh %s, devices=%s, q_heads/dev=%s, kv_heads/dev=%s",
        (rows, cols),
        n_dev,
        attn.num_heads_per_device,
        attn.num_kv_heads_per_device,
    )


def resolve_image_path(manifest_path: Path, image_field: str) -> Path:
    p = Path(image_field)
    if p.is_absolute():
        return p
    # Prefer cwd; if missing, try relative to manifest directory
    if p.exists():
        return p.resolve()
    alt = (manifest_path.parent / p).resolve()
    if alt.exists():
        return alt
    return p.resolve()


def run_batch_image_inference(
    entries: List[Dict[str, Any]],
    *,
    manifest_path: Path,
    generator,
    preprocess_image_molmo2,
    max_new_tokens: int,
    use_trace: bool,
    use_decode_trace: bool,
    use_vision_trace: bool,
    use_unified_trace: bool,
    limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Run one inference per manifest entry. Returns (per_row_results, aggregate_timings).
    """
    from models.demos.molmo2.demo.demo import IMAGE_PROMPT

    rows = entries[: limit if limit is not None else len(entries)]
    results: List[Dict[str, Any]] = []
    t0_wall = time.perf_counter()
    total_ttft = 0.0
    total_decode = 0.0
    ok_count = 0

    for idx, row in enumerate(rows):
        img_path = resolve_image_path(manifest_path, row["image"])
        prompt = row["prompt"]
        if IMAGE_PROMPT not in prompt:
            prompt = f"{IMAGE_PROMPT} {prompt}"

        logger.info(f"[{idx + 1}/{len(rows)}] {img_path.name}")

        row_out: Dict[str, Any] = {
            "index": idx,
            "image": str(img_path),
            "prompt": row["prompt"],
            "ok": False,
            "output": "",
            "error": None,
            "metrics": {},
        }

        if not img_path.is_file():
            row_out["error"] = f"missing file: {img_path}"
            results.append(row_out)
            continue

        try:
            # Drop captured traces so a new image / prompt cannot replay another row's trace program.
            if use_unified_trace or use_trace or use_decode_trace or use_vision_trace:
                generator.release_all_traces()

            generator.reset_kv_cache(0)
            generator.decode_position = 0

            image_inputs = preprocess_image_molmo2(str(img_path))
            text, metrics = generator.run_inference(
                image_inputs=image_inputs,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                use_trace=use_trace,
                use_decode_trace=use_decode_trace,
                use_vision_trace=use_vision_trace,
                use_unified_trace=use_unified_trace,
            )
            row_out["ok"] = True
            row_out["output"] = text
            row_out["metrics"] = {
                k: metrics[k]
                for k in (
                    "ttft_ms",
                    "e2e_ttft_ms",
                    "vision_ms",
                    "total_decode_ms",
                    "generated_tokens",
                    "input_tokens",
                    "tokens_per_sec",
                )
                if k in metrics
            }
            total_ttft += float(metrics.get("ttft_ms", 0) or 0)
            total_decode += float(metrics.get("total_decode_ms", 0) or 0)
            ok_count += 1
        except Exception as e:
            row_out["error"] = str(e)
            logger.exception(f"Request {idx} failed")

        results.append(row_out)

    wall_s = time.perf_counter() - t0_wall
    agg = {
        "wall_time_s": wall_s,
        "requests": len(rows),
        "succeeded": ok_count,
        "failed": len(rows) - ok_count,
        "sum_ttft_ms": total_ttft,
        "sum_decode_ms": total_decode,
    }
    return results, agg


def main(argv: Optional[List[str]] = None) -> int:
    from models.demos.molmo2.demo.demo import (
        Molmo2Generator,
        create_model,
        load_model_weights,
        load_processor,
        open_molmo_mesh_device,
        preprocess_image_molmo2,
    )

    _configure_molmo2_stdlib_logging_to_stderr()

    p = argparse.ArgumentParser(description="Molmo2 batch image demo (sequential independent requests)")
    p.add_argument(
        "--manifest",
        "--batch-manifest",
        type=Path,
        required=True,
        dest="manifest",
        help="JSON manifest (array of {image, prompt})",
    )
    p.add_argument("--max-tokens", type=int, default=64, help="Max new tokens per request")
    p.add_argument("--limit", type=int, default=None, help="Process only first N entries")
    p.add_argument("--num-layers", type=int, default=None, help="Text layers (default 36)")
    p.add_argument("--max-seq-len", type=int, default=2048, help="KV max sequence length")
    p.add_argument("--output", type=Path, default=None, help="Write JSON results here")
    p.add_argument(
        "--use-trace",
        action="store_true",
        help="Enable prefill trace (may be unsafe when seq_len repeats across different images)",
    )
    p.add_argument("--use-decode-trace", action="store_true", help="Enable decode trace")
    p.add_argument("--use-vision-trace", action="store_true", help="Enable vision trace")
    p.add_argument("--use-unified-trace", action="store_true", help="Enable unified vision+prefill trace")
    args = p.parse_args(argv)

    manifest_path = args.manifest.resolve()
    entries = load_manifest(manifest_path)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape_rc = (1, 8)
    mesh_shape = ttnn.MeshShape(*mesh_shape_rc)
    logger.info(f"Opening mesh {mesh_shape}")
    device = open_molmo_mesh_device(
        use_unified_trace=args.use_unified_trace,
        trace_region_size=None,
    )
    try:
        state_dict = load_model_weights()
        tokenizer = load_processor()
        model = create_model(
            device,
            state_dict,
            args.num_layers,
            max_seq_len=args.max_seq_len,
            image_pooling_use_tensor_parallel=not (args.use_vision_trace or args.use_unified_trace),
        )
        text_num_layers = args.num_layers if args.num_layers is not None else 36

        print_text_tensor_parallel_banner(device, model, mesh_shape_rc)

        generator = Molmo2Generator(
            mesh_device=device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=args.max_seq_len,
        )

        use_trace = args.use_trace
        use_decode = args.use_decode_trace
        use_vision = args.use_vision_trace
        use_unified = args.use_unified_trace
        if not use_trace and not use_decode and not use_vision and not use_unified:
            logger.info("All tracing disabled (recommended for multi-request manifests).")

        results, agg = run_batch_image_inference(
            entries,
            manifest_path=manifest_path,
            generator=generator,
            preprocess_image_molmo2=preprocess_image_molmo2,
            max_new_tokens=args.max_tokens,
            use_trace=use_trace,
            use_decode_trace=use_decode,
            use_vision_trace=use_vision,
            use_unified_trace=use_unified,
            limit=args.limit,
        )

        payload = {"aggregate": agg, "results": results}
        text = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
            logger.info(f"Wrote {args.output}")
        else:
            print(text)

        logger.info(
            f"Batch done: {agg['succeeded']}/{agg['requests']} ok, wall={agg['wall_time_s']:.2f}s, "
            f"sum_ttft_ms={agg['sum_ttft_ms']:.1f}, sum_decode_ms={agg['sum_decode_ms']:.1f}"
        )
        return 0 if agg["failed"] == 0 else 1
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    sys.exit(main())
