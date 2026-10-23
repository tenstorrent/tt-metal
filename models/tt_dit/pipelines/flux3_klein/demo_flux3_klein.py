# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Command-line demo for FLUX 3 Image Klein on a Galaxy 4x8: text-to-image and editing with reference images.

The flags mirror the upstream ``scripts/cli.py`` (``prompt``, ``--model``, ``--image`` repeated, ``--width``,
``--height``, ``--steps``, ``--guidance``, ``--seed``, ``--out``, ``--no-pup``, ``--boxes``). The prompt is
upsampled by default (upstream ``pup=True``): the prompt upsampler (``upsampler.py``, the BFL fine-tune of
Qwen3.5-9B on ONE Blackhole chip) runs FIRST as a subprocess with its own fabric-less 1-chip mesh, then the ring
opens for the DiT (the two cannot share a process: the ring takes all 32 chips and ``set_fabric_config`` is
process-global). A prompt that is already a finished upsampler answer (its box table fully cited) and ``--no-pup``
skip it; a refusal of the upsampler (more than 10 references, a 200:1 aspect, an over-long prompt) falls back to the
prompt as typed, as upstream (``upsampled false`` in the summary and the sidecar). ``--record`` takes one of the
upstream example records (``examples/<set>/<case>.json``) and fills prompt (the RECORDED caption, no upsampling),
model, seed and the reference images from it (the records are vendored under
``models/tt_dit/reference/flux3_klein/examples/``; their PNGs are not, so an edit record needs the BFL clone's copy
of the file, ``$FLUX3_KLEIN_REFERENCE/examples/...``, or ``--image``), so a shipped example reproduces with::

    python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py \\
        --record models/tt_dit/reference/flux3_klein/examples/distilled/man_dog.json --out out/man_dog.png
    python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py \\
        --record $FLUX3_KLEIN_REFERENCE/examples/distilled/3_bandana.json --out out/bandana.png

``--record ... --pup`` upsamples the record's ``input_prompt`` (and its boxes) on device instead: the end-to-end
demo of a shipped example. Without ``--seed`` the seed is ``sha256(caption) % 2**32`` of the caption the DiT sees
(upstream ``generate.py:73``), so the same prompt renders the same image as BFL's CLI; the sidecar records the
seed used. Your own prompt or edit::

    python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "A man walking with his dog, afternoon sun" \\
        --model distilled --seed 7 --out out/t2i.png
    python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "put a red bandana around the dog's neck" \\
        --model distilled --image dog.png --seed 7 --out out/edit.png
    python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "hot-air balloons over a green valley at sunrise" \\
        --boxes boxes.json --out out/layout.png            # 0-1 boxes [y_min, x_min, y_max, x_max] with id / desc

Every render writes a sidecar ``<out>.json`` with the upstream record fields (``model``, ``prompt`` = the caption,
``input_prompt``, ``boxes``, ``width``, ``height``, ``steps``, ``guidance``, ``seed``, ``upsampled``), so a TT render is
replayable with ``--record`` like a BFL example. Several ``--out`` renders in one process reuse the captured traces
and the one caption; pass ``--prompt-file`` for a long caption. A structurally invalid request (a side that is not a
multiple of 16 or over 4096, more than 10 references, a size whose VAE levels are not tile-aligned) is refused on the
host before anything touches the device (``policy.py``), reported as ``REFUSED`` with the check that failed, exit
code 3. There is no token budget: any admitted size runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ttnn
from models.tt_dit.pipelines.flux3_klein.packing import FLUX3_KLEIN_MAX_REFERENCES
from models.tt_dit.pipelines.flux3_klein.pipeline_flux3_klein import Flux3KleinPipeline, Flux3KleinRequest
from models.tt_dit.pipelines.flux3_klein.policy import Flux3KleinAdmissionError
from models.tt_dit.pipelines.flux3_klein.upsampler import Flux3KleinUpsamplerSubprocess
from models.tt_dit.pipelines.flux3_klein.weights_flux3_klein import resolve_weights_dir
from models.tt_dit.reference.flux3_klein import bbox

DIT_VARIANTS = ("base", "distilled")  # the pipeline's checkpoints; "pup" (the prompt upsampler) is not a DiT

# The pipeline's device parameters (tests/models/flux3_klein/device_params.py: ring, 8 KB fabric packets,
# L1_SMALL for the VAE conv2d, the trace region for the DiT + VAE traces).
MESH_SHAPE = (4, 8)
L1_SMALL_SIZE = 65536
TRACE_REGION_SIZE = 256_000_000  # 2026-10-21: base 4096x4096 DiT trace 64.27 MB + the VAE trace exceeded 64 MB
FABRIC_PACKET_BYTES = 8192


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "prompt", nargs="?", help="a prompt (upsampled on device by default) or a finished caption (passed through)"
    )
    parser.add_argument("--prompt-file", type=Path, help="read the prompt / caption from a file instead")
    parser.add_argument(
        "--record",
        type=Path,
        help="an upstream examples/<set>/<case>.json: fills the RECORDED caption (no upsampling unless --pup), "
        "model, seed, references (flags override)",
    )
    parser.add_argument(
        "--no-pup", action="store_true", help="never run the prompt upsampler (the prompt is the caption)"
    )
    parser.add_argument(
        "--pup", action="store_true", help="with --record: upsample the record's input_prompt (and boxes) on device"
    )
    parser.add_argument(
        "--boxes", type=Path, default=None, help="JSON list of 0-1 boxes ([y_min, x_min, y_max, x_max], id, desc)"
    )
    parser.add_argument(
        "--pup-weights", type=Path, default=None, help="the prompt upsampler snapshot (default: the HF cache)"
    )
    parser.add_argument(
        "--model", choices=DIT_VARIANTS, default=None, help="base (50 steps, CFG 3.5) or distilled (16 steps)"
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help=f"reference image; repeat for several (max {FLUX3_KLEIN_MAX_REFERENCES})",
    )
    parser.add_argument("--width", type=int, help="canvas width (default: the first reference's canvas, else 1024)")
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float, help="CFG scale (base only)")
    parser.add_argument("--seed", type=int, default=None, help="default: sha256(caption) %% 2**32, as upstream")
    parser.add_argument("--out", type=Path, default=Path("out.png"))
    parser.add_argument(
        "--weights-base", type=Path, default=None, help="base checkpoint dir (default: the HF cache snapshot)"
    )
    parser.add_argument("--weights-distilled", type=Path, default=None, help="distilled checkpoint dir")
    parser.add_argument("--eager", action="store_true", help="run the denoise and decode without tracing")
    parser.add_argument(
        "--layout",
        choices=("sp4tp8", "sp8tp4", "dual"),
        default="sp8tp4",
        help="DiT mesh layout: sp8tp4 (default since 2026-10-22: under the approximate-exp ring attention it is "
        "5-10 %% faster than sp4tp8 from 2K+refs up and within noise below, 31.70 dB vs the CPU golden at 1K), "
        "sp4tp8, or dual (both resident, SP8 x TP4 from 16K tokens up; buys nothing today)",
    )
    parser.add_argument("--repeat", type=int, default=1, help="render N times (the second run replays the traces)")
    return parser.parse_args(argv)


def resolve_record(args: argparse.Namespace) -> None:
    """Fill the unset fields from an upstream record; reference paths are repo-relative to the record's parent."""
    record = json.loads(args.record.read_text())
    if args.prompt is None and args.prompt_file is None:
        # The recorded caption replays as it is; --pup upsamples the raw input_prompt again (with the record's boxes).
        args.prompt = record["input_prompt"] if args.pup and record.get("input_prompt") else record["prompt"]
    if args.pup and args.boxes is None and record.get("boxes"):
        args.record_boxes = record["boxes"]
    if args.model is None:
        args.model = record.get("model", "base")
    if args.seed is None and record.get("seed") is not None:
        args.seed = int(record["seed"])
    if args.steps is None and record.get("steps") is not None:
        args.steps = int(record["steps"])
    if args.guidance is None and record.get("guidance") is not None:
        args.guidance = float(record["guidance"])
    if not args.image:
        repo_root = next((p for p in args.record.parents if (p / "examples").is_dir()), args.record.parent)
        args.image = [repo_root / entry for entry in record.get("references") or []]


def open_mesh() -> ttnn.MeshDevice:
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = FABRIC_PACKET_BYTES
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        router_config=router,
    )
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*MESH_SHAPE),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.device.DispatchCoreType.WORKER),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.record_boxes = None
    if args.record is not None:
        resolve_record(args)
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text().strip()
    if not args.prompt:
        print("error: a prompt is required (positional, --prompt-file or --record)", file=sys.stderr)
        return 2
    model = args.model or "base"
    for image in args.image:
        if not image.is_file():
            print(f"error: reference image not found: {image}", file=sys.stderr)
            return 2
    boxes = json.loads(args.boxes.read_text()) if args.boxes is not None else args.record_boxes
    try:
        if boxes:
            bbox.to_canvas(boxes)
    except ValueError as error:
        print(f"error: boxes: {error}", file=sys.stderr)
        return 2

    # The caption (generate.py:52-63), the upsampler FIRST in its own process (module docstring): a record replays its
    # recorded caption unless --pup asks for the end-to-end run; a finished caption is never upsampled.
    input_prompt = args.prompt
    finished = not boxes and bbox.is_finished(input_prompt)
    use_pup = not args.no_pup and not finished and (args.record is None or args.pup)
    caption, upsampled = None, False
    if use_pup:
        t0 = time.time()
        try:
            # scratch (boxes / caption json) next to the output; the child is waited for, never killed
            caption = Flux3KleinUpsamplerSubprocess(snapshot=args.pup_weights, scratch_dir=args.out.parent)(
                input_prompt, [str(p) for p in args.image], boxes
            )
            upsampled = True
            print(f"upsampled in {time.time() - t0:.1f}s: {caption[:120]}{'...' if len(caption) > 120 else ''}")
        except ValueError as error:
            print(f"upsampling skipped: {error}", file=sys.stderr)

    request = Flux3KleinRequest(
        prompt=caption if caption is not None else input_prompt,
        model=model,
        width=args.width,
        height=args.height,
        seed=args.seed,  # None: the pipeline hashes the caption (generate.py:73)
        steps=args.steps,
        guidance=args.guidance,
        images=tuple(str(p) for p in args.image),
        output_format=args.out.suffix.lstrip(".").lower() or "png",
        # already a caption (or passed through); the pipeline appends the box table itself when not upsampled
        prompt_upsampling=False,
        boxes=None if caption is not None else (tuple(boxes) if boxes else None),
        return_final_tokens=False,  # a serving-style request: the latent read is not part of the image time
    )
    print(
        f"model {model} seed {request.seed if request.seed is not None else 'sha256(caption)'} "
        f"steps {args.steps or 'default'} guidance {args.guidance or 'default'} "
        f"canvas {args.width or 'default'}x{args.height or 'default'} references {len(args.image)} "
        f"boxes {len(boxes) if boxes else 0} upsampled {upsampled} -> {args.out}"
    )

    overrides = {"base": args.weights_base, "distilled": args.weights_distilled}
    snapshots = {model: overrides[model] or resolve_weights_dir(model)}

    mesh = open_mesh()
    try:
        t0 = time.time()
        layout_kw = {
            "dual": dict(sp_axis=0, tp_axis=1, large_layout=(1, 0)),
            "sp4tp8": dict(sp_axis=0, tp_axis=1),
            "sp8tp4": dict(sp_axis=1, tp_axis=0),
        }[args.layout]
        pipeline = Flux3KleinPipeline(mesh, snapshots=snapshots, resident=(model,), **layout_kw)
        print(f"pipeline resident ({model}) in {time.time() - t0:.1f}s")
        for i in range(args.repeat):
            try:
                result = pipeline.generate(request, traced=not args.eager)
            except Flux3KleinAdmissionError as err:
                adm = err.admission
                print(f"REFUSED: {adm.reason} (budget {adm.budget}, measured {adm.measured})", file=sys.stderr)
                return 3
            out = args.out if args.repeat == 1 else args.out.with_name(f"{args.out.stem}_{i}{args.out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(out)
            # The upstream record fields (generate.py:80-81) next to the image: replayable with --record.
            sidecar = {
                "model": result.model,
                "prompt": result.caption,
                "input_prompt": input_prompt,
                "boxes": boxes,
                "width": result.image.size[0],
                "height": result.image.size[1],
                "steps": result.steps,
                "guidance": result.guidance,
                "seed": result.seed,
                "upsampled": bool(upsampled or result.upsampled),
                "references": [str(p) for p in args.image],
            }
            out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, ensure_ascii=False) + "\n")
            t = result.timings
            print(
                f"[{i}] saved {out} {result.image.size[0]}x{result.image.size[1]} "
                f"total {t.get('total_s', 0.0):.2f}s (trace captured {t.get('trace_captured')}); "
                f"references {result.references} crop {result.crop} upsampled {sidecar['upsampled']} "
                f"caption L {result.text_length}"
            )
        pipeline.close()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
