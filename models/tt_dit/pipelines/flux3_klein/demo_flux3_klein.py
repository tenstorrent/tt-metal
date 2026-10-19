# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Command-line demo for FLUX 3 Image Klein on a Galaxy 4x8: text-to-image and editing with reference images.

The flags mirror the upstream ``scripts/cli.py`` (``prompt``, ``--model``, ``--image`` repeated, ``--width``,
``--height``, ``--steps``, ``--guidance``, ``--seed``, ``--out``). The prompt must already be an upsampled caption
(upstream ``--no-pup``) until the prompt upsampler is on device; ``--record`` takes one of the upstream example
records (``examples/<set>/<case>.json``) and fills prompt, model, seed and the reference images from it, so a
shipped example reproduces with::

    python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py \\
        --record /data/nwoodall/flux3_klein/reference/examples/distilled/3_bandana.json --out out/bandana.png

and your own edit with::

    python models/tt_dit/pipelines/flux3_klein/demo_flux3_klein.py "<upsampled caption>" \\
        --model distilled --image dog.png --seed 7 --out out/edit.png

Several ``--out`` renders in one process reuse the captured traces; pass ``--prompt-file`` for a long caption.
A structurally invalid request (a side that is not a multiple of 16 or over 4096, more than 10 references, a size
whose VAE levels are not tile-aligned) is refused on the host before anything touches the device (``policy.py``),
reported as ``REFUSED`` with the check that failed, exit code 3. There is no token budget: any admitted size runs.
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
from models.tt_dit.pipelines.flux3_klein.weights_flux3_klein import resolve_weights_dir

DIT_VARIANTS = ("base", "distilled")  # the pipeline's checkpoints; "pup" (the prompt upsampler) is not a DiT

# The pipeline's device parameters (tests/models/flux3_klein/device_params.py: ring, 8 KB fabric packets,
# L1_SMALL for the VAE conv2d, the trace region for the DiT + VAE traces).
MESH_SHAPE = (4, 8)
L1_SMALL_SIZE = 65536
TRACE_REGION_SIZE = 64_000_000
FABRIC_PACKET_BYTES = 8192


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", help="an UPSAMPLED caption (the prompt upsampler is not on device yet)")
    parser.add_argument("--prompt-file", type=Path, help="read the caption from a file instead")
    parser.add_argument(
        "--record",
        type=Path,
        help="an upstream examples/<set>/<case>.json: fills prompt, model, seed, references (flags override)",
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("out.png"))
    parser.add_argument(
        "--weights-base", type=Path, default=None, help="base checkpoint dir (default: the HF cache snapshot)"
    )
    parser.add_argument("--weights-distilled", type=Path, default=None, help="distilled checkpoint dir")
    parser.add_argument("--eager", action="store_true", help="run the denoise and decode without tracing")
    parser.add_argument("--repeat", type=int, default=1, help="render N times (the second run replays the traces)")
    return parser.parse_args(argv)


def resolve_record(args: argparse.Namespace) -> None:
    """Fill the unset fields from an upstream record; reference paths are repo-relative to the record's parent."""
    record = json.loads(args.record.read_text())
    if args.prompt is None and args.prompt_file is None:
        args.prompt = record["prompt"]
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

    request = Flux3KleinRequest(
        prompt=args.prompt,
        model=model,
        width=args.width,
        height=args.height,
        seed=args.seed if args.seed is not None else 0,
        steps=args.steps,
        guidance=args.guidance,
        images=tuple(str(p) for p in args.image),
        output_format=args.out.suffix.lstrip(".").lower() or "png",
    )
    print(
        f"model {model} seed {request.seed} steps {args.steps or 'default'} guidance {args.guidance or 'default'} "
        f"canvas {args.width or 'default'}x{args.height or 'default'} references {len(args.image)} -> {args.out}"
    )

    overrides = {"base": args.weights_base, "distilled": args.weights_distilled}
    snapshots = {model: overrides[model] or resolve_weights_dir(model)}

    mesh = open_mesh()
    try:
        t0 = time.time()
        pipeline = Flux3KleinPipeline(mesh, snapshots=snapshots, resident=(model,))
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
            t = result.timings
            print(
                f"[{i}] saved {out} {result.image.size[0]}x{result.image.size[1]} "
                f"total {t.get('total_s', 0.0):.2f}s (trace captured {t.get('trace_captured')}); "
                f"references {result.references} crop {result.crop}"
            )
        pipeline.close()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
