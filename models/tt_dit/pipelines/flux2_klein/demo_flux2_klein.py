# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Command-line demo for FLUX.2 Klein on a Galaxy 4x8: text-to-image and editing with reference images.

The flags mirror the diffusers ``Flux2KleinPipeline.__call__`` arguments (``prompt``, ``image`` repeated,
``height`` / ``width``, ``num_inference_steps``, ``guidance_scale``, ``generator`` seed) and BFL's ``cli.py``
(``--model``). The distilled model (the default) renders in 4 steps without guidance; the base model in 50 steps
with CFG 4.0 (BFL ``util.py`` defaults). Without ``--seed`` a fresh 32-bit seed is drawn and recorded in the
sidecar. Your own prompt or edit::

    python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "A man walking with his dog, afternoon sun" \\
        --seed 7 --out out/t2i.png
    python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "put a red bandana around the dog's neck" \\
        --image dog.png --seed 7 --out out/edit.png
    python models/tt_dit/pipelines/flux2_klein/demo_flux2_klein.py "a watercolour harbour at dawn" \\
        --model base --guidance 4.0 --width 1536 --height 1024 --out out/base.png

Every render writes a sidecar ``<out>.json`` (``model``, ``size``, ``prompt``, ``width``, ``height``, ``steps``,
``guidance``, ``seed``, ``references``) so a TT render is reproducible. ``--repeat N`` renders N times in one process
(the second run replays the captured traces); pass ``--prompt-file`` for a long prompt. A structurally invalid
request (a side that is not a multiple of 16 or over 4096, more than 10 references, a size whose VAE levels are not
tile-aligned) is refused on the host before anything touches the device (``policy.py``), reported as ``REFUSED``
with the check that failed, exit code 3; a bad input exits 2. There is no token budget: any admitted size runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ttnn
from models.tt_dit.pipelines.flux2_klein.packing import FLUX2_KLEIN_MAX_REFERENCES
from models.tt_dit.pipelines.flux2_klein.pipeline_flux2_klein import (
    MODELS,
    SIZES,
    Flux2KleinPipeline,
    Flux2KleinRequest,
)
from models.tt_dit.pipelines.flux2_klein.policy import Flux2KleinAdmissionError
from models.tt_dit.pipelines.flux2_klein.weights_flux2_klein import WeightsNotFoundError, resolve_weights_dir

DIT_VARIANTS = tuple(MODELS)

# The pipeline's device parameters (ring fabric, 8 KB fabric packets, L1_SMALL for the VAE conv2d, the trace region
# for the DiT + VAE traces): the FLUX 3 Klein demo's open_mesh.
MESH_SHAPE = (4, 8)
L1_SMALL_SIZE = 65536
TRACE_REGION_SIZE = 256_000_000
FABRIC_PACKET_BYTES = 8192

LAYOUTS = {
    "sp8tp4": dict(sp_axis=1, tp_axis=0),
    "sp4tp8": dict(sp_axis=0, tp_axis=1),
    "dual": dict(sp_axis=0, tp_axis=1, large_layout=(1, 0)),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", help="the prompt")
    parser.add_argument("--prompt-file", type=Path, help="read the prompt from a file instead")
    parser.add_argument(
        "--model",
        choices=DIT_VARIANTS,
        default="distilled",
        help="distilled (4 steps, no CFG) or base (50 steps, CFG 4.0)",
    )
    parser.add_argument("--size", choices=SIZES, default="4b", help="the released size: 4b (default) or 9b")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help=f"reference image; repeat for several (max {FLUX2_KLEIN_MAX_REFERENCES})",
    )
    parser.add_argument("--width", type=int, help="canvas width (default: the first reference's size, else 1024)")
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float, help="CFG scale (base only)")
    parser.add_argument("--seed", type=int, default=None, help="default: a fresh 32-bit seed, recorded in the sidecar")
    parser.add_argument("--out", type=Path, default=Path("out.png"))
    parser.add_argument(
        "--weights", type=Path, default=None, help="the diffusers snapshot directory (default: the HF cache snapshot)"
    )
    parser.add_argument("--eager", action="store_true", help="run the denoise and decode without tracing")
    parser.add_argument(
        "--layout",
        choices=tuple(LAYOUTS),
        default="sp8tp4",
        help="DiT mesh layout: sp8tp4 (default), sp4tp8, or dual (both resident, SP8 x TP4 from 16K tokens up)",
    )
    parser.add_argument("--repeat", type=int, default=1, help="render N times (the second run replays the traces)")
    return parser.parse_args(argv)


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
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text().strip()
    if not args.prompt:
        print("error: a prompt is required (positional or --prompt-file)", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print(f"error: --repeat must be at least 1, got {args.repeat}", file=sys.stderr)
        return 2
    for image in args.image:
        if not image.is_file():
            print(f"error: reference image not found: {image}", file=sys.stderr)
            return 2

    request = Flux2KleinRequest(
        prompt=args.prompt,
        model=args.model,
        size=args.size,
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        images=tuple(str(p) for p in args.image),
        output_format=args.out.suffix.lstrip(".").lower() or "png",
        return_final_tokens=False,  # a serving-style request: the latent read is not part of the image time
        return_text=False,  # nor the text read: the encoding goes to the DiT on the device
    )
    try:
        request.resolve()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"model {args.model} size {args.size} seed {args.seed if args.seed is not None else 'random'} "
        f"steps {args.steps or 'default'} guidance {args.guidance if args.guidance is not None else 'default'} "
        f"canvas {args.width or 'default'}x{args.height or 'default'} references {len(args.image)} -> {args.out}"
    )

    try:
        snapshots = {args.model: resolve_weights_dir(args.model, args.size, weights_dir=args.weights)}
    except WeightsNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    mesh = open_mesh()
    try:
        t0 = time.time()
        pipeline = Flux2KleinPipeline(
            mesh, snapshots=snapshots, size=args.size, resident=(args.model,), **LAYOUTS[args.layout]
        )
        print(f"pipeline resident ({args.model}/{args.size}) in {time.time() - t0:.1f}s")
        for i in range(args.repeat):
            try:
                result = pipeline.generate(request, traced=not args.eager)
            except Flux2KleinAdmissionError as err:
                adm = err.admission
                print(f"REFUSED: {adm.reason} (budget {adm.budget}, measured {adm.measured})", file=sys.stderr)
                return 3
            out = args.out if args.repeat == 1 else args.out.with_name(f"{args.out.stem}_{i}{args.out.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(out)
            sidecar = {
                "model": result.model,
                "size": result.size,
                "prompt": result.input_prompt,
                "width": result.image.size[0],
                "height": result.image.size[1],
                "steps": result.steps,
                "guidance": result.guidance,
                "seed": result.seed,
                "references": [str(p) for p in args.image],
            }
            out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, ensure_ascii=False) + "\n")
            t = result.timings
            print(
                f"[{i}] saved {out} {result.image.size[0]}x{result.image.size[1]} seed {result.seed} "
                f"total {t.get('total_s', 0.0):.2f}s (trace captured {t.get('trace_captured')}); "
                f"references {result.references} layout {result.layout}"
            )
        pipeline.close()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
