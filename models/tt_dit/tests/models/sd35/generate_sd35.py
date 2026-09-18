"""Generate SD3.5-Large images from your own prompts on a 4-chip submesh.

Bringing the pipeline up costs ~25 s (weight conversion) plus a trace capture, so this script
pays that once and then keeps the pipeline alive: every prompt after the first costs only the
generation itself (~7.5 s for 28 steps at 1024px on a Blackhole QuietBox).

Usage (from the repo root):

    # interactive -- type prompts, keep generating, :q to quit
    python3 models/tt_dit/tests/models/sd35/generate_sd35.py

    # one-shot
    python3 models/tt_dit/tests/models/sd35/generate_sd35.py "a red fox in deep snow"

    # several prompts in one session
    python3 models/tt_dit/tests/models/sd35/generate_sd35.py "prompt one" "prompt two"

    # from a file, one prompt per line (blank lines and #-comments ignored)
    python3 models/tt_dit/tests/models/sd35/generate_sd35.py -f prompts.txt

Interactive commands (anything else is treated as a prompt):

    :steps N        denoising steps          :seed N|rand   fixed seed, or a new one per image
    :guidance F     CFG scale (1 = off)      :negative TXT  negative prompt ("" clears it)
    :count N        images per prompt        :show          print the current settings
    :q                                       quit

The mesh setup mirrors run_sd35_submesh.py: the system mesh is read from the SystemMesh
descriptor and a 4-chip submesh is sliced out of it, since opening a bare sub-mesh on a Galaxy
fails fabric router sync. Keep the two layout tables in sync if either changes.
"""

import argparse
import os
import random
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())
from models.tt_dit.tests.models.sd35.hf_env import ensure_hf_home

# Must run before any transformers/diffusers import caches the HF paths.
ensure_hf_home()

from loguru import logger

import ttnn
from conftest import reset_fabric, set_fabric
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    StableDiffusion3Pipeline,
    StableDiffusion3PipelineConfig,
)

# Mirrors run_sd35_submesh.py. On a Galaxy only the 2x2 corner is a physically closed ring of 4
# chips (a native 1x4 row hangs in CCLs), so the corner is relabelled to 1x4 in ring order.
_LAYOUTS = {
    "2x2": (ttnn.MeshShape(2, 2), None, (2, 0), (2, 1)),
    "1x4c": (ttnn.MeshShape(2, 2), ttnn.MeshShape(1, 4), (1, 0), (4, 1)),
    "4x1c": (ttnn.MeshShape(2, 2), ttnn.MeshShape(4, 1), (4, 0), (1, 1)),
    "1x4": (ttnn.MeshShape(1, 4), None, (1, 0), (4, 1)),
    "4x1": (ttnn.MeshShape(4, 1), None, (4, 0), (1, 1)),
    "4x1tp": (ttnn.MeshShape(4, 1), None, (1, 1), (4, 0)),
    "1x4tp": (ttnn.MeshShape(1, 4), None, (1, 0), (4, 1)),
}


def resolve_layout(system_shape, requested="auto"):
    """Pick a 4-chip layout for this system. Returns (layout name, closed_ring)."""
    if requested != "auto":
        return requested, requested in ("2x2", "1x4c", "4x1c")
    if system_shape[0] == 4:
        return "4x1tp", True
    if system_shape[1] == 4:
        return "1x4tp", True
    if system_shape == (2, 2):
        return "1x4c", True
    # Four consecutive chips of a longer chain are a line, not a ring.
    if system_shape[0] >= 4:
        return "4x1tp", False
    if system_shape[1] >= 4:
        return "1x4tp", False
    raise SystemExit(f"need four chips in a line; the system mesh is {system_shape}")


def slugify(text, limit=48):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "image"


def read_prompt_file(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate SD3.5-Large images from your own prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no prompts and no --prompt-file, the script starts an interactive session.",
    )
    p.add_argument("prompts", nargs="*", help="prompts to render; omit for an interactive session")
    p.add_argument("-f", "--prompt-file", help="file with one prompt per line (# comments ignored)")
    p.add_argument("-i", "--interactive", action="store_true", help="stay interactive after the given prompts")
    p.add_argument("-n", "--steps", type=int, default=28, help="denoising steps (default: 28)")
    p.add_argument("-g", "--guidance", type=float, default=3.5, help="CFG scale; 1 disables CFG (default: 3.5)")
    p.add_argument("-s", "--seed", default="0", help="integer seed, or 'rand' for a new one per image")
    p.add_argument("--negative", default="", help="negative prompt (default: empty)")
    p.add_argument("-c", "--count", type=int, default=1, help="images per prompt, seed incrementing (default: 1)")
    p.add_argument("-o", "--outdir", default="sd35_images", help="output directory (default: sd35_images)")
    p.add_argument("--size", type=int, default=1024, help="square image size, multiple of 128 (default: 1024)")
    p.add_argument("--layout", default=os.environ.get("SD35_LAYOUT", "auto"), help="4-chip layout (default: auto)")
    p.add_argument("--links", type=int, default=int(os.environ.get("SD35_LINKS", "2")), help="fabric links")
    p.add_argument("--no-trace", action="store_true", help="disable tracing (slower; for debugging)")
    p.add_argument("-v", "--verbose", action="store_true", help="keep the pipeline's INFO logging")
    t5 = p.add_mutually_exclusive_group()
    t5.add_argument(
        "--t5",
        dest="t5",
        action="store_true",
        default=True,
        help="use the T5 text encoder for better prompt adherence (default; first load downloads ~10 GB)",
    )
    t5.add_argument(
        "--no-t5", dest="t5", action="store_false", help="CLIP only: much faster startup, weaker prompt following"
    )
    return p.parse_args()


class Settings:
    """Generation knobs the interactive session can change between prompts."""

    def __init__(self, args):
        self.steps = args.steps
        self.guidance = args.guidance
        self.seed = args.seed
        self.negative = args.negative
        self.count = args.count

    def next_seed(self, offset):
        if str(self.seed).lower() in ("rand", "random", "-1"):
            return random.randrange(2**31)
        return int(self.seed) + offset

    def __str__(self):
        return (
            f"steps={self.steps} guidance={self.guidance} seed={self.seed} "
            f"count={self.count} negative={self.negative!r}"
        )


def apply_command(line, settings):
    """Handle a ':' command. Returns True to keep going, False to quit."""
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    val = parts[1].strip() if len(parts) > 1 else ""
    try:
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd == "steps":
            settings.steps = max(1, int(val))
        elif cmd == "seed":
            settings.seed = val if val.lower() in ("rand", "random") else str(int(val))
        elif cmd == "guidance":
            settings.guidance = float(val)
        elif cmd == "count":
            settings.count = max(1, int(val))
        elif cmd == "negative":
            settings.negative = val
        elif cmd == "show":
            pass
        else:
            print(f"unknown command ':{cmd}' -- try :steps :seed :guidance :count :negative :show :q")
            return True
    except ValueError:
        print(f"bad value for ':{cmd}': {val!r}")
        return True
    print(f"  {settings}")
    return True


def generate(pipeline, prompt, settings, args, traced):
    """Render `settings.count` images for one prompt. Returns the paths written."""
    written = []
    for i in range(settings.count):
        seed = settings.next_seed(i)
        start = time.time()
        # One prompt per call: the pipeline allocates and traces for batch 1, so a longer
        # prompt list would change the traced shapes.
        images = pipeline(
            prompts=[prompt],
            negative_prompts=[settings.negative],
            num_inference_steps=settings.steps,
            guidance_scale=settings.guidance,
            seed=seed,
            traced=traced,
        )
        elapsed = time.time() - start
        name = f"{datetime.now():%Y%m%d-%H%M%S}_{slugify(prompt)}_s{seed}.png"
        path = os.path.join(args.outdir, name)
        images[0].save(path)
        written.append(path)
        print(f"  {path}  ({elapsed:.1f}s total, {settings.steps} steps, seed {seed})")
    return written


def main():
    args = parse_args()
    if not args.verbose:
        # The per-call INFO lines interleave with the interactive prompt; warnings still show.
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    if args.size % 128 != 0:
        raise SystemExit(f"--size must be a multiple of 128, got {args.size}")

    prompts = list(args.prompts)
    if args.prompt_file:
        prompts += read_prompt_file(args.prompt_file)
    interactive = args.interactive or not prompts

    os.makedirs(args.outdir, exist_ok=True)

    system_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    layout, closed_ring = resolve_layout(system_shape, args.layout)
    mesh_shape, reshape_to, sp_cfg, tp_cfg = _LAYOUTS[layout]
    topology_name = os.environ.get("SD35_TOPOLOGY", "ring" if closed_ring else "linear")
    topology = {"linear": ttnn.Topology.Linear, "ring": ttnn.Topology.Ring}[topology_name]
    fabric = {"linear": ttnn.FabricConfig.FABRIC_1D, "ring": ttnn.FabricConfig.FABRIC_1D_RING}[
        os.environ.get("SD35_FABRIC", topology_name)
    ]

    # 8 KB fabric packets: ~5% faster per step than the 4352 B default (see run_sd35_submesh.py).
    packet = int(os.environ.get("SD35_FABRIC_PACKET", "8192"))
    if packet:
        router_config = ttnn.FabricRouterConfig()
        router_config.max_packet_payload_size_bytes = packet
        set_fabric(fabric, fabric_router_config=router_config)
    else:
        set_fabric(fabric)

    full = ttnn.open_mesh_device(
        l1_small_size=int(os.environ.get("SD35_L1_SMALL", "65536")),
        trace_region_size=50_000_000,
    )
    try:
        sub = full.create_submeshes(mesh_shape)[0]
        if reshape_to is not None:
            sub.reshape(reshape_to)
        logger.info(f"submesh {sub.shape} of {full.shape} layout={layout} topology={topology} links={args.links}")

        encoders = "CLIP+T5" if args.t5 else "CLIP"
        print(f"loading SD3.5-Large ({encoders}) on {sub.get_num_devices()} chips...")
        build_start = time.time()
        pipeline = StableDiffusion3Pipeline(
            device=sub,
            config=StableDiffusion3PipelineConfig.default(
                mesh_shape=sub.shape,
                dit_parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=sp_cfg, tp=tp_cfg),
                topology=topology,
                num_links=args.links,
                height=args.size,
                width=args.size,
                cfg_enabled=args.guidance > 1,
                enable_t5_text_encoder=args.t5,
                checkpoint_name="stabilityai/stable-diffusion-3.5-large",
            ),
        )
        print(f"ready in {time.time() - build_start:.0f}s -- images land in {os.path.abspath(args.outdir)}/")

        settings = Settings(args)
        traced = not args.no_trace

        for prompt in prompts:
            print(f'\n"{prompt}"')
            generate(pipeline, prompt, settings, args, traced)

        if interactive:
            print(f"\n{settings}")
            print("enter a prompt, or :show :steps :seed :guidance :count :negative :q")
            while True:
                try:
                    line = input("\nprompt> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line:
                    continue
                if line.startswith(":"):
                    if not apply_command(line, settings):
                        break
                    continue
                try:
                    generate(pipeline, line, settings, args, traced)
                except KeyboardInterrupt:
                    print("\n  interrupted")
    finally:
        for s in full.get_submeshes():
            ttnn.close_mesh_device(s)
        ttnn.close_mesh_device(full)
        reset_fabric(fabric)


if __name__ == "__main__":
    main()
