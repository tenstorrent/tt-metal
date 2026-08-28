# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared argument parsing, device setup and reporting for the two demos.

Both demos are the same three moves -- open the mesh, build the pipeline once,
call ONE function from `tt/pipeline.py` -- so those moves live here and each
`demo_*.py` contributes only its own task head. There is exactly one copy of the
wiring (`tt/pipeline.py`) and now exactly one copy of the plumbing around it.

The demos own their device, which is the normal shape for a demo entrypoint and
the reason `tt/pipeline.py` does not: the pipeline runs on whatever `device`
`build_pipeline` is handed, so a test fixture, a demo and the perf harness can
each supply their own without the library ever opening a second one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.demos.flux_2_klein_9b.transformer.tt import inputs as tt_inputs
from models.demos.flux_2_klein_9b.transformer.tt import pipeline as tt_pipeline
from models.demos.flux_2_klein_9b.transformer.tt import reference as tt_reference

# e2e_plan.json::device -- T3K, mesh 1x8, TP=8, FABRIC_1D, l1_small_size 24576.
DEFAULT_TP = 8
L1_SMALL_SIZE = 24576
DEFAULT_TRACE_REGION_SIZE = 768 * 1024 * 1024

DEFAULT_OUTPUT_DIR = "generated/flux_2_klein_9b_transformer"


def build_parser(description: str, *, steps: bool) -> argparse.ArgumentParser:
    """The demo flags from `e2e_plan.json::shapes.demo_default`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--height", type=int, default=tt_pipeline.DEFAULT_HEIGHT, help="image height in pixels")
    parser.add_argument("--width", type=int, default=tt_pipeline.DEFAULT_WIDTH, help="image width in pixels")
    parser.add_argument(
        "--txt-len",
        type=int,
        default=tt_pipeline.DEFAULT_TXT_LEN,
        help="prompt length in tokens (the text encoder is a separate model, so its output is seeded)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the latents and the prompt-embedding stand-in")
    if steps:
        parser.add_argument(
            "--num-steps",
            type=int,
            default=4,
            help=(
                f"flow-match Euler steps, clamped to "
                f"[{tt_inputs.MIN_STEPS}, {tt_inputs.MAX_STEPS}]; changes BOTH the TT loop and the golden"
            ),
        )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="cap EVERY repeated stack (default: every layer the checkpoint has)",
    )
    parser.add_argument("--dual-layers", type=int, default=None, help="cap transformer_blocks (8 full)")
    parser.add_argument("--single-layers", type=int, default=None, help="cap single_transformer_blocks (24 full)")
    parser.add_argument("--tp", type=int, default=DEFAULT_TP, help="tensor-parallel width (mesh is 1 x TP)")
    parser.add_argument(
        "--trace-region-size", type=int, default=DEFAULT_TRACE_REGION_SIZE, help="device trace region, bytes"
    )
    parser.add_argument(
        "--check-pcc",
        action="store_true",
        help="also run the HF golden and print the PCC (needs ~36 GB of host RAM for the float32 reference)",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="where to write the output tensors")
    return parser


def open_mesh(args):
    """Open the 1 x TP mesh with fabric and a trace region. Caller closes it."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, args.tp),
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=args.trace_region_size,
        num_command_queues=1,
    )


def close_mesh(device):
    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def load_model(args):
    """The reference module: the source of the build-time weights, and -- only
    with `--check-pcc` -- of the golden."""
    print("[flux2-demo] loading the reference checkpoint (9.08 B params)", flush=True)
    started = time.time()
    model = tt_reference.load_reference_model()
    print(f"[flux2-demo] reference loaded in {time.time() - started:.1f}s", flush=True)
    return model


def build(args, device, model):
    return tt_pipeline.build_pipeline(
        device,
        model=model,
        layers=args.layers,
        dual_layers=args.dual_layers,
        single_layers=args.single_layers,
        height=args.height,
        width=args.width,
        txt_len=args.txt_len,
    )


def make_inputs(args):
    inputs = tt_inputs.build_inputs(height=args.height, width=args.width, txt_len=args.txt_len, batch=1, seed=args.seed)
    print(f"[flux2-demo] inputs {inputs['meta']}", flush=True)
    return inputs


def report_pcc(golden, actual, label="e2e"):
    """Print the PCC on its own line, and return it. Same helper both demos use."""
    from models.common.utility_functions import comp_pcc

    _, achieved_pcc = comp_pcc(golden, actual.to(torch.float32), 0.95)
    print(f"{label} PCC={achieved_pcc}", flush=True)
    return float(achieved_pcc)


def write_outputs(args, name: str, tensors: dict, summary: dict) -> Path:
    """Save the real output tensors plus a small json summary; return the dir."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, tensor in tensors.items():
        path = out_dir / f"{name}_{key}.pt"
        torch.save(tensor, path)
        print(f"[flux2-demo] wrote {path} {tuple(tensor.shape)} {tensor.dtype}", flush=True)

    summary_path = out_dir / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[flux2-demo] wrote {summary_path}", flush=True)
    return out_dir
