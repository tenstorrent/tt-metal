#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""VAE decoder t_chunk timing sweep for Cosmos3-I2V.

Loads a saved post-denoise latent and runs the TT decoder at multiple
t_chunk sizes on the native-cfg 2x8 submesh (production configuration),
printing total decode wall time for each.

Usage:
    TT_DIT_VAE_TIMING=1 python -m models.tt_dit.experimental.cosmos3_i2v.demo.vae_tchunk_sweep \\
        --latent /tmp/pre_l3_latent.pt.latent.pt \\
        --height 720 --width 1280 --frames 189 \\
        --t-chunks 4 1 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--latent", type=Path, required=True, help="Saved latent .pt (BCTHW, post-denoise).")
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--num-links", type=int, default=None)
    p.add_argument("--hf-repo", default=None)
    p.add_argument(
        "--t-chunks",
        nargs="+",
        type=int,
        default=[4, 1],
        help="t_chunk_size values to test.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    import ttnn
    from models.tt_dit.experimental.cosmos3_i2v.demo.generate import close_mesh, open_mesh
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO
    from models.tt_dit.experimental.cosmos3_i2v.tokenizer.vae_cosmos3 import Cosmos3VAEDecoderAdapter
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager

    args = parse_args(argv)
    hf_repo = args.hf_repo or HF_REPO

    latent = torch.load(str(args.latent), map_location="cpu")
    print(f"[sweep] latent shape={tuple(latent.shape)} dtype={latent.dtype}", flush=True)

    # Open full 4x8 mesh and split into two 2x8 submeshes — mirrors native-cfg exactly.
    full_mesh = open_mesh((4, 8))
    try:
        submeshes = full_mesh.create_submeshes(ttnn.MeshShape(2, 8))
        submesh = submeshes[0]
        mesh_shape = tuple(submesh.shape)  # (2, 8)
        print(f"[sweep] submesh={mesh_shape}", flush=True)

        # tp_axis = 1 (size 8), sp_axis = 0 (size 2) — matches native-cfg VAE config.
        tp_axis, sp_axis = 1, 0
        num_links = args.num_links or (2 if ttnn.device.is_blackhole() else 1)
        parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
            width_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        )
        ccl_manager = CCLManager(mesh_device=submesh, num_links=num_links, topology=ttnn.Topology.Linear)

        results = []
        for t_chunk in args.t_chunks:
            print(f"\n[sweep] === t_chunk_size={t_chunk} ===", flush=True)
            adapter = Cosmos3VAEDecoderAdapter(
                checkpoint_name=hf_repo,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                vae_t_chunk_size=t_chunk,
                vae_dtype=ttnn.bfloat16,
            )
            t0 = time.perf_counter()
            _ = adapter.decode(latent.clone(), output_type="pt")
            dt = time.perf_counter() - t0
            print(f"[sweep] t_chunk={t_chunk} decode_wall={dt:.1f}s", flush=True)
            results.append((t_chunk, dt))
            del adapter

        print("\n[sweep] === summary ===", flush=True)
        for t_chunk, dt in results:
            print(f"  t_chunk={t_chunk}  decode={dt:.1f}s", flush=True)
    finally:
        close_mesh(full_mesh)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
