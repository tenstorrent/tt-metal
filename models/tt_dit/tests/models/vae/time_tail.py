"""Time the decode tail: host unpatchify+pull (float) against on-device YUV 4:2:0 (yuv).

Both paths run the identical device graph; only the tail differs, so the delta is the transfer.
Run with DIFFVAE_STAGE_TIMING=1 to also get the tail timer isolated from the rest of the decode.
"""

import os
import time
from pathlib import Path

import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEDecoder, decoder_config
from models.tt_dit.parallel.manager import CCLManager

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)
ITERS = int(os.environ.get("ITERS", 2))
T_LAT = int(os.environ.get("DIFFVAE_LATENT_T", 19))


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        config = decoder_config(CHECKPOINT)
        latent = torch.randn(1, config["in_channels"], T_LAT, 34, 60, generator=torch.Generator().manual_seed(3))
        ring = os.environ.get("DIFFVAE_TOPOLOGY", "ring").lower() == "ring"
        ccl = CCLManager(
            mesh,
            num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 2)),
            topology=ttnn.Topology.Ring if ring else ttnn.Topology.Linear,
        )
        dec = DiffVAEDecoder(
            config,
            mesh_device=mesh,
            ccl_manager=ccl,
            stage5_na3d_backend="op_sp_w_sharded",
            stage5_sp_axis=1,
            stage5_tp_axis=0,
            stages_na3d_backend="op_sp_w_sharded",
            stages_sp_axis=1,
            stages_tp_axis=0,
        )
        dec.load_checkpoint(CHECKPOINT)
        print(f"[setup] {8 * T_LAT - 7} frames, ring={ring}", flush=True)

        dec.forward(latent, output_type="float", seed=0)  # warm
        ttnn.synchronize_device(mesh)

        for kind in ("float", "yuv"):
            for i in range(ITERS):
                t0 = time.perf_counter()
                out = dec.forward(latent, output_type=kind, seed=0)
                ttnn.synchronize_device(mesh)
                dt = (time.perf_counter() - t0) * 1000
                print(f"[{kind:5s} {i}] {dt:9.1f} ms  out={tuple(out.shape)}", flush=True)
                del out
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
