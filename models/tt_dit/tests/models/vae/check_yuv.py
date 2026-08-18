"""Compare the on-device YUV 4:2:0 export against the float path converted on host."""

import os
from pathlib import Path

import numpy as np
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


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        config = decoder_config(CHECKPOINT)
        latent = torch.randn(1, config["in_channels"], 4, 34, 60, generator=torch.Generator().manual_seed(3))
        ccl = CCLManager(mesh, num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 2)), topology=ttnn.Topology.Ring)
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
        print(f"supports_yuv = {dec.supports_yuv}", flush=True)

        rgb = dec.forward(latent, output_type="float", seed=0)
        print(
            f"[float] {tuple(rgb.shape)} {rgb.dtype} range=({float(rgb.min()):.3f},{float(rgb.max()):.3f})", flush=True
        )

        planar = dec.forward(latent, output_type="yuv", seed=0)
        print(f"[yuv  ] {tuple(planar.shape)} {planar.dtype}", flush=True)

        # BT.601 limited range, matching the kernel's default coefficients.
        f = rgb.float().add(1.0).mul(0.5).clamp(0, 1)  # (1,3,T,H,W) -> [0,1]
        r, g, b = f[0, 0], f[0, 1], f[0, 2]
        y_ref = (16.0 + 65.481 * r + 128.553 * g + 24.966 * b).round().clamp(0, 255)
        t, h, w = y_ref.shape
        y_got = torch.from_numpy(np.ascontiguousarray(planar[:, :h, :])).float()
        d = (y_got - y_ref).abs()
        print(f"[Y] shape ref={tuple(y_ref.shape)} got={tuple(y_got.shape)}", flush=True)
        print(
            f"[Y] max|diff|={float(d.max()):.1f}  mean|diff|={float(d.mean()):.3f}  "
            f"pct within 2 = {100.0 * float((d <= 2).float().mean()):.2f}%",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
