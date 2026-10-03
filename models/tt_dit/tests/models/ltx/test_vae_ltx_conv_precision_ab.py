# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decoder conv3d precision A/B (LTX_VAE_CONV_FIDELITY) on a 2x4 submesh.

Real checkpoint weights and a real latent at 544x960/145f (each chip's shard matches 1080p on 4x8). All
arms run in one job, in AB_ARMS order; each arm builds a fresh decoder, does 1 warmup and
3 timed yuv decodes, and saves its output to $AB_OUT_DIR/yuv_<arm>.pt right away. Quality is scored off
device (compare_conv_precision_ab.py). No weight cache is read or written.
AB_CHECKPOINT: LTX safetensors with the conv VAE (2.3 monolith). AB_LATENT: saved stage-2 latent.
"""

import gc
import json
import os
import time

import pytest
import torch
from safetensors import safe_open

import ttnn

NF, H, W = 145, 544, 960
TIMED_DECODES = 3
ARMS = {"base": "", "hifi2": "HiFi2", "lofi": "LoFi"}  # base = production default (HiFi2 for bf16)


def _latent():
    lt, lh, lw = (NF - 1) // 8 + 1, H // 32, W // 32
    # Saved 2.5 stage-2 latent (1080p: T=19, H=34, W=60, packed tokens), center crop to 17x30.
    v = torch.load(os.environ["AB_LATENT"], map_location="cpu")["video"].float().reshape(1, 19, 34, 60, 128)
    h0, w0 = (34 - lh) // 2, (60 - lw) // 2
    return v[:, :lt, h0 : h0 + lh, w0 : w0 + lw].permute(0, 4, 1, 2, 3).contiguous()


def _checkpoint_decoder(path):
    """(vae config, decoder state) read straight from the checkpoint, VAE keys only."""
    from models.tt_dit.models.vae.vae_ltx import _strip_vae_prefix

    state = {}
    with safe_open(path, framework="pt") as f:
        cfg = json.loads(f.metadata()["config"])["vae"]
        for k in f.keys():
            short = _strip_vae_prefix(k, "vae.decoder.", "decoder.")
            if short is not None:
                state[short] = f.get_tensor(k)
                continue
            pcs = _strip_vae_prefix(k, "vae.per_channel_statistics.", "per_channel_statistics.")
            if pcs in ("mean-of-means", "std-of-means"):
                state[f"per_channel_statistics.{pcs}"] = f.get_tensor(k)
    return cfg, state


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_conv_precision_ab(mesh_device, device_params, monkeypatch):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.conv3d import _walk_conv3d_modules

    cfg, state = _checkpoint_decoder(os.environ["AB_CHECKPOINT"])
    lat = _latent()
    out_dir = os.environ["AB_OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=2, mesh_axis=0),
        width_parallel=ParallelFactor(factor=4, mesh_axis=1),
    )
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2)
    failed = []
    for arm in os.environ.get("AB_ARMS", ",".join(ARMS)).split(","):
        monkeypatch.setenv("LTX_VAE_CONV_FIDELITY", ARMS[arm])
        try:
            dec = LTXVideoDecoder(
                decoder_blocks=cfg["decoder_blocks"],
                causal=cfg.get("causal_decoder", False),
                base_channels=cfg.get("decoder_base_channels", 128),
                mesh_device=mesh_device,
                parallel_config=pc,
                ccl_manager=ccl,
                num_frames=NF,
                height=H,
                width=W,
            )
            convs = list(_walk_conv3d_modules(dec.up_blocks))
            n_lofi = sum(1 for c in convs if c.compute_kernel_config.math_fidelity == ttnn.MathFidelity.LoFi)
            print(f"AB arm={arm} up_convs={len(convs)} lofi={n_lofi}", flush=True)
            dec.load_torch_state_dict(state)
            out = dec(lat, output_type="yuv")  # warmup: kernel compile + program cache
            times = []
            for _ in range(TIMED_DECODES):
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                out = dec(lat, output_type="yuv")
                ttnn.synchronize_device(mesh_device)
                times.append(time.perf_counter() - t0)
            print(f"AB arm={arm} decode_s={' '.join(f'{t:.4f}' for t in times)} min={min(times):.4f}", flush=True)
            torch.save(torch.as_tensor(out), os.path.join(out_dir, f"yuv_{arm}.pt"))
        except Exception as e:  # keep the arms that work; report the rest
            print(f"AB arm={arm} FAILED {type(e).__name__}: {e}", flush=True)
            failed.append(arm)
        dec = out = None
        gc.collect()
    assert not failed, f"arms failed: {failed}"
