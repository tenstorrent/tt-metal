# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t363: full 4x8 LTX-2.5 conv VAE decode, 1088x1920/145f, old vs new conv3d blockings in one process.

Real 2.5 conv VAE weights ($VAE_CKPT), latent $AB_LATENT. Arm A uses the _BLOCKINGS table as built; arm B
patches the entries in $T363_NEW_BLOCKINGS (a Python dict literal, key -> blocking) before building its
decoder. Per arm: one warm-up yuv decode, $T363_REPS timed yuv decodes (wall, synchronized), one float
decode kept on host. Then: bit-identity, PCC and PSNR (range 2) of B vs A.
"""

import ast
import json
import os
import statistics
import time

import pytest
import torch
from safetensors.torch import load_file

import ttnn

NF, H, W = 145, 1088, 1920


def _latent():
    lt, lh, lw = (NF - 1) // 8 + 1, H // 32, W // 32
    return torch.load(os.environ["AB_LATENT"], map_location="cpu")["video"].float().reshape(1, lt, lh, lw, 128).permute(
        0, 4, 1, 2, 3
    )


def _decoder_blocks(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    cfg = json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})
    return cfg.get("decoder_blocks")


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_blk_ab_4x8(mesh_device, device_params):
    from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder, vae_key_map
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import _LTX_PROD_DECODER_BLOCKS
    from models.tt_dit.utils import conv3d

    assert tuple(mesh_device.shape) == (4, 8)
    ckpt = os.environ["VAE_CKPT"]
    new = ast.literal_eval(os.environ["T363_NEW_BLOCKINGS"])
    reps = int(os.environ.get("T363_REPS", "3"))
    blocks = _decoder_blocks(ckpt) or _LTX_PROD_DECODER_BLOCKS
    print(f"AB decoder_blocks {'metadata' if _decoder_blocks(ckpt) else 'prod default'}: {blocks}")
    raw = load_file(ckpt)
    state = {short: raw[k] for k, short in vae_key_map(raw, "decoder").items()}
    del raw
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=4, mesh_axis=0),
        width_parallel=ParallelFactor(factor=8, mesh_axis=1),
    )
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2)
    lat = _latent()

    def arm(name):
        dec = LTXVideoDecoder(
            decoder_blocks=blocks,
            in_channels=128,
            out_channels=3,
            patch_size=4,
            base_channels=128,
            causal=False,
            num_frames=NF,
            height=H,
            width=W,
            mesh_device=mesh_device,
            parallel_config=pc,
            ccl_manager=ccl,
        )
        dec.load_torch_state_dict(state)
        dec(lat, output_type="yuv")
        ttnn.synchronize_device(mesh_device)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            dec(lat, output_type="yuv")
            ttnn.synchronize_device(mesh_device)
            ts.append(time.perf_counter() - t0)
        out = dec(lat, output_type="float")
        dec.deallocate_weights()
        print(f"AB arm={name} yuv_s={[round(t, 4) for t in ts]} median={statistics.median(ts):.4f}", flush=True)
        return statistics.median(ts), out.float()

    old = {k: conv3d._BLOCKINGS[k] for k in new}
    print(f"AB old={old}\nAB new={new}")
    ta, a = arm("A_old")
    conv3d._BLOCKINGS.update(new)
    tb, b = arm("B_new")
    conv3d._BLOCKINGS.update(old)

    identical = torch.equal(a, b)
    # float64 sums over 1M-element chunks: the outputs are 0.9 G elements each.
    n, sa, sb, saa, sbb, sab, sdd, maxabs = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for ca, cb in zip(a.flatten().split(1 << 20), b.flatten().split(1 << 20)):
        ca, cb = ca.double(), cb.double()
        n += ca.numel()
        sa, sb = sa + ca.sum().item(), sb + cb.sum().item()
        saa, sbb, sab = saa + (ca * ca).sum().item(), sbb + (cb * cb).sum().item(), sab + (ca * cb).sum().item()
        sdd += ((ca - cb) ** 2).sum().item()
        maxabs = max(maxabs, (ca - cb).abs().max().item())
    cov = sab / n - (sa / n) * (sb / n)
    pcc = cov / ((saa / n - (sa / n) ** 2) ** 0.5 * (sbb / n - (sb / n) ** 2) ** 0.5)
    mse = sdd / n
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()
    print(
        f"AB RESULT median_old_s={ta:.4f} median_new_s={tb:.4f} new/old={tb / ta:.4f} "
        f"identical={identical} pcc={pcc:.6f} psnr_db={psnr:.2f} maxabs={maxabs:.4g}"
    )
    assert pcc >= 0.999 and psnr >= 45
