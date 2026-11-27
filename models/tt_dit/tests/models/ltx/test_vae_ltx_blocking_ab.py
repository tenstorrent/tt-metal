# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Conv VAE decode A/B of the 2x4 conv3d blockings for 1080p-portrait (1088x576).

For each shape in AB_SHAPES ("NFxHxW,..."; AB_ARMS picks the arms, a skipped arm loads its saved
AB_OUT_DIR output), arm "base" decodes with the table minus this shape's
per-device entries (what ran before they were added), arm "new" with the table as is, on the same
latent and weights. Prints AB timing lines and the yuv agreement; the arms must agree to PSNR > 45 dB.
"""

import math
import os
import time

import pytest
import torch

import ttnn

TIMED_DECODES = 3
DEFAULT_SHAPES = "121x1088x576,241x1088x576"


def _shapes():
    out = []
    for s in os.environ.get("AB_SHAPES", DEFAULT_SHAPES).split(","):
        nf, h, w = (int(x) for x in s.split("x"))
        out.append((nf, h, w))
    return out


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.double() - b.double()).pow(2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_vae_ltx_blocking_ab(mesh_device, device_params):
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )
    from models.tt_dit.utils import conv3d

    torch.manual_seed(42)
    tdec = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=_require_diffusers_ltx_vae(),
    ).eval()
    state = tdec.state_dict()
    state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    state["per_channel_statistics.std-of-means"] = torch.ones(128)
    tt_state = _diffusers_decoder_state_to_tt(state)
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=2, mesh_axis=0),
        width_parallel=ParallelFactor(factor=4, mesh_axis=1),
    )
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2)
    full_table = dict(conv3d._BLOCKINGS)
    out_dir = os.environ.get("AB_OUT_DIR")

    failures = []
    for nf, h, w in _shapes():
        dims = vae_ltx._compute_ltx_decoder_dims(
            decoder_blocks=_LTX_PROD_DECODER_BLOCKS, num_frames=nf, height=h, width=w, h_factor=2, w_factor=4
        )
        hw = {(d.H, d.W) for d in dims}
        own = [k for k in full_table if k[:2] == (2, 4) and (k[6], k[7]) in hw]
        torch.manual_seed(0)
        lat = torch.randn(1, 128, (nf - 1) // 8 + 1, h // 32, w // 32)
        outs = {}
        tag = f"{nf}x{h}x{w}"
        for arm in os.environ.get("AB_ARMS", "base,new").split(","):
            conv3d._BLOCKINGS.clear()
            conv3d._BLOCKINGS.update(full_table)
            if arm == "base":
                for k in own:
                    del conv3d._BLOCKINGS[k]
            conv3d._BLOCKINGS_BY_SPATIAL = None
            dec = vae_ltx.LTXVideoDecoder(
                decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
                in_channels=128,
                out_channels=3,
                patch_size=4,
                base_channels=128,
                causal=False,
                num_frames=nf,
                height=h,
                width=w,
                mesh_device=mesh_device,
                parallel_config=pc,
                ccl_manager=ccl,
            )
            dec.load_torch_state_dict(tt_state)
            out = dec(lat, output_type="yuv")  # warmup: kernel compile + program cache
            times = []
            for _ in range(TIMED_DECODES):
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                out = dec(lat, output_type="yuv")
                ttnn.synchronize_device(mesh_device)
                times.append(time.perf_counter() - t0)
            print(
                f"AB shape={tag} arm={arm} entries_dropped={len(own) if arm == 'base' else 0} "
                f"decode_s={' '.join(f'{t:.4f}' for t in times)} min={min(times):.4f}",
                flush=True,
            )
            outs[arm] = torch.as_tensor(out).flatten().to(torch.uint8)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                torch.save(outs[arm], os.path.join(out_dir, f"yuv_{tag}_{arm}.pt"))
            dec.deallocate_weights()
            del dec
        for arm in ("base", "new"):
            if arm not in outs:  # arm skipped via AB_ARMS: compare against its saved output
                outs[arm] = torch.load(os.path.join(out_dir, f"yuv_{tag}_{arm}.pt"))
        a, b = outs["base"], outs["new"]
        exact = torch.equal(a, b)
        psnr = _psnr(a, b)
        maxdiff = (a.int() - b.int()).abs().max().item()
        print(f"AB shape={nf}x{h}x{w} bit_exact={exact} psnr_db={psnr:.2f} max_abs_diff={maxdiff}", flush=True)
        if not (exact or psnr > 45):
            failures.append(f"{nf}x{h}x{w}: psnr {psnr:.2f}")
    conv3d._BLOCKINGS.clear()
    conv3d._BLOCKINGS.update(full_table)
    conv3d._BLOCKINGS_BY_SPATIAL = None
    assert not failures, failures
