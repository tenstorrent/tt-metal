# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One resnet block of FLUX 3 Klein's 4096 VAE decoder per level, as the 2D spatial decode builds it, under its conv
levers in turn: the channel-split big convs, the external halo (same conv math: byte-identical), the GroupNorm
statistics in conv1's epilogue (bf16 affine: bounded error) and that affine in conv2's halo prologue. Every output must
be finite and deterministic."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein import vae_flux3_klein as klein_vae
from ....pipelines.flux3_klein.weights_flux3_klein import resolve_weights_dir
from ....utils import tensor
from ....utils.test import ring_params_8k

SIZE, H_AXIS, W_AXIS = 4096, 1, 0
# level: (up block, channels, full height = width at that block)
LEVELS = {"512ch": (1, 512, 1024), "256ch": (2, 256, 2048), "128ch": (3, 128, 4096)}
# arm: (lever switched on in this arm on top of the previous ones, check against the previous arm: None = exact, else
# max rel L2 error); the first arm is the 2D decode with every lever off
ARMS = {
    "spatial_2d": (None, None),
    "cout_split": ("cout_split", 2e-2),
    "external_halo": ("external_halo", None),
    "fused_norm": ("fused_norm", 1.5e-2),
    "prologue_affine": ("prologue_affine", 1e-2),
}


def _levers(block) -> dict[str, list[tuple[object, str, object]]]:
    """(object, attribute, value as built) per lever of one resnet block of the decoder Klein builds."""
    convs = (block.conv1, block.conv2)
    return {
        "cout_split": [(c.inner, "cout_split", c.inner.cout_split) for c in convs],
        "external_halo": [(c, "_a3", c._a3) for c in convs],
        "fused_norm": [(block, "_a_fused", block._a_fused)],
        "prologue_affine": [(block, "_a_prologue", block._a_prologue)],
    }


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{**ring_params_8k, "l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_resnet_levers(*, mesh_device: ttnn.MeshDevice) -> None:
    ccl = lambda: CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    vae = klein_vae.Flux3KleinVae(
        resolve_weights_dir("distilled"),
        device=mesh_device,
        ccl_manager=ccl(),
        dit_ccl_manager=ccl(),
        dit_sp_axis=H_AXIS,
        sizes=(SIZE,),
    )
    gen = torch.Generator().manual_seed(0)
    inputs = {level: torch.randn(1, hw, hw, c, generator=gen) for level, (_, c, hw) in LEVELS.items()}
    decoder = vae._decoder  # its conv slicing for SIZE is set per instance at construction
    blocks = {level: decoder.up_blocks[up].resnets[1] for level, (up, _, _) in LEVELS.items()}
    levers = {level: _levers(block) for level, block in blocks.items()}
    off = {"cout_split": 1, "external_halo": False, "fused_norm": False, "prologue_affine": False}
    for per_level in levers.values():
        for name, entries in per_level.items():
            for obj, attr, _ in entries:
                setattr(obj, attr, off[name])
    previous: dict[str, torch.Tensor] = {}
    failures = []
    for arm, (lever, max_err) in ARMS.items():
        for level, block in blocks.items():
            for obj, attr, built in levers[level].get(lever, []):
                setattr(obj, attr, built)

            def run(block=block, x=inputs[level]) -> torch.Tensor:
                tt_x = tensor.from_torch(
                    x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, H_AXIS, W_AXIS, None]
                )
                out = block.forward(tt_x)
                result = tensor.to_torch(out, mesh_axes=[None, H_AXIS, W_AXIS, None])
                ttnn.deallocate(out)
                return result

            out = run()
            assert torch.isfinite(out.float()).all(), f"{arm} {level}: non-finite output"
            assert torch.equal(out, run()), f"{arm} {level}: two calls differ"
            if level in previous:
                ref = previous[level].float()
                err = ((out.float() - ref).norm() / ref.norm()).item()
                logger.info(f"{arm} {level}: rel L2 vs the previous arm {err:.3e}")
                if max_err is None and not torch.equal(out, previous[level]):
                    failures.append(f"{arm} {level}: not byte-identical ({err:.3e})")
                elif max_err is not None and err > max_err:
                    failures.append(f"{arm} {level}: rel L2 {err:.3e} > {max_err}")
            previous[level] = out
    assert not failures, "; ".join(failures)
