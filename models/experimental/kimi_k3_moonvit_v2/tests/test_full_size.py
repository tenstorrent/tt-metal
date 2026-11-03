# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device, the 4-chip ring (one LoudBox row): full-size (256x256, all 65,536 padded rows real) encoder layers, each on
its own, and the full 27-layer encoder against reference outputs saved by `goldens.py`, real weights
only.

The reference outputs are fp32 (`goldens.py --dtype fp32`, the default): the bf16 reference's own
rounding is larger than the device's error from about layer 9 on (layer 18: PCC 0.983 against fp32,
the device 0.99998). The CPU reference takes ~1.5 min per layer at this length, so it is not
recomputed here: generate it once with `python -m models.experimental.kimi_k3_moonvit_v2.tests.goldens`.
The tests skip when `$MOONVIT_GOLDEN_DIR` is unset, or the saved outputs or the checkpoint are missing.

Every chip holds 16,384 real rows, so besides the PCC over the whole image each chip's quarter
(in window order, as the device splits it) is checked on its own; a wrong chip cannot hide behind
three right ones.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.tests import goldens
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    RING,
    random_patches,
    ring_parallel,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import (
    MoonViTConfig,
    compute_kernel_config,
    resolve_parallel,
)
from models.experimental.kimi_k3_moonvit_v2.tt.patch_embed import TT_Patch_Embed
from models.experimental.kimi_k3_moonvit_v2.tt.rope import TT_Rope
from models.experimental.kimi_k3_moonvit_v2.tt.vision_attention import logical_length_tensor
from models.experimental.kimi_k3_moonvit_v2.tt.vision_block import TT_Vision_Block
from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Encoder

H, W, SEED = 256, 256, 0
CHECKPOINT = pytest.mark.parametrize("reference", ["checkpoint"], indirect=True)
# Against the fp32 outputs. Not yet measured for every layer or the 27-layer encoder: the residual
# stream reaches |x| ~ 500 before the final norm; set from the measured PCC if 0.99 proves too tight.
LAYER_PCC = 0.99
ENCODER_PCC = 0.99


@pytest.fixture
def saved():
    """`goldens.load` for this case; skips when the outputs or the checkpoint are missing."""
    if checkpoint_path() is None:
        pytest.skip("set $KIMI_K3_HF_MODEL to a Kimi-K3 checkpoint folder")
    if goldens.golden_dir() is None:
        pytest.skip(f"set ${goldens.GOLDEN_DIR_ENV} to the folder that holds the saved outputs")
    if not (goldens.case_dir(H, W, SEED) / "meta.json").is_file():
        pytest.skip(f"no saved outputs in {goldens.case_dir(H, W, SEED)}; run goldens.py --grid {H}x{W}")
    return lambda name: goldens.load(H, W, SEED, name)


def check_pcc(golden: torch.Tensor, actual: torch.Tensor, pcc: float, config, sp_factor: int, label: str) -> None:
    """PCC over every row, then over each chip's quarter. Both in raster order; the quarters are taken
    in window order, the order the rows are split across the ring."""
    results = [("all", *comp_pcc(golden.float(), actual.float(), pcc))]
    golden_w, actual_w = (host_prep.to_window_order(t, H, W, config.merge_kernel_size) for t in (golden, actual))
    rows = golden_w.shape[0] // sp_factor
    for chip in range(sp_factor):
        part = slice(chip * rows, (chip + 1) * rows)
        results.append((f"chip {chip}", *comp_pcc(golden_w[part].float(), actual_w[part].float(), pcc)))
    logger.info(f"{label}: " + ", ".join(f"{name} PCC {value}" for name, _, value in results))
    failed = [f"{name} PCC {value}" for name, passing, value in results if not passing]
    assert not failed, f"{label} below {pcc}: {failed}"


@RING
@CHECKPOINT
@pytest.mark.parametrize("layer", range(MoonViTConfig().num_hidden_layers), ids=lambda i: f"layer{i}")
def test_full_size_layer(mesh_device, sp_axis, num_links, config, reference, saved, layer):
    """`blocks[layer]` on the saved input to that block (the patch-embed output for layer 0, else the
    previous block's saved output) against its saved output. Each layer's own error, without the
    error of the layers before it."""
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    tt = TT_Vision_Block(
        config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config(mesh_device)
    )
    tt.load_torch_state_dict(reference.vision_tower.encoder.blocks[layer].state_dict())

    x = saved("patch_embed" if layer == 0 else goldens.block_file(layer - 1)[:-3])
    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(*host_prep.rope_tables(config, H, W))
    x_window = host_prep.to_window_order(x, H, W, config.merge_kernel_size)
    logical_n = logical_length_tensor(mesh_device, H * W)
    out = tt(sp_shard(x_window, mesh_device, sp_axis, config.max_patches, ttnn.float32), rope, logical_n)
    out = host_prep.from_window_order(sp_gather(out, mesh_device, sp_axis, H * W), H, W, config.merge_kernel_size)
    check_pcc(saved(goldens.block_file(layer)[:-3]), out, LAYER_PCC, config, parallel.sp_factor, f"layer {layer}")


@RING
@CHECKPOINT
def test_full_size_encoder(mesh_device, sp_axis, num_links, config, reference, saved):
    """Patch embed then all 27 blocks and the final norm on the same pixels `goldens.py` used, against
    the saved patch-embed and `final_layernorm` outputs."""
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    ckc = compute_kernel_config(mesh_device)
    tower = reference.vision_tower
    patch_embed = TT_Patch_Embed(config, mesh_device=mesh_device, compute_kernel_config=ckc)
    patch_embed.load_torch_state_dict(tower.patch_embed.state_dict())
    encoder = TT_MoonViT_Encoder(config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=ckc)
    encoder.load_torch_state_dict(tower.encoder.state_dict())

    pixels, grid_thws = random_patches(config, [(1, H, W)], SEED)
    host = host_prep.prepare_host_inputs(config, patch_embed.pos_emb_weight, pixels, grid_thws)
    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(host["cos"], host["sin"])
    patches = sp_shard(host["patches"], mesh_device, sp_axis, config.max_patches)
    pos_emb = sp_shard(host["pos_emb"], mesh_device, sp_axis, config.max_patches, ttnn.float32)

    def to_host(t):
        return host_prep.from_window_order(sp_gather(t, mesh_device, sp_axis, H * W), H, W, config.merge_kernel_size)

    x = patch_embed(patches, pos_emb)
    # Checked first so a wrong encoder input is reported as such, not as an encoder error.
    check_pcc(saved("patch_embed"), to_host(x), LAYER_PCC, config, parallel.sp_factor, "patch embed")
    out = encoder(x, rope, logical_length_tensor(mesh_device, host["logical_n"]))
    check_pcc(saved("final_layernorm"), to_host(out), ENCODER_PCC, config, parallel.sp_factor, "encoder")
