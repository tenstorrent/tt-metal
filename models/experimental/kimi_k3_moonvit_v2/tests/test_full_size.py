# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device, the 4-chip ring (one LoudBox row): full-size (256x256, all 65,536 padded rows real) encoder layers, each on
its own, the full 27-layer encoder and the full model against reference outputs saved by `goldens.py`,
real weights only.

The reference outputs are fp32 (`goldens.py --dtype fp32`, the default): the bf16 reference's own
rounding is larger than the device's error from about layer 9 on (layer 18: PCC 0.983 against fp32,
the device 0.99998). The CPU reference takes ~1.5 min per layer at this length, so it is not
recomputed here: generate it once with `python -m models.experimental.kimi_k3_moonvit_v2.tests.goldens`.
The tests skip when `$MOONVIT_GOLDEN_DIR` is unset, or the saved outputs or the checkpoint are missing.

Every chip holds 16,384 real rows, so besides the PCC over the whole image each chip's quarter
(in window order, as the device splits it) is checked on its own; a wrong chip cannot hide behind
three right ones.

The `_trace` variants are the traced perf tests: they capture the same forward in a trace, check its
first replay (it must equal the untraced run exactly and pass the same checks), then time back-to-back
replays on the saved input. The layer one runs TRACE_LAYER only.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import checkpoint_path
from models.experimental.kimi_k3_moonvit_v2.tests import goldens
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    NOT_WRITTEN,
    RING,
    RING_TRACE,
    Trace,
    assert_same_as_untraced,
    format_quality,
    quality,
    quality_failures,
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
from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Encoder, TT_MoonViT_Vision_Model

H, W, SEED = 256, 256, 0
CHECKPOINT = pytest.mark.parametrize("reference", ["checkpoint"], indirect=True)
# Against the fp32 outputs, over every row and over each chip's quarter. Relative RMSE is RMS error over
# the golden's standard deviation; limits as in common.py (measured: layers <= 1.4%, encoder <= 10.5%).
LAYERS = pytest.mark.parametrize("layer", range(MoonViTConfig().num_hidden_layers), ids=lambda i: f"layer{i}")
TRACE_LAYER = 0
LAYER_PCC = 0.99
LAYER_RELATIVE_RMSE = 0.03
ENCODER_PCC = 0.99
ENCODER_RELATIVE_RMSE = 0.15


@pytest.fixture
def saved():
    """`goldens.load` for this case; skips when the outputs or the checkpoint are missing."""
    if checkpoint_path() is None:
        pytest.skip("set $KIMI_K3_HF_MODEL to a Kimi-K3 checkpoint folder")
    if goldens.golden_dir() is None:
        pytest.skip(f"set ${goldens.GOLDEN_DIR_ENV} to the folder that holds the saved outputs")
    folder = goldens.case_dir(H, W, SEED)
    if not (folder / "meta.json").is_file():
        pytest.skip(f"no saved outputs in {folder}; run goldens.py --grid {H}x{W}")

    def load(name: str) -> torch.Tensor:
        if not (folder / f"{name}.pt").is_file():
            pytest.skip(f"no {name}.pt in {folder}; run goldens.py --grid {H}x{W} (it resumes)")
        return goldens.load(H, W, SEED, name)

    return load


def check_quality(
    golden: torch.Tensor,
    actual: torch.Tensor,
    pcc: float,
    relative_rmse: float | None,
    config,
    sp_factor: int,
    label: str,
    merged: bool = False,
) -> None:
    """`quality` over every row, then over each chip's quarter. Both in raster order; the quarters are
    taken in window order, the order the rows are split across the ring. `merged`: one row per 2x2
    window (merger and projector outputs), already in the order the ring splits them."""
    parts = [("all", golden, actual)]
    if merged:
        golden_w, actual_w = golden, actual
    else:
        golden_w, actual_w = (host_prep.to_window_order(t, H, W, config.merge_kernel_size) for t in (golden, actual))
    rows = golden_w.shape[0] // sp_factor
    for chip in range(sp_factor):
        part = slice(chip * rows, (chip + 1) * rows)
        parts.append((f"chip {chip}", golden_w[part], actual_w[part]))
    failed = []
    for name, g, a in parts:
        q = quality(g, a)
        logger.info(f"{label} {name}: {format_quality(q)}")
        failed += [f"{name} {failure}" for failure in quality_failures(q, pcc, relative_rmse)]
    assert not failed, f"{label}: {failed}"


def to_host(t: ttnn.Tensor, mesh_device, sp_axis: int, config) -> torch.Tensor:
    """Gather the device rows and put them back in raster order."""
    return host_prep.from_window_order(sp_gather(t, mesh_device, sp_axis, H * W), H, W, config.merge_kernel_size)


def layer_forward(mesh_device, sp_axis, num_links, config, reference, saved, layer):
    """`(forward, parallel)`: `forward()` runs `blocks[layer]` on the saved input to that block (the
    patch-embed output for layer 0, else the previous block's saved output), held on device."""
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    tt = TT_Vision_Block(
        config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config(mesh_device)
    )
    tt.load_torch_state_dict(reference.vision_tower.encoder.blocks[layer].state_dict())

    x = saved("patch_embed" if layer == 0 else goldens.block_file(layer - 1)[:-3])
    rope = TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(*host_prep.rope_tables(config, H, W))
    x_window = host_prep.to_window_order(x, H, W, config.merge_kernel_size)
    x_dev = sp_shard(x_window, mesh_device, sp_axis, config.max_patches, ttnn.float32)
    logical_n = logical_length_tensor(mesh_device, H * W)
    return (lambda: tt(x_dev, rope, logical_n)), parallel


def check_layer(saved, out, config, parallel, layer, label=""):
    check_quality(
        saved(goldens.block_file(layer)[:-3]),
        out,
        LAYER_PCC,
        LAYER_RELATIVE_RMSE,
        config,
        parallel.sp_factor,
        f"{label}layer {layer}",
    )


@RING
@CHECKPOINT
@LAYERS
def test_full_size_layer(mesh_device, sp_axis, num_links, config, reference, saved, layer):
    """`blocks[layer]` on the saved input to that block against its saved output. Each layer's own
    error, without the error of the layers before it."""
    forward, parallel = layer_forward(mesh_device, sp_axis, num_links, config, reference, saved, layer)
    check_layer(saved, to_host(forward(), mesh_device, sp_axis, config), config, parallel, layer)


@RING_TRACE
@CHECKPOINT
def test_full_size_layer_trace(mesh_device, sp_axis, num_links, config, reference, saved):
    """`test_full_size_layer` traced, for `blocks[TRACE_LAYER]`."""
    layer = TRACE_LAYER
    forward, parallel = layer_forward(mesh_device, sp_axis, num_links, config, reference, saved, layer)
    out = forward()
    untraced = to_host(out, mesh_device, sp_axis, config)
    ttnn.deallocate(out)

    with Trace(mesh_device, forward) as trace:
        out = to_host(trace.replay(), mesh_device, sp_axis, config)
        assert_same_as_untraced(untraced, out, f"layer {layer}")
        check_layer(saved, out, config, parallel, layer, "traced ")
        trace.time_replays(f"full-size layer {layer}")


def encoder_inputs(mesh_device, sp_axis, num_links, config, reference):
    """`(patch_embed, encoder, inputs, parallel)`: both modules with the checkpoint's weights, and the
    device inputs for the pixels `goldens.py` used, held on device."""
    parallel = resolve_parallel(mesh_device, *ring_parallel(mesh_device, sp_axis, num_links), config)
    ckc = compute_kernel_config(mesh_device)
    tower = reference.vision_tower
    patch_embed = TT_Patch_Embed(config, mesh_device=mesh_device, compute_kernel_config=ckc)
    patch_embed.load_torch_state_dict(tower.patch_embed.state_dict())
    encoder = TT_MoonViT_Encoder(config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=ckc)
    encoder.load_torch_state_dict(tower.encoder.state_dict())

    pixels, grid_thws = random_patches(config, [(1, H, W)], SEED)
    host = host_prep.prepare_host_inputs(config, patch_embed.pos_emb_weight, pixels, grid_thws)
    inputs = {
        "patches": sp_shard(host["patches"], mesh_device, sp_axis, config.max_patches),
        "pos_emb": sp_shard(host["pos_emb"], mesh_device, sp_axis, config.max_patches, ttnn.float32),
        "rope": TT_Rope(config, mesh_device=mesh_device, parallel=parallel).write(host["cos"], host["sin"]),
        "logical_n": logical_length_tensor(mesh_device, host["logical_n"]),
    }
    return patch_embed, encoder, inputs, parallel


def check_encoder(saved, out, config, parallel, label=""):
    check_quality(
        saved("final_layernorm"),
        out,
        ENCODER_PCC,
        ENCODER_RELATIVE_RMSE,
        config,
        parallel.sp_factor,
        f"{label}encoder",
    )


@RING
@CHECKPOINT
def test_full_size_encoder(mesh_device, sp_axis, num_links, config, reference, saved):
    """Patch embed then all 27 blocks and the final norm on the same pixels `goldens.py` used, against
    the saved patch-embed and `final_layernorm` outputs."""
    patch_embed, encoder, inputs, parallel = encoder_inputs(mesh_device, sp_axis, num_links, config, reference)
    x = patch_embed(inputs["patches"], inputs["pos_emb"])
    # Checked first so a wrong encoder input is reported as such, not as an encoder error.
    check_quality(
        saved("patch_embed"),
        to_host(x, mesh_device, sp_axis, config),
        LAYER_PCC,
        LAYER_RELATIVE_RMSE,
        config,
        parallel.sp_factor,
        "patch embed",
    )
    out = encoder(x, inputs["rope"], inputs["logical_n"])
    check_encoder(saved, to_host(out, mesh_device, sp_axis, config), config, parallel)


@RING_TRACE
@CHECKPOINT
def test_full_size_encoder_trace(mesh_device, sp_axis, num_links, config, reference, saved):
    """`test_full_size_encoder` traced: patch embed and the encoder in one trace, checked on the encoder
    output only (`test_full_size_encoder` checks the patch embed)."""
    patch_embed, encoder, inputs, parallel = encoder_inputs(mesh_device, sp_axis, num_links, config, reference)

    def forward():
        x = patch_embed(inputs["patches"], inputs["pos_emb"])
        out = encoder(x, inputs["rope"], inputs["logical_n"])
        ttnn.deallocate(x)
        return out

    out = forward()
    untraced = to_host(out, mesh_device, sp_axis, config)
    ttnn.deallocate(out)

    with Trace(mesh_device, forward) as trace:
        out = to_host(trace.replay(), mesh_device, sp_axis, config)
        assert_same_as_untraced(untraced, out, "encoder")
        check_encoder(saved, out, config, parallel, "traced ")
        trace.time_replays(f"full-size patch embed + {config.num_hidden_layers}-layer encoder")


@RING
@CHECKPOINT
@NOT_WRITTEN
def test_full_size_model(mesh_device, sp_axis, num_links, config, reference, saved):
    """The full model (patch embed, all 27 blocks, final norm, patch merger, projector) on the pixels
    `goldens.py` used, against the saved `mm_projector` output."""
    sequence_parallel, ccl = ring_parallel(mesh_device, sp_axis, num_links)
    tt = TT_MoonViT_Vision_Model(config, mesh_device=mesh_device, sequence_parallel=sequence_parallel, ccl=ccl)
    tt.load_torch_state_dict(reference.state_dict())
    golden = saved("mm_projector")
    pixels, grid_thws = random_patches(config, [(1, H, W)], SEED)
    check_quality(
        golden,
        tt.run(pixels, grid_thws),
        ENCODER_PCC,
        None,
        config,
        tt.parallel.sp_factor,
        "model",
        merged=True,
    )
