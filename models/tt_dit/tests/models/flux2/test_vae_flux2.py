# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import statistics
import time

import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality

_LAYERS_PER_BLOCK = 1  # pruned from pretrained (2) to keep host run fast

# Encoder perf: one compile warmup + a few timed end-to-end runs (no trace capture).
_PERF_WARMUP_RUNS = 1
_PERF_NUM_RUNS = 3


def _encoder_parallel_axes(mesh_device: ttnn.MeshDevice) -> tuple[int, int]:
    """Match pipeline / perf tests: TP on the size-8 axis, H-shard on the size-4 axis."""
    if mesh_device.shape[1] >= mesh_device.shape[0]:
        return 1, 0
    return 0, 1


def _load_pruned_torch_model() -> reference.AutoencoderKLFlux2:
    """Load the pretrained VAE and truncate each up_block to _LAYERS_PER_BLOCK resnets.

    The mid_block is left intact — it already uses num_layers=1 (2 resnets, 1 attention)
    matching TT's VaeMidBlock default, regardless of layers_per_block.
    """
    model = reference.AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    assert isinstance(model, reference.AutoencoderKLFlux2)
    model.eval()
    keep = _LAYERS_PER_BLOCK + 1
    for up_block in model.decoder.up_blocks:
        up_block.resnets = torch.nn.ModuleList(list(up_block.resnets)[:keep])
    return model


def _load_pruned_torch_encoder_model() -> reference.AutoencoderKLFlux2:
    """Load the pretrained VAE and truncate each encoder down_block to _LAYERS_PER_BLOCK resnets.

    The mid_block is left intact (num_layers=1) to match TT's VaeMidBlock default.
    """
    model = reference.AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    assert isinstance(model, reference.AutoencoderKLFlux2)
    model.eval()
    for down_block in model.encoder.down_blocks:
        down_block.resnets = torch.nn.ModuleList(list(down_block.resnets)[:_LAYERS_PER_BLOCK])
    return model


def _torch_encode_reference(torch_model: reference.AutoencoderKLFlux2, image: torch.Tensor) -> torch.Tensor:
    """Run the torch encoder + .mode() + space-to-depth patchify + batch-norm normalize.

    image: [B, 3, H, W]. Returns [B, C*p^2, H/16, W/16] channel-first, matching the TT encoder's
    patchify output permuted to NCHW.
    """
    from ....pipelines.flux2.pipeline_flux2 import _patchify_latents

    with torch.no_grad():
        encoded = torch_model.encode(image).latent_dist.mode()  # [B, C, H/8, W/8]

    encoded = _patchify_latents(encoded)  # [B, C*p^2, H/16, W/16]
    mean = torch_model.bn.running_mean.view(1, -1, 1, 1).to(dtype=encoded.dtype)
    std = torch.sqrt(torch_model.bn.running_var.view(1, -1, 1, 1) + torch_model.config.batch_norm_eps).to(
        dtype=encoded.dtype
    )
    return (encoded - mean) / std


def _torch_decode_reference(torch_model: reference.AutoencoderKLFlux2, inp: torch.Tensor) -> torch.Tensor:
    """Apply BN inv-normalize, unpatchify, then run the torch decoder.

    inp: [B, C*p^2, H_t, W_t] patchified latent.
    Returns [B, 3, H, W] (NCHW) for comparison with the TT output permuted to NCHW.
    """
    p = 2  # _PATCH_SIZE
    z_channels = torch_model.config.latent_channels
    bn_eps = torch_model.bn.eps
    s = (torch_model.bn.running_var + bn_eps).sqrt()  # [C*p^2]
    m = torch_model.bn.running_mean  # [C*p^2]
    # Matches TT _inv_normalize: z * sqrt(var + eps) + mean
    inp_norm = inp * s.view(1, -1, 1, 1) + m.view(1, -1, 1, 1)

    # Unpatchify: [B, C*p^2, H_t, W_t] -> [B, C, H_lat, W_lat]
    B, _, h_t, w_t = inp_norm.shape
    z = inp_norm.reshape(B, z_channels, p, p, h_t, w_t)
    z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
    z = z.reshape(B, z_channels, h_t * p, w_t * p)

    with torch.no_grad():
        return torch_model.decode(z).sample  # [B, 3, H, W]


def prep_data(
    vae_parallel_config: Flux2VaeParallelConfig,
    inp: torch.Tensor,
    mesh_device: ttnn.Device,
    tt_model: Flux2VaeDecoder,
) -> ttnn.Tensor:
    # inp: [B, C*p^2, H_t, W_t] patchified latent in torch.
    # Convert to patchified token format [B, H_t*W_t, C*p^2].
    inp_flat = inp.permute(0, 2, 3, 1).flatten(1, 2)

    if vae_parallel_config.h_parallel is not None:
        tt_latents = tensor.from_torch(
            inp_flat,
            device=mesh_device,
            mesh_axes=(None, vae_parallel_config.h_parallel.mesh_axis, None),
        )
    else:
        tt_latents = tensor.from_torch(inp_flat, device=mesh_device)

    p = Flux2VaeDecoder._PATCH_SIZE
    height_for_unp = inp.shape[2] * p
    width_for_unp = inp.shape[3] * p

    tt_latents = tt_model.preprocess_and_unpatchify(
        tt_latents,
        height=height_for_unp,
        width=width_for_unp,
    )

    # W-sharding can only be applied after unpatchify because H and W are interleaved
    # in the patchified token dimension.
    if vae_parallel_config.w_parallel is not None:
        tt_latents = ttnn.mesh_partition(
            tt_latents,
            dim=2,
            cluster_axis=vae_parallel_config.w_parallel.mesh_axis,
            memory_config=tt_latents.memory_config(),
        )

    return tt_latents


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 8), id="1x8"),
        pytest.param((4, 8), id="4x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [(1, 1024, 1024)],
    ids=["1024"],
)
def test_vae_flux2_decoder(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    tp_axis = 0
    h_axis = 1 - tp_axis
    w_axis = None

    torch_model = _load_pruned_torch_model()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=w_axis)

    z_channels = torch_model.config.latent_channels
    patch_size = 2
    vae_scale_factor = 8

    tt_model = Flux2VaeDecoder(
        out_channels=torch_model.config.out_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=_LAYERS_PER_BLOCK,
        z_channels=z_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    f = vae_scale_factor * patch_size
    inp = torch.randn(batch_size, z_channels * patch_size**2, height // f, width // f)

    torch_output = _torch_decode_reference(torch_model, inp)  # [B, 3, H, W]

    tt_inp = prep_data(vae_parallel_config, inp, mesh_device, tt_model=tt_model)
    tt_out = tt_model.forward(tt_inp)
    ttnn.synchronize_device(mesh_device)

    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)  # [B, 3, H, W]
    assert_quality(torch_output, tt_out_torch, pcc=0.9978, relative_rmse=0.034)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 8), id="1x8"),
        pytest.param((4, 8), id="4x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [(1, 512, 512)],
    ids=["512"],
)
def test_vae_flux2_encoder(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    tp_axis, h_axis = _encoder_parallel_axes(mesh_device)
    w_axis = None

    torch_model = _load_pruned_torch_encoder_model()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=w_axis)

    z_channels = torch_model.config.latent_channels
    vae_scale_factor = 8

    tt_model = Flux2VaeEncoder(
        in_channels=torch_model.config.in_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=_LAYERS_PER_BLOCK,
        z_channels=z_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    image = torch.randn(batch_size, torch_model.config.in_channels, height, width)

    torch_output = _torch_encode_reference(torch_model, image)  # [B, C*p^2, H/16, W/16]

    h_mesh_axis = vae_parallel_config.h_parallel.mesh_axis if vae_parallel_config.h_parallel is not None else None
    mesh_axes = [None, h_mesh_axis, None, None] if h_mesh_axis is not None else None

    tt_image = tensor.from_torch(
        image.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=mesh_axes,
    )

    patchified = tt_model.encode_and_patchify(
        tt_image, height=height // vae_scale_factor, width=width // vae_scale_factor
    )
    ttnn.synchronize_device(mesh_device)

    tt_out_torch = tensor.to_torch(patchified, mesh_axes=mesh_axes, composer_device=mesh_device).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.99, relative_rmse=0.06)


def _gather_nhwc_to_nchw(
    t: ttnn.Tensor,
    *,
    mesh_device: ttnn.Device,
    h_mesh_axis: int | None,
    tp_mesh_axis: int | None,
    channel_sharded: bool,
) -> torch.Tensor:
    """Gather a (possibly H-sharded and/or channel-TP-sharded) NHWC device tensor to host NCHW."""
    mesh_axes = [
        None,
        h_mesh_axis,
        None,
        tp_mesh_axis if channel_sharded else None,
    ]
    if all(a is None for a in mesh_axes):
        mesh_axes = None
    th = tensor.to_torch(t, mesh_axes=mesh_axes, composer_device=mesh_device)
    return th.permute(0, 3, 1, 2)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 1), id="1x1")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [(1, 512, 512)],
    ids=["512"],
)
def test_vae_flux2_encoder_components(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    """Per-component PCC test: run the TT encoder stage-by-stage (feeding each stage's output
    into the next, exactly as forward does) and assert PCC >= 0.99 at every component boundary
    against the torch reference activations.
    """
    torch.manual_seed(0)

    tp_axis, h_axis = _encoder_parallel_axes(mesh_device)
    w_axis = None

    torch_model = _load_pruned_torch_encoder_model()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=w_axis)

    z_channels = torch_model.config.latent_channels
    vae_scale_factor = 8

    tt_model = Flux2VaeEncoder(
        in_channels=torch_model.config.in_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=_LAYERS_PER_BLOCK,
        z_channels=z_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    image = torch.randn(batch_size, torch_model.config.in_channels, height, width)

    # --- torch reference activations, stage by stage ---
    enc = torch_model.encoder
    torch_stages: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        h = enc.conv_in(image)
        torch_stages["conv_in"] = h
        for i, down_block in enumerate(enc.down_blocks):
            h = down_block(h)
            torch_stages[f"down_block{i}"] = h
        h = enc.mid_block(h)
        torch_stages["mid_block"] = h
        h = enc.conv_norm_out(h)
        h = enc.conv_act(h)
        torch_stages["norm_silu"] = h
        h = enc.conv_out(h)
        torch_stages["conv_out"] = h
        q = torch_model.quant_conv(h)
        torch_stages["quant_conv"] = q

    torch_stages["patchify"] = _torch_encode_reference(torch_model, image)

    # --- TT activations, stage by stage (chained, mirroring forward) ---
    ctx = tt_model._ctx
    tp_mesh_axis = ctx.tp_axis
    h_mesh_axis = ctx.h_mesh_axis
    tp_size = mesh_device.shape[tp_mesh_axis] if tp_mesh_axis is not None else 1

    up_mesh_axes = [None, h_mesh_axis, None, None] if h_mesh_axis is not None else None
    tt_image = tensor.from_torch(
        image.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=up_mesh_axes,
    )

    def check(name: str, tt_tensor: ttnn.Tensor, *, channel_sharded: bool, pcc: float = 0.99) -> None:
        tt_nchw = _gather_nhwc_to_nchw(
            tt_tensor,
            mesh_device=mesh_device,
            h_mesh_axis=h_mesh_axis,
            tp_mesh_axis=tp_mesh_axis,
            channel_sharded=channel_sharded,
        )
        assert_quality(torch_stages[name], tt_nchw, pcc=pcc)

    x = tt_model.conv_in.forward(tt_image)
    check("conv_in", x, channel_sharded=True)

    for i, down_block in enumerate(tt_model.down_blocks):
        x = down_block.forward(x)
        check(f"down_block{i}", x, channel_sharded=True)

    x = tt_model.mid_block.forward(x)
    check("mid_block", x, channel_sharded=True)

    x = tt_model.conv_norm_out.forward(x)
    x = ttnn.silu(x)
    check("norm_silu", x, channel_sharded=True)

    if ctx.ccl_manager is not None and tp_mesh_axis is not None and tp_size > 1:
        x = ctx.ccl_manager.all_gather(x, dim=-1, mesh_axis=tp_mesh_axis, use_hyperparams=True)
    x = tt_model.conv_out.forward(x)
    check("conv_out", x, channel_sharded=False)

    x = tt_model.quant_conv.forward(x)
    check("quant_conv", x, channel_sharded=False)

    patchified = tt_model.patchify(x, height=height // vae_scale_factor, width=width // vae_scale_factor)
    ttnn.synchronize_device(mesh_device)
    check("patchify", patchified, channel_sharded=False)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((4, 8), id="4x8")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [(1, 512, 512)],
    ids=["512"],
)
def test_vae_flux2_encoder_perf_4x8(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    """Encoder perf on 4x8: timed ``encode_and_patchify`` only (no per-stage or trace capture)."""
    torch.manual_seed(0)

    tp_axis, h_axis = _encoder_parallel_axes(mesh_device)

    torch_model = _load_pruned_torch_encoder_model()
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=None)

    tt_model = Flux2VaeEncoder(
        in_channels=torch_model.config.in_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=_LAYERS_PER_BLOCK,
        z_channels=torch_model.config.latent_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    image = torch.randn(batch_size, torch_model.config.in_channels, height, width)
    enc_height = height // 8
    enc_width = width // 8

    h_mesh_axis = vae_parallel_config.h_parallel.mesh_axis if vae_parallel_config.h_parallel is not None else None
    up_mesh_axes = [None, h_mesh_axis, None, None] if h_mesh_axis is not None else None

    tt_image = tensor.from_torch(
        image.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=up_mesh_axes,
    )

    for _ in range(_PERF_WARMUP_RUNS):
        tt_model.encode_and_patchify(tt_image, height=enc_height, width=enc_width)
    ttnn.synchronize_device(mesh_device)

    samples: list[float] = []
    for _ in range(_PERF_NUM_RUNS):
        t0 = time.perf_counter()
        tt_model.encode_and_patchify(tt_image, height=enc_height, width=enc_width)
        ttnn.synchronize_device(mesh_device)
        samples.append(time.perf_counter() - t0)

    mean_s = statistics.mean(samples)
    logger.info("=" * 72)
    logger.info(f"Flux2 VAE encoder perf (4x8 WH galaxy, {height}x{width})")
    logger.info("-" * 72)
    logger.info(f"  {'encode_and_patchify':28} | {mean_s * 1e3:9.3f} ms  (mean of {_PERF_NUM_RUNS} runs)")
    logger.info(f"  {'min':28} | {min(samples) * 1e3:9.3f} ms")
    logger.info(f"  {'max':28} | {max(samples) * 1e3:9.3f} ms")
    logger.info("=" * 72)


def test_vae_encode_img2img(tt_dit_cache_dir) -> None:
    """PCC: img2img VAE encode + patchify + pack helpers match diffusers Flux2Pipeline."""
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
    from PIL import Image

    from ....pipelines.flux2.pipeline_flux2 import _pack_latents, _patchify_latents

    vae = AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    assert isinstance(vae, AutoencoderKLFlux2)
    vae.eval()

    image_processor = Flux2ImageProcessor(vae_scale_factor=16)
    image = Image.new("RGB", (512, 512), color=(32, 64, 128))
    preprocessed = image_processor.preprocess(image, height=512, width=512, resize_mode="crop")

    with torch.no_grad():
        encoded = vae.encode(preprocessed).latent_dist.mode()

    ref_patchified = Flux2Pipeline._patchify_latents(encoded)
    ours_patchified = _patchify_latents(encoded)
    assert torch.equal(ref_patchified, ours_patchified)

    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(ref_patchified.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(ref_patchified.dtype)
    ref_normalized = (ref_patchified - mean) / std
    ours_normalized = (ours_patchified - mean) / std
    assert torch.equal(ref_normalized, ours_normalized)

    ref_packed = Flux2Pipeline._pack_latents(ref_normalized)
    ours_packed = _pack_latents(ours_normalized)
    assert torch.equal(ref_packed, ours_packed)
