# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""T-sharded LTX-2 vocoder parity test (Stage B, multi-chip).

The 1×1 tests in ``test_vocoder_ltx.py`` exercise the unsharded path. This
test drives the opt-in T-sharding path: the full production vocoder runs with
``parallel_config=ParallelFactor(factor, mesh_axis)`` so the time axis fractures
across the mesh and the halo exchange (``_t_neighbor_pad`` →
``neighbor_pad_persistent_buffer``) replaces the internal conv padding. PCC is
checked against the same torch reference the 1×1 test uses — sharding must not
change the result.

Covers the AUDIO_TSHARD_PLAN.md gotchas end-to-end: T padded to
``TILE_HEIGHT * factor`` before mesh_partition, num_links capped by the product
of outer dims, and the LTXConvTranspose1d gather/run/partition shortcut.
"""

from __future__ import annotations

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.audio_vae.vocoder_ltx import LTXVocoder

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
from models.tt_dit.parallel.config import AudioTCParallelConfig, AudioTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality


@pytest.mark.parametrize(
    "mesh_device, shard_axis, num_links",
    [
        ((2, 4), 1, 1),
    ],
    ids=["2x4_shardT_axis1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ltx_vocoder_sharded(mesh_device: ttnn.MeshDevice, shard_axis: int, num_links: int):
    """Full production vocoder, T-sharded across the mesh, PCC ≥ 0.99 vs torch.

    Mirrors ``test_ltx_vocoder`` (same config, weights, input) but fractures T
    on ``shard_axis``. The auto-pick in ``pipeline_ltx._build_tt_audio_decoder``
    selects this same factor (the sequence-parallel axis).
    """
    from ltx_core.model.audio_vae.vocoder import Vocoder

    torch.manual_seed(42)

    cfg = dict(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[5, 2, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1536,
        resblock="AMP1",
        output_sampling_rate=24000,
        activation="snakebeta",
        use_tanh_at_final=False,
        apply_final_activation=True,
        use_bias_at_final=False,
    )
    torch_voc = Vocoder(**cfg)
    torch_voc.eval()
    with torch.no_grad():
        for m in torch_voc.modules():
            if hasattr(m, "alpha") and isinstance(m.alpha, torch.nn.Parameter):
                m.alpha.data = torch.randn_like(m.alpha.data) * 0.1
            if hasattr(m, "beta") and isinstance(m.beta, torch.nn.Parameter):
                m.beta.data = torch.randn_like(m.beta.data) * 0.1

    mesh_shape = tuple(mesh_device.shape)
    parallel_config = ParallelFactor(factor=mesh_shape[shard_axis], mesh_axis=shard_axis)
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    logger.info(f"Sharding vocoder T on mesh {mesh_shape}, axis {shard_axis}, factor {parallel_config.factor}")

    tt_cfg = dict(
        resblock_kernel_sizes=cfg["resblock_kernel_sizes"],
        upsample_rates=cfg["upsample_rates"],
        upsample_kernel_sizes=cfg["upsample_kernel_sizes"],
        resblock_dilation_sizes=cfg["resblock_dilation_sizes"],
        upsample_initial_channel=cfg["upsample_initial_channel"],
        resblock=cfg["resblock"],
        activation=cfg["activation"],
        use_tanh_at_final=cfg["use_tanh_at_final"],
        apply_final_activation=cfg["apply_final_activation"],
        use_bias_at_final=cfg["use_bias_at_final"],
        in_channels=128,
        out_channels=2,
    )
    tt_voc = LTXVocoder(
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **tt_cfg,
    )
    tt_voc.load_torch_state_dict(torch_voc.state_dict())

    B, S, T_frames, mel_bins = 1, 2, 64, 64
    mel = torch.randn(B, S, T_frames, mel_bins, dtype=torch.float32) * 0.5

    with torch.no_grad():
        ref_out = torch_voc(mel)
    tt_out = tt_voc(mel)

    logger.info(f"Sharded vocoder: ref {tuple(ref_out.shape)}, tt {tuple(tt_out.shape)}")
    assert tt_out.shape == ref_out.shape, f"shape mismatch: ref {ref_out.shape}, tt {tt_out.shape}"
    assert_quality(ref_out, tt_out, pcc=0.99)
    logger.info("PASSED: T-sharded LTXVocoder matches torch reference (PCC ≥ 0.99)")


@pytest.mark.parametrize(
    "mesh_device, num_links",
    [((2, 4), 1)],
    ids=["2x4_2Dshard_f8"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.xfail(
    reason=(
        "2D T-sharding on a 2×4 mesh via sequential axis halos is architecturally "
        "unsound: axis1 halo only connects within-row chips, so row-boundary T-slices "
        "(e.g. chip(0,3)→chip(1,0)) get no halo context. Correct 8-way T-sharding "
        "requires a linear (1,8) mesh topology."
    ),
    strict=True,
)
def test_ltx_vocoder_2d_sharded(mesh_device: ttnn.MeshDevice, num_links: int):
    """Full production vocoder, 2D T-sharding (factor=8) on 2×4 mesh, PCC ≥ 0.99.

    AudioTParallelConfig partitions T across both mesh axes simultaneously so
    all 8 chips do distinct work. Validates the two-axis neighbor_pad_async
    halo and two-pass mesh_partition / all_gather.
    """
    from ltx_core.model.audio_vae.vocoder import Vocoder

    torch.manual_seed(42)

    cfg = dict(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[5, 2, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1536,
        resblock="AMP1",
        output_sampling_rate=24000,
        activation="snakebeta",
        use_tanh_at_final=False,
        apply_final_activation=True,
        use_bias_at_final=False,
    )
    torch_voc = Vocoder(**cfg)
    torch_voc.eval()
    with torch.no_grad():
        for m in torch_voc.modules():
            if hasattr(m, "alpha") and isinstance(m.alpha, torch.nn.Parameter):
                m.alpha.data = torch.randn_like(m.alpha.data) * 0.1
            if hasattr(m, "beta") and isinstance(m.beta, torch.nn.Parameter):
                m.beta.data = torch.randn_like(m.beta.data) * 0.1

    mesh_shape = tuple(mesh_device.shape)
    parallel_config = AudioTParallelConfig(
        axis0=ParallelFactor(factor=mesh_shape[0], mesh_axis=0),
        axis1=ParallelFactor(factor=mesh_shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    logger.info(f"2D sharding vocoder T on mesh {mesh_shape}, factor={parallel_config.factor}")

    tt_cfg = dict(
        resblock_kernel_sizes=cfg["resblock_kernel_sizes"],
        upsample_rates=cfg["upsample_rates"],
        upsample_kernel_sizes=cfg["upsample_kernel_sizes"],
        resblock_dilation_sizes=cfg["resblock_dilation_sizes"],
        upsample_initial_channel=cfg["upsample_initial_channel"],
        resblock=cfg["resblock"],
        activation=cfg["activation"],
        use_tanh_at_final=cfg["use_tanh_at_final"],
        apply_final_activation=cfg["apply_final_activation"],
        use_bias_at_final=cfg["use_bias_at_final"],
        in_channels=128,
        out_channels=2,
    )
    tt_voc = LTXVocoder(
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **tt_cfg,
    )
    tt_voc.load_torch_state_dict(torch_voc.state_dict())

    B, S, T_frames, mel_bins = 1, 2, 64, 64
    mel = torch.randn(B, S, T_frames, mel_bins, dtype=torch.float32) * 0.5

    with torch.no_grad():
        ref_out = torch_voc(mel)
    tt_out = tt_voc(mel)

    logger.info(f"2D sharded vocoder: ref {tuple(ref_out.shape)}, tt {tuple(tt_out.shape)}")
    assert tt_out.shape == ref_out.shape
    assert_quality(ref_out, tt_out, pcc=0.99)
    logger.info(f"PASSED: 2D T-sharded LTXVocoder (factor={parallel_config.factor}) matches reference")


@pytest.mark.parametrize(
    "mesh_device, num_links",
    [((2, 4), 1)],
    ids=["2x4_T4_C2"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ltx_vocoder_channel_tp(mesh_device: ttnn.MeshDevice, num_links: int):
    """Vocoder with T-halo on axis 1 (×4) + channel-TP on axis 0 (×2), PCC ≥ 0.99.

    Both mesh axes do distinct work, soundly: T uses the halo exchange, C uses
    tensor-parallel (full convs gather C_in / scatter C_out, per-channel ops run
    on the C-shard). Channels have no sequence boundary, so the C axis is sound
    where the 2D-T scheme (above) is not.
    """
    from ltx_core.model.audio_vae.vocoder import Vocoder

    torch.manual_seed(42)

    cfg = dict(
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[5, 2, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1536,
        resblock="AMP1",
        output_sampling_rate=24000,
        activation="snakebeta",
        use_tanh_at_final=False,
        apply_final_activation=True,
        use_bias_at_final=False,
    )
    torch_voc = Vocoder(**cfg)
    torch_voc.eval()
    with torch.no_grad():
        for m in torch_voc.modules():
            if hasattr(m, "alpha") and isinstance(m.alpha, torch.nn.Parameter):
                m.alpha.data = torch.randn_like(m.alpha.data) * 0.1
            if hasattr(m, "beta") and isinstance(m.beta, torch.nn.Parameter):
                m.beta.data = torch.randn_like(m.beta.data) * 0.1

    mesh_shape = tuple(mesh_device.shape)
    parallel_config = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=mesh_shape[1], mesh_axis=1),
        channel_parallel=ParallelFactor(factor=mesh_shape[0], mesh_axis=0),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    logger.info(
        f"Vocoder T(axis1)×{parallel_config.time_parallel.factor} + "
        f"C(axis0)×{parallel_config.channel_parallel.factor} on mesh {mesh_shape}"
    )

    tt_cfg = dict(
        resblock_kernel_sizes=cfg["resblock_kernel_sizes"],
        upsample_rates=cfg["upsample_rates"],
        upsample_kernel_sizes=cfg["upsample_kernel_sizes"],
        resblock_dilation_sizes=cfg["resblock_dilation_sizes"],
        upsample_initial_channel=cfg["upsample_initial_channel"],
        resblock=cfg["resblock"],
        activation=cfg["activation"],
        use_tanh_at_final=cfg["use_tanh_at_final"],
        apply_final_activation=cfg["apply_final_activation"],
        use_bias_at_final=cfg["use_bias_at_final"],
        in_channels=128,
        out_channels=2,
    )
    tt_voc = LTXVocoder(
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **tt_cfg,
    )
    tt_voc.load_torch_state_dict(torch_voc.state_dict())

    B, S, T_frames, mel_bins = 1, 2, 64, 64
    mel = torch.randn(B, S, T_frames, mel_bins, dtype=torch.float32) * 0.5

    with torch.no_grad():
        ref_out = torch_voc(mel)
    tt_out = tt_voc(mel)

    logger.info(f"Channel-TP vocoder: ref {tuple(ref_out.shape)}, tt {tuple(tt_out.shape)}")
    assert tt_out.shape == ref_out.shape
    assert_quality(ref_out, tt_out, pcc=0.99)
    logger.info("PASSED: channel-TP LTXVocoder matches torch reference (PCC ≥ 0.99)")
