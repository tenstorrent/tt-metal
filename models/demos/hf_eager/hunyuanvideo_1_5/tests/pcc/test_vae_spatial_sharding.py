# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Property and focused device tests for the Hunyuan VAE H/W shard contract."""

import glob
import json
import os
import resource
import time
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_decoder import (
    AttnBlock,
    CausalConv3d,
    HunyuanVideo15Decoder,
    TTVAEDecodeAdapter,
    Upsample,
)
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_spatial import (
    SpatialShardPlan,
    canonicalize_host_shard_edges,
    canonicalize_replicated_shard_edges,
    causal_upsampled_frames,
    host_shard_with_halo,
    replicate_pad_to_plan,
    stitch_host_shards,
    stitch_tiles_ttnn,
)
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard


@pytest.mark.parametrize(
    "height,width,height_factor,width_factor",
    [
        (1, 1, 1, 1),
        (7, 11, 2, 3),
        (30, 53, 8, 4),  # 480x848 latent at 16x spatial compression
        (45, 80, 8, 4),  # 720x1280 latent at 16x spatial compression
    ],
)
def test_partition_round_trip_preserves_edge_and_uneven_dimensions(height, width, height_factor, width_factor):
    plan = SpatialShardPlan(height, width, height_factor, width_factor)
    source = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
    padded = replicate_pad_to_plan(source, plan)
    shards = []
    for shard in plan.shards():
        shards.append(padded[:, :, shard.h_start : shard.h_stop, shard.w_start : shard.w_stop])

    torch.testing.assert_close(stitch_host_shards(shards, plan), source, rtol=0, atol=0)
    assert padded.shape[-2:] == (plan.padded_height, plan.padded_width)
    assert all(shard.shape[-2:] == (plan.local_height, plan.local_width) for shard in shards)


@pytest.mark.parametrize("height,width,height_factor,width_factor", [(7, 11, 2, 3), (30, 53, 8, 4), (45, 80, 8, 4)])
@pytest.mark.parametrize("seed", range(4))
def test_per_layer_replicate_halo_conv_matches_unsharded(seed, height, width, height_factor, width_factor):
    """A radius-1 exchange at this layer is sufficient for an arbitrary 3x3 convolution."""
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(1, 3, height, width, generator=generator)
    weight = torch.randn(5, 3, 3, 3, generator=generator)
    plan = SpatialShardPlan(height, width, height_factor, width_factor)
    padded = replicate_pad_to_plan(source, plan)

    reference = torch.nn.functional.conv2d(torch.nn.functional.pad(source, (1, 1, 1, 1), mode="replicate"), weight)
    local_outputs = []
    for shard in plan.shards():
        with_halo = host_shard_with_halo(
            padded,
            plan,
            shard.rank_h,
            shard.rank_w,
            halo_h=1,
            halo_w=1,
        )
        local_outputs.append(torch.nn.functional.conv2d(with_halo, weight))

    torch.testing.assert_close(stitch_host_shards(local_outputs, plan), reference, rtol=1e-5, atol=1e-5)


def test_121_frame_temporal_causality_and_spatial_scale_metadata():
    plan_480p = SpatialShardPlan(30, 53, 8, 4)
    plan_720p = SpatialShardPlan(45, 80, 8, 4)

    assert causal_upsampled_frames(31, temporal_upsample_stages=2) == 121
    assert (plan_480p.scaled(16).logical_height, plan_480p.scaled(16).logical_width) == (480, 848)
    assert (plan_720p.scaled(16).logical_height, plan_720p.scaled(16).logical_width) == (720, 1280)


@pytest.mark.parametrize("base_height,base_width", [(30, 53), (45, 80)])
@pytest.mark.parametrize("upsample_stage", range(5))
def test_rank_local_edge_fill_matches_global_replicate_padding(base_height, base_width, upsample_stage):
    """Check every production-plan rank, including uneven bottom-right corners."""
    scale = 2**upsample_stage
    plan = SpatialShardPlan(base_height, base_width, 8, 4).scaled(scale)
    source = torch.arange(plan.logical_height * plan.logical_width, dtype=torch.float32).reshape(
        1, 1, plan.logical_height, plan.logical_width
    )
    expected = replicate_pad_to_plan(source, plan)

    for shard_meta in plan.shards():
        expected_shard = expected[
            :,
            :,
            shard_meta.h_start : shard_meta.h_stop,
            shard_meta.w_start : shard_meta.w_stop,
        ]
        damaged = expected_shard.clone()
        if shard_meta.logical_height < plan.local_height:
            damaged[..., shard_meta.logical_height :, :] = -1000 - shard_meta.rank_h
        if shard_meta.logical_width < plan.local_width:
            damaged[..., :, shard_meta.logical_width :] = -2000 - shard_meta.rank_w
        repaired = canonicalize_host_shard_edges(damaged, plan, shard_meta.rank_h, shard_meta.rank_w)
        torch.testing.assert_close(repaired, expected_shard, rtol=0, atol=0)

    bottom_right = canonicalize_host_shard_edges(
        expected[
            ...,
            -plan.local_height :,
            -plan.local_width :,
        ],
        plan,
        plan.height_factor - 1,
        plan.width_factor - 1,
    )
    assert bottom_right[..., -1, -1].item() == source[..., -1, -1].item()


class _StateModule:
    def __init__(self, state, **attrs):
        self._state = state
        for name, value in attrs.items():
            setattr(self, name, value)

    def state_dict(self):
        return self._state


def _conv_node(cin, cout, kernel=3, generator=None):
    weight = torch.randn(cout, cin, kernel, kernel, kernel, generator=generator) / (cin * kernel**3) ** 0.5
    bias = torch.randn(cout, generator=generator) * 0.01
    return SimpleNamespace(conv=SimpleNamespace(weight=weight, bias=bias, in_channels=cin))


def _resnet_node(channels, generator):
    state = {
        "norm1.gamma": torch.ones(channels),
        "conv1.conv.weight": _conv_node(channels, channels, generator=generator).conv.weight,
        "conv1.conv.bias": torch.zeros(channels),
        "norm2.gamma": torch.ones(channels),
        "conv2.conv.weight": _conv_node(channels, channels, generator=generator).conv.weight,
        "conv2.conv.bias": torch.zeros(channels),
    }
    return _StateModule(state)


def _attention_node(channels, generator):
    state = {"norm.gamma": torch.ones(channels)}
    for name in ("to_q", "to_k", "to_v", "proj_out"):
        state[f"{name}.weight"] = _conv_node(channels, channels, kernel=1, generator=generator).conv.weight
        state[f"{name}.bias"] = torch.zeros(channels)
    return _StateModule(state, in_channels=channels)


def _upsample_node(channels, generator, *, temporal=False):
    # repeats=4 keeps channels unchanged after the 2x2 spatial DCAE rearrange.
    return SimpleNamespace(
        conv=_conv_node(channels, channels * 4, generator=generator),
        add_temporal_upsample=temporal,
        repeats=4,
    )


def _mini_decoder_node(channels=32):
    generator = torch.Generator().manual_seed(123)
    attention = _attention_node(channels, generator)
    mid = SimpleNamespace(
        resnets=[_resnet_node(channels, generator), _resnet_node(channels, generator)],
        attentions=[attention],
    )
    up = SimpleNamespace(
        resnets=[_resnet_node(channels, generator)],
        upsamplers=[_upsample_node(channels, generator)],
    )
    return SimpleNamespace(
        repeat=1,
        conv_in=_conv_node(channels, channels, generator=generator),
        mid_block=mid,
        up_blocks=[up],
        norm_out=SimpleNamespace(gamma=torch.ones(channels)),
        conv_out=_conv_node(channels, 3, generator=generator),
    )


def _parallel_kwargs(mesh_device):
    shape = tuple(mesh_device.shape)
    config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(shape[0], 0),
        width_parallel=ParallelFactor(shape[1], 1),
    )
    manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    return config, manager


def _replicated_bthwc(mesh_device, host_bthwc):
    return ttnn.from_torch(
        host_bthwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _sharded_bthwc(mesh_device, host_bthwc, plan, config):
    padded = replicate_pad_to_plan(host_bthwc, plan, h_dim=2, w_dim=3)
    return typed_tensor_2dshard(
        padded,
        mesh_device,
        shard_mapping={
            config.height_parallel.mesh_axis: 2,
            config.width_parallel.mesh_axis: 3,
        },
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )


def _compose_bthwc(tt_tensor, mesh_device, config, manager, logical_h, logical_w):
    dims = [None, None]
    dims[config.height_parallel.mesh_axis] = 2
    dims[config.width_parallel.mesh_axis] = 3
    host = fast_device_to_host(tt_tensor, mesh_device, dims, ccl_manager=manager)
    return host[:, :, :logical_h, :logical_w, :]


def _compose_padded_bthwc(tt_tensor, mesh_device, config, manager):
    dims = [None, None]
    dims[config.height_parallel.mesh_axis] = 2
    dims[config.width_parallel.mesh_axis] = 3
    return fast_device_to_host(tt_tensor, mesh_device, dims, ccl_manager=manager)


_FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}]
_DECODER_MESHES = [(1, 2), (2, 2)]
_EDGE_DEVICE_MESHES = [(2, 2)]
if os.environ.get("HY_VAE_TEST_8X4", "0") == "1":
    _DECODER_MESHES.append((8, 4))
    _EDGE_DEVICE_MESHES.append((8, 4))


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 2)], indirect=True)
def test_device_causal_conv_halo_matches_unsharded(mesh_device):
    torch.manual_seed(0)
    source = torch.randn(1, 2, 5, 7, 32)  # uneven W on a 1x2 mesh
    weight = torch.randn(48, 32, 3, 3, 3) / (32 * 27) ** 0.5
    bias = torch.randn(48) * 0.01

    reference_module = CausalConv3d(weight, bias, device=mesh_device)
    reference_tt = reference_module(_replicated_bthwc(mesh_device, source))
    reference = ttnn.to_torch(ttnn.get_device_tensors(reference_tt)[0]).float()

    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(5, 7, config.height_parallel.factor, config.width_parallel.factor)
    sharded_module = CausalConv3d(
        weight,
        bias,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
    )
    output_tt = sharded_module(_sharded_bthwc(mesh_device, source, plan, config), 5, 7)
    output = _compose_bthwc(output_tt, mesh_device, config, manager, 5, 7)

    ok, pcc = comp_pcc(reference, output, 0.999)
    print(f"[vae halo conv] PCC={pcc}", flush=True)
    assert ok, f"halo convolution PCC {pcc} < 0.999"


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 2)], indirect=True)
def test_device_attention_gather_repartition_matches_unsharded(mesh_device):
    generator = torch.Generator().manual_seed(1)
    source = torch.randn(1, 2, 5, 7, 32, generator=generator)
    torch_attn = _attention_node(32, generator)

    reference_module = AttnBlock(torch_attn, device=mesh_device)
    reference_tt = reference_module(_replicated_bthwc(mesh_device, source))
    reference = ttnn.to_torch(ttnn.get_device_tensors(reference_tt)[0]).float()

    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(5, 7, config.height_parallel.factor, config.width_parallel.factor)
    sharded_module = AttnBlock(
        torch_attn,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
    )
    output_tt = sharded_module(_sharded_bthwc(mesh_device, source, plan, config), 5, 7)
    output = _compose_bthwc(output_tt, mesh_device, config, manager, 5, 7)

    ok, pcc = comp_pcc(reference, output, 0.999)
    print(f"[vae gathered attention] PCC={pcc}", flush=True)
    assert ok, f"gathered attention PCC {pcc} < 0.999"


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 2)], indirect=True)
def test_device_uneven_upsample_preserves_logical_extent(mesh_device):
    generator = torch.Generator().manual_seed(2)
    source = torch.randn(1, 2, 5, 7, 32, generator=generator)
    torch_up = _upsample_node(32, generator, temporal=True)

    reference_module = Upsample(torch_up, device=mesh_device)
    reference_tt = reference_module(_replicated_bthwc(mesh_device, source))
    reference = ttnn.to_torch(ttnn.get_device_tensors(reference_tt)[0]).float()

    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(5, 7, config.height_parallel.factor, config.width_parallel.factor)
    sharded_module = Upsample(
        torch_up,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
    )
    output_tt, logical_h, logical_w = sharded_module(_sharded_bthwc(mesh_device, source, plan, config), 5, 7)
    output = _compose_bthwc(output_tt, mesh_device, config, manager, logical_h, logical_w)

    assert (logical_h, logical_w) == (10, 14)
    assert output.shape[1:4] == (3, 10, 14)  # temporal causality: 2 -> 3, not 4
    ok, pcc = comp_pcc(reference, output, 0.999)
    print(f"[vae uneven upsample] PCC={pcc}", flush=True)
    assert ok, f"uneven upsample PCC {pcc} < 0.999"


@pytest.mark.parametrize("logical_h,logical_w", [(30, 53), (45, 80)])
@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", _EDGE_DEVICE_MESHES, indirect=True)
def test_device_rank_local_edge_fill_matches_replicate_padding(mesh_device, logical_h, logical_w):
    config, manager = _parallel_kwargs(mesh_device)
    plan = SpatialShardPlan(
        logical_h,
        logical_w,
        config.height_parallel.factor,
        config.width_parallel.factor,
    )
    source = torch.randn(1, 1, logical_h, logical_w, 32)
    expected = replicate_pad_to_plan(source, plan, h_dim=2, w_dim=3)
    damaged = expected.clone()
    damaged[:, :, logical_h:, :, :] = -3
    damaged[:, :, :, logical_w:, :] = 5
    damaged_tt = typed_tensor_2dshard(
        damaged,
        mesh_device,
        shard_mapping={
            config.height_parallel.mesh_axis: 2,
            config.width_parallel.mesh_axis: 3,
        },
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    gather_count = 0
    original_gather = manager.all_gather_persistent_buffer

    def counted_gather(*args, **kwargs):
        nonlocal gather_count
        gather_count += 1
        return original_gather(*args, **kwargs)

    manager.all_gather_persistent_buffer = counted_gather
    repaired_tt = canonicalize_replicated_shard_edges(damaged_tt, logical_h, logical_w, config, manager)
    repaired = _compose_padded_bthwc(repaired_tt, mesh_device, config, manager)

    assert gather_count == 0
    ok, pcc = comp_pcc(expected, repaired, 0.999)
    print(f"[vae local edge fill {logical_h}x{logical_w}] PCC={pcc}, gathers={gather_count}", flush=True)
    assert ok, f"rank-local edge-fill PCC {pcc} < 0.999"


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", _DECODER_MESHES, indirect=True)
def test_device_small_random_decoder_matches_unsharded(mesh_device):
    torch.manual_seed(3)
    mesh_shape = tuple(mesh_device.shape)
    logical_h = max(5, 2 * mesh_shape[0] - 1)
    logical_w = max(7, 2 * mesh_shape[1] - 1)
    source = torch.randn(1, 2, logical_h, logical_w, 32)
    torch_decoder = _mini_decoder_node()

    reference_decoder = HunyuanVideo15Decoder(torch_decoder, device=mesh_device)
    reference_tt = reference_decoder(_replicated_bthwc(mesh_device, source))
    reference = ttnn.to_torch(ttnn.get_device_tensors(reference_tt)[0]).float()

    config, manager = _parallel_kwargs(mesh_device)
    gather_calls = []
    original_gather = manager.all_gather_persistent_buffer

    def counted_gather(tensor, /, **kwargs):
        gather_calls.append((kwargs["dim"], kwargs["mesh_axis"]))
        return original_gather(tensor, **kwargs)

    manager.all_gather_persistent_buffer = counted_gather
    plan = SpatialShardPlan(
        logical_h,
        logical_w,
        config.height_parallel.factor,
        config.width_parallel.factor,
    )
    sharded_decoder = HunyuanVideo15Decoder(
        torch_decoder,
        device=mesh_device,
        parallel_config=config,
        ccl_manager=manager,
    )
    output_tt, output_h, output_w = sharded_decoder(
        _sharded_bthwc(mesh_device, source, plan, config), logical_h, logical_w
    )
    graph_gathers = len(gather_calls)
    output = _compose_bthwc(output_tt, mesh_device, config, manager, output_h, output_w)

    assert (output_h, output_w) == (logical_h * 2, logical_w * 2)
    active_axes = sum(factor > 1 for factor in (config.height_parallel.factor, config.width_parallel.factor))
    old_correctness_gathers = 11 * active_axes
    assert graph_gathers == active_axes
    ok, pcc = comp_pcc(reference, output, 0.999)
    print(
        f"[vae small sharded decoder] PCC={pcc}, graph gathers {old_correctness_gathers}->{graph_gathers}",
        flush=True,
    )
    assert ok, f"small decoder PCC {pcc} < 0.999"


@pytest.mark.parametrize("device_params", _FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_device_real_weight_decoder_offline_gate(mesh_device):
    """Offline-only real-weight PCC/timing gate; never calls the Hub."""
    if os.environ.get("HY_VAE_REAL_WEIGHT_GATE", "0") != "1":
        pytest.skip("set HY_VAE_REAL_WEIGHT_GATE=1 to run the cached real-weight gate")

    from diffusers import AutoencoderKLHunyuanVideo15

    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snapshots = glob.glob(
        os.path.join(
            hub,
            "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_i2v",
            "snapshots",
            "*",
        )
    )
    snapshots = [path for path in snapshots if os.path.isfile(os.path.join(path, "vae", "config.json"))]
    if not snapshots:
        pytest.skip("cached 480p I2V VAE snapshot is unavailable")

    mode = os.environ.get("HY_VAE_REAL_WEIGHT_MODE", "hw")
    assert mode in {"hw", "replicated", "tile"}
    latent_t = int(os.environ.get("HY_VAE_REAL_LATENT_T", "2"))
    latent_h = int(os.environ.get("HY_VAE_REAL_LATENT_H", "8"))
    latent_w = int(os.environ.get("HY_VAE_REAL_LATENT_W", "4"))
    if mode == "hw":
        assert latent_h >= mesh_device.shape[0] and latent_w >= mesh_device.shape[1]

    load_start = time.perf_counter()
    vae = AutoencoderKLHunyuanVideo15.from_pretrained(
        snapshots[-1], subfolder="vae", local_files_only=True, torch_dtype=torch.float32
    ).eval()
    host_load_seconds = time.perf_counter() - load_start
    torch.manual_seed(19)
    latent = torch.randn(1, vae.config.latent_channels, latent_t, latent_h, latent_w)

    skip_host = os.environ.get("HY_VAE_REAL_SKIP_HOST", "0") == "1"
    reference = None
    host_seconds = None
    if not skip_host:
        with torch.no_grad():
            host_start = time.perf_counter()
            reference = vae.decode(latent).sample.float()
            host_seconds = time.perf_counter() - host_start

    build_start = time.perf_counter()
    if mode == "hw":
        config, manager = _parallel_kwargs(mesh_device)
        decoder = HunyuanVideo15Decoder(
            vae.decoder,
            device=mesh_device,
            parallel_config=config,
            ccl_manager=manager,
        )
        plan = SpatialShardPlan(latent_h, latent_w, 8, 4)
        latent_tt = _sharded_bthwc(mesh_device, latent.permute(0, 2, 3, 4, 1), plan, config)
    elif mode == "replicated":
        config = manager = None
        decoder = HunyuanVideo15Decoder(vae.decoder, device=mesh_device)
        latent_tt = _replicated_bthwc(mesh_device, latent.permute(0, 2, 3, 4, 1))
    else:
        config = manager = latent_tt = None
        os.environ["HY_VAE_HW_SHARD"] = "0"
        os.environ["HY_VAE_TILE"] = "1"
        decoder = TTVAEDecodeAdapter(vae, mesh_device)
    build_seconds = time.perf_counter() - build_start

    gather_count = 0
    if manager is not None:
        original_gather = manager.all_gather_persistent_buffer

        def counted_gather(*args, **kwargs):
            nonlocal gather_count
            gather_count += 1
            return original_gather(*args, **kwargs)

        manager.all_gather_persistent_buffer = counted_gather

    def dram_allocated_mib():
        ttnn.synchronize_device(mesh_device)
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return view.num_banks * view.total_bytes_allocated_per_bank / (1024**2)

    observed_dram_mib = [dram_allocated_mib()]
    device_start = time.perf_counter()
    if mode == "hw":
        output_tt, output_h, output_w = decoder(latent_tt, latent_h, latent_w)
    elif mode == "replicated":
        output_tt = decoder(latent_tt)
        output_h, output_w = latent_h * 16, latent_w * 16
    else:
        output = decoder.decode(latent).sample.float()
        output_tt = None
    ttnn.synchronize_device(mesh_device)
    device_seconds = time.perf_counter() - device_start
    observed_dram_mib.append(dram_allocated_mib())

    if mode != "tile":
        readback_start = time.perf_counter()
        if mode == "hw":
            output = _compose_bthwc(output_tt, mesh_device, config, manager, output_h, output_w)
        else:
            output = ttnn.to_torch(ttnn.get_device_tensors(output_tt)[0]).float()
        readback_seconds = time.perf_counter() - readback_start
        output = output.permute(0, 4, 1, 2, 3).contiguous()
    else:
        readback_seconds = None

    pcc = max_abs = None
    ok = True
    if reference is not None:
        ok, pcc = comp_pcc(reference, output, 0.95)
        max_abs = float(torch.max(torch.abs(reference - output)))
    compare_path = os.environ.get("HY_VAE_REAL_COMPARE_PATH")
    compare_pcc = compare_final_frame_pcc = None
    if compare_path:
        comparison = torch.load(compare_path, map_location="cpu", weights_only=True)
        _, compare_pcc = comp_pcc(comparison, output, 0.0)
        _, compare_final_frame_pcc = comp_pcc(comparison[:, :, -1], output[:, :, -1], 0.0)
    save_path = os.environ.get("HY_VAE_REAL_SAVE_PATH")
    if save_path:
        torch.save(output, save_path)
    metrics = {
        "mode": mode,
        "latent_shape": list(latent.shape),
        "output_shape": list(output.shape),
        "pcc": pcc,
        "max_abs": max_abs,
        "comparison_pcc": compare_pcc,
        "comparison_final_frame_pcc": compare_final_frame_pcc,
        "host_load_seconds": host_load_seconds,
        "host_decode_seconds": host_seconds,
        "tt_build_seconds": build_seconds,
        "tt_first_decode_seconds": device_seconds,
        "final_d2h_seconds": readback_seconds,
        "graph_gathers": gather_count,
        "observed_dram_allocated_mib": max(observed_dram_mib),
        "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    print(f"[vae real-weight offline gate] {json.dumps(metrics, sort_keys=True)}", flush=True)
    assert ok, f"real-weight {mode} decoder PCC {pcc} < 0.95"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_device_tile_blend_crop_stitch_matches_host(mesh_device):
    """Focused PCC gate for TTNN blend/crop/stitch with odd edge tiles."""
    torch.manual_seed(0)
    coords = [(0, 0, 11, 13), (0, 10, 11, 7), (8, 0, 5, 13), (8, 10, 5, 7)]
    padded_tiles = torch.randn(4, 5, 11, 13, 3)  # tile-major BTHWC
    decoded_ncthw = [
        padded_tiles[index : index + 1, :, :real_h, :real_w, :].permute(0, 4, 1, 2, 3).contiguous()
        for index, (_, _, real_h, real_w) in enumerate(coords)
    ]
    reference = TTVAEDecodeAdapter._stitch_tiles(
        decoded_ncthw,
        coords,
        ncol=2,
        blend_h=3,
        blend_w=4,
        row_limit_h=8,
        row_limit_w=10,
    )

    tt_tiles = ttnn.from_torch(
        padded_tiles,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    tt_result = stitch_tiles_ttnn(
        tt_tiles,
        coords,
        ncol=2,
        blend_h=3,
        blend_w=4,
        row_limit_h=8,
        row_limit_w=10,
        spatial_scale=1,
        device=mesh_device,
    )
    result = ttnn.to_torch(ttnn.get_device_tensors(tt_result)[0]).float().permute(0, 4, 1, 2, 3)
    ok, pcc = comp_pcc(reference, result, 0.999)
    print(f"[vae device stitch] PCC={pcc} -> {'PASS' if ok else 'FAIL'}", flush=True)
    assert ok, f"device tile blend/crop/stitch PCC {pcc} < 0.999"
