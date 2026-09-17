# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn

from ...layers.normalization import (
    DistributedGroupNorm,
    DistributedLayerNorm,
    DistributedRMSNorm,
    GroupNorm,
    LayerNorm,
    RMSNorm,
)
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.tensor import bf16_tensor


class TorchRMSNorm(torch.nn.Module):
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = bias
        if norm_elementwise_affine:
            self.weight = torch.nn.Parameter(torch.randn(embedding_dim))
            if bias:
                self.bias = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.norm_eps)
        if self.norm_elementwise_affine:
            x = x * self.weight
            if self.use_bias:
                x = x + self.bias
        return x


class TorchLayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True):
        super().__init__()
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = bias
        if norm_elementwise_affine:
            self.weight = torch.nn.Parameter(torch.randn(embedding_dim))
            if bias:
                self.bias = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.norm_eps)
        if self.norm_elementwise_affine:
            x = x * self.weight
            if self.use_bias:
                x = x + self.bias
        return x


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (1, 2), (2, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("input_shape"),
    [
        (1, 38, 4096, 64),  # Q, K spatial norm
        (1, 38, 333, 64),  # Q, K prompt norm
    ],
)
@pytest.mark.parametrize(
    ("norm_eltwise_affine, bias"),
    [
        (True, False),
    ],
)
def test_rmsnorm(
    mesh_device: ttnn.MeshDevice,
    input_shape: tuple[int, int, int, int],
    norm_eltwise_affine: bool,
    bias: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchRMSNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    tt_model = RMSNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias, mesh_device=mesh_device
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    for t in ttnn.get_device_tensors(tt_output):
        t = ttnn.to_torch(t)
        assert_quality(torch_output, t, pcc=0.999_500)


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (1, 2), (2, 1)],
    indirect=True,
)
@pytest.mark.parametrize(
    ("input_shape", "use_row_major_workaround"),
    [
        ((1, 1, 4096, 2432), False),  # spatial norm
        ((1, 1, 333, 2432), False),  # prompt norm
        ((1, 1, 22528, 3072), True),  # Mochi large layernorm
    ],
)
@pytest.mark.parametrize(
    ("norm_eltwise_affine, bias"),
    [
        (True, False),
    ],
)
def test_layernorm(
    mesh_device: ttnn.MeshDevice,
    input_shape: tuple[int, int, int, int],
    norm_eltwise_affine: bool,
    bias: bool,
    use_row_major_workaround: bool,
) -> None:
    MIN_PCC = 0.982_000 if input_shape[-2] < 20000 else 0.961_000
    torch_dtype = torch.bfloat16
    torch_model = TorchLayerNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    tt_model = LayerNorm(
        embedding_dim=input_shape[-1],
        norm_elementwise_affine=norm_eltwise_affine,
        bias=bias,
        mesh_device=mesh_device,
        use_row_major_workaround=use_row_major_workaround,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    logger.info(f"Running torch model with input shape {torch_input_tensor.shape}")
    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    logger.info(f"Running TT model with input shape {tt_input_tensor.shape}")
    tt_output = tt_model(tt_input_tensor)

    for t in ttnn.get_device_tensors(tt_output):
        t = ttnn.to_torch(t)
        assert_quality(torch_output, t, pcc=MIN_PCC)


@pytest.mark.parametrize(
    "mesh_device, mesh_axis",
    [
        [(1, 2), 1],
        [(2, 1), 0],
        [(2, 2), 0],
        [(2, 2), 1],
        [(2, 4), 0],
        [(4, 2), 1],
    ],
    ids=[
        "1x2_1",
        "2x1_0",
        "2x2_0",
        "2x2_1",
        "2x4_0",
        "4x2_1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("input_shape"),
    [
        (1, 1, 4096, 2432),  # spatial norm
        (1, 1, 333, 2432),  # prompt norm
        (1, 1, 32768, 384),
    ],
    ids=["shape1", "shape2", "shape3"],
)
@pytest.mark.parametrize(
    ("norm_eltwise_affine, bias"),
    [
        (True, False),
        (False, False),
    ],
    ids=["yes_eltwise_no_bias", "no_eltwise_no_bias"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}],
    ids=["fabric1d"],
    indirect=True,
)
def test_distributed_rms_norm(
    mesh_device: ttnn.MeshDevice,
    mesh_axis: int,
    input_shape: tuple[int, int, int, int],
    norm_eltwise_affine: bool,
    bias: bool,
) -> None:
    torch_dtype = torch.bfloat16
    torch_model = TorchRMSNorm(
        embedding_dim=input_shape[-1], norm_elementwise_affine=norm_eltwise_affine, bias=bias
    ).to(dtype=torch_dtype)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)

    tt_model = DistributedRMSNorm(
        embedding_dim=input_shape[-1],
        norm_elementwise_affine=norm_eltwise_affine,
        bias=bias,
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype) * 2 + 4

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    shard_dims = [None, None]
    shard_dims[mesh_axis] = -1
    shard_dims[1 - mesh_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_300)


TP_SWEEP = [
    pytest.param((1, 1), 0, {"fabric_config": None, "require_exact_physical_num_devices": False}, id="tp1_axis0"),
    pytest.param(
        (1, 2),
        1,
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
        id="tp2_axis1",
    ),
    pytest.param(
        (1, 4),
        1,
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
        id="tp4_axis1",
    ),
]


@pytest.mark.parametrize(
    "embedding_dim",
    [1920, 2048, 2432, 3072, 5120],
    ids=["dim_motif", "dim0", "dim1", "dim2", "dim3"],
)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 512),
        (1, 2048),
        (1, 4096),
        (1, 9472),
        (2, 334),  # Motif prompt
        (2, 4100),  # Motif spatial
    ],
    ids=["b1_len0", "b1_len1", "b1_len2", "b1_len3", "motif_prompt", "motif_spatial"],
)
@pytest.mark.parametrize(
    "affine_parameters, affine_dynamic",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
    ids=["no_affine", "static_affine", "dynamic_affine"],
)
@pytest.mark.parametrize("mesh_device, mesh_axis, device_params", TP_SWEEP, indirect=["mesh_device", "device_params"])
def test_distributed_layernorm(
    mesh_device: ttnn.MeshDevice,
    mesh_axis: int,
    embedding_dim: int,
    batch_size: int,
    seq_len: int,
    affine_parameters: bool,
    affine_dynamic: bool,
) -> None:
    """Covers all DistributedLayerNorm instantiations seen in tt_dit models."""
    torch_dtype = torch.bfloat16

    torch_model = TorchLayerNorm(
        embedding_dim=embedding_dim, norm_elementwise_affine=affine_parameters, bias=affine_parameters
    ).to(dtype=torch_dtype)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)

    tt_model = DistributedLayerNorm(
        embedding_dim=embedding_dim,
        norm_elementwise_affine=affine_parameters,
        bias=affine_parameters,
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, batch_size, seq_len, embedding_dim), dtype=torch_dtype) * 2 + 4
    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1)

    if affine_dynamic:
        torch_model.norm_elementwise_affine = True
        torch_model.use_bias = True
        # Note: Use batch_size if using dynamic affine since Motif dynamics include batch dim
        torch_model.weight = torch.nn.Parameter(torch.randn((batch_size, 1, embedding_dim), dtype=torch_dtype))
        torch_model.bias = torch.nn.Parameter(torch.randn((batch_size, 1, embedding_dim), dtype=torch_dtype))
        # Tilized weights and bias for dynamic affine
        tt_dynamic_weight_tensor = bf16_tensor(
            torch_model.weight.data, device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1
        )
        tt_dynamic_bias_tensor = bf16_tensor(
            torch_model.bias.data, device=mesh_device, mesh_axis=mesh_axis, shard_dim=-1
        )
    else:
        tt_dynamic_weight_tensor = None
        tt_dynamic_bias_tensor = None

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor, dynamic_weight=tt_dynamic_weight_tensor, dynamic_bias=tt_dynamic_bias_tensor)

    shard_dims = [None, None]
    shard_dims[mesh_axis] = -1
    shard_dims[1 - mesh_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )

    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_300)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}],
    ids=["fabric1d"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_axis",
    [1, None],
)
@pytest.mark.parametrize(
    ("group_count", "input_shape"),
    [
        (32, (1, 512, 128, 128)),
        (32, (1, 512, 256, 256)),
        (32, (1, 512, 512, 512)),
        (32, (1, 256, 512, 512)),
        (32, (1, 256, 1024, 1024)),
        (8, (1, 64, 1024, 1024)),
    ],
)
def test_group_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: tuple[int, int, int, int],
    group_count: int,
    mesh_axis: int,
) -> None:
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=input_shape[1])
    torch.nn.init.normal_(torch_model.weight)
    torch.nn.init.normal_(torch_model.bias)
    torch_model.eval()

    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype)

    tt_model = GroupNorm.from_torch(
        torch_ref=torch_model,
        mesh_device=mesh_device,
        mesh_axis=mesh_axis,
        core_grid=ttnn.CoreGrid(x=8, y=8),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1) if mesh_axis is not None else None,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input_tensor)

    tt_torch = ttnn.to_torch(
        tt_output if mesh_axis is not None else ttnn.get_device_tensors(tt_output)[0],
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1) if mesh_axis is not None else None,
    )

    tt_torch = tt_torch.permute(0, 3, 1, 2)

    assert_quality(torch_output, tt_torch, pcc=0.999_300)


@pytest.mark.parametrize(
    "mesh_device, cluster_axis, device_params",
    [
        pytest.param(
            (1, 1),
            1,
            {"fabric_config": None, "require_exact_physical_num_devices": False},
            id="local_1x1",
        ),
        pytest.param(
            (1, 2),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="sp2_axis1",
        ),
        pytest.param(
            (1, 4),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="sp4_axis1",
        ),
        pytest.param(
            (2, 1),
            0,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="sp2_axis0",
        ),
        # 8 devices: the only width that clears require_exact_physical_num_devices on a t3k, so the
        # only config that compiles GN_DISTRIBUTED_AG there.
        pytest.param(
            (1, 8),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="sp8_axis1",
        ),
        pytest.param(
            (4, 8),
            0,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="mesh4x8_axis0",
        ),
        pytest.param(
            (4, 8),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="mesh4x8_axis1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("group_count", "input_shape"),
    [
        # NCHW; H must be divisible by cluster width. C % 32 == 0 for fused v1.
        (32, (1, 128, 64, 64)),
        (32, (1, 256, 128, 128)),
        (8, (1, 64, 64, 64)),
    ],
    ids=["c128_h64", "c256_h128", "c64_g8"],
)
@pytest.mark.parametrize("activation_fn", [None, "silu"], ids=["no_act", "silu"])
def test_distributed_group_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    cluster_axis: int,
    input_shape: tuple[int, int, int, int],
    group_count: int,
    activation_fn: str | None,
) -> None:
    """Spatially shard H on cluster_axis; match torch GroupNorm on the full tensor."""
    torch_dtype = torch.bfloat16
    torch.manual_seed(0)

    cluster_size = tuple(mesh_device.shape)[cluster_axis]
    n, c, h, w = input_shape
    assert h % cluster_size == 0, f"H={h} must be divisible by cluster_size={cluster_size}"

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=c)
    torch.nn.init.normal_(torch_model.weight)
    torch.nn.init.normal_(torch_model.bias)
    torch_model.eval()

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    with torch.no_grad():
        torch_output = torch_model(torch_input)
        if activation_fn == "silu":
            torch_output = torch.nn.functional.silu(torch_output)

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    tt_model = DistributedGroupNorm.from_torch(
        torch_ref=torch_model,
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        mesh_axis=None,
        ccl_manager=ccl_manager,
        activation_fn=activation_fn,
    )

    # NHWC, shard height (dim 1) across cluster_axis.
    nhwc = torch_input.permute(0, 2, 3, 1).contiguous()
    tt_input = bf16_tensor(nhwc, device=mesh_device, mesh_axis=cluster_axis, shard_dim=1)

    tt_output = tt_model(tt_input)

    tt_torch = _gather_group_norm_output(tt_output, mesh_device, cluster_axis, torch_output.shape[0])

    assert_quality(torch_output, tt_torch, pcc=0.999_300)


def _gather_group_norm_output(
    tt_output: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    cluster_axis: int,
    batch: int,
) -> torch.Tensor:
    """Gather the H-sharded NHWC output back to a torch NCHW tensor."""
    shard_dims = [None, None]
    shard_dims[cluster_axis] = 1  # gather H
    shard_dims[1 - cluster_axis] = 0
    tt_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    # ConcatMesh2dToTensor may introduce a leading mesh-replica dim when the other axis is used as batch.
    if tt_torch.ndim == 5:
        tt_torch = tt_torch[0]
    # The non-cluster mesh axis holds identical replicas of the (replicated) input; keep one so the
    # batch matches torch's. The meaningful distributed comparison (H gathered over cluster_axis)
    # is preserved by the retained replica.
    if tt_torch.shape[0] != batch:
        tt_torch = tt_torch[:batch]
    return tt_torch.permute(0, 3, 1, 2)


@pytest.mark.parametrize(
    "mesh_device, cluster_axis, device_params",
    [
        pytest.param(
            (4, 8),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="mesh4x8_axis1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_distributed_group_norm_fused_vs_unfused_silu(
    *,
    mesh_device: ttnn.MeshDevice,
    cluster_axis: int,
) -> None:
    """Fused SiLU vs unfused GroupNorm + ``ttnn.silu``, both in one session.

    Two things under test:
      1. Both paths match the torch golden. The fused path applies SiLU to the fp32 DEST value
         while the unfused path applies it to the bf16-rounded GroupNorm output, so the two are
         NOT bit-exact — fused is expected to score equal or better against torch. Only a PCC
         bound is asserted between them.
      2. The program cache distinguishes them: identical shapes/tensors with the activation as
         the only differing attribute, launched in the same session. If ``activation_fn``
         were missing from ``compute_program_hash``, the second launch would reuse the first
         one's compiled program and one of the golden checks would fail.
    """
    torch_dtype = torch.bfloat16
    torch.manual_seed(0)

    input_shape = (1, 128, 64, 64)
    group_count = 32
    n, c = input_shape[0], input_shape[1]

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=c)
    torch.nn.init.normal_(torch_model.weight)
    torch.nn.init.normal_(torch_model.bias)
    torch_model.eval()

    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    with torch.no_grad():
        torch_gn = torch_model(torch_input)
        torch_silu = torch.nn.functional.silu(torch_gn)

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    nhwc = torch_input.permute(0, 2, 3, 1).contiguous()

    def build(activation_fn: str | None) -> DistributedGroupNorm:
        return DistributedGroupNorm.from_torch(
            torch_ref=torch_model,
            mesh_device=mesh_device,
            cluster_axis=cluster_axis,
            mesh_axis=None,
            ccl_manager=ccl_manager,
            activation_fn=activation_fn,
        )

    # Unfused: GroupNorm, then a separate ttnn.silu (what the model did before fusion).
    tt_input = bf16_tensor(nhwc, device=mesh_device, mesh_axis=cluster_axis, shard_dim=1)
    unfused_out = build(None)(tt_input)
    unfused_out = ttnn.silu(unfused_out, output_tensor=unfused_out)
    unfused = _gather_group_norm_output(unfused_out, mesh_device, cluster_axis, n)

    # Fused: same shapes, activation is the only differing op attribute.
    tt_input = bf16_tensor(nhwc, device=mesh_device, mesh_axis=cluster_axis, shard_dim=1)
    fused = _gather_group_norm_output(build("silu")(tt_input), mesh_device, cluster_axis, n)

    logger.info("unfused vs torch:")
    assert_quality(torch_silu, unfused, pcc=0.999_300)
    logger.info("fused vs torch:")
    assert_quality(torch_silu, fused, pcc=0.999_300)
    logger.info("fused vs unfused:")
    assert_quality(unfused, fused, pcc=0.999_500)


@pytest.mark.parametrize(
    "mesh_device, cluster_axis, device_params",
    [
        pytest.param(
            (1, 4),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="sp4_axis1",
        ),
        pytest.param(
            (1, 8),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="sp8_axis1",
        ),
        pytest.param(
            (4, 8),
            1,
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
            id="mesh4x8_axis1",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_distributed_group_norm_program_cache_hit(
    *,
    mesh_device: ttnn.MeshDevice,
    cluster_axis: int,
) -> None:
    """Two launches with identical attributes but different buffers must both be correct.

    The second launch hits the program cache, so it runs ``override_runtime_arguments`` instead of
    rebuilding the program. That override patches the input/output/gamma/beta/mask/stats addresses
    and, on the master readers, the per-core stats-DRAM arg index. Nothing else in this file
    exercises it: ``test_distributed_group_norm`` gets a fresh device per parametrization, and the
    fused-vs-unfused test deliberately varies the activation to force a cache miss.
    """
    torch_dtype = torch.bfloat16
    torch.manual_seed(0)

    input_shape = (1, 128, 64, 64)
    group_count = 32
    n, c = input_shape[0], input_shape[1]

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=c)
    torch.nn.init.normal_(torch_model.weight)
    torch.nn.init.normal_(torch_model.bias)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)
    tt_model = DistributedGroupNorm.from_torch(
        torch_ref=torch_model,
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        mesh_axis=None,
        ccl_manager=ccl_manager,
    )

    # Distinct data per launch, and the first input stays alive so the second lands at a
    # different address -- otherwise the override could be a no-op and still look correct.
    torch_inputs = [torch.randn(input_shape, dtype=torch_dtype) for _ in range(2)]
    tt_inputs = [
        bf16_tensor(t.permute(0, 2, 3, 1).contiguous(), device=mesh_device, mesh_axis=cluster_axis, shard_dim=1)
        for t in torch_inputs
    ]

    outputs = []
    outputs.append(_gather_group_norm_output(tt_model(tt_inputs[0]), mesh_device, cluster_axis, n))
    entries_after_first = mesh_device.num_program_cache_entries()
    outputs.append(_gather_group_norm_output(tt_model(tt_inputs[1]), mesh_device, cluster_axis, n))
    entries_after_second = mesh_device.num_program_cache_entries()

    logger.info(f"program cache entries: {entries_after_first} after first launch, {entries_after_second} after second")
    assert entries_after_second == entries_after_first, (
        f"second launch added {entries_after_second - entries_after_first} program cache entries; it must reuse "
        "the cached program so that override_runtime_arguments is what patches the buffer addresses"
    )

    with torch.no_grad():
        for i, (torch_input, tt_out) in enumerate(zip(torch_inputs, outputs)):
            logger.info(f"launch {i} vs torch:")
            assert_quality(torch_model(torch_input), tt_out, pcc=0.999_300)
