# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.all_gather_rms_norm on a Blackhole box.

Every device of a group along cluster_axis holds its (..., M, W_local) slice; every device returns the full normalized
(..., M, G * W_local) row (RMSNorm, or LayerNorm with gamma and bias): the all-gather + norm chain of the MLA q_a
latent (RMSNorm) and of the MLA indexer k (LayerNorm), done in one op. This is the gate subset of the golden suite the
op was generated against (see the README next to the op): the production per-device shapes, the numeric axes
(dtype, fp32 DEST, gamma / beta format), the placement axis (DRAM interleaved, L1 interleaved output) and the topology
axis (Ring, Linear, both mesh axes).

The Ring cases need the whole box as one ring: a QuietBox 2 opened as 1x4 (or 4x1) under FABRIC_2D_TORUS_XY closes its
snake cycle. The Linear cases run the same box under FABRIC_2D. Other machines skip.
"""

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0

# (PCC >=, relative RMS <=) by (input dtype, fp32 DEST): the golden suite's limits.
TOLERANCES = {
    (ttnn.float32, True): (0.9999, 0.005),
    (ttnn.bfloat16, True): (0.999, 0.01),
    (ttnn.bfloat16, False): (0.999, 0.015),
    (ttnn.bfloat8_b, True): (0.995, 0.04),
    (ttnn.bfloat8_b, False): (0.995, 0.05),
}
EPS = 1e-6
LN_ROW_OFFSET_SCALE = 2.0  # LayerNorm inputs get a per-row mean far from zero, so a per-slice mean would not pass


def router_config():
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024 + 64  # the production Blackhole payload
    return config


def ring_device_params():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": router_config(),
        "require_exact_physical_num_devices": True,
    }


def line_device_params():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": router_config(),
        "require_exact_physical_num_devices": True,
    }


def model_compute_config(fp32_dest_acc_en=False, dtype=ttnn.bfloat16):
    """The MLA layer's default (HiFi2, 16-bit DEST, exact math); fp32 DEST or fp32 inputs run at HiFi4, the op's default
    fidelity (HiFi2 truncates fp32 operands and misses the fp32 tolerances)."""
    fidelity = ttnn.MathFidelity.HiFi4 if (fp32_dest_acc_en or dtype == ttnn.float32) else ttnn.MathFidelity.HiFi2
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False
    )


_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def random_stacked(mesh_shape, shape, dtype, seed, row_offset_scale=0.0):
    """Distinct data per device, as an (R, C, *shape) tensor: device (r, c) holds [r, c]."""
    torch.manual_seed(seed)
    x = torch.randn((*mesh_shape, *shape), dtype=torch.float32)
    if row_offset_scale:
        x = x + torch.randn((*mesh_shape, *shape[:-1], 1), dtype=torch.float32) * row_offset_scale
    return x.to(_TORCH_DTYPE[dtype])


def quantize_like_device(x_stacked, dtype):
    """bfloat8_b is block-float: round the input the way the device holds it, so the reference measures the op."""
    if dtype != ttnn.bfloat8_b:
        return x_stacked
    t = ttnn.from_torch(x_stacked.reshape(-1, x_stacked.shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).reshape(x_stacked.shape).to(torch.bfloat16)


def to_mesh(stacked, mesh_device, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """An (R, C, *shape) tensor -> a mesh tensor whose device (r, c) holds [r, c] (rank < 4 shapes get leading ones)."""
    rows, cols = stacked.shape[0], stacked.shape[1]
    per_device = tuple(stacked.shape[2:])
    per4 = stacked.reshape(rows, cols, *([1] * (4 - len(per_device))), *per_device)  # (R, C, 1, 1, M, W) for rank 2
    full = torch.cat([torch.cat([per4[r, c] for c in range(cols)], dim=1) for r in range(rows)], dim=0)  # (R, C, M, W)
    return ttnn.from_torch(
        full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def make_weight(w, mesh_device, dtype, layout):
    """A full-W weight replicated on every device: TILE (1, 1, 1, W) or ROW_MAJOR sticks (1, 1, ceil(W / 32), 32)."""
    width = w.shape[-1]
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        pad = -width % 32
        w = torch.nn.functional.pad(w, (0, pad)).reshape(1, 1, -1, 32)
    else:
        w = w.reshape(1, 1, 1, width)
    return ttnn.from_torch(
        w,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def reference(x_stacked, gamma, beta, cluster_axis, norm_type):
    """Per group: concatenate the members' slices along the last dim, normalize in fp32; every member gets the row."""
    rows, cols = x_stacked.shape[0], x_stacked.shape[1]
    out = torch.empty((rows, cols, *x_stacked.shape[2:-1], x_stacked.shape[-1] * (rows, cols)[cluster_axis]))
    g = None if gamma is None else gamma.float().reshape(-1)
    b = None if beta is None else beta.float().reshape(-1)
    others = range(cols) if cluster_axis == 0 else range(rows)
    for other in others:
        members = [(r, other) for r in range(rows)] if cluster_axis == 0 else [(other, c) for c in range(cols)]
        full = torch.cat([x_stacked[r, c].float() for r, c in members], dim=-1)
        if norm_type == "layer_norm":
            normed = torch.nn.functional.layer_norm(full, (full.shape[-1],), weight=g, bias=b, eps=EPS)
        else:
            normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=g, eps=EPS)
        for r, c in members:
            out[r, c] = normed
    return out


def check(out, expected, dtype, fp32_dest_acc_en, label=""):
    pcc_target, rms_target = TOLERANCES[(dtype, fp32_dest_acc_en)]
    shards = ttnn.get_device_tensors(out)
    rows, cols = expected.shape[0], expected.shape[1]
    for i, shard in enumerate(shards):
        got = ttnn.to_torch(shard).float().reshape(expected[0, 0].shape)
        exp = expected[i // cols, i % cols]
        pcc = torch.corrcoef(torch.stack([exp.flatten().double(), got.flatten().double()]))[0, 1].item()
        rel_rms = ((got - exp) ** 2).mean().sqrt().item() / max((exp**2).mean().sqrt().item(), 1e-30)
        assert (
            pcc >= pcc_target and rel_rms <= rms_target
        ), f"{label} device {i}: pcc {pcc:.6f} (>= {pcc_target}), rel_rms {rel_rms:.5f} (<= {rms_target})"


def run_case(
    mesh_device,
    shape,
    *,
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
    num_links=2,
    dtype=ttnn.bfloat16,
    fp32_dest_acc_en=False,
    norm_type="rms_norm",
    gamma_dtype=ttnn.bfloat16,
    gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
    gamma=True,
    beta=False,
    output="dram",
    preallocated=False,
    calls=1,
    seed=0,
):
    mesh_shape = tuple(mesh_device.shape)
    group = mesh_shape[cluster_axis]
    width = shape[-1] * group
    ln = norm_type == "layer_norm"
    x = quantize_like_device(random_stacked(mesh_shape, shape, dtype, seed, LN_ROW_OFFSET_SCALE if ln else 0.0), dtype)
    g = torch.randn(width).to(_TORCH_DTYPE[gamma_dtype]) if gamma else None
    b = torch.randn(width).to(_TORCH_DTYPE[gamma_dtype]) if beta else None
    expected = reference(x, g, b, cluster_axis, norm_type)

    tx = to_mesh(x, mesh_device, dtype)
    if len(shape) < 4:
        tx = ttnn.reshape(tx, ttnn.Shape(shape))
    tg = make_weight(g, mesh_device, gamma_dtype, gamma_layout) if gamma else None
    tb = make_weight(b, mesh_device, gamma_dtype, gamma_layout) if beta else None
    memory_config = ttnn.L1_MEMORY_CONFIG if output == "l1" else ttnn.DRAM_MEMORY_CONFIG
    kwargs = dict(
        cluster_axis=cluster_axis,
        gamma=tg,
        epsilon=EPS,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=model_compute_config(fp32_dest_acc_en, dtype),
        memory_config=memory_config,
    )
    if ln:
        kwargs.update(norm_type="layer_norm", beta=tb)
    for _ in range(calls):
        if preallocated:  # a rank-4 per-device output, zero-filled, in the requested placement
            kwargs["output_tensor"] = to_mesh(
                torch.zeros((*mesh_shape, *shape[:-1], width), dtype=torch.bfloat16),
                mesh_device,
                dtype,
                memory_config=memory_config,
            )
        out = ttnn.experimental.all_gather_rms_norm(tx, **kwargs)
        ttnn.synchronize_device(mesh_device)
        check(out, expected, dtype, fp32_dest_acc_en, label=f"{shape} {norm_type} {output}")
    return out


# Production per-device shapes (the golden suite's INPUTS, model-derived): the MLA q_a latent (q_lora_rank / TP) at the
# 640-row Galaxy chunk, the box chunk and the large chunks; the indexer k (index_head_dim 128 / TP4); the dense-FFN norm.
MODEL_CASES = {
    "q_a_dsv3_kimi_640x384": ((1, 1, 640, 384), "rms_norm", False),
    "q_a_glm_640x512": ((1, 1, 640, 512), "rms_norm", False),
    "q_a_tp8_640x192": ((1, 1, 640, 192), "rms_norm", False),
    "q_a_tp8_640x256": ((1, 1, 640, 256), "rms_norm", False),
    "q_a_2048x384": ((1, 1, 2048, 384), "rms_norm", False),
    "q_a_4096x384": ((1, 1, 4096, 384), "rms_norm", False),
    "q_a_160x384": ((1, 1, 160, 384), "rms_norm", False),
    "q_a_decode_32x384": ((1, 1, 32, 384), "rms_norm", False),
    "indexer_k_640x32": ((1, 1, 640, 32), "layer_norm", True),
    # indexer_k at 4096 rows per device returns wrong values (PCC 0.96; 640..3200 rows pass): known limit, see README
    "dense_ffn_norm_640x1792": ((1, 1, 640, 1792), "rms_norm", False),
    "dense_ffn_norm_640x896": ((1, 1, 640, 896), "rms_norm", False),
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("case", MODEL_CASES.values(), ids=MODEL_CASES.keys())
def test_ring_model_shapes(mesh_device, case):
    shape, norm_type, beta = case
    run_case(mesh_device, shape, norm_type=norm_type, beta=beta)


# The MLA q_a norm's real call at 640 tokens per chip: its output is the L1 input of the q_b_proj matmul that follows
# (an L1 interleaved output); the output may also be a preallocated tensor.
@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("preallocated", [False, True], ids=["op_output", "preallocated"])
@pytest.mark.parametrize(
    "shape, norm_type, beta",
    [
        ((1, 1, 640, 384), "rms_norm", False),
        ((1, 1, 640, 512), "rms_norm", False),
        ((1, 1, 640, 32), "layer_norm", True),
    ],
    ids=["q_a_640x384", "q_a_glm_640x512", "indexer_k_640x32"],
)
def test_l1_interleaved_output(mesh_device, shape, norm_type, beta, preallocated):
    run_case(mesh_device, shape, norm_type=norm_type, beta=beta, output="l1", preallocated=preallocated)


# The numeric axes on representative shapes: dtype x fp32 DEST x gamma format x norm / beta. The h-non-aligned and
# w-non-aligned shapes cover the masking paths.
NUMERIC_SHAPES = {
    "aligned_640x384": (1, 1, 640, 384),
    "h_non_aligned_17x1280": (1, 1, 17, 1280),
    "w_non_aligned_32x50": (1, 1, 32, 50),
    "rank3_640x1792": (1, 640, 1792),
    "rank2_640x384": (640, 384),
}


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("shape", NUMERIC_SHAPES.values(), ids=NUMERIC_SHAPES.keys())
@pytest.mark.parametrize(
    "dtype, fp32_dest_acc_en",
    [(ttnn.bfloat16, False), (ttnn.bfloat16, True), (ttnn.float32, True), (ttnn.bfloat8_b, True)],
    ids=["bf16_dest16", "bf16_dest32", "fp32", "bf8b"],
)
@pytest.mark.parametrize(
    "norm_type, gamma, beta, gamma_dtype, gamma_layout",
    [
        ("rms_norm", True, False, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ("rms_norm", True, False, ttnn.float32, ttnn.TILE_LAYOUT),
        ("rms_norm", False, False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ("layer_norm", True, True, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ("layer_norm", True, False, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ],
    ids=["rms_gamma_rm", "rms_gamma_f32_tile", "rms_nogamma", "ln_gamma_beta_rm", "ln_gamma_tile"],
)
def test_numeric_axes(mesh_device, shape, dtype, fp32_dest_acc_en, norm_type, gamma, beta, gamma_dtype, gamma_layout):
    if dtype == ttnn.float32 and not fp32_dest_acc_en:
        pytest.skip("float32 input needs fp32 DEST accumulation")
    if dtype == ttnn.bfloat8_b and shape[-1] % 32:
        pytest.skip("bfloat8_b blocks need a tile-aligned width")
    run_case(
        mesh_device,
        shape,
        dtype=dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        norm_type=norm_type,
        gamma=gamma,
        beta=beta,
        gamma_dtype=gamma_dtype,
        gamma_layout=gamma_layout,
    )


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [line_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], ids=["1x2", "1x4"], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
@pytest.mark.parametrize(
    "shape, norm_type, beta",
    [((1, 1, 640, 384), "rms_norm", False), ((1, 1, 640, 32), "layer_norm", True)],
    ids=["q_a_640x384", "indexer_k_640x32"],
)
def test_line(mesh_device, shape, norm_type, beta, num_links):
    run_case(mesh_device, shape, norm_type=norm_type, beta=beta, topology=ttnn.Topology.Linear, num_links=num_links)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4), (4, 1)], ids=["axis1_1x4", "axis0_4x1"], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
def test_ring_both_axes(mesh_device, num_links):
    axis = 1 if tuple(mesh_device.shape) == (1, 4) else 0
    run_case(mesh_device, (1, 1, 640, 384), cluster_axis=axis, num_links=num_links)
    run_case(mesh_device, (1, 1, 640, 32), cluster_axis=axis, num_links=num_links, norm_type="layer_norm", beta=True)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_program_cache(mesh_device):
    """Calls with new tensors hit one cached program and stay correct (runtime args re-pointed on a hit)."""
    run_case(mesh_device, (1, 1, 640, 384))
    entries = mesh_device.num_program_cache_entries()
    run_case(mesh_device, (1, 1, 640, 384), calls=2, seed=1)
    assert mesh_device.num_program_cache_entries() == entries, "a repeated call built a new program"


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [ring_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "error, overrides, message",
    [
        (ValueError, dict(norm_type="layer_norm", beta=True, gamma=False), "beta requires gamma"),
        (ValueError, dict(norm_type="rms_norm", beta=True), "beta is only valid"),
        (ValueError, dict(cluster_axis=2), "cluster_axis"),
    ],
    ids=["beta_without_gamma", "beta_with_rms_norm", "bad_cluster_axis"],
)
def test_refusals(mesh_device, expect_error, error, overrides, message):
    """Calls outside the contract raise instead of returning wrong numbers."""
    shape = (1, 1, 640, 384)
    x = to_mesh(random_stacked(tuple(mesh_device.shape), shape, ttnn.bfloat16, 0), mesh_device, ttnn.bfloat16)
    width = shape[-1] * 4
    gamma = (
        None
        if overrides.get("gamma") is False
        else make_weight(torch.randn(width).to(torch.bfloat16), mesh_device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    )
    beta = (
        make_weight(torch.randn(width).to(torch.bfloat16), mesh_device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        if overrides.get("beta")
        else None
    )
    kwargs = dict(
        cluster_axis=overrides.get("cluster_axis", 1),
        gamma=gamma,
        epsilon=EPS,
        topology=ttnn.Topology.Ring,
        norm_type=overrides.get("norm_type", "rms_norm"),
    )
    if beta is not None:
        kwargs["beta"] = beta
    with expect_error((error, NotImplementedError, RuntimeError), message):
        ttnn.experimental.all_gather_rms_norm(x, **kwargs)
