# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for BEVFormer's fused multi-scale deformable attention.

Same shapes as ``tests/pcc/test_fused_msda.py`` -- nuscenes tiny
(Q=900, H=4, L=1) and base (Q=2500, H=8, L=4) -- and the same structure as
``test_encoder_layer_perf.py``: a PCC gate that doubles as the warmup, then a signposted
region so the report covers already-compiled, already-cached programs.

Two measured surfaces, because the interesting question is which of the two the
time is in:

``test_fused_msda_module_perf``
    The whole ``TTMSDeformableAttention`` forward: value/attn/offset Linears,
    softmax, the fused op, ``output_proj`` and the residual. This is what the
    encoder actually pays.

``test_fused_msda_kernel_perf``
    Only ``ttnn.experimental.fused_msda_from_offsets``. The projections and the
    softmax run once, before the signpost, so the measured region is the op and
    nothing else. Three variants isolate the design questions that are still
    open on #55198:

    * ``packed_multi``  -- the production contract: packed value/offsets/attn,
      one call covering all levels.
    * ``canonical_multi`` -- the rank-4 / rank-6 / rank-5 operand forms the op
      also accepts. The delta against ``packed_multi`` is what byte-offset
      addressing (#55232-#55236) buys, measured rather than assumed.
    * ``packed_per_level`` -- L single-level calls accumulated on device, which
      is the contract #55201 replaced. Operands are pre-sliced on the host
      outside the measured region, so this is a *lower bound* on the per-level
      cost: it counts L launches plus L-1 adds, but not the slicing that a real
      per-level path would also pay.

Read the signposted rows only, and attribute the fused op separately from the
Linears around it. For kernel work ``CB-COMPUTE-WAIT-FRONT`` is the number that
decides #55231: while the reader derives sampling geometry in soft float on a
core with no FPU, that counter approaches the op's whole duration and no amount
of compute-side work matters.
"""

import subprocess
from functools import partial

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.bevformer.config import DeformableAttentionConfig
from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.tests.test_utils import check_with_pcc
from models.experimental.bevformer.tt.model_preprocessing import create_ms_deformable_attention_parameters
from models.experimental.bevformer.tt.tt_ms_deformable_attention import (
    TTMSDeformableAttention,
    _spatial_shapes_list,
)

DEVICE_PERF_ITERS = 1

# (config_name, batch_size, num_queries) -- mirrors the PCC file's parametrization.
WORKLOADS = [
    ("nuscenes_tiny", 1, 900),
    ("nuscenes_base", 1, 2500),
]

KERNEL_VARIANTS = ["packed_multi", "canonical_multi", "packed_per_level"]


def _head_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _build(device, config_name, batch_size, num_queries):
    """Reference module, preprocessed TT params and the torch inputs.

    Deliberately the same construction as the PCC file: a perf number measured
    on a different shape than the correctness gate is not comparable to it.
    """
    preset_config = get_preset_config(config_name)
    assert preset_config is not None, f"Configuration '{config_name}' not found"

    model_config = preset_config.model_config
    num_levels = model_config.num_levels
    spatial_shapes = torch.tensor(preset_config.dataset_config.spatial_shapes[:num_levels], dtype=torch.long)
    num_keys = int(spatial_shapes.prod(dim=1).sum().item())

    config = DeformableAttentionConfig(
        embed_dims=model_config.embed_dims,
        num_heads=model_config.num_heads,
        num_levels=num_levels,
        num_points=model_config.num_points,
    )

    query = torch.randn(batch_size, num_queries, config.embed_dims, dtype=torch.float32)
    value = torch.randn(batch_size, num_keys, config.embed_dims, dtype=torch.float32)
    reference_points = torch.rand(batch_size, num_queries, num_levels, 2, dtype=torch.float32)

    ref_model = MSDeformableAttention(config)
    ref_model.eval()

    tt_parameters = create_ms_deformable_attention_parameters(
        torch_model=ref_model, device=device, config=config, dtype=ttnn.float32
    )

    logger.info(
        f"{config_name}: Q={num_queries} heads={config.num_heads} D={config.embed_dims // config.num_heads} "
        f"L={num_levels} P={config.num_points} keys={num_keys} "
        f"work_units={batch_size * config.num_heads * ((num_queries + 31) // 32)} "
        f"reduction={4 * num_levels * config.num_points}"
    )

    return config, ref_model, tt_parameters, (query, value, reference_points, spatial_shapes)


def _to_device(device, torch_inputs):
    query, value, reference_points, _ = torch_inputs
    return (
        ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
    )


def _tt_model(device, config, tt_parameters, spatial_shapes):
    return TTMSDeformableAttention(
        config=config,
        device=device,
        params=tt_parameters,
        spatial_shapes=spatial_shapes,
    )


def _packed_operands(model, tt_query, tt_value, tt_reference_points):
    """The op's four inputs, in the packed ROW_MAJOR forms the module produces.

    This is the projection/softmax prefix of ``TTMSDeformableAttention.forward``
    lifted out so it runs before the signpost. Keeping it identical to the
    module matters: the packed shapes are what make the reader's byte-offset
    addressing legal, so a different prefix would measure a different op.
    """
    bs, num_queries, _ = tt_query.shape

    value = ttnn.to_layout(tt_value, ttnn.TILE_LAYOUT)
    value = ttnn.linear(value, model.params.value_proj.weight, bias=model.params.value_proj.bias)

    query = ttnn.to_layout(tt_query, ttnn.TILE_LAYOUT)

    attn = ttnn.linear(query, model.params.attention_weights.weight, bias=model.params.attention_weights.bias)
    attn = ttnn.reshape(attn, (bs, num_queries, model.num_heads, model.num_levels * model.num_points))
    attn = ttnn.softmax(attn, dim=-1)

    offsets = ttnn.linear(
        query, model.params.sampling_offsets.weight, bias=getattr(model.params.sampling_offsets, "bias", None)
    )
    offsets = ttnn.to_layout(offsets, ttnn.ROW_MAJOR_LAYOUT)
    offsets = ttnn.reshape(offsets, (bs, num_queries, model.num_heads, model.num_levels * model.num_points * 2))

    value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
    attn = ttnn.to_layout(attn, ttnn.ROW_MAJOR_LAYOUT)
    refs = ttnn.to_layout(tt_reference_points, ttnn.ROW_MAJOR_LAYOUT)
    return value, refs, offsets, attn


def _canonical_operands(value, offsets, attn, model):
    """The rank-4 / rank-6 / rank-5 forms, for the packed-vs-canonical A/B.

    Every split here widens the operand into an axis the op would otherwise
    reach by byte offset, and narrows its ROW_MAJOR page in the process: a
    2-wide offsets page is 4 B of data in a 32 B allocation. That is the tax
    #55232 is about, and the reason to measure it rather than argue it.
    """
    bs, num_keys, _ = value.shape
    num_queries = offsets.shape[1]
    return (
        ttnn.reshape(value, (bs, num_keys, model.num_heads, model.head_dim)),
        ttnn.reshape(offsets, (bs, num_queries, model.num_heads, model.num_levels, model.num_points, 2)),
        ttnn.reshape(attn, (bs, num_queries, model.num_heads, model.num_levels, model.num_points)),
    )


def _per_level_operands(device, value, offsets, attn, model, spatial_shapes):
    """Host-side per-level slices of the packed operands, one set per level.

    Sliced through torch rather than ``ttnn.slice`` on purpose: this runs
    outside the measured region, and a host round trip has no device cost to
    attribute, whereas a device slice would land inside the A/B and confuse the
    launch-overhead question #55201 asks.

    ``reference_points`` is not sliced. In ``reference_mode="pillar"`` the
    reference index is ``p % R``, independent of the level, so each single-level
    call needs the full (B, Q, R, 2) tensor -- which is also why the per-level
    sum is numerically the same as one multi-level call.
    """
    value_t = ttnn.to_torch(value)
    offsets_t = ttnn.to_torch(offsets)
    attn_t = ttnn.to_torch(attn)

    p = model.num_points
    per_level = []
    start = 0
    for level, (height, width) in enumerate(_spatial_shapes_list(spatial_shapes)):
        keys = height * width
        per_level.append(
            (
                (height, width),
                ttnn.from_torch(
                    value_t[:, start : start + keys, :].contiguous(),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                ttnn.from_torch(
                    offsets_t[..., level * p * 2 : (level + 1) * p * 2].contiguous(),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                ttnn.from_torch(
                    attn_t[..., level * p : (level + 1) * p].contiguous(),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            )
        )
        start += keys
    return per_level


def _call_one(value, refs, offsets, attn, spatial_shapes):
    return ttnn.experimental.fused_msda_from_offsets(
        value,
        refs,
        offsets,
        attn,
        _spatial_shapes_list(spatial_shapes),
        reference_mode="pillar",
    )


def _call_per_level(refs, per_level):
    """L single-level launches, accumulated on device. The pre-#55201 shape."""
    acc = None
    for (height, width), value_l, offsets_l, attn_l in per_level:
        out = ttnn.experimental.fused_msda_from_offsets(
            value_l,
            refs,
            offsets_l,
            attn_l,
            [(height, width)],
            reference_mode="pillar",
        )
        acc = out if acc is None else ttnn.add(acc, out)
    return acc


def _signposted(device, op_fn, iters=DEVICE_PERF_ITERS):
    ttnn.synchronize_device(device)
    # Drain and reset the profiler buffers so the signposted region starts empty;
    # the warmup call's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    outputs = []
    signpost("start")
    for _ in range(iters):
        outputs.append(op_fn())
        ttnn.synchronize_device(device)
    signpost("stop")
    for out in outputs:
        ttnn.deallocate(out)


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("config_name, batch_size, num_queries", WORKLOADS)
@pytest.mark.parametrize("expected_pcc", [0.999])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_fused_msda_module_perf(
    device,
    config_name,
    batch_size,
    num_queries,
    expected_pcc,
    reset_seeds,
    ensure_gc,
):
    """Whole-module device time: what the encoder pays per MSDA call."""
    logger.info(f"device-perf run of commit {_head_sha()} -- module {config_name} q={num_queries}")

    config, ref_model, tt_parameters, torch_inputs = _build(device, config_name, batch_size, num_queries)
    query, value, reference_points, spatial_shapes = torch_inputs
    tt_query, tt_value, tt_refs = _to_device(device, torch_inputs)
    model = _tt_model(device, config, tt_parameters, spatial_shapes)

    def op_fn():
        return model(query=tt_query, value=tt_value, reference_points=tt_refs)

    expected = ref_model(query, value, reference_points=reference_points, spatial_shapes=spatial_shapes)

    # Doubles as the warmup: this call compiles the kernels and fills the program
    # cache, so the signposted iterations already run at steady state.
    tt_output = op_fn()
    passed, message = check_with_pcc(expected, ttnn.to_torch(tt_output, dtype=torch.float32), expected_pcc)
    assert passed, f"PCC check failed: {message}"
    logger.info(f"PCC gate: {message}")
    ttnn.deallocate(tt_output)

    _signposted(device, op_fn)


@torch.no_grad()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("config_name, batch_size, num_queries", WORKLOADS)
@pytest.mark.parametrize("variant", KERNEL_VARIANTS)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32 * 1024}], indirect=True)
def test_fused_msda_kernel_perf(
    device,
    config_name,
    batch_size,
    num_queries,
    variant,
    reset_seeds,
    ensure_gc,
):
    """Fused op alone, with the operand-form and level-count A/Bs."""
    logger.info(f"device-perf run of commit {_head_sha()} -- kernel {config_name} q={num_queries} variant={variant}")

    config, _, tt_parameters, torch_inputs = _build(device, config_name, batch_size, num_queries)
    spatial_shapes = torch_inputs[3]

    if variant == "packed_per_level" and config.num_levels == 1:
        pytest.skip("per-level is the same single call as multi-level when num_levels == 1")

    tt_query, tt_value, tt_refs = _to_device(device, torch_inputs)
    model = _tt_model(device, config, tt_parameters, spatial_shapes)

    value_p, refs, offsets_p, attn_p = _packed_operands(model, tt_query, tt_value, tt_refs)

    if variant == "packed_multi":
        op_fn = partial(_call_one, value_p, refs, offsets_p, attn_p, spatial_shapes)
    elif variant == "canonical_multi":
        value_c, offsets_c, attn_c = _canonical_operands(value_p, offsets_p, attn_p, model)
        op_fn = partial(_call_one, value_c, refs, offsets_c, attn_c, spatial_shapes)
    elif variant == "packed_per_level":
        per_level = _per_level_operands(device, value_p, offsets_p, attn_p, model, spatial_shapes)
        op_fn = partial(_call_per_level, refs, per_level)
    else:
        pytest.fail(f"unknown variant {variant}")

    # Gate every variant against the production contract rather than against the
    # torch reference: all three are meant to compute the same thing, and a
    # variant that quietly samples different points would otherwise show up as a
    # speedup. Doubles as the warmup.
    golden = _call_one(value_p, refs, offsets_p, attn_p, spatial_shapes)
    warmup = op_fn()
    expected_shape = [batch_size, num_queries, config.embed_dims]
    assert (
        list(warmup.shape) == expected_shape
    ), f"variant {variant} produced {list(warmup.shape)}, want {expected_shape}"
    passed, message = check_with_pcc(
        ttnn.to_torch(golden, dtype=torch.float32),
        ttnn.to_torch(warmup, dtype=torch.float32),
        pcc=0.999,
    )
    assert passed, f"variant {variant} diverged from packed_multi: {message}"
    logger.info(f"variant gate vs packed_multi: {message}")
    ttnn.deallocate(golden)
    ttnn.deallocate(warmup)

    _signposted(device, op_fn)
