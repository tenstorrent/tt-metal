# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in (TEST_SDPA_DIT_PARITY=1): tt_dit SDPA recipes vs each model's legacy SDPA setup.

Every SDPA call in models/tt_dit selects a named recipe on Blackhole (with op-selected blocking). This
file keeps the legacy configuration each call used before that change -- HiFi2, BF16 dest,
exp_approx_mode=False, and the model's tuned chunk sizes and grid -- and runs it through ttnn directly,
next to the recipe candidates, on the model's per-device production shapes.

For each case and variant it records
- accuracy against an FP32 torch reference on a sample of query rows (two heads), as relative L2 and
  PCC, on two input regimes: unit-variance Q/K/V ("unit", scores ~N(0, 1)) and Q/K scaled by 2
  ("sharp", scores ~N(0, 16), peaked softmax);
- the median trace wall time (unit regime), so ``recipe_over_legacy`` = recipe / legacy time.

Two connected Blackholes as a 1x2 FABRIC_1D_RING mesh (the ring/exp-ring perf fixture). Dense and
joint cases run a single chip's shape (replicated on both chips); ring and exp ring run a two-chip
ring with the per-device length a larger mesh gives each chip (heads = heads per TP shard). One
``PARITY {...}`` JSON line per test is printed for collection.
"""

import json
import math
import os
from collections import namedtuple

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .test_sdpa_recipe_ring_perf import time_trace

pytestmark = pytest.mark.skipif(os.getenv("TEST_SDPA_DIT_PARITY") != "1", reason="Opt-in tt_dit recipe parity")

VARIANTS = os.getenv("PARITY_VARIANTS", "legacy,FAST,COMPENSATED,BALANCED").split(",")
SAMPLE_ROWS = 512

# op: dense | joint | ring | ring_cross | exp. q/k: per-device primary Q rows / K rows (ring: local
# rows per chip, K = Q). joint: joint (prompt) rows. legacy: (q_chunk, k_chunk[, grid_x]) the model
# used before recipes. grid: SDPA worker grid relative to the 11x10 P150b grid. ccl: ring CCL layout.
Case = namedtuple("Case", "name op heads q k joint d legacy grid ccl")
CASES = [
    # blocks/attention.py (FLUX.1, Qwen-Image, Motif): grid (x, y-1), row CCL.
    Case("flux1_joint_tp2", "joint", 12, 4096, 4096, 512, 128, (128, 512), "y-1", None),
    Case("flux1_ring_2x2", "ring", 12, 2048, 2048, 512, 128, (128, 512), "y-1", "row"),
    Case("flux1_ring_8x4", "ring", 6, 512, 512, 512, 128, (64, 512), "y-1", "row"),
    Case("motif_joint_tp2", "joint", 16, 4096, 4096, 256, 64, (128, 1024), "y-1", None),
    # blocks/attention_opt.py (FLUX.2): dense (x, y-1); ring grid from get_ring_sdpa_core_grid.
    Case("flux2_joint_tp2", "joint", 24, 4096, 4096, 512, 128, (128, 512), "y-1", None),
    Case("flux2_ring_2x2", "ring", 24, 2048, 2048, 512, 128, (128, 512), "y-1", "row"),
    Case("flux2_ring_4x8_1024", "ring", 6, 1024, 1024, 512, 128, (128, 256), "y-5", "row"),
    Case("flux2_ring_4x8_2048", "ring", 6, 4096, 4096, 512, 128, (256, 512), "y-1", "row"),
    # attention_sd35.py (D64, 38 heads padded to 40): grid (x, y-1), row CCL.
    Case("sd35_joint_tp2", "joint", 20, 4096, 4096, 352, 64, (256, 512), "y-1", None),
    Case("sd35_ring_2x2", "ring", 20, 2048, 2048, 352, 64, (256, 512), "y-1", "row"),
    Case("sd35_ring_4x4", "ring", 10, 1024, 1024, 352, 64, (128, 512), "y-1", "row"),
    # attention_mochi.py: joint on the full grid (256, 512); ring (x, y-1) row CCL.
    Case("mochi_joint_tp2", "joint", 12, 8192, 8192, 256, 128, (256, 512), "full", None),
    Case("mochi_ring_2x2", "ring", 12, 11136, 11136, 256, 128, (128, 512), "y-1", "row"),
    Case("mochi_ring_8x4", "ring", 6, 5568, 5568, 256, 128, (128, 512), "y-1", "row"),
    # attention_wan.py: dense self/cross on the full grid (256, 256); ring (x-1, y) column CCL.
    Case("wan_self_dense_tp2", "dense", 20, 8192, 8192, 0, 128, (256, 256), "full", None),
    Case("wan_cross_480p_2x2", "dense", 20, 8192, 512, 0, 128, (256, 256), "full", None),
    Case("wan_cross_720p_8x4", "dense", 10, 9472, 512, 0, 128, (256, 256), "full", None),
    Case("wan_ring_480p_2x2", "ring", 20, 8192, 8192, 0, 128, (128, 512), "x-1", "col"),
    Case("wan_ring_480p_8x4", "ring", 10, 4096, 4096, 0, 128, (288, 512), "x-1", "col"),
    Case("wan_ring_720p_8x4", "ring", 10, 9472, 9472, 0, 128, (288, 512), "x-1", "col"),
    Case("wan_exp_720p_4x32", "exp", 10, 2368, 2368, 0, 128, (224, 512), "full", None),
    # attention_ltx.py: video ring per-N tuned; text/A2V cross per-shape tuned; audio D64.
    Case("ltx_ring_s1_8x4", "ring", 8, 1216, 1216, 0, 128, (96, 256), "x-1", "col"),
    Case("ltx_ring_s2_8x4", "ring", 8, 4864, 4864, 0, 128, (192, 512), "x-1", "col"),
    Case("ltx_ring_2x2", "ring", 16, 4864, 4864, 0, 128, (128, 512), "x-1", "col"),
    Case("ltx_text_cross_s2", "dense", 8, 4864, 32, 0, 128, (192, 128), "full", None),
    Case("ltx_a2v_cross_s2", "dense", 8, 4864, 256, 0, 128, (192, 256), "full", None),
    Case("ltx_audio_self_d64", "dense", 8, 256, 256, 0, 64, (256, 256), "full", None),
    Case("ltx_v2a_ring_cross_8x4", "ring_cross", 8, 32, 4864, 0, 64, (32, 256), "x-1", "col"),
    # transformer_ideogram4.py (D256): dense full grid (128, 256); ring (x, y-1) row CCL.
    Case("ideogram4_dense_tp2", "dense", 10, 4096, 4096, 0, 256, (128, 256), "full", None),
    Case("ideogram4_ring_2x2", "ring", 10, 2048, 2048, 0, 256, (128, 256), "y-1", "row"),
    # attention_minimax_h3.py: ring (x-1, y) column CCL at measured per-length chunks; exp ring search.
    Case("h3_ring_5s", "ring", 14, 4768, 4768, 0, 128, (320, 384), "x-1", "col"),
    Case("h3_ring_10s", "ring", 14, 9216, 9216, 0, 128, (256, 512), "x-1", "col"),
    Case("h3_exp_4x32", "exp", 14, 1216, 1216, 0, 128, (128, 512, 11), "full", None),
    Case("h3_refiner_dense", "dense", 14, 256, 256, 0, 128, (256, 256), "full", None),
]


@pytest.fixture(scope="module")
def parity_mesh():
    """1x2 FABRIC_1D_RING mesh at the default worker L1 (the models' pipelines do not shrink it)."""
    if not is_blackhole() or ttnn.GetNumAvailableDevices() != 2:
        pytest.skip("requires two connected Blackholes")
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    mesh = manager = None
    try:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), trace_region_size=33554432)
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        subdevice = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)]
        yield mesh, subdevice, semaphores, grid
    finally:
        if mesh is not None:
            if manager is not None:
                mesh.reset_sub_device_stall_group()
                mesh.clear_loaded_sub_device_manager()
                mesh.remove_sub_device_manager(manager)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def legacy_compute_config(fidelity=ttnn.MathFidelity.HiFi2):
    """Every tt_dit SDPA call's legacy compute config: HiFi2, BF16 dest, no approximate math.

    The Wan/LTX quant presets swap the fidelity (``legacy_lofi``) or cast the inputs to BFP8
    (``legacy_bfp8``) on the self-attention ring SDPA.
    """
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=False)


# Recipe LOW_PRECISION variants: KV storage dtype.
E_KV = {"LOW_PRECISION_bf16": ttnn.bfloat16, "LOW_PRECISION_bfp8": ttnn.bfloat8_b, "LOW_PRECISION_bfp4": ttnn.bfloat4_b}


def prepare_variant(tensors, variant):
    """Variant-specific input handling for one Q/K/V triple (legacy quant cast or E preparation)."""
    if variant == "legacy_bfp8":
        return [ttnn.typecast(x, ttnn.bfloat8_b) for x in tensors]
    if variant in E_KV:
        q, k, v = tensors
        prepare = ttnn.transformer.prepare_sdpa_input
        return [
            prepare(q, is_query=True),
            prepare(k, is_query=False, dtype=E_KV[variant]),
            prepare(v, is_query=False, dtype=E_KV[variant]),
        ]
    return tensors


def worker_grid(full, grid):
    return {
        "full": (full.x, full.y),
        "x-1": (full.x - 1, full.y),
        "y-1": (full.x, full.y - 1),
        "y-5": (full.x, full.y - 5),
    }[grid]


def variant_kwargs(case, variant, full):
    grid = worker_grid(full, case.grid)
    if variant.startswith("legacy") or variant == "FAST_legacy_chunks":
        q, k = case.legacy[:2]
        if len(case.legacy) > 2:
            grid = (case.legacy[2], full.y)
    if variant == "FAST_legacy_chunks":
        # Decomposition: FAST pinned to the legacy tuned chunks and grid (isolates op-selected blocking).
        config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q, k_chunk_size=k)
        return {"program_config": config, "precision": ttnn.SDPAPrecision.FAST, "inputs_prepared": False}
    if variant.startswith("legacy"):
        # Decomposition: legacy_approx is the legacy config with the approximate exponential
        # (isolates exact vs approximate exp from the kernel path and blocking).
        config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=q,
            k_chunk_size=k,
            exp_approx_mode=variant == "legacy_approx",
        )
        fidelity = ttnn.MathFidelity.LoFi if variant == "legacy_lofi" else ttnn.MathFidelity.HiFi2
        return {"program_config": config, "compute_kernel_config": legacy_compute_config(fidelity)}
    config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid)
    if variant in E_KV:
        return {"program_config": config, "precision": ttnn.SDPAPrecision.LOW_PRECISION, "inputs_prepared": True}
    return {"program_config": config, "precision": getattr(ttnn.SDPAPrecision, variant), "inputs_prepared": False}


def host_inputs(case, regime):
    ring = case.op in ("ring", "ring_cross", "exp")
    n_q = 2 * case.q if ring else case.q
    n_k = 2 * case.k if ring else case.k
    gen = torch.Generator().manual_seed(20260924)
    scale = 2.0 if regime == "sharp" else 1.0
    shape = lambda n: (1, case.heads, n, case.d)  # noqa: E731
    q = torch.randn(shape(n_q), generator=gen) * scale
    k = torch.randn(shape(n_k), generator=gen) * scale
    v = torch.randn(shape(n_k), generator=gen)
    joint = None
    if case.joint:
        joint = [
            torch.randn(shape(case.joint), generator=gen) * scale,
            torch.randn(shape(case.joint), generator=gen) * scale,
            torch.randn(shape(case.joint), generator=gen),
        ]
        joint = [x.bfloat16() for x in joint]
    return [x.bfloat16() for x in (q, k, v)], joint


def reference_rows(q, k, v, joint, rows, heads):
    """FP32 attention for the sampled (concatenated primary+joint) query rows of the sampled heads."""
    if joint is not None:
        q, k, v = (torch.cat([a, b], dim=2) for a, b in zip((q, k, v), joint))
    q = q[:, heads][:, :, rows].float()
    scores = q @ k[:, heads].float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    return torch.softmax(scores, dim=-1) @ v[:, heads].float()


def metrics(out, ref):
    out, ref = out.double().flatten(), ref.double().flatten()
    rel_l2 = ((out - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([out, ref]))[0, 1].item()
    return rel_l2, pcc


def build(mesh, case, host, joint, variant, extras):
    subdevice, semaphores, full = extras
    replicate = ttnn.ReplicateTensorToMesh(mesh)
    ring = case.op in ("ring", "ring_cross", "exp")
    upload = lambda x, mapper: ttnn.from_torch(
        x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
    )  # noqa: E731
    mapper = ttnn.ShardTensorToMesh(mesh, dim=2) if ring else replicate
    inputs = prepare_variant([upload(x, mapper) for x in host], variant)
    joint_inputs = prepare_variant([upload(x, replicate) for x in joint], variant) if joint is not None else None
    kwargs = variant_kwargs(case, variant, full)
    if case.op == "dense":
        return lambda: ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, **kwargs)
    if case.op == "joint":
        return lambda: ttnn.transformer.joint_scaled_dot_product_attention(
            *inputs, *joint_inputs, joint_strategy="rear", **kwargs
        )
    backing = [
        ttnn.allocate_tensor_on_device(list(x.shape), t.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x, t in zip(host[1:], inputs[1:])
    ]
    joint_args = joint_inputs if joint_inputs is not None else [None, None, None]
    if case.op == "exp":
        return lambda: ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
            *inputs,
            persistent_output_buffer_k=backing[0],
            persistent_output_buffer_v=backing[1],
            joint_strategy="rear",
            logical_n=host[1].shape[2],
            dim=2,
            multi_device_global_semaphore=semaphores[:2],
            num_links=2,
            cluster_axis=1,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=subdevice,
            num_workers_per_link=full.y // 2,
            num_buffers_per_channel=16,
            **kwargs,
        )
    grid = kwargs["program_config"].compute_with_storage_grid_size
    ccl = (
        {"ccl_core_grid_offset": (grid.x, 0), "use_column_major_ccl": True}
        if case.ccl == "col"
        else {"ccl_core_grid_offset": (0, grid.y)}
    )
    return lambda: ttnn.transformer.ring_joint_scaled_dot_product_attention(
        *inputs,
        *joint_args,
        persistent_output_buffer_k=backing[0],
        persistent_output_buffer_v=backing[1],
        joint_strategy="rear",
        logical_n=host[1].shape[2],
        is_cross=case.op == "ring_cross",
        dim=2,
        multi_device_global_semaphore=semaphores,
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh,
        topology=ttnn.Topology.Linear,
        subdevice_id=subdevice,
        **ccl,
        **kwargs,
    )


def resolved_chunks(case, variant, host, joint, mesh, full):
    """The op-selected blocking for a recipe variant, as Q/K[/grid x]."""
    T = ttnn._ttnn.operations.transformer
    ring = case.op in ("ring", "ring_cross", "exp")
    mapper = ttnn.ShardTensorToMesh(mesh, dim=2) if ring else ttnn.ReplicateTensorToMesh(mesh)
    q, k = (ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in host[:2])
    config = variant_kwargs(case, variant, full)["program_config"]
    op = {"dense": "dense", "joint": "joint", "ring": "ring", "ring_cross": "ring", "exp": "exp_ring"}[case.op]
    kwargs = {"program_config": config}
    precision = ttnn.SDPAPrecision.LOW_PRECISION if variant in E_KV else getattr(ttnn.SDPAPrecision, variant)
    if variant in E_KV:
        k = ttnn.typecast(k, E_KV[variant]) if E_KV[variant] != ttnn.bfloat16 else k
    if ring:
        kwargs["ring_size"] = 2
    if joint is not None:
        jq, jk = (
            ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            for x in joint[:2]
        )
        kwargs.update(joint_q=jq, joint_k=jk)
    try:
        r = T._sdpa_recipe_resolved_program_config(op, precision, q, k, **kwargs)
    except (RuntimeError, TypeError) as error:
        return f"unresolved: {str(error)[:80]}"
    return f"Q{r.q_chunk_size}/K{r.k_chunk_size}/x{r.compute_with_storage_grid_size.x}"


def outputs_to_host(mesh, case, result):
    ring = case.op in ("ring", "ring_cross", "exp")
    if isinstance(result, (list, tuple)):
        spatial, joint_out = result[0], result[1] if case.joint else None
    else:
        spatial, joint_out = result, None
    if ring:
        spatial = ttnn.to_torch(spatial, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
    else:
        spatial = ttnn.to_torch(spatial, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
    if joint_out is not None:
        joint_out = ttnn.to_torch(joint_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
        spatial = torch.cat([spatial, joint_out], dim=2)
    return spatial


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_dit_recipe_parity(parity_mesh, case, variant, record_property):
    mesh, subdevice, semaphores, full = parity_mesh
    extras = (subdevice, semaphores, full)
    result = {"case": case.name, "variant": variant}
    heads = [0, case.heads - 1]
    for regime in ("unit", "sharp"):
        host, joint = host_inputs(case, regime)
        n_rows = host[0].shape[2] + (case.joint or 0)
        rows = torch.linspace(0, n_rows - 1, min(SAMPLE_ROWS, n_rows)).long()
        try:
            invoke = build(mesh, case, host, joint, variant, extras)
            out = outputs_to_host(mesh, case, invoke())
        except RuntimeError as error:
            result["rejected"] = str(error).split("\n")[0][:300]
            print("PARITY " + json.dumps(result), flush=True)
            record_property("rejected", result["rejected"])
            pytest.skip(f"{variant} rejected: {result['rejected']}")
        ref = reference_rows(*host, joint, rows, heads)
        rel_l2, pcc = metrics(out[:, heads][:, :, rows], ref)
        result[f"{regime}_rel_l2"] = round(rel_l2, 7)
        result[f"{regime}_pcc"] = round(pcc, 8)
        if regime == "unit":
            if not variant.startswith("legacy"):
                result["chunks"] = resolved_chunks(case, variant, host, joint, mesh, full)
            median, _minimum = time_trace(mesh, invoke)
            result["ms"] = round(median, 4)
    for key, value in result.items():
        record_property(key, value)
    print("PARITY " + json.dumps(result), flush=True)


# VAE attention (dense, noncausal, replicated per chip). legacy: (fidelity, fp32_dest_acc, q, k) or
# None for the op defaults (no program/compute config). valid: logical key length of a key-padding
# mask (None: unmasked). Recipe variants with a mask run it as attn_mask; "<recipe>_sliced" drops the
# padded keys instead (bit-identical to the mask on the recipe path).
VaeCase = namedtuple("VaeCase", "name batch heads n d legacy valid")
VAE_CASES = [
    VaeCase("wan_vae_480p", 11, 1, 6240, 384, (ttnn.MathFidelity.HiFi2, True, 32, 256), None),
    VaeCase("wan_vae_720p", 11, 1, 14400, 384, (ttnn.MathFidelity.HiFi2, True, 32, 256), None),
    VaeCase("flux_vae_1024", 1, 1, 16384, 512, (ttnn.MathFidelity.HiFi2, True, 128, 128), None),
    VaeCase("sd35_vae_1024", 1, 1, 16384, 512, None, None),
    VaeCase("h3_decoder", 1, 32, 1824, 64, (ttnn.MathFidelity.HiFi2, False, 192, 192), 1797),
]
VAE_VARIANTS = os.getenv("PARITY_VAE_VARIANTS", "legacy,FAST,BALANCED,ACCURATE,FAST_sliced,BALANCED_sliced").split(",")


@pytest.mark.parametrize("variant", VAE_VARIANTS)
@pytest.mark.parametrize("case", VAE_CASES, ids=[c.name for c in VAE_CASES])
def test_vae_recipe_parity(parity_mesh, case, variant, record_property):
    mesh, _subdevice, _semaphores, full = parity_mesh
    sliced = variant.endswith("_sliced")
    if sliced and case.valid is None:
        pytest.skip("unmasked case")
    precision = variant.removesuffix("_sliced")
    result = {"case": case.name, "variant": variant}
    replicate = ttnn.ReplicateTensorToMesh(mesh)
    upload = lambda x: ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate)  # noqa: E731
    heads = [0, case.heads - 1]
    for regime in ("unit", "sharp"):
        gen = torch.Generator().manual_seed(20260924)
        scale = 2.0 if regime == "sharp" else 1.0
        shape = (case.batch, case.heads, case.n, case.d)
        q, k = (torch.randn(shape, generator=gen) * scale for _ in range(2))
        v = torch.randn(shape, generator=gen)
        q, k, v = (x.bfloat16() for x in (q, k, v))
        kwargs = {}
        keys = case.n
        if case.valid is not None and sliced:
            keys = case.valid
        elif case.valid is not None:
            mask = torch.zeros(1, 1, case.n, case.n)
            mask[..., case.valid :] = float("-inf")
            kwargs["attn_mask"] = upload(mask.bfloat16())
        tensors = [upload(q), upload(k[:, :, :keys]), upload(v[:, :, :keys])]
        if precision == "legacy":
            if case.legacy is not None:
                fidelity, fp32, qc, kc = case.legacy
                kwargs["program_config"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=full, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
                )
                kwargs["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
                    mesh.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32
                )
        else:
            kwargs["precision"] = getattr(ttnn.SDPAPrecision, precision)
        invoke = lambda: ttnn.transformer.scaled_dot_product_attention(
            *tensors, is_causal=False, **kwargs
        )  # noqa: E731
        try:
            out = ttnn.to_torch(invoke(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[: case.batch]
        except RuntimeError as error:
            result["rejected"] = str(error).split("\n")[0][:300]
            print("PARITY " + json.dumps(result), flush=True)
            record_property("rejected", result["rejected"])
            pytest.skip(f"{variant} rejected: {result['rejected']}")
        rows = torch.linspace(0, case.n - 1, min(SAMPLE_ROWS, case.n)).long()
        b = [0, case.batch - 1]
        qs = q[b][:, heads][:, :, rows].float()
        valid = case.valid or case.n  # the mask and the slice both drop the padded keys
        scores = qs @ k[b][:, heads, :valid].float().transpose(-1, -2) / math.sqrt(case.d)
        ref = torch.softmax(scores, dim=-1) @ v[b][:, heads, :valid].float()
        rel_l2, pcc = metrics(out[b][:, heads][:, :, rows], ref)
        result[f"{regime}_rel_l2"] = round(rel_l2, 7)
        result[f"{regime}_pcc"] = round(pcc, 8)
        if regime == "unit":
            median, _minimum = time_trace(mesh, invoke)
            result["ms"] = round(median, 4)
    for key, value in result.items():
        record_property(key, value)
    print("PARITY " + json.dumps(result), flush=True)
