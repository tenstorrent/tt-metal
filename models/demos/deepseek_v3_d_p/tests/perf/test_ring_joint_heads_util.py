# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level ring_joint_sdpa (KV-cache / chunked-prefill mode, streaming compute) utilization vs head layout.

Galaxy-proxy geometry on the 2x2 box: 640 query rows/chip, SP=2 ring, qk 192 / v 128, bfp8 K/V, HiFi2. The
chunk sits at kv_actual = P + 3840 (chips = Galaxy chips 6, 7). Cases vary q heads / kv heads per chip to
separate MHA-vs-GQA (K/V bytes per score) from work-unit count. Each case: warm-up + one timed call inside
`RJ_<case>_START/_END` signposts. Env RJ_CASES="nq:nkv:q:k[:l1acc],..." overrides the case list."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology

QK_D, V_D = 192, 128
DEFAULT_CASES = (
    "32:2:128:1024,16:16:128:1024,16:1:128:1024,16:2:128:1024,32:32:128:1024,16:16:128:1024:0,32:2:128:1024:0"
)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((2, 2), fabric2d_device_params(l1_small_size=1152), id="fabric2d-2x2")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("prefix", [51200], ids=["pre50k"])
@pytest.mark.timeout(0)
def test_ring_joint_heads_util(mesh_device, device_params, prefix):
    sp, tp = tuple(mesh_device.shape)
    SQ_LOCAL = int(os.environ.get("RJ_SQ_LOCAL", "640"))  # query rows/chip (Galaxy proxy: 640)
    CHUNK = SQ_LOCAL * sp
    LINKS = int(os.environ.get("RJ_LINKS", "2"))
    kv_actual = prefix + 5120 - CHUNK
    n_local = (kv_actual + CHUNK) // sp
    tt_ccl = get_tt_ccl(mesh_device)
    sp_topo, _ = per_axis_topology()
    grid = mesh_device.compute_with_storage_grid_size()
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, 1))

    def dev(shape, dtype):
        return ttnn.from_torch(
            torch.randn(*shape) * 0.5,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    for case in os.environ.get("RJ_CASES", DEFAULT_CASES).split(","):
        f = [int(x) for x in case.split(":")]
        nq, nkv, qc, kc = f[:4]
        l1acc = bool(f[4]) if len(f) > 4 else True
        q = dev((1, nq * tp, SQ_LOCAL * sp, QK_D), ttnn.bfloat16)
        k = dev((1, nkv * tp, n_local * sp, QK_D), ttnn.bfloat8_b)
        v = dev((1, nkv * tp, n_local * sp, V_D), ttnn.bfloat8_b)
        bk = ttnn.empty(
            [1, nkv, n_local * sp, QK_D], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        bv = ttnn.empty(
            [1, nkv, n_local * sp, V_D], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=l1acc,
        )
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
        )

        def run():
            out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                None,
                None,
                None,
                persistent_output_buffer_k=bk,
                persistent_output_buffer_v=bv,
                joint_strategy="rear",
                logical_n=kv_actual + CHUNK,
                program_config=pc,
                compute_kernel_config=ckc,
                dim=2,
                multi_device_global_semaphore=tt_ccl.ring_attention_ccl_semaphore_handles,
                num_links=LINKS,
                cluster_axis=0,
                mesh_device=mesh_device,
                topology=sp_topo,
                ccl_core_grid_offset=tt_ccl.ring_attention_ccl_core_grid_offset,
                use_column_major_ccl=True,
                is_causal=True,
                scale=QK_D**-0.5,
                is_balanced=False,
                kv_cache_batch_idx=0,
                kv_actual_isl=kv_actual,
            )
            return out

        tag = f"RJ_links{LINKS}_sq{SQ_LOCAL}_nq{nq}_nkv{nkv}_q{qc}k{kc}_l1acc{int(l1acc)}"
        try:
            ttnn.deallocate(run())
        except RuntimeError as e:
            logger.warning(f"{tag} failed: {str(e).splitlines()[0][:200]}")
            continue
        ttnn.synchronize_device(mesh_device)
        ttnn.tracy_message(f"`TT_SIGNPOST: {tag}_START`")
        ttnn.deallocate(run())
        ttnn.synchronize_device(mesh_device)
        ttnn.tracy_message(f"`TT_SIGNPOST: {tag}_END`")
        logger.info(f"{tag} done (kv_actual={kv_actual}, n_local={n_local})")
        for t in (q, k, v, bk, bv):
            ttnn.deallocate(t)
