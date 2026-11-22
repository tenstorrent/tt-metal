# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.GAP (PACKED_CWRITE_FAST=on): ttnn.bringup.update_padded_kv_cache's input row window with tp_axis (allowed by
the fork in PB.GAP) writes exactly what the same call on the sliced window writes.

Mesh only (0 model loads; the resident runner's open mesh). The packed served writes' geometry: the input [1, 1, n cr,
width] a chip (mesh row r's server-order rows of n segments, replicated over the columns), each segment j written from
rows [j cr, (j + 1) cr) into its own slot at its own start (metadata path: slot / start / valid as 1-element uint32
device tensors). For every (dtype, tp_axis, start) case: cache A takes the windowed call, cache B the call on
ttnn.slice of the same rows; both caches start from the same random content and are compared bit for bit on every chip.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd \\
        "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_cwrite_window_ops.py"
"""

from __future__ import annotations

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True

N_SEG = 4
CR = 512  # compressed rows a mesh row a segment (2048 tokens / 2 ... the 8x4's 4 x 2k at ratio 2)
SLOTS, LAYERS = 2, 1
# (cache width, dtype, tp_axis): the ratio-2 compressed KV (bf16 512, TP-dedup) and the fp8 index-key rows (uint8 192)
KINDS = (("ckv", 512, "bf16", 1), ("idx8", 192, "u8", None), ("idx8_tp", 192, "u8", 1))
STARTS = (69632, 10608, 245760)  # 139264 / 2; a start inside a stripe; a far start (the cache holds 2^18 rows)


@mesh_parametrize(timeout_s=1800)
def test_window_tp_equals_slice(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

    mesh = mesh_device
    sp, tp = int(mesh.shape[0]), int(mesh.shape[1])
    g = torch.Generator().manual_seed(7)
    bad, cases = [], 0

    def u32(v):
        return ttnn.from_torch(
            torch.tensor([[int(v)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    for name, width, dt, tp_axis in KINDS:
        # rows a chip of a 2^18-row slot: striped over SP x TP (TP-dedup) or over SP only (tp_axis None)
        depth = -(-(262144 + 16384) // (sp * (tp if tp_axis is not None else 1)) // 32) * 32
        ttype = ttnn.bfloat16 if dt == "bf16" else ttnn.uint8
        if dt == "bf16":
            prior = torch.randn(sp * tp, SLOTS * LAYERS, 1, depth, width, generator=g).to(torch.bfloat16)
            inp = torch.randn(sp, 1, 1, N_SEG * CR, width, generator=g).to(torch.bfloat16)
        else:
            prior = torch.randint(0, 256, (sp * tp, SLOTS * LAYERS, 1, depth, width), generator=g).to(torch.uint8)
            inp = torch.randint(0, 256, (sp, 1, 1, N_SEG * CR, width), generator=g).to(torch.uint8)

        host_prior = prior.reshape(sp, tp, SLOTS * LAYERS, 1, depth, width)
        x = ttnn.from_torch(
            inp.reshape(sp, 1, N_SEG * CR, width),
            dtype=ttype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, None)),
        )

        def fresh():
            return ttnn.from_torch(
                host_prior.reshape(sp, tp * SLOTS * LAYERS, depth, width),
                dtype=ttype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, 1)),
            )

        for start in STARTS:
            ca, cb = fresh(), fresh()
            ca = ttnn.reshape(ca, [SLOTS * LAYERS, 1, depth, width])
            cb = ttnn.reshape(cb, [SLOTS * LAYERS, 1, depth, width])
            for j in range(N_SEG):
                slot, layer = j % SLOTS, j % LAYERS
                kv0 = start + j * 4096
                valid = kv0 + CR - 16 * (j % 2)
                s_t, k_t, v_t = u32(slot), u32(kv0), u32(-(-valid // 16) * 16)
                args = dict(layer_idx=layer, num_layers=LAYERS, cluster_axis=0, tp_axis=tp_axis, max_grid=max_grid())
                ttnn.bringup.update_padded_kv_cache(
                    ca, x, s_t, k_t, valid_global=v_t, input_row_offset=j * CR, input_rows=CR, **args
                )
                sl = ttnn.slice(
                    x, [0, 0, j * CR, 0], [1, 1, (j + 1) * CR, width], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.bringup.update_padded_kv_cache(cb, sl, s_t, k_t, valid_global=v_t, **args)
                ttnn.deallocate(sl)
                for t in (s_t, k_t, v_t):
                    ttnn.deallocate(t)
            da = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(ca)]
            db = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(cb)]
            same = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(da, db))
            moved = any(
                not torch.equal(a.view(torch.uint8), p.reshape(a.shape).view(torch.uint8))
                for a, p in zip(da, host_prior.reshape(sp * tp, SLOTS * LAYERS, 1, depth, width))
            )
            cases += 1
            print(f"PB.GAP window x tp: {name} start {start}: equal {same}, wrote {moved}")
            if not (same and moved):
                bad.append((name, start, same, moved))
            ttnn.deallocate(ca)
            ttnn.deallocate(cb)
        ttnn.deallocate(x)
    metrics.record("cwrite_window_tp_mismatches", len(bad))
    metrics.record("cwrite_window_tp_cases", cases)
    assert not bad, bad
