# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load-time expert placement for the M3 EP MoE (M3_KA_EXPERT_PLACEMENT, see utils/expert_placement.py).

Two pieces, both pure byte moves (no rounding, no re-tilize):
  * ``permute_router_columns``: the router gate weight [hidden, E] and the e_score_correction_bias [1, E] get
    their expert columns permuted (bf16 -> torch -> bf16), so the router emits label n for expert perm[n].
  * ``make_placed_routed_expert_cls``: a TtRoutedExpert whose cache-only loader reads the 8 cached local-slot
    tensors of each projection (bf4 multi-device host tensors, one shard per chip), and rebuilds every slot from
    the shards of the experts the placement puts there before moving it to the device exactly as
    TtRoutedExpert does. It never writes the cache.
"""

from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert, routed_expert_weight_memory_config
from models.demos.minimax_m3.utils import expert_placement as ep


def permute_router_columns(t, perm, mesh_device):
    """Replicated device tensor [..., E] -> the same with columns reordered as out[..., n] = t[..., perm[n]]."""
    host = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    assert host.shape[-1] == len(perm), (tuple(host.shape), len(perm))
    out = ttnn.from_torch(
        host[..., perm].contiguous(),
        device=mesh_device,
        layout=t.layout,
        dtype=t.dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.deallocate(t)
    return out


def make_placed_routed_expert_cls(perm):
    """TtRoutedExpert subclass serving expert perm[n] at label n's (chip, local slot)."""
    perm = [int(x) for x in perm]
    ep.validate(perm)

    class PlacedRoutedExpert(TtRoutedExpert):
        expert_perm = perm

        @staticmethod
        def _convert_and_cache_expert_weights(
            torch_weights,
            experts_per_chip,
            mesh_device,
            weights_dtype,
            cache_path,
            cache_name_prefix,
            device=None,
            *,
            emb_dim=None,
            hidden_dim=None,
            weights_dram_nd_sharded=False,
        ):
            if torch_weights is not None or device is None or cache_path is None or cache_name_prefix is None:
                raise NotImplementedError("M3_KA_EXPERT_PLACEMENT supports the cache-only expert load path only")
            rows, cols = tuple(mesh_device.shape)
            assert (rows, cols, experts_per_chip) == (
                ep.DISPATCH_GROUP_SIZE,
                ep.NUM_DISPATCH_GROUPS,
                ep.EXPERTS_PER_CHIP,
            ), f"placement assumes the 4x4 / 8-experts-per-chip layout, got {rows}x{cols}/{experts_per_chip}"

            def _to_device(host_tt):  # identical to TtRoutedExpert's own placement
                host_tt = ttnn.squeeze(ttnn.squeeze(host_tt, dim=0), dim=0)
                return ttnn.to_device(
                    host_tt,
                    device,
                    memory_config=routed_expert_weight_memory_config(
                        device, host_tt.shape[-1], dram_nd_sharded=weights_dram_nd_sharded
                    ),
                )

            result = []
            for proj in ("gate", "up", "down"):
                # old[l][r * cols + g] = the shard of local slot l on mesh device (row r, col g), i.e. expert
                # (g * rows + r) * epc + l (ExpertMapping column-major; mesh mapper dims=(0, 1), row-major shards)
                old = []
                for l in range(experts_per_chip):
                    path = Path(cache_path) / (
                        f"{cache_name_prefix}.local_{l}_{proj}_dtype_{weights_dtype.name}_layout_TILE.tensorbin"
                    )
                    if not path.is_file():
                        raise FileNotFoundError(f"expert placement needs the cached expert weight {path}")
                    host = ttnn._ttnn.tensor.load_tensor_flatbuffer(str(path), device=None)
                    shards = ttnn.get_device_tensors(host)
                    assert len(shards) == rows * cols, (path, len(shards))
                    old.append(shards)
                placed = []
                for l in range(experts_per_chip):
                    shards = []
                    for r in range(rows):
                        for g in range(cols):
                            n = (g * rows + r) * experts_per_chip + l
                            ge, re_, le = ep.label_position(perm[n])
                            shards.append(old[le][re_ * cols + ge])
                    placed.append(_to_device(ttnn.from_host_shards(shards, ttnn.MeshShape(rows, cols))))
                del old
                result.append(placed)
            return tuple(result)

    return PlacedRoutedExpert
