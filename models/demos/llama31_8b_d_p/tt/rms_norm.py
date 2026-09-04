# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plain RMSNorm for Llama-3.1-8B: `out = rms_norm(x) * weight`.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaRMSNorm`.
**Template:** `models/demos/gpt_oss_d_p/tt/rms_norm.py:17`, adapted rather than imported
(`DEC-014`), with three changes:

1. **The Gemma `+1` fold is deleted.** Llama is a plain RMSNorm — there is no `use_gemma_norm` key
   in its config (`bringup_log/00_MODEL_CARD.md` §2, §3), so the template's fold at
   `models/demos/gpt_oss_d_p/tt/rms_norm.py:25-26` is dead weight and exactly the kind of
   copied-in feature the card's "does NOT have" section exists to stop.
2. **`eps` is a dict lookup**, not `hf_config.rms_norm_eps` as an attribute
   (`models/demos/gpt_oss_d_p/tt/rms_norm.py:46`). This package normalises `hf_config` to a dict
   once and holds it (recipe P1 trap 2).
3. **An explicit `compute_kernel_config`.** The template calls `ttnn.rms_norm` with none
   (`models/demos/gpt_oss_d_p/tt/rms_norm.py:94`), and this is the one op in the model whose
   default does *not* already enable fp32 accumulation: recipe §2.4 measures the op at 0.9999652
   with no config versus 0.9999971 with `fp32_dest_acc_en=True`, against a 0.9999986 floor. The
   `G-RMS` gate A/Bs it in-suite so the claim is a number on this box, not a quotation.

The `is_distributed` branch is **kept and defaults off** (`BRINGUP_RECIPE.md:1271-1274`). It is
only reachable under residual scheme B, and `DEC-025` takes scheme A for this iteration, so the
branch is dormant: a dormant branch is a claim about code that has never run, and it must not be
counted as "the distributed norm works". Its raw `ttnn.all_gather` is the single sanctioned
exception to the "modules only call `MeshConfig` wrappers" rule (`DEC-028`).
"""

from torch import nn

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name, get_default_num_links

from .config import MeshConfig, default_compute_kernel_config


class RMSNorm(nn.Module):
    """`out = rms_norm(x, eps) * weight`. Weight replicated; input `[1, 1, S_loc, hidden]` bf16 TILE."""

    def __init__(
        self,
        mesh_device,
        hf,
        state_dict,
        *,
        tensor_cache_path=None,
        mesh_config=None,
        is_distributed=False,
        fp32_dest_acc_en=True,
    ):
        """
        Args:
            mesh_device: the open mesh.
            hf: the normalised config **dict** (recipe P1 trap 2).
            state_dict: already stripped to this norm's own keys, i.e. `{"weight": ...}` — the
                caller splits with `substate` (`models/demos/gpt_oss_d_p/utils/substate.py:15`).
                May be empty in cache-only mode, which requires `tensor_cache_path`.
            tensor_cache_path: where `ttnn.as_tensor` persists / reloads the tilized weight.
            mesh_config: the model's `MeshConfig`; defaults to TP over the whole column axis.
            is_distributed: run the 3-op distributed norm on an emb/TP-sharded input. Dormant in
                this iteration (`DEC-025`, `DEC-028`).
            fp32_dest_acc_en: exposed **only** so `G-RMS` can A/B recipe §2.4's flag in-suite. The
                default is the correct value; passing `False` is a measurement, never a
                configuration.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.is_distributed = is_distributed
        self.eps = hf["rms_norm_eps"]
        self.compute_kernel_config = default_compute_kernel_config(mesh_device, fp32_dest_acc_en=fp32_dest_acc_en)

        if state_dict:
            # Reshaped to (1, 1, hidden/TILE_SIZE, TILE_SIZE) and stored ROW_MAJOR, which is the
            # layout `ttnn.rms_norm`'s `weight=` wants (`models/demos/gpt_oss_d_p/tt/rms_norm.py:27`).
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        elif not tensor_cache_path:
            # Fail loud rather than running weight-free: a norm whose gain silently became `None`
            # is the "one layer runs on garbage" failure mode in Appendix B.
            raise ValueError("RMSNorm needs either a state_dict with 'weight' or a tensor_cache_path to load from")
        else:
            torch_weight = None

        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,  # `DEC-022`: norm weights stay bf16 ROW_MAJOR
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2)) if self.is_distributed else None
            ),
        )

    def forward(self, x):
        """`[1, 1, S_loc, hidden]` -> `[1, 1, S_loc, hidden]`, bf16 TILE in and out."""
        if self.is_distributed:
            return self._forward_distributed(x)
        return ttnn.rms_norm(
            x,
            weight=self.tt_weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _forward_distributed(self, x):
        """The 3-op distributed norm for an emb/TP-sharded input. **Dormant** — see `DEC-025`.

        `rms_norm_pre_all_gather` -> all-gather the `[1, 1, 32, 32*tp]` stats tensor ->
        `rms_norm_post_all_gather`. The all-gather is a raw `ttnn.all_gather` rather than
        `MeshConfig.allgather`, which is the one exception `BRINGUP_RECIPE.md:1156-1158` grants and
        `DEC-028` logs: this tensor needs a norm-specific width-sharded L1 memory config, and
        `ttnn.all_gather` is non-experimental, so it takes no ping-pong semaphores and carries none
        of the lifetime hazard the wrapper rule guards against.
        """
        activation_grid_bounding_box_size = x.memory_config().shard_spec.grid.bounding_box().grid_size()
        shard_height, shard_width = x.memory_config().shard_spec.shape
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=activation_grid_bounding_box_size,
            subblock_w=1,
            block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
            block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
            inplace=False,
        )
        gathered_stats_memory_config = ttnn.create_sharded_memory_config(
            shape=[1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * self.mesh_device.shape[1]],
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        tt_stats = ttnn.rms_norm_pre_all_gather(
            x,
            program_config=program_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        tt_gathered_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=get_default_num_links(self.mesh_device),
            cluster_axis=self.mesh_config.tp_axis,  # collectives go on the TP axis only
            mesh_device=self.mesh_device,
            memory_config=gathered_stats_memory_config,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(tt_stats)

        # `stats` is passed ONCE, positionally. The template passes the same tensor both
        # positionally and as `stats=` (`models/demos/gpt_oss_d_p/tt/rms_norm.py:82-90`), which
        # raises `TypeError: incompatible function arguments` — measured, see `DEC-031`. The branch
        # is dormant there, so nothing has ever executed it.
        tt_output = ttnn.rms_norm_post_all_gather(
            x,
            tt_gathered_stats,
            program_config=program_config,
            epsilon=self.eps,
            weight=self.tt_weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(tt_gathered_stats)
        return tt_output
