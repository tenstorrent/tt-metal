# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B RMSNorm.

HF anchor: ``transformers.models.llama.modeling_llama.LlamaRMSNorm``.
Template: ``models/demos/gpt_oss_d_p/tt/rms_norm.py:17`` (class), ``:49`` (``forward``),
``:27`` (the ``(1, 1, -1, TILE_SIZE)`` weight reshape), ``:34`` (``as_tensor`` ROW_MAJOR +
``cache_file_name``), ``:94`` (the plain single-pass ``ttnn.rms_norm``).

Llama's norm is **plain**: ``out = rms_norm(x) * weight``. There is no Gemma-style ``(1 + weight)``
fold — the template's ``use_gemma_norm`` branch (``rms_norm.py:22``, ``:25-26``) is deleted rather
than defaulted off, per convention 12 (assert/omit, do not branch, on features Llama lacks);
``00_MODEL_CARD.md`` §2 records the config has no such key. ``eps`` is
``hf_config.rms_norm_eps`` = 1e-05.

Two branches exist, one dormant:

* ``is_distributed=False`` (default, this iteration) — one ``ttnn.rms_norm`` over the full-width
  replicated residual. This is residual **scheme A** (``DEC-018``).
* ``is_distributed=True`` — the 3-op distributed norm (``rms_norm_pre_all_gather`` -> stats
  all-gather -> ``rms_norm_post_all_gather``) for a TP-sharded residual, i.e. scheme B. Unlike the
  template, which pins the flag to a literal with the condition commented out
  (``gpt_oss_d_p/tt/rms_norm.py:33``), it is a **constructor argument** here, so enabling scheme B
  is a caller decision and not a source edit — ``DEC-024``. It stays ``False`` for P5-P7; P8 owns
  the first exercise of it, and the stats all-gather is the one sanctioned raw-``ttnn`` collective
  in this package (``bringup_log/04_CCL_PLAN.md`` §7.1 row 7).
"""

from torch import nn

import ttnn
from models.demos.llama31_8b_d_p.utils.general_utils import get_cache_file_name, get_default_num_links


def norm_compute_kernel_config(mesh_device):
    """Compute-kernel config for the norm ops. ``fp32_dest_acc_en=True`` is the load-bearing field.

    Measured on a Blackhole card against an fp32-weight torch reference (real layer-0 norm weights,
    hidden 4096), ``ttnn.rms_norm`` PCC:

    ==============================================  ==================  ==================
    compute_kernel_config                           rand[0,1) 32/512    randn 32/512
    ==============================================  ==================  ==================
    none (the template's implicit default)          0.9999440/0.9999531 0.9999652/0.9999648
    HiFi2, fp32_dest_acc_en=False                   0.9999369/0.9999407 0.9999607/0.9999590
    HiFi4, fp32_dest_acc_en=True (this)             0.9999969/0.9999968 0.9999971/0.9999971
    torch bf16 input-rounding floor                 0.9999986/0.9999987 0.9999986
    ==============================================  ==================  ==================

    So ``MathFidelity`` alone is a no-op (HiFi2 is marginally worse than the default), while
    ``fp32_dest_acc_en=True`` removes ~25x of the error and lands on the numerical floor. Passing no
    config -- which every template norm does -- is a silent precision regression. See ``DEC-031`` and
    ``BRINGUP_RECIPE.md`` Appendix E.3.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class RMSNorm(nn.Module):
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        *,
        mesh_config,
        tensor_cache_path=None,
        is_distributed: bool = False,
    ):
        """
        Args:
            mesh_device: the open mesh.
            hf_config: a ``LlamaHFConfig`` (``tt/model_config.py``). Only ``rms_norm_eps`` is read.
            state_dict: the already-stripped sub-dict for this norm, i.e. ``{"weight": ...}``.
                May be empty in cache-only mode (convention 5): ``as_tensor`` then reads the
                tilized file named by ``tensor_cache_path``.
            mesh_config: the package ``MeshConfig``. Required — the weight's mesh mapper depends on
                it whenever ``is_distributed``.
            tensor_cache_path: directory for the ttnn weight cache, or ``None`` to skip caching.
            is_distributed: run the 3-op distributed norm (residual scheme B). ``DEC-024``.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.is_distributed = is_distributed
        self.eps = hf_config.rms_norm_eps
        # Built once per module, not per forward: it is a pure value object, but rebuilding it per
        # call would put an avoidable host-side allocation in the per-layer path.
        self.compute_kernel_config = norm_compute_kernel_config(mesh_device)

        if state_dict:
            # ttnn's norm wants the gain laid out one tile-row wide: [1, 1, hidden/32, 32].
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        # Norm gains stay bfloat16 regardless of the model's weight dtype (convention 11): they are
        # tiny, and a bf8_b gain is a direct multiplicative error on every activation.
        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2)) if is_distributed else None,
        )

    def forward(self, x):
        if self.is_distributed:
            activation_grid_bounding_box_size = x.memory_config().shard_spec.grid.bounding_box().grid_size()
            shard_height, shard_width = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=activation_grid_bounding_box_size,
                subblock_w=1,
                block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
                block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
                inplace=False,
            )
            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * self.mesh_config.tp],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            # Distributed rmsnorm part 1: per-shard sum of squares.
            tt_stats = ttnn.rms_norm_pre_all_gather(
                x,
                program_config=program_config,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config,
            )

            # All-gather the [1,1,32,32] stats across the TP axis. The one sanctioned raw-ttnn
            # collective in this package (04_CCL_PLAN.md section 7.1): it is the non-async op, and
            # MeshConfig.allgather wraps the async one.
            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                num_links=get_default_num_links(self.mesh_device),
                cluster_axis=self.mesh_config.tp_axis,
                mesh_device=self.mesh_device,
                memory_config=tt_gathered_stats_memory_config,
                topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(tt_stats)

            # Distributed rmsnorm part 2: normalise with the global statistics and apply the gain.
            tt_output = ttnn.rms_norm_post_all_gather(
                x,
                tt_gathered_stats,
                program_config=program_config,
                epsilon=self.eps,
                weight=self.tt_weight,
                dtype=ttnn.bfloat16,
                stats=tt_gathered_stats,
                compute_kernel_config=self.compute_kernel_config,
            )
            ttnn.deallocate(tt_gathered_stats)
            return tt_output

        return ttnn.rms_norm(
            x, weight=self.tt_weight, epsilon=self.eps, compute_kernel_config=self.compute_kernel_config
        )
