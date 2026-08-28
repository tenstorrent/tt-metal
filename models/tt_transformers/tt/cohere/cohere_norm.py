# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# CohereLayerNorm for Command-R (c4ai-command-r-v01).
#
# HF reference (transformers v4.39.3, models/cohere/modeling_cohere.py lines 78-94):
#   - fp32 internal compute: mean, variance, (x - mean) * rsqrt(var + eps)
#   - affine: * weight (fp32), no bias for this checkpoint
#   - cast back to input dtype; eps = config.layer_norm_eps = 1e-5
#
# This is a MEAN-CENTERING LayerNorm — NOT the RMSNorm used across tt_transformers.
#
# TP-mesh path (P5b, 2026-08-28): mirrors the proven tt_distributed_rmsnorm wiring
# (models/tt_transformers/tt/ccl.py) using the layernorm analog ops — verified present
# in the on-box ttnn build (0.19.0-era, Blackhole firmware 19.11.0):
#   ttnn.layer_norm_pre_all_gather  -> tt_all_gather(dim=3, cluster_axis=1)
#   ttnn.layer_norm_post_all_gather (per-device gamma shard)
# Gamma is sharded per-device via ShardTensor2dMesh(dims=(None, 2)) exactly like
# models/common/rmsnorm.py weight_distributed. On a multi-device mesh (cluster axis 1
# width > 1) forward() ALWAYS takes the distributed path — the same choice
# DistributedNorm makes for the stock blocks on this stack. Plain ttnn.layer_norm
# remains as the single-device fallback only; NOTE: the plain interleaved op
# over-allocates L1 circular buffers at dim 8192 on this build (observed on-box
# 2026-08-28: CBs grow to 3,363,712 B > 1,572,864 B) — do not use it on the mesh.

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import SHARD_HEIGHT
from models.tt_transformers.tt.common import Mode


class TtCohereLayerNorm(LightweightModule):
    """LayerNorm (mean-centering) for Command-R norms.

    Used for: per-layer input_layernorm (x40 — converted state-dict key
    "attention_norm": map_hf_to_meta_keys renames HF input_layernorm ->
    attention_norm; the parallel block feeds it to BOTH branches) and the final
    model norm (key "norm").

    Weight-only (no bias) for c4ai-command-r-v01: checkpoint tensors are *.weight
    only (attention_bias=false; HF CohereLayerNorm defaults bias=None).

    Weights: `weight` is the replicated full-width gamma (single-device fallback);
    `weight_distributed` is the per-device width shard used by the distributed
    path (mirrors rmsnorm.py).
    """

    def __init__(
        self,
        device,
        dim,
        eps,
        state_dict,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        weight_key="attention_norm",
        tt_ccl=None,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.eps = eps
        self.tt_ccl = tt_ccl

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        else:
            weight_name = f"{weight_key}.weight"

        # Same reshape RMSNorm uses for its norm weight (SHARD_HEIGHT == one tile).
        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )

        # Compatibility with models that don't use mesh devices (mirrors RMSNorm).
        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        # Distributed iff the mesh has >1 device on cluster axis 1 (the TP width
        # axis) — mirrors DistributedNorm always distributing on this stack.
        self.distributed = is_mesh_device and list(device.shape)[1] > 1

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=None if weight_cache_path is None else weight_cache_path / weight_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if self.distributed:
            # Per-device gamma shard: [1, 1, dim/SHARD_HEIGHT, SHARD_HEIGHT] sharded
            # on dim 2 across cluster axis 1 (mirrors rmsnorm.py weight_distributed).
            self.weight_distributed = ttnn.as_tensor(
                torch_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=(
                    None if weight_cache_path is None else weight_cache_path / (weight_name + "_distributed")
                ),
                mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape)),
            )

            # Full-width TILE gamma for the decode-mode plain layernorm: the
            # RMSNorm-style [1,1,dim/32,32] ROW_MAJOR weight selects a program
            # that over-allocates L1 at dim 8192 (TT_THROW program.cpp:1722,
            # 3.36 MB CB on core [0-0] — observed on-box 2026-08-28); the
            # TILE [1,1,1,dim] layout is probe-verified to fit.
            self.weight_fullwidth = ttnn.as_tensor(
                state_dict[weight_name].reshape(1, 1, 1, dim),
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=None if weight_cache_path is None else weight_cache_path / (weight_name + "_fullwidth"),
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
            )

        # Arch-aware compute config — a hardcoded WormholeComputeKernelConfig
        # fatals at DECODE shapes on Blackhole (std::get: wrong index for
        # variant in layer_norm_pre_all_gather, observed on-box 2026-08-28);
        # init_device_compute_kernel_config is the pattern the image's own
        # test_distributed_layernorm.py uses.
        self.compute_kernel_config_hifi2 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _distributed_layernorm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """tt_distributed_layernorm — layernorm analog of ccl.tt_distributed_rmsnorm.

        pre_all_gather computes per-row partial stats on each device's width shard;
        the stats are all-gathered along cluster axis 1; post_all_gather combines
        them into the true row mean/variance and applies the per-device gamma shard;
        the result is then all-gathered back to full width (replicated) — the same
        contract DistributedNorm gives the stock blocks (enable_all_gather=True).
        """
        # Proven pattern from the image's own tests
        # (tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm.py):
        # pre_all_gather output goes DIRECTLY into ttnn.all_gather — the reshape
        # workaround in the rmsnorm path does NOT apply to layernorm (it fatals
        # at reshape.cpp:637 in this build — observed on-box 2026-08-28).
        tt_stats = ttnn.layer_norm_pre_all_gather(
            x, compute_kernel_config=self.compute_kernel_config_hifi2, dtype=ttnn.bfloat16
        )
        tt_stats_gathered = ttnn.all_gather(
            tt_stats,
            dim=3,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_stats.deallocate(True)

        x = ttnn.layer_norm_post_all_gather(
            x,
            tt_stats_gathered,
            epsilon=self.eps,
            weight=self.weight_distributed,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        tt_stats_gathered.deallocate(True)

        # Output all-gather: mirrors DistributedNorm (distributed_norm.py forward,
        # enable_all_gather=True) — the normed tensor is REPLICATED back to full
        # width so the downstream column-parallel attention / row-parallel MLP
        # matmuls see the full 8192-wide input per device (observed on-box
        # 2026-08-28: without this gather the qkv matmul fatals width=2048 vs
        # height=8192).
        x = ttnn.all_gather(
            x,
            dim=3,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return x

    def _decode_fullwidth_layernorm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Decode-mode distributed layernorm for Command-R.

        The pre/post-all-gather layernorm op family is broken at decode shapes
        in this build (TT_THROW dataflow_buffer.cpp:2581 CB-alloc at
        [1,1,32,2048]/device, and ttnn.all_gather scrambles the layout to
        [1,1,8,32768] with maxdiff 7.35 — all observed on-box 2026-08-28).
        Mirror the stock DistributedNorm TG=False decode pattern instead:
        all_gather_async the width shards to full width (the proven CCL op
        every stock model decodes with), then a plain full-width layernorm
        (probe-verified at [1,1,32,8192] on Blackhole, DRAM and L1).
        """
        x = ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.tt_ccl.get_num_links(1),
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        return ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.weight_fullwidth,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

    def forward(
        self, x: ttnn.Tensor, mode="decode", norm_config=None, in_sharded=False, out_sharded=False
    ) -> ttnn.Tensor:
        sharded_program_config = norm_config.get("sharded_program_config") if norm_config else None
        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        output_mem_config = norm_config.get("output_mem_config") if norm_config else None

        if self.distributed and not in_sharded:
            if mode == Mode.DECODE or mode == "decode":
                x = self._decode_fullwidth_layernorm(x)
            else:
                x = self._distributed_layernorm(x)
        else:
            # Single-device / sharded-input fallback only (see module docstring:
            # plain interleaved layernorm over-allocates L1 at dim 8192 on this
            # build — the mesh path above is the production path).
            program_config = sharded_program_config if in_sharded else None
            memory_config = sharded_output_config if out_sharded else None
            x = ttnn.layer_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )

        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(x)
        if output_mem_config is not None:
            x = ttnn.to_memory_config(x, output_mem_config)
        return x


def build_cohere_final_norm(args, mesh_device, state_dict, weight_cache_path, dtype, tt_ccl):
    """Final model norm for Command-R (model.py final_norm_builder hook).

    Same distributed TtCohereLayerNorm over the converted "norm.weight" key (the
    final norm input is TP width-sharded like every residual stream tensor).
    Norm weights stay bfloat16 regardless of the model dtype (mirrors RMSNorm).
    """
    return TtCohereLayerNorm(
        device=mesh_device,
        dim=args.dim,
        eps=args.norm_eps,
        state_dict=state_dict,
        state_dict_prefix=args.get_state_dict_prefix("", None),
        weight_cache_path=None if args.dummy_weights else weight_cache_path,
        weight_dtype=ttnn.bfloat16,
        weight_key="norm",
        tt_ccl=tt_ccl,
    )
