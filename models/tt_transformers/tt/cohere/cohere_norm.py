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
# Weight loading mirrors models/common/rmsnorm.py (SHARD_HEIGHT reshape, replicated
# row-major weight, cache_file_name) so the on-device weight format matches what the
# ttnn normalization ops expect on this stack.

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import SHARD_HEIGHT


class TtCohereLayerNorm(LightweightModule):
    """LayerNorm (mean-centering) for Command-R norms.

    Used for: per-layer input_layernorm (x40 — converted state-dict key
    "attention_norm": map_hf_to_meta_keys renames HF input_layernorm ->
    attention_norm; the parallel block feeds it to BOTH branches) and the final
    model norm (key "norm").

    Weight-only (no bias) for c4ai-command-r-v01: checkpoint tensors are *.weight
    only (attention_bias=false; HF CohereLayerNorm defaults bias=None).

    TODO(PCC): validate ttnn.layer_norm vs HF CohereLayerNorm fp32 compute at
    PCC >= 0.99 on captured CPU-reference activations (tt-rd scripts/command-r/)
    before Stage-1 bring-up. compute_kernel_config mirrors RMSNorm's HiFi2 +
    fp32_dest_acc_en settings; re-tune if PCC < 0.99.
    TODO(Stage-2): sharded program/memory configs + a DistributedNorm equivalent
    for TP meshes (the RMSNorm path uses DistributedNorm(RMSNorm(...)); LayerNorm
    needs its own all-gather wiring — models/tt_transformers/tt/distributed_norm.py).
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

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=None if weight_cache_path is None else weight_cache_path / weight_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self, x: ttnn.Tensor, mode="decode", norm_config=None, in_sharded=False, out_sharded=False
    ) -> ttnn.Tensor:
        sharded_program_config = norm_config.get("sharded_program_config") if norm_config else None
        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        output_mem_config = norm_config.get("output_mem_config") if norm_config else None

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

    Plain (non-distributed) TtCohereLayerNorm over the converted "norm.weight" key.
    Norm weights stay bfloat16 regardless of the model dtype (mirrors RMSNorm).
    TODO(Stage-2): DistributedNorm strategy for TP meshes (see class TODO above).
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
