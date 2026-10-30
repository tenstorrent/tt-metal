# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""
import functools
import os

import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .experts import ExpertConfig, Experts
from .experts_throughput import (
    DeepSeekPrefillConfig,
    ThroughputExpertConfig,
    ThroughputExperts,
    create_fused_moe_gpt_config,
)
from .topk import TopKRouter

# gpt-oss-20b's experts rounded offline onto the bfloat4_b grid by GPTQ: one layer_<L>.pt per layer holding
# gate_up_proj [E, H, 2I] (gate/up interleaved), gate_up_proj_bias, down_proj [E, I, H], down_proj_bias, with the
# intermediate channels reordered so each 16-wide bfloat4_b block holds similar-scale channels (an exact rewrite: gate,
# up, their bias and the down rows move together). bfloat4_b keeps these values exactly, so prefill and decode run the
# same bytes at the same speed with less rounding error than converting the checkpoint's MXFP4 experts. The repo's
# down_proj_bias.pt holds the checkpoint's expert down biases (bf16), which the GPTQ files carry unchanged: experts
# whose biases match them are the gpt-oss-20b checkpoint's. The revision is pinned, so a later upload cannot change
# what this code loads.
GPTQ_EXPERTS_REPO = "jerrywangTT/gpt-oss-20b-p150-gptq-experts"
GPTQ_EXPERTS_REVISION = "de25c21b074ca7bf7f55b1ec887579a350704fcf"
GPTQ_EXPERTS_CACHE_NAME = "experts_gptq_" + GPTQ_EXPERTS_REVISION[:12]


def _gptq_hub_file(name):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(GPTQ_EXPERTS_REPO, name, revision=GPTQ_EXPERTS_REVISION)


@functools.lru_cache(maxsize=1)
def _checkpoint_down_biases():
    """{layer: [E, H] bf16} from the Hub, or {} once when it cannot be reached (every layer then keeps its experts)."""
    try:
        return torch.load(_gptq_hub_file("down_proj_bias.pt"))
    except Exception as e:  # no network or no access to the Hub: run with the checkpoint's experts
        logger.warning(f"GPTQ experts unavailable ({e}); using the checkpoint's experts")
        return {}


@functools.lru_cache(maxsize=1)
def _warn_cache_without_gptq_experts():
    logger.warning(
        "The weight cache holds no GPTQ experts, using the checkpoint's experts; "
        "run once with GPT_OSS_FORCE_MODEL_LOAD=1 to add them"
    )


def gptq_experts(experts_state_dict, layer_idx, tensor_cache_path):
    """The experts this layer loads: (cache name, state dict).

    GPT_OSS_EXPERTS_GPTQ_DIR unset (the default): the GPTQ experts from GPTQ_EXPERTS_REPO when this layer's experts are
    the gpt-oss-20b checkpoint's (their down biases match); any other weights (another model, random test weights) are
    kept as given. With no weights on host (a warm weight cache) the GPTQ experts' cache is used if a previous run built
    it. GPT_OSS_EXPERTS_GPTQ_DIR=<directory of layer_<L>.pt>: those files. GPT_OSS_EXPERTS_GPTQ_DIR=none: the
    checkpoint's experts. The GPTQ experts get their own cache names, so an original-weight cache is never read for them.
    """
    gptq_dir = os.environ.get("GPT_OSS_EXPERTS_GPTQ_DIR")
    if gptq_dir is not None:
        gptq_dir = gptq_dir.strip()
        if gptq_dir.lower() in ("", "0", "none", "off"):
            return "experts", experts_state_dict
        assert layer_idx is not None, "GPT_OSS_EXPERTS_GPTQ_DIR needs the layer index"
        if experts_state_dict:
            rounded = torch.load(os.path.join(gptq_dir, f"layer_{int(layer_idx)}.pt"))
            experts_state_dict = {k: v.float() for k, v in rounded.items()}
        return "experts_gptq_" + os.path.basename(gptq_dir.rstrip("/")), experts_state_dict

    if layer_idx is None:
        return "experts", experts_state_dict
    if not experts_state_dict:
        built = tensor_cache_path is not None and os.path.isdir(f"{tensor_cache_path}/{GPTQ_EXPERTS_CACHE_NAME}")
        if not built:
            _warn_cache_without_gptq_experts()
        return (GPTQ_EXPERTS_CACHE_NAME if built else "experts"), experts_state_dict
    bias = experts_state_dict.get("down_proj_bias")
    reference = _checkpoint_down_biases().get(int(layer_idx))
    if bias is None or reference is None or bias.shape != reference.shape:
        return "experts", experts_state_dict
    if not torch.equal(bias.float(), reference.float()):
        return "experts", experts_state_dict
    rounded = torch.load(_gptq_hub_file(f"layer_{int(layer_idx)}.pt"))
    return GPTQ_EXPERTS_CACHE_NAME, {k: v.float() for k, v in rounded.items()}


class MLP:
    """Streamlined MoE MLP combining router and experts"""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        use_throughput_experts=True,
        tokens_per_device=32,
        layer_idx=None,
    ):
        # Split state dict
        router_state_dict = substate(state_dict, "router")
        experts_state_dict = substate(state_dict, "experts")
        experts_cache_name, experts_state_dict = gptq_experts(experts_state_dict, layer_idx, tensor_cache_path)

        # Initialize components with mesh_config
        self.router = TopKRouter(
            mesh_device,
            hf_config,
            router_state_dict,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
        )

        # Throughput experts rely on all_to_all_dispatch/combine across a mesh axis,
        # which has no meaning on a single device and would require fabric.
        if use_throughput_experts and mesh_device.get_num_devices() == 1:
            use_throughput_experts = False

        self.use_throughput_experts = use_throughput_experts
        if self.use_throughput_experts:
            # Create TT config
            throughput_expert_config = ThroughputExpertConfig(
                intermediate_size=hf_config.intermediate_size,
                num_experts=hf_config.num_local_experts,
                hidden_size=hf_config.hidden_size,
                num_experts_per_tok=hf_config.num_experts_per_tok,
                num_devices=mesh_device.get_num_devices(),
            )

            # Create fused MoE config if requested
            fused_config = None
            # moe_gpt and its packed weights require twelve DRAM-aligned cores.
            # Blackhole uses the generic gather/matmul/reduce decode path.
            if not ttnn.device.is_blackhole(mesh_device):
                fused_config = create_fused_moe_gpt_config(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    state_dict=experts_state_dict,
                    tokens_per_device=tokens_per_device,
                    num_links=ccl_manager.num_links,
                    tensor_cache_path=get_cache_file_name(tensor_cache_path, experts_cache_name),
                )

            # DeepSeek prefill config: always created when throughput experts are
            # enabled. The two were previously gated by separate flags but were always
            # set together by every caller; now bundled.
            prefill_config = None
            deepseek_permuted_weights = None
            if use_throughput_experts:
                import torch as _torch

                from .experts_throughput.prefill import _compute_weight_permutation

                prefill_config = DeepSeekPrefillConfig(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    dispatch_group_size=mesh_device.shape[0],
                    num_dispatch_groups=mesh_device.shape[1],
                    capacity_factor=2.0,
                    seq_len_per_chip=1024,
                    num_links=ccl_manager.num_links,
                )
                # Permute expert state_dict to GROUP-BASED ordering before loading
                perm = _compute_weight_permutation(
                    mesh_device.shape[0],
                    mesh_device.shape[1],
                    throughput_expert_config.num_experts // (mesh_device.shape[0] * mesh_device.shape[1]),
                )
                perm_t = _torch.tensor(perm, dtype=_torch.long)
                permuted_sd = {
                    k: v.index_select(0, perm_t) if v.shape[0] == throughput_expert_config.num_experts else v
                    for k, v in experts_state_dict.items()
                }
                from .experts_throughput.weights import load_throughput_expert_weights

                deepseek_permuted_weights = load_throughput_expert_weights(
                    mesh_device=mesh_device,
                    config=throughput_expert_config,
                    state_dict=permuted_sd,
                    weight_dtype=ttnn.bfloat4_b,
                    tensor_cache_path=get_cache_file_name(tensor_cache_path, experts_cache_name + "_ds_perm"),
                )
                prefill_config.permuted_weights = deepseek_permuted_weights

            # Create TT experts module
            self.experts = ThroughputExperts(
                mesh_device=mesh_device,
                config=throughput_expert_config,
                state_dict=experts_state_dict,
                weight_dtype=ttnn.bfloat4_b,
                dispatch_cluster_axis=0,
                decode_memory_config=ttnn.L1_MEMORY_CONFIG,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, experts_cache_name),
                mesh_config=mesh_config,
                ccl_manager=ccl_manager,
                fused_config=fused_config,
                prefill_config=prefill_config,
            )
        else:
            # Create expert config from HF config
            expert_config = ExpertConfig(
                intermediate_size=hf_config.intermediate_size,
                num_experts=hf_config.num_local_experts,
                hidden_size=hf_config.hidden_size,
                num_experts_per_tok=hf_config.num_experts_per_tok,
                swiglu_limit=hf_config.swiglu_limit,
            )

            # Use GPT-OSS specific program config
            program_config = GPTOSSProgramConfig()

            # Create experts with new modular implementation
            self.experts = Experts(
                mesh_device=mesh_device,
                config=expert_config,
                state_dict=experts_state_dict,
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                program_config=program_config,
                weight_dtype=ttnn.bfloat4_b,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, experts_cache_name),
            )

    def __call__(self, hidden_states, is_decode):
        """Forward pass: route -> experts
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
        Returns:
            Expert output tensor [batch, seq_len, hidden_size]
        """
        expert_indices, expert_weights = self.router(hidden_states, self.use_throughput_experts)
        expert_output = self.experts(
            hidden_states, topk_expert_indices=expert_indices, topk_expert_weights=expert_weights, is_decode=is_decode
        )
        return expert_output
