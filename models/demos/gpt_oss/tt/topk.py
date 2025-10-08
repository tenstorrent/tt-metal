import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name


def topk_router(g, experts_per_token):
    expert_weights, expert_indices = ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)
    compute_config = ttnn.init_device_compute_kernel_config(
        g.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    expert_weights = ttnn.softmax(expert_weights, dim=1, numeric_stable=True, compute_kernel_config=compute_config)
    router_scores = ttnn.scatter(ttnn.zeros_like(g), dim=1, index=expert_indices, src=expert_weights)
    return router_scores, expert_weights, expert_indices


class TopKRouter:
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.weight = ttnn.as_tensor(
            state_dict["weight"].transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            state_dict["bias"].unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # NOTE: bad outputs when I provide any reasonable compute config
        # self.compute_config = ttnn.init_device_compute_kernel_config(
        #     mesh_device.arch(),
        #     math_fidelity=ttnn.MathFidelity.HiFi2,
        #     math_approx_mode=False,
        #     fp32_dest_acc_en=False,
        #     packer_l1_acc=False,
        # )
        self.compute_config = None

    def __call__(self, hidden_states):
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
        router_logits = ttnn.linear(
            hidden_states, self.weight, bias=self.bias, compute_kernel_config=self.compute_config
        )
        # hidden_states.deallocate(True)
        router_scores, _expert_weights, router_indices = topk_router(router_logits, self.top_k)
        # router_logits.deallocate(True)
        return router_scores, router_indices, router_logits
