"""Native prefill expert grid/fidelity candidate."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedMoE

class NativePrefillMoE(OptimizedMoE):
    prefill_grid = (5, 4)
    prefill_fidelity = ttnn.MathFidelity.LoFi

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj.dedicated_compute = ttnn.init_device_compute_kernel_config(kwargs['mesh_device'].arch(),
            math_fidelity=cls.prefill_fidelity, math_approx_mode=False, fp32_dest_acc_en=False)
        return obj

    def _group(self, x):
        router_shared = ttnn.linear(x, self.router_shared_packed, compute_kernel_config=self.compute)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        counts = ttnn.typecast(ttnn.where(ttnn.max(routing, dim=2, keepdim=True) > 0, 32, 0), ttnn.uint32)
        counts = ttnn.reshape(ttnn.to_layout(counts, ttnn.ROW_MAJOR_LAYOUT), (self.experts,))
        output = ttnn.zeros_like(self.output_template)
        capacity_x = self._capacity_input(x)
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            ttnn.to_layout(capacity_x, ttnn.ROW_MAJOR_LAYOUT),
            self.expert_gates,
            self.expert_ups,
            self.expert_downs,
            counts,
            self.expert_ids,
            input_m_tiles=1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.dedicated_compute,
            core_grid=ttnn.CoreCoord(*self.prefill_grid),
            output=output,
            expert_region_offsets=self.expert_offsets,
            read_x_at_offset=False,
        )
        down = ttnn.reshape(down, (1, self.experts, 32, self.hidden))
        if self.retain_intermediates:
            self.intermediates = dict(routing=routing, counts=counts, down=down)
        result = self._weighted_reduce(down, routing)
        sg, su = router_shared[..., 512:1152], router_shared[..., 1152:1792]
        scale = router_shared[..., 1792:1793]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
