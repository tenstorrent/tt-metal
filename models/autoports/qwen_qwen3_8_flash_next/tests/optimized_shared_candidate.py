"""Shared expert/router packed-versus-separate and DRAM geometry trials."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedMoE
class SharedMoE(OptimizedMoE):
    shared_dtype=ttnn.bfloat16
    shared_fidelity=ttnn.MathFidelity.HiFi4
    shared_split=False
    shared_readers=0
    shared_cores=(5,2)
    shared_blocks=(16,10)
    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        obj=super().from_state_dict(*args,**kwargs)
        obj.device=obj.mesh
        obj.projection_compute=ttnn.init_device_compute_kernel_config(obj.device.arch(),math_fidelity=cls.shared_fidelity,
            math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
        for name in ('router_shared_packed','shared_down'):
            setattr(obj,name,ttnn.typecast(getattr(obj,name),cls.shared_dtype))
        names=['router_shared_packed','shared_down'];cores=cls.shared_cores;blocks=cls.shared_blocks
        if cls.shared_split:
            obj.shared_parts=[]
            for i,(a,b) in enumerate(((0,512),(512,1152),(1152,1792),(1792,1824))):
                w=obj.router_shared_packed[:,a:b];setattr(obj,f'shared_part_{i}',w);obj.shared_parts.append(w)
            names=[f'shared_part_{i}' for i in range(4)]+['shared_down']
            cores=[cores[0]]*4+[cores[1]];blocks=[blocks[0]]*4+[blocks[1]]
        if cls.shared_readers:
            from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import configure
            configure(obj,names,cores=cores,blocks=blocks,readers=cls.shared_readers)
        return obj
    def _shared_linear(self,x,w,**kwargs):
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import linear
        def mm(weight):
            if self.shared_readers:return linear(self,x,weight)
            return ttnn.linear(x,weight,compute_kernel_config=self.projection_compute)
        if self.shared_split and w.buffer_address()==self.router_shared_packed.buffer_address():
            return ttnn.concat([mm(v) for v in self.shared_parts],dim=-1)
        return mm(w)
    def _group(self, x):
        router_shared = self._shared_linear(x, self.router_shared_packed, compute_kernel_config=self.compute)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        counts = ttnn.typecast(ttnn.where(ttnn.max(routing, dim=2, keepdim=True) > 0, 32, 0), ttnn.uint32)
        counts = ttnn.reshape(ttnn.to_layout(counts, ttnn.ROW_MAJOR_LAYOUT), (self.experts,))
        output = ttnn.zeros_like(self.output_template)
        capacity_x = self._capacity_input(x)
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(ttnn.to_layout(capacity_x, ttnn.ROW_MAJOR_LAYOUT), self.expert_gates, self.expert_ups, self.expert_downs, counts, self.expert_ids, input_m_tiles=1, dtype=ttnn.bfloat16, compute_kernel_config=self.dedicated_compute, core_grid=ttnn.CoreCoord(5, 4), output=output, expert_region_offsets=self.expert_offsets, read_x_at_offset=False)
        down = ttnn.reshape(down, (1, self.experts, 32, self.hidden))
        if self.retain_intermediates:
            self.intermediates = dict(routing=routing, counts=counts, down=down)
        result = self._weighted_reduce(down, routing)
        sg, su = (router_shared[..., 512:1152], router_shared[..., 1152:1792])
        scale = router_shared[..., 1792:1793]
        shared = self._shared_linear(ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), self.shared_down, compute_kernel_config=self.compute)
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))

    def _group_unified(self, x):
        router_input = x[:, :, :1, :] if self.logical_router else x
        router_shared = self._shared_linear(router_input, self.router_shared_packed, compute_kernel_config=self.compute)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        active = ttnn.max(routing, dim=2, keepdim=True) > 0
        counts = ttnn.typecast(ttnn.where(active, 32, 0), ttnn.uint32)
        counts = ttnn.reshape(ttnn.to_layout(counts, ttnn.ROW_MAJOR_LAYOUT), (self.experts,))
        dispatched = self._dispatch_input(x, counts)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(dispatched, self.expert_offsets, counts, self.expert_ids, self.expert_gates, self.expert_ups, self.expert_downs, 32)
        result = self._compact(down, indices, selected)
        sg, su = (router_shared[..., 512:1152], router_shared[..., 1152:1792])
        scale = router_shared[..., 1792:1793]
        shared = self._shared_linear(ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), self.shared_down, compute_kernel_config=self.compute)
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
