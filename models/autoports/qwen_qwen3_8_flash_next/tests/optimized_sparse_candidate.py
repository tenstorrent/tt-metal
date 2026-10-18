"""Sparse active expert geometry/precision experiment with compact decode reduction."""
import math
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import _PackedMoE
from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_compact_moe_candidate import CompactMoE


class SparseMoE(CompactMoE):
    sparse_cores = 20
    sparse_down_cores = 0
    gate_block = 4
    down_block = 4
    sparse_fidelity = ttnn.MathFidelity.LoFi
    sparse_output_dtype = ttnn.bfloat8_b
    sparse_l1 = True
    sparse_packed = True
    sparse_fuse_silu = False
    exact_nnz = False
    prefill_inplace = False

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        import torch
        obj = _PackedMoE.from_state_dict.__func__(cls, state, expert_dtype=cls.dedicated_weight_dtype, **kwargs)
        obj.mesh=kwargs['mesh_device']
        if cls.down_dtype!=cls.dedicated_weight_dtype:
            obj.down=ttnn.from_torch(state['mlp.experts.down_proj'].transpose(-2,-1).unsqueeze(0).contiguous(),
                dtype=cls.down_dtype,layout=ttnn.TILE_LAYOUT,device=obj.mesh)
        if not cls.sparse_packed:
            gu=state['mlp.experts.gate_up_proj']
            obj.gate=ttnn.from_torch(gu[:,:obj.intermediate].transpose(-2,-1).unsqueeze(0).contiguous(),
                dtype=cls.dedicated_weight_dtype,layout=ttnn.TILE_LAYOUT,device=obj.mesh)
            obj.up=ttnn.from_torch(gu[:,obj.intermediate:].transpose(-2,-1).unsqueeze(0).contiguous(),
                dtype=cls.dedicated_weight_dtype,layout=ttnn.TILE_LAYOUT,device=obj.mesh)
        obj.sparse_compute=ttnn.init_device_compute_kernel_config(obj.mesh.arch(),math_fidelity=cls.sparse_fidelity,
            math_approx_mode=False,fp32_dest_acc_en=False,packer_l1_acc=True)
        for width in (obj.intermediate,obj.intermediate*2,obj.hidden):
            ncores=(cls.sparse_down_cores or cls.sparse_cores) if width==obj.hidden else cls.sparse_cores
            gx=min(ncores,10 if ncores>50 else 5); gy=math.ceil(ncores/gx)
            n=math.ceil(width/32/ncores)
            obj.programs[width]=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx,gy),
                in0_block_w=cls.down_block if width==obj.hidden else cls.gate_block,
                out_subblock_h=1,out_subblock_w=min(n,4) if n%min(n,4)==0 else 1,
                out_block_h=1,out_block_w=n,per_core_M=1,per_core_N=n,fuse_batch=False,mcast_in0=True)
        if cls.sparse_fuse_silu and not cls.sparse_packed:
            n=math.ceil(obj.intermediate/32/cls.sparse_cores)
            obj.programs['fused_gate']=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(5,math.ceil(cls.sparse_cores/5)),
                in0_block_w=cls.gate_block,out_subblock_h=1,out_subblock_w=1,out_block_h=1,out_block_w=n,
                per_core_M=1,per_core_N=n,fuse_batch=False,mcast_in0=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU))
        obj.compact_indices=ttnn.from_torch(torch.arange(10,dtype=torch.int32).reshape(1,1,1,10),
            dtype=ttnn.uint16,layout=ttnn.ROW_MAJOR_LAYOUT,device=obj.mesh)
        obj._reduce_mapping=ttnn.from_torch(torch.zeros((1,512),dtype=torch.int32),dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,device=obj.mesh)
        obj.route_ones=ttnn.from_torch(torch.cat([torch.ones(10),torch.zeros(22)]).reshape(1,1,1,32),
            dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=obj.mesh)
        obj.active_nnz=None
        return obj

    def _sparse(self,x,w,sparsity,width,*,input_sparse=False):
        return ttnn.sparse_matmul(x,w,sparsity=sparsity,nnz=self.active_nnz,is_input_a_sparse=input_sparse,
            is_input_b_sparse=True,program_config=self.programs['fused_gate'] if self.sparse_fuse_silu and not self.sparse_packed and w.buffer_address()==self.gate.buffer_address() else self.programs[width],compute_kernel_config=self.sparse_compute,
            dtype=self.sparse_output_dtype,memory_config=ttnn.L1_MEMORY_CONFIG if self.sparse_l1 else ttnn.DRAM_MEMORY_CONFIG)

    def _group(self, x):
        router_shared = ttnn.linear(x, self.router_shared_packed, compute_kernel_config=self.compute)
        logits = router_shared[..., :512]
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        sparsity = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        if self.sparse_packed:
            packed = ttnn.reshape(
                self._sparse(x, self.gate_up, sparsity, self.intermediate * 2), (1, self.experts, 32, self.intermediate * 2)
            )
            gate, up = packed[..., : self.intermediate], packed[..., self.intermediate :]
        else:
            gate=self._sparse(x,self.gate,sparsity,self.intermediate)
            up=self._sparse(x,self.up,sparsity,self.intermediate)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[] if self.sparse_fuse_silu and not self.sparse_packed else [ttnn.UnaryOpType.SILU])
        down = self._sparse(hidden, self.down, sparsity, self.hidden, input_sparse=True)
        del hidden, gate, up
        if self.sparse_packed: del packed
        routing = ttnn.permute(routing, (0, 3, 2, 1))
        weighted=ttnn.multiply(down,routing,output_tensor=down) if self.prefill_inplace else ttnn.multiply(down,routing)
        result = ttnn.sum(weighted, dim=1, keepdim=True)
        sg, su = router_shared[..., 512:1152], router_shared[..., 1152:1792]
        scale = router_shared[..., 1792:1793]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))

    def _group_unified(self,x):
        rs=ttnn.linear(x[:,:,:1,:],self.router_shared_packed,compute_kernel_config=self.compute)
        logits=rs[...,:512]
        values,indices=ttnn.topk(logits,k=32,dim=-1,sorted=True)
        selected=ttnn.softmax(ttnn.add(values,self.top_mask),dim=-1,numeric_stable=True,compute_kernel_config=self.compute)
        routing=ttnn.scatter(ttnn.zeros_like(logits),dim=3,index=indices,src=selected)
        # Exactly ten distinct indices from topk, with exact BF16 ones; score
        # underflow cannot change the work count. Scores remain unchanged.
        self.active_nnz=10 if self.exact_nnz else None
        mask=ttnn.scatter(ttnn.zeros_like(logits),dim=3,index=indices,src=self.route_ones) if self.exact_nnz else routing
        sparsity=ttnn.to_layout(mask,ttnn.ROW_MAJOR_LAYOUT)
        if self.sparse_packed:
            gu=self._sparse(x,self.gate_up,sparsity,self.intermediate*2)
            gu=ttnn.reshape(gu,(1,512,32,self.intermediate*2))
            gate,up=gu[...,:self.intermediate],gu[...,self.intermediate:]
        else:
            gate=self._sparse(x,self.gate,sparsity,self.intermediate)
            up=self._sparse(x,self.up,sparsity,self.intermediate)
        hidden=ttnn.multiply(gate,up,input_tensor_a_activations=[] if self.sparse_fuse_silu and not self.sparse_packed else [ttnn.UnaryOpType.SILU],
            memory_config=ttnn.L1_MEMORY_CONFIG if self.sparse_l1 else ttnn.DRAM_MEMORY_CONFIG)
        down=self._sparse(hidden,self.down,sparsity,self.hidden,input_sparse=True)
        result=self._compact(down,indices,selected)
        sg,su=rs[...,512:1152],rs[...,1152:1792]
        shared=ttnn.linear(ttnn.multiply(sg,su,input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,compute_kernel_config=self.compute)
        return ttnn.add(result,ttnn.multiply(shared,rs[...,1792:1793],input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
