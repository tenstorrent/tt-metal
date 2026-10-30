# SPDX-License-Identifier: Apache-2.0
"""Four-chip TP decoder. Baseline math: optimized_decoder. See stage04 evidence."""
import copy
import math
import ttnn
from .optimized_decoder import OptimizedDecoder, LinearState, PagedState, default_policy, reshape
from .fp32_recurrence import _KERNEL, _UPSTREAM
from models.common.modules.tt_ccl import TT_CCL


class MultichipDecoder(OptimizedDecoder):
    supports_prepared_prefill = True
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None, ccl_workspace=None):
        import torch
        config = getattr(hf_config, "text_config", hf_config)
        assert mesh_device.get_num_devices() == 4 and list(mesh_device.shape) == [1, 4]
        assert (config.hidden_size, config.intermediate_size) == (5120, 17408)
        assert (config.num_attention_heads, config.num_key_value_heads) == (24, 4)
        assert (config.linear_num_key_heads, config.linear_num_value_heads) == (16, 48)
        assert config.layer_types[layer_idx] in ('linear_attention','full_attention')
        self = cls()
        self.config, self.mesh_device = config, mesh_device
        self.layer_idx, self.layer_type = layer_idx, config.layer_types[layer_idx]
        self.policy = default_policy(self.layer_type)
        # Stage05 policy; final evidence and inter-layer contract live in doc/optimized_multichip_decoder.
        self.policy.update(
            {'direct_all_reduce': True,
             'carry_residual': True,
             'l1_mlp': True,
             'ccl_dtype': 'bfloat8_b',
             'pooled_all_reduce': True,
             'ar_fused_move': True,
             'mix': {'cores': 8, 'block': 20, 'readers': 1, 'prefill_block': 4, 'weight_pad_readers': 2},
             'out': {'cores': 8, 'block': 6, 'readers': 2, 'prefill_block': 4},
             'down': {'cores': 8, 'block': 17, 'readers': 1, 'prefill_block': 4},
             'prefill_chunk': 4096,
             'prefill_ccl_workers': 8,
             'prefill_native_rs': True,
             # Full-stack long generation needs finer out-projection K grouping;
             # retain native fusion and the measured down-projection geometry.
             'prefill_rs_geometry': {'grid': [11,8], 'out_k': 4, 'down_k': 17, 'block_h': 4, 'subh': 2, 'subw': 3},
             'decode_rs_geometry': {'grid': [10,1], 'out_k': 6, 'down_k': 17, 'subh': 1, 'subw': 4},
             'packed_decode_rope_min_batch': 4,
             'packed_prefill_rope_min_batch': 8,
             'prefill_ccl_dtype': 'bfloat8_b',
             'prefill_block': 4,
             'prefill_subh': 1,
             'gateup': {'cores': 80, 'block': 2, 'readers': 2, 'prefill_block': 4},
             'prefill_unit_subw': 5,
             'sdpa_fidelity': 'LoFi',
             'sdpa_fp32': False,
             'separate_prefill': True,
             'residual_layout': 'phase_specific',
             'sharded_decode_min_batch': 8,
             'policy_name': 'tp4_p300c_bfp4_lofi_compact_v5',
             'packed_decode_conv': True,
             'batched_native_rs': {'persistent_ccl': False,
                                   'persistent_gather': False,
                                   'ccl_dtype': None,
                                   'compact_column_gather': True,
                                   'compact_gather_move': True,
                                   'fused_intermediate_dram': True,
                                   'fused_compact_input': True}}
        )
        # Separate prefill projections share one BF8 gather. Decode and the
        # one-row prefill specialization retain the packed weight.
        for role in ('gate', 'up'):
            self.policy[role] = dict(cores=80, block=2, readers=1,
                prefill_block=20, prefill_grid=[11,8], prefill_subh=8,
                prefill_out_block_h=8, prefill_out_subblock_w=1)
        if self.layer_type == 'linear_attention':
            self.policy['mix']['prefill_block']=16
            self.policy['gateup']['prefill_block']=8
        for role in ('mix', 'gateup', 'gate', 'up'):
            self.policy[role]['prefill_activation']='bfloat8_b'
            self.policy[role]['prefill_ccl_dtype']='bfloat8_b'
        if self.layer_type == 'full_attention':
            self.policy['decode_rs_geometry']['out_k']=24
            # The 11x8 mix grid permits a 4x1 subblock at a 4096-token
            # chunk. Other projections keep their measured geometry.
            self.policy['mix'].update(prefill_grid=[11,8], prefill_subh=4,
                                      prefill_l1=True, prefill_l1_max_tokens=4096)
        if policy:
            self.policy.update(copy.deepcopy(policy))
        if self.policy.get('precision_prefill_activation'):
            for role in ('mix', 'gateup', 'gate', 'up'):
                self.policy[role]['prefill_activation'] = self.policy['precision_prefill_activation']
        # The inherited BFP4 reader geometry does not fit larger BFP8 tiles.
        # Resolve after dtype overrides and before packing/copying phase views.
        # Preserve explicit legal alternatives used by precision microchecks.
        if self.policy['attention_dtype'] == 'bfloat8_b':
            if self.policy['mix']['block'] == 20 and self.policy['mix']['readers'] == 1:
                self.policy['mix']['block'] = 10
            if self.layer_type == 'linear_attention' and self.policy['mix'].get('prefill_block') == 16:
                self.policy['mix']['prefill_block'] = 8
        if self.policy['mlp_dtype'] == 'bfloat8_b':
            for role in ('gate', 'up'):
                if self.policy[role].get('prefill_block') == 20:
                    self.policy[role]['prefill_block'] = 10
        if self.policy.get('ccl_dtype') == 'bfloat16':
            if self.policy['mix']['block'] == 20 and self.policy['mix']['readers'] == 1:
                self.policy['mix']['block'] = 10
            if self.layer_type == 'linear_attention' and self.policy['mix'].get('prefill_block') == 16:
                self.policy['mix']['prefill_block'] = 8
        if (self.policy['down_dtype'] == 'bfloat8_b' or self.policy.get('ccl_dtype') == 'bfloat16') and self.policy['down']['readers'] == 1:
            # K/shard width is17 tiles; halving to8 would be illegal.
            self.policy['down']['readers'] = 2
        if self.layer_type == 'full_attention' and self.policy.get('kv_dtype') == 'bfloat16':
            # Native page-table staging plus BF16 K/V double buffers exceed L1
            # at K256/C32. K128 retains the existing core cap and page guard.
            self.policy['sdpa_chunk'] = min(self.policy.get('sdpa_chunk', 256), 128)
        if self.policy["residual_layout"] != "phase_specific":
            self.policy["prefill_native_rs"]=False
        self.ar_workspace=None
        if self.policy.get("direct_all_reduce") and self.policy.get("pooled_all_reduce") and self.policy["residual_layout"] in ("replicated","phase_specific"):
            from .decode_workspace import DecodeAllReduceWorkspace
            payload_dtype=getattr(ttnn,self.policy.get("ccl_dtype") or "bfloat16")
            self.ar_workspace=ccl_workspace or DecodeAllReduceWorkspace(mesh_device,payload_dtype)
            self.ar_workspace.validate_owner(mesh_device,payload_dtype)
        elif ccl_workspace is not None:
            raise ValueError("A shared workspace requires pooled_all_reduce")
        self.ccl = self.ar_workspace.ccl if self.ar_workspace is not None else TT_CCL(mesh_device)
        self._ccl_buffers = {}
        self.weight_roles, self.logical_widths = {}, {}
        self.ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        self.sdpa_ckc = self.ckc
        if self.policy.get("sdpa_fidelity"):
            self.sdpa_ckc=ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity,self.policy["sdpa_fidelity"]),
                math_approx_mode=False,fp32_dest_acc_en=self.policy.get("sdpa_fp32",True),packer_l1_acc=True)
        self.matmul_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.get('fused_prefill_fidelity', 'LoFi')),
            math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
        prefix = f"model.language_model.layers.{layer_idx}."
        if not any(k.startswith(prefix) for k in state_dict): prefix = ""
        def weight(name): return state_dict[prefix+name].detach().float()
        def put(value, dtype=ttnn.bfloat16, role=None):
            sharded = isinstance(value, list)
            pieces = value if sharded else [value]
            logical_width = pieces[0].shape[-1]
            mem = ttnn.DRAM_MEMORY_CONFIG
            if role:
                grid = mesh_device.dram_grid_size()
                readers = self.policy[role].get("readers",2)
                padding_readers=math.lcm(readers,self.policy[role].get('weight_pad_readers',readers))
                physical_n = math.ceil(logical_width/(32*grid.x*padding_readers))*32*grid.x*padding_readers
                pieces = [torch.nn.functional.pad(x,(0,physical_n-logical_width)) for x in pieces]
                cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(grid.x-1,grid.y-1))])
                mem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(cores,[pieces[0].shape[0],physical_n//grid.x],ttnn.ShardOrientation.ROW_MAJOR))
            tensor=ttnn.from_torch(torch.cat(pieces,dim=0) if sharded else pieces[0],
                device=mesh_device,dtype=dtype,layout=ttnn.TILE_LAYOUT,memory_config=mem,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device,dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh_device))
            self.logical_widths[id(tensor)] = logical_width
            if role: self.weight_roles[id(tensor)] = role
            return tensor
        def chunks(name,dim=0): return list(weight(name).chunk(4,dim=dim))
        def matrix(parts,role):
            group = "down" if role == "down" else ("mlp" if role in ("gate","up","gateup") else "attention")
            return put([p.T.contiguous() for p in parts],getattr(ttnn,self.policy[group+"_dtype"]),role)
        self.in_norm=put(1+weight("input_layernorm.weight"))
        self.post_norm=put(1+weight("post_attention_layernorm.weight"))
        self.norm_shards={id(self.in_norm):put(list((1+weight("input_layernorm.weight")).chunk(4))),
                          id(self.post_norm):put(list((1+weight("post_attention_layernorm.weight")).chunk(4)))}
        gates, ups=chunks("mlp.gate_proj.weight"),chunks("mlp.up_proj.weight")
        if self.policy.get("separate_prefill") or not self.policy.get("packed_mlp"):
            self.gate, self.up=matrix(gates,"gate"),matrix(ups,"up")
        self.gateup=matrix([torch.cat([g,u],dim=0) for g,u in zip(gates,ups)],"gateup")
        self.down=matrix(chunks("mlp.down_proj.weight",1),"down")
        if self.layer_type == "linear_attention":
            eye=torch.eye(32).reshape(1,1,32,32)
            masks=torch.zeros(1,1,32,96)
            masks[...,:16,:16]=1; masks[...,16:,48:64]=1; masks[...,16:,64:80]=1
            self.gdn_constants=dict(eye=put(eye,ttnn.float32),tril=put(torch.ones_like(eye).tril(),ttnn.float32),
                ones=put(torch.ones_like(eye),ttnn.float32),masks=put(masks,ttnn.float32))
            self.l2_weight=put(torch.full((128,),128**-0.5),ttnn.float32)
            self.conv_start=ttnn.from_torch(torch.tensor([0],dtype=torch.int32),dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,device=mesh_device,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
            q,k,v=weight("linear_attn.in_proj_qkv.weight").split([2048,2048,6144],dim=0)
            z=weight("linear_attn.in_proj_z.weight")
            a,b=weight("linear_attn.in_proj_a.weight"),weight("linear_attn.in_proj_b.weight")
            mix=[]
            for rank in range(4):
                parts=[t.chunk(4,dim=0)[rank] for t in (q,k,v,z,a,b)]
                parts[-2:]=[torch.nn.functional.pad(t,(0,0,0,20)) for t in parts[-2:]]
                mix.append(torch.cat(parts,dim=0))
            self.mix=matrix(mix,"mix")
            mix_widths=[512,512,1536,1536,32,32]
            self.out=matrix(chunks("linear_attn.out_proj.weight",1),"out")
            conv=weight("linear_attn.conv1d.weight")[:,0,:]
            cq,ck,cv=conv.split([2048,2048,6144],dim=0)
            cs=[torch.cat([t.chunk(4)[r] for t in (cq,ck,cv)]) for r in range(4)]
            self.taps=[put([t[:,i].reshape(1,1,-1) for t in cs]) for i in range(4)]
            self.neg_A=put([(-t.exp()).reshape(1,1,12) for t in chunks("linear_attn.A_log")],ttnn.float32)
            self.dt=put([t.reshape(1,1,12) for t in chunks("linear_attn.dt_bias")],ttnn.float32)
            self.out_norm=put(weight("linear_attn.norm.weight"))
        else:
            qg=weight("self_attn.q_proj.weight").reshape(24,2,256,5120)
            ks,vs=chunks("self_attn.k_proj.weight"),chunks("self_attn.v_proj.weight")
            mix=[torch.cat([qg[r*6:(r+1)*6,0].reshape(1536,5120),ks[r],vs[r],
                qg[r*6:(r+1)*6,1].reshape(1536,5120)]) for r in range(4)]
            self.mix=matrix(mix,"mix")
            mix_widths=[1536,256,256,1536]
            self.out=matrix(chunks("self_attn.o_proj.weight",1),"out")
            self.q_norm=put(1+weight("self_attn.q_norm.weight"))
            self.k_norm=put(1+weight("self_attn.k_norm.weight"))
            self.chunk_positions=[ttnn.from_torch(torch.tensor([p],dtype=torch.int32),dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,device=mesh_device,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
                for p in range(0,config.max_position_embeddings,self.chunk_size)]
        if self.policy.get("split_mix"):
            self.mix_parts=[]
            begin=0
            for index,width in enumerate(mix_widths):
                role=f"mix_part_{index}"
                self.policy[role]=copy.deepcopy(self.policy["mix"])
                self.mix_parts.append(matrix([value[begin:begin+width] for value in mix],role))
                begin+=width
        if self.policy.get('prefill_native_rs'):
            from .fused_prefill import prepare_prefill_rs
            prepare_prefill_rs(self)
        if self.policy['residual_layout'] == 'phase_specific':
            if self.policy.get('sharded_decode_min_batch'):
                batched=copy.copy(self)
                batched.policy=copy.deepcopy(self.policy)
                batched.policy.update(residual_layout='sharded',direct_all_reduce=False,pooled_all_reduce=False,
                    ccl_dtype=None,persistent_ccl=True,ccl_compact=True,ccl_l1=True,residual_l1=True,
                    distributed_norm_l1=True,distributed_norm_direct=True)
                batched.ar_workspace=None
                batched._ccl_buffers={}
                if self.policy.get('batched_native_rs'):
                    batched.ar_workspace=self.ar_workspace
                    from .fused_decode import prepare_native_rs
                    batched.policy.update(self.policy['batched_native_rs'])
                    batched.policy['native_rs_decode']=True
                    prepare_native_rs(batched)
                self._batched_decode_impl=batched
            prefill_impl=copy.copy(self)
            prefill_impl.policy=copy.deepcopy(self.policy)
            prefill_impl.policy['residual_layout']='sharded'
            prefill_impl.policy['_prefill_phase']=True
            self.policy['residual_layout']='replicated'
            self._prefill_impl=prefill_impl
        return self

    def decode_implementation(self,batch_size):
        """Setup/query API: choose the batch's inter-layer residual contract.

        Views share weights; warm every required signature before trace capture.
        No tensor conversion occurs here or between decoder layers.
        """
        if hasattr(self,'_batched_decode_impl') and self.policy['sharded_decode_min_batch']<=batch_size<=32:
            return self._batched_decode_impl
        return self

    def decode_residual_layout(self,batch_size):
        return self.decode_implementation(batch_size).policy['residual_layout']

    def prefill_forward(self,hidden_states,**kwargs):
        impl=getattr(self,'_prefill_impl',self)
        width=1280 if impl.policy['residual_layout']=='sharded' else 5120
        if hidden_states.shape[-1] != width:
            raise ValueError(f'Prefill expects {width} hidden channels per device')
        owner=impl.ar_workspace if impl.policy.get('prefill_native_rs') else None
        if owner is None:
            return OptimizedDecoder.prefill_forward(impl,hidden_states,**kwargs)
        owner.begin_prefill()
        try:
            output=OptimizedDecoder.prefill_forward(impl,hidden_states,**kwargs)
            owner.finish_prefill()
            return output
        except BaseException:
            owner.abort()
            raise

    def decode_forward(self,hidden_states,*,active_mask=None,**kwargs):
        if active_mask is not None:
            if tuple(active_mask.shape) != (hidden_states.shape[0], 1, 1):
                raise ValueError("Decode active_mask must have shape [B,1,1]")
            kwargs["active_mask"] = active_mask
        impl=self.decode_implementation(hidden_states.shape[0])
        if not (hidden_states.shape[-1]==1280 and impl is not self):
            impl=self
        expected=1280 if impl.policy['residual_layout']=='sharded' else 5120
        if hidden_states.shape[-1]!=expected:
            raise ValueError(f'Decode expects {expected} hidden channels for this execution layout')
        owner=impl.ar_workspace
        if owner is None or hidden_states.shape[0]>32:
            return OptimizedDecoder.decode_forward(impl,hidden_states,**kwargs)
        owner.begin()
        try:
            output=OptimizedDecoder.decode_forward(impl,hidden_states,**kwargs)
            owner.finish()
            return output
        except BaseException:
            owner.abort()
            raise

    def _gather(self,x,key=None,*,compact_output=False):
        shape=list(x.shape)
        decode=math.prod(shape[:-1]) <= 32 and not self.policy.get("_prefill_phase",False)
        memory=ttnn.L1_MEMORY_CONFIG if decode and self.policy.get("ccl_l1") else ttnn.DRAM_MEMORY_CONFIG
        prefill_cfg=self.policy.get(key,{}) if self.policy.get("_prefill_phase") and isinstance(key,str) else {}
        prefill_l1=prefill_cfg.get("prefill_l1",False) and math.prod(shape[:-1])<=prefill_cfg.get("prefill_l1_max_tokens",4096)
        if prefill_l1:memory=ttnn.L1_MEMORY_CONFIG
        physical_shape=[1,1,math.prod(shape[:-1]),shape[-1]] if decode and self.policy.get("ccl_compact") else [1]*(4-len(shape))+shape
        payload_name=self.policy.get('ccl_dtype') if decode else prefill_cfg.get('prefill_ccl_dtype',self.policy.get('prefill_ccl_dtype'))
        payload=getattr(ttnn,payload_name) if payload_name and not isinstance(key,tuple) else x.dtype
        if decode and self.policy.get('compact_gather_move'):
            x=ttnn.reshape(x,physical_shape)
            x=ttnn.assign(x,memory_config=memory,dtype=payload) if x.dtype!=payload else ttnn.to_memory_config(x,memory)
        else:
            if x.dtype!=payload:x=ttnn.typecast(x,payload)
            x=ttnn.reshape(x if prefill_l1 else ttnn.to_memory_config(x,memory),physical_shape)
        buffer=None
        if self.policy.get("persistent_gather",self.policy.get("persistent_ccl")) and key is not None and math.prod(shape[:-1]) <= 32:
            signature=("ag",key,tuple(x.shape),str(x.dtype))
            if signature not in self._ccl_buffers:
                self._ccl_buffers[signature]=ttnn.zeros([*list(x.shape)[:-1],x.shape[-1]*4],
                    device=self.mesh_device,dtype=x.dtype,layout=ttnn.TILE_LAYOUT,memory_config=memory)
            buffer=self._ccl_buffers[signature]
        x=ttnn.experimental.all_gather_async(x,dim=3,persistent_output_buffer=buffer,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.policy.get("ccl_links",2) if decode else 2,topology=ttnn.Topology.Ring,memory_config=memory,
            chunks_per_sync=self.policy.get("ccl_chunks",10) if decode else 10,num_workers_per_link=self.policy.get("ccl_workers",2) if decode else self.policy.get("prefill_ccl_workers",2),num_buffers_per_channel=self.policy.get("ccl_buffers",2) if decode else 2)
        output_shape=[1,math.prod(shape[:-1]),shape[-1]*4] if compact_output else [*shape[:-1],shape[-1]*4]
        column_dtype=self.policy.get(key,{}).get('prefill_activation') if isinstance(key,str) else None
        keep_dtype=self.policy.get('_prefill_phase') and column_dtype is not None and column_dtype==payload_name
        return ttnn.reshape(ttnn.typecast(x,ttnn.bfloat16) if x.dtype != ttnn.bfloat16 and not keep_dtype else x,output_shape)

    def _reduce(self,x,key=None):
        shape=list(x.shape)
        decode=math.prod(shape[:-1]) <= 32 and not self.policy.get("_prefill_phase",False)
        memory=ttnn.L1_MEMORY_CONFIG if decode and self.policy.get("ccl_rs_l1",self.policy.get("ccl_l1")) else ttnn.DRAM_MEMORY_CONFIG
        direct = decode and self.policy.get("direct_all_reduce") and self.policy["residual_layout"] == "replicated"
        fused_move = direct and self.policy.get("ar_fused_move")
        if not fused_move:
            payload_name=self.policy.get('ccl_dtype') if decode else self.policy.get('prefill_ccl_dtype')
            if payload_name and not isinstance(key,tuple):
                x=ttnn.typecast(x,getattr(ttnn,payload_name))
            physical_shape=[1,1,math.prod(shape[:-1]),shape[-1]] if decode and self.policy.get("ccl_compact") else [1]*(4-len(shape))+shape
            x=ttnn.reshape(ttnn.to_memory_config(x,memory),physical_shape)
        if direct:
            x=ttnn.reshape(x,[1,1,math.prod(shape[:-1]),shape[-1]])
            def ar_mem(width):
                return ttnn.create_sharded_memory_config([32,width//8],ttnn.CoreGrid(x=8,y=1),
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
            memory=ar_mem(shape[-1])
            if fused_move:
                payload_dtype=getattr(ttnn,self.policy.get("ccl_dtype") or "bfloat16")
                # Explicit copy combines sharding and conversion. to_memory_config
                # returns its input for identical memory configs, ignoring dtype.
                x=ttnn.assign(x,memory_config=memory,dtype=payload_dtype)
                assert x.dtype == payload_dtype
            else:
                x=ttnn.to_memory_config(x,memory)
            if self.ar_workspace is not None:
                scratch,semaphore=self.ar_workspace.borrow(key)
            else:
                signature=("ar",key,tuple(x.shape),str(x.dtype))
                if signature not in self._ccl_buffers:
                    self._ccl_buffers[signature]=ttnn.zeros([*list(x.shape)[:-1],x.shape[-1]*4],
                        device=self.mesh_device,dtype=x.dtype,layout=ttnn.TILE_LAYOUT,memory_config=ar_mem(shape[-1]*4))
                scratch=self._ccl_buffers[signature]
                semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle()
            y=ttnn.experimental.all_reduce_async(x,scratch,cluster_axis=1,
                mesh_device=self.mesh_device,multi_device_global_semaphore=semaphore,
                memory_config=memory,dtype=ttnn.bfloat16,topology=ttnn.Topology.Ring,
                num_links=self.policy.get("ccl_links",2))
            if math.prod(shape[:-1])>1:
                # Restoring [B,1,H] expands to B physical tile rows. Spread those
                # rows over a rectangular80-core boundary instead of overflowing8.
                expanded_mem=ttnn.create_sharded_memory_config([32,64],ttnn.CoreGrid(x=10,y=8),
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
                return ttnn.reshape(y,shape,memory_config=expanded_mem)
            return ttnn.reshape(y,shape)
        buffers=None
        if self.policy.get("persistent_ccl") and key is not None and math.prod(shape[:-1]) <= 32:
            signature=("rs",key,tuple(x.shape),str(x.dtype))
            if signature not in self._ccl_buffers:
                intermediate,penult=ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(x,dim=3,topology=ttnn.Topology.Ring)
                output=ttnn.zeros([*list(x.shape)[:-1],x.shape[-1]//4],device=self.mesh_device,
                    dtype=x.dtype,layout=ttnn.TILE_LAYOUT,memory_config=memory)
                self._ccl_buffers[signature]=[intermediate,output,penult]
            buffers=self._ccl_buffers[signature]
        x=ttnn.experimental.reduce_scatter_minimal_async(x,dim=3,persistent_output_buffers=buffers,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.policy.get("ccl_links",2) if decode else 2,topology=ttnn.Topology.Ring,memory_config=memory,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=self.policy.get("ccl_chunks",10) if decode else 10,num_workers_per_link=self.policy.get("ccl_workers",2) if decode else self.policy.get("prefill_ccl_workers",2),num_buffers_per_channel=self.policy.get("ccl_buffers",2) if decode else 2)
        x=ttnn.reshape(ttnn.typecast(x,ttnn.bfloat16) if x.dtype != ttnn.bfloat16 else x,[*shape[:-1],shape[-1]//4])
        x=self._gather(x,key) if self.policy["residual_layout"] == "replicated" else x
        if decode and self.policy.get("residual_l1"):
            output_shape=list(x.shape)
            width=output_shape[-1]
            x=ttnn.reshape(x,[1,1,math.prod(output_shape[:-1]),width])
            mem=ttnn.create_sharded_memory_config([32,width//8],ttnn.CoreGrid(x=8,y=1),
                ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
            x=ttnn.to_memory_config(x,mem)
            if output_shape[0]>1 and width==1280:
                expanded=ttnn.create_sharded_memory_config([output_shape[0]*32,32],ttnn.CoreGrid(x=8,y=5),
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
                x=ttnn.reshape(x,output_shape,memory_config=expanded)
            else:
                x=ttnn.reshape(x,output_shape)
        return x

    def _linear(self,x,w,activation=None,**kwargs):
        role=self.weight_roles[id(w)]
        owner=self.ar_workspace if self.policy.get('_prefill_phase') and self.policy.get('prefill_native_rs') and role in ('out','down') else None
        if owner is not None:owner.require_prefill_role(role)
        y=self._linear_impl(x,w,activation,**kwargs)
        if owner is not None:owner.complete_prefill_role(role)
        return y

    def _linear_impl(self,x,w,activation=None,**kwargs):
        role=self.weight_roles[id(w)]
        if self.policy.get('_prefill_phase') and self.policy.get('prefill_native_rs') and role in ('out','down') and list(x.shape)[:2]==[1,4096]:
            from .fused_prefill import prefill_rs_projection
            return prefill_rs_projection(self,x,role)
        compact_column=self.policy.get('compact_column_gather') and not self.policy.get('_prefill_phase') and x.shape[0]>1 and x.shape[1]==1
        if self.policy.get('native_rs_decode') and role in ('out','down') and (x.shape[1]==1 or kwargs.get('decode_batch')):
            from .fused_decode import native_rs_projection
            return native_rs_projection(self,x,role,kwargs.get('decode_batch') or x.shape[0])
        if self.ar_workspace is not None and not self.policy.get('_prefill_phase') and role in ('out','down') and (kwargs.get('decode_batch') or x.shape[0])<=32 and (x.shape[1]==1 or kwargs.get('decode_batch')):
            self.ar_workspace.require_role(role)
        if w is self.mix and self.policy.get("split_mix"):
            if x.shape[-1] == 1280:
                if compact_column:kwargs['decode_batch']=x.shape[0]
                x=self._gather(x,"mix",compact_output=compact_column)
            return ttnn.concat([super(MultichipDecoder,self)._linear(x,part,activation,**kwargs) for part in self.mix_parts],dim=-1)
        if x.shape[-1] == 1280 and self.weight_roles[id(w)] in ("mix","gate","up","gateup"):
            if compact_column:kwargs['decode_batch']=x.shape[0]
            x=self._gather(x,self.weight_roles[id(w)],compact_output=compact_column)
        batch=kwargs.get('decode_batch') or x.shape[0]
        compact_row=self.policy.get('compact_row_reduce') and not self.policy.get('_prefill_phase') and role in ('out','down') and batch>1 and (x.shape[1]==1 or kwargs.get('decode_batch'))
        if compact_row:kwargs['packed_output']=True
        y=super()._linear(x,w,activation,**kwargs)
        if role in ('out','down'):y=self._reduce(y,role)
        if compact_row:
            shape=[batch,1,y.shape[-1]]
            if y.is_sharded():
                width=y.shape[-1]
                grid=ttnn.CoreGrid(x=10,y=8) if width==5120 else ttnn.CoreGrid(x=8,y=5)
                mem=ttnn.create_sharded_memory_config([batch*32,width//(grid.x*grid.y)],grid,
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
                return ttnn.reshape(y,shape,memory_config=mem)
            return ttnn.reshape(y,shape)
        return y

    def _norm(self,x,w):
        if x.shape[-1] == 1280 and w.shape[-1] == 5120:
            shape=list(x.shape)
            distributed_l1=self.policy.get("distributed_norm_l1") and shape[-2]==1
            if distributed_l1 and self.policy.get("distributed_norm_direct"):
                x=ttnn.reshape(x,[1,*shape])
            else:
                x=ttnn.reshape(ttnn.to_memory_config(x,ttnn.DRAM_MEMORY_CONFIG),[1,*shape])
            options={}
            if distributed_l1:
                x=ttnn.reshape(x,[1,1,shape[0],1280])
                mem=ttnn.create_sharded_memory_config([32,160],ttnn.CoreGrid(x=8,y=1),
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
                x=ttnn.to_memory_config(x,mem)
                options['program_config']=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(8,1),subblock_w=1,block_h=1,block_w=5,inplace=False)
            stats=ttnn.rms_norm_pre_all_gather(x,compute_kernel_config=self.ckc,dtype=ttnn.bfloat16,**options)
            stats=self._gather(stats,("norm",id(w)))
            if distributed_l1:
                stats=ttnn.to_memory_config(stats,ttnn.create_sharded_memory_config([32,128],ttnn.CoreGrid(x=1,y=1),
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True))
            normalized=ttnn.rms_norm_post_all_gather(x,stats,epsilon=self.config.rms_norm_eps,
                weight=self.norm_shards[id(w)],compute_kernel_config=self.ckc,**options)
            if distributed_l1 and shape[0]>1:
                expanded=ttnn.create_sharded_memory_config([shape[0]*32,32],ttnn.CoreGrid(x=8,y=5),
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
                return ttnn.reshape(normalized,shape,memory_config=expanded)
            return ttnn.reshape(normalized,shape)
        if x.shape[-1] == 5120 and x.shape[-2] == 1 and (self.policy.get("norm_cores",8)>11 or (self.policy.get("carry_batched_norm") and x.shape[0]>1)):
            # Norm reducers use a bounding rectangle. A prefix of the11-wide
            # device grid is not a valid40/80-core collective norm contract.
            shape=list(x.shape)
            rows=math.prod(shape[:-1])
            nc=self.policy.get("norm_cores",8)
            device_grid=self.mesh_device.compute_with_storage_grid_size()
            nx=next(v for v in range(device_grid.x,0,-1) if nc%v==0 and nc//v<=device_grid.y)
            grid=ttnn.CoreGrid(x=nx,y=nc//nx)
            mem=ttnn.create_sharded_memory_config([math.ceil(rows/32)*32,5120//nc],grid,
                ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
            x=ttnn.to_memory_config(ttnn.reshape(x,[1,1,rows,5120]),mem)
            pc=ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(nx,nc//nx),
                subblock_w=1,block_h=math.ceil(rows/32),block_w=160//nc,inplace=False)
            x=ttnn.rms_norm(x,weight=w,epsilon=self.config.rms_norm_eps,program_config=pc,
                memory_config=mem,compute_kernel_config=self.ckc)
            if rows>1 and self.policy.get("carry_batched_norm"):
                expanded_mem=ttnn.create_sharded_memory_config([32,64],ttnn.CoreGrid(x=10,y=8),
                    ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
                return ttnn.reshape(x,shape,memory_config=expanded_mem)
            if not (self.policy.get("carry_residual") and rows==1):
                x=ttnn.to_memory_config(x,ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.reshape(x,shape)
        return super()._norm(x,w)

    def allocate_state(self, *, batch_size, context_length=None):
        """Setup only. Owners release state after all traces referencing it retire."""
        def zeros(shape, dtype):
            return ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                              device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.layer_type == "linear_attention":
            return LinearState(ttnn.zeros([batch_size, 3, 2560], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                               zeros([batch_size, 12, 128, 128], ttnn.float32))
        if context_length is None or not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("Supply a valid context_length for the paged pool")
        # Flexible chunked attention reads page-table rows in 32-byte sticks:
        # round physical capacity to eight INT32 page IDs, not logical context.
        pages = batch_size * ((context_length + 255) // 256) * 8
        return PagedState(zeros([pages, 1, 32, 256], getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))),
                          zeros([pages, 1, 32, 256], getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))))

    def _finish(self, residual, mixer):
        if mixer.is_sharded():
            residual = ttnn.to_memory_config(residual, mixer.memory_config())
        x = ttnn.add(residual, mixer, memory_config=mixer.memory_config())
        h = self._norm(x, self.post_norm)
        packed_prefill = bool(self.policy.get("_prefill_phase") and self.policy.get("packed_mlp")
                              and not self.policy.get("separate_prefill") and not self.policy.get("minimal_mlp"))
        if h.shape[-1] == 1280 and h.shape[1] > 1 and not packed_prefill:
            h=self._gather(h,"gateup")  # shared by separate projections; packed path retains selected input dtype
        compact = self.policy.get("packed_mlp") and h.shape[1] == 1 and h.shape[0] > 1 and not (self.policy.get("minimal_mlp") and self.policy.get("minimal_decode"))
        if self.policy.get("packed_mlp") and (h.shape[1] == 1 or not self.policy.get("separate_prefill")):
            if self.policy.get("minimal_mlp") and (h.shape[1] > 1 or self.policy.get("minimal_decode")):
                grid = self.mesh_device.compute_with_storage_grid_size()
                inter = ttnn.experimental.minimal_matmul(ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG),
                    self.gateup, fuse_swiglu=True,
                    config=ttnn.MinimalMatmulConfig(M_block_size=min(self.policy.get("minimal_m", 4), math.ceil(h.shape[-2]/32)),
                        K_block_size=self.policy.get("minimal_k", 8), N_block_size=self.policy.get("minimal_n", 8),
                        subblock_h=1, subblock_w=2, compute_with_storage_grid_size=grid),
                    compute_kernel_config=self.matmul_ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16)
            else:
                gu = self._linear(h, self.gateup, packed_output=compact)
                inter = ttnn.multiply(gu[..., :4352], gu[..., 4352:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    memory_config=ttnn.L1_MEMORY_CONFIG if self.policy.get("l1_intermediates") and h.shape[1] == 1 else ttnn.DRAM_MEMORY_CONFIG)
        else:
            gate = self._linear(h, self.gate, activation="silu")
            inter = ttnn.multiply(gate, self._linear(h, self.up))
        mlp_decode_batch = h.shape[0] if compact else None
        if packed_prefill:
            # The activation product owns its output; these projection inputs
            # and split-output backing storage are no longer consumed.
            del h, gu
        elif self.policy.get('_prefill_phase') and self.policy.get('separate_prefill') and h.shape[1] > 1:
            # Both projections have completed and multiply owns its output.
            del h, gate
        elif self.policy.get("prefill_l1"):
            del h
        mlp = self._linear(inter, self.down, decode_batch=mlp_decode_batch)
        if mlp.is_sharded():
            x = ttnn.to_memory_config(x, mlp.memory_config())
        return ttnn.add(x, mlp, memory_config=mlp.memory_config())

    def _packed_decode_convolution(self, projected, state):
        """Each retained fourth row sees only its own user's three history rows.

        Assemble/extract in row-major layout so tile padding cannot become data.
        The other three outputs per window are discarded, never fed back.
        """
        batch=projected.shape[0]
        current = ttnn.to_layout(projected[..., :2560], ttnn.ROW_MAJOR_LAYOUT)
        windows = ttnn.concat([state.conv, current], dim=1)
        packed = ttnn.reshape(windows, [1, batch*4, 2560])
        physical = ((batch*4 + 31)//32)*32
        if physical != batch*4:
            packed = ttnn.pad(packed, [(0,0),(0,physical-batch*4),(0,0)], value=0)
        packed = ttnn.to_layout(packed, ttnn.TILE_LAYOUT)
        hist = state.conv[:1]
        outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            packed, hist, *self.taps, 512, 512, 1536,
            actual_start=self.conv_start, predecessor_carry=hist,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
        qflat, kflat, vflat = [ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(t,ttnn.ROW_MAJOR_LAYOUT)[:,:batch*4], [batch,4,width])[:,3:4],
            ttnn.TILE_LAYOUT) for t,width in zip(outputs,(512,512,1536))]
        next_conv = windows[:,1:4]
        return qflat,kflat,vflat,next_conv

    def _delta(self, x, state, *, prepared_metadata=None, active_mask=None):
        batch, length, _ = x.shape
        if active_mask is not None and (length != 1 or prepared_metadata is not None):
            raise ValueError("Decode active_mask requires a one-token non-prefill call")
        projected = self._linear(x, self.mix)
        if prepared_metadata is None and self.policy.get('packed_decode_conv') and batch>=8 and length==1 and not self.policy.get('_prefill_phase'):
            qflat,kflat,vflat,next_conv=self._packed_decode_convolution(projected,state)
        else:
            physical_conv = ((length + 31) // 32) * 32
            padded_projected = ttnn.pad(projected, [(0, 0), (0, physical_conv-length), (0, 0)], value=0) if physical_conv != length else projected
            qs, ks, vs = [], [], []
            for bidx in range(batch):
                hist = ttnn.to_layout(state.conv[bidx:bidx+1], ttnn.ROW_MAJOR_LAYOUT)
                cq, ck, cv = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                    padded_projected[bidx:bidx+1], hist, *self.taps, 512, 512, 1536,
                    actual_start=self.conv_start, predecessor_carry=hist,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
                qs.append(cq[:, :length]); ks.append(ck[:, :length]); vs.append(cv[:, :length])
            qflat, kflat, vflat = [parts[0] if batch == 1 else ttnn.concat(parts, dim=0) for parts in (qs, ks, vs)]
            if prepared_metadata is not None:
                # Select the last three *logical* projected inputs, including
                # incoming history for n=0/1/2. Indices vary as device data;
                # shapes depend only on the physical prepared bucket.
                projected_qkv = ttnn.to_layout(projected[..., :2560], ttnn.ROW_MAJOR_LAYOUT)
                history_table = ttnn.reshape(ttnn.concat([state.conv, projected_qkv], dim=1),
                                             [batch * (length + 3), 2560])
                next_conv = ttnn.embedding(prepared_metadata["history_indices"], history_table,
                                           layout=ttnn.ROW_MAJOR_LAYOUT)
            elif self.policy.get("history_tile_tail") and length >= 32 and length % 32 == 0:
                # Slice a complete tile before untilize, then select logical rows.
                # Direct nonaligned tiled slicing would add an implicit retile.
                tail = projected[:, length-32:length, :2560]
                next_conv = ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT)[:, -3:, :]
            else:
                next_conv = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)[:, length-3:length, :2560] if length >= 3 else ttnn.concat(
                    [state.conv[:, length:, :], ttnn.to_layout(projected[..., :2560], ttnn.ROW_MAJOR_LAYOUT)], dim=1)
        if active_mask is not None:
            # Row-major convolution carry is preserved for inactive decode slots.
            # where uses tiled operands; round-trip only in this masked variant.
            conv_active = ttnn.to_layout(active_mask, ttnn.TILE_LAYOUT)
            next_conv = ttnn.to_layout(ttnn.where(conv_active,
                ttnn.to_layout(next_conv, ttnn.TILE_LAYOUT),
                ttnn.to_layout(state.conv, ttnn.TILE_LAYOUT)), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(next_conv, state.conv)

        def l2(t):
            t = ttnn.typecast(t, ttnn.float32)
            return ttnn.rms_norm(t, weight=self.l2_weight, epsilon=1e-6/128, compute_kernel_config=self.ckc)

        a = ttnn.typecast(projected[..., 4096:4108], ttnn.float32)
        decay = ttnn.multiply(self.neg_A, ttnn.softplus(ttnn.add(a, self.dt)))
        beta = ttnn.sigmoid(ttnn.typecast(projected[..., 4128:4140], ttnn.float32))
        if prepared_metadata is not None or active_mask is not None:
            # Mask after convolution and nonlinear gates: zero hidden padding
            # alone leaves convolution carry and nonzero decay/beta behind.
            valid = prepared_metadata["valid_mask"] if prepared_metadata is not None else ttnn.to_layout(active_mask, ttnn.TILE_LAYOUT)
            qflat, kflat, vflat = (ttnn.multiply(t, ttnn.typecast(valid, t.dtype))
                                  for t in (qflat, kflat, vflat))
            valid_fp32 = ttnn.typecast(valid, ttnn.float32)
            decay = ttnn.multiply(decay, valid_fp32)
            beta = ttnn.multiply(beta, valid_fp32)
        if length > 1:
            physical = ((length + 31) // 32) * 32
            def padded(t):
                return ttnn.pad(t, [(0, 0), (0, physical-length), (0, 0)], value=0) if physical != length else t
            chunk_outputs, chunk_states = [], []
            kernel_batch = (self.mesh_device.compute_with_storage_grid_size().x * self.mesh_device.compute_with_storage_grid_size().y) // 12
            pq, pk, pv, pg, pb = map(padded, (qflat, kflat, vflat, decay, beta))
            for begin in range(0, batch, kernel_batch):
                end = min(begin + kernel_batch, batch)
                out, final = ttnn.transformer.chunk_gated_delta_rule(
                    pq[begin:end], pk[begin:end], pv[begin:end], pg[begin:end], pb[begin:end],
                    initial_state=state.recurrent[begin:end], output_final_state=True,
                    chunk_size=32, output_head_major=True, compute_kernel_config=self.ckc, **self.gdn_constants)
                chunk_outputs.append(out); chunk_states.append(final)
            y = chunk_outputs[0] if len(chunk_outputs) == 1 else ttnn.concat(chunk_outputs, dim=0)
            new_state = chunk_states[0] if len(chunk_states) == 1 else ttnn.concat(chunk_states, dim=0)
            norm_gate = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                ttnn.typecast(y, ttnn.bfloat16), padded_projected, self.out_norm, 12,
                gate_column_offset=2560, epsilon=self.config.rms_norm_eps, output_dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
            if prepared_metadata is not None:
                # An entirely inactive row must preserve even the low FP32 bits
                # that an all-zero GDN chunk's identity matmul could round.
                active = ttnn.reshape(prepared_metadata["valid_mask"][:, :1, :], [batch, 1, 1, 1])
                new_state = ttnn.where(active, new_state, state.recurrent)
            ttnn.copy(new_state, state.recurrent)
            return self._linear(ttnn.multiply(norm_gate[:, :length], projected[..., 2560:4096]), self.out)
        else:
            q = l2(reshape(qflat, [batch, length, 4, 128]))
            k = l2(reshape(kflat, [batch, length, 4, 128]))
            v = ttnn.typecast(reshape(vflat, [batch, length, 12, 128]), ttnn.float32)
            # This kernel assigns one value head to each core. Split its batch axis,
            # retaining batched projections and an explicit state for every request.
            grid = self.mesh_device.compute_with_storage_grid_size()
            kernel_batch = (grid.x * grid.y) // 12
            ys, states = [], []
            for begin in range(0, batch, kernel_batch):
                end = min(begin + kernel_batch, batch)
                y_part, state_part = fp32_recurrence(
                    q[begin:end], k[begin:end], v[begin:end], decay[begin:end], beta[begin:end],
                    initial_state=state.recurrent[begin:end], output_final_state=True, compute_kernel_config=self.ckc,
                    # Keep the original state available for inactive prepared
                    # slots. The FP32 recurrence math is unchanged; only its
                    # destination differs from the default in-place decode.
                    inplace_state=(self.policy.get("inplace_recurrence", False)
                                   and prepared_metadata is None and active_mask is None),
                )
                ys.append(y_part)
                states.append(state_part)
            y = ttnn.concat(ys, dim=0) if len(ys) > 1 else ys[0]
            new_state = ttnn.concat(states, dim=0) if len(states) > 1 else states[0]
        if prepared_metadata is not None or active_mask is not None:
            row_active = (prepared_metadata["valid_mask"][:, :1, :] if prepared_metadata is not None
                          else ttnn.to_layout(active_mask, ttnn.TILE_LAYOUT))
            active = ttnn.reshape(row_active, [batch, 1, 1, 1])
            new_state = ttnn.where(active, new_state, state.recurrent)
        if new_state.buffer_address() != state.recurrent.buffer_address():
            ttnn.copy(new_state, state.recurrent)
        y = self._norm(ttnn.typecast(y, ttnn.bfloat16), self.out_norm)
        z = reshape(projected[..., 2560:4096], [batch, length, 12, 128])
        return self._linear(reshape(ttnn.multiply(z, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), [batch, length, 1536]), self.out)

    # Pack users as independent sequence rows; never cache position tables.
    def _packed_decode_rope_tables(self,cos,sin,batch):
        return tuple(ttnn.to_layout(ttnn.reshape(ttnn.to_layout(t,ttnn.ROW_MAJOR_LAYOUT),[1,1,batch,64]),
            ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG) for t in (cos,sin))

    def _packed_decode_rope(self,x,tables):
        batch=x.shape[0]
        assert x.shape[2]==1 and 1<batch<=32
        rot=ttnn.to_layout(x[...,:64],ttnn.ROW_MAJOR_LAYOUT)
        packed=ttnn.to_layout(ttnn.permute(rot,(2,1,0,3),memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG)
        y=ttnn.experimental.rotary_embedding(packed,*tables,compute_kernel_config=self.ckc,
            memory_config=ttnn.L1_MEMORY_CONFIG)
        y=ttnn.to_layout(y,ttnn.ROW_MAJOR_LAYOUT)[:,:,:batch,:]
        y=ttnn.to_layout(ttnn.permute(y,(2,1,0,3),memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.concat([y,x[...,64:]],dim=-1,memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _packed_prefill_rope(self,value,tables,batch,length):
        heads=value.shape[1]
        rot,tail=value[...,:64],value[...,64:]
        packed=ttnn.reshape(ttnn.permute(rot,(1,0,2,3)),[1,heads,batch*length,64])
        y=ttnn.experimental.rotary_embedding(packed,*tables,compute_kernel_config=self.ckc)
        y=ttnn.permute(ttnn.reshape(y,[heads,batch,length,64]),(1,0,2,3))
        return ttnn.concat([y,tail],dim=-1,memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _attention(self, x, state, *, page_table, cos, sin, current_pos=None, chunk_start=0,
                   prepared_metadata=None):
        batch, length, _ = x.shape
        if prepared_metadata is not None:
            chunk_start=prepared_metadata.get('start_alignment',0)
        projected = self._linear(x, self.mix)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            projected[..., :2048], num_heads=6, num_kv_heads=1, transpose_key=False)
        if current_pos is not None and batch>=self.policy['packed_decode_rope_min_batch']:
            tables=self._packed_decode_rope_tables(cos,sin,batch)
            q=self._packed_decode_rope(self._norm(q,self.q_norm),tables)
            k=self._packed_decode_rope(self._norm(k,self.k_norm),tables)
        elif current_pos is None and batch>=self.policy['packed_prefill_rope_min_batch'] and length%32==0:
            # Internal working rows are already padded/masked by prefill_forward.
            # Fold only tile-aligned rows; keep the original path for other shapes.
            tables=tuple(ttnn.reshape(t,[1,1,batch*length,64]) for t in (cos,sin))
            q=self._packed_prefill_rope(self._norm(q,self.q_norm),tables,batch,length)
            k=self._packed_prefill_rope(self._norm(k,self.k_norm),tables,batch,length)
        else:
            q = self._rope(self._norm(q, self.q_norm), cos, sin)
            k = self._rope(self._norm(k, self.k_norm), cos, sin)
        if current_pos is not None:
            # Decode kernel layout is [1,B,H,D]. Cache update requires height sharding.
            q = ttnn.permute(q, (2, 0, 1, 3))
            k, v = ttnn.permute(k, (2, 0, 1, 3)), ttnn.permute(v, (2, 0, 1, 3))
            mem = ttnn.create_sharded_memory_config(
                [32, 256], core_grid=ttnn.num_cores_to_corerangeset(batch, self.mesh_device.compute_with_storage_grid_size(), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            for cache, update in ((state.key, k), (state.value, v)):
                ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(update, mem),
                                                     update_idxs_tensor=current_pos, page_table=page_table)
            # The reader fetches complete chunks before applying the causal mask.
            # External page tables own their mappings: never read alignment bytes
            # as page IDs or invent padding-page ownership. Choose a chunk that
            # divides the explicitly mapped capacity, using static shape metadata.
            sdpa_chunk = self.policy.get("sdpa_chunk", 256)
            if sdpa_chunk < 32 or sdpa_chunk & (sdpa_chunk - 1):
                raise ValueError("sdpa_chunk must be a power of two >=32")
            mapped_tokens = page_table.shape[-1] * 32
            while mapped_tokens % sdpa_chunk:
                sdpa_chunk //= 2
            self.effective_sdpa_chunk = sdpa_chunk
            y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, state.key, state.value, page_table_tensor=page_table, cur_pos_tensor=current_pos,
                scale=256**-0.5, compute_kernel_config=self.sdpa_ckc,
                # The default is 32 tokens and one core/head: 2048 serial BF16
                # softmax-stat updates at full context. Bound the accumulation
                # chain with larger chunks and sequence-parallel cores.
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32, k_chunk_size=sdpa_chunk, max_cores_per_head_batch=self.policy.get("sdpa_cores", 16),
                    exp_approx_mode=self.policy.get("decode_sdpa_approx",False),
                ),
            )
            y = reshape(y, [batch, 1, 1536])
        else:
            fill_table = (page_table[:, chunk_start // 32:] if prepared_metadata is None
                          else prepared_metadata["fill_page_table"])
            # Prefill is scheduled eagerly over prepared fixed-slot groups.
            # Warming all rows prepares every slice/batch_idx program; selecting
            # a subset here adds no signature and must not overwrite another
            # group's KV pages. The mask/indices must describe the same rows.
            active_rows = range(batch) if prepared_metadata is None else prepared_metadata.get("active_rows", range(batch))
            active_rows = tuple(active_rows)
            if len(set(active_rows)) != len(active_rows) or any(not 0 <= user < batch for user in active_rows):
                raise ValueError("Prepared active_rows must contain distinct configured slot indices")
            for user in active_rows:
                ttnn.experimental.paged_fill_cache(state.key, ttnn.typecast(k[user:user + 1], state.key.dtype), fill_table, batch_idx=user)
                ttnn.experimental.paged_fill_cache(state.value, ttnn.typecast(v[user:user + 1], state.value.dtype), fill_table, batch_idx=user)
            sdpa_options = {}
            if self.policy.get("prefill_sdpa_q"):
                qc = 32
                while qc * 2 <= min(self.policy["prefill_sdpa_q"], length):
                    qc *= 2
                while length % qc or (chunk_start and chunk_start % qc):
                    qc //= 2
                kc = self.policy.get("prefill_sdpa_k", 128)
                while chunk_start and chunk_start % kc:
                    kc //= 2
                sdpa_options["program_config"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=self.policy.get("prefill_sdpa_approx", False))
            y = ttnn.transformer.chunked_scaled_dot_product_attention(
                q, state.key, state.value, page_table_tensor=page_table, chunk_start_idx=None,
                chunk_start_idx_tensor=(self.chunk_positions[chunk_start // self.chunk_size]
                                        if prepared_metadata is None else prepared_metadata["chunk_start"]),
                compute_kernel_config=self.ckc, **sdpa_options,
            )
            y = ttnn.transformer.concatenate_heads(y)
        # Matches the installed HF reference. Gate semantics are independently audited.
        y = ttnn.multiply(projected[..., 2048:], y, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return self._linear(y, self.out)


def fp32_recurrence(q, k, v, g, beta, *, initial_state, scale=None, output_final_state=True,
                    compute_kernel_config=None, inplace_state=False):
    """[B,T,Hq,128], [B,T,12,128], log g/beta [B,T,12] -> output, FP32 state.

    Only final-state mode is supported. The compute policy is fixed HiFi4/FP32;
    the optional compute_kernel_config argument preserves call-site compatibility.
    """
    batch, length, qheads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if key_dim != 128 or value_dim != 128 or value_heads != 12 or qheads not in (4, 12):
        raise ValueError("FP32 recurrence supports the Qwen3.8-27B head dimensions only")
    if not output_final_state or initial_state is None:
        raise ValueError("An explicit initial state and final state output are required")
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    heads = batch * value_heads
    if heads > grid.x * grid.y:
        raise ValueError("Split recurrence batches to at most the available cores / 12")
    for x in (q, k, v, g, beta, initial_state):
        if x.dtype != ttnn.float32:
            raise ValueError("Recurrence inputs and state must be FP32")

    def split(x, h, width):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), [batch * h, length, width])
        if h != value_heads:
            x = ttnn.repeat_interleave(x, value_heads // h, dim=0)
        return reshape(x, [heads * length, 1, width]) if length > 1 else x

    q = split(ttnn.multiply(q, key_dim ** -0.5 if scale is None else scale), qheads, key_dim)
    k = split(k, qheads, key_dim)
    v = split(v, value_heads, value_dim)

    def split_scalar(x):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1)), [heads, length])
        return reshape(x, [heads * length, 1, 1])

    # Exponentiation remains the public FP32 TTNN op, as in the upstream wrapper.
    decay = split_scalar(ttnn.exp(g))
    beta = split_scalar(beta)
    state = ttnn.reshape(initial_state, [heads, key_dim, value_dim])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([heads * length, 1, value_dim]), ttnn.float32,
                                        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    # Each head reads its entire initial state once, before computing final state.
    # Final-only writer writes that same disjoint head after its computation.
    final = state if inplace_state else ttnn.allocate_tensor_on_device(
        ttnn.Shape([heads, key_dim, value_dim]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    coords = [ttnn.CoreCoord(h // grid.y, h % grid.y) for h in range(heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    kt, vt, tile_bytes = 4, 4, 4096
    cb_sizes = [kt, kt, vt, 2 if length > 1 else 1, 2 if length > 1 else 1,
                kt * vt, 2 * vt, 2 * kt * vt, kt * vt, kt * vt, kt * vt, vt, vt, kt, kt * vt, kt * vt]
    cbs = [ttnn.CBDescriptor(total_size=tiles * tile_bytes, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index,
                                     data_format=ttnn.float32, page_size=tile_bytes)])
           for index, tiles in enumerate(cb_sizes)]
    reader_ct = [kt, vt, 0]
    # Ring indexing is disabled. A valid dummy accessor keeps the unused template
    # argument well-formed without any scratch buffer or new persistent state.
    for tensor in (q, k, v, decay, beta, state, state):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_ct = [kt, vt, 0]
    for tensor in (out, final):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for h, core in enumerate(coords):
        rr[core.x][core.y] = [h, length, q.buffer_address(), k.buffer_address(), v.buffer_address(),
                              decay.buffer_address(), beta.buffer_address(), state.buffer_address(), 0, 0, heads]
        wr[core.x][core.y] = [h, length, out.buffer_address(), final.buffer_address(), heads]
        cr[core.x][core.y] = [length]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # Only copy/SFPU consumers get direct-to-DST FP32. CB 14/15 are separate
    # packed copies of state/sd for matmul; q/k/u/kcol remain matrix operands.
    for index in (2, 3, 4, 5, 8, 9, 10, 11):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    descriptor = ttnn.ProgramDescriptor(kernels=[
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "reader_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=reader_ct, runtime_args=rr,
                             config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "writer_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=writer_ct, runtime_args=wr,
                             config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_KERNEL, core_ranges=cores, compile_time_args=[kt, vt, 0],
                             runtime_args=cr, config=config),
    ], cbs=cbs, semaphores=[])
    ttnn.generic_op([q, k, v, decay, beta, state, out, final], descriptor)
    out = ttnn.permute(ttnn.reshape(out, [batch, value_heads, length, value_dim]), (0, 2, 1, 3))
    return out, ttnn.reshape(final, [batch, value_heads, key_dim, value_dim])
