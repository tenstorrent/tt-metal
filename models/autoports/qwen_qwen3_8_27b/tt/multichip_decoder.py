# SPDX-License-Identifier: Apache-2.0
"""Four-chip TP decoder. Baseline math: optimized_decoder. See stage04 evidence."""
import copy
import math
import ttnn
from .optimized_decoder import OptimizedDecoder, LinearState, PagedState, default_policy, reshape
from .fp32_recurrence import _KERNEL, _UPSTREAM
from models.common.modules.tt_ccl import TT_CCL


class MultichipDecoder(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None):
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
        self.policy.update(mix={"cores":40,"block":4,"readers":2}, out={"cores":16,"block":3,"prefill_block":16,"readers":2},
                           down={"cores":17,"block":8,"prefill_block":8,"readers":2},
                           carry_residual=False, l1_mlp=False, separate_prefill=False, residual_layout="phase_specific")
        if policy:
            self.policy.update(copy.deepcopy(policy))
        self.ccl = TT_CCL(mesh_device)
        self.weight_roles, self.logical_widths = {}, {}
        self.ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        self.matmul_ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
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
                physical_n = math.ceil(logical_width/(32*grid.x*readers))*32*grid.x*readers
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
        def matrix(parts,role): return put([p.T.contiguous() for p in parts],ttnn.bfloat4_b,role)
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
            self.mix=matrix([torch.cat([qg[r*6:(r+1)*6,0].reshape(1536,5120),ks[r],vs[r],
                qg[r*6:(r+1)*6,1].reshape(1536,5120)]) for r in range(4)],"mix")
            self.out=matrix(chunks("self_attn.o_proj.weight",1),"out")
            self.q_norm=put(1+weight("self_attn.q_norm.weight"))
            self.k_norm=put(1+weight("self_attn.k_norm.weight"))
            self.chunk_positions=[ttnn.from_torch(torch.tensor([p],dtype=torch.int32),dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,device=mesh_device,mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
                for p in range(0,config.max_position_embeddings,self.chunk_size)]
        if self.policy['residual_layout'] == 'phase_specific':
            prefill_impl=copy.copy(self)
            prefill_impl.policy=copy.deepcopy(self.policy)
            prefill_impl.policy['residual_layout']='sharded'
            self.policy['residual_layout']='replicated'
            self._prefill_impl=prefill_impl
        return self

    def prefill_forward(self,hidden_states,**kwargs):
        impl=getattr(self,'_prefill_impl',self)
        width=1280 if impl.policy['residual_layout']=='sharded' else 5120
        if hidden_states.shape[-1] != width:
            raise ValueError(f'Prefill expects {width} hidden channels per device')
        return OptimizedDecoder.prefill_forward(impl,hidden_states,**kwargs)

    def _gather(self,x):
        shape=list(x.shape)
        x=ttnn.reshape(ttnn.to_memory_config(x,ttnn.DRAM_MEMORY_CONFIG),[1]*(4-len(shape))+shape)
        x=ttnn.experimental.all_gather_async(x,dim=3,persistent_output_buffer=None,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,topology=ttnn.Topology.Ring,memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=10,num_workers_per_link=2,num_buffers_per_channel=2)
        return ttnn.reshape(x,[*shape[:-1],shape[-1]*4])

    def _reduce(self,x):
        shape=list(x.shape)
        x=ttnn.reshape(ttnn.to_memory_config(x,ttnn.DRAM_MEMORY_CONFIG),[1]*(4-len(shape))+shape)
        x=ttnn.experimental.reduce_scatter_minimal_async(x,dim=3,persistent_output_buffers=None,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,topology=ttnn.Topology.Ring,memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=10,num_workers_per_link=2,num_buffers_per_channel=2)
        x=ttnn.reshape(x,[*shape[:-1],shape[-1]//4])
        return self._gather(x) if self.policy["residual_layout"] == "replicated" else x

    def _linear(self,x,w,activation=None,**kwargs):
        if x.shape[-1] == 1280 and self.weight_roles[id(w)] in ("mix","gate","up","gateup"):
            x=self._gather(x)
        y=super()._linear(x,w,activation,**kwargs)
        return self._reduce(y) if self.weight_roles[id(w)] in ("out","down") else y

    def _norm(self,x,w):
        if x.shape[-1] == 1280 and w.shape[-1] == 5120:
            shape=list(x.shape)
            x=ttnn.reshape(ttnn.to_memory_config(x,ttnn.DRAM_MEMORY_CONFIG),[1,*shape])
            stats=ttnn.rms_norm_pre_all_gather(x,compute_kernel_config=self.ckc,dtype=ttnn.bfloat16)
            stats=self._gather(stats)
            normalized=ttnn.rms_norm_post_all_gather(x,stats,epsilon=self.config.rms_norm_eps,
                weight=self.norm_shards[id(w)],compute_kernel_config=self.ckc)
            return ttnn.reshape(normalized,shape)
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
        if h.shape[-1] == 1280 and h.shape[1] > 1:
            h=self._gather(h)  # shared by both separate prefill projections
        compact = self.policy.get("packed_mlp") and h.shape[1] == 1 and h.shape[0] > 1 and not self.policy.get("minimal_mlp")
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
        if self.policy.get("prefill_l1"):
            del h
        mlp = self._linear(inter, self.down, decode_batch=mlp_decode_batch)
        if mlp.is_sharded():
            x = ttnn.to_memory_config(x, mlp.memory_config())
        return ttnn.add(x, mlp, memory_config=mlp.memory_config())

    def _delta(self, x, state):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
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
        if self.policy.get("history_tile_tail") and length >= 32 and length % 32 == 0:
            # Slice a complete tile before untilize, then select logical rows.
            # Direct nonaligned tiled slicing would add an implicit retile.
            tail = projected[:, length-32:length, :2560]
            next_conv = ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT)[:, -3:, :]
        else:
            next_conv = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)[:, length-3:length, :2560] if length >= 3 else ttnn.concat(
                [state.conv[:, length:, :], ttnn.to_layout(projected[..., :2560], ttnn.ROW_MAJOR_LAYOUT)], dim=1)
        ttnn.copy(next_conv, state.conv)

        def l2(t):
            t = ttnn.typecast(t, ttnn.float32)
            return ttnn.rms_norm(t, weight=self.l2_weight, epsilon=1e-6/128, compute_kernel_config=self.ckc)

        a = ttnn.typecast(projected[..., 4096:4108], ttnn.float32)
        decay = ttnn.multiply(self.neg_A, ttnn.softplus(ttnn.add(a, self.dt)))
        beta = ttnn.sigmoid(ttnn.typecast(projected[..., 4128:4140], ttnn.float32))
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
                    inplace_state=self.policy.get("inplace_recurrence", False),
                )
                ys.append(y_part)
                states.append(state_part)
            y = ttnn.concat(ys, dim=0) if len(ys) > 1 else ys[0]
            new_state = ttnn.concat(states, dim=0) if len(states) > 1 else states[0]
        if new_state.buffer_address() != state.recurrent.buffer_address():
            ttnn.copy(new_state, state.recurrent)
        y = self._norm(ttnn.typecast(y, ttnn.bfloat16), self.out_norm)
        z = reshape(projected[..., 2560:4096], [batch, length, 12, 128])
        return self._linear(reshape(ttnn.multiply(z, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), [batch, length, 1536]), self.out)

    def _attention(self, x, state, *, page_table, cos, sin, current_pos=None, chunk_start=0):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            projected[..., :2048], num_heads=6, num_kv_heads=1, transpose_key=False)
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
            y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, state.key, state.value, page_table_tensor=page_table, cur_pos_tensor=current_pos,
                scale=256**-0.5, compute_kernel_config=self.ckc,
                # The default is 32 tokens and one core/head: 2048 serial BF16
                # softmax-stat updates at full context. Bound the accumulation
                # chain with larger chunks and sequence-parallel cores.
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32, k_chunk_size=self.policy.get("sdpa_chunk", 256), max_cores_per_head_batch=self.policy.get("sdpa_cores", 16),
                    exp_approx_mode=False,
                ),
            )
            y = reshape(y, [batch, 1, 1536])
        else:
            fill_table = page_table[:, chunk_start // 32:]
            for user in range(batch):
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
                chunk_start_idx_tensor=self.chunk_positions[chunk_start // self.chunk_size],
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
