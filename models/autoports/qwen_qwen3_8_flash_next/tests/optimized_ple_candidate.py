"""PLE real-weight precision, DRAM geometry, separation, and sharded norm candidates."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import ngram_indices,lookup_rows,_conv_taps,_mask_padding
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedPLE

class TunedPLE(OptimizedPLE):
    weight_dtype=ttnn.bfloat16
    fidelity=ttnn.MathFidelity.HiFi4
    readers=0
    cores=5
    block=4
    split=False
    norm_cores=0
    keep_sharded=False
    adapt_l1=False
    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        obj=super().from_state_dict(*args,**kwargs)
        obj.device=kwargs['mesh_device']
        obj.packed_projection=ttnn.typecast(obj.packed_projection,cls.weight_dtype)
        obj.projection_compute=ttnn.init_device_compute_kernel_config(obj.device.arch(),math_fidelity=cls.fidelity,
            math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
        names=['packed_projection']
        if cls.split:
            obj.key=obj.packed_projection[:,:10240];obj.value=obj.packed_projection[:,10240:]
            names=['key','value']
        if cls.readers:
            from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import configure
            configure(obj,names,cores=[cls.cores]*len(names),blocks=[cls.block]*len(names),readers=cls.readers)
        return obj
    def _project(self,x):
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import linear
        def mm(w):
            if self.readers: return linear(self,x,w)
            return ttnn.linear(x,w,compute_kernel_config=self.projection_compute)
        if self.split: return ttnn.concat([mm(self.key),mm(self.value)],dim=-1)
        return mm(self.packed_projection)

    def forward(self, x, *, tokens, positions, slots, valid_lengths, cache):
        batch, seq = tokens.shape
        if tuple(x.shape) != (1, batch, seq, 10240):
            raise ValueError("PLE x must have shape [1,B,S,10240]")
        ws = self.workspaces[(batch, seq)]
        ngram_indices(
            tokens,
            positions,
            slots,
            cache.token_history,
            valid_lengths=valid_lengths,
            output=ws.indices,
            multipliers=self.multipliers,
            primes=self.primes,
            offsets=self.offsets,
            eos=self.eos,
        )
        lookup_rows(self.tables, ws.indices, output=ws.embeddings)
        n = batch * seq
        emb = ttnn.reshape(ttnn.to_layout(ws.embeddings, ttnn.TILE_LAYOUT), (1, 1, n, 2560))
        projections = self._project(emb)
        key = self._norm(projections[..., :10240], self.norm_key, n)
        query = self._norm(x, self.norm_query, n)
        # Gate definition: sign(g) * sqrt(max(abs(g), 1e-6)).
        # The fused unary chain removes intermediate BF16 packs; matched
        # correctness checks qualify this rounding change.
        dot = ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True)
        gate = ttnn.multiply(dot, 2560**-0.5)
        gate = ttnn.multiply(
            gate,
            gate,
            input_tensor_a_activations=[
                ttnn.UnaryOpType.ABS,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 1e-6),
                ttnn.UnaryOpType.SQRT,
            ],
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGN],
        )
        value = projections[..., 10240:]
        value = ttnn.reshape(value, (1, n, 1, 2560))
        gated = ttnn.reshape(
            ttnn.multiply(gate, value, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]), (1, 1, n, 10240)
        )
        normalized = self._norm(gated, self.norm_conv, n)
        normalized = ttnn.to_layout(ttnn.reshape(normalized, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        # The gather zeros padding, resets only position-zero requests, and
        # updates all nine historical vectors after reading all old taps.
        _conv_taps(
            normalized, positions, slots, valid_lengths, cache.conv_history, output=ws.taps, batch=batch, seq=seq
        )
        taps = ttnn.to_layout(ws.taps, ttnn.TILE_LAYOUT)
        convolved = ttnn.sum(ttnn.multiply(ttnn.typecast(taps, ttnn.float32), self.conv), dim=0, keepdim=True)
        convolved = ttnn.typecast(convolved, ttnn.bfloat16)
        # Mask gated values via the same device controls; tap index 3 is the
        # normalized current token, so cannot be used as a numerical mask.
        gated_rm = ttnn.to_layout(ttnn.reshape(gated, (n, 10240)), ttnn.ROW_MAJOR_LAYOUT)
        _mask_padding(gated_rm, valid_lengths, batch=batch, seq=seq)
        result = ttnn.add(
            ttnn.to_layout(gated_rm, ttnn.TILE_LAYOUT),
            ttnn.reshape(convolved, (n, 10240)),
            input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
        )
        return ttnn.reshape(result, (1, batch, seq, 10240))

    def _norm(self, x, weight, n):
        x = ttnn.typecast(ttnn.reshape(x, (1, n, 4, 2560)), ttnn.float32)
        if not self.norm_cores or n!=1:
            normed = ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=self.compute)
            return ttnn.multiply(normed, weight, dtype=ttnn.bfloat16)
        count=self.norm_cores;gx=min(10,count);gy=count//gx;kw=80//count
        grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(gx-1,gy-1))])
        mem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.L1,
            ttnn.ShardSpec(grid,[32,2560//count],ttnn.ShardOrientation.ROW_MAJOR))
        prog=ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[gx,gy],
            subblock_w=2 if kw%2==0 else 1,block_h=1,block_w=kw,inplace=False)
        normed=ttnn.rms_norm(ttnn.to_memory_config(x,mem),epsilon=self.eps,program_config=prog,
            compute_kernel_config=self.compute,memory_config=mem)
        normed=ttnn.multiply(normed,weight,dtype=ttnn.bfloat16,memory_config=mem)
        if self.adapt_l1: return ttnn.to_memory_config(normed,ttnn.L1_MEMORY_CONFIG)
        return normed if self.keep_sharded else ttnn.to_memory_config(normed,ttnn.DRAM_MEMORY_CONFIG)
