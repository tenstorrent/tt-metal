"""Hyperconnection weight precision and sharded normalization experiment."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedHyperConnection

class ShardedHyperConnection(OptimizedHyperConnection):
    weight_dtype=ttnn.bfloat16
    fidelity=ttnn.MathFidelity.HiFi4
    norm_cores=0
    keep_sharded=False
    adapt_flatten=False
    adapt_binary=False
    split_read=False
    fused_up=True
    dram_readers=0
    dram_cores=(20,2)
    dram_blocks=(16,5)

    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        obj=super().from_state_dict(*args,**kwargs)
        obj.mesh=kwargs['mesh_device']
        for name in ('read_packed','up'):
            w=getattr(obj,name)
            if w.dtype != cls.weight_dtype: setattr(obj,name,ttnn.typecast(w,cls.weight_dtype))
        obj.hc_compute=ttnn.init_device_compute_kernel_config(obj.mesh.arch(),math_fidelity=cls.fidelity,
            math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=False)
        if cls.dram_readers:
            from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import configure
            obj.device=obj.mesh
            obj.projection_compute=obj.hc_compute
            names=('read_packed','up');cores=cls.dram_cores;blocks=cls.dram_blocks
            if cls.split_read:
                obj.read_down=obj.read_packed[:,:320];obj.read_inject=obj.read_packed[:,320:]
                names=('read_down','read_inject','up');cores=(cores[0],cores[0],cores[1]);blocks=(blocks[0],blocks[0],blocks[1])
            configure(obj,names,cores=cores,blocks=blocks,
                readers=cls.dram_readers,activations={'up':ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)} if cls.fused_up else None)
        if cls.adapt_binary and cls.norm_cores:
            n=cls.norm_cores;gx=min(n,10);gy=n//gx
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(gx-1,gy-1))])
            mem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.L1,
                ttnn.ShardSpec(grid,[32,obj.width//n],ttnn.ShardOrientation.ROW_MAJOR))
            obj.norm_sharded=ttnn.to_memory_config(obj.norm_packed,mem)
        return obj

    def _project(self,x,w,activation=None):
        if self.split_read and w.buffer_address()==self.read_packed.buffer_address():
            return ttnn.concat([self._project(x,self.read_down),self._project(x,self.read_inject)],dim=-1)
        if self.dram_readers and x.shape[1]==1 and x.shape[2]==1:
            from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import linear
            y=linear(self,x,w)
            return ttnn.sigmoid(y) if activation and not self.fused_up else y
        return ttnn.linear(x,w,activation=activation,compute_kernel_config=self.hc_compute)

    def _norm_candidate(self,x):
        if not self.norm_cores:
            return ttnn.rms_norm(x,epsilon=self.eps,compute_kernel_config=self.compute)
        n=self.norm_cores; gx=min(n,10); gy=n//gx; k=self.width//32//n
        height=1
        for size in tuple(x.padded_shape)[:-1]: height*=size
        grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(gx-1,gy-1))])
        mem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.L1,
            ttnn.ShardSpec(grid,[height,self.width//n],ttnn.ShardOrientation.ROW_MAJOR))
        prog=ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[gx,gy],
            subblock_w=2 if k%2==0 else 1,block_h=height//32,block_w=k,inplace=False)
        y=ttnn.rms_norm(ttnn.to_memory_config(x,mem),epsilon=self.eps,compute_kernel_config=self.compute,
            program_config=prog,memory_config=mem)
        return y if self.keep_sharded and x.shape[1]==1 else ttnn.to_memory_config(y,ttnn.DRAM_MEMORY_CONFIG)

    def _read_prefill(self, x):
        streams = [
            ttnn.rms_norm(self._part(x, i), weight=self.norm[i], epsilon=self.eps, compute_kernel_config=self.compute)
            for i in range(self.count)
        ]
        normalized = ttnn.concat(streams, dim=-1)
        packed = self._project(normalized, self.read_packed)
        mix = ttnn.silu(packed[..., :320])
        mix = self._project(mix, self.up, activation="sigmoid")
        mixed = ttnn.multiply(self._part(mix, 0), streams[0])
        for i in range(1, self.count):
            mixed = ttnn.add(mixed, ttnn.multiply(self._part(mix, i), streams[i]))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return ttnn.multiply(mixed, 1 / self.count), inject

    def read(self, x):
        # Up to four tokens use no more padded rows than four split streams.
        # This geometry also preserves the short-tail routing accuracy.
        if x.shape[2] > 4:
            return self._read_prefill(x)
        b, seq = tuple(x.shape)[1:3]
        retain = self.keep_sharded and b*seq==1
        streams = ttnn.reshape(x, (1, b * seq, self.count, self.width))
        streams = ttnn.multiply(
            self._norm_candidate(streams), self.norm_sharded if self.adapt_binary and retain else self.norm_packed
        )
        flat_source=ttnn.to_memory_config(streams,ttnn.L1_MEMORY_CONFIG) if self.adapt_flatten and retain else streams
        normalized = ttnn.reshape(flat_source, (1, b, seq, self.count * self.width))
        packed = self._project(normalized, self.read_packed)
        mix = ttnn.silu(packed[..., :320])
        mix = self._project(mix, self.up, activation="sigmoid")
        mix = ttnn.reshape(mix, (1, b * seq, self.count, self.width))
        if self.adapt_binary and retain: mix=ttnn.to_memory_config(mix,streams.memory_config())
        mixed = ttnn.multiply(mix, streams)
        if self.adapt_binary and retain: mixed=ttnn.to_memory_config(mixed,ttnn.L1_MEMORY_CONFIG)
        mixed = ttnn.mean(mixed, dim=2, keepdim=True)
        mixed = ttnn.reshape(mixed, (1, b, seq, self.width))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return mixed, inject
