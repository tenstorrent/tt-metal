"""Explicit minimal_matmul candidate on material prefill projections."""
import math
import ttnn

def install(*,cast=False,blocks=(4,4,8)):
    from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder as mod
    for cls in (mod.OptimizedGatedDeltaNet,mod.OptimizedQSA):
        original=cls._linear
        def linear(self,x,w,_original=original):
            if x.shape[-2]<32 or math.prod(tuple(x.shape)[:-2])!=1:return _original(self,x,w)
            config=ttnn.MinimalMatmulConfig(M_block_size=blocks[0],K_block_size=blocks[1],N_block_size=blocks[2],
                subblock_h=1,subblock_w=min(4,blocks[2]),compute_with_storage_grid_size=ttnn.CoreCoord(11,10))
            xx=ttnn.typecast(x,w.dtype) if cast and x.dtype!=w.dtype else x
            return ttnn.experimental.minimal_matmul(xx,w,config=config,dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=self.projection_compute)
        cls._linear=linear
