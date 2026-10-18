"""2D program-config experiment on the actual packed attention projections."""
import math
import ttnn


def install(grid_x=10, k_block=8, subblock=4):
    from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder as mod
    for cls in (mod.OptimizedGatedDeltaNet,mod.OptimizedQSA):
        old=cls._linear
        def linear(self,x,w,_old=old):
            if x.shape[-2]<32 or math.prod(tuple(x.shape)[:-2])!=1:
                return _old(self,x,w)
            m=math.ceil(x.shape[-2]/32); k=w.shape[-2]//32; n=math.ceil(w.shape[-1]/32)
            gy=min(m,10); gx=min(grid_x,n)
            pm=math.ceil(m/gy); pn=math.ceil(n/gx)
            block=k_block
            while k%block: block-=1
            sw=min(subblock,pn)
            while pn%sw: sw-=1
            pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx,gy),
                in0_block_w=block,out_subblock_h=1,out_subblock_w=sw,
                per_core_M=pm,per_core_N=pn,transpose_mcast=False,fused_activation=None)
            return ttnn.linear(x,w,program_config=pc,compute_kernel_config=self.projection_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cls._linear=linear
