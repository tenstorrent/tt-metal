"""L1 residual/working-shard family through both HC norms and residual writes."""
import argparse
import sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--norm-cores', type=int, default=5)
    p.add_argument('--validate', action='store_true')
    a, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    original_norm = target.MeshOps.norm
    original_write = target.ParallelHyperConnection.write
    original_read = target.ParallelHyperConnection.read

    def norm(self, value, weight, eps):
        if tuple(value.shape) != (1, 1, 4, 640):
            return original_norm(self, value, weight, eps)
        if a.norm_cores == 0:
            return original_norm(self, ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG), weight, eps)
        n = a.norm_cores
        grid = ttnn.CoreGrid(x=min(10,n), y=max(1,n//10))
        mem = ttnn.create_sharded_memory_config((32,640//n), grid,
                    ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True)
        subblock = min(4, 20 // n)
        while (20 // n) % subblock:
            subblock -= 1
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(grid.x,grid.y), subblock_w=subblock,
                    block_h=1, block_w=20//n, inplace=False)
        x = ttnn.to_memory_config(value, mem)
        stats = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.float32,
                    compute_kernel_config=self.compute, program_config=prog)
        stats = self.gather(ttnn.to_memory_config(stats, ttnn.DRAM_MEMORY_CONFIG))
        smem = ttnn.create_sharded_memory_config((32,128), ttnn.CoreGrid(x=1,y=1),
                    ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True)
        y = ttnn.rms_norm_post_all_gather(x, ttnn.to_memory_config(stats,smem), epsilon=eps,
                    compute_kernel_config=self.compute, program_config=prog, memory_config=mem)
        y = ttnn.multiply(y, ttnn.to_memory_config(weight,mem), memory_config=mem)
        # Flatten four logical streams for the existing packed HC matmul. The
        # interleaved L1 transition remains internal; residual ownership is unchanged.
        return ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)

    def read(self, x, **kwargs):
        if tuple(x.shape) == (1,1,1,2560):
            x=ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG)
        return original_read(self,x,**kwargs)

    def write(self,x,update,inject):
        if tuple(x.shape) != (1,1,1,2560):
            return original_write(self,x,update,inject)
        y=ttnn.multiply(ttnn.reshape(update,(1,1,1,640)),ttnn.reshape(inject,(1,1,4,1)),
                        memory_config=ttnn.L1_MEMORY_CONFIG)
        y=ttnn.add(ttnn.reshape(x,(1,1,4,640)),y,memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.reshape(y,x.shape)

    target.MeshOps.norm=norm
    target.ParallelHyperConnection.read=read
    target.ParallelHyperConnection.write=write
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
    bench.CANDIDATE={'norm_cores':a.norm_cores,'residual':'hidden_sharded_L1','restore':'none'}
    if a.validate:
        from models.autoports.qwen_qwen3_8_flash_next.tests import validate_multichip_decoder as validation
        import os
        from pathlib import Path
        tag=rest[rest.index('--tag')+1]
        directory=Path(__file__).resolve().parents[1]/os.environ.get('QWEN38_STAGE_DIR','doc/multichip_decoder')/tag
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'norm_candidate.py.txt').write_text(Path(__file__).read_text())
        validation.main()
    else:
        bench.main()

if __name__=='__main__': main()
