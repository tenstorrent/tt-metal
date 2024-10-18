import ttnn
from models.common.lightweightmodule import LightweightModule


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args):
        self.norm = norm
        self.args = args

    def forward(self, x, mode):
        """Apply a norm, possibly gathering inputs if required."""
        input_mem_cfg = self.norm.sharded_output_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology(), memory_config=input_mem_cfg)
        elif mode == "decode":
            # Gathered norms will be sharded for decode mode, so single-chip should be too
            x = ttnn.interleaved_to_sharded(x, input_mem_cfg)

        # x sharded in decode mode here
        x = self.norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))

        # Distributed norm already performs a gather
        if self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology())

        return x
