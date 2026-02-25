# layernorm.py
import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
SHARD_HEIGHT = TILE  # keep same convention as RMSNorm

class LayerNorm(LightweightModule):
    """
    LayerNorm that matches RMSNorm's call signature so it can be wrapped by DistributedNorm.

    Required by DistributedNorm:
      - forward(x, mode, in_sharded=False, out_sharded=False)
      - attributes: eps, sharded_program_config, sharded_output_config
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        eps: float = 1e-5,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        # keep args for parity / future use; LN distributed path not implemented here
        is_distributed=None,
        ccl_topology=ttnn.Topology.Ring,
        tt_ccl=None,
    ):
        super().__init__()
        self.device = device
        self.eps = 1e-5 if eps is None else float(eps)

        # Match RMSNorm attribute names so DistributedNorm can read them
        self.sharded_program_config = sharded_program_config
        self.sharded_output_config = sharded_output_config
        self.output_mem_config = output_mem_config

        # Keep these fields for API parity (even if unused)
        self.is_distributed = is_distributed
        self.ccl_topology = ccl_topology
        self.tt_ccl = tt_ccl
        self.is_final_norm = (weight_key == "norm") and (layer_num is None)


        # Resolve parameter names
        if state_dict_prefix:
            w_name = f"{state_dict_prefix}{weight_key}.weight"
            b_name = f"{state_dict_prefix}{weight_key}.bias"
        else:
            if layer_num is None:
                w_name = f"{weight_key}.weight"
                b_name = f"{weight_key}.bias"
            else:
                w_name = f"layers.{layer_num}.{weight_key}.weight"
                b_name = f"layers.{layer_num}.{weight_key}.bias"

        
        # Reshape like RMSNorm does: [1,1,dim//32,32]
        # This tends to work well with TT kernels expecting tile-shaped last dim.
        torch_gamma = state_dict[w_name].view(1, 1, dim // 32, 32)
        torch_beta  = state_dict[b_name].view(1, 1, dim // 32, 32)
        
        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        self.gamma = ttnn.as_tensor(
            torch_gamma,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=None if weight_cache_path is None else weight_cache_path / w_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )
        self.beta = ttnn.as_tensor(
            torch_beta,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=None if weight_cache_path is None else weight_cache_path / b_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )
        

    def forward(self, x: ttnn.Tensor, mode, in_sharded=False, out_sharded=False) -> ttnn.Tensor:
        """
        Must match RMSNorm's signature because DistributedNorm calls us like:
          norm(x, mode=mode, in_sharded=..., out_sharded=...)  :contentReference[oaicite:5]{index=5}
        """

        # Mirror RMSNorm logic: use sharded configs only when input is sharded
        program_config = self.sharded_program_config if in_sharded else None
        memory_config = self.sharded_output_config if out_sharded else None

        # Same constraints RMSNorm enforces:
        # - If input isn't sharded, we shouldn't be asked to output sharded (unless you explicitly support that)
        if not in_sharded:
            assert not out_sharded, "Non-sharded LayerNorm cannot output a sharded tensor"
        
        y = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.gamma,
            bias=self.beta,
            program_config=program_config,
            memory_config=memory_config,
        )
        
        
        # Match RMSNorm behavior: if we normalized sharded but caller wants interleaved, de-shard
        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(y)
        return y
