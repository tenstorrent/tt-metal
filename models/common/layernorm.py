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

        
        # --- DEBUG / SAFETY CHECK: make sure FINAL norm wires to norm.weight, not attention_norm ---
        is_final_norm = (weight_key == "norm") and (layer_num is None)
        
        # quick stats from host weights (safe even with tracing)
        import torch
        _w = state_dict[w_name].detach()
        _b = state_dict[b_name].detach()
        w_mean = float(_w.float().mean())
        w_min  = float(_w.float().min())
        w_max  = float(_w.float().max())
        b_mean = float(_b.float().mean())
        
        tag = "FINAL_NORM" if is_final_norm else f"LN({weight_key})"
        print(f"[LN-WIRE] {tag}")
        print(f"  w_name: {w_name}")
        print(f"  gamma: shape={tuple(_w.shape)} min/max/mean=({w_min:.6f}, {w_max:.6f}, {w_mean:.6f})")
        print(f"  beta : shape={tuple(_b.shape)} mean={b_mean:.6f}")
        
        # Only enforce the mean-threshold on the *final* norm.
        # attention_norm mean ~0.25 is normal for Phi-1, so don't assert there.
        if is_final_norm:
            assert w_mean > 0.8, (
                f"[FINAL_NORM] wrong gamma wired (mean={w_mean}). "
                f"Expected norm.weight, got something like attention_norm?"
            )
        # --- end debug ---
        
        
        # Reshape like RMSNorm does: [1,1,dim//32,32]
        # This tends to work well with TT kernels expecting tile-shaped last dim.
        torch_gamma = state_dict[w_name].view(1, 1, dim // 32, 32)
        torch_beta  = state_dict[b_name].view(1, 1, dim // 32, 32)
        
        
        print("[FINAL LN PARAM CHECK] gamma first4:", torch_gamma[:4].tolist())
        print("[FINAL LN PARAM CHECK] beta first4:", torch_beta[:4].tolist())

        # Put this right after you assign self.gamma/self.beta (or load them)
        def _fingerprint_ln(name, w, b):
            import torch
            def stats(t):
                t = t.detach().float().cpu()
                return float(t.min()), float(t.max()), float(t.mean())
            print(f"[LN-WIRE] {name}")
            print("  gamma:", tuple(w.shape), "min/max/mean:", stats(w))
            if b is not None:
                print("  beta :", tuple(b.shape), "min/max/mean:", stats(b))
            else:
                print("  beta : None")
        
        # Example usage after you load tensors:
        _fingerprint_ln("FINAL_NORM", torch_gamma, torch_beta)

        

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

        # If your build’s ttnn.layer_norm supports program_config/memory_config for sharded inputs,
        # this should work. Otherwise, you'll need to convert sharded->interleaved, run LN, then reshard.
        print("[LN] eps:", self.eps)
        print("[LN] x.shape:", x.shape, "x.dtype:", x.dtype, "x.layout:", x.layout)
        print("[LN] gamma.shape:", self.gamma.shape, "gamma.dtype:", self.gamma.dtype, "gamma.layout:", self.gamma.layout)
        print("[LN] beta.shape:", self.beta.shape, "beta.dtype:", self.beta.dtype, "beta.layout:", self.beta.layout)
        
        y = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.gamma,
            bias=self.beta,
            program_config=program_config,
            memory_config=memory_config,
        )
        try:
            import torch
        
            # Only for FINAL norm (avoid printing for attention_norm / ffn_norm)
            if getattr(self, "is_final_norm", False):
                # print only once per instance
                if not getattr(self, "_printed_once", False):
                    self._printed_once = True
        
                    dev_ts = ttnn.get_device_tensors(y)
                    h0 = ttnn.to_torch(dev_ts[0])   # [1,1,32,2048]
                    v0 = h0[0, 0, 16].float()
        
                    print("\n=== TT FINAL LN OUT dev0 @16 ===")
                    print("H:", v0.numel(), "norm:", float(torch.norm(v0)))
                    print("first8:", v0[:8].tolist())
        
                    # Check replication / consistency across devices (no concat!)
                    if len(dev_ts) > 1:
                        h1 = ttnn.to_torch(dev_ts[1])
                        v1 = h1[0, 0, 16].float()
                        diff = (v0 - v1).abs()
                        print("dev0-dev1 max_abs_diff:", float(diff.max()))
                        print("dev0-dev1 mean_abs_diff:", float(diff.mean()))
        
                    print("=== END TT FINAL LN OUT ===\n")
        
        except Exception as e:
            print("TT FINAL LN debug failed:", repr(e))
        
        


        # Match RMSNorm behavior: if we normalized sharded but caller wants interleaved, de-shard
        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(y)
        return y
