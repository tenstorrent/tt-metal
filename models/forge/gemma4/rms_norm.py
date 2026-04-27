"""Distributed RMSNorm — shared between prefill and decode."""
import ttnn


class RMSNorm:
    """Distributed RMSNorm body shared by every in-decoder norm site:
    input_layernorm, post_attn_ln, pre_ff_ln, post_ff_ln in each decoder
    layer, plus the final norm in the LM head postlude.

    Computes y = x * rsqrt(mean(x*x) + eps) * weight where the mean
    reduction is sharded mean -> all_gather across mesh -> mean.

    Op sequence is bit-identical to the legacy `_rms_norm(x, weight, eps)`
    helper. Does not deallocate `x` or `weight`; caller owns lifetimes.
    """

    def __init__(self, weight, eps):
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        x_squared = ttnn.multiply(
            x,
            x,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        mean_local = ttnn.mean(
            x_squared,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(x_squared, False)
        mean_gathered = ttnn.all_gather(
            input_tensor=mean_local,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(mean_local, False)
        mean_global = ttnn.mean(
            mean_gathered,
            [2],
            True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(mean_gathered, False)
        rms = ttnn.add(
            mean_global,
            self.eps,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(mean_global, False)
        rsqrt = ttnn.rsqrt(
            rms,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(rms, False)
        normalized = ttnn.multiply(
            x,
            rsqrt,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(rsqrt, False)
        output = ttnn.multiply(
            normalized,
            self.weight,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(normalized, False)
        return output

    @classmethod
    def from_consteval(cls, cached_main, weight_ce_key, eps):
        """Bootstrap from an existing consteval cache entry. Used during
        Phase 1 migration; replaced by `from_state_dict` in Phase 2.
        """
        return cls(cached_main[f"main_const_eval_{weight_ce_key}"][0], eps)

    @classmethod
    def from_state_dict(cls, state_dict, hf_key, eps, mesh_device, *, dtype=None, role=None):
        """Build RMSNorm from an HF state_dict entry.

        NOTE: gemma-4's HF RMSNorm convention does NOT use the `1 + w`
        offset (verified empirically against the consteval cache; the
        cache stores `w_hf` directly). Pass-through.
        """
        import torch
        from gemma4 import weights as gw

        if dtype is None:
            dtype = ttnn.DataType.BFLOAT16
        if role is None:
            role = gw.role_for_hf_key(hf_key)
        torch_w = state_dict[hf_key].to(torch.bfloat16)
        ttnn_w = ttnn.as_tensor(
            torch_w,
            dtype=dtype,
            layout=ttnn.Layout.TILE,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED,
                ttnn.BufferType.DRAM,
                None,
            ),
            mesh_mapper=gw.mesh_mapper_for_role(role, mesh_device),
        )
        return cls(ttnn_w, eps)
