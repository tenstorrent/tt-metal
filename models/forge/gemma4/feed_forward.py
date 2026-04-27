"""Distributed gated-GELU FeedForward — shared between prefill and decode."""
import ttnn


class FeedForward:
    """Distributed gated-GELU FeedForward body shared between sliding
    and full decoder layers. Computes:
        gated = gelu(matmul(x, gate_proj_w.T)) * matmul(x, up_proj_w.T)
        out   = matmul(gated, down_proj_w.T)
    with mesh communication (all_gather before matmuls, reduce_scatter
    after down_proj).

    The seq_len dimension is inferred from x.shape[-2], so the same
    class handles prefill (seq_len=19) and decode (seq_len=1) without
    branching.

    Consumes (deallocates internally): `x`. Does NOT deallocate weights.
    """

    def __init__(self, gate_proj_w, up_proj_w, down_proj_w):
        self.gate_proj_w = gate_proj_w
        self.up_proj_w = up_proj_w
        self.down_proj_w = down_proj_w

    def __call__(self, x):
        seq_len = x.shape[-2]
        flat = ttnn.reshape(
            x,
            [seq_len, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(x, False)
        gathered = ttnn.all_gather(
            input_tensor=flat,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(flat, False)
        gate = ttnn.matmul(
            gathered,
            self.gate_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="gelu",
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        up = ttnn.matmul(
            gathered,
            self.up_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(gathered, False)
        gated = ttnn.multiply(
            gate,
            up,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(up, False)
        ttnn.deallocate(gate, False)
        down = ttnn.matmul(
            gated,
            self.down_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(gated, False)
        before_rs = ttnn.reshape(
            down,
            [1, 1, seq_len, 5376],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(down, False)
        scattered = ttnn.reduce_scatter(
            input_tensor=before_rs,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(before_rs, False)
        out = ttnn.reshape(
            scattered,
            [1, seq_len, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(scattered, False)
        return out

    @classmethod
    def from_consteval(cls, cached_main, gate_ce, up_ce, down_ce):
        return cls(
            cached_main[f"main_const_eval_{gate_ce}"][0],
            cached_main[f"main_const_eval_{up_ce}"][0],
            cached_main[f"main_const_eval_{down_ce}"][0],
        )

    @classmethod
    def from_state_dict(cls, state_dict, layer_idx, mesh_device, *, dtype=None):
        """Build FeedForward from HF state_dict.

        Default dtype is BFLOAT8_B (verified bit-equal against the
        consteval cache before consteval was retired in Phase 4).
        """

        if dtype is None:
            dtype = ttnn.DataType.BFLOAT8_B
        prefix = f"model.language_model.layers.{layer_idx}.mlp"
        gate = _load_proj(state_dict, f"{prefix}.gate_proj.weight", "gate_proj", mesh_device, dtype)
        up = _load_proj(state_dict, f"{prefix}.up_proj.weight", "up_proj", mesh_device, dtype)
        down = _load_proj(state_dict, f"{prefix}.down_proj.weight", "down_proj", mesh_device, dtype)
        return cls(gate, up, down)


def _load_proj(state_dict, hf_key, role, mesh_device, dtype):
    """Load a projection weight matching the consteval pipeline:
    bf16 → on-device → TILE layout → typecast to target dtype.

    Loading directly via ttnn.as_tensor(dtype=BFLOAT8_B) produces a
    DIFFERENT BFLOAT8_B byte pattern than the on-device typecast that
    consteval does, because the ttnn quantization path differs from
    the device-side typecast kernel.
    """
    import torch
    from gemma4 import weights as gw

    torch_w = state_dict[hf_key].to(torch.bfloat16)
    # Step 1: bf16 ttnn tensor on device, TILE layout (matches consteval).
    bf16_ttnn = ttnn.as_tensor(
        torch_w,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        mesh_mapper=gw.mesh_mapper_for_role(role, mesh_device),
    )
    if dtype == ttnn.DataType.BFLOAT16:
        return bf16_ttnn
    # Step 2: on-device typecast to target dtype (BFLOAT8_B).
    out = ttnn.typecast(
        bf16_ttnn,
        dtype,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(bf16_ttnn, False)
    return out
