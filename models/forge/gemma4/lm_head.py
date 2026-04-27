"""Postlude: final RMSNorm + lm_head matmul + softcap."""
from gemma4.rms_norm import RMSNorm

import ttnn


class LMHead:
    """Tail of `Gemma4ForCausalLM.forward`: applies the last layer's
    layer_scalar, distributed RMSNorm, lm_head matmul, then softcap
    via tanh. Shared between prefill and decode. seq_len inferred
    from input shape so [1, 1, 65536] (decode) and [1, 19, 65536]
    (prefill) both work.

    Composes RMSNorm internally for the final norm step.
    """

    def __init__(self, last_layer_scalar, rms_eps, norm_weight, lm_head_weight, softcap):
        self.last_layer_scalar = last_layer_scalar
        self.rms_eps = rms_eps
        self.norm_weight = norm_weight
        self.lm_head_weight = lm_head_weight
        self.softcap = softcap
        self._final_norm = RMSNorm(norm_weight, rms_eps)

    def __call__(self, last_layer_residual):
        scaled = ttnn.multiply(
            last_layer_residual,
            self.last_layer_scalar,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(last_layer_residual, False)

        normed = self._final_norm(scaled)
        ttnn.deallocate(scaled, False)

        seq_len = normed.shape[-2]
        flat = ttnn.reshape(
            normed,
            [seq_len, 1344],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(normed, False)
        gathered_in = ttnn.all_gather(
            input_tensor=flat,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(flat, False)
        logits_local = ttnn.matmul(
            gathered_in,
            self.lm_head_weight,
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
        ttnn.deallocate(gathered_in, False)
        logits_3d = ttnn.reshape(
            logits_local,
            [1, seq_len, 65536],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(logits_local, False)
        gathered_out = ttnn.all_gather(
            input_tensor=logits_3d,
            dim=2,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(logits_3d, False)
        divided = ttnn.divide(
            gathered_out,
            self.softcap,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(gathered_out, False)
        tanhed = ttnn.tanh(
            divided,
            fast_and_approximate_mode=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(divided, False)
        capped = ttnn.multiply(
            tanhed,
            self.softcap,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(tanhed, False)
        return capped

    @classmethod
    def from_consteval(cls, cached_main, last_layer_scalar_ce, rms_eps, norm_weight_ce, lm_head_weight_ce, softcap):
        return cls(
            cached_main[f"main_const_eval_{last_layer_scalar_ce}"][0],
            rms_eps,
            cached_main[f"main_const_eval_{norm_weight_ce}"][0],
            cached_main[f"main_const_eval_{lm_head_weight_ce}"][0],
            softcap,
        )

    @classmethod
    def from_state_dict(
        cls, state_dict, mesh_device, *, rms_eps, last_layer_scalar, softcap, lm_head_dtype=None, norm_dtype=None
    ):
        """Build LMHead from HF state_dict + caller-supplied scalars.

        norm_weight comes from `model.language_model.norm.weight`
        (no +1 offset for gemma-4; verified against consteval).

        lm_head_weight comes from `model.language_model.embed_tokens.weight`
        (gemma-4 uses tied embeddings; tie_word_embeddings=True).
        Sharded along vocab dim (dim=0), distinct from the embed_tokens
        instance which is sharded along hidden (dim=1).

        last_layer_scalar and softcap are caller-supplied (built by
        the decoder layer / shared-scalars factories).
        """
        import torch
        from gemma4 import weights as gw

        if lm_head_dtype is None:
            lm_head_dtype = ttnn.DataType.BFLOAT8_B
        if norm_dtype is None:
            norm_dtype = ttnn.DataType.BFLOAT16

        torch_norm = state_dict["model.language_model.norm.weight"].to(torch.bfloat16)
        norm_weight = ttnn.as_tensor(
            torch_norm,
            dtype=norm_dtype,
            layout=ttnn.Layout.TILE,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            mesh_mapper=gw.mesh_mapper_for_role("norm", mesh_device),
        )

        torch_lm = state_dict["model.language_model.embed_tokens.weight"].to(torch.bfloat16)
        # Two-step load to match consteval's bf16→typecast pipeline.
        bf16_lm = ttnn.as_tensor(
            torch_lm,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            mesh_mapper=gw.mesh_mapper_for_role("lm_head", mesh_device),
        )
        if lm_head_dtype == ttnn.DataType.BFLOAT16:
            lm_head_weight = bf16_lm
        else:
            lm_head_weight = ttnn.typecast(
                bf16_lm,
                lm_head_dtype,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            )
            ttnn.deallocate(bf16_lm, False)

        return cls(
            last_layer_scalar=last_layer_scalar,
            rms_eps=rms_eps,
            norm_weight=norm_weight,
            lm_head_weight=lm_head_weight,
            softcap=softcap,
        )
