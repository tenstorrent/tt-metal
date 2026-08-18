# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native Cosmos3OmniTransformer trunk for I2V.

First-cut scope: the trunk owns the 64 native decoder layers + the two
final RMSNorms (`norm`, `norm_moe_gen`). Pre/post pieces (text
embedding, patchify+pack, timestep embedding, proj_in/proj_out, mRoPE
cos/sin generation) stay in the host-side pipeline wrapper for now —
they're cheap once-per-step ops. We'll fold them in here in later
commits as it matters.

Why a native trunk instead of the previous tt-symbiote-wrapped
`Cosmos3OmniTransformer`:
  - tt-symbiote shimming routes every `nn.Linear` through TT but leaves
    the rest of the trunk on host PyTorch. Activations ping-pong over
    the host↔device boundary per op, and `TTNNLinearMeshShard`'s output
    topology metadata interacts badly with `auto_compose` over many
    layers, producing NaN end-to-end (confirmed by the diagnostic run
    that returned 15360/15360 NaN latents).
  - Native trunk keeps activations on device for the full decoder
    stack. No boundary crossings inside the trunk → no NaN-class bugs
    from layout-metadata races.

Forward contract:
    forward(und_seq, gen_seq, cos_und, sin_und, cos_gen, sin_gen)
        -> (und_out, gen_out)
where the inputs are replicated `[1, 1, N, hidden_size]` TILE ttnn
tensors (matching the decoder-layer contract we validated for PCC),
the rotary cos/sin are replicated `[1, 1, N_*, head_dim]` TILE ttnn
tensors, and the outputs are replicated `[1, 1, N_*, hidden_size]`
ready for the host wrapper to concat, index, and run proj_out on.

PCC validation (Stage D): build the trunk + a truncated `max_layers`
config, load weights via `load_torch_state_dict` from the HF
checkpoint, compare against the reference `Cosmos3OmniTransformer.forward`
on a 1x1 LoudBox mesh.
"""

from __future__ import annotations

import os

import ttnn

from ....layers.linear import Linear
from ....layers.module import Module, ModuleList
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig, ParallelFactor
from .decoder_layer import Cosmos3VLTextMoTDecoderLayer


def _default_parallel_config() -> DiTParallelConfig:
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        tensor_parallel=ParallelFactor(1, 1),
        sequence_parallel=ParallelFactor(1, 0),
    )


class Cosmos3OmniTransformer(Module):
    """Native Cosmos3 trunk: 64 decoder layers + 2 final RMSNorms."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        num_hidden_layers: int,
        patch_latent_dim: int | None = None,
        enable_proj_in: bool = False,
        enable_proj_out: bool = False,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig | None = None,
        ccl_manager=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if parallel_config is None:
            parallel_config = _default_parallel_config()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        layer_kw = {
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "intermediate_size": intermediate_size,
            "attention_bias": attention_bias,
            "rms_norm_eps": rms_norm_eps,
            "mesh_device": mesh_device,
            "parallel_config": parallel_config,
            "ccl_manager": ccl_manager,
            "dtype": dtype,
        }
        self.layers = ModuleList(Cosmos3VLTextMoTDecoderLayer(**layer_kw) for _ in range(num_hidden_layers))

        norm_kw = {
            "norm_eps": rms_norm_eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "dtype": dtype,
        }
        self.norm = RMSNorm(hidden_size, **norm_kw)
        self.norm_moe_gen = RMSNorm(hidden_size, **norm_kw)

        # Vision proj_in/proj_out on device. Replicated weights (no TP) — both are
        # tiny (5120 x 192). proj_in shrinks the gen upload 26x; proj_out shrinks the
        # gen download 26x. Each is independently gated so callers can adopt them
        # incrementally (Phase A: proj_out only; Phase B: both).
        if (enable_proj_in or enable_proj_out) and patch_latent_dim is None:
            msg = "patch_latent_dim is required when enable_proj_in or enable_proj_out is True"
            raise ValueError(msg)
        # proj_in runs float32 + HiFi4: the trunk's normal bf16/HiFi2 precision leaves
        # RMSE≈0.0017 after proj_in, which amplifies ~61× over 64 layers → ~9% gen_out
        # error → visible noise.
        self.proj_in = (
            Linear(patch_latent_dim, hidden_size, bias=True, mesh_device=mesh_device, dtype=ttnn.float32)
            if enable_proj_in
            else None
        )
        self._proj_in_compute_kernel_config = (
            ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            if enable_proj_in
            else None
        )
        self.proj_out = (
            Linear(hidden_size, patch_latent_dim, bias=True, mesh_device=mesh_device, dtype=dtype)
            if enable_proj_out
            else None
        )

    def forward(
        self,
        und_seq: ttnn.Tensor,
        gen_seq: ttnn.Tensor,
        cos_und: ttnn.Tensor,
        sin_und: ttnn.Tensor,
        cos_gen: ttnn.Tensor,
        sin_gen: ttnn.Tensor,
        logical_n_gen: int | None = None,
        time_embed: ttnn.Tensor | None = None,
        noisy_mask_gen: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the full 64-layer decoder stack + final norms entirely on device.

        und tensors are replicated on the mesh. gen tensors are replicated at
        sp_factor=1; at sp_factor>1 they enter sp-sharded on `sp_axis` with
        sequence length padded to a multiple of `k_chunk_size * sp_factor`
        (the trunk wrapper handles padding + scattering). `logical_n_gen` is
        the unpadded N_gen, required by the ring SDPA op inside each layer.

        Returns gen_out replicated even at sp>1 — a final all-gather on
        sp_axis collapses the sequence-sharded gen back to the proxy's
        expected layout. The all-gather is a single CCL per generate step
        (not per layer), amortized across the 64-layer stack.
        """
        # When proj_in is on-device, gen_seq enters as raw patches [N_gen, patch_latent_dim].
        # Project to hidden, then add the per-step timestep embedding to noisy rows only.
        # noisy_mask_gen [N,1] @ time_embed [1,hidden] → [N,hidden] outer product: clean rows
        # multiply by 0, noisy rows by 1. Uses matmul (K=1) instead of ttnn.multiply because
        # ttnn.multiply's 2D broadcast ([1,1,1,H] × [1,1,N,1]) corrupts tile boundaries.
        if self.proj_in is not None:
            gen_seq_fp32 = ttnn.typecast(gen_seq, ttnn.float32)
            # default_block_size=4: float32 tiles are 2× larger than bf16, halving the block
            # size keeps circular buffers within BH's 1.5 MB L1 limit.
            gen_seq = self.proj_in(
                gen_seq_fp32,
                compute_kernel_config=self._proj_in_compute_kernel_config,
                default_block_size=(4, 4, 8),
            )
            ttnn.deallocate(gen_seq_fp32)
            gen_seq = ttnn.typecast(gen_seq, und_seq.dtype)
            if time_embed is not None and noisy_mask_gen is not None:
                gen_seq = ttnn.add(gen_seq, ttnn.matmul(noisy_mask_gen, time_embed))

        _profile_flush_every = int(os.environ.get("TT_COSMOS3_PROFILE_FLUSH_EVERY", "0"))

        for i, layer in enumerate(self.layers):
            und_seq, gen_seq = layer(und_seq, gen_seq, cos_und, sin_und, cos_gen, sin_gen, logical_n_gen=logical_n_gen)
            # drain device profiler ring mid-trunk to prevent overflow on long captures;
            # each drain adds a host sync — no-op in production (env unset → 0)
            if _profile_flush_every and (i + 1) % _profile_flush_every == 0:
                # sync before reading so all in-flight zone writes reach DRAM
                ttnn.synchronize_device(self.mesh_device)
                ttnn.ReadDeviceProfiler(self.mesh_device)

        und_out = self.norm(und_seq)
        gen_out = self.norm_moe_gen(gen_seq)

        if self.parallel_config.sequence_parallel.factor > 1:
            gen_out = self.ccl_manager.all_gather_persistent_buffer(
                gen_out, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )
        if self.proj_out is not None:
            gen_out = self.proj_out(gen_out)
        return und_out, gen_out
