# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 Transformer Model.

Extends the base Transformer with:
- Gemma4TransformerBlock (hybrid attention, layer scalar)
- Dual RoPE setup (sliding vs full attention layers)
- Logit softcapping after LM head
"""

import math

import torch
from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.sampling.generator import SamplingGenerator
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding, ScaledEmbedding
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.rope import RotarySetup

from .gemma4_decoder import Gemma4TransformerBlock


class Gemma4Transformer(Transformer):
    """
    Gemma 4 Transformer extending the base Transformer.

    Key differences from base:
    - Uses Gemma4TransformerBlock with per-layer hybrid attention config
    - Dual RoPE: sliding (theta=10K, head_dim=256) and full (theta=1M)
    - Logit softcapping: tanh(logits / cap) * cap
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        rope_setup_class=None,
        prefetcher=None,
    ):
        # We override __init__ to use Gemma4TransformerBlock
        # Call grandparent __init__ (LightweightModule) to skip base Transformer __init__
        LightweightModule.__init__(self)

        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        state_dict_prefix = args.get_state_dict_prefix("", None)
        self.decoders_optimizations = args.decoders_optimizations
        self.prefetcher = prefetcher
        self.tt_ccl = TT_CCL(self.mesh_device)

        # Embedding (scaled by sqrt(hidden_dim) for Gemma)
        embd_kwargs = {
            "mesh_device": mesh_device,
            "args": args,
            "weight_cache_path": args.weight_cache_path(dtype),
            "state_dict": state_dict,
            "dtype": ttnn.bfloat16,
        }
        if self.args.embed_scale is not None:
            embd_cls = ScaledEmbedding
            embd_kwargs["embed_scale"] = self.args.embed_scale
        else:
            embd_cls = Embedding
        self.embd = embd_cls(**embd_kwargs)

        # Dual RoPE setup
        # Global RoPE (for full attention layers): theta=1M, uses sliding head_dim=256 for now
        # TODO: Implement partial rotary (128 dims) for correct full attention RoPE
        ActualRopeSetupClass = rope_setup_class if rope_setup_class is not None else RotarySetup

        # Full attention RoPE (global) - HF-format cos/sin for partial rotation
        # partial_rotary_factor=0.25, global_head_dim=512 → rotary_dim=128
        global_rope_theta = getattr(args, "rope_theta_global", 1000000.0)
        partial_rotary_factor = getattr(args, "partial_rotary_factor", 0.25)
        global_rotary_dim = int(args.global_head_dim * partial_rotary_factor)  # 128

        # Create Meta-style RotarySetup for transformation matrices
        self.rope_setup = ActualRopeSetupClass(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=global_rotary_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=global_rope_theta,
            use_qk_fused=args.use_qk_fused,
            prefetcher=prefetcher,
        )

        # Sliding attention RoPE (local) - theta=10K, head_dim=256
        self.rope_local_setup = ActualRopeSetupClass(
            mesh_device,
            args.max_batch_size,
            args.head_dim,  # 256
            args.max_seq_len,
            args.rope_theta_local if args.rope_theta_local else 10000.0,
            use_qk_fused=args.use_qk_fused,
            prefetcher=None,
        )

        # Transformation matrices from LOCAL rope (head_dim=256) for sliding layers
        self.trans_mats_dict = self.rope_local_setup.get_both_trans_mats()

        # For full attention partial RoPE: convert Meta-format cos/sin to HF-format
        # Meta: [cos_f0, cos_f0, cos_f1, cos_f1, ...] (pairs)
        # HF:   [cos_f0, cos_f1, ..., cos_f63, cos_f0, cos_f1, ..., cos_f63] (halves)
        self._create_hf_format_global_rope(mesh_device, global_rotary_dim)

        # Gemma4 decoder layers
        self.layers = [
            Gemma4TransformerBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
                transformation_mats_global=None,  # Full attn uses HF-format cos/sin directly
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                attention_class=attention_class,
                prefetcher=prefetcher,
            )
            for i in tqdm(range(self.n_layers), desc="Loading Gemma4 layers")
        ]

        # Final norm
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                is_distributed=self.args.is_distributed_norm,
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
        )

        # Override final norm to HiFi4 (Gemma4-specific, same as decoder norms)
        self.norm.norm.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # LM Head - use BF16 weights to prevent BFP8 quantization loss
        # on the massive 5376→262K projection (11% PCC drop with BFP8)
        lm_head_weight_dtype = ttnn.bfloat16
        lm_head_cache_path = args.weight_cache_path(lm_head_weight_dtype)
        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            dtype=lm_head_weight_dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=lm_head_cache_path,
            max_columns_per_device=self.args.max_columns_per_device_lm_head,
            prefetcher=prefetcher,
        )

        # Override LM head compute kernel: default HiFi2 with fp32_dest_acc_en=False
        # loses precision on the massive 5376→262K projection (1.7% PCC drop).
        self.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # On-device sampling
        sampling_splits = self.args.num_devices if list(self.mesh_device.shape) != [1, 1] else 2
        self._supports_on_device_sampling = (
            prefetcher is None and self.args.vocab_size // sampling_splits <= 64 * 1024
        )
        if self._supports_on_device_sampling:
            self.sampling = SamplingGenerator(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
            )
        else:
            self.sampling = None

        # Logit softcapping value
        self.final_logit_softcapping = getattr(args, "final_logit_softcapping", None)

    def _create_hf_format_global_rope(self, mesh_device, rotary_dim):
        """Create HF-format cos/sin matrices for full attention partial RoPE.

        Converts Meta-format (interleaved pairs) to HF-format (duplicated halves)
        so that ttnn.experimental.rotary_embedding works correctly with unpermuted Q/K.
        """
        # Get Meta-format cos/sin from RotarySetup
        cos_meta_dev = self.rope_setup.cos_matrix_prefill  # [1, 1, max_seq, rotary_dim]
        sin_meta_dev = self.rope_setup.sin_matrix_prefill

        # Convert to torch, reshape, and convert format
        cos_meta = ttnn.to_torch(ttnn.get_device_tensors(cos_meta_dev)[0])  # [1, 1, seq, dim]
        sin_meta = ttnn.to_torch(ttnn.get_device_tensors(sin_meta_dev)[0])

        # Meta format: [cos_f0, cos_f0, cos_f1, cos_f1, ...] → extract unique: [cos_f0, cos_f1, ...]
        cos_unique = cos_meta[:, :, :, 0::2]  # Take every other element: [1, 1, seq, dim/2]
        sin_unique = sin_meta[:, :, :, 0::2]

        # HF format: [cos_f0, cos_f1, ..., cos_f(n/2-1), cos_f0, cos_f1, ..., cos_f(n/2-1)]
        cos_hf = torch.cat([cos_unique, cos_unique], dim=-1)  # [1, 1, seq, dim]
        sin_hf = torch.cat([sin_unique, sin_unique], dim=-1)

        # Store on device
        self.cos_global_hf = ttnn.from_torch(
            cos_hf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_global_hf = ttnn.from_torch(
            sin_hf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def prepare_inputs_prefill(self, tokens, start_pos=0, **kwargs):
        """Override to use HF-format cos/sin for global (full attention) RoPE."""
        result = super().prepare_inputs_prefill(tokens, start_pos=start_pos, **kwargs)
        tokens_or_embd, rot_mats_global, rot_mats_local, tt_page_table, tt_chunk_page_table = result

        # Replace global rot_mats with HF-format cos/sin
        trace_enabled = kwargs.get("trace_enabled", False)
        S = tokens.shape[-1] if tokens.dim() == 2 else tokens.shape[-1]
        seq_len = kwargs.get("last_token_idx", S - 1) + 1 if kwargs.get("last_token_idx") is not None else S
        prefill_start_pos = 0 if trace_enabled else start_pos
        slice_end = self.args.max_seq_len if trace_enabled else min(self.cos_global_hf.shape[2], start_pos + S)

        cos_global = self.cos_global_hf[:, :, prefill_start_pos:slice_end, :]
        sin_global = self.sin_global_hf[:, :, prefill_start_pos:slice_end, :]
        rot_mats_global_hf = [cos_global, sin_global]

        return tokens_or_embd, rot_mats_global_hf, rot_mats_local, tt_page_table, tt_chunk_page_table

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode: Mode = Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    ):
        if mode == Mode.DECODE:
            if self.prefetcher is not None:
                self.prefetcher.run()

        for i, layer in enumerate(self.layers):
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )

            if mode == Mode.DECODE and not self.args.is_galaxy:
                if self.prefetcher is not None:
                    x = ttnn.to_memory_config(
                        x,
                        self.args.get_residual_mem_config(mode, self.prefetcher),
                        activation_dtype,
                    )
                elif activation_dtype is not None and x.dtype != activation_dtype:
                    x = ttnn.typecast(x, activation_dtype)
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                batch_size=batch_size,
            )

        if mode == Mode.DECODE:
            if self.prefetcher is not None:
                self.prefetcher.stop()

        if mode == Mode.PREFILL and get_last_token == -1:
            return x

        # Slice to get last token for prefill
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        x = self.norm(x, mode=mode, norm_config=self.args.get_norm_config("lm_head", mode, self.prefetcher))

        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(
            mode, None if mode == Mode.PREFILL else self.prefetcher
        )
        if mode == Mode.PREFILL and lm_head_input_mem_cfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)
        if mode == Mode.DECODE and self.prefetcher is not None:
            x = ttnn.to_memory_config(x, self.args.get_lm_head_input_mem_config(mode, self.prefetcher))

        x = self.lm_head(x)

        # Apply logit softcapping: tanh(logits / cap) * cap
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            x = ttnn.multiply(x, 1.0 / cap)
            x = ttnn.tanh(x)
            x = ttnn.multiply(x, cap)

        if mode == Mode.PREFILL:
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return x
