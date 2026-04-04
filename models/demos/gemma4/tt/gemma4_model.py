# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 E4B Text Model

Extends the base Transformer with Gemma 4 specific features:
- Dual head_dim and partial rotary via custom Gemma4Attention
- V-norm on value projections
- Per-layer input gating
- Layer scalar per decoder block
- Final logit soft-capping: tanh(logits / 30) * 30

Note: KV cache sharing is NOT implemented in this initial version.
All 42 layers compute their own QKV independently.
"""

import torch
from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.sampling.generator import SamplingGenerator
from models.demos.gemma4.tt.gemma4_decoder import Gemma4Decoder
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding, ScaledEmbedding
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup


class TtGemma4TextModel(Transformer):
    """
    Gemma 4 E4B text model with dual head_dim, partial rotary, V-norm,
    per-layer gating, logit soft-capping.
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
        # Skip Transformer.__init__ — we override everything
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
        self.final_logit_softcapping = getattr(args, "final_logit_softcapping", None)

        # Embedding (scaled by sqrt(dim) for Gemma family)
        embd_kwargs = {
            "mesh_device": mesh_device,
            "args": args,
            "weight_cache_path": args.weight_cache_path(dtype),
            "state_dict": state_dict,
            "dtype": ttnn.bfloat16,
        }
        if self.args.embed_scale is not None:
            self.embd = ScaledEmbedding(**embd_kwargs, embed_scale=self.args.embed_scale)
        else:
            self.embd = Embedding(**embd_kwargs)

        # RoPE setup with dual head_dim:
        # Global rope: proportional RoPE with partial_rotary_factor=0.25
        #   - rotary_dim = int(global_head_dim * partial_rotary_factor) = 128
        #   - But inv_freq uses denominator = global_head_dim (512), not rotary_dim (128)
        # Local rope: head_dim = 256 (full rotation, standard frequencies)
        DefaultRopeSetup = HfRotarySetup if self.args.use_hf_rope else RotarySetup
        ActualRopeSetupClass = rope_setup_class if rope_setup_class is not None else DefaultRopeSetup

        global_rotary_dim = int(args.global_head_dim * args.partial_rotary_factor)
        self.rope_setup = self._create_proportional_rope_setup(
            device=mesh_device,
            global_head_dim=args.global_head_dim,  # 512
            partial_rotary_factor=args.partial_rotary_factor,  # 0.25
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,  # 1000000
        )

        if args.rope_theta_local:
            self.rope_local_setup = DefaultRopeSetup(
                mesh_device,
                args.max_batch_size,
                args.head_dim,  # 256 full rotation
                args.max_seq_len,
                args.rope_theta_local,  # 10000
                use_qk_fused=False,
                prefetcher=None,
            )

        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        # Build Gemma4 decoder layers
        self.layers = [
            Gemma4Decoder(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                prefetcher=prefetcher,
            )
            for i in tqdm(range(self.n_layers), desc="Building Gemma4 layers")
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

        # LM Head
        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            max_columns_per_device=self.args.max_columns_per_device_lm_head,
            prefetcher=prefetcher,
        )
        # Override LM head compute kernel for higher accuracy during bring-up
        # Default is HiFi2/no-fp32-acc; use HiFi4 for better precision on logits
        self.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # On-device sampling
        sampling_splits = self.args.num_devices if list(self.mesh_device.shape) != [1, 1] else 2
        self._supports_on_device_sampling = prefetcher is None and self.args.vocab_size // sampling_splits <= 64 * 1024
        if self._supports_on_device_sampling:
            self.sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl)
        else:
            self.sampling = None

    def _create_proportional_rope_setup(self, device, global_head_dim, partial_rotary_factor, max_seq_len, rope_theta):
        """Create RoPE setup with proportional frequency scaling for global attention.

        HF Gemma4's proportional RoPE for global layers (head_dim=512):
        1. Computes 64 real inv_freq values + 192 zeros = 256 total inv_freq
        2. freqs = outer(positions, inv_freq) → [seq_len, 256]
        3. cos/sin duplicated: [cos, cos] → [seq_len, 512]
        4. rotate_half pairs dim[i] with dim[i+256] (half of 512)

        We generate full 512-dim cos/sin so ttnn.rotary_embedding's rotate_half
        (which pairs dim[i] with dim[head_dim/2+i]) matches HF exactly.
        """
        rope_angles = int(partial_rotary_factor * global_head_dim // 2)  # 64
        nope_angles = global_head_dim // 2 - rope_angles  # 192

        # 64 real frequencies + 192 zeros = 256 inv_freq (matching HF's _compute_proportional_rope_parameters)
        inv_freq_rotated = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_angles, 2).float() / global_head_dim))
        inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)], dim=0)  # [256]

        # Generate cos/sin for all positions: [max_seq_len, 256]
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, 256]

        # HF format: duplicate [cos_0..cos_255, cos_0..cos_255] → [seq_len, 512]
        cos_hf = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        sin_hf = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

        # Add batch dimensions: [1, 1, seq_len, rotary_dim]
        cos_4d = cos_hf.unsqueeze(0).unsqueeze(0)
        sin_4d = sin_hf.unsqueeze(0).unsqueeze(0)

        mapper = ttnn.ReplicateTensorToMesh(device)
        cos_matrix = ttnn.from_torch(
            cos_4d, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
        )
        sin_matrix = ttnn.from_torch(
            sin_4d, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
        )

        # Create a lightweight wrapper matching HfRotarySetup interface
        class ProportionalRopeSetup:
            def __init__(self, cos_mat, sin_mat):
                self.cos_matrix_prefill = cos_mat
                self.sin_matrix_prefill = sin_mat
                # For decode, use the same matrices (will be indexed by position)
                self.cos_matrix = cos_mat
                self.sin_matrix = sin_mat

            def get_both_trans_mats(self):
                return {}

            def get_rot_mats(self, rot_mat_idxs):
                """Get rotation matrices for decode mode."""
                cos_gathered = ttnn.embedding(rot_mat_idxs, self.cos_matrix_prefill[0, 0], layout=ttnn.TILE_LAYOUT)
                sin_gathered = ttnn.embedding(rot_mat_idxs, self.sin_matrix_prefill[0, 0], layout=ttnn.TILE_LAYOUT)
                return [cos_gathered, sin_gathered]

        return ProportionalRopeSetup(cos_matrix, sin_matrix)

    def _apply_logit_softcapping(self, logits):
        """Apply logit soft-capping: tanh(logits / cap) * cap"""
        if self.final_logit_softcapping is None:
            return logits
        cap = self.final_logit_softcapping
        logits = ttnn.multiply(logits, 1.0 / cap)
        logits = ttnn.tanh(logits)
        logits = ttnn.multiply(logits, cap)
        return logits

    def _apply_norm_and_lm_head(self, x):
        """Override to add logit soft-capping after LM head."""
        logits = super()._apply_norm_and_lm_head(x)
        logits = self._apply_logit_softcapping(logits)
        return logits

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
        if mode == Mode.DECODE and self.prefetcher is not None:
            self.prefetcher.run()

        # KV sharing: track source layer K/V for shared layers
        # kv_sharing_map: {shared_layer_idx: source_layer_idx}
        shared_kv_store = {}  # source_layer_idx -> (k_heads, v_heads)

        # Determine which layers are KV sources (last non-shared layer of each type)
        kv_source_layers = set()
        if hasattr(self.args, "kv_sharing_map"):
            for src_idx in self.args.kv_sharing_map.values():
                kv_source_layers.add(src_idx)

        for i, layer in enumerate(self.layers):
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )

            if mode == Mode.DECODE and not self.args.is_galaxy:
                x = ttnn.to_memory_config(
                    x,
                    self.args.get_residual_mem_config(mode, self.prefetcher),
                    activation_dtype,
                )
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            layer_kv = kv_cache[i] if kv_cache is not None else None

            # Set shared K/V for KV-shared layers (prefill only)
            if mode == Mode.PREFILL and hasattr(self.args, "kv_sharing_map") and i in self.args.kv_sharing_map:
                src_idx = self.args.kv_sharing_map[i]
                if src_idx in shared_kv_store:
                    layer.attention.shared_kv = shared_kv_store[src_idx]

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
                kv_cache=layer_kv,
                batch_size=batch_size,
            )

            # Store K/V from source layers for sharing
            if mode == Mode.PREFILL and i in kv_source_layers:
                shared_kv_store[i] = (layer.attention.last_k_heads, layer.attention.last_v_heads)

        if mode == Mode.DECODE and self.prefetcher is not None:
            self.prefetcher.stop()

        if mode == Mode.PREFILL and get_last_token == -1:
            return x

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

        # Logit soft-capping
        x = self._apply_logit_softcapping(x)

        if mode == Mode.PREFILL:
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return x
