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
        # Global rope: rotary_dim = int(global_head_dim * partial_rotary_factor) = 128
        # Local rope: head_dim = 256 (full rotation)
        DefaultRopeSetup = HfRotarySetup if self.args.use_hf_rope else RotarySetup
        ActualRopeSetupClass = rope_setup_class if rope_setup_class is not None else DefaultRopeSetup

        global_rotary_dim = int(args.global_head_dim * args.partial_rotary_factor)
        self.rope_setup = ActualRopeSetupClass(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=global_rotary_dim,  # 128 for partial rotation
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,  # 1000000
            rope_scaling=args.rope_scaling,
            use_qk_fused=False,
            prefetcher=prefetcher,
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

        # On-device sampling
        sampling_splits = self.args.num_devices if list(self.mesh_device.shape) != [1, 1] else 2
        self._supports_on_device_sampling = prefetcher is None and self.args.vocab_size // sampling_splits <= 64 * 1024
        if self._supports_on_device_sampling:
            self.sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl)
        else:
            self.sampling = None

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

            # Each layer gets its own KV cache (no sharing in initial bring-up)
            layer_kv = kv_cache[i] if kv_cache is not None else None

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
