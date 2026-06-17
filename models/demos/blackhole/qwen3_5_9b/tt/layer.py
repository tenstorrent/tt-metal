# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hybrid Qwen3.5 decoder layer (ttnn port of transformers' Qwen3_5DecoderLayer).

Each layer is either a full (softmax) attention layer or a Gated DeltaNet (linear
attention) layer, chosen per index by args.is_full_attention_layer(layer_num). Both
kinds share the identical RMSNorm + residual + SwiGLU-MLP wiring of the HF reference:

    x -> input_layernorm -> token_mixer -> + x
      -> post_attention_layernorm -> mlp -> + residual

See Qwen3_5DecoderLayer in transformers.models.qwen3_5.modeling_qwen3_5 for the golden.
"""
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen3_5_9b.tt.attention import Qwen35Attention
from models.demos.blackhole.qwen3_5_9b.tt.gdn import Qwen35GatedDeltaNet
from models.demos.blackhole.qwen3_5_9b.tt.mlp import Qwen35MLP
from models.demos.blackhole.qwen3_5_9b.utils.substate import substate
from models.tt_transformers.tt.common import Mode


class Qwen35DecoderLayer(LightweightModule):
    """One transformer block: a full-attention OR gated-deltanet token mixer, then a SwiGLU MLP.

    The only thing that varies between the two layer kinds is which token mixer the layer
    holds; the norm/residual/MLP structure is identical, mirroring transformers'
    Qwen3_5DecoderLayer.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.args = args
        self.device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        # Per-layer tensor cache (sharded mesh weights are written here once, then re-read on
        # subsequent runs to skip the slow single-threaded reorder+shard of the full model).
        cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None

        # Decoder norms stay on the framework RMSNorm + DistributedNorm.
        #   * The new tt/rms_norm.py Qwen35RMSNorm is the attention's internal q/k-norm primitive
        #     and is deliberately non-distributed (plain local RMS over the last dim).
        #   * On tensor parallelism the residual stream is fractured along the hidden dim, so the
        #     decoder norm must do the fractured->replicated transition (distributed stats-reduce +
        #     all-gather) before feeding the column-parallel token mixer — exactly what
        #     DistributedNorm provides. add_unit_offset bakes the Qwen3_5RMSNorm (1 + weight) scale.
        self.input_layernorm = self._make_norm(state_dict, "input_layernorm", tensor_cache_path, "attention_norm")
        self.post_attention_layernorm = self._make_norm(
            state_dict, "post_attention_layernorm", tensor_cache_path, "ff_norm"
        )

        # Token mixer: the new attention/gdn classes are themselves tensor-parallel (they take
        # tt_ccl and reduce-scatter their output), so there is a single class per kind regardless
        # of mesh size — the old TP-vs-single-device class fork is gone. Note the constructor arg
        # order differs between the two (attention leads with mesh_device, gdn leads with args).
        if self.is_full_attention:
            # create_kv_cache=True allocates the internal contiguous cache the demo/eager decode
            # path reads; a paged (vLLM) deployment rebinds it later via set_paged_kv_cache.
            self.attention = Qwen35Attention(
                mesh_device,
                substate(state_dict, f"layers.{layer_num}.self_attn"),
                args,
                tt_ccl,
                create_kv_cache=True,
                tensor_cache_path=cache,
            )
        else:
            # GDN carries its own conv + recurrent state internally (no external KV cache, no RoPE).
            self.attention = Qwen35GatedDeltaNet(
                args,
                substate(state_dict, f"layers.{layer_num}.linear_attn"),
                mesh_device,
                tt_ccl=tt_ccl,
                tensor_cache_path=cache,
            )

        self.feed_forward = Qwen35MLP(
            mesh_device,
            substate(state_dict, f"layers.{layer_num}.mlp"),
            args,
            tensor_cache_path=cache,
            tt_ccl=tt_ccl,
        )

    def _make_norm(self, state_dict, weight_key, tensor_cache_path, ag_key):
        """Build a per-layer RMSNorm, wrapped in DistributedNorm when TP>1.

        On a single device this is the plain framework RMSNorm the validated 9B path used; the
        DistributedNorm wrapper (TP>1) mirrors tt_transformers' decoder and handles the
        fractured->replicated transition the non-distributed token-mixer input requires.
        """
        norm = RMSNorm(
            device=self.device,
            dim=self.args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{self.layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=self.args.norm_eps,
            **(
                dict(
                    is_distributed=self.args.is_distributed_norm,
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                )
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(norm, self.args, tt_ccl=self.tt_ccl, TG=self.args.is_galaxy, ag_config_key=ag_key)
        return norm

    def forward_prefill(self, x, cos=None, sin=None, page_table=None, user_id=0):
        """Prefill one stream: input_norm → token mixer → residual → post_norm → MLP → residual.

        The full-attention mixer takes the RoPE cos/sin plus the KV-fill paging args:
          * page_table=None drives the contiguous KV fill the demo/eager path reads.
          * a non-None page_table drives the paged fill (the path vLLM uses).
        GDN takes none of those — it has no RoPE and folds its conv/recurrent state in internally.
        """
        attn_norm_config, ff_norm_config = self._norm_configs(Mode.PREFILL)

        attn_in = self.input_layernorm(x, mode=Mode.PREFILL, norm_config=attn_norm_config)
        if self.is_full_attention:
            attn_out = self.attention.forward_prefill(attn_in, cos, sin, page_table=page_table, user_id=user_id)
        else:
            attn_out = self.attention.forward_prefill(attn_in)  # captures GDN conv/recurrent state
        ttnn.deallocate(attn_in)
        h = ttnn.add(x, attn_out)  # residual; x is the caller's tensor and is left live for it
        ttnn.deallocate(attn_out)

        ff_in = self.post_attention_layernorm(h, mode=Mode.PREFILL, norm_config=ff_norm_config)
        ff_out = self.feed_forward.forward(ff_in)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def forward_decode(self, x, position_tensor=None, cos=None, sin=None, page_table=None):
        """Decode one step: same input_norm → mixer → residual → post_norm → MLP → residual block as prefill.

        Only the token-mixer call differs from prefill — full attention reads its KV cache at
        position_tensor (advancing it) under RoPE cos/sin and the optional page_table, while GDN's
        forward_decode reads + advances its recurrent state and ignores all of those (it carries its
        position implicitly). The norm/residual/MLP wiring around it is identical (only the norm memory
        config changes), so it's spelled out here too rather than hidden behind a shared helper.
        """
        attn_norm_config, ff_norm_config = self._norm_configs(Mode.DECODE)

        attn_in = self.input_layernorm(x, mode=Mode.DECODE, norm_config=attn_norm_config)
        if self.is_full_attention:
            attn_out = self.attention.forward_decode(attn_in, position_tensor, cos, sin, page_table=page_table)
        else:
            attn_out = self.attention.forward_decode(attn_in)
        ttnn.deallocate(attn_in)
        h = ttnn.add(x, attn_out)  # residual; x is the caller's tensor and is left live for it
        ttnn.deallocate(attn_out)

        ff_in = self.post_attention_layernorm(h, mode=Mode.DECODE, norm_config=ff_norm_config)
        ff_out = self.feed_forward.forward(ff_in)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def _norm_configs(self, norm_mode):
        """Memory configs (attn_norm, ff_norm) for the two decoder norms, which vary by phase and mesh size.

        Pulled out of forward_* because it's an orthogonal config lookup, not part of the block's data
        flow: on TP (>1 device) DistributedNorm wants the framework's per-norm configs; on a single device
        we keep the decode norm output in L1 (as the validated single-device path did) and let prefill fall
        back to the framework default (interleaved DRAM).
        """
        if self.num_devices > 1:
            return self.args.get_norm_config("attn", norm_mode), self.args.get_norm_config("ff", norm_mode)
        cfg = {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if norm_mode == Mode.DECODE else None
        return cfg, cfg
