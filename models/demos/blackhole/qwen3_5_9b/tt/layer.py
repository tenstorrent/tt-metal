# models/demos/blackhole/qwen3_5_9b/tt/layer.py
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""
import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen3_5_9b.tt.attention import AttentionConfig, Qwen35GatedAttention
from models.demos.blackhole.qwen3_5_9b.tt.gdn import GDNConfig, Qwen35GatedDeltaNet
from models.demos.blackhole.qwen3_5_9b.tt.mlp import Qwen35MLP
from models.demos.blackhole.qwen3_5_9b.utils.substate import substate
from models.tt_transformers.tt.common import Mode


class Qwen35DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        # NOTE (multi-device/TP handoff): is_distributed=None is correct on single device.
        # For 27B TP, the framework Embedding shards the hidden dim, so the hidden state
        # entering RMSNorm is sharded -> these norms must then pass is_distributed=args.is_distributed_norm
        # + tt_ccl=<the model's self.tt_ccl> (or wrap in tt_transformers DistributedNorm) to all-gather.
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key="input_layernorm",
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
        )
        self.ffn_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key="post_attention_layernorm",
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
        )

        if self.is_full_attention:
            attn_state = substate(state_dict, f"layers.{layer_num}.self_attn")
            attn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen35GatedAttention(mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache)
        else:
            gdn_state = substate(state_dict, f"layers.{layer_num}.linear_attn")
            gdn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen35GatedDeltaNet(mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache)

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        self.feed_forward = Qwen35MLP(mesh_device, mlp_state, mlp_cache)

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
    ):
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
        # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
        _norm_config = {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
        attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_norm_config)

        if self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_norm_config)

        ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
