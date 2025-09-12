# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mamba_falcon import FalconH1Mamba
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup


class TransformerBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl

        self.num_devices = args.num_devices
        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()

        self.layer_num = layer_num

        ActualAttentionClass = attention_class if attention_class is not None else DefaultAttention

        self.attention = ActualAttentionClass(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )
        # Optional Mamba branch (Falcon-H1 hybrid). Instantiate only if weights exist and not disabled.
        has_mamba = any(k.startswith(f"layers.{layer_num}.mamba.") for k in state_dict.keys())
        disable_mamba_env = os.environ.get("TT_DISABLE_MAMBA", "0").lower() in ("1", "true")
        disable_mamba_arg = getattr(args, "disable_mamba", False)
        enable_mamba = has_mamba and not (disable_mamba_env or disable_mamba_arg)
        self.mamba = (
            FalconH1Mamba(
                mesh_device=mesh_device,
                args=args,
                state_dict=state_dict,
                layer_num=layer_num,
                weight_cache_path=weight_cache_path,
            )
            if enable_mamba
            else None
        )
        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=self.args.is_distributed_norm,
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
        )

        # Resolve FFN norm key dynamically for broader model compatibility (Falcon, etc.)
        def _resolve_ffn_norm_key(sd, ln):
            base = f"layers.{ln}."
            # Preferred aliases in order
            preferred = [
                "ffn_norm",
                "post_attention_layernorm",
                "post_feedforward_layernorm",
                "ln_mlp",
                "mlp_layernorm",
                "mlp_ln",
            ]
            for key in preferred:
                if f"{base}{key}.weight" in sd:
                    return key
            # Fallback: scan for any layer norm that is not the attention norm
            suffixes = []
            for k in sd.keys():
                if k.startswith(base) and k.endswith(".weight") and ("norm" in k or "layernorm" in k):
                    suffix = k[len(base) : -len(".weight")]
                    if suffix != "attention_norm" and not suffix.startswith("pre_feedforward"):
                        suffixes.append(suffix)
            if suffixes:
                return suffixes[0]
            return "ffn_norm"

        ffn_norm_key = _resolve_ffn_norm_key(state_dict, layer_num)

        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=ffn_norm_key,
                is_distributed=self.args.is_distributed_norm,
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
        )
        if f"layers.{layer_num}.pre_feedforward_layernorm.weight" in state_dict:
            self.pre_ff_norm = DistributedNorm(  # pre_feedforward_layernorm
                RMSNorm(
                    device=mesh_device,
                    dim=args.dim,
                    eps=args.norm_eps,
                    state_dict=state_dict,
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                    weight_cache_path=None if args.dummy_weights else weight_cache_path,
                    weight_dtype=ttnn.bfloat16,
                    weight_key="pre_feedforward_layernorm",
                    is_distributed=self.args.is_distributed_norm,
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                ),
                args,
                tt_ccl=self.tt_ccl,
                TG=args.is_galaxy,
            )
        else:
            # If pre_feedforward_layernorm is not in state_dict, we do not use it
            self.pre_ff_norm = None

        if f"layers.{layer_num}.post_feedforward_layernorm.weight" in state_dict:
            self.post_ff_norm = DistributedNorm(  # post_feedforward_layernorm
                RMSNorm(
                    device=mesh_device,
                    dim=args.dim,
                    eps=args.norm_eps,
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    state_dict=state_dict,
                    state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                    weight_cache_path=None if args.dummy_weights else weight_cache_path,
                    weight_dtype=ttnn.bfloat16,
                    weight_key="post_feedforward_layernorm",
                    is_distributed=self.args.is_distributed_norm,
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                ),
                args,
                tt_ccl=self.tt_ccl,
                TG=args.is_galaxy,
            )
        else:
            # If post_feedforward_layernorm is not in state_dict, we do not use it
            self.post_ff_norm = None

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        residual = x
        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # Choose the correct rotation matrices based on the mode
        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        # Norms take fractured inputs and output replicated across devices
        attn_in = self.attention_norm(x, mode)
        # Apply attention input multiplier if provided by config (Falcon-H1)
        if hasattr(self.args, "attention_in_multiplier"):
            attn_in = ttnn.multiply(attn_in, self.args.attention_in_multiplier)
        # Compute Mamba branch BEFORE attention (attention may deallocate its input)
        mamba_out = None
        if mode == "prefill" and self.mamba is not None:
            mamba_out = self.mamba.forward(attn_in, mode)

        # Attention takes replicated inputs and produces fractured outputs
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        # Apply attention out multiplier if provided by config (Falcon-H1)
        if hasattr(self.args, "attention_out_multiplier"):
            attn_out = ttnn.multiply(attn_out, self.args.attention_out_multiplier)

        # mamba_out computed above
        # Ensure mamba_out matches attn_out layout/memory_config before add (avoid broadcast errors)
        if mamba_out is not None:
            try:
                # Force both tensors to DRAM + ROW_MAJOR + 4D shape [1,1,S,H]
                attn_out = ttnn.to_memory_config(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                mamba_out = ttnn.to_memory_config(mamba_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if attn_out.layout != ttnn.ROW_MAJOR_LAYOUT:
                    attn_out = ttnn.to_layout(attn_out, ttnn.ROW_MAJOR_LAYOUT)
                if mamba_out.layout != ttnn.ROW_MAJOR_LAYOUT:
                    mamba_out = ttnn.to_layout(mamba_out, ttnn.ROW_MAJOR_LAYOUT)
                attn_out = ttnn.unsqueeze_to_4D(attn_out)
                mamba_out = ttnn.unsqueeze_to_4D(mamba_out)
            except Exception:
                pass

        if self.pre_ff_norm is None:
            # Residual connection after attention (no mamba path implemented yet)
            hidden_states = ttnn.add(
                residual,
                attn_out if mamba_out is None else ttnn.add(attn_out, mamba_out),
                memory_config=skip_mem_cfg,
                dtype=ttnn.bfloat16 if TG else None,
            )
            residual = hidden_states
            if mode == "prefill":
                x.deallocate(True)
            # Apply FFN norm only in architectures without pre-FF norm
            hidden_states = self.ff_norm(hidden_states, mode)
        else:
            # Falcon-H1 style: add residual first, then apply pre-FF norm; no extra FFN norm here
            hidden_states = attn_out if mamba_out is None else ttnn.add(attn_out, mamba_out)
            # The attention output is replicated; residual is fractured. Gather before add if multi-device
            if self.num_devices > 1:
                hidden_states = tt_all_reduce(
                    hidden_states,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=0,
                    dim=3,
                    num_reduce_scatter_links=self.args.num_reduce_scatter_links,
                    num_all_gather_links=self.args.num_all_gather_links,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=self.args.ccl_dtype,
                )
                hidden_states = ttnn.div(hidden_states, self.num_devices)
            hidden_states = ttnn.add(
                residual, hidden_states, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
            )
            residual = hidden_states
            hidden_states = self.pre_ff_norm(hidden_states, mode)

        ttnn.deallocate(attn_out)

        if TG and mode == "decode":
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.model_config["MLP_ACT_MEMCFG"])
        # MLP takes replicated inputs and produces fractured outputs

        hidden_states = self.feed_forward.forward(hidden_states, mode)

        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        if self.post_ff_norm is not None:
            hidden_states = self.post_ff_norm(hidden_states, mode)  # Gathered
            if self.num_devices > 1:
                hidden_states = tt_all_reduce(
                    hidden_states,
                    self.mesh_device,
                    tt_ccl=self.tt_ccl,
                    cluster_axis=0,
                    dim=3,
                    num_reduce_scatter_links=self.args.num_reduce_scatter_links,
                    num_all_gather_links=self.args.num_all_gather_links,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=self.args.ccl_dtype,
                )

                hidden_states = ttnn.div(hidden_states, self.num_devices)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )

        return out  # fractured across devices
