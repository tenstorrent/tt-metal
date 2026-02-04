# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ttnn
import os
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mixtral_mlp import TtMixtralMLP
from models.tt_transformers.tt.mixtral_moe import TtMoeLayer
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
        self.is_mixture_of_experts = False
        self.layer_num = layer_num
        # Check the HF_MODEL environment variable
        hf_model = os.getenv("HF_MODEL", "").strip()
        # If the model explicitly matches Phi-1 or Phi-1.5, set flag
        is_phi1 = hf_model in {"microsoft/Phi-1"}
        self.is_phi1 = is_phi1

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

        if getattr(self.args, "is_mixture_of_experts", False):
            self.feed_forward = TtMoeLayer(
                mesh_device=mesh_device,
                state_dict=state_dict,
                experts=TtMixtralMLP(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    args=args,
                    layer_num=layer_num,
                    dtypes={
                        "w1": dtype,
                        "w2": dtype,
                        "w3": dtype,
                    },
                ),
                args=args,
                layer_num=layer_num,
                dtype=dtype,
                tt_ccl=self.tt_ccl,
            )
        else:
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
                force_weight_tile=is_phi1,
                is_distributed=(False if is_phi1 else self.args.is_distributed_norm),
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
            force_local_norm=is_phi1,
        )
        # Phi-1 does not have post_attention_layernorm / ffn_norm; it only has input_layernorm,
        # which we map to attention_norm. In that case, reuse attention_norm for the FFN path.
        ffn_norm_key = "ffn_norm"
        if f"layers.{layer_num}.ffn_norm.weight" not in state_dict:
            ffn_norm_key = "attention_norm"

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
                force_weight_tile=is_phi1,
                is_distributed=(False if is_phi1 else self.args.is_distributed_norm),
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
            force_local_norm=is_phi1,
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
                    force_weight_tile=is_phi1,
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
                    force_weight_tile=is_phi1,
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

        def _phi1_safe_add(a, b, mem_cfg, dtype):
            if self.num_devices <= 1:
                return ttnn.add(a, b, memory_config=mem_cfg, dtype=dtype)

            def _get_device_tensors(t):
                if hasattr(ttnn, "get_device_tensors"):
                    return ttnn.get_device_tensors(t)
                if hasattr(ttnn, "distributed") and hasattr(ttnn.distributed, "get_device_tensors"):
                    return ttnn.distributed.get_device_tensors(t)
                return None

            def _get_dist_cfg(t):
                return t.distributed_tensor_config() if hasattr(t, "distributed_tensor_config") else None

            def _aggregate_as_tensor(device_tensors, like_tensor):
                cfg = _get_dist_cfg(like_tensor)
                if cfg is not None:
                    if hasattr(ttnn, "aggregate_as_tensor"):
                        return ttnn.aggregate_as_tensor(device_tensors, cfg)
                    if hasattr(ttnn, "distributed") and hasattr(ttnn.distributed, "aggregate_as_tensor"):
                        return ttnn.distributed.aggregate_as_tensor(device_tensors, cfg)
                return ttnn.combine_device_tensors(device_tensors)

            a_shards = _get_device_tensors(a)
            b_shards = _get_device_tensors(b)

            a_is_sharded = isinstance(a_shards, (list, tuple)) and len(a_shards) > 1
            b_is_sharded = isinstance(b_shards, (list, tuple)) and len(b_shards) > 1

            # sharded + sharded
            if a_is_sharded and b_is_sharded and len(a_shards) == len(b_shards):
                out_shards = [
                    ttnn.add(ai, bi, memory_config=mem_cfg, dtype=dtype)
                    for ai, bi in zip(a_shards, b_shards)
                ]
                return _aggregate_as_tensor(out_shards, a)
            
            # a is sharded (list of per-device tensors), b is not (single full-width tensor)
            if a_is_sharded and not b_is_sharded:
                # residual shards define the sharding scheme
                shard_w = a_shards[0].shape[-1]
                num = len(a_shards)
                total_w = shard_w * num

                # Only handle the exact mismatch you're seeing: b is full width, a shards are partial width
                if hasattr(b, "shape") and b.shape[-1] == total_w:
                    # Bring b to host as a torch.Tensor, slice in Python, then rebuild a TT tensor
                    b_torch = ttnn.to_torch(b)  # docs: ttnn.to_torch :contentReference[oaicite:4]{index=4}

                    out_shards = []
                    for i, ai in enumerate(a_shards):
                        start = i * shard_w
                        end = (i + 1) * shard_w

                        # Slice last dim in torch (safe, no TTNN device-op invoked)
                        bi_torch = b_torch[..., start:end]

                        # Recreate as TT tensor in TILE so it can add with ai (also TILE)
                        bi_host = ttnn.from_torch(
                            bi_torch,
                            dtype=ai.dtype,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )  # docs: ttnn.from_torch :contentReference[oaicite:5]{index=5}

                        # Put on mesh, then select the i-th device shard (same as your existing flow)
                        bi_mesh = ttnn.to_device(bi_host, device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                        bi_list = ttnn.get_device_tensors(bi_mesh)
                        bi = bi_list[i]

                        out_shards.append(ttnn.add(ai, bi, memory_config=mem_cfg, dtype=dtype))

                                        # Host aggregate (torch.cat), then upload as a SHARDED mesh tensor
                    import torch

                    out_torch = [ttnn.to_torch(s).detach().cpu() for s in out_shards]
                    full = torch.cat(out_torch, dim=-1)

                    num_devices = len(a_shards)

                    mesh_mapper_config = ttnn.MeshMapperConfig(
                        [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
                        ttnn.MeshShape(1, num_devices),
                    )
                    mesh_mapper = ttnn.create_mesh_mapper(self.mesh_device, mesh_mapper_config)

                    full_mesh_sharded = ttnn.from_torch(
                        full,
                        dtype=out_shards[0].dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mesh_mapper,
                    )

                    return ttnn.to_memory_config(full_mesh_sharded, mem_cfg)

                # Don’t fall back to mesh add — it will hit invalid subtile broadcast again
                raise RuntimeError(
                    f"_phi1_safe_add: cannot align shapes; a_shard_last={a_shards[0].shape[-1]} "
                    f"num_shards={len(a_shards)} b_shape={getattr(b,'shape',None)}"
                )


            if b_is_sharded and not a_is_sharded:
                a2 = ttnn.to_memory_config(a, mem_cfg)
                out_shards = [
                    ttnn.add(a2, bi, memory_config=mem_cfg, dtype=dtype)
                    for bi in b_shards
                ]
                return _aggregate_as_tensor(out_shards, b)

            # neither sharded (safe normal add)
            a2 = ttnn.to_memory_config(a, mem_cfg)
            b2 = ttnn.to_memory_config(b, mem_cfg)
            return ttnn.add(a2, b2, memory_config=mem_cfg, dtype=dtype)


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
        # TODO: create correct memory config in RopeSetup (issue is in ttnn.add op because of different shape in memory config for residual and rot_mats)
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)

        if self.pre_ff_norm is None:
            if getattr(self, "is_phi1", False) and mode == "prefill":
                if getattr(self, "is_phi1", False) and mode == "prefill" and not hasattr(self, "_meta_dumped"):
                    self._meta_dumped = True
                    #with open("/tmp/phi1_add_meta.txt", "w") as f:
                    #    f.write(f"residual: {residual}\n")
                    #    f.write(f"attn_out: {attn_out}\n")

                hidden_states = _phi1_safe_add(residual, attn_out, skip_mem_cfg, ttnn.bfloat16 if TG else None)
            else:
                hidden_states = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None)
            residual = hidden_states
            if mode == "prefill":
                x.deallocate(True)
        else:
            hidden_states = attn_out
        hidden_states = self.ff_norm(hidden_states, mode)
        if self.pre_ff_norm is not None:
            # The output of the ff_norm is replicated across the device
            # but the residual is fractured across the devices
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

        # For Phi-1 prefill, keep residual + MLP output in same memcfg before the final add
        if getattr(self, "is_phi1", False) and mode == "prefill":
            hidden_states = ttnn.to_memory_config(hidden_states, skip_mem_cfg)
            residual = ttnn.to_memory_config(residual, skip_mem_cfg)

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

        final_dtype = (
            self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16
        )
        if getattr(self, "is_phi1", False) and mode == "prefill":
            out = _phi1_safe_add(residual, hidden_states, skip_mem_cfg, final_dtype)
        else:
            out = ttnn.add(residual, hidden_states, memory_config=skip_mem_cfg, dtype=final_dtype)

        return out  # fractured across devices
