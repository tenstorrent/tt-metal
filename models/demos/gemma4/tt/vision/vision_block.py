# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.gemma4.tt.vision.vision_attention import VisionAttention
from models.demos.gemma4.tt.vision.vision_mlp import Gemma4VisionMLP
from models.tt_transformers.tt.common import Mode


class VisionDistributedNorm(LightweightModule):
    """RMSNorm over a hidden-fractured residual.

    Gathers dim=3 along the TP axis only, then runs a local RMSNorm. On 1D
    meshes that is a mesh-wide gather (no ``cluster_axis``). On 2D meshes
    (DP axis 0, TP axis 1) the gather is ``cluster_axis=1`` so DP groups stay
    isolated — the stock ``DistributedNorm`` all-gathers the whole mesh and
    would concatenate hidden shards from every DP rank.
    """

    def __init__(self, norm, args, tt_ccl):
        super().__init__()
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl

    def forward(self, x, mode: Mode, norm_config=None):
        if self.args.is_multichip:
            # 2D: gather only along TP. 1D: omit cluster_axis so the gather
            # covers the whole line (tt_all_gather short-circuits cluster_axis=1
            # when 1 is in mesh_shape).
            cluster_axis = 1 if self.args.is_2d_mesh else None
            gather_kwargs = dict(
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=self.tt_ccl.get_num_links(1 if cluster_axis is None else cluster_axis),
                topology=self.args.ccl_topology(cluster_axis),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            if cluster_axis is not None:
                gather_kwargs["cluster_axis"] = cluster_axis
            x = ttnn.experimental.all_gather_async(x, **gather_kwargs)

        return self.norm(x, mode=mode, in_sharded=False, out_sharded=False, norm_config=norm_config)


class VisionBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        tt_ccl,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
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
        self.tt_ccl = tt_ccl

        if self.tt_ccl is None:
            raise ValueError("VisionBlock requires a `tt_ccl` instance")

        self.layer_num = layer_num

        self.attention = VisionAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            tt_ccl=tt_ccl,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
        )
        self.feed_forward = Gemma4VisionMLP(
            mesh_device=mesh_device,
            args=args,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
        )

        # Block I/O is fractured along dim=3, so the norms all-gather first
        # (mirrors `DistributedNorm` in the LLM decoder).
        def make_norm(prefix):
            return VisionDistributedNorm(
                RMSNorm(
                    device=mesh_device,
                    dim=args.dim,
                    eps=1e-6,
                    state_dict=state_dict,
                    state_dict_prefix=args.get_state_dict_prefix(prefix, layer_num),
                    weight_cache_path=None if args.dummy_weights else weight_cache_path,
                    weight_dtype=ttnn.bfloat16,
                    weight_key="",
                    tt_ccl=tt_ccl,
                ),
                args,
                tt_ccl,
            )

        self.input_norm = make_norm("input_layernorm")
        self.post_attention_norm = make_norm("post_attention_layernorm")
        self.pre_ff_norm = make_norm("pre_feedforward_layernorm")
        self.post_ff_norm = make_norm("post_feedforward_layernorm")

    def _fracture_hidden(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """Re-shard a replicated (post-norm) tensor along dim=3 to match residual TP layout."""
        if self.args.is_multichip and not self.args.is_distributed_norm(Mode.PREFILL):
            frac = ttnn.mesh_partition(
                t,
                memory_config=t.memory_config(),
                dim=3,
                cluster_axis=1,
            )
            if frac is not t:
                ttnn.deallocate(t)
            return frac
        return t

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats,
    ) -> ttnn.Tensor:
        """Run the vision block.

        I/O contract: ``x`` is fractured along dim=3 (each device holds dim/TP),
        and on 2D meshes also sharded on batch along DP (axis 0). Output matches.
        Norms internally all-gather along TP only; attention/MLP end with a
        reduce-scatter along TP. Post-norms are ``mesh_partition``ed before the
        residual add so both sides of the add stay fractured.
        """
        skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"VisionBlock input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        attn_in = self.input_norm(x, mode=Mode.PREFILL)
        attn_out = self.attention.forward(
            attn_in,
            rot_mats=rot_mats,
        )
        attn_normed = self.post_attention_norm(attn_out, mode=Mode.PREFILL)
        attn_normed = self._fracture_hidden(attn_normed)
        h = ttnn.add(x, attn_normed, memory_config=skip_mem_cfg, dtype=None)
        ttnn.deallocate(attn_normed)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(x)

        ff_in = self.pre_ff_norm(h, mode=Mode.PREFILL)
        ff_out = self.feed_forward.forward(ff_in, mode=Mode.PREFILL)
        ff_normed = self.post_ff_norm(ff_out, mode=Mode.PREFILL)
        ff_normed = self._fracture_hidden(ff_normed)
        ttnn.deallocate(ff_in)
        out = ttnn.add(
            h,
            ff_normed,
            memory_config=skip_mem_cfg,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        ttnn.deallocate(ff_normed)

        return out
