# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os

from loguru import logger

import ttnn
from models.demos.qwen3_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.model_config import ModelArgs

# MESH_DEVICE → (DP axis 0, TP axis 1). 2D Galaxy: TG=(8, 4), TG_4x8=(4, 8).
VISION_MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
    "TG_4x8": (4, 8),
    "4x8": (4, 8),
    "8x4": (8, 4),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def vision_mesh_shape_from_env():
    """Cluster shape for vision unit tests from ``MESH_DEVICE``."""
    env = os.environ.get("MESH_DEVICE")
    if env in VISION_MESH_DEVICE_MAP:
        return VISION_MESH_DEVICE_MAP[env]
    return len(ttnn.get_device_ids())


def vision_tp_reduce_scatter(tensor, mesh_device, tt_ccl, args):
    """Reduce-scatter along the TP axis so the result is fractured on dim=3.

    1D meshes (T3K/QB2) go through ``tt_all_reduce(cluster_axis=0)``: that helper
    short-circuits ``cluster_axis==1`` when ``1 in mesh_shape``, and the 1D path
    is already a reduce-scatter. 2D meshes (DP on axis 0, TP on axis 1) must
    reduce-scatter with ``cluster_axis=1`` — ``tt_all_reduce``'s 2D path is a
    full all-reduce (replicated), which would break the fractured block I/O
    contract and mix DP groups if aimed at axis 0.
    """
    if not args.is_multichip:
        return tensor
    if args.is_2d_mesh:
        cluster_axis = 1
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=tt_ccl.get_num_links(cluster_axis),
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=args.ccl_topology(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        return reduced
    return tt_all_reduce(
        tensor,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        sharded=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=args.ccl_dtype,
        topology=args.ccl_topology(),
    )


class ModelOptimizations:
    def __init__(self, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        self.bfp4_mlp = False
        # self.bfp4_mlp = "Qwen3-VL-32B" in model_name


class VisionModelArgs(ModelArgs):
    # Base __init__ checks the TEXT config's KV heads; the vision tower's own MHA
    # heads (set below) shard along cluster axis 1, so only the base check needs relaxing.
    SUPPORTS_KV_REPLICATION = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parallelism: DP on cluster axis 0 (batch), TP on cluster axis 1 (hidden).
        # 1D meshes (N150/N300/T3K/QB2) have dp=1 and only TP. 2D Galaxy meshes
        # (4, 8) and (8, 4) run both: one image per DP rank, Megatron TP on hidden.
        self.dp = self.cluster_shape[0]
        self.tp = self.cluster_shape[1]
        self.is_2d_mesh = self.dp > 1 and self.tp > 1

        # Core dimensions from HF config
        self.dim = self.hf_config.vision_config.hidden_size
        self.unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        # Pad the MLP intermediate to a tile multiple per TP rank (not per mesh device).
        self.hidden_dim = nearest_multiple(self.unpadded_hidden_dim, self.tile_size * self.tp)
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_key_value_heads
        self.n_heads = self.hf_config.vision_config.num_key_value_heads
        self.n_kv_heads = self.hf_config.vision_config.num_key_value_heads

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size

        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding head dim from {self.head_dim} to {self.padded_head_dim}")

        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        self.optimizations = ModelOptimizations(
            self.model_name
        )  # todo)) implement finer grained control similar to tt_transformers'

        num_rows = lambda seq_len: min(seq_len, 1024 if self.is_2d_mesh else 2048)
        # WO is row-sharded on the concat-heads dim across TP; the linear emits
        # a full-width partial sum that is then reduce-scattered. On 2D, k is
        # the local concat-heads width (not dim/dp — axis 0 is batch, not TP).
        if self.is_2d_mesh:
            k_dim = (self.n_heads // self.tp) * self.padded_head_dim
            n_dim = self.dim
            in0_block_w = 1
        else:
            k_dim = self.dim
            n_dim = self.dim
            in0_block_w = self.dim // 1024
        self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
            m=num_rows(seq_len),
            k=k_dim,
            n=n_dim,
            grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
            in0_block_w=in0_block_w,
            fuse_batch=seq_len <= 1024,
        )

        assert self.n_kv_heads % self.tp == 0, f"n_kv_heads ({self.n_kv_heads}) must be divisible by TP={self.tp}"
        assert self.n_heads % self.tp == 0, f"vision n_heads ({self.n_heads}) must be divisible by TP={self.tp}"
        assert self.qkv_size % self.tp == 0, f"vision qkv_size ({self.qkv_size}) must be divisible by TP={self.tp}"
        assert self.dim % self.tp == 0, f"vision dim ({self.dim}) must be divisible by TP={self.tp}"
        assert (
            self.hidden_dim % self.tp == 0
        ), f"vision hidden_dim ({self.hidden_dim}) must be divisible by TP={self.tp}"

    def ccl_topology(self, cluster_axis=None):
        """Topology for a collective along ``cluster_axis`` (defaults to TP, axis 1).

        ``ModelArgs.ccl_topology`` picks Ring from the *total* device count, but a
        Ring needs a wraparound link on the axis actually being gathered. On a 2D
        Galaxy the TP axis is only 4 wide and ``FABRIC_1D`` has no wraparound
        there, so a Ring request fails routing in ``fabric.cpp`` with "Could not
        find any forwarding direction from src (M0, D0) to dst (M0, D3)". Gate
        Ring on the width of the gathered axis instead, which is what the base
        class's own 1x4-submesh fallback intends.
        """
        topology = super().ccl_topology()
        if not self.is_2d_mesh or topology != ttnn.Topology.Ring:
            return topology
        axis = 1 if cluster_axis is None else cluster_axis
        return topology if self.cluster_shape[axis] >= 8 else ttnn.Topology.Linear

    def activation_shard_dims(self, batch_size, batch_dim=1):
        """Shard dims for a 2D mapper: DP (axis 0) on batch, TP (axis 1) on hidden.

        When ``batch_size`` is not a multiple of ``dp`` the batch axis is
        replicated (unit tests with batch=1 on a 2D mesh).
        """
        shard_batch = self.dp > 1 and batch_size % self.dp == 0
        return (batch_dim if shard_batch else None, -1)

    def dp_mesh_mapper(self, batch_size, batch_dim):
        """Shard ``batch_dim`` across DP (axis 0); replicate on TP (axis 1).

        Used for tensors that have no hidden dim (pixels, position ids, pooling
        weights). Falls back to full replication when batch cannot be split
        evenly across DP ranks.
        """
        if self.mesh_device is None or self.mesh_device.__class__.__name__ != "MeshDevice":
            return None
        if self.dp > 1 and batch_size % self.dp == 0:
            return ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(batch_dim, None),
                mesh_shape=self.cluster_shape,
            )
        return ttnn.ReplicateTensorToMesh(self.mesh_device)

    def prepare_residual_tensor_prefill(self, x_bsh):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch
        S: sequence len
        H: dim

        Vision blocks consume tensors fractured along the hidden dim (dim=3)
        across TP (axis 1). When batch is a multiple of ``dp``, batch is also
        sharded across DP (axis 0).
        """
        x_1BSH = x_bsh.unsqueeze(0)
        batch = x_1BSH.shape[1]

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            dims=self.activation_shard_dims(batch, batch_dim=1),
            mesh_shape=self.cluster_shape,
        )

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

    # Visual model does not use distributed norm for now
    def is_distributed_norm(self, mode):
        return False

    def get_state_dict_prefix(self, module_name, layer_num=None, deepstack_merger_num=None):
        layer_prefix = f"visual.encoder.layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "mlp",
            "Gemma4VisionMLP": "mlp",
            "VisionAttention": "self_attn",
            "VisionBlock": "",
            "VisionTransformer": "visual",
            "PatchMerger": "visual.merger",
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            "pre_feedforward_layernorm": "pre_feedforward_layernorm",
            "post_feedforward_layernorm": "post_feedforward_layernorm",
            "DeepstackMerger": f"visual.deepstack_merger_list.{deepstack_merger_num}",
            "": "",  # If no module is given, just get layer prefix
        }
        return layer_prefix + module_map[module_name]

    def reference_vision_model(self, depth=None):
        # Workaround until Qwen2.5-VL is fully integrated into a HF release
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as AutoModelForCausalLM

        print("Loading Gemma-4 model: ", AutoModelForCausalLM)
        config = AutoModelForCausalLM.config_class.from_pretrained(self.CKPT_DIR)
        config.vision_config.num_hidden_layers = depth if depth is not None else config.vision_config.num_hidden_layers
        model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR, config=config)
        return model.model.vision_tower

    def reference_vision_block(self, layer_num=0):
        return self.reference_vision_model().encoder.layers[layer_num]

    def reference_mlp(self):
        return self.reference_vision_block().mlp

    def reference_attention(self):
        return self.reference_vision_block().self_attn

    def reference_rms_norm(self):
        return self.reference_vision_block().norm2

    def reference_pooler(self):
        return self.reference_vision_model().pooler

    def reference_rotary_emb(self):
        return self.reference_vision_model().encoder.rotary_emb

    def reference_patch_merger(self):
        return self.reference_vision_model().merger

    def reference_deepstack_merger(self):
        return self.reference_vision_model().deepstack_merger_list[0]

    def reference_patch_embedder(self):
        return self.reference_vision_model().patch_embedder

    def reference_patch_embed(self):
        return self.reference_vision_model().patch_embed
