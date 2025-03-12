import math
import re
from loguru import logger
import ttnn
from models.tt_transformers.tt.common import nearest_multiple


class VisionModelArgs:
    def __init__(self, model_args):
        self.mesh_device = model_args.mesh_device
        self.hf_config = model_args.hf_config
        self.model_args = model_args

        # Core dimensions from HF config
        self.dim = self.hf_config.vision_config.hidden_size
        self.unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(  # pad to a tile multiple per device
            self.unpadded_hidden_dim, model_args.tile_size * model_args.num_devices
        )
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_heads
        self.n_heads = self.hf_config.vision_config.num_heads
        self.n_kv_heads = self.hf_config.vision_config.num_heads
        self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / (
            self.n_kv_heads // self.model_args.cluster_shape[1]
        )
        self.MAX_QKV_MM_SEQ_LEN = self.model_args.MAX_QKV_MM_SEQ_LEN

        self.model_config = model_args.model_config
        self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(
                1, 8 if seq_len >= self.MAX_QKV_MM_SEQ_LEN else seq_len // self.tile_size // 8  # 8 rows
            ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=math.ceil(self.qkv_size / self.cluster_shape[1] / 32 / 8),  # N / TILE_WIDTH / grid width
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= self.MAX_QKV_MM_SEQ_LEN,
        )

        assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"
        self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
            (((self.n_kv_heads // self.cluster_shape[1]) * seq_len // (8 * 8)), self.head_dim),
            ttnn.CoreGrid(y=8, x=8),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def map_keys_to_hf_format(self, vision_state_dict):
        # Whole name is start or end of the string or prefixed/suffixed by a dot
        replace_whole_name = lambda pattern, repl: lambda s: re.sub(rf"(^|\.)({pattern})($|\.)", rf"\1{repl}\3", s)
        output = {}
        for k, v in vision_state_dict.items():
            k = replace_whole_name("qkv", "qkv_proj")(k)
            k = replace_whole_name("proj", "o_proj")(k)
            output[k] = v
        return output

    # Device and optimization settings - forwarded from model_args
    def is_distributed_norm(self, mode):
        return False

    def get_model_config(self):
        return self.model_config

    @property
    def tile_padded_batch_rows(self):
        return self.model_args.tile_padded_batch_rows

    @property
    def compute_kernel_config_hifi2(self):
        return self.model_args.compute_kernel_config_hifi2

    @property
    def compute_kernel_config_hifi4(self):
        return self.model_args.compute_kernel_config_hifi4

    @property
    def max_grid_size(self):
        return self.model_args.max_grid_size

    @property
    def tile_size(self):
        return self.model_args.tile_size

    @property
    def max_batch_size(self):
        return self.model_args.max_batch_size

    @property
    def max_seq_len(self):
        return self.model_args.max_seq_len

    @property
    def is_multichip(self):
        return self.model_args.is_multichip

    @property
    def cluster_shape(self):
        return self.model_args.cluster_shape

    @property
    def dummy_weights(self):
        return self.model_args.dummy_weights

    @property
    def num_devices(self):
        return self.model_args.num_devices

    @property
    def is_galaxy(self):
        return self.model_args.is_galaxy

    @property
    def optimizations(self):
        return self.model_args.optimizations

    @property
    def compute_kernel_config_lofi(self):
        return self.model_args.compute_kernel_config_lofi

    @property
    def compute_kernel_config_hifi2_fp16(self):
        return self.model_args.compute_kernel_config_hifi2_fp16

    @property
    def ccl_dtype(self):
        return self.model_args.ccl_dtype

    @property
    def num_reduce_scatter_links(self):
        return self.model_args.num_reduce_scatter_links

    @property
    def num_all_gather_links(self):
        return self.model_args.num_all_gather_links

    @property
    def ccl_topology(self):
        return self.model_args.ccl_topology

    def get_state_dict_prefix(self, module_name, layer_num):
        base_prefix = self.model_args.get_state_dict_prefix(module_name, layer_num)
        return "visual." + base_prefix.replace("layers.", "blocks.")

    @property
    def create_dram_sharded_mem_config(self):
        return self.model_args.create_dram_sharded_mem_config
