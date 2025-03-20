import math
import re
from loguru import logger
import ttnn
from models.tt_transformers.tt.common import nearest_multiple
from models.tt_transformers.tt.model_config import ModelArgs


class VisionModelArgs(ModelArgs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Core dimensions from HF config
        self.dim = self.hf_config.vision_config.hidden_size
        self.unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(  # pad to a tile multiple per device
            self.unpadded_hidden_dim, self.tile_size * self.num_devices
        )
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_heads
        self.n_heads = self.hf_config.vision_config.num_heads
        self.n_kv_heads = self.hf_config.vision_config.num_heads

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size

        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding head dim from {self.head_dim} to {self.padded_head_dim}")

        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / (self.n_kv_heads // self.cluster_shape[1])
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

    def map_keys_to_hf_format(self, vision_state_dict):
        # Whole name is start or end of the string or prefixed/suffixed by a dot
        replace_whole_name = lambda pattern, repl: lambda s: re.sub(rf"(^|\.)({pattern})($|\.)", rf"\1{repl}\3", s)
        output = {}
        for k, v in vision_state_dict.items():
            k = replace_whole_name("qkv", "qkv_proj")(k)
            k = replace_whole_name("proj", "o_proj")(k)
            output[k] = v
        return output

    # Visual model does not use distributed norm for now
    def is_distributed_norm(self, mode):
        return False

    def get_state_dict_prefix(self, module_name, layer_num):
        base_module_name = "Attention" if module_name == "VisionAttention" else module_name
        base_prefix = super().get_state_dict_prefix(base_module_name, layer_num)
        return "visual." + base_prefix.replace("layers.", "blocks.")
