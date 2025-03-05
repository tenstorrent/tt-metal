from loguru import logger
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

    # Device and optimization settings - forwarded from model_args
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
