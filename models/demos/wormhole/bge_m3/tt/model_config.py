# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import os

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0


class ModelArgs:
    """
    BGE-M3 TTv2 model loading contract.

    Canonical flow:
        args = ModelArgsTTv2(...)
        state_dict = args.load_state_dict()
        model = args.load_model(state_dict=state_dict)
    """

    def __init__(
        self,
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=8192,
        cache_hf=False,
        hf_model_name=None,
    ):
        super().__init__()
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.arch_name = ttnn.get_arch_name()
        self.dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None

        self.device_name = determine_device_name(self.mesh_device)

        self.cluster_shape = list(self.mesh_device.shape) if self.mesh_device is not None else None
        self.is_galaxy = self.num_devices == 32

        self.tile_size = 32
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.prefill_len_cutoff = 512 if is_blackhole() else 1024

        self.dummy_weights = dummy_weights
        self.cache_hf = cache_hf

        self.hf_model_name = hf_model_name if hf_model_name else os.getenv("HF_MODEL")

        if self.hf_model_name:
            self.model_name = self.hf_model_name.strip("/").split("/")[-1]
        else:
            raise ValueError("No HF model name provided.")

        self.from_hf_url = True if self.hf_model_name else False
        self.tokenizer = self.create_tokenizer()
        self._set_hf_params()

    def ccl_topology(self):
        pass

    def _resolve_checkpoint(self, checkpoint_dir=None):
        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for bge-m3 models for now.")

        ckp_point = ""
        if self.from_hf_url:
            ckp_point = self.hf_model_name
        elif checkpoint_dir:
            ckp_point = checkpoint_dir
        return ckp_point

    def _set_params_from_dict(self, config):
        eos_token_id = config.get("eos_token_id", None)
        self.eos_token_id = None if isinstance(eos_token_id, int) else eos_token_id

        self.dim = config.get("dim", config.get("hidden_size"))
        self.n_heads = config.get("n_heads", config.get("num_attention_heads"))
        self.n_kv_heads = config.get("n_kv_heads", config.get("num_key_value_heads"))
        self.n_layers = config.get("n_layers", config.get("num_hidden_layers"))
        self.full_model_n_layers = self.n_layers
        self.norm_eps = config.get("norm_eps", config.get("layer_norm_eps"))
        self.vocab_size = config["vocab_size"]

        self.head_dim = config.get("head_dim", self.dim // self.n_heads)
        self.max_context_len = config.get("max_position_embeddings")

        if "intermediate_size" in config:
            self.intermediate_size = config["intermediate_size"]
        else:
            # TODO: Implement manual calculation of intermediate_size.
            raise ValueError(
                "intermediate_size not found in config. Please implement manual calculation of intermediate_size."
            )

        self.layer_types = config.get("layer_types", None)

        self.mlp_activation_type = config.get("hidden_act")

        self.pad_token_id = config.get("pad_token_id", None)

    def _set_hf_params(self, checkpoint_dir=None):
        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for bge-m3 models for now.")

        ckp_point = self._resolve_checkpoint(checkpoint_dir)
        if not ckp_point:
            raise ValueError("No checkpoint directory or HF URL provided.")

        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(ckp_point).to_dict()
        self._set_params_from_dict(self.hf_config)

    def _set_model_params(self, checkpoint):
        import json

        params_file = os.path.join(checkpoint, "params.json")
        assert os.path.exists(params_file), f"params.json file not found at {params_file}"
        with open(params_file, "r") as f:
            params = json.load(f)
        self._set_params_from_dict(params)

    def load_state_dict(self):
        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for bge-m3 models for now.")

        ckp_point = self._resolve_checkpoint()
        if not ckp_point:
            raise ValueError("No checkpoint directory or HF URL provided.")

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(ckp_point, torch_dtype="auto")
        state_dict = model.state_dict()

        # TODO: do layer checks here.

        return state_dict

    def create_tokenizer(self):
        ckp_point = self._resolve_checkpoint()
        if not ckp_point:
            raise ValueError("No checkpoint directory or HF URL provided.")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ckp_point)

        return tokenizer

    # def weight_cache_path(self, dtype: ttnn.DataType) -> Path:
    #     cache_root_env = os.getenv("TT_CACHE_PATH")
    #     if cache_root_env:
    #         cache_root = Path(cache_root_env)
    #     else:
    #         model_path = Path(self.model_path)
    #         if model_path.exists():
    #             cache_root = model_path
    #         else:
    #             cache_root = Path.home() / ".cache" / "tt-metal" / "bge-m3"

    #     dtype_name = _dtype_name(dtype)
    #     cache_path = cache_root / f"tensor_cache_{dtype_name}"
    #     cache_path.mkdir(parents=True, exist_ok=True)
    #     self.resolved_weight_cache_path = cache_path
    #     return cache_path

    def num_to_corerange(x):
        assert x < 8 or x % 8 == 0
        num_x = min(x, 8)
        num_y = x // num_x
        assert num_x * num_y == x
        return ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(num_x - 1, num_y - 1),
        )

    def encode_prompts(self, prompts: list[str] | str) -> ttnn.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_seq_len,
        )
        max_prompt_length = max(len(prompt_input_ids) for prompt_input_ids in tokenized["input_ids"])
        pad_multiple = 2048 if max_prompt_length > 2048 else 128
        padded_length = ((max_prompt_length + pad_multiple - 1) // pad_multiple) * pad_multiple

        if padded_length > self.max_seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} is too small for padded prompt length {padded_length}. "
                "Increase max_seq_len or use shorter prompts."
            )

        return self.tokenizer.pad(
            tokenized,
            padding="max_length",
            max_length=padded_length,
            return_tensors="pt",
        )


def num_to_coregrid(x):
    if x % 8 == 0:
        return ttnn.CoreGrid(y=x // 8, x=8)
    if x == 12:
        return ttnn.CoreGrid(y=2, x=6)
    if x == 20:
        return ttnn.CoreGrid(y=4, x=5)


def determine_device_name(mesh_device):
    """
    Determine device name based on number of devices and architecture.

    Args:
        mesh_device (MeshDevice): MeshDevice object

    Returns:
        str: Device name (e.g., "CPU", "N150", "P100", etc.)

    Raises:
        ValueError: If architecture or device count is unsupported
    """
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None  # CoreCoord with (x, y)

    if num_devices == 0:
        return "CPU"

    if is_blackhole():
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",  # P100 DRAM grid is 7x1, P150 is 8x1
            2: "P300",
            4: "P150x4",
            8: "P150x8",
        }
    elif is_wormhole_b0():
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices in dict_device_names:
        return dict_device_names[num_devices]
    else:
        raise ValueError(f"Unsupported number of devices: {num_devices} for {arch_name}")


# class ModelOptimizations:
#     def __init__(self):

#         core_grid_8x8 = ttnn.CoreGrid(y=8, x=8)

#         self.BGE_L1_SMALL_SIZE = 0
#         self.BGE_SEQ_LENGTH = 8192

#         seqL = self.BGE_SEQ_LENGTH
#         if seqL <= self.BGE_SEQ_LENGTH:
#             seqL_factor = 1
#         else:
#             seqL_factor = 2

#         self.TILE_HEIGHT = 32
#         seqL_padded = (((seqL - 1) // self.TILE_HEIGHT) + 1) * self.TILE_HEIGHT
#         seqL_t = seqL_padded // self.TILE_HEIGHT  # 12
#         dim_t = 1024 // self.TILE_HEIGHT  # 32
#         dim_t__x = dim_t // core_grid_8x8.x  # 4
#         head_num = 16
#         head_seqL_t__x = (head_num * seqL_t) // core_grid_8x8.x  # 24
#         head_size_t = dim_t // head_num  # 2

#         self.layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
#             compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
#             subblock_w=dim_t__x,  # Revert to 4
#             block_h=seqL_t,  # Keep same as sentence_bert for seq_len handling
#             block_w=dim_t__x,  # 1024 / 32 / 8 = 4
#             inplace=True,
#             legacy_reduction=True,
#             legacy_rsqrt=True,
#         )

#         self.ff1_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#             compute_with_storage_grid_size=(self.core_grid_8x8.x, self.core_grid_8x8.y),
#             in0_block_w=dim_t__x,  # Keep 4 (Kt=32, per_core=4, max is 4)
#             out_subblock_h=1,
#             out_subblock_w=dim_t__x * 2,  # Keep 8 (no FP32 accumulation, max for BF16 is 8)
#             per_core_M=seqL_t,
#             per_core_N=dim_t__x * 4,
#             transpose_mcast=False,
#             fused_activation=(ttnn.UnaryOpType.GELU, True),
#         )

#         self.ff2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#             compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
#             in0_block_w=(dim_t__x * 4) // seqL_factor,  # Increase from 8 to 16 for better reuse (4096 / 32 / 8 = 16)
#             out_subblock_h=1,  # Keep 1 (per_core_M=12, so 1 divides evenly)
#             out_subblock_w=dim_t__x // seqL_factor,  # Try 4 (max for FP32: 1*4=4, divides per_core_N=4)
#             per_core_M=seqL_t,
#             per_core_N=dim_t__x,  # Calculated: ceil(1024 / (32 * 8)) = 4
#             transpose_mcast=False,
#             fused_activation=None,
#         )

#         self.query_key_value_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#             compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
#             in0_block_w=dim_t__x // seqL_factor,  # Revert to 4 (must divide K evenly: 1024/32=32, 32/8=4)
#             out_subblock_h=1,
#             out_subblock_w=dim_t__x // seqL_factor,  # Keep 4 for FP32 accumulation (1*4=4 <= 4, max for FP32)
#             per_core_M=seqL_t,
#             per_core_N=dim_t__x * 3,
#             transpose_mcast=False,
#             fused_activation=None,
#         )

#         self.self_out_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#             compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
#             in0_block_w=dim_t__x,
#             out_subblock_h=1,  # Restore to 2 (no FP32 accumulation on self-output)
#             out_subblock_w=dim_t__x,  # Restore to 4
#             per_core_M=seqL_t,
#             per_core_N=dim_t__x,
#             transpose_mcast=False,
#             fused_activation=None,
#         )

#         self.pre_softmax_config = ttnn.MatmulMultiCoreReuseProgramConfig(
#             compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
#             in0_block_w=head_size_t,
#             out_subblock_h=1,
#             out_subblock_w=seqL_t // 2,  # Keep 6 (best balance between performance and PCC)
#             per_core_M=head_seqL_t__x,
#             per_core_N=seqL_t,
#         )

#         self.softmax_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
#             compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
#             subblock_w=seqL_t,  # Revert to 6 (best for accuracy)
#             block_h=head_seqL_t__x,  # Keep same for seq_len handling
#             block_w=seqL_t,  # Keep same for seq_len handling (384 / 32 / 8 = 1.5, but using 12 for compatibility)
#         )
