# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from ttnn.device import is_blackhole as ttnn_is_blackhole
from ttnn.device import is_wormhole_b0 as ttnn_is_wormhole_b0

import ttnn


class ModelArgs:
    """
    BGE-M3 model loading contract (HF checkpoint, tokenizer, and tensor layout).

    Typical usage passes this object into the TT encoder after ``load_state_dict``.
    """

    def __init__(
        self,
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=8192,
        cache_hf=False,
        hf_model_name=None,
        dtype=ttnn.bfloat16,
        data_parallel=False,
        use_experimental_encoder_sdpa=False,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.device_name, self.num_devices = determine_device_name(self.mesh_device)
        self.arch_name = ttnn.get_arch_name()
        self.dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None
        self.grid_size = mesh_device.compute_with_storage_grid_size() if mesh_device else None

        self.cluster_shape = list(self.mesh_device.shape) if self.mesh_device is not None else None
        self.is_galaxy = self.num_devices == 32

        self.tile_size = 32
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        # Data-parallel serving path (DP=2): batch B12 split to B6 per chip on a
        # 1x2 N300, each chip an independent full-sequence replica with no
        # collectives. Requested via the data_parallel argument; only valid on
        # the 2x1 / S8192 serving shape.
        if data_parallel and not (
            self.max_seq_len == 8192
            and mesh_device is not None
            and self.num_devices == 2
            and tuple(mesh_device.shape) == (2, 1)
        ):
            raise ValueError(
                "data_parallel=True requires max_seq_len=8192 on a (2, 1) 2-device mesh, "
                f"got seq_len={self.max_seq_len}, shape={tuple(mesh_device.shape) if mesh_device else None}"
            )
        self.data_parallel = data_parallel
        # Opt-in model-local JIT encoder SDPA (DP S8192 path only). Runs the
        # non-FP32-dest / half-sync configuration (DEST=8), measured -2.3ms/SDPA
        # call and -57ms full-model wall vs stock, with equal-or-better PCC.
        # Only takes effect on the exact head-folded DP S8192 contract; any
        # deviation falls back to stock SDPA (see attention.py guard).
        self.use_experimental_encoder_sdpa = use_experimental_encoder_sdpa
        self.attention_mask_dtype = (
            dtype if self.max_seq_len == 512 and max(1, int(self.max_batch_size)) in (1, 32) else ttnn.bfloat16
        )
        self.prefill_len_cutoff = 512 if ttnn_is_blackhole(mesh_device) else 1024

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

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        from torch import load as torch_load
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(ckp_point, torch_dtype="auto")

        state_dict = model.state_dict()

        files_to_check = ["colbert_linear.pt", "sparse_linear.pt"]
        download_repo_id = self.hf_model_name if os.path.isdir(ckp_point) else ckp_point

        for file_name in files_to_check:
            if os.path.isdir(ckp_point):
                local_file_path = os.path.join(ckp_point, file_name)
                if os.path.exists(local_file_path):
                    file_path = local_file_path
                elif download_repo_id:
                    try:
                        file_path = hf_hub_download(
                            repo_id=download_repo_id,
                            filename=file_name,
                            local_files_only=True,
                        )
                    except LocalEntryNotFoundError:
                        file_path = hf_hub_download(repo_id=download_repo_id, filename=file_name)
                else:
                    raise FileNotFoundError(f"Missing required file '{file_name}' in local checkpoint '{ckp_point}'")
            else:
                try:
                    file_path = hf_hub_download(
                        repo_id=ckp_point,
                        filename=file_name,
                        local_files_only=True,
                    )
                except LocalEntryNotFoundError:
                    file_path = hf_hub_download(repo_id=ckp_point, filename=file_name)

            model_weights = torch_load(file_path, map_location="cpu")
            name = file_name.split(".")[0]
            state_dict[f"{name}.weight"] = model_weights["weight"]
            state_dict[f"{name}.bias"] = model_weights["bias"]

        return state_dict

    def create_tokenizer(self):
        ckp_point = self._resolve_checkpoint()
        if not ckp_point:
            raise ValueError("No checkpoint directory or HF URL provided.")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ckp_point)

        return tokenizer

    def num_to_corerange(x):
        assert x < 8 or x % 8 == 0
        num_x = min(x, 8)
        num_y = x // num_x
        assert num_x * num_y == x
        return ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(num_x - 1, num_y - 1),
        )

    def encode_prompts(
        self,
        prompts: list[str] | str,
        prompt_length: int | None = None,
        *,
        attention_mask_4d: bool = True,
        inputs_mesh_mapper: ttnn.TensorToMesh | None = None,
    ) -> ttnn.Tensor:
        """Tokenize ``prompts`` and build BGE-M3 model inputs.

        ``attention_mask_4d`` (default True) controls the shape of the
        returned ``attention_mask``:
          * True — SDPA-ready 4D additive mask ``[B, 1, S, S]`` (model
            consumes it directly without rebuilding from a 2D keep-mask).
          * False — raw 2D boolean keep-mask ``[B, S]`` (HF convention,
            same as ``tokenizer_attention_mask``). Use this when callers
            expect a 2D mask, e.g. ``BgeM3ForEmbedding._pad_inputs`` /
            pooling helpers.

        ``tokenizer_attention_mask`` is always populated with the raw 2D
        keep-mask regardless of this flag.

        ``inputs_mesh_mapper`` (default None) controls how device inputs are
        distributed across a multi-device mesh:
          * None — ttnn default (replicate to every chip). Use for single
            device or when every chip processes the same batch.
          * ``ttnn.ShardTensorToMesh(mesh_device, dim=0)`` — shard the global
            batch along dim 0 across the mesh (data parallel). Built via
            ``models.demos.utils.common_demo_utils.get_mesh_mappers(device)``.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_seq_len,
        )
        max_prompt_length = max(len(prompt_input_ids) for prompt_input_ids in tokenized["input_ids"])
        padded_length = (
            int(prompt_length) if prompt_length is not None else get_padded_sequence_length(max_prompt_length)
        )

        if padded_length < max_prompt_length:
            raise ValueError(
                f"prompt_length={padded_length} is shorter than tokenized prompt length {max_prompt_length}. "
                "Increase prompt_length or use a shorter prompt."
            )

        if padded_length > self.max_seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} is too small for padded prompt length {padded_length}. "
                "Increase max_seq_len or use shorter prompts."
            )

        encoded = self.tokenizer.pad(
            tokenized,
            padding="max_length",
            max_length=padded_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = input_ids.new_zeros(input_ids.shape)
        encoded["tokenizer_attention_mask"] = encoded["attention_mask"]

        if attention_mask_4d:
            keep = encoded["tokenizer_attention_mask"].bfloat16()
            additive = (1.0 - keep) * -100000.0
            encoded["attention_mask"] = (
                additive.unsqueeze(1).unsqueeze(1).expand(-1, -1, padded_length, -1).contiguous()
            )

        mask = input_ids.ne(int(self.pad_token_id)).to(dtype=input_ids.dtype)
        incremental_indices = mask.cumsum(dim=1) * mask
        encoded["position_ids"] = (incremental_indices + int(self.pad_token_id)).to(dtype=input_ids.dtype)

        if self.mesh_device is not None:
            encoded["model_inputs"] = {
                "input_ids": ttnn.from_torch(
                    encoded["input_ids"].int(),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=inputs_mesh_mapper,
                ),
                "attention_mask": (
                    ttnn.from_torch(
                        encoded["attention_mask"].bfloat16(),
                        device=self.mesh_device,
                        dtype=self.attention_mask_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=inputs_mesh_mapper,
                    )
                    if attention_mask_4d
                    else ttnn.from_torch(
                        encoded["attention_mask"].int(),
                        device=self.mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=inputs_mesh_mapper,
                    )
                ),
                "token_type_ids": ttnn.from_torch(
                    encoded["token_type_ids"].int(),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=inputs_mesh_mapper,
                ),
                "position_ids": ttnn.from_torch(
                    encoded["position_ids"].int(),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=inputs_mesh_mapper,
                ),
            }
        return encoded


def get_padded_sequence_length(seq_len: int) -> int:
    # Attention requires seq_len % 32 (tile height); for seq_len > 128 it must be % 128
    # (see ``BgeM3Attention.forward``). Padding **≤128** to 32-token steps avoids forcing
    # e.g. 32→128 (4× wasted device work) while keeping alignment; above 128, keep 128-wide steps.
    # Long-sequence paths: 1024 / 2048 alignment for large kernels.
    if seq_len <= 1024:
        pad_multiple = 32 if seq_len <= 128 else 128
    elif seq_len <= 2048:
        pad_multiple = 1024
    else:
        pad_multiple = 2048

    return ((seq_len + pad_multiple - 1) // pad_multiple) * pad_multiple


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
        tuple[str, int]: Device name and number of devices.

    Raises:
        ValueError: If architecture or device count is unsupported
    """
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None  # CoreCoord with (x, y)

    if num_devices == 0:
        return "CPU", num_devices

    if ttnn_is_blackhole(mesh_device):
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",  # P100 DRAM grid is 7x1, P150 is 8x1
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "P150x32",  # Blackhole Galaxy
        }
    elif ttnn_is_wormhole_b0(mesh_device):
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
        return dict_device_names[num_devices], num_devices
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
