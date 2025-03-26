# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.llama3_subdevices.tt.generator import LlamaGenerator
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.model_config import LlamaOptimizations, TtModelArgs


class TtLlamaForCausalLM(LlamaGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, n_layers=None):
        instruct_mode = "Instruct" in hf_config._name_or_path
        max_seq_len = 131072  # TODO: modify this for different models/devices
        optimizations = LlamaOptimizations.performance  # TODO: maybe change to accuracy
        dtype = ttnn.bfloat8_b

        # Load model args, weights
        model_args = TtModelArgs(
            mesh_device,
            instruct=instruct_mode,
            max_batch_size=max_batch_size,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
        )
        if n_layers is not None:
            model_args.n_layers = n_layers
        state_dict = model_args.load_state_dict()

        tt_model = TtTransformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            use_paged_kv_cache=True,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)
