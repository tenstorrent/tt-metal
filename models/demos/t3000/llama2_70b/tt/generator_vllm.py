# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
import json
import torch

from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from models.demos.t3000.llama2_70b.tt.llama_common import (
    load_llama_state_dict,
    setup_llama_env,
    check_mesh_device,
)
from models.demos.t3000.llama2_70b.reference.llama.llama.model import ModelArgs as ReferenceModelArgs


class TtLlamaForCausalLM(TtLlamaModelForGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(cls, hf_config, t3k_mesh_device, max_batch_size):
        # TODO: pass in model args and tt args as parameters from vllm
        @dataclass
        class ModelArgs:
            llama_version: str = None
            ckpt_dir: str = None
            max_batch_size: int = 32  # overwritten by max_num_seqs from vllm
            num_layers: int = 80
            max_kv_context_len: int = 131072

        @dataclass
        class TTArgs:
            mesh_device: object = None
            cache_path: str = None

        # setup configs
        llama_version = "llama3"
        model_config, ckpt_dir, _, cache_path = setup_llama_env(
            llama_version=llama_version,
        )

        check_mesh_device(t3k_mesh_device, model_config)

        # initialize arg classes
        model_args = ModelArgs(llama_version=llama_version, ckpt_dir=ckpt_dir, max_batch_size=max_batch_size)
        tt_args = TTArgs(mesh_device=t3k_mesh_device, cache_path=cache_path)

        # load state dict
        state_dict = load_llama_state_dict(model_args.ckpt_dir, n_layers=model_args.num_layers)

        # TODO: delete this configuration setup once llama can directly accept hf_config

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        configuration = ReferenceModelArgs(
            max_seq_len=model_args.max_kv_context_len,
            max_batch_size=model_args.max_batch_size,
            **params,
        )

        return cls(
            configuration=configuration, state_dict=state_dict, model_args=model_args, tt_args=tt_args, vllm=True
        )

    @property
    def cache_path(self):
        return self.tt_model.cache_path

    def prefill_forward(self, tokens: torch.Tensor, page_table, kv_cache, prompt_lens):
        return super().prefill_forward(tokens, 0, page_table, kv_cache, prompt_lens)
