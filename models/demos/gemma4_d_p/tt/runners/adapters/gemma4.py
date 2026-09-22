# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4-31B-it integration with the shared prefill service."""

import os

from models.demos.common.prefill.adapter import PrefillModelAdapter


class Gemma4ServiceConfig:
    NUM_LAYERS = 60
    FABRIC_PAYLOAD_SIZE = 8192
    MESH_SHAPE = (8, 4)
    CHUNK_SIZE = 8192
    MAX_SEQ_LEN = 262144
    MAX_USER_SLOTS = 6


def validate_params(params):
    expected = {
        "mesh_shape": Gemma4ServiceConfig.MESH_SHAPE,
        "num_layers": Gemma4ServiceConfig.NUM_LAYERS,
        "first_layer_idx": 0,
        "is_first_rank": True,
        "is_last_rank": True,
        "max_seq_len": Gemma4ServiceConfig.MAX_SEQ_LEN,
        "chunk_size": Gemma4ServiceConfig.CHUNK_SIZE,
        "sp_axis": 0,
        "tp_axis": 1,
        "use_trace": True,
        "dflash_enabled": False,
    }
    for name, value in expected.items():
        if getattr(params, name) != value:
            raise ValueError(f"Gemma4 prefill requires {name}={value}, got {getattr(params, name)}")
    if not 1 <= params.num_users <= Gemma4ServiceConfig.MAX_USER_SLOTS:
        raise ValueError(f"Gemma4 prefill requires 1 to {Gemma4ServiceConfig.MAX_USER_SLOTS} KV slots")


class Gemma4PrefillAdapter(PrefillModelAdapter):
    name = "gemma4_d_p"
    model_config = Gemma4ServiceConfig
    hf_model_default = "google/gemma-4-31B-it"
    ttnn_cache_default = ""
    prefill_trace_default = ""
    pipeline_activation_emb_tp_sharded = False

    @property
    def hf_model_id(self):
        return self.hf_model_default

    def load_hf_config(self):
        from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs, validate_31b_config

        config = Gemma4ModelArgs.load_hf_config(self.hf_model_id)
        config = getattr(config, "text_config", config)
        validate_31b_config(config)
        return config

    @property
    def tt_cache_path(self):
        return os.getenv("PREFILL_TTNN_CACHE") or os.getenv("TT_CACHE_PATH")

    def weight_cache_path(self, mesh_shape):
        if tuple(mesh_shape) != Gemma4ServiceConfig.MESH_SHAPE:
            raise ValueError("Gemma4 prefill requires an 8x4 mesh")
        import ttnn
        from models.demos.gemma4_d_p.tt.model_config import resolve_cache_dir_from_tt_cache_path

        return resolve_cache_dir_from_tt_cache_path(
            self.tt_cache_path,
            dtype=ttnn.bfloat16,
            mesh_shape=mesh_shape,
        )

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        validate_params(params)
        from models.demos.gemma4_d_p.config import MeshConfig
        from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
        from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches

        return allocate_ring_kv_caches(
            MeshConfig(mesh_device),
            Gemma4ModelArgs.from_hf_config(hf_config),
            num_users=params.num_users,
            max_seq_len=params.max_seq_len,
            prefill_chunk_size=params.chunk_size,
        )

    def build_runtime(self, *, mesh_device, hf_config, params):
        validate_params(params)
        from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime

        return Gemma4PrefillRuntime(
            mesh_device=mesh_device, hf_model_id=self.hf_model_id, tt_cache_path=self.tt_cache_path, config=params
        )
