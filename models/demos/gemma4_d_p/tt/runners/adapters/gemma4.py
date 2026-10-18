# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4-31B-it integration with the shared prefill service."""

import os

from models.demos.common.prefill.adapter import PrefillModelAdapter


class Gemma4ServiceConfig:
    NUM_LAYERS = 60
    FABRIC_PAYLOAD_SIZE = 8192
    MESH_SHAPE = (8, 4)
    # Validation runs may set GEMMA4_TEST_CHUNK_SIZE to test another chunk size.
    CHUNK_SIZE = int(os.environ.get("GEMMA4_TEST_CHUNK_SIZE", 8192))
    MAX_SEQ_LEN = 262144
    MAX_USER_SLOTS = 6


def is_global_layer_by_index(layer_idx):
    return layer_idx % 6 == 5


def validate_params(params):
    expected = {
        "mesh_shape": Gemma4ServiceConfig.MESH_SHAPE,
        "num_layers": Gemma4ServiceConfig.NUM_LAYERS,
        "first_layer_idx": 0,
        "is_first_rank": True,
        "is_last_rank": True,
        "max_seq_len": Gemma4ServiceConfig.MAX_SEQ_LEN,
        "sp_axis": 0,
        "tp_axis": 1,
        "use_trace": True,
        "dflash_enabled": False,
    }
    for name, value in expected.items():
        if getattr(params, name) != value:
            raise ValueError(f"Gemma4 prefill requires {name}={value}, got {getattr(params, name)}")
    if params.chunk_size <= 0 or params.chunk_size % (8 * 4 * 32) or params.chunk_size > params.max_seq_len:
        raise ValueError("Gemma4 packed chunk size must be a positive multiple of 1024 within the context limit")
    if not 1 <= params.num_users <= Gemma4ServiceConfig.MAX_USER_SLOTS:
        raise ValueError(f"Gemma4 prefill requires 1 to {Gemma4ServiceConfig.MAX_USER_SLOTS} KV slots")


class Gemma4PrefillAdapter(PrefillModelAdapter):
    name = "gemma4_d_p"
    model_config = Gemma4ServiceConfig
    hf_model_default = "google/gemma-4-31B-it"
    ttnn_cache_default = ""
    prefill_trace_default = "/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135"
    pipeline_activation_emb_tp_sharded = False

    def prefill_metadata_size_bytes(self, chunk_size, cp, num_users):
        from models.demos.gemma4_d_p.tt.runners.packing import metadata_size_bytes

        return metadata_size_bytes(chunk_size, cp, num_users)

    def decode_prefill_metadata(self, words):
        from models.demos.gemma4_d_p.tt.runners.packing import decode_metadata

        segments = decode_metadata([int(v) & 0xFFFFFFFF for v in words])
        if segments is None:
            return dict(slot_id=-1, actual_start=-1, actual_end=-1)
        first = segments[0]
        return dict(
            slot_id=first.slot_id, actual_start=first.actual_start, actual_end=first.actual_end, segments=segments
        )

    def create_prefill_sender(self, send, *, chunk_size, cp, num_users, max_seq_len):
        from models.demos.gemma4_d_p.tt.runners.packing import PackedPrefillSender

        return PackedPrefillSender(send, chunk_size=chunk_size, cp=cp, num_users=num_users, max_seq_len=max_seq_len)

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

    def cache_layer_rows(self, config_id, num_layers):
        if not 0 <= config_id < 36:
            raise ValueError(f"Invalid Gemma4 cache config {config_id}")
        return {layer: layer for layer in range(num_layers) if is_global_layer_by_index(layer) == (config_id < 4)}

    def cache_head_dim(self, config_id):
        if not 0 <= config_id < 36:
            raise ValueError(f"Invalid Gemma4 cache config {config_id}")
        return 640 if config_id < 4 else 256

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
