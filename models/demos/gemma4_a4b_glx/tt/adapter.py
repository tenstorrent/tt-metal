# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""gemma4_a4b_glx (Gemma-4 26B-A4B) behind tt-metal's prefill runner (models/demos/common/prefill/adapter.py).

The KV cache is tt.attention.TtKVCache per layer (the device model's own state); its table is tt.kv_chunk_table.
Import-light: the device stack loads inside the methods.
"""

from __future__ import annotations

import os

from models.demos.common.prefill.adapter import PrefillModelAdapter

HF_MODEL = (
    "/mnt/models/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/20da991ab4afab98e8f910c4a2e8f4fbefc404ad"
)


class Gemma4A4bGlxConfig:
    NUM_LAYERS = 30
    FABRIC_PAYLOAD_SIZE = 8192
    MESH_SHAPE = (8, 4)


def _validate(params):
    if tuple(params.mesh_shape) != Gemma4A4bGlxConfig.MESH_SHAPE or (params.sp_axis, params.tp_axis) != (0, 1):
        raise ValueError(f"gemma4_a4b_glx prefill runs on an SP=8 x TP=4 mesh, got {params.mesh_shape}")
    if params.first_layer_idx != 0 or not (params.is_first_rank and params.is_last_rank):
        raise ValueError("gemma4_a4b_glx prefill runs all its layers on one rank")
    if params.use_trace or params.dflash_enabled or params.mtp_levels:
        raise ValueError("gemma4_a4b_glx prefill runs untraced, without DFlash or MTP")


class Gemma4A4bGlxPrefillAdapter(PrefillModelAdapter):
    name = "gemma4_a4b_glx"
    model_config = Gemma4A4bGlxConfig
    hf_model_default = HF_MODEL
    ttnn_cache_default = ""
    prefill_trace_default = ""
    pipeline_activation_emb_tp_sharded = False

    # hf_model_path (PREFILL_HF_MODEL, else hf_model_default) comes from PrefillModelAdapter.

    def load_hf_config(self):
        from models.demos.gemma4_a4b_glx.reference.gemma4_ref import Gemma4TextConfig

        return Gemma4TextConfig.from_json(os.path.join(self.hf_model_path, "config.json"))

    def weight_cache_path(self, mesh_shape):
        if tuple(mesh_shape) != Gemma4A4bGlxConfig.MESH_SHAPE:
            raise ValueError("gemma4_a4b_glx prefill requires an 8x4 mesh")
        return None  # weights load from the HF checkpoint

    # ---- the table's configs (tt.kv_chunk_table: column c x {K, V} x sub; every record 256 wide)
    def cache_kind(self, config_id: int) -> str:
        return "kv"

    def cache_layer_rows(self, config_id, num_layers):
        if not 0 <= config_id < 16:
            raise ValueError(f"invalid gemma4_a4b_glx cache config {config_id}")
        return {layer: layer for layer in range(num_layers)}

    def cache_head_dim(self, config_id):
        if not 0 <= config_id < 16:
            raise ValueError(f"invalid gemma4_a4b_glx cache config {config_id}")
        return 256

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        _validate(params)
        from models.demos.gemma4_a4b_glx.tt.runtime import Gemma4A4bGlxKvCaches

        return Gemma4A4bGlxKvCaches(
            mesh_device, hf_config, range(params.num_layers), params.num_users, params.max_seq_len
        )

    def build_runtime(self, *, mesh_device, hf_config, params):
        _validate(params)
        from models.demos.gemma4_a4b_glx.tt.runtime import Gemma4A4bGlxPrefillRuntime

        return Gemma4A4bGlxPrefillRuntime(mesh_device=mesh_device, model_path=self.hf_model_path, config=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir):
        """The producer's read-back: every layer's K and V through the table vs the golden's kv_cache/ (min PCC)."""
        from pathlib import Path

        from safetensors.torch import load_file

        from models.demos.gemma4_a4b_glx.tt.kv_chunk_table import kv_pcc_mins, read_slot_kv

        cfg = self.load_hf_config()
        n = table.config(0).num_layers
        got = read_slot_kv(table, device_map, slot_id, real_len, {i: cfg.is_sliding(i) for i in range(n)})

        def golden(i):
            d = load_file(str(Path(trace_dir) / "kv_cache" / f"layer_{i}.safetensors"))
            return {k: d[f"{k}_cache_layer_{i}"] for k in ("key", "value")}

        return kv_pcc_mins(got, golden, real_len)
