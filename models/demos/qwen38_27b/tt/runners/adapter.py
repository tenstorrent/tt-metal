# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B behind tt-metal's prefill runner (models/demos/common/prefill/adapter.py), serving_contract.md
"Adapter and table".

The engine-owned cache (allocate_kv_cache) is the device model's attention KV state, tt/kv_cache.py TtKVCache: bfp8
ND-sharded [1, 1, 32, 256] ROUND_ROBIN_1D records, batch slot * n_attn_layers + attn_layer. The runtime runs the
all-device model (tt/model.py) on it; the Gated DeltaNet state (live state and restart snapshots, not in the KV table)
lives in the runtime's TtModelState bound to that cache at compile.

One rank, mesh 8 x 4, eager (no trace), host-sink acks: sink(global layer, request_id) once per served layer per
chunk, in layer order, after an event on the queue has passed that layer's last op (its KV rows, or conv / recurrent
state and snapshots, are then final in DRAM).

Import-light: no model / reference import at module load (the runner imports the adapter in the serving process).
"""

from __future__ import annotations

from pathlib import Path

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter

HF_SNAPSHOT = "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class Qwen38ServiceConfig:
    NUM_LAYERS = 64
    FABRIC_PAYLOAD_SIZE = 4352  # the 1D fabric's default packet payload, as the bring-up opens the mesh
    MESH_SHAPE = (8, 4)
    CHUNK_SIZE = 5120
    MAX_SEQ_LEN = 56320  # the served cache's length (7040 rows per chip)
    HIDDEN_SIZE = 5120
    NUM_KEY_VALUE_HEADS = 4
    HEAD_DIM = 256


def resolve_hf_dir(path_or_repo: str) -> str:
    """A checkpoint directory, or a HF repo id resolved in the local HF cache."""
    if Path(path_or_repo).is_dir():
        return str(path_or_repo)
    from huggingface_hub import snapshot_download

    return snapshot_download(path_or_repo, local_files_only=True)


def served_layers(params) -> list[int]:
    return list(range(params.first_layer_idx, params.first_layer_idx + params.num_layers))


def cache_max_seq(params) -> int:
    """The cache always has the served length: the ring SDPA's cache read needs more cache rows per chip than one
    chunk's (Q.seq < K.seq per chip), whatever max_seq_len the run asks for."""
    return max(int(params.max_seq_len), Qwen38ServiceConfig.MAX_SEQ_LEN)


def validate_params(params) -> None:
    if tuple(params.mesh_shape) != Qwen38ServiceConfig.MESH_SHAPE or (params.sp_axis, params.tp_axis) != (0, 1):
        raise ValueError(f"qwen38_27b prefill needs mesh {Qwen38ServiceConfig.MESH_SHAPE} (SP axis 0, TP axis 1)")
    if not (params.is_first_rank and params.is_last_rank):
        raise ValueError("qwen38_27b prefill runs on one rank (all served layers on one mesh)")
    if params.chunk_size != Qwen38ServiceConfig.CHUNK_SIZE:
        raise ValueError(f"qwen38_27b prefill needs chunk_size {Qwen38ServiceConfig.CHUNK_SIZE}")
    if params.use_trace or params.dflash_enabled or params.mtp_levels:
        raise ValueError("qwen38_27b prefill runs eager, without DFlash or MTP (PREFILL_USE_TRACE=0)")


class Qwen38KvCaches(KvCaches):
    """The engine-owned attention KV cache of the served layers (cache None when none of them is an attention layer).
    attn_layers: the served attention layers (global ids) in the cache's layer order."""

    def __init__(self, cache, attn_layers: list[int], first_layer: int, num_layers: int):
        self.cache, self.attn_layers = cache, list(attn_layers)
        self.first_layer, self.num_layers = first_layer, num_layers

    def attn_of_table_layer(self) -> dict:
        """{table layer (served layer - first): index in the cache's layer dim} of the attention layers."""
        return {g - self.first_layer: a for a, g in enumerate(self.attn_layers)}


class Qwen38PrefillAdapter(PrefillModelAdapter):
    name = "qwen38_27b"
    model_config = Qwen38ServiceConfig
    hf_model_default = HF_SNAPSHOT
    ttnn_cache_default = ""
    prefill_trace_default = "/mnt/models/agentic-prefill-goldens/Qwen/Qwen3.8-27B/isl56320"
    pipeline_activation_emb_tp_sharded = False  # the residual is replicated over the TP columns
    hf_repo_id = "Qwen/Qwen3.8-27B"

    def hf_dir(self) -> str:
        return resolve_hf_dir(self.hf_model_path())

    def load_hf_config(self):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.hf_dir())
        return getattr(config, "text_config", config)

    def weight_cache_path(self, mesh_shape):
        return None  # weights load from the checkpoint (no TTNN weight cache)

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        validate_params(params)
        from models.demos.qwen38_27b.tt.kv_cache import TtKVCache

        layers = served_layers(params)
        attn = [i for i in layers if hf_config.layer_types[i] == "full_attention"]
        cache = None
        if attn:
            cache = TtKVCache(
                mesh_device,
                hf_config.num_key_value_heads,
                hf_config.head_dim,
                cache_max_seq(params),
                params.chunk_size,
                num_slots=params.num_users,
                num_layers=len(attn),
            )
        return Qwen38KvCaches(cache, attn, params.first_layer_idx, params.num_layers)

    def build_runtime(self, *, mesh_device, hf_config, params):
        validate_params(params)
        from models.demos.qwen38_27b.tt.runners.runtime import Qwen38PrefillRuntime

        return Qwen38PrefillRuntime(mesh_device=mesh_device, hf_dir=self.hf_dir(), config=params)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir):
        """The producer's KV read-back (prefill_producer._read_slot_kv_and_check_pcc dispatches here)."""
        from models.demos.qwen38_27b.tt.runners.kv_table import read_slot_kv_and_check_pcc

        return read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir)


def contract_state_pcc(spec, runtime, kv, slot: int, length: int, golden) -> dict:
    """{"conv", "recurrent": min PCC over the served Gated DeltaNet layers} of slot's live state (at length) vs the
    golden snapshot at length (the fixed-size state is not in the KV table). Harness boundary (host reads)."""
    from models.demos.qwen38_27b.tt.runners.kv_table import _pcc

    state = runtime.state_for(kv)
    out = {}
    for layer in runtime.model.layers:
        if runtime.model.is_attn(layer):
            continue
        pos = state.gdn[(slot, layer)].pos
        if pos != length:
            raise ValueError(f"GDN layer {layer} slot {slot}: the state is at {pos}, not {length}")
        dev = state.to_torch(layer, length, slot=slot)
        gold = golden.state(layer, at=length)
        for name in ("conv", "recurrent"):
            out[name] = min(out.get(name, 1.0), _pcc(dev[name].float(), gold[name].float()))
    return out
