# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B (hybrid Gated DeltaNet + gated full attention) behind tt-metal's prefill runner
(models/demos/common/prefill/adapter.py; serving_contract.md "Adapter and table").

One rank, an 8x4 mesh (SP 8 on axis 0, TP 4 on axis 1), eager. The engine owns the serving state
(tt/runners/kv_cache.py): the served full-attention KV cache, which the address table publishes, and the linear
layers' conv / recurrent state, which it does not (fixed-size state is out of this bring-up's migration). Every served
layer acks once per chunk, linear layers included.

Import-light: nothing device-side is imported at module load.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from models.demos.common.prefill.adapter import PrefillModelAdapter

from ..settings import settings

RECORD_TOKENS = 32


class Qwen38ServiceConfig:
    NUM_LAYERS = 64
    MESH_SHAPE = (8, 4)
    # The bring-up tests open the fabric with the default packet payload; the runner opens it with this one.
    FABRIC_PAYLOAD_SIZE = 4352
    NUM_KEY_VALUE_HEADS = 4
    HEAD_DIM = 256


def text_config(hf_dir) -> dict:
    raw = json.loads((Path(hf_dir) / "config.json").read_text())
    return raw.get("text_config", raw)


def _pcc(a, b) -> float:
    import torch

    a, b = a.double().flatten(), b.double().flatten()
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else 0.0


class Qwen38PrefillAdapter(PrefillModelAdapter):
    name = "qwen38_27b"
    model_config = Qwen38ServiceConfig
    hf_model_default = settings.table["HF_DIR"].default  # the runner prints it; QWEN38_27B_HF_DIR overrides
    ttnn_cache_default = ""
    prefill_trace_default = ""
    pipeline_activation_emb_tp_sharded = False

    def load_hf_config(self):
        hf_dir = settings.get("HF_DIR")
        cfg = SimpleNamespace(**text_config(hf_dir))
        cfg.hf_dir = hf_dir
        return cfg

    def weight_cache_path(self, mesh_shape):
        return None  # weights load from the checkpoint (bfloat8_b at load time)

    def cache_kind(self, config_id: int) -> str:
        n = Qwen38ServiceConfig.NUM_KEY_VALUE_HEADS
        return "gqa_key" if config_id < n else "gqa_value"

    def layer_split_boundaries(self, num_layers: int):
        return {0}  # one rank

    def _validate(self, params):
        if tuple(params.mesh_shape) != Qwen38ServiceConfig.MESH_SHAPE or (params.sp_axis, params.tp_axis) != (0, 1):
            raise ValueError(f"Qwen3.8-27B prefill runs an 8x4 mesh (SP axis 0, TP axis 1), got {params.mesh_shape}")
        if not (params.is_first_rank and params.is_last_rank):
            raise ValueError("Qwen3.8-27B prefill serves one rank")
        if params.use_trace:
            raise ValueError("Qwen3.8-27B prefill runs eager: PREFILL_USE_TRACE=0")
        if params.max_seq_len % params.chunk_size:
            raise ValueError(f"max_seq_len {params.max_seq_len} is not a multiple of the chunk {params.chunk_size}")

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        self._validate(params)
        from models.demos.qwen38_27b.reference.model import Config
        from models.demos.qwen38_27b.tt.runners.kv_cache import allocate

        layers = range(params.first_layer_idx, params.first_layer_idx + params.num_layers)
        return allocate(
            mesh_device, Config(hf_config.hf_dir), layers, params.num_users, params.chunk_size, params.max_seq_len
        )

    def build_runtime(self, *, mesh_device, hf_config, params):
        self._validate(params)
        from models.demos.qwen38_27b.tt.runners.runtime import Qwen38PrefillRuntime

        return Qwen38PrefillRuntime(mesh_device=mesh_device, hf_dir=hf_config.hf_dir, config=params)

    # ---- producer read-back of this layout (prefill_producer._read_slot_kv_and_check_pcc)
    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers):
        """{"k": min PCC, "v": min PCC} of slot ``slot_id``'s KV [0, real_len) read through the table, vs the golden
        trace's ``kv_cache/layer_<i>.safetensors`` (key / value [n_kv, seq, head_dim], natural order). Table layer i is
        global layer i (one rank from layer 0); layers the table leaves unset (linear attention) are skipped."""
        from safetensors import safe_open

        from models.demos.common.prefill.runners import prefill_producer as producer

        n_kv, hd = Qwen38ServiceConfig.NUM_KEY_VALUE_HEADS, Qwen38ServiceConfig.HEAD_DIM
        read_len = -(-real_len // RECORD_TOKENS) * RECORD_TOKENS
        mins, checked = {"k": 1.0, "v": 1.0}, 0
        for layer in range(num_layers):
            if not int(table.lookup(layer, 0, slot_id, 0).size_bytes):
                continue
            checked += 1
            path = Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"
            with safe_open(str(path), framework="pt") as f:
                gold = {
                    "k": f.get_tensor(f"key_cache_layer_{layer}").float(),
                    "v": f.get_tensor(f"value_cache_layer_{layer}").float(),
                }
            for kind, base in (("k", 0), ("v", n_kv)):
                for h in range(n_kv):
                    dev = producer._read_kv_slice(
                        table, device_map, base + h, layer, slot_id, read_len, hd, producer._decode_bfp8_chunk
                    )[:real_len]
                    p = _pcc(dev.float(), gold[kind].reshape(n_kv, -1, hd)[h, :real_len])
                    mins[kind] = min(mins[kind], p)
        if not checked:
            raise ValueError(f"no attention layer among the table's {num_layers} layers to read back")
        return mins
