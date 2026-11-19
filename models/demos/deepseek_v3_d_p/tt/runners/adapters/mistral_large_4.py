# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mistral Large 4 prefill adapter. Import-light: the device model is imported in build_runtime.

Serves a contiguous run of layers on one rank (all 72 by default). The caches (config "00" kvpe, "01" index) are
allocated for params.num_users slots; mistral_large_4/kv_table.py has their layout.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from models.demos.common.prefill.adapter import KvCaches, PrefillModelAdapter

from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import fabric

CHECKPOINT_DIR = "/mnt/models/mistralai/Mistral-Large-4-1T-A52B"
# most cache tokens (slots x max_seq) one Galaxy holds next to the weights: one slot of 1044480 (204 chunks) with
# SERVE_CONTEXT=long
MAX_CACHE_TOKENS = 1044480


def hf_model_dir(adapter: PrefillModelAdapter) -> str:
    """PREFILL_HF_MODEL, else the adapter's ``hf_model_default``."""
    return os.environ.get("PREFILL_HF_MODEL") or adapter.hf_model_default


class MistralLarge4Config:
    """Static model constants the runner reads off ``model_config``."""

    NUM_LAYERS = 72
    # fabric packet payload the runner opens the mesh with (mistral_large_4/fabric.py); 0 = the fabric's default
    FABRIC_PAYLOAD_SIZE = fabric.payload() or fabric.RUNNER_DEFAULT_PAYLOAD


class MistralLarge4KvCaches(KvCaches):
    """The served caches {"00": kvpe, "01": index} (mistral_large_4/kv_table.py)."""

    def __init__(self, kvpe, index):
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import kv_table as KT

        self._caches = {KT.KV: kvpe, KT.INDEX: index}

    def tensor(self, name: str):
        return self._caches[name]

    def tensors(self) -> dict:
        return dict(self._caches)


class _ServeSpec:
    """The spec fields the device model's builder (mistral_large_4/hooks.py) reads, from the runner's params."""

    def __init__(self, hf_dir: str, max_seq: int, chunk: int):
        from pathlib import Path

        self.hf_dir = Path(hf_dir)
        self._v = {"target.seq": max_seq, "target.chunk": chunk, "reference.weight_dtype": "bfloat16"}

    def get(self, key: str, default=None):
        return self._v.get(key, default)


class MistralLarge4PrefillAdapter(PrefillModelAdapter):
    name = "mistral_large_4"
    model_config = MistralLarge4Config
    hf_model_default = CHECKPOINT_DIR
    ttnn_cache_default = ""  # weights load from the checkpoint each run (no TTNN weight cache)
    prefill_trace_default = ""
    default_gate_mode = "DEVICE_FP32"
    # the model's residual stream [1, 1, S / 8, H / 4] per chip is TP-sharded on the hidden dim
    pipeline_activation_emb_tp_sharded = True
    hf_repo_id = "mistralai/Mistral-Large-4-1T-A52B"

    def hf_model_dir(self) -> str:
        return hf_model_dir(self)

    def load_hf_config(self):
        from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config

        hf_dir = hf_model_dir(self)
        cfg = Config(hf_dir)
        return SimpleNamespace(
            hf_dir=hf_dir, cfg=cfg, hidden_size=cfg.hidden, num_hidden_layers=cfg.n_layers, vocab_size=cfg.vocab
        )

    def weight_cache_path(self, mesh_shape: tuple):
        return None

    def layer_split_boundaries(self, num_layers: int):
        """A rank must start on a full-indexer layer (the shared-index layers reuse the previous full layer's top-k)."""
        from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config

        cfg = Config(hf_model_dir(self))
        return {i for i in range(num_layers) if cfg.is_full(i)}

    def _geometry(self, hf_config, params):
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import kv_table as KT

        _check_params(params)
        layers = range(params.first_layer_idx, params.first_layer_idx + params.num_layers)
        return KT.Geometry.of(
            hf_config.cfg,
            layers,
            sp=params.sp_factor,
            chunk=params.chunk_size,
            max_seq=params.max_seq_len,
            slots=params.num_users,
        )

    def allocate_kv_cache(self, *, mesh_device, hf_config, params) -> MistralLarge4KvCaches:
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import kv_table as KT
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.model import alloc_caches

        geo = self._geometry(hf_config, params)
        kvpe, index = alloc_caches(
            mesh_device, hf_config.cfg, geo.max_seq, geo.regions(KT.KV), geo.regions(KT.INDEX), geo.slots
        )
        return MistralLarge4KvCaches(kvpe, index)

    def build_runtime(self, *, mesh_device, hf_config, params):
        """The runtime over the served layers. The device model is built without caches; the runner allocates them
        afterwards and the runtime binds them before any module is built."""
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.hooks import device_model
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4.runtime import MistralLarge4PrefillRuntime

        geo = self._geometry(hf_config, params)
        spec = _ServeSpec(hf_config.hf_dir, geo.max_seq, geo.chunk)
        model = device_model(
            mesh_device,
            spec,
            list(geo.layers),
            lm_head=False,
            num_users=geo.slots,
            caches={"kvpe": None, "index": None},
        )
        return MistralLarge4PrefillRuntime(mesh_device, model, params, geo)

    def read_slot_kv_and_check_pcc(self, table, device_map, slot_id, real_len, trace_dir, num_layers=None):
        """The producer's read-back of this layout (kv_table.read_slot_kv_and_check_pcc) over the served layers."""
        from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config
        from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import kv_table as KT

        cfg = Config(hf_model_dir(self))
        n = table.config(table.config_id_of(KT.KV)).num_layers if num_layers is None else int(num_layers)
        return KT.read_slot_kv_and_check_pcc(
            table,
            device_map,
            slot_id,
            real_len,
            trace_dir,
            list(range(n)),
            cfg.kv_lora + cfg.qk_rope,
            kv_source=cfg.kv_source,
        )


def _check_params(params):
    if not (params.is_first_rank and params.is_last_rank):
        raise ValueError("mistral_large_4 serves its layers on one rank (no pipeline split)")
    if params.mtp_levels or params.dflash_enabled:
        raise ValueError("mistral_large_4 runs without MTP or DFlash")
    if int(params.max_seq_len) * int(params.num_users) > MAX_CACHE_TOKENS:
        raise ValueError(
            f"{params.num_users} slots x {params.max_seq_len} tokens: one Galaxy holds {MAX_CACHE_TOKENS} cache tokens "
            "(MISTRAL_LARGE_4_SERVE_CONTEXT=long serves one slot of that)"
        )
    if tuple(params.mesh_shape) != (8, 4) or (params.sp_axis, params.tp_axis) != (0, 1):
        raise ValueError(f"mistral_large_4 runs on an [8, 4] mesh (SP axis 0, TP axis 1), got {params.mesh_shape}")
