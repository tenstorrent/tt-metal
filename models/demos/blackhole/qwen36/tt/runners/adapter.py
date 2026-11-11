# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``Qwen35PupPrefillAdapter``: the FLUX.3 Klein prompt upsampler (Qwen3.5-9B VLM) as a tt-d-gen prefill runner model.

Plugs into ``models/demos/common/prefill/runners/prefill_runner.py`` (``PREFILL_MODEL=qwen35_9b_pup``, manifest
``manifests/qwen35_9b_pup.json``). The adapter is a factory + descriptor; the runtime (``runtime.py``) runs chunks and
publishes state. Precedents: MiniMax-M3 (``minimax_m3/tt/runners/adapters/minimax_m3.py``: GQA K/V per TP head,
media spliced from the embedding store, tt-metal #64000) and Kimi-K3 (``deepseek_v3_d_p/.../kimi_k3.py``: linear-
attention state as Snapshot table configs, #63958).

What it publishes (``kv_table.py``): the 8 attention layers' K / V (configs 00-07, Kv) and the 24 GDN layers'
recurrent + convolution state (configs 08 / 09, Snapshot). Nothing else: no logits / first token (decode computes the
last prompt token), no hidden state (one rank).

Env read here (the runner's convention, not model knobs): ``PREFILL_HF_MODEL`` / ``HF_MODEL`` (checkpoint dir),
``TT_CACHE_PATH`` (weight cache root, read by ``Qwen36ModelArgs``). Every model setting is ``config.PupPrefillConfig``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from models.demos.common.prefill.adapter import PrefillModelAdapter, PrefillRunParams
from models.demos.blackhole.qwen36.tt.runners.config import DEFAULT_CONFIG, PupPrefillConfig

_DEFAULT_CKPT = (
    "/mnt/models/huggingface/hub/models--black-forest-labs--flux-3-image-klein-pup/snapshots/"
    "3f3edb52b2c55de0d49feab28598b68f7f8e4830"
)
_FULL_ATTENTION_INTERVAL = 4  # Qwen3.5: every 4th layer is full attention (3, 7, ..., 31)


def _model_path() -> str:
    return os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL") or _DEFAULT_CKPT


def layer_ids(num_layers: int, layer_types=None):
    """(attention layer ids, GDN layer ids) of the text model."""
    if layer_types:
        attn = tuple(i for i, t in enumerate(layer_types[:num_layers]) if t == "full_attention")
    else:
        attn = tuple(i for i in range(num_layers) if (i + 1) % _FULL_ATTENTION_INTERVAL == 0)
    return attn, tuple(i for i in range(num_layers) if i not in attn)


class Qwen35PupPrefillAdapter(PrefillModelAdapter):
    name = "qwen35_9b_pup"
    model_config = SimpleNamespace(NUM_LAYERS=32, EMB_SIZE=4096, FABRIC_PAYLOAD_SIZE=4096, NUM_KEY_VALUE_HEADS=4)
    hf_model_default = _DEFAULT_CKPT
    ttnn_cache_default = ""
    prefill_trace_default = ""
    default_gate_mode = "DEVICE_FP32"  # unused (dense)
    # tt-metal #63999: the runner passes media refs only to adapters that declare it
    supports_media = True
    # The traced short prefill and the trace region the s5 profile needs (the runner opens the mesh with these).
    l1_small_size = 24576
    trace_region_size = 1024 * 1024 * 1024
    supports_pretrained = False

    def __init__(self, pup_config: PupPrefillConfig = DEFAULT_CONFIG, media_reader=None):
        super().__init__()
        self.pup_config = pup_config
        self.media_reader = media_reader

    # --- HF config / weights -----------------------------------------------------------------------------------
    def load_hf_config(self):
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(_model_path())
        return getattr(cfg, "text_config", cfg)

    def weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        root = os.environ.get("TT_CACHE_PATH")
        return Path(root) if root else None

    # --- table semantics ---------------------------------------------------------------------------------------
    def cache_kind(self, config_id: int) -> str:
        from models.demos.blackhole.qwen36.tt.runners.kv_table import cache_kind

        return cache_kind(config_id, int(self.model_config.NUM_KEY_VALUE_HEADS))

    def layer_position_range(self, layer_idx: int, real_len: int) -> tuple[int, int]:
        from models.demos.blackhole.qwen36.tt.runners.gdn_state import GdnGeometry
        from models.demos.blackhole.qwen36.tt.runners.kv_table import layer_position_range

        _, gdn = layer_ids(int(self.model_config.NUM_LAYERS))
        return layer_position_range(layer_idx, real_len, gdn, GdnGeometry(conv_per_value_head=True))

    # --- caches + runtime --------------------------------------------------------------------------------------
    def _serving_config(self, params: PrefillRunParams) -> PupPrefillConfig:
        # PUP_PREFILL_FAST=0: the generic B=1 runtime (A/B against the fast one); unset = the config's choice
        fast = os.environ.get("PUP_PREFILL_FAST")
        return self.pup_config.with_serving(
            num_users=params.num_users,
            max_seq_len=params.max_seq_len,
            chunk_size=params.chunk_size,
            fast=None if fast is None else fast != "0",
        )

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        from models.demos.blackhole.qwen36.tt.runners import kv_table
        from models.demos.blackhole.qwen36.tt.runners.gdn_state import GdnGeometry

        assert params.is_first_rank and params.is_last_rank, "the PUP prefill is one rank (one (1, 4) column)"
        cfg = self._serving_config(params)
        n = int(getattr(hf_config, "num_hidden_layers", self.model_config.NUM_LAYERS))
        attn, gdn = layer_ids(n, getattr(hf_config, "layer_types", None))
        geometry = GdnGeometry(
            num_k_heads=int(getattr(hf_config, "linear_num_key_heads", 16)),
            num_v_heads=int(getattr(hf_config, "linear_num_value_heads", 32)),
            head_dim=int(getattr(hf_config, "linear_key_head_dim", 128)),
            conv_history=int(getattr(hf_config, "linear_conv_kernel_dim", 4)) - 1,
            sequence_parallel_size=int(params.mesh_shape[params.sp_axis]),
            tensor_parallel_size=int(params.mesh_shape[params.tp_axis]),
            conv_per_value_head=True,  # the decode ring's form (tt-blaze qwen35_9b cache_migration)
        )
        return kv_table.allocate(
            mesh_device,
            attn_layer_ids=attn,
            gdn_layer_ids=gdn,
            geometry=geometry,
            num_users=cfg.num_users,
            max_seq_len=cfg.max_seq_len,
            head_dim=int(getattr(hf_config, "head_dim", 256)),
        )

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        import ttnn
        from models.demos.blackhole.qwen36.tt.model import Qwen36Model
        from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
        from models.demos.blackhole.qwen36.tt.runners.runtime import RuntimeConfig, TtQwen35PupPrefillRuntime

        os.environ["HF_MODEL"] = _model_path()  # Qwen36ModelArgs reads the checkpoint from HF_MODEL
        cfg = self._serving_config(params)
        args = Qwen36ModelArgs(mesh_device=mesh_device, max_batch_size=1, max_seq_len=cfg.max_seq_len, enable_mtp=False)
        cfg.apply(args)
        model = Qwen36Model(mesh_device, args, args.load_state_dict(), tensor_cache_path=args.weight_cache_path())
        # the model's working paged cache: slot s owns blocks [s * bps, (s + 1) * bps), plus one trash block for the
        # padded rows of a bucket; the published copy is the engine's M3-layout cache
        nb = cfg.num_users * cfg.blocks_per_slot + 1
        nb = -(-nb // 32) * 32  # paged SDPA page-table sticks are multiples of 32 blocks
        model.allocate_kv_caches(
            [nb, getattr(args, "n_local_kv_heads", args.n_kv_heads), cfg.block_size, args.head_dim],
            ttnn.bfloat8_b,
            batch_size=1,
        )
        rc = RuntimeConfig(
            chunk_size=cfg.chunk_size,
            max_seq_len=cfg.max_seq_len,
            num_users=cfg.num_users,
            num_layers=int(self.model_config.NUM_LAYERS),
            use_trace=True,  # the model's own per-bucket traces (not the runner's segmented capture_trace)
            pup=cfg,
        )
        reader = self.media_reader
        if reader is None:
            # The encoder service writes image embeddings into this host's segment of the media store
            # (PREFILL_MEDIA_STORE_*, tt-metal #64095); the store is on when PREFILL_MEDIA_STORE_MASTER is set.
            from models.common.multimedia_store.config import MediaEmbeddingStoreConfig

            store = MediaEmbeddingStoreConfig.from_env()
            if store is not None:
                from models.common.multimedia_store.mooncake_store import MooncakeEmbeddingReader

                reader = self.media_reader = MooncakeEmbeddingReader.open(store)
        if cfg.fast:
            from transformers import AutoTokenizer

            from models.demos.blackhole.qwen36.tt.runners.fast_runtime import TtQwen35PupFastPrefillRuntime

            # the system turn ends at the first <|im_end|> (Qwen chat template): the snapshot boundary
            end_tok = AutoTokenizer.from_pretrained(_model_path()).convert_tokens_to_ids("<|im_end|>")
            if os.environ.get("PUP_PREFILL_PREFIX", "1") == "0":  # A/B: no system-turn snapshot (every request from 0)
                end_tok = None
            return TtQwen35PupFastPrefillRuntime(
                mesh_device=mesh_device, model=model, config=rc, media_reader=reader, prefix_end_token=end_tok
            )
        return TtQwen35PupPrefillRuntime(mesh_device=mesh_device, model=model, config=rc, media_reader=reader)
