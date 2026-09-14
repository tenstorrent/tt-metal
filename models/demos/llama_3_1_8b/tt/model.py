# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole model: embedding -> N decoder layers -> final norm -> LM head.

New at model scale relative to the single decoder layer (recipe M1/M2):

* **per-layer weight slicing** — the state dict is sliced by ``model.layers.N.`` and handed to each
  layer, so a layer can only ever see its own tensors;
* **KV cache sized for N layers** — allocated once here (or handed in by the runtime) with
  ``num_layers`` slots per user, not one;
* **per-layer type dispatch** — *not applicable*: every Llama-3.1 layer is the same dense GQA block.

Prefill is headless by default: the populated KV cache is the deliverable, so ``skip_lm_head``
defaults to True and the final norm + LM head are skipped with it.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule

from .attention.kv_cache import allocate_kv_cache
from .ccl import CCLManager
from .layer import DecoderLayer
from .lm_head import LMHead
from .mesh import MeshConfig
from .parallel_embedding import ParallelEmbedding, cache_name_for, embed_shard_2d
from .rms_norm import RMSNorm
from .rope import RopeSetup
from ..utils.general import cache_name, default_num_links, substate


class Model(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg,
        *,
        mesh_config: Optional[MeshConfig] = None,
        ccl_manager: Optional[CCLManager] = None,
        state_dict: Optional[dict] = None,
        max_seq_len: int = 10240,
        num_layers: Optional[int] = None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        sequence_parallel: bool = True,
        shard_vocab_on_sp: Optional[bool] = None,
        build_lm_head: bool = True,
    ):
        """``num_layers`` under ``cfg.num_hidden_layers`` builds a DEPTH-REDUCED model.

        That is a diagnostic, never a result (recipe §4) — it exists so a host can hold two copies of
        the model as random weights for a whole-model PCC test. The graded number is the full-depth
        acceptance run.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl_manager = ccl_manager or CCLManager(mesh_device, num_links=default_num_links(mesh_device))
        self.num_layers = cfg.num_hidden_layers if num_layers is None else num_layers
        self.max_seq_len = max_seq_len
        self.sequence_parallel = sequence_parallel
        self.rope_setup = RopeSetup(mesh_device, cfg, self.mesh_config)

        shard_vocab = embed_shard_2d() if shard_vocab_on_sp is None else shard_vocab_on_sp
        self.embedding = ParallelEmbedding(
            mesh_device,
            cfg.vocab_size,
            cfg.hidden_size,
            self.mesh_config,
            self.ccl_manager,
            torch_weight=substate(state_dict, "model.embed_tokens")["weight"] if state_dict else None,
            cache_file_name=cache_name(tensor_cache_path, cache_name_for(shard_vocab)),
            shard_vocab_on_sp=shard_vocab,
        )
        self.layers = [
            DecoderLayer(
                mesh_device,
                cfg,
                self.mesh_config,
                self.ccl_manager,
                self.rope_setup,
                layer_idx=i,
                state_dict=substate(state_dict, f"model.layers.{i}") if state_dict else None,
                max_seq_len=max_seq_len,
                weight_dtype=weight_dtype,
                tensor_cache_path=cache_name(tensor_cache_path, f"layers.{i}"),
                sequence_parallel=sequence_parallel,
            )
            for i in range(self.num_layers)
        ]
        self.norm = RMSNorm(
            mesh_device,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            state_dict=substate(state_dict, "model.norm") if state_dict else None,
            cache_file_name=cache_name(tensor_cache_path, "norm"),
        )
        self.lm_head = (
            LMHead(
                mesh_device,
                cfg,
                self.mesh_config,
                state_dict=substate(state_dict, "lm_head") if state_dict else None,
                weight_dtype=weight_dtype,
                cache_file_name=cache_name(tensor_cache_path, "lm_head"),
            )
            if build_lm_head
            else None
        )

    def allocate_kv_cache(self, *, num_users: int = 1, cache_dtype=ttnn.bfloat8_b):
        """The model's own KV cache, ``num_layers`` slots per user. The runtime may allocate its own
        instead and pass it in — the forward never owns one."""
        return allocate_kv_cache(
            self.mesh_device,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            num_kv_heads=self.cfg.num_key_value_heads,
            tp=self.mesh_config.tp,
            sp_axis=self.mesh_config.sp_axis,
            num_users=num_users,
            head_dim=self.cfg.head_dim,
            cache_dtype=cache_dtype,
        )

    def prefill_forward(
        self,
        x,
        *,
        rope_mats,
        kv_cache=None,
        cached_len: int = 0,
        user_id: int = 0,
        indexed_rope: bool = False,
        skip_lm_head: bool = True,
        on_layer_complete=None,
    ):
        """``x`` is the already-embedded hidden state ``[1, 1, s_local, hidden]``.

        ``on_layer_complete(layer_idx)`` is the per-layer migration/LayerAck seam: it is called after
        each layer's KV is in the cache. Nothing in this package consumes it — serving and KV
        migration are a separate follow-on — but the callback is where they would attach, and having
        the seam costs nothing.
        """
        for layer in self.layers:
            x = layer(
                x,
                rope_mats,
                kv_cache=kv_cache,
                user_id=user_id,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            if on_layer_complete is not None:
                on_layer_complete(layer.layer_idx)
        if skip_lm_head:
            return x
        normed = self.norm(x)
        x.deallocate(True)
        assert self.lm_head is not None, "model was built with build_lm_head=False"
        logits = self.lm_head(normed)
        normed.deallocate(True)
        return logits
