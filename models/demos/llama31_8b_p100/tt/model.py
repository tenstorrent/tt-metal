# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.ttt_compat.tt.model import *  # noqa: F401,F403  # isort: skip
import torch

import models.ttt_compat.tt.model as _tt
import ttnn
from models.demos.llama31_8b_p100.tt import wiring
from models.demos.llama31_8b_p100.tt.attention import Attention
from models.demos.llama31_8b_p100.tt.decoder import TransformerBlock
from models.demos.llama31_8b_p100.tt.decoder import fuse_norms as llama31_8b_fuse_norms
from models.demos.llama31_8b_p100.tt.lm_head import LMHead
from models.ttt_compat.tt.common import Mode
from models.ttt_compat.tt.model_config import TensorGroup


class Transformer(_tt.Transformer):
    def __init__(self, *args, **kwargs):
        wiring.reset()
        kwargs.setdefault("attention_class", Attention)
        kwargs.setdefault("block_class", TransformerBlock)
        kwargs.setdefault("lm_head_cls", LMHead)
        super().__init__(*args, **kwargs)
        greedy = wiring.on("greedy_argmax", self.args)
        if greedy is not None and getattr(self, "sampling", None) is not None:
            greedy.use_for(self.sampling)

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode: Mode = Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        page_tables_per_layer=None,
    ):
        if mode == Mode.DECODE:
            if self.prefetcher is not None:
                self.prefetcher.run()

        if mode == Mode.PREFILL:
            prefill_seq_len = x.shape[2] // batch_size
            rot_mats_global = self._slice_prefill_rot_mats(rot_mats_global, chunk_start_idx, prefill_seq_len)
            if rot_mats_local is not None:
                rot_mats_local = self._slice_prefill_rot_mats(rot_mats_local, chunk_start_idx, prefill_seq_len)

        if page_tables_per_layer is not None and len(page_tables_per_layer) != len(self.layers):
            raise ValueError(
                f"page_tables_per_layer has {len(page_tables_per_layer)} entries "
                f"but model has {len(self.layers)} layers"
            )

        fuse_norms = llama31_8b_fuse_norms(self.args, mode)
        normed_x = None
        for i, layer in enumerate(self.layers):
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )

            if mode == Mode.DECODE and not self.args.is_galaxy:
                x = ttnn.to_memory_config(
                    x,
                    self.args.get_residual_mem_config(mode, self.prefetcher),
                    activation_dtype,
                )
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)

            layer_page_table = page_tables_per_layer[i] if page_tables_per_layer is not None else page_table

            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=layer_page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                batch_size=batch_size,
                **(
                    {
                        "normed_x": normed_x,
                        "next_attention_norm": self.layers[i + 1].attention_norm
                        if i + 1 < len(self.layers)
                        and self.args.decoders_optimizations.get_tensor_dtype(
                            decoder_id=i + 1, tensor=TensorGroup.ACTIVATION
                        )
                        in (None, ttnn.bfloat16)
                        else None,
                    }
                    if fuse_norms
                    else {}
                ),
            )
            normed_x = layer.next_normed if fuse_norms else None

        if mode == Mode.DECODE:
            if self.prefetcher is not None:
                self.prefetcher.stop()

        if mode == Mode.PREFILL and get_last_token == -1:
            return x

        if get_last_token != -1:
            seq_len = int(x.shape[2])
            if seq_len % 32 == 0:
                for device_tensor, values in (
                    (self._tail_slice_start, [0, 0, get_last_token, 0]),
                    (self._tail_slice_end, [1, 1, get_last_token + 32, int(x.shape[-1])]),
                ):
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(
                            torch.tensor(values, dtype=torch.int32),
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        ),
                        device_tensor,
                    )
                x = ttnn.slice(
                    input_tensor=x,
                    starts=self._tail_slice_start,
                    ends=self._tail_slice_end,
                    slice_dim=2,
                    num_devices=seq_len // 32,
                )
            else:
                x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        x = self.norm(x, mode=mode, norm_config=self.args.get_norm_config("lm_head", mode, self.prefetcher))

        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(
            mode, None if mode == Mode.PREFILL else self.prefetcher
        )
        if mode == Mode.PREFILL and lm_head_input_mem_cfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)
        if mode == Mode.DECODE and self.prefetcher is not None:
            x = ttnn.to_memory_config(x, self.args.get_lm_head_input_mem_config(mode, self.prefetcher))

        x = self.lm_head(x)
        x = self._apply_final_logit_softcapping(x)
        if mode == Mode.PREFILL:
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return x
