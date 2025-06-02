# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.model import Transformer
from models.experimental.phi3_mini.tt.phi3_mini_rope import Phi3MiniRotarySetup


class Phi3Transformer(Transformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        self.rope_setup = Phi3MiniRotarySetup(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            scale_factor=args.rope_scaling_factor,
            ext_scale_tensors=args.rope_scaling,
            orig_context_len=args.orig_context_len,
            datatype=ttnn.bfloat16,
        )
