# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.layernorm import LayerNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.phi_decoder import Phi1TransformerBlock


class Phi1Transformer(Transformer):
    def _get_block_class(self):
        return Phi1TransformerBlock

    def _build_norm(self, args, mesh_device, state_dict, weight_cache_path, prefetcher):
        return DistributedNorm(
            LayerNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                is_distributed=self.args.is_distributed_norm,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=prefetcher,
            TG=False,
        )
