# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for Ling-mini-2.0 (BailingMoeV2) via tt_symbiote.

Bridges vLLM's serving interface (initialize_vllm_model, prefill_forward,
decode_forward, allocate_kv_cache) to the HF BailingMoeV2 model whose
decoder layers, norm, embedding, rotary embedding, linear projections,
SiLU, and text-model wrapper have been replaced with TTNN equivalents.

The module replacement pattern follows tests/test_ling_mini_2_0.py.
The boilerplate (DIAG, watchdog, _to_host_tensor, prefill/decode forward,
warmup loops, KV cache hand-off) lives in
models.experimental.tt_symbiote.vllm.symbiote_adapter_base.SymbioteAdapterBase;
this module only carries the Ling-specific transformers 5.x shims, the
3-pass replacement, and the single-paged KV cache build.
"""

# ---------------------------------------------------------------------------
# Transformers 5.x compatibility shims for Ling-mini-2.0's HF custom code.
# These must be applied at module-import time, BEFORE
# AutoModelForCausalLM.from_pretrained triggers any HF dynamic-import chain
# (which itself references the missing symbols). Centralising them in
# SymbioteAdapterBase would defeat the ordering guarantee, so they live here.
#
# The same shims also live in models/experimental/tt_symbiote/tests/conftest.py
# for the standalone pytest path that bypasses this adapter.
#
# (1) is_torch_fx_available was removed in transformers 5.x. Returning False
#     skips the torch.fx wrapping which is a tracing-only optimisation.
# (2) ROPE_INIT_FUNCTIONS['default'] was dropped in transformers 5.x. The HF
#     custom rotary embedding sets rope_type='default' when rope_scaling is
#     absent. We add the key back with the original plain inv-freq formula
#     (base ** (-2i/dim)). The HF rotary emb is replaced by
#     TTNNBailingRotaryEmbedding anyway, so this only needs to survive __init__.
# ---------------------------------------------------------------------------
import torch
import transformers.utils.import_utils as _tui

if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: False

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_INIT

if "default" not in _ROPE_INIT:

    def _default_rope_init(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        return inv_freq, 1.0  # attention_factor unused for default RoPE

    _ROPE_INIT["default"] = _default_rope_init

import logging
from math import ceil
from typing import Optional

from torch import nn
from tqdm import tqdm

from models.experimental.tt_symbiote.vllm.symbiote_adapter_base import SymbioteAdapterBase

logger = logging.getLogger(__name__)


class SymbioteBailingMoeV2ForCausalLM(SymbioteAdapterBase):
    """vLLM-compatible adapter for Ling-mini-2.0 (BailingMoeV2) on TT hardware.

    Implements the four-method contract expected by TTModelLoader / TTModelRunner.
    The contract methods are inherited from SymbioteAdapterBase; this subclass
    only overrides _build_model_and_kv_cache (3-pass HF module replacement and
    single paged KV cache allocation).
    """

    MODEL_KEY = "LING"
    WATCHDOG_PREFILL_KERNEL_HINT = "bailing attention"

    # Sequence lengths primed during warmup. Covers every ISL the benchmark
    # sweep exercises against the spec's max_context=2048 cap: rows
    # (128,128), (128,1024), (1024,128), (2048,128) all hit a warmed
    # program-cache bucket so first-request TTFT does not pay JIT compile
    # cost. Values <= max_position_embeddings are filtered at runtime in
    # warmup_model_prefill. If max_context is raised in the future, extend
    # this tuple to match (e.g. add 3072 for max_context=3072).
    WARMUP_PREFILL_SEQ_LENS = (128, 1024, 2048)

    @classmethod
    def _build_model_and_kv_cache(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel,
        optimizations: Optional[str] = None,
        **kwargs,
    ):
        """Load HF BailingMoeV2 and replace modules with TTNN equivalents.

        Mirrors the loading and replacement logic from test_ling_mini_2_0.py:
        three-pass module replacement, weight preprocessing, device transfer,
        and a single paged KV cache allocation. Returns (model, kv_cache,
        model_device); the base then runs model.eval(), grad disable, and the
        final device-property patch.
        """
        from transformers import AutoModelForCausalLM

        from models.experimental.tt_symbiote.models.bailing_moe_v2 import (
            TTNNBailingMoeV2Model,
        )
        from models.experimental.tt_symbiote.modules.activation import TTNNSilu
        from models.experimental.tt_symbiote.modules.attention import (
            PagedAttentionConfig,
            TTNNPagedAttentionKVCache,
        )
        from models.experimental.tt_symbiote.modules.decoder_layer import (
            TTNNBailingMoEDecoderLayerPadded,
        )
        from models.experimental.tt_symbiote.modules.embedding import (
            TTNNBailingPaddedEmbedding,
            TTNNBailingRotaryEmbedding,
        )
        from models.experimental.tt_symbiote.modules.linear import (
            TTNNLinearIColShardedWRowSharded,
        )
        from models.experimental.tt_symbiote.modules.normalization import (
            TTNNDistributedRMSNorm,
        )
        from models.experimental.tt_symbiote.utils.device_management import set_device
        from models.experimental.tt_symbiote.utils.module_replacement import (
            register_module_replacement_dict,
        )

        model_name = hf_config._name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Capture model_device BEFORE module replacement -- after replacement,
        # TTNN modules lack _parameters so model.parameters() would fail.
        model_device = next(model.parameters()).device

        # BailingMoeV2 is text-only: the BailingMoeV2Model lives at model.model.
        text_model = model.model
        decoder_class = text_model.layers[0].__class__
        norm_class = text_model.norm.__class__
        rotary_class = text_model.rotary_emb.__class__
        text_model_class = text_model.__class__

        # Three-pass replacement (mirrors test_ling_mini_2_0.py:87-99). The
        # passes are sequenced so per-instance class replacements happen
        # before the blanket nn.Linear / nn.SiLU sweep, and the text-model
        # wrapper replacement runs last so its from_torch sees its children
        # already replaced.
        nn_to_ttnn = {
            decoder_class: TTNNBailingMoEDecoderLayerPadded,
            norm_class: TTNNDistributedRMSNorm,
            nn.Embedding: TTNNBailingPaddedEmbedding,
            rotary_class: TTNNBailingRotaryEmbedding,
        }
        nn_to_ttnn2 = {
            nn.Linear: TTNNLinearIColShardedWRowSharded,
            nn.SiLU: TTNNSilu,
        }
        nn_to_ttnn_3 = {text_model_class: TTNNBailingMoeV2Model}

        modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
        modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
        modules3 = register_module_replacement_dict(model, nn_to_ttnn_3, model_config=None)
        all_modules = {**modules1, **modules2, **modules3}

        # After replacing all nn.Modules with TTNNModules, HF's model.device
        # (which calls next(self.parameters())) fails. Patch it to return cpu
        # while set_device runs; the base's initialize_vllm_model will install
        # the final, authoritative patch (model.device -> captured model_device)
        # after _build_model_and_kv_cache returns.
        type(model).device = property(lambda self: torch.device("cpu"))

        set_device(model, mesh_device)

        logger.info(f"Preprocessing {len(all_modules)} TTNN modules weights...")
        for name, mod in tqdm(all_modules.items(), desc="Preprocessing & moving weights"):
            mod.preprocess_weights()
            mod.move_weights_to_device()

        # Allocate a single paged-attention KV cache sized to the configured
        # max_seq_len. block_size matches vllm_args["block_size"]=64 in the
        # ModelSpecTemplate; max_num_blocks covers max_seq_len * max_batch_size.
        block_size = 64
        max_num_blocks = max(1, ceil(max_seq_len / block_size)) * max(1, max_batch_size)
        kv_cache = TTNNPagedAttentionKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            config=PagedAttentionConfig(
                block_size=block_size,
                max_num_blocks=max_num_blocks,
                batch_size=max_batch_size,
            ),
            device=None,
        ).to_device(mesh_device)

        return model, kv_cache, model_device
