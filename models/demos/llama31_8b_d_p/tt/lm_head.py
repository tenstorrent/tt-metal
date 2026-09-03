# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B LM head — the last-token logits projection.

HF anchor: ``transformers.models.llama.modeling_llama.LlamaForCausalLM.lm_head``
(``nn.Linear(hidden_size, vocab_size, bias=False)``).
Template: ``models/demos/gpt_oss_d_p/tt/model.py:127`` (the transpose to ``[hidden, vocab]``),
``:134`` (``as_tensor``), ``:141`` (``column_parallel``), ``:241`` (the matmul).

**Prefill's real product is the KV cache, not logits** — ``prefill_forward`` runs with
``skip_lm_head=True`` on the deployment path. This module exists for ``G-MODEL``'s top-1 token
agreement check, which is the only test in this iteration that needs a distribution over the
vocabulary.

**Two deviations from the template, both ``DEC-015``:**

1. **No power-of-2 vocab padding.** gpt-oss rounds the per-device vocab up to a power of two
   (``models/demos/gpt_oss_d_p/tt/model.py:31`` ``compute_per_device_vocab``, ``:38``) purely so
   ``ttnn.topk``'s multi-core bitonic path works for on-device sampling. This iteration has no
   on-device sampling, so the plain ``vocab_size / tp`` shard is used and
   ``compute_per_device_vocab`` / ``padded_vocab_size`` / ``_supports_on_device_sampling``
   (``model.py:145``) are all deleted rather than defaulted off. At TP=8 that is
   ``128256 / 8 = 16032`` = 501 tiles, exactly aligned.
2. **No device-side all-gather on the vocab shard.** The TP concat happens on the host in
   ``Model.process_output_prefill``, exactly as the template does (``model.py:322``). So the LM head
   contributes **zero** collectives (``bringup_log/04_CCL_PLAN.md`` §7, row 6).

``lm_head.weight`` is a real, separate tensor: Llama-3.1-8B has ``tie_word_embeddings: false``
(``configs/Llama-3.1-8B-Instruct/config.json:33``), unlike Llama-3.2-1B/3B. Asserted at
construction, because aliasing the embedding table would produce plausible-looking logits that are
wrong for every token.
"""

from __future__ import annotations

import ttnn
from models.demos.llama31_8b_d_p.utils.general_utils import get_cache_file_name


class LMHead:
    """``[1, 1, S, hidden]`` -> ``[1, 1, S, vocab/tp]``. Column-parallel, no bias, no collective."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        *,
        mesh_config,
        tensor_cache_path=None,
        weight_dtype=ttnn.bfloat8_b,
        compute_kernel_config=None,
    ):
        """
        Args:
            mesh_device: the ttnn mesh device.
            hf_config: a ``LlamaHFConfig``.
            state_dict: the already-stripped ``lm_head.*`` sub-dict, i.e.
                ``{"weight": [vocab_size, hidden_size]}`` in HF ``[out, in]`` layout. ``{}`` means
                cache-only mode, which requires ``tensor_cache_path``.
            mesh_config: ``MeshConfig``; supplies the column-parallel mapper.
            tensor_cache_path: directory for the tilized weight cache, or ``None``.
            weight_dtype: on-device weight dtype (default ``bfloat8_b``, as the template).
            compute_kernel_config: passed to the matmul; ``None`` builds the package default
                (HiFi4 + ``fp32_dest_acc_en=True``, ``DEC-031``).
        """
        from models.demos.llama31_8b_d_p.tt.mlp import default_compute_kernel_config

        assert hf_config.tie_word_embeddings is False, (
            "hf_config.tie_word_embeddings is True, but this LMHead loads a separate lm_head.weight. "
            "Llama-3.1-8B is untied (config.json:33); a tied variant (Llama-3.2-1B/3B) needs the "
            "embedding table transposed here instead, which is not implemented (03_OUTLINE.md §3.15)."
        )

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.compute_kernel_config = (
            default_compute_kernel_config(mesh_device) if compute_kernel_config is None else compute_kernel_config
        )

        # No padding path (DEC-015), so the shard must be exact and tile-aligned rather than
        # rounded up: assert instead of working around.
        if mesh_config.tp > 1:
            assert self.vocab_size % mesh_config.tp == 0, (
                f"vocab_size {self.vocab_size} is not divisible by tp {mesh_config.tp}; this LMHead "
                f"deletes gpt-oss's power-of-2 vocab padding (DEC-015) and cannot shard raggedly"
            )
            assert (self.vocab_size // mesh_config.tp) % ttnn.TILE_SIZE == 0, (
                f"vocab_size/tp = {self.vocab_size // mesh_config.tp} is not a multiple of "
                f"TILE_SIZE ({ttnn.TILE_SIZE})"
            )

        if state_dict:
            assert "weight" in state_dict, (
                "LMHead needs the stripped sub-dict {'weight': ...}; pass substate(sd, 'lm_head'). "
                "An empty sub-dict here is the signature of Meta-renamed keys (DEC-039)."
            )
            assert "bias" not in state_dict, "lm_head carries a bias; Llama-3.1 has none"
            weight = state_dict["weight"]
            assert weight.shape == (self.vocab_size, self.hidden_size), (
                f"lm_head.weight is {tuple(weight.shape)}, expected "
                f"{(self.vocab_size, self.hidden_size)} from hf_config"
            )
            # HF [out, in] -> ttnn [in, out] == [hidden, vocab], once, at load time.
            weight = weight.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        else:
            assert tensor_cache_path, (
                "LMHead got an empty state_dict and no tensor_cache_path; there is nothing to load "
                "from (cache-only mode needs the cache, DEC-038)"
            )
            weight = None

        self.weight = ttnn.as_tensor(
            weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
        )

    def __call__(self, x):
        """``[1, 1, S, hidden]`` -> ``[1, 1, S, vocab/tp]`` bf8_b, per chip. No collective."""
        return ttnn.linear(
            x,
            self.weight,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
        )
