# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM wrapper for Gemma4 with hybrid kv cache support.

Gemma4 has a 5:1 sliding-window:full-attention layer pattern; routing
each layer to its own kv_cache_group's page table is required to avoid
the asymmetric-hybrid memory waste described in vLLM's hybrid kv cache
manager design. ``Gemma4ForCausalLM`` registers as a vLLM model class
and handles the runtime-side hybrid plumbing on top of the standalone
Gemma4 model in ``models.demos.gemma4``.
"""

from collections import defaultdict

from models.demos.gemma4.tt.common import create_tt_model
from models.tt_transformers.tt.generator import create_submeshes
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM


class Gemma4ForCausalLM(HybridAttentionForCausalLM):
    """vLLM wrapper for the Gemma4 26B-A4B / 31B-it models.

    Inherits ``get_kv_cache_spec`` from :class:`HybridAttentionForCausalLM`
    (per-layer specs from ``hf_config.text_config.layer_types``) and the
    default ``allocate_kv_cache_per_layer`` (delegates to
    :func:`allocate_vllm_kv_cache_per_layer`).

    ``prefill_forward`` and ``decode_forward`` are intentionally inherited
    as :class:`NotImplementedError` stubs in this commit: full vLLM
    serving for Gemma4 needs the standalone ``Gemma4Model``'s prefill /
    decode pipeline to be wired into ``Generator``'s text forward path,
    which is its own follow-up. The model-level routing
    (``page_tables_per_group`` + ``layer_to_group`` plumbed through
    ``Gemma4Model.__call__``) is in place and ready for that integration.

    Includes :meth:`_build_layer_to_group`: a helper that mirrors
    upstream's :func:`vllm.v1.core.kv_cache_utils._get_kv_cache_groups_
    uniform_page_size` grouping logic so the wrapper can map each model
    layer index back to its kv_cache_group when invoking
    ``Gemma4Model``.
    """

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
    }

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=131072,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str | None = None,
    ):
        # Gemma4's standalone model handles its own optimizations; we
        # ignore the vLLM-supplied ``optimizations`` arg here.
        del optimizations
        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        models = []
        model_args_list = []
        for submesh in submesh_devices:
            model_args, model, _kv_cache, _state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                num_layers=n_layers,
                # vLLM allocates the KV cache via allocate_kv_cache_per_layer
                # after this returns; skip allocation here so we don't
                # double-allocate buffers.
                create_kv_cache=False,
            )
            models.append(model)
            model_args_list.append(model_args)

        return cls(models, model_args_list, mesh_device)

    @property
    def cache_path(self):
        # Gemma4 model_args expose model_cache_path directly.
        return self.model_args[0].model_cache_path

    def _build_layer_to_group(self, num_groups):
        """Map each model layer index → kv_cache_group index.

        Mirrors upstream's grouping in
        ``vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size``:
        group_size = min(per-type counts), with the larger type split
        into multiple groups via ``layers[i::num_groups]`` (interleaved
        so PP partitioning works). Cached on the instance.
        """
        cached = getattr(self, "_layer_to_group_cache", None)
        if cached is not None and cached["num_groups"] == num_groups:
            return cached["layer_to_group"]

        hf_config = self.model_args[0]
        text_config = getattr(hf_config, "text_config", hf_config)
        layer_types = list(getattr(text_config, "layer_types", []))
        if not layer_types:
            raise ValueError(
                "Gemma4ForCausalLM requires hf_config.text_config.layer_types "
                "to build per-layer kv_cache_group routing"
            )

        same_type_layers: dict[str, list[int]] = defaultdict(list)
        for i, lt in enumerate(layer_types):
            same_type_layers[lt].append(i)

        min_count = min(len(v) for v in same_type_layers.values())
        max_count = max(len(v) for v in same_type_layers.values())
        group_size = max_count if max_count < min_count * 1.25 else min_count

        layer_to_group: list[int | None] = [None] * len(layer_types)
        next_group_idx = 0
        for indices in same_type_layers.values():
            num_groups_for_type = -(-len(indices) // group_size)  # cdiv
            for i in range(num_groups_for_type):
                for layer_idx in indices[i::num_groups_for_type]:
                    layer_to_group[layer_idx] = next_group_idx
                next_group_idx += 1

        if next_group_idx != num_groups:
            raise ValueError(
                f"Computed {next_group_idx} kv_cache_groups from layer_types "
                f"but vLLM reported {num_groups}; the per-model grouping is "
                "out of sync with upstream's kv_cache_utils. Update "
                "_build_layer_to_group to match."
            )

        self._layer_to_group_cache = {
            "num_groups": num_groups,
            "layer_to_group": layer_to_group,
        }
        return layer_to_group
