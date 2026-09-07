# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Medium-3.5 dense SwiGLU MLP — every one of the 88 layers.

Donor: ``minimax_m3/tt/dense_mlp.py`` (``compute.mlp_dense``). ``gpt_oss_d_p`` has NO dense MLP —
every GPT-OSS layer is MoE, so its ``tt/mlp.py`` is a router + expert-parallel wrapper and the wrong
shape entirely — which is why this one entry comes from a different package than the rest of the map.

What transfers, and is the reason this entry exists: gate/up are COLUMN-parallel (they shard the
intermediate dim across TP), down is ROW-parallel followed by a TP all-reduce, and the weight-cache
discipline is per-tensor tilized ``.tensorbin`` via ``ttnn.as_tensor(cache_file_name=)``.

Adapted from the donor:
  * **activation**: M3's clamped SwiGLU-OAI (``moe/activation.swiglu`` with ``swiglu_limit`` /
    ``alpha``) -> plain silu SwiGLU. HF ``Ministral3MLP`` is exactly
    ``down_proj(act_fn(gate_proj(x)) * up_proj(x))`` with ``hidden_act == "silu"``, so there is no
    clamp, no ``alpha``, and no ``(up + 1)`` term. Composed from ``ttnn.silu`` + ``ttnn.mul``
    (there is no ``ttnn.swiglu``, and the decomposition is exact).
  * **intermediate**: 12288 -> 28672.
  * **every layer is dense** — M3 gates its dense MLP to layers 0-2 on ``moe_layer_freq``; Mistral
    has no ``moe_layer_freq`` and no experts anywhere, so there is no gating.
  * **package-local imports**: the donor pulls ``minimax_m3`` utils (``general_utils``,
    ``profiler_utils``, ``substate``, ``residual``) while the rest of this map is ``gpt_oss_d_p``.
    Per the donor note, the structure is ported onto this package's own equivalents rather than
    imported across packages, and the M3 profiler zones are dropped (perf tooling, out of scope).
  * **residual scheme**: the donor supports M3's ``emb/tp``-sharded residual stream (closing with a
    reduce-scatter instead of an all-reduce). This package follows the gpt-oss donor's REPLICATED
    residual throughout — full hidden on every TP column, all-reduce tails — so the whole model is
    one scheme. Sharding the residual is a memory/perf optimization and out of bring-up scope; the
    ``scatter_output`` seam is kept so it can be turned on later without reshaping this file.

Weight keys ``mlp.{gate,up,down}_proj.weight`` match the donor's.
"""

import ttnn
from models.demos.mistral_3_5_d_p.utils.general_utils import get_cache_file_name, get_matmul_compute_config
from models.demos.mistral_3_5_d_p.utils.substate import substate


class MLP:
    """Dense silu-SwiGLU FFN: ``down(silu(gate(x)) * up(x))`` at intermediate_size 28672."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        weight_dtype=None,
        tensor_cache_path=None,
        scatter_output=False,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            hf_config: HF text config (reads ``hidden_size``, ``intermediate_size``, ``hidden_act``)
            state_dict: ``{gate,up,down}_proj.weight`` in HF ``[out, in]`` layout. Empty dict ->
                cache-only load (weights come from the tilized cache).
            mesh_config: Mesh parallelization config
            ccl_manager: Communication manager — REQUIRED when TP > 1 (down_proj is row-parallel, so
                its per-device partial sums are wrong until the collective runs)
            weight_dtype: defaults to the spec's ``dataformats.weights.mlp.*``
            tensor_cache_path: Optional path for weight caching
            scatter_output: close with a TP reduce-scatter (``hidden/tp`` out) instead of an
                all-reduce. False for this package — the residual stream is replicated.
        """
        from models.demos.mistral_3_5_d_p.spec import SPEC

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.scatter_output = scatter_output
        act = getattr(hf_config, "hidden_act", "silu")
        # Fail loud rather than silently applying the wrong nonlinearity: this file implements silu
        # SwiGLU only, and a config that changes hidden_act changes the model.
        assert act == "silu", f"MLP implements silu SwiGLU only, but hf_config.hidden_act is {act!r}"
        if mesh_config.tp > 1:
            assert ccl_manager is not None, "TP > 1 needs a CCLManager for the down_proj collective"

        # fp32 destination accumulation for the three projections: the 28672-deep down_proj is the
        # deepest contraction in the model. See get_matmul_compute_config for the measured effect.
        self.matmul_config = get_matmul_compute_config(mesh_device)

        col_mapper = mesh_config.column_parallel(mesh_device)  # shard the output (intermediate) dim
        row_mapper = mesh_config.row_parallel(mesh_device)  # shard the input (intermediate) dim

        def _load(name, weight, mapper, dtype):
            # `weight` is None in cache-only mode (empty state_dict) — still build the tensor so
            # ttnn.as_tensor loads the tilized copy straight from the cache. A dense FFN always has
            # all three projections, so return None only when there is no cache path to load from.
            if weight is None and not tensor_cache_path:
                return None
            return ttnn.as_tensor(
                weight,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                mesh_mapper=mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if state_dict:
            # HF stores a Linear weight as [out, in]; ttnn.linear wants [in, out] -> transpose.
            def _prep(key):
                return substate(state_dict, key)["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

            gate_w, up_w, down_w = _prep("gate_proj"), _prep("up_proj"), _prep("down_proj")
        else:
            gate_w = up_w = down_w = None

        # Per-projection dtypes, as the spec's dataformats.weights.mlp block allows.
        gate_dtype = SPEC.mlp_gate_dtype if weight_dtype is None else weight_dtype
        up_dtype = SPEC.mlp_up_dtype if weight_dtype is None else weight_dtype
        down_dtype = SPEC.mlp_down_dtype if weight_dtype is None else weight_dtype
        self.gate_proj = _load("gate_proj", gate_w, col_mapper, gate_dtype)
        self.up_proj = _load("up_proj", up_w, col_mapper, up_dtype)
        self.down_proj = _load("down_proj", down_w, row_mapper, down_dtype)

    def __call__(self, x, activation_dtype=ttnn.bfloat16):
        """x: [1, 1, tokens, hidden] -> [1, 1, tokens, hidden] (replicated across TP after the
        all-reduce), or [1, 1, tokens, hidden/tp] when ``scatter_output`` is set."""
        gate = ttnn.linear(x, self.gate_proj, dtype=activation_dtype, compute_kernel_config=self.matmul_config)
        up = ttnn.linear(x, self.up_proj, dtype=activation_dtype, compute_kernel_config=self.matmul_config)
        # There is no ttnn.swiglu; SwiGLU is silu(gate) * up, which composes exactly from two ops.
        act = ttnn.multiply(ttnn.silu(gate), up)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(act, self.down_proj, dtype=activation_dtype, compute_kernel_config=self.matmul_config)
        act.deallocate(True)

        # down_proj is row-parallel: each TP device holds a partial sum over its intermediate shard,
        # so a TP collective is required for correctness, not for speed.
        if self.mesh_config.tp > 1:
            if self.scatter_output:
                scattered = self.mesh_config.reduce_scatter(out, self.ccl_manager, dim=3, axis=self.mesh_config.tp_axis)
                out.deallocate(True)
                out = scattered
            else:
                out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
