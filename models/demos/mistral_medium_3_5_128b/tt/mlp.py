# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 dense SwiGLU MLP. Ported from ``minimax_m3/tt/dense_mlp.py``.

    down(silu(gate(x)) * up(x))     hidden 12288 -> intermediate 28672 -> hidden 12288

Differences from the source, all because this model is simpler:

* **Plain ``silu``**, not M3's clamped ``swigluoai`` (``hidden_act: "silu"``, no ``swiglu_limit`` /
  ``swiglu_alpha`` in the config). ``ttnn.silu`` is the exact op, so there is nothing to compose.
* **No sharded-residual branch.** The residual stream here is hidden-replicated across TP (see
  ``tt/rms_norm.py``), so ``down_proj``'s row-parallel tail always closes with a full TP all-reduce.
  M3's ``scatter_output`` mode belongs to a hidden-sharded residual and has no caller here.
* **Every layer is dense.** No MoE, no shared expert, no router — all 88 layers use this block.

``gate_proj`` / ``up_proj`` are column-parallel (the intermediate dim shards over TP: 28672/4 = 7168);
``down_proj`` is row-parallel over the same dim, so each chip produces a partial sum over hidden and
the all-reduce completes it.
"""

import ttnn
from models.demos.mistral_medium_3_5_128b.tt.compute import matmul_compute_kernel_config
from models.demos.mistral_medium_3_5_128b.utils.general_utils import get_cache_file_name
from models.demos.mistral_medium_3_5_128b.utils.substate import substate


class MLP:
    """Dense SwiGLU feed-forward network, TP-sharded on the intermediate dimension."""

    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        *,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: the open mesh.
            config: a :class:`~...reference.model_config.MistralMediumConfig`; ``hidden_act`` must be
                ``"silu"`` (asserted) and ``intermediate_size`` must divide TP.
            state_dict: ``{gate,up,down}_proj.weight`` in HF ``[out, in]`` layout. Empty => cache-only.
            mesh_config: :class:`~...tt.config.MeshConfig`.
            ccl_manager: required when ``mesh_config.tp > 1`` (the ``down_proj`` all-reduce).
            weight_dtype: on-device weight dtype (the spec's ``dataformats.weights``: bfloat8_b).
            tensor_cache_path: directory for the tilized-weight cache, or None.
        """
        assert config.hidden_act == "silu", f"this MLP implements silu only, config says {config.hidden_act}"
        assert config.intermediate_size % mesh_config.tp == 0, (
            f"intermediate_size ({config.intermediate_size}) must divide TP ({mesh_config.tp}); a ragged "
            f"column split has no valid mesh mapping"
        )
        assert mesh_config.tp == 1 or ccl_manager is not None, "down_proj's row-parallel tail needs a CCLManager"

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = config.hidden_size

        col_mapper = mesh_config.column_parallel(mesh_device)  # shard the output (intermediate) dim
        row_mapper = mesh_config.row_parallel(mesh_device)  # shard the input (intermediate) dim

        def _prep(key):
            # HF stores Linear weight as [out, in]; ttnn.linear wants [in, out].
            return substate(state_dict, key)["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        def _load(name, weight, mapper):
            return ttnn.as_tensor(
                weight,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        gate_w, up_w, down_w = (_prep(k) for k in ("gate_proj", "up_proj", "down_proj")) if state_dict else (None,) * 3
        self.gate_proj = _load("gate_proj", gate_w, col_mapper)
        self.up_proj = _load("up_proj", up_w, col_mapper)
        self.down_proj = _load("down_proj", down_w, row_mapper)

    def __call__(self, x):
        """``x``: ``[1, 1, tokens_local, hidden_size]`` -> same shape (post all-reduce)."""
        # HiFi4 + fp32 accumulate on all three matmuls: omitting the config selects ttnn's LoFi
        # default, whose per-layer error is invisible per block and fatal over 88 (tt/compute.py).
        ckc = matmul_compute_kernel_config()
        gate = ttnn.linear(x, self.gate_proj, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
        up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
        act = ttnn.mul(ttnn.silu(gate), up)
        gate.deallocate(True)
        up.deallocate(True)

        out = ttnn.linear(act, self.down_proj, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
        act.deallocate(True)

        # down_proj is row-parallel: each TP column holds a partial sum over hidden, so the sum is
        # only complete after the collective. The residual stream is hidden-replicated, so this is
        # a full all-reduce (never a reduce-scatter).
        if self.mesh_config.tp > 1:
            reduced = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
            out.deallocate(True)
            out = reduced
        return out
