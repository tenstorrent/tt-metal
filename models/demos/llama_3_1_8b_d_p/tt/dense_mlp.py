# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B dense MLP: `down(silu(gate(x)) * up(x))` at intermediate 14336.

Structure borrowed from `minimax_m3/tt/dense_mlp.py` (its dense layers 0-2, measured at hidden 6144
/ dense intermediate 12288 / sp8×tp4 on this mesh):

* `gate_proj` / `up_proj` are **column-parallel** — they shard the intermediate dim across TP, so
  each chip computes `intermediate/tp = 3584` columns from a full-width input.
* `down_proj` is **row-parallel** — it shards the intermediate (contraction) dim, so every TP chip
  ends holding a PARTIAL SUM over its shard. A TP collective is therefore required either way:
  reduce-scatter under a sharded residual (out `emb/tp`, which the caller adds straight into its
  residual), all-reduce under the replicated one (out full emb).

## What was NOT borrowed: the activation

The donor applies clamped **swigluoai** (α=1.702, clamp limit 7.0) — the gpt-oss activation, shared
by minimax_m3's dense MLP and its experts. Llama's `hidden_act` is plain `silu` and its MLP is
`down(silu(gate) * up)` with no clamp and no alpha. Porting the donor's structure while carrying its
activation over would be a silent accuracy loss, so the activation is written fresh here;
`tests/torch_ref/test_llama_reference.py::test_mlp_activation_is_plain_silu_swiglu` pins the
distinction on the reference side.

There is no `ttnn.mlp`, and that is fine — an MLP is a matmul, an activation and a second matmul, so
D3 composes it from `ttnn.linear` / `ttnn.silu` / `ttnn.multiply` rather than reaching for a torch
fallback.

## Numerics

Every projection matmul takes an explicit HiFi4 `compute_kernel_config` with
`fp32_dest_acc_en=True` (the bring-up default, recipe §2.3). This is not decoration: at bf16
destination accumulation a deep contraction caps a decoder layer around PCC 0.989, and the
accumulation dtype dominates both weight dtype and math fidelity. `down_proj` contracts over 14336,
which is the deepest contraction in the layer.
"""

import ttnn
from models.demos.llama_3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama_3_1_8b_d_p.utils.substate import substate

from .attention.operations import assert_sharded_residual_unpadded
from .residual import use_sharded_residual


def hifi4_compute_config():
    """HiFi4 + fp32 destination accumulation — the bring-up default for every projection matmul.

    A narrower setting is a MEASUREMENT, not an inheritance: if it is ever used, record what it
    costs. Note the donors pin their SDPA to `fp32_dest_acc_en=False`; that constraint belongs only
    to the ring cache-read op's streaming online softmax, not to matmuls and not to SDPA in general.
    """
    return ttnn.WormholeComputeKernelConfig(  # name is historical; this is the BH config too
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class DenseMLP:
    """Llama's per-layer SwiGLU FFN."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        scatter_output=None,
    ):
        """
        Args:
            state_dict: this MLP's substate, i.e. `{gate_proj,up_proj,down_proj}.weight`, or `{}`
                for cache-only loading.
            weight_dtype: from the spec's `dataformats.weights` (default `bfloat8_b`).
            scatter_output: True => close with a TP reduce-scatter (output `emb/tp`, the
                sharded-residual contract); False => all-reduce (output full emb). None derives it
                from the residual scheme, which is what the layer wants.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        self.weight_dtype = weight_dtype
        self.tensor_cache_path = tensor_cache_path
        self.state_dict = state_dict
        self.scatter_output = use_sharded_residual() if scatter_output is None else scatter_output
        assert hf_config.hidden_act == "silu", f"expected silu, got {hf_config.hidden_act!r}"
        assert not getattr(hf_config, "mlp_bias", False), "Llama's MLP projections are bias-free"
        assert self.intermediate_size % mesh_config.tp == 0, (
            f"intermediate {self.intermediate_size} must divide tp {mesh_config.tp}"
        )
        self.compute_kernel_config = hifi4_compute_config()
        self.gate_proj = self.up_proj = self.down_proj = None
        self._load_weights()

    def _load_weights(self):
        """Tilize gate/up (column-parallel) and down (row-parallel) onto the mesh.

        HF stores a Linear weight as `[out, in]` and `ttnn.linear` wants `[in, out]`, so each is
        transposed on the way in.
        """
        col_mapper = self.mesh_config.column_parallel(self.mesh_device)  # shard the output dim
        row_mapper = self.mesh_config.row_parallel(self.mesh_device)  # shard the input dim

        def prep(name):
            key = f"{name}.weight"
            if not self.state_dict or key not in self.state_dict:
                return None
            return self.state_dict[key].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        def load(name, mapper):
            weight = prep(name)
            # `weight is None` is cache-only mode: as_tensor still loads the tilized tensor straight
            # from disk. With neither a weight nor a cache path there is nothing to build.
            if weight is None and not self.tensor_cache_path:
                return None
            return ttnn.as_tensor(
                weight,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.weight_dtype,
                mesh_mapper=mapper,
                cache_file_name=get_cache_file_name(self.tensor_cache_path, name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.gate_proj = load("gate_proj", col_mapper)
        self.up_proj = load("up_proj", col_mapper)
        self.down_proj = load("down_proj", row_mapper)

    def __call__(self, x):
        """x [1, 1, tokens_local, hidden] -> `emb/tp` (scatter_output) or full emb."""
        # No ttnn.mlp exists, and none is needed: this IS the MLP, composed from the ops that do.
        gate = ttnn.linear(
            x, self.gate_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
        )
        up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
        # Plain SiLU SwiGLU — NOT the donors' clamped swigluoai. See the module docstring.
        act = ttnn.multiply(ttnn.silu(gate), up)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(
            act, self.down_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
        )
        act.deallocate(True)

        # down is row-parallel: each TP device holds a partial sum over its intermediate shard, so a
        # TP collective is required either way. Sharded residual -> reduce-scatter only (emb/tp out,
        # which the caller adds straight into its residual); replicated -> full all-reduce.
        if self.mesh_config.tp > 1:
            if self.scatter_output:
                # A non-tile-aligned hidden/tp would land output-dim padding inside one TP column's
                # residual slice after the scatter, where nothing masks it off.
                assert_sharded_residual_unpadded(self.mesh_config, self.hidden_size)
                scattered = self.mesh_config.reduce_scatter(
                    out, self.ccl_manager, dim=3, axis=self.mesh_config.tp_axis
                )
                out.deallocate(True)
                out = scattered
            else:
                out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
