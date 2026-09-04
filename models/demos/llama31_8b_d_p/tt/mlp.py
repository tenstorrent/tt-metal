# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dense SwiGLU FFN for Llama-3.1-8B: `down(silu(gate(x)) * up(x))`, no biases.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaMLP`.
**Template:** `models/demos/minimax_m3/tt/dense_mlp.py:26` — structure taken whole (the
column/row mapper split, the cache-only `_load` branch at `:58-72`, the `scatter_output` flag, and
the TP tail at `:96-112`), with three changes:

1. **Plain SwiGLU.** M3's clamped `swigluoai` activation (`models/demos/minimax_m3/tt/dense_mlp.py:92`,
   which reads `swiglu_limit` and `alpha`) is replaced by `silu(gate) * up`. Llama's `hidden_act` is
   `silu` with no clamp and no alpha (`bringup_log/00_MODEL_CARD.md` §2), so both knobs disappear
   rather than being defaulted.
2. **An explicit `compute_kernel_config` on all three matmuls.**
   `models/demos/minimax_m3/tt/dense_mlp.py:89-90` and `:94` pass none. On a matmul
   `fp32_dest_acc_en=True` is bit-identical to the default and `False` costs 96x-1168x
   (`BRINGUP_RECIPE.md:652-653`), so the risk is inheriting an explicit `False` and the defence is
   passing an explicit `True` from `tt/config.py`'s single factory (`DEC-030`).
3. **`scatter_output=True` refuses.** M3 implements both residual schemes; this package takes
   scheme A (`DEC-025`) and the seam is wired but not honoured until P8 (`DEC-038`), because a
   module returning `emb/TP` while every other module returns full emb is a half-wired scheme, and
   `BRINGUP_RECIPE.md:1207-1209` requires a loud refusal instead.

**No biases.** `mlp_bias` is `false` (`bringup_log/00_MODEL_CARD.md` §2, §3), so unlike
`models/demos/minimax_m3/tt/mlp.py` there is no bias tensor to fail loud about in cache-only mode —
the three projections are the whole weight surface, and an empty `state_dict` with no cache path
raises.

**The TP tail runs only at `tp > 1`** (`models/demos/minimax_m3/tt/dense_mlp.py:99`), so `G-MLP` at
`(1,1)` does not execute a collective; `bringup_log/04_CCL_PLAN.md` §5 row 2 is the P8 wiring.
"""

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.substate import substate

from .config import MeshConfig, default_compute_kernel_config

# Residual scheme A (`DEC-025`): every module returns the full `[1, 1, S_loc, hidden]` residual, so
# the derived value of `scatter_output` is False. Named rather than inlined because the day scheme B
# lands, this constant is the flag `bringup_log/04_CCL_PLAN.md` §6 promises.
_SCHEME_A_SCATTER_OUTPUT = False


class MLP:
    """`down(silu(gate(x)) * up(x))`. gate/up column-parallel, down row-parallel + the TP collective."""

    def __init__(
        self,
        mesh_device,
        hf,
        state_dict,
        *,
        mesh_config=None,
        ccl_manager=None,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        scatter_output=None,
        fused_silu=True,
        fp32_dest_acc_en=True,
    ):
        """
        Args:
            mesh_device: the open mesh.
            hf: the normalised config **dict** (recipe P1 trap 2).
            state_dict: already stripped to this MLP's own keys, i.e.
                `{"gate_proj.weight": ..., "up_proj.weight": ..., "down_proj.weight": ...}` — the
                caller splits with `substate` (`models/demos/gpt_oss_d_p/utils/substate.py:15`).
                May be empty in cache-only mode, which requires `tensor_cache_path`.
            mesh_config: the model's `MeshConfig`; defaults to TP over the whole column axis.
            ccl_manager: the model's `CCLManager`. Required only when `tp > 1`.
            weight_dtype: on-device weight dtype, `bfloat8_b` (`DEC-022`).
            activation_dtype: matmul output dtype, `bfloat16` (`DEC-022`).
            tensor_cache_path: where `ttnn.as_tensor` persists / reloads the tilized weights.
            scatter_output: `None` derives it from the residual scheme (A -> `False`). `True` is
                refused until P8 wires scheme B end to end (`DEC-025`, `DEC-038`).
            fused_silu: fold SiLU into the `gate * up` multiply as `input_tensor_a_activations`
                rather than running a separate `ttnn.silu`. `DEC-039` records the measurement that
                chose the default; `False` exists so `G-MLP` can A/B it in-suite.
            fp32_dest_acc_en: exposed **only** so `G-MLP` can A/B recipe §2.4's flag on the three
                matmuls, the same way `tt/rms_norm.py` does for the norm. The default is the correct
                value; passing `False` is a measurement, never a configuration.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl_manager = ccl_manager
        self.hidden_size = hf["hidden_size"]
        self.intermediate_size = hf["intermediate_size"]
        self.activation_dtype = activation_dtype
        self.fused_silu = fused_silu
        self.compute_kernel_config = default_compute_kernel_config(mesh_device, fp32_dest_acc_en=fp32_dest_acc_en)

        self.scatter_output = _SCHEME_A_SCATTER_OUTPUT if scatter_output is None else scatter_output
        if self.scatter_output:
            # Refuse loudly rather than half-wire scheme B: `MeshConfig.reduce_scatter` would run,
            # but every other module still returns full emb, so the layer's residual add would be
            # comparing a 512-wide tail against a 4096-wide stream (`DEC-025`, `DEC-038`).
            raise NotImplementedError(
                "MLP(scatter_output=True) needs residual scheme B, which is not wired in this "
                "iteration: DEC-025 takes scheme A (replicated full-emb residual) and P8 owns the "
                "switch. bringup_log/04_CCL_PLAN.md section 5 row 4 is the seam."
            )
        if self.mesh_config.tp > 1 and ccl_manager is None:
            raise ValueError(f"MLP at tp={self.mesh_config.tp} needs a ccl_manager for its TP all-reduce")

        assert (
            self.intermediate_size % self.mesh_config.tp == 0
        ), f"intermediate_size {self.intermediate_size} is not divisible by tp {self.mesh_config.tp}"

        if state_dict:
            # HF stores a Linear weight as `[out, in]`; `ttnn.linear` wants `[in, out]`, so
            # transpose at LOAD time and never at runtime
            # (`models/demos/minimax_m3/tt/dense_mlp.py:77`).
            def _prep(name):
                return substate(state_dict, name)["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

            gate_w, up_w, down_w = _prep("gate_proj"), _prep("up_proj"), _prep("down_proj")
        elif not tensor_cache_path:
            # Fail loud rather than build three `None` projections: Appendix B's "cache-only build
            # silently wrong" row. A dense FFN always has all three weights, so there is no
            # optional-weight case to tolerate here.
            raise ValueError("MLP needs either a state_dict with gate/up/down_proj or a tensor_cache_path")
        else:
            gate_w = up_w = down_w = None

        col_mapper = self.mesh_config.column_parallel(mesh_device)  # shard the output (intermediate) dim
        row_mapper = self.mesh_config.row_parallel(mesh_device)  # shard the input (intermediate) dim

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

        self.gate_proj = _load("gate_proj", gate_w, col_mapper)
        self.up_proj = _load("up_proj", up_w, col_mapper)
        self.down_proj = _load("down_proj", down_w, row_mapper)

    def __call__(self, x):
        """`[1, 1, S_loc, hidden]` -> `[1, 1, S_loc, hidden]`, bf16 TILE in and out.

        `x` is **not** deallocated: it is the caller's residual branch
        (`models/demos/minimax_m3/tt/dense_mlp.py:87-94` likewise leaves it live).
        """
        gate = ttnn.linear(
            x, self.gate_proj, dtype=self.activation_dtype, compute_kernel_config=self.compute_kernel_config
        )
        up = ttnn.linear(x, self.up_proj, dtype=self.activation_dtype, compute_kernel_config=self.compute_kernel_config)

        if self.fused_silu:
            # SiLU folded into the multiply's `input_tensor_a` — i.e. onto `gate`, never `up`.
            # `G-MLP`'s negative control is that same call with the arguments swapped, which is what
            # proves the unary is on the argument this comment claims (recipe P5.4).
            act = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        else:
            silu_gate = ttnn.silu(gate)
            act = ttnn.mul(silu_gate, up)
            silu_gate.deallocate(True)
        gate.deallocate(True)
        up.deallocate(True)

        out = ttnn.linear(
            act, self.down_proj, dtype=self.activation_dtype, compute_kernel_config=self.compute_kernel_config
        )
        act.deallocate(True)

        # `down_proj` is row-parallel, so each TP device holds a PARTIAL SUM over its intermediate
        # shard: a TP collective is mandatory, not an optimisation
        # (`bringup_log/04_CCL_PLAN.md` §4). Under scheme A that is a full all-reduce; the
        # reduce-scatter alternative is scheme B's seam and is refused in `__init__` above.
        if self.mesh_config.tp > 1:
            out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
