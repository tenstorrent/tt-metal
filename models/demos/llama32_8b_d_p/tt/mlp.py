# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dense SwiGLU feed-forward network for Llama-3.1-8B-Instruct prefill.

HF anchor: ``transformers.models.llama.modeling_llama.LlamaMLP`` — ``down_proj(silu(gate_proj(x)) *
up_proj(x))``, ``mlp_bias: false`` so there is no bias anywhere.

Template: ``models/demos/minimax_m3/tt/dense_mlp.py:26`` (class), ``:29`` (``__init__``), ``:38``
(``scatter_output``), ``:58`` (the ``_load`` closure), ``:62`` (the cache-only branch), ``:77`` (the
HF ``[out, in]`` -> ttnn ``[in, out]`` transpose), ``:87`` (``__call__``), ``:99`` (``if tp > 1``),
``:105`` (reduce-scatter), ``:112`` (all-reduce).

Three deliberate changes from that template (``03_OUTLINE.md`` §3.6):

1. M3's clamped ``swigluoai`` activation (``dense_mlp.py:92`` -> ``moe/activation.py``) becomes the
   ONE fused op ``ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])``. Llama's
   ``hidden_act`` is plain ``silu`` with no clamp and no alpha; in-tree usage of the fused form is
   ``models/common/modules/mlp/mlp_1d.py:259`` and its default activation is SILU at ``:84``.
   ``hidden_act`` is asserted, not branched on (``03_OUTLINE.md`` §1 convention 12).
2. The ``zone(...)`` profiler wrappers are dropped (``02_SURVEY.md`` §3).
3. ``scatter_output`` defaults to a literal ``False`` — residual scheme **A** (``DEC-018``) — rather
   than being derived from an env-gated ``use_sharded_residual()`` (``dense_mlp.py:48``); this
   package plans exactly two env vars and that is not one of them (``03_OUTLINE.md`` §1
   convention 10). The parameter stays wired so scheme B remains a flag, not a rewrite.

Parallelism (``04_CCL_PLAN.md`` §7 rows 3-4): ``gate_proj`` / ``up_proj`` are column-parallel
(shard the intermediate dim on the TP axis), ``down_proj`` is row-parallel (shard the contraction
dim), so every TP chip holds a partial sum and a TP collective on ``cluster_axis=tp_axis``,
``dim=3`` closes the block. At TP=1 no collective is entered at all.

**Every matmul is given an explicit compute-kernel config with ``fp32_dest_acc_en=True``**
(``DEC-031``), built ONCE per module through ``ttnn.init_device_compute_kernel_config`` (``DEC-013``
— there is no ``ttnn.BlackholeComputeKernelConfig``, Appendix F.8).

Measured on this box, seq 512, and worth stating precisely because the polarity is **not** the one
``DEC-031`` found on ``ttnn.rms_norm`` (``R-021``): for ``ttnn.linear`` the op's own default is
**bit-identical** to HiFi4 + ``fp32_dest_acc_en=True``, so passing nothing loses nothing here.
Passing ``fp32_dest_acc_en=False`` — the template's own default,
``models/demos/gpt_oss_d_p/tt/attention/config.py:71`` — costs **96x** (bf8_b weights, 0.9925 vs
0.9999) to **1168x** (bf16, 0.9918 vs 0.99999) of the measured error. So the config is explicit to
pin the good behaviour against a default change and to keep the template's ``False`` from being
copied forward, not to fix a current regression. Full A/B in ``06_GATES.md`` -> ``G-MLP``.
"""

from __future__ import annotations

import ttnn
from models.demos.llama32_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama32_8b_d_p.utils.substate import substate

# Llama-3.1's only activation. Asserted rather than branched on: `hidden_act` is a field on
# LlamaHFConfig (DEC-009 / DEC-025), and anything other than silu would need a different fused
# op than the one below.
_SUPPORTED_HIDDEN_ACT = "silu"


def default_compute_kernel_config(mesh_device):
    """The package's default matmul compute-kernel config: HiFi4 + ``fp32_dest_acc_en=True``.

    Built through ``ttnn.init_device_compute_kernel_config`` rather than by naming a class:
    ``ttnn.BlackholeComputeKernelConfig`` does not exist (``hasattr`` is ``False``;
    ``ttnn/ttnn/__init__.py:305`` exports only the Wormhole name) and where it *is* defined it is
    the same object (``ttnn/ttnn/types.py:61``), so an arch branch is a no-op — ``DEC-013``.

    ``fp32_dest_acc_en=True`` is the load-bearing field, not ``math_fidelity``: measured on this
    Blackhole, raising fidelity alone changes nothing while enabling fp32 destination accumulation
    removes most of the residual error (``DEC-031``). The attention SP ring path is the one place it
    must stay ``False`` (``models/demos/gpt_oss_d_p/tt/attention/prefill.py:200``); that is P8's,
    and it is why this is a function and not a constant.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class MLP:
    """Dense SwiGLU FFN: ``down(silu(gate(x)) * up(x))``.

    Column-parallel ``gate``/``up``, row-parallel ``down``, then the TP collective.
    """

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        *,
        mesh_config,
        ccl_manager=None,
        tensor_cache_path=None,
        weight_dtype=ttnn.bfloat8_b,
        scatter_output=False,
        compute_kernel_config=None,
    ):
        """
        Args:
            mesh_device: the ttnn mesh device.
            hf_config: a ``LlamaHFConfig`` (``tt/model_config.py``). An OBJECT, never a dict, and
                never read through ``getattr(..., default)`` (``DEC-009``, Appendix F.2).
            state_dict: the already-stripped ``mlp.*`` sub-dict (``substate(sd, "mlp")`` is the
                caller's job — ``03_OUTLINE.md`` §1 convention 4). ``{}`` means cache-only mode.
            mesh_config: ``MeshConfig``; supplies the mappers and the collectives.
            ccl_manager: ``CCLManager``; only touched when ``mesh_config.tp > 1``.
            tensor_cache_path: directory for the tilized weight cache, or ``None``.
            weight_dtype: on-device weight dtype (default ``bfloat8_b``; Appendix E measured the
                existing implementation at 0.9995823 there).
            scatter_output: ``False`` (default, scheme A) closes with a TP all-reduce and returns
                the full ``hidden_size``. ``True`` (scheme B) closes with the reduce-scatter half
                only and returns ``hidden_size / tp``, which the caller adds into its own sharded
                residual. ``DEC-018``.
            compute_kernel_config: passed to all three matmuls. ``None`` builds
                :func:`default_compute_kernel_config` (HiFi4, ``fp32_dest_acc_en=True``). Pass an
                explicit config to A/B the precision; never leave the op's own default in place
                (``DEC-031``).
        """
        assert (
            hf_config.hidden_act == _SUPPORTED_HIDDEN_ACT
        ), f"tt/mlp.py implements SwiGLU with {_SUPPORTED_HIDDEN_ACT!r}; hf_config.hidden_act is {hf_config.hidden_act!r}"
        assert not hf_config.mlp_bias, "Llama-3.1 has mlp_bias: false; this MLP has no bias path"

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        self.scatter_output = scatter_output
        self.compute_kernel_config = (
            default_compute_kernel_config(mesh_device) if compute_kernel_config is None else compute_kernel_config
        )

        # Both collective tails scatter `hidden_size` on the TP axis (dim 3), and the reduce-scatter
        # half of an all-reduce does too, so a non-tile-aligned hidden/tp would put output-dim
        # padding inside one TP column's slice. For Llama 4096/tp is tile-aligned for every
        # admissible tp (00_MODEL_CARD.md §4.3), which is exactly why the gpt-oss o_proj padding
        # path is deleted throughout this package (03_OUTLINE.md §3.8).
        if mesh_config.tp > 1:
            assert (self.hidden_size // mesh_config.tp) % ttnn.TILE_SIZE == 0, (
                f"hidden_size/tp = {self.hidden_size}/{mesh_config.tp} = "
                f"{self.hidden_size // mesh_config.tp} is not a multiple of TILE_SIZE "
                f"({ttnn.TILE_SIZE}); the TP collective would scatter padding into one column"
            )
            assert (self.intermediate_size % mesh_config.tp) == 0, (
                f"intermediate_size {self.intermediate_size} is not divisible by tp {mesh_config.tp}; "
                f"gate/up column-parallel sharding would be ragged"
            )

        col_mapper = mesh_config.column_parallel(mesh_device)  # shard the output (intermediate) dim
        row_mapper = mesh_config.row_parallel(mesh_device)  # shard the input (intermediate) dim

        def _load(name, weight, mapper):
            # weight is None in cache-only mode (empty state_dict) — still build, so ttnn.as_tensor
            # reads the tilized tensor straight from disk. A dense FFN always has all three
            # projections, so return None only when there is no cache to load from either.
            if weight is None and not tensor_cache_path:
                return None
            return ttnn.as_tensor(
                weight,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if state_dict:
            # HF stores nn.Linear weight as [out, in]; ttnn.linear wants [in, out]. Transposed once
            # here at load time, never at runtime (03_OUTLINE.md §1 convention 6).
            def _prep(key):
                sub = substate(state_dict, key)
                assert "bias" not in sub, f"{key} carries a bias, but Llama-3.1 has mlp_bias: false"
                return sub["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

            gate_w, up_w, down_w = _prep("gate_proj"), _prep("up_proj"), _prep("down_proj")
        else:
            gate_w = up_w = down_w = None

        self.gate_proj = _load("gate_proj", gate_w, col_mapper)
        self.up_proj = _load("up_proj", up_w, col_mapper)
        self.down_proj = _load("down_proj", down_w, row_mapper)

    def __call__(self, x):
        """``[1, 1, S_loc, hidden]`` -> ``[1, 1, S_loc, hidden]`` (scheme A) or
        ``[1, 1, S_loc, hidden/tp]`` (scheme B)."""
        gate = ttnn.linear(x, self.gate_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)

        # The ONE fused op: SiLU on `gate` folded into the elementwise multiply, so the whole
        # activation is a single device op with no intermediate SiLU tensor. Consumes neither
        # input in place — deallocate both explicitly.
        act = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16)
        gate.deallocate(True)
        up.deallocate(True)

        out = ttnn.linear(act, self.down_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
        act.deallocate(True)

        # down_proj is row-parallel, so each TP chip holds a partial sum over its slice of the
        # intermediate dim: a TP collective is mandatory either way. Scheme A -> full all-reduce
        # (RS + AG, full hidden out); scheme B -> the reduce-scatter half only (hidden/tp out).
        if self.mesh_config.tp > 1:
            if self.scatter_output:
                scattered = self.mesh_config.reduce_scatter(out, self.ccl_manager, dim=3, axis=self.mesh_config.tp_axis)
                out.deallocate(True)
                out = scattered
            else:
                # `MeshConfig.allreduce` frees its own input between the RS and the AG
                # (tt/config.py:134); do not deallocate `out` again after this.
                out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
