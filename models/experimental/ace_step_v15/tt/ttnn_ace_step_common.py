# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for the ACE-Step 1.5 bringup (Block 1 owns this file).

Blocks 2 (condition encoder) and 3 (VAE decoder) import from here; the API below is
intended to be stable. Everything in this module is device-agnostic apart from needing a
1x1 ``ttnn.MeshDevice`` (``models.tt_dit.layers.module.Parameter`` requires a mesh device,
so open the single Wormhole as ``ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))``).

Contents
--------
``AceStepDiTConfig``            the 2 B DiT config, with the derived SDPA window and layer types.
``linear_compute_config`` etc.  compute-kernel configs, built once per module constructor.
``make_rms_norm`` / ``make_linear``
                                thin wrappers over ``models.tt_dit.layers`` that pin the
                                conventions ACE-Step needs (no norm bias, no linear bias).
``Modulation``                  the folded adaLN constants for one block (see `fold` notes).
``rms_norm_modulated``          ``RMSNorm(x) * (1 + scale) + shift`` in one place.
``build_rope_tables``           HF half-split cos/sin for a given sequence length.
``apply_rope``                  RoPE application (device op, with a composite fallback).
``timestep_sinusoid``           the 256-wide sinusoid, computed on the **host in fp32**.
``fold_time_embed_r``           runs ``time_embed_r`` at ``t - t_r == 0`` on the host.
``conv1d_patch_to_linear`` / ``conv_transpose1d_patch_to_linear``
                                the ``proj_in`` / ``proj_out`` weight folds (kernel == stride).

Verified traps encoded here
---------------------------
* **TRAP-1** ``AceStepDiTConfig.sdpa_window_size`` is ``2 * sliding_window``. TTNN's
  ``sliding_window_size`` is the *total* window width (``|i-j| <= W/2``); the value in
  ``config.json`` (128) is the ``|i-j|`` bound, so 256 must be passed. Confirmed both by
  on-device probe and statically in
  ``ttnn/cpp/.../sdpa/device/kernels/sliding_window_geometry.hpp`` (``half_window = W / 2``
  for the non-causal branch).
* **TRAP-2** GQA is handled in-kernel by SDPA; nothing here repeats K/V.
* RoPE is the **HF half-split** convention (``rotate_half``), *not* the interleaved one.
  ``ttnn.alt_complex_rotate90`` (used by ``models/tt_dit/blocks/attention.py::_apply_rope``)
  implements the *interleaved* convention and is therefore **wrong** for ACE-Step.
* The timestep sinusoid must be evaluated in fp32. The reference feeds ``t * 1000`` into
  ``cos``/``sin``; in bfloat16 the argument (~950 rad) carries an absolute error of ~2 rad,
  which destroys the embedding. See ``timestep_sinusoid``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Sequence

import torch

import ttnn
from models.tt_dit.layers.feedforward import FeedForward
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, Parameter
from models.tt_dit.layers.normalization import RMSNorm

if TYPE_CHECKING:  # pragma: no cover
    pass

TILE = 32


def layer_types_for(num_layers: int) -> tuple[str, ...]:
    """The ACE-Step layer-type alternation for a stack of ``num_layers``.

    Reference: ``"sliding_attention" if bool((i + 1) % 2) else "full_attention"``, i.e.
    **EVEN** layers (0, 2, ...) use the symmetric ``|i-j| <= sliding_window`` band and **ODD**
    layers are global (``mask=None``).

    A free function rather than only a config method, because the condition encoder's stacks
    (8 lyric + 4 timbre layers) share this rule with the 24-layer DiT and should not have to
    fabricate a ``num_hidden_layers`` to ask about it.
    """
    return tuple("sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(num_layers))


# --------------------------------------------------------------------------------------- #
#                                        config                                            #
# --------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AceStepDiTConfig:
    """Config for the 2 B ACE-Step 1.5 DiT (``acestep-v15-turbo``).

    Field names and defaults mirror ``diffusers.AceStepTransformer1DModel.__init__`` so a
    deployed ``config.json`` can be splatted in directly.
    """

    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    in_channels: int = 192
    audio_acoustic_hidden_dim: int = 64
    patch_size: int = 2
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    sliding_window: int = 128
    layer_types: tuple[str, ...] | None = None
    encoder_hidden_size: int | None = None
    time_embed_in_channels: int = 256
    time_embed_scale: float = 1000.0

    #: Number of adaLN chunks per block: 3 for self-attn + 3 for the MLP. Cross-attn is
    #: unmodulated, so there is no 7th/8th chunk. A ``ClassVar`` rather than a field so
    #: ``dataclasses.replace`` stays usable.
    num_modulation_chunks: ClassVar[int] = 6

    def __post_init__(self) -> None:
        # Every dimension in this model is a clean multiple of the tile size; if that ever
        # stops being true the whole "no padding in the transformer" assumption goes with it.
        for name in (
            "hidden_size",
            "intermediate_size",
            "head_dim",
            "in_channels",
            "audio_acoustic_hidden_dim",
        ):
            value = getattr(self, name)
            assert value % TILE == 0, f"{name}={value} must be a multiple of {TILE}"
        assert self.hidden_size == self.num_attention_heads * self.head_dim
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @classmethod
    def from_diffusers_config(cls, config: dict, **overrides) -> AceStepDiTConfig:
        """Build from a deployed ``transformer/config.json`` (or ``meta['transformer_config']``).

        Unknown keys (``_class_name``, ``is_turbo``, ``attention_dropout``, ...) are ignored,
        and ``layer_types`` is normalised to a tuple so the dataclass stays hashable.
        """
        fields = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in config.items() if k in fields}
        if isinstance(kwargs.get("layer_types"), list):
            kwargs["layer_types"] = tuple(kwargs["layer_types"])
        kwargs.update(overrides)
        return cls(**kwargs)

    # -- derived ------------------------------------------------------------------------ #

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def qkv_width(self) -> int:
        return self.q_width + 2 * self.kv_width

    @property
    def attention_scale(self) -> float:
        return self.head_dim**-0.5

    @property
    def cross_attention_input_dim(self) -> int:
        return self.encoder_hidden_size if self.encoder_hidden_size is not None else self.hidden_size

    @property
    def sdpa_window_size(self) -> int:
        """TRAP-1: TTNN's ``sliding_window_size`` is the TOTAL window width.

        ACE-Step's ``create_4d_mask`` keeps ``|i - j| <= sliding_window`` (128), and the
        non-causal TTNN kernel keeps ``|i - j| <= sliding_window_size / 2``. Passing 128
        straight through halves the window and scores PCC ~0.762 against the correct band
        while looking perfectly healthy.
        """
        window = 2 * self.sliding_window
        assert window // 2 == self.sliding_window, "sdpa_window_size must halve back to sliding_window"
        return window

    def resolved_layer_types(self) -> tuple[str, ...]:
        if self.layer_types is not None:
            assert len(self.layer_types) == self.num_hidden_layers
            return tuple(self.layer_types)
        return layer_types_for(self.num_hidden_layers)

    def is_sliding(self, layer_index: int) -> bool:
        return self.resolved_layer_types()[layer_index] == "sliding_attention"

    def window_for_layer(self, layer_index: int) -> int | None:
        """``sdpa_window_size`` for sliding layers, ``None`` (no mask at all) for global ones."""
        return self.sdpa_window_size if self.is_sliding(layer_index) else None


# --------------------------------------------------------------------------------------- #
#                                  compute kernel configs                                  #
# --------------------------------------------------------------------------------------- #


def linear_compute_config(mesh_device: ttnn.MeshDevice, dtype: ttnn.DataType = ttnn.bfloat16):
    """The tt_dit house matmul config (HiFi2 + fp32 dest acc + packer L1 acc for bf16)."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4 if dtype == ttnn.float32 else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def norm_compute_config(mesh_device: ttnn.MeshDevice):
    """RMSNorm: the reference computes the variance in fp32 on both paths, so do the same."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def sdpa_compute_config(mesh_device: ttnn.MeshDevice):
    """HiFi4, **no** fp32 dest accumulation. ``exp_approx_mode`` is left off in the program config.

    TRAP-13: ``fp32_dest_acc_en=True`` together with ``sliding_window_size`` **deadlocks**
    ``ttnn.transformer.scaled_dot_product_attention`` -- the op enqueues, then the first readback
    of its output never returns and the watcher reports no stuck core. Isolated matrix on
    Wormhole b0, q[1,16,128,128] / k,v[1,8,128,128]:

        fp32=0 packer=0 window=256  -> ok        fp32=1 packer=0 window=256  -> STALL
        fp32=0 packer=1 window=256  -> ok        fp32=1 packer=1 window=none -> ok
                                                 HiFi4 fp32=1 window=256     -> STALL

    So either knob alone is safe; it is specifically fp32-dest-acc **plus** a window. Math
    fidelity and ``packer_l1_acc`` are irrelevant. The window is required for the sliding layers
    (TRAP-1 -- dropping it silently degrades PCC to 0.762), so fp32 accumulation is what goes.
    Do not set it back to True here without re-checking that the stall is fixed upstream.
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def sdpa_program_config(mesh_device: ttnn.MeshDevice, seq_len: int, *, chunk_size: int = 256) -> ttnn.SDPAProgramConfig:
    """Optional explicit flash tiling. NOT used by default.

    The TRAP-1 window measurements were taken with ``program_config=None`` (op-chosen
    tiling), so the modules default to that. This helper exists for the perf phase; note
    that ``q_chunk_size``/``k_chunk_size`` interact with which blocks the sliding-window
    stamp can skip, so re-run the PCC gates after enabling it.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    padded = -(-seq_len // TILE) * TILE
    chunk = min(padded, chunk_size)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )


# --------------------------------------------------------------------------------------- #
#                                    module factories                                      #
# --------------------------------------------------------------------------------------- #


def make_rms_norm(
    dim: int, *, eps: float, mesh_device: ttnn.MeshDevice, dtype: ttnn.DataType = ttnn.bfloat16
) -> RMSNorm:
    """``diffusers.models.normalization.RMSNorm(dim, eps)`` — affine weight, **no bias**.

    ``models.tt_dit.layers.normalization.RMSNorm`` defaults to ``bias=True``; ACE-Step never
    has a norm bias (neither the hidden-size norms nor the head_dim QK norms), so it must be
    disabled or ``load_torch_state_dict`` reports a missing ``bias`` key.
    """
    return RMSNorm(embedding_dim=dim, norm_eps=eps, bias=False, mesh_device=mesh_device, dtype=dtype)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    mesh_device: ttnn.MeshDevice,
    activation_fn: str | None = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> Linear:
    return Linear(
        in_features,
        out_features,
        bias=bias,
        activation_fn=activation_fn,
        dtype=dtype,
        mesh_device=mesh_device,
    )


class AceStepMLP(FeedForward):
    """SwiGLU MLP: ``down_proj(silu(gate_proj(x)) * up_proj(x))``, no bias.

    Shared by the DiT block and the condition encoder's layers (identical shapes and identical
    checkpoint key names), which is why it lives here rather than in ``ttnn_ace_step_dit``.

    Uses ``tt_dit``'s fused-SwiGLU ``Linear`` (``activation_fn="swiglu"``), which packs the two
    ``hidden -> inner`` projections into one ``hidden -> 2*inner`` matmul and applies
    ``silu(gate) * up`` inside the kernel.

    ``_prepare_torch_state`` builds the packed weight as ``cat([up_proj, gate_proj], dim=0)``.
    That order is required, not cosmetic: ``models/tt_dit/utils/tensor.py::
    prepare_for_fused_swiglu`` defaults to ``gate_is_first=False``, i.e. it expects
    ``[up | gate]`` (the torch/HuggingFace convention) and re-interleaves gate into the even
    tile slot itself. Verified on the host: the correct order reproduces
    ``silu(gate_proj(x)) * up_proj(x)`` at max|diff| = 0.0, the swapped order is off by 418.

    ``dtype`` is threaded to both Linears (``tt_dit``'s ``FeedForward`` hard-wires bfloat16),
    so an fp32 precision bisect covers the MLP too.
    """

    def __init__(
        self,
        config: AceStepDiTConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__(
            dim=config.hidden_size,
            dim_out=config.hidden_size,
            inner_dim=config.intermediate_size,
            activation_fn="swiglu",
            bias=False,
            mesh_device=mesh_device,
        )
        # FeedForward does not thread dtype to its Linears; rebuild them. Parameters are lazy
        # (no device allocation until load), so this costs nothing.
        self.dtype = dtype
        self.ff1 = make_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            mesh_device=mesh_device,
            activation_fn="swiglu",
            dtype=dtype,
        )
        self.ff2 = make_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        up = state.pop("up_proj.weight", None)
        gate = state.pop("gate_proj.weight", None)
        down = state.pop("down_proj.weight", None)
        if up is not None and gate is not None:
            state["ff1.weight"] = torch.cat([up, gate], dim=0)
        if down is not None:
            state["ff2.weight"] = down


# --------------------------------------------------------------------------------------- #
#                                    adaLN modulation                                      #
# --------------------------------------------------------------------------------------- #


class Modulation(Module):
    """The per-site adaLN constants of one block, with three folds baked in.

    The reference computes, per block::

        shift, scale, gate, c_shift, c_scale, c_gate = (scale_shift_table + temb).chunk(6, dim=1)

    where ``temb = timestep_proj_t + timestep_proj_r`` and ``timestep_proj_r`` is a *constant*
    at inference (``time_embed_r`` always sees ``t - t_r == 0``). Three folds follow:

    1.  ``timestep_proj_r`` is folded into ``scale_shift_table`` at load, so ``time_embed_r``
        never runs (§3.7 of the master doc).
    2.  the ``dim=1`` chunk of a size-6 axis is replaced by six separate ``[1, 1, 1, H]``
        parameters, so nothing is sliced on a non-tile-aligned axis at runtime.
    3.  the ``+1`` of ``x * (1 + scale) + shift`` is folded into the *scale* constants, so the
        forward pass emits ``one_plus_scale`` directly (48 saved elementwise adds per step).

    Consequently ``forward()`` returns, in order,
    ``(shift, one_plus_scale, gate, c_shift, one_plus_c_scale, c_gate)`` — note the second and
    fifth entries are ``1 + scale``, **not** ``scale``.

    Gates are **bare**: the caller must compute ``x + o * gate``, never ``o * (1 + gate)``.
    """

    #: Parameter names in reference chunk order. ``*_scale`` entries carry the folded ``+1``.
    NAMES_6 = ("shift", "one_plus_scale", "gate", "c_shift", "one_plus_c_scale", "c_gate")
    #: ``norm_out`` uses a 2-way table: ``shift, scale = (scale_shift_table + temb).chunk(2)``.
    NAMES_2 = ("shift", "one_plus_scale")

    def __init__(
        self,
        hidden_size: int,
        *,
        num_chunks: int = 6,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        assert num_chunks in (2, 6), f"unsupported modulation width {num_chunks}"
        self.hidden_size = hidden_size
        self.num_chunks = num_chunks
        self.names = self.NAMES_6 if num_chunks == 6 else self.NAMES_2
        for name in self.names:
            setattr(
                self,
                name,
                Parameter(total_shape=[1, 1, 1, hidden_size], device=mesh_device, dtype=dtype),
            )

    def forward(self, per_step: Sequence[ttnn.Tensor]) -> list[ttnn.Tensor]:
        """Add the per-step timestep projection chunks to the folded constants.

        ``per_step`` must have ``num_chunks`` entries, each ``[1, 1, 1, hidden_size]``. For the
        2-way (``norm_out``) case both entries are the *same* ``temb`` tensor, matching the
        reference's ``temb.unsqueeze(1)`` broadcast against a ``[1, 2, H]`` table.
        """
        assert len(per_step) == self.num_chunks, f"expected {self.num_chunks} chunks, got {len(per_step)}"
        return [ttnn.add(getattr(self, name).data, chunk) for name, chunk in zip(self.names, per_step)]

    # -- weight preparation ------------------------------------------------------------- #

    @staticmethod
    def fold_state(
        scale_shift_table: torch.Tensor,
        timestep_proj_r: torch.Tensor | None,
        *,
        num_chunks: int,
    ) -> dict[str, torch.Tensor]:
        """Build the ``{name: [1, 1, 1, H]}`` state for one ``Modulation``.

        Args:
            scale_shift_table: the reference parameter, ``[1, num_chunks, H]``.
            timestep_proj_r: the constant ``time_embed_r`` output for this site,
                ``[num_chunks, H]`` (6-way) or ``[H]`` (2-way ``temb_r``). ``None`` skips
                the fold, which is what a caller wants when it intends to run
                ``time_embed_r`` explicitly.
            num_chunks: 6 for a transformer block, 2 for ``norm_out``.
        """
        names = Modulation.NAMES_6 if num_chunks == 6 else Modulation.NAMES_2
        table = scale_shift_table.reshape(num_chunks, -1).to(torch.float32)
        hidden = table.shape[-1]

        if timestep_proj_r is None:
            r = torch.zeros(num_chunks, hidden, dtype=torch.float32)
        else:
            r = timestep_proj_r.to(torch.float32).reshape(-1, hidden)
            if r.shape[0] == 1 and num_chunks != 1:
                # The 2-way norm_out site adds the SAME temb to both chunks.
                r = r.expand(num_chunks, hidden)
            assert r.shape[0] == num_chunks, f"timestep_proj_r has {r.shape[0]} chunks, expected {num_chunks}"

        folded = table + r
        state: dict[str, torch.Tensor] = {}
        for i, name in enumerate(names):
            value = folded[i]
            if name.startswith("one_plus_"):
                value = value + 1.0  # fold (1 + scale)
            state[name] = value.reshape(1, 1, 1, hidden).contiguous()
        return state


def rms_norm_modulated(
    norm: RMSNorm,
    x: ttnn.Tensor,
    one_plus_scale: ttnn.Tensor,
    shift: ttnn.Tensor,
    *,
    compute_kernel_config=None,
    fused: bool = False,
) -> ttnn.Tensor:
    """``RMSNorm(x) * (1 + scale) + shift``.

    Two implementations:

    ``fused=False`` (default): plain ``norm`` then ``multiply`` then ``add``. Only documented
    ops, matches the reference's ``.type_as(hidden_states)`` semantics (modulation in the
    activation dtype).

    ``fused=True``: folds the modulation into the fused norm kernel's affine terms —
    ``dit_rms_norm_unary_fused(x, weight=norm_w * (1 + scale), bias=shift)`` — saving two
    elementwise passes per adaLN site (144 per DiT forward) and doing the modulation in the
    kernel's fp32 internals. **Not yet validated on hardware**; ``test_dit_ops_pcc.py``
    measures both so the default can be flipped once it is.
    """
    if not fused:
        h = norm(x, compute_kernel_config=compute_kernel_config)
        # In-place on the norm output: the modulation would otherwise leak two full-size
        # activations per adaLN site (~290 MB across a 24-layer forward at S=768).
        ttnn.multiply(h, one_plus_scale, output_tensor=h)
        ttnn.add(h, shift, output_tensor=h)
        return h

    assert norm.weight is not None, "fused adaLN needs an affine norm weight to fold into"
    # The norm's Parameter is [1, embedding_dim]; match one_plus_scale's rank explicitly rather
    # than relying on implicit rank promotion in the binary op.
    static = ttnn.reshape(norm.weight.data, [1, 1, 1, norm.embedding_dim])
    weight = ttnn.multiply(static, one_plus_scale)
    out = ttnn.experimental.dit_rms_norm_unary_fused(
        x,
        weight=weight,
        bias=shift,
        epsilon=norm.norm_eps,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(weight)
    return out


# --------------------------------------------------------------------------------------- #
#                                          RoPE                                            #
# --------------------------------------------------------------------------------------- #


def rope_tables_torch(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Host fp32 HF half-split ``cos``/``sin``, ``[1, 1, seq_len, head_dim]``.

    Bit-identical to ``diffusers.get_1d_rotary_pos_embed(head_dim, arange(S), theta=theta,
    use_real=True, repeat_interleave_real=False)``, which is what ACE-Step's
    ``_ace_step_rotary_freqs`` calls.
    """
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)  # [S, D/2]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos[None, None], sin[None, None]


def build_rope_tables(
    mesh_device: ttnn.MeshDevice,
    seq_len: int,
    *,
    head_dim: int = 128,
    theta: float = 1000000.0,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Device ``(cos, sin)`` of shape ``[1, 1, seq_len, head_dim]``, TILE layout.

    Equivalent to ``models.tt_transformers.tt.rope.get_rot_mats_hf(head_dim, device, seq_len,
    theta, rope_scaling=None)`` but without that helper's ``precompute_freqs(.., seq_len * 2)``
    over-allocation, and with no dependency on the tt_transformers model-args plumbing.

    ``dtype`` defaults to bfloat16 because ``ttnn.experimental.rotary_embedding_hf`` requires
    bf16 inputs; this also matches the deployed checkpoint, where the reference rounds the
    fp32 frequencies down to the activation dtype before the multiply.
    """
    cos, sin = rope_tables_torch(seq_len, head_dim, theta)
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return to_dev(cos), to_dev(sin)


def _dims(t) -> list[int]:
    """``list(tensor.shape)`` for a ttnn tensor.

    ``ttnn.Shape.__getitem__`` accepts only an ``int`` — slicing it raises
    ``TypeError: incompatible function arguments``. torch tensors slice fine, so this bug only
    surfaces on the device path, which is exactly where it is least convenient. Use this helper
    anywhere a ttnn shape needs slicing.
    """
    return [int(t.shape[i]) for i in range(len(t.shape))]


def apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, *, composite: bool = False) -> ttnn.Tensor:
    """HF half-split rotary embedding on a ``[1, heads, S, head_dim]`` tensor.

    ``composite=False`` (default) uses ``ttnn.experimental.rotary_embedding_hf``, a single
    device op. It requires bfloat16 inputs and ``head_dim`` either 32 or a multiple of 64
    (128 qualifies).

    ``composite=True`` builds ``x * cos + rotate_half(x) * sin`` out of slice/neg/concat. It
    works at any dtype, which is the only way to run RoPE in fp32 for a precision bisect.

    ⚠ Do **not** substitute ``models/tt_dit/blocks/attention.py::_apply_rope``: it uses
    ``ttnn.alt_complex_rotate90``, i.e. the *interleaved* pair convention
    ``(x_2i, x_2i+1) -> (-x_2i+1, x_2i)``. ACE-Step follows Qwen3 / HF ``rotate_half``, which
    splits the last dim into halves. The two differ for every head_dim > 2.
    """
    if not composite:
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=False)

    dim = int(x.shape[-1])
    half = dim // 2
    prefix = _dims(x)[:-1]
    x1 = ttnn.slice(x, [0, 0, 0, 0], [*prefix, half])
    x2 = ttnn.slice(x, [0, 0, 0, half], [*prefix, dim])
    ttnn.neg(x2, output_tensor=x2)  # rotate_half: cat([-x2, x1], -1)
    rotated = ttnn.concat([x2, x1], dim=-1)
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    out = ttnn.multiply(x, cos)
    ttnn.multiply(rotated, sin, output_tensor=rotated)
    ttnn.add(out, rotated, output_tensor=out)
    ttnn.deallocate(rotated)
    return out


# --------------------------------------------------------------------------------------- #
#                                   timestep embedding                                     #
# --------------------------------------------------------------------------------------- #


def timestep_sinusoid(
    timestep: float | torch.Tensor,
    *,
    num_channels: int = 256,
    scale: float = 1000.0,
    max_period: int = 10000,
    downscale_freq_shift: float = 0.0,
    flip_sin_to_cos: bool = True,
) -> torch.Tensor:
    """The 256-wide sinusoid of ``AceStepTimestepEmbedding``, on the **host in fp32**.

    Returns ``[B, num_channels]`` fp32.

    ⚠ This must not run in bfloat16. The reference evaluates
    ``Timesteps(...)(t * 1000.0)`` in fp32 and only rounds the *result* to the activation
    dtype. The largest frequency factor is 1.0, so the sinusoid argument reaches ~1000 rad;
    bfloat16 quantises that to ~±2 rad, which randomises ``cos``/``sin`` completely. At 8
    denoising steps there is nothing to gain from putting it on device anyway — §3.7 of the
    master doc lists it as a host precompute.

    Matches ``diffusers.get_timestep_embedding(t * scale, num_channels,
    flip_sin_to_cos=True, downscale_freq_shift=0)`` exactly.
    """
    t = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1)
    half = num_channels // 2
    exponent = -math.log(max_period) * torch.arange(0, half, dtype=torch.float32)
    exponent = exponent / (half - downscale_freq_shift)
    emb = (t * scale)[:, None] * torch.exp(exponent)[None, :]
    parts = [emb.cos(), emb.sin()] if flip_sin_to_cos else [emb.sin(), emb.cos()]
    return torch.cat(parts, dim=-1)


def timestep_sinusoid_tt(
    timestep: float | torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    num_channels: int = 256,
    scale: float = 1000.0,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """``timestep_sinusoid`` uploaded as ``[1, 1, B, num_channels]``."""
    freq = timestep_sinusoid(timestep, num_channels=num_channels, scale=scale)
    return ttnn.from_torch(
        freq.reshape(1, 1, *freq.shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def fold_time_embed_r(
    state: dict[str, torch.Tensor], *, num_channels: int = 256, scale: float = 1000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``time_embed_r`` at ``t - t_r == 0`` on the host and return its constant output.

    Inference always passes ``timestep_r == timestep``, so ``time_embed_r`` sees exactly 0 and
    its output never changes. Folding it removes a 256->2048->2048->12288 MLP from every
    denoising step (§3.7).

    Args:
        state: the ``time_embed_r`` substate, i.e. keys ``linear_1.{weight,bias}``,
            ``linear_2.{weight,bias}``, ``time_proj.{weight,bias}``.

    Returns:
        ``(temb_r, timestep_proj_r)`` as fp32 ``[1, 2048]`` and ``[6, 2048]``. ``temb_r`` folds
        into the ``norm_out`` 2-way table; ``timestep_proj_r`` folds into each block's 6-way
        ``scale_shift_table``.
    """
    required = (
        "linear_1.weight",
        "linear_1.bias",
        "linear_2.weight",
        "linear_2.bias",
        "time_proj.weight",
        "time_proj.bias",
    )
    missing = [k for k in required if k not in state]
    if missing:
        msg = f"time_embed_r state is incomplete, missing {missing}"
        raise KeyError(msg)

    f32 = lambda k: state[k].detach().to(torch.float32)  # noqa: E731
    t_freq = timestep_sinusoid(0.0, num_channels=num_channels, scale=scale)  # [1, 256]
    h = torch.nn.functional.linear(t_freq, f32("linear_1.weight"), f32("linear_1.bias"))
    h = torch.nn.functional.silu(h)
    temb_r = torch.nn.functional.linear(h, f32("linear_2.weight"), f32("linear_2.bias"))
    proj = torch.nn.functional.linear(torch.nn.functional.silu(temb_r), f32("time_proj.weight"), f32("time_proj.bias"))
    hidden = temb_r.shape[-1]
    return temb_r.reshape(1, hidden), proj.reshape(6, hidden)


# --------------------------------------------------------------------------------------- #
#                          proj_in / proj_out: convs that are matmuls                      #
# --------------------------------------------------------------------------------------- #


def conv1d_patch_to_linear(weight: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``nn.Conv1d(C_in, C_out, k=patch_size, stride=patch_size, padding=0)`` -> ``nn.Linear``.

    With ``kernel == stride`` the windows are non-overlapping, so the conv is exactly a
    reshape of the NSC input from ``[B, T, C_in]`` to ``[B, T/p, p * C_in]`` followed by
    ``Linear(p * C_in, C_out)``.

    ``weight`` is the conv weight ``[C_out, C_in, p]``; the returned tensor is the
    ``nn.Linear`` weight ``[C_out, p * C_in]`` matching the row-major ``(k, c)`` flattening of
    the reshape. The conv bias needs no change.
    """
    c_out, c_in, k = weight.shape
    assert k == patch_size, f"conv kernel {k} != patch_size {patch_size}"
    # y[o, s] = sum_k sum_c x[c, p*s + k] * W[o, c, k]  ->  column index k * C_in + c
    return weight.permute(0, 2, 1).reshape(c_out, k * c_in).contiguous()


def conv_transpose1d_patch_to_linear(
    weight: torch.Tensor, bias: torch.Tensor | None, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``nn.ConvTranspose1d(C_in, C_out, k=p, stride=p, padding=0)`` -> ``nn.Linear``.

    Also non-overlapping, so it is ``Linear(C_in, p * C_out)`` followed by a reshape of the
    NSC output from ``[B, S, p * C_out]`` to ``[B, p * S, C_out]``.

    ⚠ ``nn.ConvTranspose1d`` stores its weight as ``[C_in, C_out, k]`` — the in/out axes are
    swapped relative to ``nn.Conv1d``. The bias is ``[C_out]`` and must be **tiled ``p``
    times** to ``[p * C_out]``, because the output column index is ``k * C_out + c`` and the
    bias depends only on ``c``.
    """
    c_in, c_out, k = weight.shape
    assert k == patch_size, f"conv kernel {k} != patch_size {patch_size}"
    # y[c, p*s + k] = sum_i x[i, s] * W[i, c, k]  ->  Linear out index k * C_out + c
    lin_w = weight.permute(2, 1, 0).reshape(k * c_out, c_in).contiguous()
    lin_b = None if bias is None else bias.repeat(k).contiguous()
    return lin_w, lin_b


# --------------------------------------------------------------------------------------- #
#                                   small tensor helpers                                   #
# --------------------------------------------------------------------------------------- #


def to_device(
    x: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    """``torch`` -> replicated ``ttnn`` tensor on a (1, 1) or larger mesh."""
    return ttnn.from_torch(
        x,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def to_host(x: ttnn.Tensor) -> torch.Tensor:
    """``ttnn`` -> host ``torch`` fp32, with any mesh replication collapsed."""
    from models.tt_dit.utils.tensor import to_torch as _to_torch

    return _to_torch(x).to(torch.float32)


class Capture(dict):
    """Host-snapshot collector for PCC bisection, with an optional key filter.

    A plain ``dict`` also works as a ``capture`` argument and records everything; use this
    class when that would be too much data. At S=768 the DiT emits ~15 intermediates per
    layer at 6 MB fp32 each, so an unfiltered 24-layer capture is ~2 GB of host memory.

    Args:
        keys: exact keys to keep.
        suffixes: keep any key ending in one of these (e.g. ``(".out",)``).

    Passing neither keeps everything.
    """

    def __init__(self, keys=None, *, suffixes=None) -> None:
        super().__init__()
        self._keys = set(keys) if keys is not None else None
        self._suffixes = tuple(suffixes) if suffixes is not None else None

    def wants(self, key: str) -> bool:
        if self._keys is None and self._suffixes is None:
            return True
        if self._keys is not None and key in self._keys:
            return True
        return bool(self._suffixes is not None and key.endswith(self._suffixes))


def capture_tensor(capture: dict | None, key: str, value: ttnn.Tensor) -> None:
    """Snapshot a device tensor to host fp32 *now*.

    Deliberately eager: on-device intermediates stashed for later inspection read back as
    garbage once their DRAM has been reused (XTTS BUG-4 / master doc §5). Only active when
    ``capture`` is not ``None``, so the fast path costs one ``is None`` test.
    """
    if capture is None:
        return
    wants = getattr(capture, "wants", None)
    if wants is not None and not wants(key):
        return
    capture[key] = to_host(value)


def reshape_last_two(x: ttnn.Tensor, new_rows: int, new_cols: int) -> ttnn.Tensor:
    """Row-major-contiguous reshape of the trailing ``[rows, cols]`` pair.

    Used for the ``proj_in`` / ``proj_out`` patch (un)folding, where ``[T, C] <-> [T/p, p*C]``
    moves tile boundaries and therefore is not a TILE-layout view. Round-trips through
    ROW_MAJOR, where the reshape is a genuine view.
    """
    was_tile = x.layout == ttnn.TILE_LAYOUT
    if was_tile:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, _dims(x)[:-2] + [new_rows, new_cols])
    if was_tile:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    return x
