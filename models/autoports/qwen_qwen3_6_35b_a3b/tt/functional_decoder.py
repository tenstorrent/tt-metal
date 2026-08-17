# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Functional TTNN decoder layer for ``Qwen/Qwen3.6-35B-A3B`` (HF ``Qwen3_5MoeDecoderLayer``).

One class, :class:`FunctionalDecoder`, covers **both** decoder layer kinds the model has
(`config.text_config.layer_types`):

===================  =========================================  ==========================
layer kind           token mixer                                per-sequence state
===================  =========================================  ==========================
``full_attention``   GQA 16/2, head_dim 256, output-gated,       paged K/V cache
                     partial RoPE (rotary_dim 64), q/k RMSNorm
``linear_attention``  gated DeltaNet: depthwise causal conv1d    conv state (3 taps) +
                     (k=4) + gated delta rule, 32 v-heads        recurrent state
                                                                 [32, 128, 128]
===================  =========================================  ==========================

Everything after the token mixer is shared: RMSNorm (``1 + w``) around a 256-expert /
top-8 MoE MLP with a sigmoid-gated shared expert.

Public contract
---------------

``FunctionalDecoder.from_state_dict(state_dict, hf_config=..., layer_idx=..., mesh_device=...)``
builds the layer. All host-side (torch) weight preparation, cache allocation and RoPE table
construction happens there; the two forward methods below run pure TTNN on device.

``prefill_forward(x, *, user_id=0, page_table=None, start_pos=0) -> ttnn.Tensor``
    ``x``          : ``[1, 1, seq_len, hidden]``, TILE, DRAM. **Any** ``seq_len >= 1`` with
                     ``start_pos + seq_len <= supported_context``. Non-tile / non-chunk
                     aligned lengths are padded and masked internally and the output is
                     sliced back to ``seq_len``.
    ``user_id``    : sequence slot. Selects the ``page_table`` row for the paged K/V cache
                     and the conv/recurrent state slot for linear-attention layers.
    ``page_table`` : ``int32 [max_batch, ceil(supported_context / block_size)]``, ROW_MAJOR.
                     Required for ``full_attention`` layers, ignored for ``linear_attention``.
    ``start_pos``  : absolute position of ``x[..., 0, :]``. Must be a multiple of
                     ``PREFILL_ALIGN`` (128) == ``sdpa_chunk``, because chunked SDPA turns the
                     offset into a chunk index by integer division and a misaligned offset is
                     silently wrong rather than an error (see ``prefill_forward``). Lets a
                     caller stream a long prompt through several calls; the KV cache, conv
                     state and recurrent state all carry over. A fresh request needs
                     ``start_pos = 0``, so this never restricts prompt length.
    returns        : ``[1, 1, seq_len, hidden]``, TILE, DRAM.

    One sequence per call (``user_id``), matching the per-request prefill that vLLM and the
    downstream full-model stage issue. Batch >1 prefill = call once per ``user_id``; each
    lands in its own cache slot / page-table row.

``decode_forward(x, *, current_pos=None, page_table=None) -> ttnn.Tensor``
    ``x``           : ``[1, 1, max_batch, hidden]``, TILE, DRAM — one token per slot.
    ``current_pos`` : ``int32 [max_batch]`` device tensor, ROW_MAJOR. Absolute position of
                      the token being decoded per slot. Used as the paged-cache write index,
                      the SDPA ``cur_pos`` and (via an on-device typecast) the RoPE table
                      lookup index. A value of ``-1`` marks a slot inactive: its attention is
                      skipped and its paged K/V is left untouched (verified by
                      ``test_decode_skips_inactive_slots_with_negative_position``). Required
                      for ``full_attention``; ``linear_attention`` layers ignore it, since
                      their recurrence carries no position, so it may be omitted there.
    ``page_table``  : same tensor as prefill. Required for ``full_attention``.
    returns         : ``[1, 1, max_batch, hidden]``, TILE, DRAM.

    ``current_pos`` is a *device tensor*, never a Python list, so one captured trace can be
    replayed at any position by writing new values into it. The decode batch is fixed at
    ``max_batch`` because the linear-attention conv/recurrent state buffers are updated
    whole-tensor and in place (a requirement for trace-safe replay).

Both methods are free of ``torch``, ``ttnn.from_torch``, ``ttnn.to_torch`` and any host
round-trip; see ``tests/test_functional_decoder.py::test_no_runtime_host_fallback``.

Implementation notes for the non-obvious parts live in
``doc/functional_decoder/work_log.md`` §3 and are unit-tested on CPU in
``tests/test_reference_math.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32

#: Public prefill inputs are padded up to this multiple internally. It is the lcm of the
#: SDPA q/k chunk (128), the delta-rule chunk (64), the paged block size (64) and the MoE
#: tile group (32), so every internal offset derived from it is legal for every op.
PREFILL_ALIGN = 128

_SPARSE_CORE_GRID = (8, 8)  # sparse_matmul rejects the full 11x10 Blackhole grid


# =======================================================================================
# config
# =======================================================================================
@dataclass
class DecoderConfig:
    """Everything the layer needs, derived from the HF text config plus runtime knobs."""

    # --- from hf_config ---
    layer_kind: str
    layer_idx: int
    hidden_size: int
    rms_norm_eps: float
    # full attention
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    rotary_dim: int
    rope_theta: float
    # linear attention
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    conv_kernel: int
    # moe
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    # capability
    hf_max_position_embeddings: int

    # --- runtime knobs ---
    mesh_device: Any = None
    max_batch_size: int = 1
    supported_context: int = 262144
    block_size: int = 64
    prefill_chunk_size: int = 2048
    delta_chunk_size: int = 64
    moe_prefill_chunk_tokens: int = 512
    sdpa_chunk: int = 128
    #: Keys per k-chunk for the decode SDPA. **This is the variable behind long-context decode
    #: accuracy** (README section 3.8): each chunk is one sequential bf16 accumulation step, so a
    #: bigger chunk means a shallower accumulation over the same keys. At 262144 keys and 1 core
    #: per head the op PCC is 0.7664 at 32 (the op's own default), 0.9704 at 128, 0.9809 at 256 and
    #: 0.9825 at 512. 512 is the **largest legal value**: 1024 needs 2371456 B of statically
    #: allocated circular buffers against a 1572864 B L1 limit.
    decode_sdpa_k_chunk_size: int = 512
    #: Cores the decode SDPA splits each KV head's keys across. 1 is both the op's own default for
    #: the paged variant (``sdpa_decode.cpp:122-129``) and the only value measured correct at every
    #: ``cur_pos``: every value above 1 returns a silently wrong answer below some context that
    #: grows with it (README section 3.8). It is pinned rather than left implicit so that a later
    #: stage changing it has to read why. ``None`` means "pass no program config at all", which is
    #: *not* neutral -- it selects ``k_chunk_size=32``, the worst measured setting.
    decode_sdpa_max_cores_per_head: int | None = 1
    #: Escape hatch: a fully-specified ``ttnn.SDPAProgramConfig`` for the decode SDPA, used
    #: verbatim and overriding ``decode_sdpa_max_cores_per_head``. For sweeps in later stages.
    decode_sdpa_program_config: Any = None
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    expert_weight_dtype: ttnn.DataType = ttnn.bfloat16
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat16
    #: gated-delta-rule internals (cumulative gates, decay masks, UT transform, recurrent
    #: state). fp32 by default: the chunk scan is a length-``seq/64`` recurrence and the HF
    #: config itself pins the SSM state to float32 (``mamba_ssm_dtype``).
    delta_dtype: ttnn.DataType = ttnn.float32

    # derived
    num_v_head_groups: int = field(init=False)
    #: width of the q|k|v block that the delta rule consumes (q/k duplicated up to v heads)
    delta_qkv_width: int = field(init=False)
    #: total depthwise-conv width: delta_qkv_width plus the z block, which rides along with
    #: an identity conv tap so one conv+silu emits post-conv q/k/v *and* silu(z)
    conv_dim: int = field(init=False)
    max_blocks_per_seq: int = field(init=False)

    def __post_init__(self):
        if self.layer_kind not in ("full_attention", "linear_attention"):
            raise ValueError(f"unsupported layer kind {self.layer_kind!r}")
        if self.supported_context % self.block_size:
            raise ValueError("supported_context must be a multiple of block_size")
        if self.supported_context % PREFILL_ALIGN:
            # a prefill of the whole context pads up to a PREFILL_ALIGN multiple, and the
            # page table / rope table must still cover the padded length
            raise ValueError(f"supported_context must be a multiple of {PREFILL_ALIGN}")
        if self.prefill_chunk_size % PREFILL_ALIGN:
            raise ValueError(f"prefill_chunk_size must be a multiple of {PREFILL_ALIGN}")
        if self.delta_chunk_size & (self.delta_chunk_size - 1):
            raise ValueError("delta_chunk_size must be a power of two")
        if self.moe_prefill_chunk_tokens % TILE:
            raise ValueError("moe_prefill_chunk_tokens must be a multiple of 32")
        # Q/K of the delta rule are repeat_interleaved from key heads up to value heads. The
        # duplication is folded into the projection and conv weights at load time so the
        # runtime path never needs a repeat_interleave (see from_state_dict).
        self.num_v_head_groups = self.linear_num_value_heads // self.linear_num_key_heads
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        hv = self.linear_num_value_heads
        self.delta_qkv_width = 2 * hv * self.linear_key_head_dim + hv * self.linear_value_head_dim
        self.conv_dim = self.delta_qkv_width + hv * self.linear_value_head_dim
        self.max_blocks_per_seq = self.supported_context // self.block_size

    @classmethod
    def from_hf(cls, hf_config, layer_idx: int, mesh_device, **kwargs) -> "DecoderConfig":
        partial = hf_config.rope_parameters.get("partial_rotary_factor", 1.0)
        return cls(
            layer_kind=hf_config.layer_types[layer_idx],
            layer_idx=layer_idx,
            hidden_size=hf_config.hidden_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            head_dim=hf_config.head_dim,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            rotary_dim=int(hf_config.head_dim * partial),
            rope_theta=hf_config.rope_parameters["rope_theta"],
            linear_num_key_heads=hf_config.linear_num_key_heads,
            linear_num_value_heads=hf_config.linear_num_value_heads,
            linear_key_head_dim=hf_config.linear_key_head_dim,
            linear_value_head_dim=hf_config.linear_value_head_dim,
            conv_kernel=hf_config.linear_conv_kernel_dim,
            num_experts=hf_config.num_experts,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            shared_expert_intermediate_size=hf_config.shared_expert_intermediate_size,
            hf_max_position_embeddings=hf_config.max_position_embeddings,
            mesh_device=mesh_device,
            **kwargs,
        )

    @property
    def is_linear(self) -> bool:
        return self.layer_kind == "linear_attention"

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5


# =======================================================================================
# helpers
# =======================================================================================
def _hifi_config(mesh_device) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _sparse_program_config(m: int, n: int, k: int, in0_block_w: int = 8, cores: tuple[int, int] = _SPARSE_CORE_GRID):
    """1D-multicast matmul config for ``ttnn.sparse_matmul`` (mcast_in0 is mandatory).

    ``in0_block_w`` is snapped down to a divisor of ``Kt`` because the sparse kernel asserts
    ``Kt % in0_block_w == 0``.
    """
    core_x, core_y = cores
    num_cores = core_x * core_y
    n_tiles = math.ceil(n / TILE)
    k_tiles = math.ceil(k / TILE)
    if k_tiles % in0_block_w:
        divisors = [d for d in range(2, in0_block_w + 1) if k_tiles % d == 0]
        in0_block_w = max(divisors) if divisors else k_tiles
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=max(TILE, m) // TILE,
        per_core_N=(n_tiles + num_cores - 1) // num_cores,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _sdpa_program_config(chunk: int) -> ttnn.SDPAProgramConfig:
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )


def _decode_sdpa_program_config(cfg: DecoderConfig):
    """The decode SDPA program config, built once at setup.

    ``cfg.decode_sdpa_program_config`` wins if given. Otherwise the two knobs that matter are
    ``cfg.decode_sdpa_k_chunk_size`` (accumulation depth -- the accuracy/latency lever) and
    ``cfg.decode_sdpa_max_cores_per_head`` (parallel decomposition, which must stay 1). See README
    section 3.8 for the 2-D sweep both come from.

    **Passing no config is not neutral.** The paged entry point substitutes its own config before
    the device op sees one (``sdpa_decode.cpp:122-129``): device grid, ``q_chunk_size=32``,
    ``k_chunk_size=32``, ``exp_approx_mode`` unset, ``max_cores_per_head_batch=1``. So the factory's
    ``program_config.has_value() ? max_cores_per_head_batch : num_cores_available`` branch
    (``sdpa_decode_program_factory.cpp:192-193``) is unreachable from here, the struct default of 16
    (``sdpa_config.hpp:18``) is unreachable, and "no config" means ``k_chunk_size=32`` -- the worst
    setting measured. ``diag_sdpa_decode.py``'s identity control proves this: no config and an
    explicit config spelling out that substitution are bit-identical at all 11 contexts.

    ``exp_approx_mode=False`` matches what the substituted config resolves to (``nullopt`` ->
    ``false``, ``sdpa_decode_program_factory.cpp:211-212``) and is spelled out only for provenance;
    the sweep shows approx and exact are bit-identical. ``q_chunk_size`` is unused by the decode
    factory but the struct requires a value.
    """
    if cfg.decode_sdpa_program_config is not None:
        return cfg.decode_sdpa_program_config
    if cfg.decode_sdpa_max_cores_per_head is None:
        return None
    grid = cfg.mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=cfg.decode_sdpa_k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=cfg.decode_sdpa_max_cores_per_head,
    )


def _tilized(tensor):
    """``(tile_layout_tensor, owned)``. ``owned`` is False when the input already was TILE.

    ``ttnn.to_layout`` returns the *input* if no conversion is needed, so the caller cannot
    unconditionally deallocate the result -- the same aliasing trap as ``_subview`` / ``_move``.
    """
    if tensor.layout == ttnn.TILE_LAYOUT:
        return tensor, False
    return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT), True


def _zero_(tensor) -> None:
    """Zero ``tensor`` in place, keeping its buffer address (so it stays trace-safe).

    ``ttnn.fill`` with ``output_tensor=tensor`` is the right primitive for two independent reasons,
    both measured by ``probe_ttnn_ops.py``:

    * **It writes rather than scales.** Zeroing is how a slot is handed to a new sequence (see
      ``reset_state`` and ``prefill_forward``'s ``start_pos == 0`` path) and the README contract says
      a reused slot starts clean *unconditionally*, so the operation must not depend on what the
      buffer currently holds. ``fill`` clears NaN and Inf. (An earlier version used
      ``ttnn.zeros_like`` + ``ttnn.copy`` and justified it by "``0 * NaN`` is ``NaN``, so a multiply
      cannot clear a poisoned buffer". That is true in IEEE arithmetic but **not** what this build's
      ``ttnn.mul(t, 0.0)`` does -- the probe shows it zeroes a NaN too. The stated reason was wrong;
      the preference for a write still stands, on the weaker ground that it cannot depend on kernel
      details.)
    * **It allocates nothing.** ``zeros_like`` materialises a full-size peer, which at the shape the
      capability contract certifies -- batch 32 at 262144 context, 8 GiB per K/V cache -- is a
      transient 8 GiB on top of the 16 GiB already held. ``fill`` writes into the existing buffer.
    """
    ttnn.fill(tensor, 0.0, output_tensor=tensor)


def _dealloc(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            ttnn.deallocate(tensor)


def _move(tensor, memory_config):
    """``to_memory_config`` that is a real no-op when the layout already matches.

    ``ttnn.to_memory_config`` returns the *input* tensor when nothing has to move, so an
    unconditional convert-then-deallocate frees the tensor it just returned.
    """
    if tensor.memory_config() == memory_config:
        return tensor
    out = ttnn.to_memory_config(tensor, memory_config)
    _dealloc(tensor)
    return out


def _subview(tensor, starts, ends):
    """``(sliced, owned)`` — read-only sub-view.

    ``ttnn.slice`` hands back a *view* of the input (same buffer address) when the slice
    covers the whole tensor, so an unconditional deallocate of the result would free the
    caller's tensor — e.g. the page table, or a persistent state buffer.
    """
    sub = ttnn.slice(tensor, list(starts), list(ends))
    return sub, sub.buffer_address() != tensor.buffer_address()


def _view(tensor, shape):
    """Reinterpret ``tensor``'s shape. The result **aliases** the input buffer.

    ``ttnn.reshape`` returns a view, so exactly one of (input, view) may ever be deallocated —
    freeing both double-frees, and freeing the view while the input is still needed is a
    use-after-free. Callers keep the input as the owner.
    """
    return ttnn.reshape(tensor, shape)


def _owned_slice(tensor, starts, ends):
    """A sub-tensor the caller may mutate and deallocate, even for a whole-tensor slice."""
    sub, owned = _subview(tensor, starts, ends)
    if owned:
        return sub
    return ttnn.clone(sub, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=sub.dtype)


# =======================================================================================
# host-side weight preparation (setup only)
# =======================================================================================
def _prepare_weights(state_dict, cfg: DecoderConfig) -> dict:
    """torch -> plain torch tensors laid out the way the TTNN forward wants them.

    Kept in one function so the runtime path has no reshaping/transposing left to do and no
    torch import at all. Returns a name -> (tensor, dtype) mapping consumed by
    :func:`_to_device`.
    """
    import torch

    def t(name):
        return state_dict[name].to(torch.float32)

    act, wgt = cfg.activation_dtype, cfg.weight_dtype
    out: dict[str, tuple[Any, ttnn.DataType]] = {}

    # ---- norms: HF applies (1 + w) ----
    out["input_norm_w"] = ((1.0 + t("input_layernorm.weight")).reshape(1, 1, 1, -1), act)
    out["post_norm_w"] = ((1.0 + t("post_attention_layernorm.weight")).reshape(1, 1, 1, -1), act)

    # ---- MoE ----
    out["router_w"] = (t("mlp.gate.weight").T.reshape(1, 1, cfg.hidden_size, cfg.num_experts), act)
    # HF: gate_up_proj[e] is [2*I, H] and linear(x, W) -> chunk(2, -1) => rows 0..I-1 gate,
    # rows I..2I-1 up. Transposing to [H, 2*I] keeps that column split.
    gate_up = t("mlp.experts.gate_up_proj").transpose(1, 2).contiguous()  # [E, H, 2I]
    out["expert_gate_up"] = (gate_up.unsqueeze(0), cfg.expert_weight_dtype)
    down = t("mlp.experts.down_proj").transpose(1, 2).contiguous()  # [E, I, H]
    out["expert_down"] = (down.unsqueeze(0), cfg.expert_weight_dtype)
    out["shared_gate_proj"] = (t("mlp.shared_expert.gate_proj.weight").T.unsqueeze(0).unsqueeze(0), wgt)
    out["shared_up_proj"] = (t("mlp.shared_expert.up_proj.weight").T.unsqueeze(0).unsqueeze(0), wgt)
    out["shared_down_proj"] = (t("mlp.shared_expert.down_proj.weight").T.unsqueeze(0).unsqueeze(0), wgt)
    out["shared_expert_gate"] = (t("mlp.shared_expert_gate.weight").T.unsqueeze(0).unsqueeze(0), wgt)

    if cfg.layer_kind == "full_attention":
        nh, hd = cfg.num_attention_heads, cfg.head_dim
        q_full = t("self_attn.q_proj.weight")  # [nh * 2 * hd, H], head-interleaved [q|gate]
        q_full = q_full.reshape(nh, 2, hd, cfg.hidden_size)
        q_part = q_full[:, 0].reshape(nh * hd, cfg.hidden_size)
        gate_part = q_full[:, 1].reshape(nh * hd, cfg.hidden_size)
        fused = torch.cat([q_part, t("self_attn.k_proj.weight"), t("self_attn.v_proj.weight"), gate_part], dim=0)
        out["qkvg"] = (fused.T.unsqueeze(0).unsqueeze(0), wgt)
        out["q_norm_w"] = ((1.0 + t("self_attn.q_norm.weight")).reshape(1, 1, 1, -1), act)
        out["k_norm_w"] = ((1.0 + t("self_attn.k_norm.weight")).reshape(1, 1, 1, -1), act)
        out["o_proj"] = (t("self_attn.o_proj.weight").T.unsqueeze(0).unsqueeze(0), wgt)
    else:
        hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        rep = cfg.num_v_head_groups
        qkv = t("linear_attn.in_proj_qkv.weight")  # [2*hk*dk + hv*dv, H]
        q_w = qkv[: hk * dk].reshape(hk, dk, -1)
        k_w = qkv[hk * dk : 2 * hk * dk].reshape(hk, dk, -1)
        v_w = qkv[2 * hk * dk :]
        # Fold HF's repeat_interleave(q/k, hv // hk, dim=heads) into the weights. The conv is
        # depthwise so duplicating channels commutes with it as long as the per-channel conv
        # taps are duplicated identically (done below).
        q_w = q_w.repeat_interleave(rep, dim=0).reshape(hv * dk, -1)
        k_w = k_w.repeat_interleave(rep, dim=0).reshape(hv * dk, -1)
        z_w = t("linear_attn.in_proj_z.weight")  # [hv*dv, H]
        b_w = t("linear_attn.in_proj_b.weight")  # [hv, H]
        a_w = t("linear_attn.in_proj_a.weight")  # [hv, H]
        fused_in = torch.cat([q_w, k_w, v_w, z_w, b_w, a_w], dim=0)
        out["in_proj"] = (fused_in.T.unsqueeze(0).unsqueeze(0), wgt)

        # conv taps, duplicated for q/k and an identity tap for the z block so that one
        # depthwise conv + silu produces post-conv q/k/v *and* silu(z).
        conv = t("linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel)  # [2*hk*dk + hv*dv, K]
        cq = conv[: hk * dk].reshape(hk, dk, cfg.conv_kernel).repeat_interleave(rep, dim=0)
        ck = conv[hk * dk : 2 * hk * dk].reshape(hk, dk, cfg.conv_kernel).repeat_interleave(rep, dim=0)
        cv = conv[2 * hk * dk :].reshape(hv, dv, cfg.conv_kernel)
        identity = torch.zeros(hv, dv, cfg.conv_kernel)
        identity[:, :, -1] = 1.0
        conv_full = torch.cat(
            [
                cq.reshape(-1, cfg.conv_kernel),
                ck.reshape(-1, cfg.conv_kernel),
                cv.reshape(-1, cfg.conv_kernel),
                identity.reshape(-1, cfg.conv_kernel),
            ],
            dim=0,
        )  # [conv_dim, K]
        for j in range(cfg.conv_kernel):
            out[f"conv_tap_{j}"] = (conv_full[:, j].reshape(1, 1, 1, -1), act)

        out["delta_norm_w"] = (t("linear_attn.norm.weight").reshape(1, 1, 1, -1), act)
        out["out_proj"] = (t("linear_attn.out_proj.weight").T.unsqueeze(0).unsqueeze(0), wgt)
        # g = -exp(A_log) * softplus(a + dt_bias); the constant factor folds into a weight.
        out["neg_exp_a_log"] = ((-t("linear_attn.A_log").exp()).reshape(1, 1, 1, -1), ttnn.float32)
        out["dt_bias"] = (t("linear_attn.dt_bias").reshape(1, 1, 1, -1), ttnn.float32)
    return out


def _build_rope_tables(cfg: DecoderConfig):
    """cos/sin lookup tables ``[1, 1, supported_context, rotary_dim]`` (setup only).

    Text-only Qwen3.6 mRoPE collapses onto standard 1-D RoPE (all three position rows are
    the text positions), proved in ``tests/test_reference_math.py``.
    """
    import torch

    dim = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    pos = torch.arange(cfg.supported_context, dtype=torch.float64)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float().reshape(1, 1, -1, dim), emb.sin().float().reshape(1, 1, -1, dim)


def _to_device(tensor, dtype, mesh_device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )


# =======================================================================================
# the layer
# =======================================================================================
class FunctionalDecoder(LightweightModule):
    """A single Qwen3.6-35B-A3B decoder layer on a TTNN mesh device."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.compute_config = _hifi_config(cfg.mesh_device)
        self.decode_sdpa_config = _decode_sdpa_program_config(cfg)
        self.w: dict[str, ttnn.Tensor] = {}

    # -----------------------------------------------------------------------------------
    # construction
    # -----------------------------------------------------------------------------------
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        """Build the layer from an HF layer-relative state dict.

        ``state_dict`` keys are the HF names with the ``model.language_model.layers.<i>.``
        prefix stripped (e.g. ``self_attn.q_proj.weight``, ``mlp.experts.gate_up_proj``).
        All torch work — transposes, the q/gate de-interleave, the q/k head duplication,
        conv-tap split, ``1 + w`` norm folding, RoPE tables, cache allocation — happens
        here, so ``prefill_forward`` / ``decode_forward`` are pure TTNN.
        """
        import torch

        cfg = DecoderConfig.from_hf(hf_config, layer_idx, mesh_device, **kwargs)
        self = cls(cfg)

        prepared = _prepare_weights(state_dict, cfg)
        for name, (tensor, dtype) in prepared.items():
            if tensor is None:
                continue
            self.w[name] = _to_device(tensor, dtype, mesh_device)

        # RoPE tables (full attention only), stored **ROW_MAJOR**. That is decode's requirement,
        # not a detail: `ttnn.embedding` converts a TILE weight to ROW_MAJOR *on every call*
        # (`ttnn/cpp/ttnn/operations/embedding/embedding.cpp:30-32`), so a TILE table would
        # untilize the whole `supported_context` x `rotary_dim` table per decode step -- 2 x 32 MiB
        # at the advertised context, to read `max_batch_size` rows. Prefill wants TILE instead, but
        # only for the chunk it slices, so it tilizes that slice (<= `prefill_chunk_size` rows)
        # rather than paying for a second full table: keeping both layouts costs 64 MiB per layer
        # at the advertised context, which is real DRAM pressure once several layers are live.
        if not cfg.is_linear:
            cos, sin = _build_rope_tables(cfg)
            self.w["rope_cos"] = _to_device(cos, cfg.activation_dtype, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.w["rope_sin"] = _to_device(sin, cfg.activation_dtype, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # delta-rule constants
        if cfg.is_linear:
            c = cfg.delta_chunk_size
            eye = torch.eye(c).reshape(1, 1, c, c)
            self.w["delta_eye"] = _to_device(eye, cfg.delta_dtype, mesh_device)
            # additive mask: 0 on/below the diagonal, -1e30 above. Applied *before* exp so
            # the upper triangle can never overflow (cumulative gates reach ~-1e5).
            causal = torch.zeros(c, c)
            causal.masked_fill_(torch.ones(c, c, dtype=torch.bool).triu(1), -1e30)
            self.w["delta_causal_add"] = _to_device(causal.reshape(1, 1, c, c), cfg.delta_dtype, mesh_device)
            self.w["delta_strict_mul"] = _to_device(
                torch.ones(c, c).tril(-1).reshape(1, 1, c, c), cfg.delta_dtype, mesh_device
            )
            # upper-triangular ones: row-vector @ triu = within-chunk cumulative sum.
            self.w["delta_cumsum"] = _to_device(
                torch.ones(c, c).triu(0).reshape(1, 1, c, c), cfg.delta_dtype, mesh_device
            )
            # ones column used to build the padding mask for a non-aligned prefill tail
            # (sliced + zero-padded on device; see _linear_attention_prefill).
            self.w["ones_column"] = _to_device(
                torch.ones(1, 1, cfg.prefill_chunk_size, 1), cfg.delta_dtype, mesh_device
            )
            # one-hot / complement rows used to write a single sequence slot of the conv
            # state without a host round-trip.
            b = cfg.max_batch_size
            onehot = torch.eye(b).reshape(b, 1, b, 1)
            self.w["slot_onehot"] = _to_device(onehot, cfg.activation_dtype, mesh_device)
            self.w["slot_keep"] = _to_device(1.0 - onehot, cfg.activation_dtype, mesh_device)

        self._init_state()
        return self

    def _init_state(self) -> None:
        """Allocate the persistent per-sequence state (paged K/V, or conv + recurrent)."""
        import torch

        cfg = self.cfg
        dev = cfg.mesh_device
        if cfg.is_linear:
            hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
            self.recurrent_state = _to_device(torch.zeros(cfg.max_batch_size, hv, dk, dv), cfg.delta_dtype, dev)
            # conv_kernel-1 single-row taps, oldest first. Separate buffers (rather than one
            # [.., K-1, .., ..] tensor) so decode can shift them with whole-tensor in-place
            # copies, which is what makes the traced decode replay-safe.
            self.conv_state = [
                _to_device(torch.zeros(1, 1, cfg.max_batch_size, cfg.conv_dim), cfg.activation_dtype, dev)
                for _ in range(cfg.conv_kernel - 1)
            ]
            self.kv_cache = None
        else:
            num_blocks = cfg.max_batch_size * cfg.max_blocks_per_seq
            shape = (num_blocks, cfg.num_key_value_heads, cfg.block_size, cfg.head_dim)
            self.kv_cache = [
                _to_device(torch.zeros(*shape), cfg.kv_cache_dtype, dev),
                _to_device(torch.zeros(*shape), cfg.kv_cache_dtype, dev),
            ]
            self.recurrent_state = None
            self.conv_state = None

    def release(self) -> None:
        """Free every device tensor this layer owns (weights, caches, state).

        Not needed for correctness — the tensors are freed when the layer is dropped — but a
        layer holds ~1.5 GiB of MoE weights plus the paged cache, so an explicit release lets a
        caller reclaim DRAM before opening another layer or closing the device.
        """
        for tensor in list(self.w.values()):
            _dealloc(tensor)
        self.w.clear()
        for tensor in (self.recurrent_state, *(self.conv_state or ()), *(self.kv_cache or ())):
            _dealloc(tensor)
        self.recurrent_state = None
        self.conv_state = None
        self.kv_cache = None

    def reset_state(self) -> None:
        """Zero the persistent state in place (keeps buffer addresses, so trace-safe).

        All slots. There is deliberately no per-slot variant: a slot is handed to a new sequence by
        prefilling it from ``start_pos = 0``, which zeroes that slot's carry on its own.
        """
        if self.cfg.is_linear:
            _zero_(self.recurrent_state)
            for tap in self.conv_state:
                _zero_(tap)
        else:
            for cache in self.kv_cache:
                _zero_(cache)

    # -----------------------------------------------------------------------------------
    # forward dispatch
    # -----------------------------------------------------------------------------------
    def forward(self, x, *, mode: str, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(x, **kwargs)
        if mode == "decode":
            return self.decode_forward(x, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    # -----------------------------------------------------------------------------------
    # prefill
    # -----------------------------------------------------------------------------------
    def prefill_forward(self, x, *, user_id: int = 0, page_table=None, start_pos: int = 0):
        cfg = self.cfg
        seq_len = int(x.shape[-2])
        if seq_len < 1:
            raise ValueError("prefill needs seq_len >= 1")
        if start_pos % PREFILL_ALIGN:
            # Op contract, not a shortcut: chunked SDPA converts the absolute offset to a chunk
            # index by integer division, in both entry points this checkout offers --
            # sdpa_program_factory.cpp:133 (`chunk_start_idx / q_chunk_size`, scalar offset) and
            # kernels/dataflow/reader_interleaved.cpp:260 (same expression, device-tensor offset).
            # A misaligned offset therefore places the causal-mask diagonal in the wrong tile and
            # returns silently wrong values, so reject it instead of rounding. `sdpa_chunk` is the
            # knob: it sets q_chunk_size == PREFILL_ALIGN, and TILE (32) is the floor.
            raise ValueError(f"start_pos must be a multiple of {PREFILL_ALIGN}, got {start_pos}")
        if start_pos + seq_len > cfg.supported_context:
            raise ValueError(
                f"start_pos + seq_len = {start_pos + seq_len} exceeds supported context " f"{cfg.supported_context}"
            )
        if not cfg.is_linear and page_table is None:
            raise ValueError("full_attention prefill requires a page_table")
        if user_id >= cfg.max_batch_size:
            raise ValueError(f"user_id {user_id} >= max_batch_size {cfg.max_batch_size}")

        padded_len = math.ceil(seq_len / PREFILL_ALIGN) * PREFILL_ALIGN
        if padded_len != seq_len:
            x = ttnn.pad(x, padding=[(0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)], value=0.0)

        # linear-attention layers thread the conv left-context and the recurrent state
        # across the internal chunks of this call.
        carry = self._load_linear_carry(user_id) if cfg.is_linear else None
        if carry is not None and start_pos == 0:
            # `start_pos == 0` means this slot is starting a new sequence, so the carry must not
            # be whatever the previous occupant of the slot left behind. Full attention
            # self-heals (this prefill rewrites every block it will later read); the DeltaNet
            # conv left-context and recurrent state do not, so zero them here. Both pieces of
            # `carry` are owned copies (`_load_linear_carry`), so this cannot touch the shared
            # `conv_state` / `recurrent_state` buffers -- they are updated per slot by
            # `_store_linear_carry` at the end of the call.
            for piece in carry:
                _zero_(piece)

        pieces = []
        for offset in range(0, padded_len, cfg.prefill_chunk_size):
            length = min(cfg.prefill_chunk_size, padded_len - offset)
            valid_len = max(0, min(length, seq_len - offset))
            chunk, owns_chunk = _subview(x, [0, 0, offset, 0], [1, 1, offset + length, cfg.hidden_size])
            out, carry = self._prefill_chunk(
                chunk,
                user_id=user_id,
                page_table=page_table,
                abs_pos=start_pos + offset,
                carry=carry,
                valid_len=valid_len,
            )
            if owns_chunk:
                _dealloc(chunk)
            pieces.append(out)
        if padded_len != seq_len:
            _dealloc(x)

        if cfg.is_linear:
            self._store_linear_carry(user_id, carry)

        out = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2)
        if len(pieces) > 1:
            _dealloc(*pieces)
        if padded_len != seq_len:
            sliced, owns = _subview(out, [0, 0, 0, 0], [1, 1, seq_len, cfg.hidden_size])
            if owns:
                _dealloc(out)
            out = sliced
        return out

    def _prefill_chunk(self, x, *, user_id, page_table, abs_pos, carry, valid_len):
        cfg = self.cfg
        normed = ttnn.rms_norm(
            x,
            weight=self.w["input_norm_w"],
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )
        if cfg.is_linear:
            mixed, carry = self._linear_attention_prefill(normed, carry, valid_len)
        else:
            mixed = self._full_attention_prefill(normed, user_id=user_id, page_table=page_table, abs_pos=abs_pos)
        _dealloc(normed)

        hidden = ttnn.add(x, mixed)
        _dealloc(mixed)

        normed2 = ttnn.rms_norm(
            hidden,
            weight=self.w["post_norm_w"],
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )
        moe = self._moe_prefill(normed2)
        _dealloc(normed2)
        out = ttnn.add(hidden, moe)
        _dealloc(hidden, moe)
        return out, carry

    # ---- full attention -----------------------------------------------------------------
    def _full_attention_prefill(self, x, *, user_id, page_table, abs_pos):
        cfg = self.cfg
        seq = int(x.shape[-2])
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        qkv_width = (nh + 2 * nkv) * hd

        fused = ttnn.linear(
            x,
            self.w["qkvg"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        qkv = ttnn.slice(fused, [0, 0, 0, 0], [1, 1, seq, qkv_width])
        gate = ttnn.slice(fused, [0, 0, 0, qkv_width], [1, 1, seq, int(fused.shape[-1])])
        _dealloc(fused)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _dealloc(qkv)

        q = self._norm_heads(q, "q_norm_w")
        k = self._norm_heads(k, "k_norm_w")

        # The tables are ROW_MAJOR (decode's requirement, see `from_state_dict`), so slice the
        # chunk's rows and tilize just those -- `rotary_embedding_hf` wants TILE. `_subview`, not a
        # raw slice: a chunk covering the whole table (abs_pos == 0 and chunk length ==
        # supported_context, reachable whenever supported_context <= prefill_chunk_size) aliases it,
        # and deallocating that would free the layer's weights.
        rope_end = [1, 1, abs_pos + seq, cfg.rotary_dim]
        cos_rows, owns_cos = _subview(self.w["rope_cos"], [0, 0, abs_pos, 0], rope_end)
        sin_rows, owns_sin = _subview(self.w["rope_sin"], [0, 0, abs_pos, 0], rope_end)
        # `_tilized` rather than a bare `to_layout`: `to_layout` returns its *input* when the layout
        # already matches, and the two deallocates below would then double-free. Correct today
        # because the tables are ROW_MAJOR, but that is exactly the assumption items 1 and 5 of
        # README section 7 were bugs about, so it is checked rather than assumed.
        cos, owns_cos_tile = _tilized(cos_rows)
        sin, owns_sin_tile = _tilized(sin_rows)
        if owns_cos:
            _dealloc(cos_rows)
        if owns_sin:
            _dealloc(sin_rows)
        q = self._partial_rope_prefill(q, cos, sin)
        k = self._partial_rope_prefill(k, cos, sin)
        if owns_cos_tile:
            _dealloc(cos)
        if owns_sin_tile:
            _dealloc(sin)

        keys, values = self.kv_cache
        k_fill = ttnn.typecast(k, dtype=keys.dtype) if k.dtype != keys.dtype else k
        v_fill = ttnn.typecast(v, dtype=values.dtype) if v.dtype != values.dtype else v
        block_off = abs_pos // cfg.block_size
        n_blocks = seq // cfg.block_size
        chunk_pt, owns_chunk_pt = _subview(page_table, [0, block_off], [cfg.max_batch_size, block_off + n_blocks])
        ttnn.experimental.paged_fill_cache(keys, k_fill, chunk_pt, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(values, v_fill, chunk_pt, batch_idx=user_id)
        if owns_chunk_pt:
            _dealloc(chunk_pt)
        if k_fill is not k:
            _dealloc(k_fill)
        if v_fill is not v:
            _dealloc(v_fill)
        _dealloc(k, v)

        user_pt, owns_user_pt = _subview(page_table, [user_id, 0], [user_id + 1, int(page_table.shape[-1])])
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q,
            input_tensor_k=keys,
            input_tensor_v=values,
            page_table_tensor=user_pt,
            chunk_start_idx=abs_pos,
            scale=cfg.attn_scale,
            program_config=_sdpa_program_config(cfg.sdpa_chunk),
            compute_kernel_config=self.compute_config,
        )
        _dealloc(q)
        if owns_user_pt:
            _dealloc(user_pt)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _dealloc(attn)
        gated = ttnn.mul(concat, ttnn.sigmoid(gate))
        _dealloc(concat, gate)
        out = ttnn.linear(
            gated,
            self.w["o_proj"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(gated)
        return out

    def _norm_heads(self, heads, weight_key):
        """RMSNorm over head_dim on a ``[1, n_heads, seq, head_dim]`` tensor."""
        return ttnn.rms_norm(
            heads,
            weight=self.w[weight_key],
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )

    def _partial_rope_prefill(self, heads, cos, sin):
        """Rotate only the first ``rotary_dim`` channels of each head (partial RoPE).

        ``rotary_embedding_hf`` implements exactly HF's ``rotate_half`` pairing (i with
        i + d/2) *within the tensor it is given*, so slicing to the rotary block first is
        what makes partial RoPE correct — feeding it the full 256-wide head with a padded
        cos/sin would pair channel i with i+128 and silently mis-rotate.
        """
        cfg = self.cfg
        nh, seq = int(heads.shape[1]), int(heads.shape[2])
        rot = ttnn.slice(heads, [0, 0, 0, 0], [1, nh, seq, cfg.rotary_dim])
        tail = ttnn.slice(heads, [0, 0, 0, cfg.rotary_dim], [1, nh, seq, cfg.head_dim])
        _dealloc(heads)
        rotated = ttnn.experimental.rotary_embedding_hf(rot, cos, sin, is_decode_mode=False)
        _dealloc(rot)
        out = ttnn.concat([rotated, tail], dim=-1)
        _dealloc(rotated, tail)
        return out

    # ---- linear attention (gated DeltaNet) -----------------------------------------------
    def _load_linear_carry(self, user_id):
        """(conv left-context rows, working recurrent state) for ``user_id``."""
        cfg = self.cfg
        views = [_subview(tap, [0, 0, user_id, 0], [1, 1, user_id + 1, cfg.conv_dim]) for tap in self.conv_state]
        rows = [view for view, _ in views]
        ctx = (
            ttnn.concat(rows, dim=2)
            if len(rows) > 1
            else _owned_slice(self.conv_state[0], [0, 0, user_id, 0], [1, 1, user_id + 1, cfg.conv_dim])
        )
        for view, owned in views:
            if owned:
                _dealloc(view)
        hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        # Owned copy: the chunk scan mutates the working state in place, and it is written
        # back through fill_cache, so it must not alias self.recurrent_state.
        slot = _owned_slice(self.recurrent_state, [user_id, 0, 0, 0], [user_id + 1, hv, dk, dv])
        state = ttnn.reshape(slot, (hv, 1, dk, dv))
        return ctx, state

    def _store_linear_carry(self, user_id, carry) -> None:
        cfg = self.cfg
        ctx, state = carry
        hv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        ttnn.fill_cache(self.recurrent_state, ttnn.reshape(state, (1, hv, dk, dv)), user_id)
        _dealloc(state)

        keep, owns_keep = _subview(self.w["slot_keep"], [user_id, 0, 0, 0], [user_id + 1, 1, cfg.max_batch_size, 1])
        onehot, owns_onehot = _subview(
            self.w["slot_onehot"], [user_id, 0, 0, 0], [user_id + 1, 1, cfg.max_batch_size, 1]
        )
        for j, tap in enumerate(self.conv_state):
            row, owns_row = _subview(ctx, [0, 0, j, 0], [1, 1, j + 1, cfg.conv_dim])
            broadcast = ttnn.repeat(row, ttnn.Shape([1, 1, cfg.max_batch_size, 1]))
            owns_broadcast = broadcast.buffer_address() != row.buffer_address()
            merged = ttnn.add(ttnn.mul(tap, keep), ttnn.mul(broadcast, onehot))
            ttnn.copy(merged, tap)
            if owns_broadcast:
                _dealloc(broadcast)
            if owns_row:
                _dealloc(row)
            _dealloc(merged)
        _dealloc(ctx)
        if owns_keep:
            _dealloc(keep)
        if owns_onehot:
            _dealloc(onehot)

    def _linear_attention_prefill(self, x, carry, valid_len):
        cfg = self.cfg
        seq = int(x.shape[-2])
        ctx, state = carry
        hv = cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        kernel = cfg.conv_kernel

        projected = ttnn.linear(
            x,
            self.w["in_proj"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        conv_in = ttnn.slice(projected, [0, 0, 0, 0], [1, 1, seq, cfg.conv_dim])
        ba = ttnn.slice(projected, [0, 0, 0, cfg.conv_dim], [1, 1, seq, cfg.conv_dim + 2 * hv])
        _dealloc(projected)

        # depthwise causal conv1d + silu, as shifted multiply-accumulates over the
        # left-context-prepended input. The z block carries an identity tap so the same
        # silu produces silu(z).
        padded = ttnn.concat([ctx, conv_in], dim=2)
        acc = None
        for j in range(kernel):
            shifted = ttnn.slice(padded, [0, 0, j, 0], [1, 1, j + seq, cfg.conv_dim])
            term = ttnn.mul(shifted, self.w[f"conv_tap_{j}"])
            _dealloc(shifted)
            acc = term if acc is None else ttnn.add(acc, term, output_tensor=acc)
            if acc is not term:
                _dealloc(term)
        # The stored conv context is the last kernel-1 *logical* pre-conv rows. padded row i
        # holds conv_in row i-(kernel-1), so rows [valid_len, valid_len+kernel-1) are exactly
        # conv_in rows [valid_len-(kernel-1), valid_len) — the tail of the real sequence, not
        # of the zero padding.
        new_ctx = ttnn.slice(padded, [0, 0, valid_len, 0], [1, 1, valid_len + kernel - 1, cfg.conv_dim])
        _dealloc(padded, ctx, conv_in)
        activated = ttnn.silu(acc)
        _dealloc(acc)

        qkv = ttnn.slice(activated, [0, 0, 0, 0], [1, 1, seq, cfg.delta_qkv_width])
        silu_z = ttnn.slice(activated, [0, 0, 0, cfg.delta_qkv_width], [1, 1, seq, cfg.conv_dim])
        _dealloc(activated)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=hv, num_kv_heads=hv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _dealloc(qkv)

        beta, g = self._delta_gates(ba, seq)
        _dealloc(ba)
        if valid_len < seq:
            # A padded token must leave the recurrent state untouched:
            #   S <- S * exp(g) + k^T ((v - k S) * beta)
            # so beta = 0 and g = 0 make the step the identity. Masking after the
            # nonlinearities is required (sigmoid(0) = 0.5, softplus(dt_bias) != 0).
            valid = self._valid_mask(valid_len, seq)
            beta = self._masked(beta, valid)
            g = self._masked(g, valid)
            _dealloc(valid)

        core, state = self._gated_delta_rule_prefill(q, k, v, beta, g, state)
        _dealloc(q, k, v, beta, g)

        normed = ttnn.rms_norm(
            core,
            weight=self.w["delta_norm_w"],
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(core)
        concat = ttnn.experimental.nlp_concat_heads(normed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _dealloc(normed)
        gated = ttnn.mul(concat, silu_z)
        _dealloc(concat, silu_z)
        out = ttnn.linear(
            gated,
            self.w["out_proj"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(gated)
        return out, (new_ctx, state)

    def _valid_mask(self, valid_len: int, total: int):
        """``[1, 1, total, 1]`` — 1 for logical tokens, 0 for the padded tail. Device-only."""
        head, owns_head = _subview(self.w["ones_column"], [0, 0, 0, 0], [1, 1, valid_len, 1])
        mask = ttnn.pad(head, padding=[(0, 0), (0, 0), (0, total - valid_len), (0, 0)], value=0.0)
        if owns_head:
            _dealloc(head)
        return mask

    def _masked(self, tensor, valid):
        out = ttnn.mul(tensor, valid)
        _dealloc(tensor)
        return out

    def _delta_gates(self, ba, tokens):
        """(beta, g) as ``[1, hv, tokens, 1]`` fp32 from the fused b|a projection block.

        ``beta = sigmoid(b)``; ``g = -exp(A_log) * softplus(a + dt_bias)``. softplus must run
        in fp32: ``dt_bias`` reaches +15.6 and the bf16 kernel is ~3% off there, which the
        length-``seq`` cumulative gate then amplifies.
        """
        cfg = self.cfg
        hv = cfg.linear_num_value_heads
        b_part = ttnn.slice(ba, [0, 0, 0, 0], [1, 1, tokens, hv])
        a_part = ttnn.slice(ba, [0, 0, 0, hv], [1, 1, tokens, 2 * hv])

        beta = ttnn.permute(ttnn.sigmoid(b_part), (0, 3, 2, 1))
        _dealloc(b_part)

        a32 = ttnn.typecast(a_part, dtype=ttnn.float32)
        _dealloc(a_part)
        shifted = ttnn.add(a32, self.w["dt_bias"])
        _dealloc(a32)
        soft = ttnn.softplus(shifted, beta=1.0, threshold=20.0)
        _dealloc(shifted)
        g = ttnn.mul(soft, self.w["neg_exp_a_log"])
        _dealloc(soft)
        g = ttnn.permute(g, (0, 3, 2, 1))

        beta = self._cast(beta, cfg.delta_dtype)
        g = self._cast(g, cfg.delta_dtype)
        return beta, g

    def _cast(self, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor
        out = ttnn.typecast(tensor, dtype=dtype)
        _dealloc(tensor)
        return out

    def _l2_normalize(self, heads, extra_scale: float = 1.0):
        """Exact ``x * rsqrt(sum(x^2) + 1e-6)`` in fp32, optionally rescaled.

        Written out rather than folded into ``ttnn.rms_norm``: the fused norm is ~4e-3 off
        relative here even at HiFi4/fp32-acc, and these vectors' inner products drive the
        delta-rule recurrence.
        """
        x = self._cast(heads, ttnn.float32) if heads.dtype != ttnn.float32 else heads
        squares = ttnn.mul(x, x)
        total = ttnn.sum(squares, dim=-1, keepdim=True)
        _dealloc(squares)
        inv = ttnn.rsqrt(ttnn.add(total, 1e-6))
        _dealloc(total)
        if extra_scale != 1.0:
            inv = ttnn.mul(inv, extra_scale, output_tensor=inv)
        out = ttnn.mul(x, inv)
        _dealloc(inv)
        if x is not heads:
            _dealloc(x)
        return self._cast(out, self.cfg.delta_dtype)

    def _ut_transform(self, a_mat):
        """``(I - A)^-1`` for strictly-lower-triangular A, by repeated squaring.

        ``A^C = 0`` for a strictly lower triangular ``C x C`` matrix, so
        ``(I-A)^-1 = prod_j (I + A^(2^j))`` for ``j < log2(C)`` — 2*log2(C) batched matmuls
        instead of HF's C-1 serial row updates. Proved in
        ``tests/test_reference_math.py::test_ut_transform_squaring_matches_serial``.
        """
        eye = self.w["delta_eye"]
        acc = ttnn.add(eye, a_mat)
        power = a_mat
        steps = self.cfg.delta_chunk_size.bit_length() - 1
        for step in range(1, steps):
            squared = ttnn.matmul(power, power, compute_kernel_config=self.compute_config)
            if step > 1:
                _dealloc(power)
            power = squared
            factor = ttnn.add(eye, power)
            new_acc = ttnn.matmul(acc, factor, compute_kernel_config=self.compute_config)
            _dealloc(acc, factor)
            acc = new_acc
        if steps > 1:
            _dealloc(power)
        return acc

    def _gated_delta_rule_prefill(self, q, k, v, beta, g, state):
        """Chunked gated delta rule. Returns (core_attn_out [1, hv, seq, dv], new state)."""
        cfg = self.cfg
        chunk = cfg.delta_chunk_size
        hv = cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        seq = int(q.shape[2])
        n_chunks = seq // chunk
        mm = dict(compute_kernel_config=self.compute_config)

        q = self._l2_normalize(q, extra_scale=dk**-0.5)
        k = self._l2_normalize(k)
        v = self._cast(v, cfg.delta_dtype)

        def chunked(tensor, width):
            return ttnn.reshape(tensor, (hv, n_chunks, chunk, width))

        q = chunked(q, dk)
        k = chunked(k, dk)
        v = chunked(v, dv)
        beta = chunked(beta, 1)
        g = chunked(g, 1)

        # within-chunk cumulative gate, as a row-vector @ upper-triangular-ones matmul
        g_row = ttnn.transpose(g, -2, -1)
        _dealloc(g)
        g_cum_row = ttnn.matmul(g_row, self.w["delta_cumsum"], **mm)
        _dealloc(g_row)
        g_cum = ttnn.transpose(g_cum_row, -2, -1)  # [hv, nc, chunk, 1]

        decay = ttnn.exp(ttnn.add(ttnn.sub(g_cum, g_cum_row), self.w["delta_causal_add"]))
        _dealloc(g_cum_row)
        decay_strict = ttnn.mul(decay, self.w["delta_strict_mul"])

        k_beta = ttnn.mul(k, beta)
        v_beta = ttnn.mul(v, beta)
        _dealloc(beta, v)

        k_t = ttnn.transpose(k, -2, -1)
        a_mat = ttnn.mul(ttnn.matmul(k_beta, k_t, **mm), decay_strict)
        _dealloc(decay_strict)
        a_mat = ttnn.mul(a_mat, -1.0, output_tensor=a_mat)
        t_mat = self._ut_transform(a_mat)
        _dealloc(a_mat)

        v_tilde = ttnn.matmul(t_mat, v_beta, **mm)
        _dealloc(v_beta)
        exp_g = ttnn.exp(g_cum)
        k_cumdecay = ttnn.matmul(t_mat, ttnn.mul(k_beta, exp_g), **mm)
        _dealloc(t_mat, k_beta)

        # everything the serial scan needs, precomputed for all chunks at once
        q_scaled = ttnn.mul(q, exp_g)
        g_last = ttnn.slice(g_cum, [0, 0, chunk - 1, 0], [hv, n_chunks, chunk, 1])
        decay_last = ttnn.exp(g_last)
        k_scaled_t = ttnn.transpose(ttnn.mul(k, ttnn.exp(ttnn.sub(g_last, g_cum))), -2, -1)
        _dealloc(exp_g, g_cum, g_last, k)

        # One split call per tensor instead of n_chunks slices per tensor: for a 262144-token
        # prefill that is 8 op launches per prefill chunk rather than 8 * n_chunks.
        pre_split = (q, q_scaled, k_t, k_scaled_t, v_tilde, k_cumdecay, decay, decay_last)
        splits = [ttnn.split(tensor, 1, dim=1) for tensor in pre_split]
        q_c, qg_c, kt_c, kst_c, vt_c, kcd_c, decay_c, dlast_c = splits

        outputs = []
        for i in range(n_chunks):
            attn = ttnn.mul(ttnn.matmul(q_c[i], kt_c[i], **mm), decay_c[i])
            v_new = ttnn.sub(vt_c[i], ttnn.matmul(kcd_c[i], state, **mm))
            out_i = ttnn.add(ttnn.matmul(qg_c[i], state, **mm), ttnn.matmul(attn, v_new, **mm))
            _dealloc(attn)
            outputs.append(out_i)
            ttnn.mul(state, dlast_c[i], output_tensor=state)
            ttnn.add(state, ttnn.matmul(kst_c[i], v_new, **mm), output_tensor=state)
            _dealloc(v_new)
        # A single-chunk split can hand back the input itself, so free the parents and let the
        # (possibly aliasing) per-chunk handles fall out of scope rather than freeing both.
        for group, parent in zip(splits, pre_split):
            for piece in group:
                if piece.buffer_address() != parent.buffer_address():
                    _dealloc(piece)
        _dealloc(*pre_split)

        core = ttnn.concat(outputs, dim=1) if n_chunks > 1 else outputs[0]
        if n_chunks > 1:
            _dealloc(*outputs)
        core = ttnn.reshape(core, (1, hv, seq, dv))
        return self._cast(core, cfg.activation_dtype), state

    # -----------------------------------------------------------------------------------
    # decode
    # -----------------------------------------------------------------------------------
    def decode_forward(self, x, *, current_pos=None, page_table=None):
        cfg = self.cfg
        batch = int(x.shape[-2])
        if batch != cfg.max_batch_size:
            raise ValueError(
                f"decode batch must equal max_batch_size ({cfg.max_batch_size}), got {batch}. "
                "The linear-attention conv/recurrent state buffers are updated whole-tensor "
                "in place so a traced replay can reuse their addresses."
            )
        if not cfg.is_linear:
            if page_table is None:
                raise ValueError("full_attention decode requires a page_table")
            if current_pos is None:
                raise ValueError("full_attention decode requires current_pos")

        normed = ttnn.rms_norm(
            x,
            weight=self.w["input_norm_w"],
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )
        if cfg.is_linear:
            mixed = self._linear_attention_decode(normed)
        else:
            mixed = self._full_attention_decode(normed, current_pos=current_pos, page_table=page_table)
        _dealloc(normed)

        hidden = ttnn.add(x, mixed)
        _dealloc(mixed)
        normed2 = ttnn.rms_norm(
            hidden,
            weight=self.w["post_norm_w"],
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )
        moe = self._moe_decode(normed2)
        _dealloc(normed2)
        out = ttnn.add(hidden, moe)
        _dealloc(hidden, moe)
        return out

    def _full_attention_decode(self, x, *, current_pos, page_table):
        cfg = self.cfg
        batch = cfg.max_batch_size
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        qkv_width = (nh + 2 * nkv) * hd

        fused = ttnn.linear(
            x,
            self.w["qkvg"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        qkv = ttnn.slice(fused, [0, 0, 0, 0], [1, 1, batch, qkv_width])
        gate = ttnn.slice(fused, [0, 0, 0, qkv_width], [1, 1, batch, int(fused.shape[-1])])
        _dealloc(fused)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=nh,
            num_kv_heads=nkv,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
        )
        _dealloc(qkv)

        # q/k RMSNorm needs an interleaved input; the paged cache update needs a sharded one.
        k_shard_cfg = k.memory_config()
        q = self._decode_norm_interleaved(q, "q_norm_w")
        k_int = self._decode_norm_interleaved(k, "k_norm_w")

        cos, sin = self._decode_rope(current_pos)
        q = self._partial_rope_decode(q, cos, sin, nh)
        k_int = self._partial_rope_decode(k_int, cos, sin, nkv)
        _dealloc(cos, sin)

        keys, values = self.kv_cache
        k_sharded = _move(k_int, k_shard_cfg)
        ttnn.experimental.paged_update_cache(keys, k_sharded, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(values, v, update_idxs_tensor=current_pos, page_table=page_table)
        _dealloc(k_sharded, v)

        q_sdpa = _move(q, ttnn.DRAM_MEMORY_CONFIG)
        sdpa_kwargs = {}
        if self.decode_sdpa_config is not None:
            sdpa_kwargs["program_config"] = self.decode_sdpa_config
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_sdpa,
            keys,
            values,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=cfg.attn_scale,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **sdpa_kwargs,
        )
        _dealloc(q_sdpa)

        # view of attn: freeing `concat` below releases the single shared buffer
        concat = _view(attn, (1, 1, batch, nh * hd))
        gated = ttnn.mul(concat, ttnn.sigmoid(gate))
        _dealloc(concat, gate)
        out = ttnn.linear(
            gated,
            self.w["o_proj"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(gated)
        return out

    def _decode_norm_interleaved(self, heads, weight_key):
        interleaved = _move(heads, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.rms_norm(
            interleaved,
            weight=self.w[weight_key],
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(interleaved)
        return out

    def _decode_rope(self, current_pos):
        """cos/sin ``[1, batch, 1, rotary_dim]`` looked up from the on-device RoPE table.

        ``current_pos`` is clamped to ``>= 0`` first: a slot marked inactive with ``-1`` would
        otherwise reach ``ttnn.embedding`` as a huge unsigned index and read outside the table.
        The clamped row is discarded downstream (SDPA skips that slot), so the clamp only has to
        keep the lookup in bounds.
        """
        cfg = self.cfg
        idx = ttnn.reshape(current_pos, (1, cfg.max_batch_size))
        idx = ttnn.maximum(idx, 0)
        idx = ttnn.typecast(idx, dtype=ttnn.uint32)
        # The tables are ROW_MAJOR (see `from_state_dict`), which is what keeps this a gather of
        # `max_batch_size` rows instead of an untilize of the whole table.
        cos = ttnn.embedding(idx, self.w["rope_cos"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.w["rope_sin"], layout=ttnn.TILE_LAYOUT)
        _dealloc(idx)
        cos = ttnn.reshape(cos, (1, cfg.max_batch_size, 1, cfg.rotary_dim))
        sin = ttnn.reshape(sin, (1, cfg.max_batch_size, 1, cfg.rotary_dim))
        return cos, sin

    def _partial_rope_decode(self, heads, cos, sin, n_heads):
        """Partial RoPE on ``[1, batch, n_heads, head_dim]`` with per-slot positions.

        Written out instead of calling ``rotary_embedding_hf``: its decode mode needs a
        height-sharded input, and binary broadcasting cannot span the heads (height) dim
        inside a tile, so cos/sin are materialised over heads first.
        """
        cfg = self.cfg
        batch = cfg.max_batch_size
        half = cfg.rotary_dim // 2
        cos_b = ttnn.repeat(cos, ttnn.Shape([1, 1, n_heads, 1]))
        sin_b = ttnn.repeat(sin, ttnn.Shape([1, 1, n_heads, 1]))
        rot = ttnn.slice(heads, [0, 0, 0, 0], [1, batch, n_heads, cfg.rotary_dim])
        tail = ttnn.slice(heads, [0, 0, 0, cfg.rotary_dim], [1, batch, n_heads, cfg.head_dim])
        _dealloc(heads)
        first = ttnn.slice(rot, [0, 0, 0, 0], [1, batch, n_heads, half])
        second = ttnn.slice(rot, [0, 0, 0, half], [1, batch, n_heads, cfg.rotary_dim])
        swapped = ttnn.concat([ttnn.neg(second), first], dim=-1)
        _dealloc(first, second)
        rotated = ttnn.add(ttnn.mul(rot, cos_b), ttnn.mul(swapped, sin_b))
        _dealloc(rot, swapped, cos_b, sin_b)
        out = ttnn.concat([rotated, tail], dim=-1)
        _dealloc(rotated, tail)
        return out

    def _linear_attention_decode(self, x):
        cfg = self.cfg
        batch = cfg.max_batch_size
        hv = cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        mm = dict(compute_kernel_config=self.compute_config)

        projected = ttnn.linear(
            x,
            self.w["in_proj"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        conv_in = ttnn.slice(projected, [0, 0, 0, 0], [1, 1, batch, cfg.conv_dim])
        ba = ttnn.slice(projected, [0, 0, 0, cfg.conv_dim], [1, 1, batch, cfg.conv_dim + 2 * hv])
        _dealloc(projected)

        # depthwise conv over [oldest tap, ..., newest tap, current token]
        acc = ttnn.mul(conv_in, self.w[f"conv_tap_{cfg.conv_kernel - 1}"])
        for j, tap in enumerate(self.conv_state):
            ttnn.add(acc, ttnn.mul(tap, self.w[f"conv_tap_{j}"]), output_tensor=acc)
        # shift the taps: oldest <- next, ..., newest <- current token (in place, so the
        # captured trace keeps writing the same buffers)
        for j in range(len(self.conv_state) - 1):
            ttnn.copy(self.conv_state[j + 1], self.conv_state[j])
        ttnn.copy(conv_in, self.conv_state[-1])
        _dealloc(conv_in)

        activated = ttnn.silu(acc)
        _dealloc(acc)
        qkv = ttnn.slice(activated, [0, 0, 0, 0], [1, 1, batch, cfg.delta_qkv_width])
        silu_z = ttnn.slice(activated, [0, 0, 0, cfg.delta_qkv_width], [1, 1, batch, cfg.conv_dim])
        _dealloc(activated)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=hv,
            num_kv_heads=hv,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
        )
        _dealloc(qkv)
        q, k, v = (_move(h, ttnn.DRAM_MEMORY_CONFIG) for h in (q, k, v))

        # [1, batch, hv, d] -> [batch, hv, 1, d] so the state matmuls batch over (slot, head)
        q = ttnn.permute(q, (1, 2, 0, 3))
        k = ttnn.permute(k, (1, 2, 0, 3))
        v = ttnn.permute(v, (1, 2, 0, 3))

        beta, g = self._delta_gates(ba, batch)  # [1, hv, batch, 1]
        _dealloc(ba)
        beta = ttnn.permute(beta, (2, 1, 0, 3))  # -> [batch, hv, 1, 1]
        g = ttnn.permute(g, (2, 1, 0, 3))

        q = self._l2_normalize(q, extra_scale=dk**-0.5)
        k = self._l2_normalize(k)
        v = self._cast(v, cfg.delta_dtype)

        state = self.recurrent_state
        ttnn.mul(state, ttnn.exp(g), output_tensor=state)
        kv_mem = ttnn.matmul(k, state, **mm)
        delta = ttnn.mul(ttnn.sub(v, kv_mem), beta)
        _dealloc(kv_mem, v, beta, g)
        ttnn.add(state, ttnn.matmul(ttnn.transpose(k, -2, -1), delta, **mm), output_tensor=state)
        _dealloc(delta, k)
        core = ttnn.matmul(q, state, **mm)  # [batch, hv, 1, dv]
        _dealloc(q)

        core = self._cast(core, cfg.activation_dtype)
        normed = ttnn.rms_norm(
            core,
            weight=self.w["delta_norm_w"],
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(core)
        permuted = ttnn.permute(normed, (2, 0, 1, 3))
        concat = _view(permuted, (1, 1, batch, hv * dv))
        _dealloc(normed)
        gated = ttnn.mul(concat, silu_z)
        _dealloc(concat, silu_z)
        out = ttnn.linear(
            gated,
            self.w["out_proj"],
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(gated)
        return out

    # -----------------------------------------------------------------------------------
    # MoE
    # -----------------------------------------------------------------------------------
    def _router(self, x, tokens):
        """(dense top-k weights, binary sparsity mask) both ``[1, 1, tokens, num_experts]``.

        HF softmaxes over all 256 experts, takes the top 8 and renormalises by their sum.
        Softmax is monotonic, so top-k of the raw logits selects the same experts, and a
        softmax over just those 8 values *is* the renormalised weight — no 256-wide softmax
        needed. The mask is kept separate from the weights so the sparse-matmul sparsity
        pattern is exactly ``num_experts_per_tok`` per token even when a routing weight
        rounds to zero in bf16.
        """
        cfg = self.cfg
        logits = ttnn.linear(
            x,
            self.w["router_w"],
            dtype=ttnn.bfloat16,  # ttnn.topk accepts bfloat16 only
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        values, indices = ttnn.topk(logits, k=cfg.num_experts_per_tok, dim=-1, sorted=True)

        v32 = ttnn.typecast(values, dtype=ttnn.float32)
        _dealloc(values)
        shifted = ttnn.sub(v32, ttnn.max(v32, dim=-1, keepdim=True))
        exps = ttnn.exp(shifted)
        _dealloc(shifted, v32)
        weights = ttnn.div(exps, ttnn.sum(exps, dim=-1, keepdim=True))
        _dealloc(exps)
        weights = ttnn.typecast(weights, dtype=ttnn.bfloat16)

        zeros = ttnn.zeros_like(logits)
        dense = ttnn.scatter(zeros, dim=3, index=indices, src=weights)
        mask = ttnn.scatter(zeros, dim=3, index=indices, src=ttnn.ones_like(weights))
        _dealloc(zeros, logits, weights, indices)
        return dense, mask

    def _shared_expert(self, x):
        cfg = self.cfg
        mm = dict(
            dtype=cfg.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        gate = ttnn.silu(ttnn.linear(x, self.w["shared_gate_proj"], **mm))
        up = ttnn.linear(x, self.w["shared_up_proj"], **mm)
        hidden = ttnn.mul(gate, up)
        _dealloc(gate, up)
        out = ttnn.linear(hidden, self.w["shared_down_proj"], **mm)
        _dealloc(hidden)
        sig = ttnn.sigmoid(ttnn.linear(x, self.w["shared_expert_gate"], **mm))
        gated = ttnn.mul(out, sig)
        _dealloc(out, sig)
        return gated

    def _experts(self, x_groups, sparsity, weights_per_row, group_tokens):
        """Active-expert MoE over ``[1, G, group_tokens, hidden]``.

        ``sparsity`` is ``[1, G, 1, E]`` (which experts the group needs) and
        ``weights_per_row`` is ``[G, E, group_tokens, 1]`` (per-token routing weight).
        ``nnz`` is deliberately left inferred: a static count that disagrees with the actual
        non-zeros deadlocks the sparse-matmul mcast receivers.
        """
        cfg = self.cfg
        groups = int(x_groups.shape[1])
        e, h, inter = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size

        gate_up = ttnn.sparse_matmul(
            x_groups,
            self.w["expert_gate_up"],
            sparsity=sparsity,
            nnz=None,
            program_config=_sparse_program_config(group_tokens, 2 * inter, h),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=cfg.activation_dtype,
            compute_kernel_config=self.compute_config,
        )
        gate_up = _view(gate_up, (groups, e, group_tokens, 2 * inter))
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [groups, e, group_tokens, inter])
        up = ttnn.slice(gate_up, [0, 0, 0, inter], [groups, e, group_tokens, 2 * inter])
        _dealloc(gate_up)
        down_in = ttnn.mul(ttnn.silu(gate), up)
        _dealloc(gate, up)

        # down projection consumes a per-expert input, so the sparsity is [1, 1, G, E].
        # This is a *view* of the caller's sparsity tensor - do not deallocate it here.
        down_sparsity = _view(sparsity, (1, 1, groups, e))
        down = ttnn.sparse_matmul(
            down_in,
            self.w["expert_down"],
            sparsity=down_sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=_sparse_program_config(group_tokens, h, inter),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=cfg.activation_dtype,
            compute_kernel_config=self.compute_config,
        )
        _dealloc(down_in)
        weighted = ttnn.mul(down, weights_per_row)
        _dealloc(down)
        reduced = ttnn.sum(weighted, dim=1, keepdim=True)  # [G, 1, group_tokens, hidden]
        _dealloc(weighted)
        return reduced

    def _moe_prefill(self, x):
        cfg = self.cfg
        seq = int(x.shape[-2])
        e, h = cfg.num_experts, cfg.hidden_size
        shared = self._shared_expert(x)

        # Chunked over tokens: the sparse-matmul intermediates are dense over all 256 experts
        # ([tokens, 256, 32, 2048] for the down projection), so the chunk size is what bounds
        # peak DRAM rather than any correctness concern.
        reduced_chunks = []
        step = cfg.moe_prefill_chunk_tokens
        for offset in range(0, seq, step):
            tokens = min(step, seq - offset)
            block, owns_block = _subview(x, [0, 0, offset, 0], [1, 1, offset + tokens, h])
            groups = tokens // TILE
            dense, mask = self._router(block, tokens)
            # views of block / mask: those two stay the owners
            grouped = _view(block, (1, groups, TILE, h))
            mask_grouped = _view(mask, (1, groups, TILE, e))
            # one sparsity pattern per tile group = the union of its 32 tokens' top-k
            sparsity = ttnn.max(mask_grouped, dim=2, keepdim=True)
            sparsity_rm = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
            weights = ttnn.permute(_view(dense, (1, groups, TILE, e)), (1, 3, 2, 0))
            _dealloc(sparsity)

            reduced_chunks.append(self._experts(grouped, sparsity_rm, weights, TILE))
            _dealloc(sparsity_rm, weights, dense, mask)
            if owns_block:
                _dealloc(block)

        # [G, 1, 32, hidden] -> [1, 1, G*32, hidden]; the chunk tensors stay the owners
        views = [_view(chunk, (1, 1, int(chunk.shape[0]) * TILE, h)) for chunk in reduced_chunks]
        experts_out = views[0] if len(views) == 1 else ttnn.concat(views, dim=2)
        if len(views) > 1:
            _dealloc(*reduced_chunks)
        out = ttnn.add(experts_out, shared)
        _dealloc(experts_out, shared)
        return out

    def _moe_decode(self, x):
        cfg = self.cfg
        batch = cfg.max_batch_size
        e, h = cfg.num_experts, cfg.hidden_size
        shared = self._shared_expert(x)

        dense, mask = self._router(x, batch)
        # one token per group: [1, 1, batch, *] -> [1, batch, 1, *]
        rows = ttnn.transpose(x, 1, 2)
        mask_rows = ttnn.transpose(mask, 1, 2)
        sparsity = ttnn.to_layout(mask_rows, ttnn.ROW_MAJOR_LAYOUT)
        weights = ttnn.permute(dense, (2, 3, 0, 1))  # [batch, e, 1, 1]
        _dealloc(dense, mask, mask_rows)

        reduced = self._experts(rows, sparsity, weights, 1)
        _dealloc(rows, sparsity, weights)
        experts_out = ttnn.permute(reduced, (1, 2, 0, 3))  # [G,1,1,h] -> [1,1,batch,h]
        _dealloc(reduced)
        out = ttnn.add(experts_out, shared)
        _dealloc(experts_out, shared)
        return out
