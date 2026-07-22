# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor

MAX_CHUNK_SIZE = 128


@dataclass
class StateConversion:
    """Declarative torch-state-dict key remapping: ordered regex renames, then removes."""

    rename: Sequence[tuple[str, str]] = ()
    remove: Sequence[str] = ()

    def convert(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        in_ = dict(state_dict)
        out: dict[str, torch.Tensor] = {}
        compiled = [(re.compile(p), t) for (p, t) in self.rename]
        removes = [re.compile(p) for p in self.remove]
        for k in list(in_):
            transformed = False
            for pattern, t in compiled:
                new_k, count = pattern.subn(t, k, count=1)
                if count == 1:
                    out[new_k] = in_.pop(k)
                    transformed = True
                    break
            if transformed:
                continue
            for pattern in removes:
                if pattern.search(k):
                    in_.pop(k)
                    transformed = True
                    break
            if not transformed:
                warnings.warn(f"unprocessed key: {k}", stacklevel=2)
        return {**in_, **out}


# Strip a leading `model.` from encoder submodules; drop the LM head and rotary buffers.
# Works on both the full SmolLM3ForCausalLM dict (model.* + lm_head) and the inner
# SmolLM3Model dict (already-stripped keys pass through the rename as no-ops).
STATE_CONVERSION = StateConversion(
    rename=[(r"^(?:model\.)?(embed_tokens|layers|norm)", r"\1")],
    remove=[r"(?:^|\.)lm_head(?:\.|$)", r"(?:^|\.)rotary_emb(?:\.|$)"],
)


def create_rope_tensors(
    batch_size: int, sequence_length: int, head_dim: int, rope_theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain single-axis RoPE tables matching HF SmolLM3RotaryEmbedding (attention_scaling=1.0).

    Returns (cos, sin) each shaped (batch, 1, seq, head_dim), full-width (non-interleaved).
    """
    position_ids = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)  # (B, seq)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)  # (B, hd/2, 1)
    position_ids_expanded = position_ids[:, None, :].float()  # (B, 1, seq)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # (B, seq, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, seq, hd)
    cos = emb.cos().unsqueeze(1)  # (B, 1, seq, hd)
    sin = emb.sin().unsqueeze(1)
    return cos, sin


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    return x * cos + _rotate_half(x) * sin


def optimal_groups(group_count: int, group_size: int, device_count: int) -> tuple[int, int, int]:
    # In order to distribute heads evenly on devices, three operations are possibly performed:
    # 1. Pad to increase group size.
    # 2. Pad to increase group count (= number of key/value heads).
    # 3. Split groups into smaller groups defined by a split factor.
    # For a particular split factor, padding sizes follow from the requirements that the padded
    # group size must be divisible by this factor and the new group count must be divisible by the
    # device count. We choose this factor such that memory requirements are minimized.

    best_split_factor = 1
    best_size = math.inf
    best_group_count = group_count
    best_group_size = group_size

    for s in range(1, group_size + 1):
        new_group_size = -(-group_size // s)  # = ceil(group_size / s)
        new_group_count = -(-group_count * s // device_count) * device_count

        # query heads + 2 * key/value heads
        size = new_group_size * new_group_count + 2 * new_group_count

        if size < best_size:
            best_size = size
            best_split_factor = s
            best_group_count = new_group_count
            best_group_size = new_group_size

    return best_group_count, best_group_size, best_split_factor


def _pad(t: torch.Tensor, amount: int, *, dim: int) -> torch.Tensor:
    """Pad tensor with `amount` zeros on the end of dimension `dim`."""
    padding = [0] * (2 * t.ndim)
    padding[-(dim * 2 + 1)] = amount
    return torch.nn.functional.pad(t, padding)


def prepare_attention_bias(attention_mask: ttnn.Tensor) -> ttnn.Tensor:
    batch_size, seq_len = attention_mask.shape

    # convert to causal attention mask
    attention_mask = attention_mask.reshape([batch_size, 1, 1, seq_len])
    attention_mask = ttnn.expand(attention_mask, [-1, -1, seq_len, -1])
    attention_mask = tensor.tril(attention_mask)

    attention_mask = (attention_mask - 1.0) * math.inf

    return ttnn.clone(attention_mask, dtype=ttnn.bfloat4_b)


def build_sp_causal_bias(seq_local: int, sp_factor: int, *, device: ttnn.MeshDevice, sp_axis: int) -> ttnn.Tensor:
    """Per-shard rectangular causal bias for sequence-parallel attention.

    The sequence is sharded across ``sp_axis``; shard ``r`` holds query rows for global positions
    ``[r*seq_local, (r+1)*seq_local)`` while K/V are all-gathered to the full ``seq_local*sp_factor``.
    Row ``i`` (global ``r*seq_local+i``) may attend key ``j`` iff ``j <= r*seq_local+i``. The result
    is sharded along ``sp_axis`` so each device receives only its own ``(1,1,seq_local,seq_total)`` slice.
    """
    seq_total = seq_local * sp_factor
    q_idx = torch.arange(seq_local)
    k_idx = torch.arange(seq_total)
    masks = []
    for r in range(sp_factor):
        allow = k_idx[None, :] <= (r * seq_local + q_idx)[:, None]  # (seq_local, seq_total)
        masks.append(torch.where(allow, 0.0, float("-inf")))
    stacked = torch.stack(masks, dim=0).reshape(sp_factor, 1, seq_local, seq_total)
    return tensor.from_torch(stacked, device=device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None])


@dataclass
class SmolLM3Context:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    fsdp_mesh_axis: int | None = None
    sp_axis: int | None = None
    sp_factor: int = 1


class SmolLM3RmsNorm(RMSNorm):
    def __init__(self, size: int, *, eps: float, ctx: SmolLM3Context) -> None:
        super().__init__(size, norm_eps=eps, bias=False, mesh_device=ctx.device)
        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(x, compute_kernel_config=self._compute_kernel_config)


class SmolLM3Mlp(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ctx: SmolLM3Context,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        # intermediate_size is much greater than hidden_size
        self.gate_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.up_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        if hidden_act != "silu":
            msg = f"unsupported activation function: {hidden_act}"
            raise ValueError(msg)
        self.act_fn = ttnn.silu

        self._ccl_manager = ctx.ccl_manager
        self._tp_axis = ctx.tp_axis
        self._tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.act_fn(self.gate_proj.forward(x)) * self.up_proj.forward(x)
        x = self.down_proj(x)
        if self._tp_factor > 1:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
        return x


class SmolLM3Attention(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        use_rope: bool,
        ctx: SmolLM3Context,
    ) -> None:
        super().__init__()

        self._use_rope = use_rope

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        if hidden_size % num_heads != 0:
            msg = f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            raise ValueError(msg)

        head_dim = hidden_size // num_heads
        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        group_count = num_key_value_heads
        group_size = num_heads // num_key_value_heads

        opt_group_count, opt_group_size, split_factor = optimal_groups(group_count, group_size, tp_factor)
        padded_heads = opt_group_count * opt_group_size

        self.qkv_proj = ColParallelLinear(
            hidden_size,
            (padded_heads + 2 * opt_group_count) * head_dim,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.o_proj = ColParallelLinear(
            padded_heads * head_dim,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True,
        )

        self._head_dim = head_dim
        self._group_count = group_count
        self._group_size = group_size
        self._num_local_heads = padded_heads // tp_factor
        self._num_local_kv_heads = opt_group_count // tp_factor
        self._group_size_padding = opt_group_size * split_factor - group_size
        self._group_count_padding = opt_group_count - group_count * split_factor
        self._split_factor = split_factor
        self._tp_axis = ctx.tp_axis
        self._tp_factor = tp_factor
        self._sp_axis = ctx.sp_axis
        self._sp_factor = ctx.sp_factor
        self._device = ctx.device
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def _prepare_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            q = q.unflatten(0, [self._group_count, self._group_size, self._head_dim])
            k = k.unflatten(0, [self._group_count, 1, self._head_dim])
            v = v.unflatten(0, [self._group_count, 1, self._head_dim])

            # pad group size
            q = _pad(q, self._group_size_padding, dim=1)

            # split groups
            s = self._split_factor
            q = q.flatten(0, 1).unflatten(0, [self._group_count * s, -1])
            k = k.repeat_interleave(s, dim=0)
            v = v.repeat_interleave(s, dim=0)

            # pad group count
            q = _pad(q, self._group_count_padding, dim=0)
            k = _pad(k, self._group_count_padding, dim=0)
            v = _pad(v, self._group_count_padding, dim=0)

            # fuse
            q = q.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_heads])
            k = k.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])
            v = v.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])

            return torch.cat([q, k, v], dim=1).flatten(0, 2)

        if "q_proj.weight" in state and "k_proj.weight" in state and "v_proj.weight" in state:
            state["qkv_proj.weight"] = _prepare_qkv(
                state.pop("q_proj.weight"), state.pop("k_proj.weight"), state.pop("v_proj.weight")
            )

        if "o_proj.weight" in state:
            o = state["o_proj.weight"]

            o = o.unflatten(1, [self._group_count, self._group_size, self._head_dim])

            # pad group size
            o = _pad(o, self._group_size_padding, dim=2)

            # split groups
            o = o.flatten(1, 2).unflatten(1, [self._group_count * self._split_factor, -1])

            # pad group count
            o = _pad(o, self._group_count_padding, dim=1)

            state["o_proj.weight"] = o.flatten(1, 3)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attention_bias: ttnn.Tensor | None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        x = self.qkv_proj.forward(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(x, 1),
            num_heads=self._num_local_heads,
            num_kv_heads=self._num_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos, sin = pos_embeds
        if self._use_rope:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        if self._sp_factor > 1:
            # Sequence-parallel: gather full-sequence K/V (already RoPE'd) across the sp axis so each
            # shard's local Q attends the whole sequence. The rectangular causal bias carries the
            # shard's global offset, so is_causal cannot be used here.
            k = self._ccl_manager.all_gather_persistent_buffer(k, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)
            v = self._ccl_manager.all_gather_persistent_buffer(v, dim=2, mesh_axis=self._sp_axis, use_hyperparams=True)

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            is_causal=attention_bias is None,
            program_config=self._sdpa_program_config(k.shape[2]),
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        x = ttnn.transformer.concatenate_heads(x)

        if self._tp_factor > 1:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.o_proj.forward(x)

        if self._tp_factor > 1:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x

    def _sdpa_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        grid_size = self._device.compute_with_storage_grid_size()

        seq_len = -(-seq_len // 32) * 32
        chunk_size = min(seq_len, MAX_CHUNK_SIZE)

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=chunk_size,
            k_chunk_size=chunk_size,
            exp_approx_mode=False,
        )


class SmolLM3DecoderLayer(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float,
        use_rope: bool,
        ctx: SmolLM3Context,
    ) -> None:
        super().__init__()

        self.self_attn = SmolLM3Attention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            use_rope=use_rope,
            ctx=ctx,
        )
        self.mlp = SmolLM3Mlp(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act, ctx=ctx
        )
        self.input_layernorm = SmolLM3RmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)
        self.post_attention_layernorm = SmolLM3RmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attention_bias: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x, attention_bias=attention_bias, pos_embeds=pos_embeds)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        return x + residual


class SmolLM3TextEncoder(Module):
    def __init__(
        self,
        config: "SmolLM3Config",
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
        ccl_manager: CCLManager | None = None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        tp_factor = parallel_config.tensor_parallel.factor
        fsdp_mesh_axis = None
        if is_fsdp and tp_factor > 1:
            other = 1 - tp_axis
            if device.shape[other] > 1:
                fsdp_mesh_axis = other
        sp = parallel_config.sequence_parallel
        sp_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None
        sp_factor = sp.factor if sp is not None else 1
        ctx = SmolLM3Context(
            device=device,
            tp_axis=tp_axis if tp_factor > 1 else None,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
        )
        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            raise ValueError("ccl_manager must be provided if tensor parallelism is used")
        if ctx.sp_axis is not None and ctx.ccl_manager is None:
            raise ValueError("ccl_manager must be provided if sequence parallelism is used")
        self._sp_axis = ctx.sp_axis
        self._sp_factor = ctx.sp_factor

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, device=device)
        self.layers = ModuleList(
            SmolLM3DecoderLayer(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                rms_norm_eps=config.rms_norm_eps,
                use_rope=bool(config.no_rope_layers[i]),
                ctx=ctx,
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = SmolLM3RmsNorm(config.hidden_size, eps=config.rms_norm_eps, ctx=ctx)

        self._config = config
        self._device = device
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta
        self._sp_bias_cache: dict[int, ttnn.Tensor] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        converted = STATE_CONVERSION.convert(state)
        state.clear()
        state.update(converted)

    def create_rope_tensors(self, batch_size: int, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return create_rope_tensors(batch_size, sequence_length, self._head_dim, self._rope_theta)

    def encode(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]:
        """Return (prompt_embeds, all_hidden_states) matching the FIBO output contract.

        prompt_embeds = concat(all_hidden_states[-1], all_hidden_states[-2], dim=-1)
        shape: [B, T, 2 * hidden_size]
        """
        all_hidden_states = self.forward(input_ids, attention_mask=attention_mask, pos_embeds=pos_embeds)
        prompt_embeds = ttnn.concat([all_hidden_states[-1], all_hidden_states[-2]], dim=-1)
        return prompt_embeds, all_hidden_states

    def forward(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> list[ttnn.Tensor]:
        # When sequence-parallel, ``.shape`` reports the LOCAL (per-shard) sequence length; the full
        # sequence is ``seq_len * sp_factor`` and lives across the sp axis after the K/V all-gather.
        batch_size, seq_len = input_ids.shape

        if self._sp_factor > 1:
            # Sharded seq: per-shard rectangular causal bias (global offset baked in), threaded through
            # every layer. It is constant for a given local seq length, so build it once and cache it:
            # that keeps the host->device build out of a captured trace (it runs during the tracer's
            # prep_run and is only read inside capture/replay). No local padding -- the wrapper buckets.
            padded = seq_len
            attention_bias = self._sp_bias_cache.get(seq_len)
            if attention_bias is None:
                attention_bias = build_sp_causal_bias(
                    seq_len, self._sp_factor, device=self._device, sp_axis=self._sp_axis
                )
                self._sp_bias_cache[seq_len] = attention_bias
        elif attention_mask is not None:
            padded = (
                -(-seq_len // 32) * 32 if seq_len < MAX_CHUNK_SIZE else -(-seq_len // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE
            )
            input_ids = ttnn.pad(input_ids, [(0, 0), (0, padded - seq_len)], value=0)
            pos_embeds = tuple(
                ttnn.pad(x, [(0, 0), (0, 0), (0, padded - seq_len), (0, 0)], value=0) for x in pos_embeds
            )
            attention_mask = ttnn.pad(attention_mask, [(0, 0), (0, padded - seq_len)], value=0)
            attention_bias = prepare_attention_bias(attention_mask)
        else:
            padded = seq_len
            attention_bias = None

        hidden_states = self.embed_tokens.forward(input_ids)

        # HF output_hidden_states convention: append the INPUT to each layer, then the final norm.
        all_hidden_states: list[ttnn.Tensor] = []
        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            hidden_states = layer.forward(hidden_states, attention_bias=attention_bias, pos_embeds=pos_embeds)
        hidden_states = self.norm.forward(hidden_states)
        all_hidden_states.append(hidden_states)

        if padded != seq_len:
            all_hidden_states = [x[:, :seq_len, :] for x in all_hidden_states]
        return all_hidden_states
