# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""2D tensor-parallel MLA projections for the Hy4-preview SC1 layout (TP32 / DP1, #65885).

ttMLA shards its projections over ONE mesh axis (TP) and replicates them over the other (SP, which carries the
sequence). ``ttMLA2DTP`` shards every projection over BOTH axes, once, with nothing replicated:

  * the HEAD axis (``tp_axis``, 8 chips): heads, 8 per chip column;
  * the HIDDEN axis (``sp_axis``, 4 chips): the 6144 hidden dimension (1536 per chip), and inside the attention
    core the sequence, as ttMLA's SP axis.

Activation contract (Llama-70B-on-TG style): the sublayer input ``hn`` and the output are ``[1, 1, S, hidden / 4]``,
hidden-sharded over the hidden axis and replicated over the head axis, for the whole chunk ``S`` (no sequence split).

Per-weight sharding (``hp`` = hidden-axis size 4, ``tp`` = head-axis size 8; per-chip shapes for Hy4):

  ============  ==========================  ===========================================  =========================
  weight        [K, N]                      split                                        per chip
  ============  ==========================  ===========================================  =========================
  q_a_proj      [6144, 2048]                K over hidden, N over heads                  [1536, 256]
  q_b_proj      [2048, 64 * 256]            K (q_lora) over hidden, N (heads) over heads [512, 8 heads * 256]
  kv_a_proj     [6144, 576]                 K over hidden x heads (576 is not 8 tiles)   [192, 576]
  wkv_b1        [64, 192, 512] per head     heads over heads x hidden (2 per chip)       [2, 192, 512]
  wkv_b2        [64, 512, 256] per head     heads over heads x hidden (2 per chip)       [2, 512, 256]
  g_proj        [6144, 64 * 256]            K over hidden, N (heads) over heads          [1536, 8 heads * 256]
  o_proj        [64 * 256, 6144]            K (heads) over heads, N over hidden          [8 heads * 256, 1536]
  ============  ==========================  ===========================================  =========================

  Chip (r, c) (r on the hidden axis, c on the head axis) owns heads ``8c + 2r`` and ``8c + 2r + 1`` for the
  per-head ops (wkv_b1, wkv_b2, the gate multiply). The two norms (2048 + 512) and the 64-value sink are replicated.

Collectives (the hidden contraction is a reduce over the hidden axis; every reduce accumulates in fp32 inside
``matmul_reduce_scatter_prefill``):

  q_a:   x @ Wqa -> reduce-scatter(hidden, N) -> all-gather(hidden) -> all-gather(heads) -> q_a_layernorm (full row)
  q_b:   q_lat[:, K_r] @ Wqb -> reduce-scatter(hidden, N): this chip's 2 heads, every token
         -> wkv_b1 (2 heads) -> all-to-all(hidden) + all-to-all(heads), heads -> sequence: [64 heads, S / 32]
         -> RoPE (ttMLA's indexed op) -> the unchanged sparse core
  kv_a:  x[:, K_rc] @ Wkva -> reduce-scatter(hidden, sequence) -> ttMLA's KV stem (TP gather + fp32 reduce)
         -> kv_a_layernorm, RoPE, KV cache write (ttMLA's SP x TP block-cyclic cache, SP on the hidden axis)
  core:  ttMLA's sparse path, unchanged: sparse_sdpa over all 64 heads per chip with the learnable sink
         (``attention_sink``), each query's whole top-k on one chip -> all-to-all back to [8 heads, S / 4]
  wkv_b2: all-to-all(hidden), sequence -> heads: [2 heads, S] -> wkv_b2 (2 heads)
  gate:  x @ Wg -> reduce-scatter(hidden, N) -> sigmoid. The reduce over the hidden axis completes BEFORE the
         sigmoid (sigmoid of a partial sum would be wrong); the scatter only picks this chip's 2 heads.
  o:     (v * g) -> all-gather(hidden) [8 heads, S] -> @ Wo -> reduce-scatter(heads, N) -> all-gather(heads)
         -> [S, hidden / 4], the input layout.

The attention core is ttMLA's: per chip, all 64 heads over S / 32 queries, with the selected KV rows gathered
(or pushed) to that chip, so each query's softmax is complete on one chip and the sink goes into ``sparse_sdpa``
exactly once. The context-parallel stripe-and-LSE-merge core of #65884 (each chip attends over its own KV stripe,
partials merged by LSE) is NOT used here; if it lands (#65886 / #65887) the sink must move to the merge,
``out = merged * sigmoid(lse - sink)`` (``tests/test_hy4_attention_math.py::test_lse_fold_is_the_same_function``).

Not here: the DSA indexer on this input layout (TtIndexer needs 128 / 8 = 16 index dims per chip, not tile
aligned, #65887); this module takes the top-k as ``indexer_indices``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCache
from models.demos.deepseek_v3_d_p.utils.weight_cache_dump import weight_cache_dump_mode

# The config switch: models that set it (Hy4-preview on the tp32 layout) build ttMLA2DTP; GLM-5.x, DeepSeek and
# Kimi never set it and keep ttMLA.
MLA_2D_TP_FLAG = "mla_2d_tp"


def uses_2d_tp(config) -> bool:
    return bool(getattr(config, MLA_2D_TP_FLAG, False))


def mla_class(config):
    """ttMLA2DTP when the config opts in to 2D TP attention, else ttMLA."""
    return ttMLA2DTP if uses_2d_tp(config) else ttMLA


class ttMLA2DTP(ttMLA):
    CACHE_PREFIX = "mla2d"  # its shards differ from ttMLA's TP-only ones; never share their cache files

    # ------------------------------------------------------------------------------------------- host math
    @staticmethod
    def heads_of_chip(num_heads: int, hp: int, tp: int, r: int, c: int) -> list[int]:
        """The heads chip (r on the hidden axis, c on the head axis) owns for the per-head ops."""
        per_col = num_heads // tp
        per_chip = per_col // hp
        return [c * per_col + r * per_chip + j for j in range(per_chip)]

    @staticmethod
    def check_dims(config, hp: int, tp: int) -> None:
        H, hidden, Q = config.num_attention_heads, config.hidden_size, config.q_lora_rank
        v, qk = config.v_head_dim, config.qk_nope_head_dim + config.qk_rope_head_dim
        T = ttnn.TILE_SIZE
        checks = {
            "heads over heads x hidden": H % (tp * hp) == 0,
            "hidden over hidden x heads, whole tiles": hidden % (hp * tp * T) == 0,
            "q_lora over hidden x heads, whole tiles": Q % (hp * tp * T) == 0,
            "q_b / gate heads per chip, whole tiles": (H // (tp * hp)) * qk % T == 0 and (H // (tp * hp)) * v % T == 0,
            "o_proj output over heads, whole tiles": hidden // hp % (tp * T) == 0,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"2D TP attention (hidden over {hp}, heads over {tp}) does not tile: {bad}")

    @staticmethod
    def host_weights_2d(state_dict: dict | None, config, hp: int, tp: int) -> dict:
        """Host tensors laid out for the 2D mesh mappers, as ``name -> (tensor, hidden_dim, head_dim)``: the tensor
        dim sharded over the hidden axis and the one sharded over the head axis (None = replicated over it).

        ``state_dict`` holds ttMLA's canonical weights (``q_a_proj.weight`` ... ``g_proj.weight``, ``attn_sink.weight``,
        with Hy4's RoPE frames already reordered by ``hy4_checkpoint.hy4_mla_weights``). None/empty -> placeholders
        of the right shapes (cache-only load). Pure torch, so the CPU tests check the same layout the device loads."""
        H, hidden = config.num_attention_heads, config.hidden_size
        Q, L = config.q_lora_rank, config.kv_lora_rank
        N, R, V = config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim
        use_gate = bool(getattr(config, "mla_use_output_gate", False))
        use_sink = bool(getattr(config, "mla_learnable_sink", False))
        real = bool(state_dict) and "q_a_proj.weight" in state_dict
        hpc = H // tp  # heads per head-axis column
        if real:
            sd = state_dict
            q_a = sd["q_a_proj.weight"].transpose(-2, -1)
            q_b = sd["q_b_proj.weight"].transpose(-2, -1)
            kv_a = sd["kv_a_proj_with_mqa.weight"].transpose(-2, -1)
            kv_b = sd["kv_b_proj.weight"].reshape(H, N + V, L)
            w_uk = kv_b[:, :N, :]  # [H, N, L]: q_nope @ w_uk -> latent
            w_uv = kv_b[:, N:, :].transpose(-2, -1)  # [H, L, V]: latent @ w_uv -> v
            o = sd["o_proj.weight"].transpose(-2, -1)
            g = sd["g_proj.weight"].transpose(-2, -1) if use_gate else None
            q_a_ln = sd["q_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            kv_a_ln = sd["kv_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
            sink = ttMLA.sink_for_sparse_sdpa(sd["attn_sink.weight"], ttMLA.softmax_scale(config)) if use_sink else None
        else:
            e = torch.empty
            q_a, q_b, kv_a = e(hidden, Q), e(Q, H * (N + R)), e(hidden, L + R)
            w_uk, w_uv, o = e(H, N, L), e(H, L, V), e(H * V, hidden)
            g = e(hidden, H * V) if use_gate else None
            q_a_ln = e(1, 1, Q // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            kv_a_ln = e(1, 1, L // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            sink = torch.empty(1, 1, 1, H, dtype=torch.bfloat16) if use_sink else None
        out = {
            "q_a_proj": (q_a, 0, 1),
            "q_b_proj": (q_b, 0, 1),
            # K over both axes: hidden index r * (hidden / hp) + c * (hidden / (hp * tp)) + k.
            "kv_a_proj_with_mqa": (kv_a.reshape(hp, tp, hidden // (hp * tp), L + R), 0, 1),
            # Head h = c * hpc + r * (hpc / hp) + j: dim 0 is c (head axis), dim 1 is r * (hpc / hp) + j (hidden axis).
            "wkv_b1": (w_uk.reshape(tp, hpc, N, L), 1, 0),
            "wkv_b2": (w_uv.reshape(tp, hpc, L, V), 1, 0),
            "o_proj": (o, 1, 0),
            "q_a_layernorm": (q_a_ln, None, None),
            "kv_a_layernorm": (kv_a_ln, None, None),
        }
        if use_gate:
            out["g_proj"] = (g, 0, 1)
        if use_sink:
            out["attn_sink"] = (sink, None, None)
        return out

    @staticmethod
    def chip_shard(t: torch.Tensor, hidden_dim, head_dim, hp: int, tp: int, r: int, c: int) -> torch.Tensor:
        """What ``ShardTensor2dMesh`` puts on chip (r, c) for a ``host_weights_2d`` entry (CPU tests)."""
        if hidden_dim is not None:
            t = t.chunk(hp, dim=hidden_dim)[r]
        if head_dim is not None:
            t = t.chunk(tp, dim=head_dim)[c]
        return t

    # ------------------------------------------------------------------------------------------- caches
    @staticmethod
    def check_cache_complete(
        cache_path: Path,
        cache_name_prefix: str,
        has_indexer: bool = False,
        has_output_gate: bool = False,
        has_sink: bool = False,
    ) -> bool:
        """``cache_name_prefix`` is ttMLA's ``.../layer_{i}.mla``; the 2D shards live under ``layer_{i}.mla2d``."""
        from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        prefix = str(cache_name_prefix)
        prefix_2d = prefix[: -len(".mla")] + "." + ttMLA2DTP.CACHE_PREFIX if prefix.endswith(".mla") else prefix
        for name in ttMLA.weight_names(has_output_gate, has_sink):
            if not pattern_exists(f"{prefix_2d}.{name}*.tensorbin", "MLA"):
                return False
        return not has_indexer or TtIndexer.check_cache_complete(cache_path, cache_name_prefix)

    @staticmethod
    def _convert_and_cache_weights(
        state_dict: dict | None,
        mesh_device: ttnn.MeshDevice,
        config,
        layer_idx: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        cache_path: Path | None = None,
        device: ttnn.MeshDevice | None = None,
        kv_only: bool = False,
        replicate_wkv_b1: bool = False,  # ttMLA's early-Q replica; the 2D layout shards wkv_b1 instead
        tp_shard_q_a_norm: bool = False,
    ) -> dict | None:
        assert not tp_shard_q_a_norm, "2D TP normalizes the full q_a row after its gathers"
        hp, tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
        ttMLA2DTP.check_dims(config, hp, tp)
        host = ttMLA2DTP.host_weights_2d(state_dict, config, hp, tp)
        mem = ttnn.DRAM_MEMORY_CONFIG if device else None
        kv_names = ("kv_a_proj_with_mqa", "kv_a_layernorm")
        result = {}
        for name, (t, hidden_dim, head_dim) in host.items():
            if kv_only and name not in kv_names:
                continue
            if hidden_dim is None and head_dim is None:
                mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            else:
                dims = [None, None]
                dims[sp_axis], dims[tp_axis] = hidden_dim, head_dim
                mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
            vector = name in ("q_a_layernorm", "kv_a_layernorm", "attn_sink")
            result[name] = ttnn.as_tensor(
                t,
                device=device,
                dtype=ttnn.bfloat16 if vector else ttnn.bfloat8_b,
                layout=ttnn.ROW_MAJOR_LAYOUT if vector else ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
                cache_file_name=(
                    str(cache_path / f"layer_{layer_idx}.{ttMLA2DTP.CACHE_PREFIX}.{name}") if cache_path else None
                ),
                cache_dump_mode=weight_cache_dump_mode(),
            )
        if device is None:
            return None
        return result

    @staticmethod
    def build_ttnn_cache(
        state_dict: dict,
        cache_path: Path,
        mesh_device: ttnn.MeshDevice,
        config,
        layer_idx: int,
        seq_len: int,
        sp_axis: int = 0,
        tp_axis: int = 1,
        kv_only: bool = False,
        has_indexer: bool | None = None,
        use_fused_rmsnorm: bool = False,
    ):
        """The 2D shards only: the DSA indexer has no 2D layout yet (#65887)."""
        ttMLA2DTP._convert_and_cache_weights(
            state_dict, mesh_device, config, layer_idx, sp_axis, tp_axis, cache_path, device=None, kv_only=kv_only
        )

    # ------------------------------------------------------------------------------------------- device
    def __init__(self, config, state_dict, mesh_device, *args, **kwargs):
        super().__init__(config, state_dict, mesh_device, *args, **kwargs)
        if not self._has_indexer:
            raise NotImplementedError("2D TP MLA runs the sparse (DSA) core only")
        if not isinstance(self._indexer, type(None)) and type(self._indexer).__name__ == "TtIndexer":
            raise NotImplementedError(
                f"layer {self.layer_idx}: the DSA indexer has no 2D-TP layout yet (#65887); build the layer as a "
                "shared layer and pass its top-k as indexer_indices"
            )
        # The K-over-both-axes shard loads as [1, 1, K, 576]; matmul_reduce_scatter_prefill takes a rank-2 weight.
        kv_a = self.kv_a_proj_with_mqa_weight
        self.kv_a_proj_with_mqa_weight = ttnn.reshape(kv_a, (kv_a.shape[-2], kv_a.shape[-1]))
        self.hp = self.sp_factor  # the hidden axis is ttMLA's SP axis
        self.heads_per_chip = self.num_heads // (self.tp_factor * self.hp)
        # All-gather outputs: persistent, shared by every layer through TT_CCL, and gathered with ttMLA's shared
        # high-bandwidth semaphores (plain ttnn.all_gather allocates its own in L1_SMALL, which the prefill's 1536 B
        # do not hold). Gather outputs alias these buffers, so the forward never deallocates them.
        S = self.active_seq_len
        heads_v = self.num_heads // self.tp_factor * self.v_head_dim
        self._gather_out = {
            name: self.tt_ccl.get_mla_high_bw_all_gather_buffer(
                name=f"mla2d_{name}", shape=[1, 1, S, width], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            for name, width in (
                ("q_a_col", self.q_lora_rank // self.tp_factor),
                ("q_a", self.q_lora_rank),
                ("o_in", heads_v),
                ("o_out", self.hidden_size // self.hp),
            )
        }
        ready, data_valid = self.tt_ccl.get_mla_high_bw_gather_semaphores()
        self._gather_semaphores = dict(ready_semaphore=ready, data_valid_semaphore=data_valid)
        logger.info(
            f"MLA layer {self.layer_idx}: 2D TP, hidden over axis {self.sp_axis} ({self.hp}), heads over axis "
            f"{self.tp_axis} ({self.tp_factor}), {self.heads_per_chip} heads per chip in the per-head ops"
        )

    def _axis_topology(self, axis: int):
        return self.sp_ccl_topology if axis == self.sp_axis else self.tp_ccl_topology

    def _mm_rs(self, x: ttnn.Tensor, w: ttnn.Tensor, axis: int, scatter_dim: int) -> ttnn.Tensor:
        """x @ w with the partials summed (fp32) over ``axis`` and scattered over ``scatter_dim``."""
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            x,
            w,
            cluster_axis=axis,
            scatter_dim=scatter_dim,
            topology=self._axis_topology(axis),
            num_links=self.ccl_num_links,
        )

    def _all_gather(self, x: ttnn.Tensor, axis: int, name: str) -> ttnn.Tensor:
        """All-gather the last dim over ``axis`` into the persistent buffer ``name`` (returned; not to be freed)."""
        return ttnn.experimental.high_bw_all_gather(
            x,
            dim=3,
            output_tensor=self._gather_out[name],
            cluster_axis=axis,
            num_links=self.ccl_num_links,
            **self._gather_semaphores,
        )

    def _all_to_all(self, x: ttnn.Tensor, in_dim: int, out_dim: int, axis: int) -> ttnn.Tensor:
        """Gather ``in_dim`` and split ``out_dim`` over ``axis``."""
        return ttnn.experimental.all_to_all_async_generic(
            x,
            in_dim=in_dim,
            out_dim=out_dim,
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=axis,
        )

    def _q_a_latent_2d(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """q_a_layernorm(x @ Wqa), the whole 2048-wide row on every chip: [1, 1, S, q_lora]."""
        part = self._mm_rs(x, self.q_a_proj_weight, self.sp_axis, -1)  # [S, q_lora / (hp * tp)], reduced
        col = self._all_gather(part, self.sp_axis, "q_a_col")  # [S, q_lora / tp]
        ttnn.deallocate(part)
        qr = self._all_gather(col, self.tp_axis, "q_a")  # [S, q_lora]
        return self._q_a_norm(qr, ttnn.DRAM_MEMORY_CONFIG)

    def _q_stem_2d(self, qr: ttnn.Tensor, rope_tensors, kv_actual_isl, seq_len_local: int, metadata) -> ttnn.Tensor:
        """q_b (K over hidden) -> this chip's heads -> wkv_b1 -> heads to sequence -> RoPE: [1, H, S / (hp * tp), 576]
        row-major, the q ttMLA's sparse core takes (as _q_stem_seq_sharded returns it)."""
        k_slice = ttnn.mesh_partition(qr, dim=3, cluster_axis=self.sp_axis)  # [S, q_lora / hp]
        ttnn.deallocate(qr)
        q = self._mm_rs(k_slice, self.q_b_proj_weight, self.sp_axis, -1)  # [S, heads_per_chip * qk_head_dim]
        ttnn.deallocate(k_slice)
        q_nope, q_rope = ttnn.experimental.nlp_create_q_heads_split(
            q,
            num_heads=self.heads_per_chip,
            split_head_dim=self.qk_nope_head_dim,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        chunk = q_nope.shape[2]
        q_lat = ttnn.linear(q_nope, self.wkv_b1_weight, **self._make_batched_mm_kwargs("wkv_b1", chunk))
        ttnn.deallocate(q_nope)
        moved = []
        for t in (q_lat, q_rope):
            rows = self._all_to_all(t, 1, 2, self.sp_axis)  # [heads / tp, S / hp]
            ttnn.deallocate(t)
            moved.append(self._all_to_all(rows, 1, 2, self.tp_axis))  # [heads, S / (hp * tp)]
            ttnn.deallocate(rows)
        prefix, rope_seq = moved
        tt_q = self._apply_rope_padded(
            rope_seq,
            rope_tensors,
            kv_actual_isl,
            metadata=metadata,
            concat_prefix=prefix,
            seq_subshard_axis=self.tp_axis,
            row_major_output=True,
        )
        ttnn.deallocate(prefix)
        ttnn.deallocate(rope_seq)
        return tt_q

    def _kv_a_proj(self, hidden_states: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """x[:, K_rc] @ Wkva reduced over the hidden axis and sequence-scattered over it: [1, 1, S / hp, 576], still
        partial over the head axis, which ttMLA's KV stem then gathers and reduces in fp32."""
        x_rc = ttnn.mesh_partition(hidden_states, dim=3, cluster_axis=self.tp_axis)
        kv = self._mm_rs(x_rc, self.kv_a_proj_with_mqa_weight, self.sp_axis, -2)
        ttnn.deallocate(x_rc)
        return kv

    def _apply_wkv_b2(self, t: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """[1, heads / tp, S / hp, 512] -> sequence to heads over the hidden axis -> wkv_b2 on this chip's heads:
        [1, heads_per_chip, S, v_head_dim]."""
        mine = self._all_to_all(t, 2, 1, self.sp_axis)
        # bf16 out (ttMLA's default is bf8): the gated v is all-gathered into a bf16 buffer before o_proj.
        kwargs = {**self._make_batched_mm_kwargs("wkv_b2", mine.shape[2]), "dtype": ttnn.bfloat16}
        v = ttnn.linear(mine, self.wkv_b2_weight, **kwargs)
        ttnn.deallocate(mine)
        return v

    def _output_gate(self, hidden_states: ttnn.Tensor, seq_len_local: int) -> ttnn.Tensor:
        """sigmoid(x @ Wg) for this chip's heads: the hidden-axis reduce finishes before the sigmoid."""
        g = self._mm_rs(hidden_states, self.g_proj_weight, self.sp_axis, -1)
        out = ttnn.sigmoid(g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(g)
        return out

    def _o_proj_epilogue(
        self, attn_out: ttnn.Tensor, seq_len_local: int, hidden_states: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        v = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [S, hpc_chip * V]
        if self._use_gate:
            g = self._output_gate(hidden_states, seq_len_local)
            gated = ttnn.multiply(v, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(g)
            ttnn.deallocate(v)
            v = gated
        col = self._all_gather(v, self.sp_axis, "o_in")  # [S, heads / tp * V]: this column's heads
        ttnn.deallocate(v)
        part = self._mm_rs(col, self.o_proj_weight, self.tp_axis, -1)  # [S, hidden / (hp * tp)]
        out = self._all_gather(part, self.tp_axis, "o_out")  # [S, hidden / hp]
        ttnn.deallocate(part)
        return ttnn.clone(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # the caller owns the output

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rope_tensors: dict,
        kvpe_cache: MlaKvCache,
        cache_layer_idx: int = 0,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        return_kv_intermediates: bool = False,
        index_kv_cache: Optional[ttnn.Tensor] = None,
        indexer_indices: Optional[ttnn.Tensor] = None,
        return_indexer_indices: bool = False,
        metadata: Optional[ttnn.Tensor] = None,
        force_kv_only: bool = False,
    ):
        """``hidden_states``: ``[1, 1, S, hidden / hp]`` (the whole chunk, hidden over the hidden axis, replicated
        over the head axis). Returns the same layout. ``indexer_indices``: this chunk's top-k as ttMLA takes it,
        ``[1, 1, S / (hp * tp), k]`` per chip."""
        if kvpe_cache.geometry != self.kv_cache_geometry:
            raise ValueError(f"MLA configured for KV geometry {self.kv_cache_geometry}, got {kvpe_cache.geometry}")
        assert kvpe_cache.format == self.sparse_kv_cache_format
        chunk = hidden_states.shape[2]
        assert chunk == self.active_seq_len, f"2D TP MLA was built for {self.active_seq_len}-token chunks, got {chunk}"
        seq_len_local = chunk // self.sp_factor
        kv_actual_isl = actual_start or 0
        ttnn.tracy_message("`TT_SIGNPOST: MLA_START`")

        if self.kv_only or force_kv_only:
            tt_kvpe, tt_kv_nope, _ = self._kv_stem(
                hidden_states, rope_tensors, kv_actual_isl, seq_len_local, False, kvpe_cache, metadata=metadata
            )
            if tt_kv_nope is not None:
                ttnn.deallocate(tt_kv_nope)
            self._update_kv_cache(
                kvpe_cache,
                tt_kvpe,
                cache_user_id=cache_user_id,
                cache_layer_idx=cache_layer_idx,
                kv_actual_isl=kv_actual_isl,
                actual_end=actual_end,
                metadata=metadata,
                tp_axis=self.tp_shard_kv_axis,
            )
            for value in tt_kvpe if isinstance(tt_kvpe, tuple) else (tt_kvpe,):
                ttnn.deallocate(value)
            return None

        if indexer_indices is None:
            raise NotImplementedError("2D TP MLA takes its top-k as indexer_indices (indexer layout: #65887)")
        if self.tt_ccl.sparse_kv_push_planner is not None and self.layer_idx == self._first_layer_idx:
            self.tt_ccl.sparse_kv_push_planner.invalidate()

        qr = self._q_a_latent_2d(hidden_states)
        tt_q = self._q_stem_2d(qr, rope_tensors, kv_actual_isl, seq_len_local, metadata)
        tt_kvpe, tt_kv_nope, kv_intermediates = self._kv_stem(
            hidden_states,
            rope_tensors,
            kv_actual_isl,
            seq_len_local,
            return_kv_intermediates,
            kvpe_cache,
            metadata=metadata,
        )
        attn_out = self._attention(
            tt_q=tt_q,
            tt_kvpe=tt_kvpe,
            tt_kv_nope=tt_kv_nope,
            indices=indexer_indices,
            kvpe_cache=kvpe_cache,
            cache_layer_idx=cache_layer_idx,
            cache_user_id=cache_user_id,
            seq_len_local=seq_len_local,
            kv_actual_isl=kv_actual_isl,
            actual_end=actual_end,
            metadata=metadata,
            injected=True,
        )
        out = self._o_proj_epilogue(attn_out, seq_len_local, hidden_states=hidden_states)
        ttnn.deallocate(attn_out)
        ttnn.tracy_message("`TT_SIGNPOST: MLA_END`")
        result = (out,)
        if return_kv_intermediates:
            result += (kv_intermediates,)
        if return_indexer_indices:
            result += (indexer_indices,)
        return result[0] if len(result) == 1 else result
