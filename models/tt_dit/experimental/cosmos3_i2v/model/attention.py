# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native Cosmos3 joint-attention with tensor + sequence parallelism.

Mirrors `Cosmos3AttnProcessor` from `reference/transformer_cosmos3.py` but
runs Q/K/V projections, per-head RMSNorm, RoPE, both attention pathways,
and the output projections on device with ttnn ops.

und (text) is causal self-attention; gen (video) is full attention keyed
against `[und_K; gen_K]` / `[und_V; gen_V]`. With SP, the gen pathway
uses ring-joint SDPA across the sp_axis with und passed as the joint
tensor (the only mechanism the kernel exposes for extending K/V beyond
the ring K/V). und stays local on every chip — it's 50–200× shorter
than gen at 720p, sharding it costs more in comm than it saves.

The joint-Q output of the ring call is discarded: it would compute
und→[und, gen] attention (non-causal, including gen tokens) which is
wrong for und. The local und SDPA above produces the real und output.

Joint inputs (und Q/K/V) are padded to a multiple of `q_chunk_size=128`
because the ring op requires joint L ≡ 0 (mod q_chunk_size). The K pad
rows are filled with a large negative value so `softmax(Q·K_pad) ≈ 0`;
V pad rows are zero (weight is ~0 anyway); Q pad rows are zero (output
is discarded). N_und is ≤ 512 in practice — pad cost is < 1% of the
gen workload.

Sharding:
  - TP: Q/K/V projections use `ColParallelLinear` on `tp_axis`.
    Per-head RMSNorm reduces on head_dim (not sharded). RoPE is per-chip.
    SDPA runs on `[B, H_local, N, head_dim]`. ttnn SDPA handles GQA.
    `to_out`/`to_add_out` are `RowParallelLinear` followed by TP
    all-gather so heads are replicated downstream.
  - SP (gen only, when sp_factor > 1): gen Q/K/V come in already
    sequence-sharded on `sp_axis` (the trunk wrapper scatters at trunk
    entry). Ring-joint SDPA fuses the ring K/V all-gather. gen output
    stays sequence-sharded — the trunk wrapper gathers at trunk exit so
    each decoder layer's MLP + RMSNorm + residual sees the sharded gen
    seq and only pays per-chip cost.

Input contract:
  - und_seq, cos_und, sin_und: replicated `[1, 1, N_und, *]` (TP and SP).
  - gen_seq, cos_gen, sin_gen at sp_factor=1: replicated `[1, 1, N_gen, *]`.
  - gen_seq, cos_gen, sin_gen at sp_factor>1: replicated on TP, sharded
    on sp_axis with shape `[1, 1, N_gen_padded / sp, *]`. The trunk
    wrapper pads N_gen to a multiple of `k_chunk_size * sp_factor` so
    the ring op's per-chip seq is tile-aligned and divisible by
    `k_chunk_size`.
  - logical_n_gen: unpadded N_gen, threaded through the trunk for the
    ring op's `logical_n` arg. Defaults to gen_seq.shape[-2] (correct
    when nothing was padded).
  - Outputs match the input layout per pathway (und replicated, gen
    matches its input sharding).

RoPE is HF half-split style (`x*cos + rotate_half(x)*sin`), fused via
`ttnn.experimental.rotary_embedding_hf`. The host still pre-computes
cos/sin via `Cosmos3VLTextRotaryEmbedding` — native interleaved 3D mRoPE
is item 2 in the Phase 2 plan.

When `parallel_config` is `None`, defaults to a degenerate (tp=1, sp=1)
config — preserves the single-device contract from the MVP.
"""

from __future__ import annotations

import os

import ttnn

from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.matmul import get_matmul_core_grid


def sp_ring_enabled() -> bool:
    """SP ring SDPA on by default: the native-cfg dual-submesh path builds each trunk with
    sequence_parallel factor=2, and the ring SDPA + scatter must match that sharding or the
    denoised latent is corrupt. Set TT_COSMOS3_ENABLE_SP_RING=0 to disable if the ring op
    trips power on a given board."""
    return os.environ.get("TT_COSMOS3_ENABLE_SP_RING", "1") not in ("", "0", "false", "False")


# Module-level latch so TT_COSMOS3_DUMP_ATTN_DIR captures the FIRST attention call
# of the process (layer 0, step 0, cond pass). Reset is not needed — process is short-lived.
_attn_dump_done = False


def _default_parallel_config() -> DiTParallelConfig:
    """Single-device fallback: tp=1, sp=1, cfg=1."""
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        tensor_parallel=ParallelFactor(1, 1),
        sequence_parallel=ParallelFactor(1, 0),
    )


def _gqa_interleave_broadcast(t, kv_repeat: int, n_kv_heads: int, sub_core_grids):
    """Emulate `repeat_interleave(t, kv_repeat, dim=1)` while staying within `sub_core_grids`.

    `ttnn.repeat_interleave` defaults to the full grid — a power-trip risk on BH
    Galaxy. `concat([t] * kv_repeat, dim=1)` matches repeat_interleave only when
    `n_kv_heads == 1`. For `n_kv_heads > 1` this slices per-head, replicates each
    slice `kv_repeat` times in order, then concats: `[h0,h0,...,h0, h1,h1,...,h1]`.
    """
    if kv_repeat == 1:
        return t
    if n_kv_heads == 1:
        return ttnn.concat([t] * kv_repeat, dim=1, sub_core_grids=sub_core_grids)
    B, _, N, D = t.shape
    slices = []
    per_head = []
    for h in range(n_kv_heads):
        head_h = ttnn.slice(t, [0, h, 0, 0], [B, h + 1, N, D])
        per_head.append(head_h)
        slices.extend([head_h] * kv_repeat)
    result = ttnn.concat(slices, dim=1, sub_core_grids=sub_core_grids)
    for head_t in per_head:
        ttnn.deallocate(head_t)
    return result


class Cosmos3JointAttention(Module):
    """Dual-pathway joint attention for the Cosmos3 MoT trunk."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig | None = None,
        ccl_manager=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if num_attention_heads % num_key_value_heads != 0:
            msg = (
                f"num_attention_heads ({num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({num_key_value_heads})"
            )
            raise ValueError(msg)

        if parallel_config is None:
            parallel_config = _default_parallel_config()

        tp_factor = parallel_config.tensor_parallel.factor
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        sp_factor = parallel_config.sequence_parallel.factor
        sp_axis = parallel_config.sequence_parallel.mesh_axis

        if num_attention_heads % tp_factor != 0:
            msg = f"num_attention_heads ({num_attention_heads}) must be divisible by tp_factor ({tp_factor})"
            raise ValueError(msg)
        if num_key_value_heads % tp_factor != 0:
            msg = (
                f"num_key_value_heads ({num_key_value_heads}) must be divisible by tp_factor ({tp_factor}); "
                "TP can't split a single KV head"
            )
            raise ValueError(msg)
        if (tp_factor > 1 or sp_factor > 1) and ccl_manager is None:
            msg = "ccl_manager is required when tp_factor > 1 or sp_factor > 1"
            raise ValueError(msg)

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_local_heads = num_attention_heads // tp_factor
        self.n_local_kv_heads = num_key_value_heads // tp_factor
        self.sp_factor = sp_factor
        self.sp_axis = sp_axis
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        _use_fsdp = os.environ.get("TT_COSMOS3_FSDP_ON_SP") in ("1", "true", "True") and sp_factor > 1
        fsdp_axis = sp_axis if _use_fsdp else None

        col_kw = {
            "bias": attention_bias,
            "mesh_device": mesh_device,
            "mesh_axis": tp_axis,
            "fsdp_mesh_axis": fsdp_axis,
            "ccl_manager": ccl_manager,
            "dtype": dtype,
        }
        row_kw = {
            "bias": attention_bias,
            "mesh_device": mesh_device,
            "mesh_axis": tp_axis,
            "fsdp_mesh_axis": fsdp_axis,
            "ccl_manager": ccl_manager,
            "dtype": dtype,
        }
        norm_kw = {
            "norm_eps": rms_norm_eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "dtype": dtype,
        }

        # Understanding (causal) pathway
        self.to_q = ColParallelLinear(hidden_size, num_attention_heads * head_dim, **col_kw)
        self.to_k = ColParallelLinear(hidden_size, num_key_value_heads * head_dim, **col_kw)
        self.to_v = ColParallelLinear(hidden_size, num_key_value_heads * head_dim, **col_kw)
        self.to_out = RowParallelLinear(num_attention_heads * head_dim, hidden_size, **row_kw)
        self.norm_q = RMSNorm(head_dim, **norm_kw)
        self.norm_k = RMSNorm(head_dim, **norm_kw)

        # Generation (full) pathway
        self.add_q_proj = ColParallelLinear(hidden_size, num_attention_heads * head_dim, **col_kw)
        self.add_k_proj = ColParallelLinear(hidden_size, num_key_value_heads * head_dim, **col_kw)
        self.add_v_proj = ColParallelLinear(hidden_size, num_key_value_heads * head_dim, **col_kw)
        self.to_add_out = RowParallelLinear(num_attention_heads * head_dim, hidden_size, **row_kw)
        self.norm_added_q = RMSNorm(head_dim, **norm_kw)
        self.norm_added_k = RMSNorm(head_dim, **norm_kw)

        # SDPA precision: HiFi4 + fp32 accumulator. At Cosmos3-scale (hidden=5120, 64 layers,
        # num_attention_heads=64, head_dim=128), HiFi2 + bf16-acc produced 100% NaN end-to-end
        # on Galaxy even though small-config PCC tests passed at HiFi2. The softmax inside
        # ttnn.transformer.scaled_dot_product_attention is the most likely overflow point at
        # scale — exp() on slightly-out-of-range logits saturates fast in bf16. fp32 destination
        # accumulation gives the softmax enough headroom; HiFi4 matches what diffusers uses
        # downstream for the attention math.
        # The NaN was attributed to bf16 accumulation, not the math fidelity. fp32_dest_acc
        # is kept unconditionally; TT_COSMOS3_SDPA_HIFI2=1 drops the QK/PV matmul fidelity to
        # HiFi2 (~half the math cycles) to test whether HiFi4 is actually required for parity.
        import os as _os_sdpa

        _sdpa_fidelity = (
            ttnn.MathFidelity.HiFi2
            if _os_sdpa.environ.get("TT_COSMOS3_SDPA_HIFI2") in ("1", "true", "True")
            else ttnn.MathFidelity.HiFi4
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=_sdpa_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # Clamp SDPA's compute grid to 11x10 on Blackhole Galaxy. The raw
        # compute_with_storage_grid_size() returns 13x10 on BH P150, which can
        # cause a group of chips to overdraw power from the PDU and bring down
        # a whole tray (8 chips). SDPA is matmul-heavy internally — same risk
        # class as the trunk's linears, which already use get_matmul_core_grid.
        grid = get_matmul_core_grid(mesh_device)
        self.sdpa_q_chunk_size = 128
        self.sdpa_k_chunk_size = 128
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=self.sdpa_q_chunk_size,
            k_chunk_size=self.sdpa_k_chunk_size,
            exp_approx_mode=False,
        )
        # Ring-joint SDPA needs cores reserved for the CCL fabric workers.
        # Mirror the Flux/Motif pattern (blocks/attention.py:70-73, 301): reserve the
        # last row of the clamped SDPA grid for CCL workers, set ccl_core_grid_offset
        # to (0, sdpa_worker_grid.y). Power clamp is preserved.
        if self.sp_factor > 1:
            self.ring_sdpa_worker_grid = ttnn.CoreCoord(grid.x, max(grid.y - 1, 1))
            self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.ring_sdpa_worker_grid,
                q_chunk_size=self.sdpa_q_chunk_size,
                k_chunk_size=self.sdpa_k_chunk_size,
                exp_approx_mode=False,
            )
        else:
            self.ring_sdpa_worker_grid = None
            self.ring_sdpa_program_config = None

        # Power-safe sub_core_grids for ttnn.pad / ttnn.concat in the SP path. Without
        # this, those ops default to the full 13x10 BH P150 grid and can trip the PDU
        # when stacked on top of the ring SDPA workers. Reuses the same 11x10 clamp the
        # matmul path already enforces.
        self.safe_core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )

        self._k_pad_mask_cache: dict[tuple[int, int, int], tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def _k_pad_mask(self, padded_n_gen: int, logical_n_gen: int, head_dim: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (padded_n_gen, logical_n_gen, head_dim)
        cached = self._k_pad_mask_cache.get(key)
        if cached is not None:
            return cached
        import torch as _torch

        mask_host = _torch.ones(1, 1, padded_n_gen, head_dim, dtype=_torch.bfloat16)
        mask_host[:, :, logical_n_gen:, :] = 0
        neg_pad_host = _torch.zeros(1, 1, padded_n_gen, head_dim, dtype=_torch.bfloat16)
        neg_pad_host[:, :, logical_n_gen:, :] = -1.0e4
        shard_dims = (2, None) if self.sp_axis == 0 else (None, 2)
        mesh_shape_tup = tuple(self.mesh_device.shape)

        def _shard_mask(t: "_torch.Tensor") -> ttnn.Tensor:
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=mesh_shape_tup, dims=shard_dims),
            )

        result = (_shard_mask(mask_host), _shard_mask(neg_pad_host))
        self._k_pad_mask_cache[key] = result
        return result

    def _tp_factor(self) -> int:
        return self.parallel_config.tensor_parallel.factor

    def _tp_axis(self) -> int:
        return self.parallel_config.tensor_parallel.mesh_axis

    def _split_heads(self, x_11NF: ttnn.Tensor, num_heads_total: int) -> ttnn.Tensor:
        """Reshape TP-sharded projection output to per-head layout.

        Input:  [1, 1, N, num_heads_total * head_dim / tp]  (output of ColParallelLinear)
        Output: [1, num_heads_total / tp, N, head_dim]      (per-chip local heads)
        """
        local_heads = num_heads_total // self._tp_factor()
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            x_11NF,
            num_heads=local_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
        )
        return out

    def _apply_rope_half_split(
        self,
        x_BHNE: ttnn.Tensor,
        cos_11NE: ttnn.Tensor,
        sin_11NE: ttnn.Tensor,
    ) -> ttnn.Tensor:
        return ttnn.experimental.rotary_embedding_hf(x_BHNE, cos_11NE, sin_11NE, is_decode_mode=False)

    def _dump_attn_stage(self, t: ttnn.Tensor, dump_dir: str, name: str) -> None:
        """Gather an sp-sharded-or-replicated tensor (seq dim=2) and save as torch .pt."""
        import torch as _torch

        mesh_shape = tuple(self.mesh_device.shape)
        devs = ttnn.get_device_tensors(t)
        if self.sp_factor > 1 and sp_ring_enabled():
            tp_factor = mesh_shape[1 - self.sp_axis]
            if self.sp_axis == 0:
                slices = [ttnn.to_torch(devs[i * tp_factor]) for i in range(self.sp_factor)]
            else:
                slices = [ttnn.to_torch(devs[i]) for i in range(self.sp_factor)]
            full = _torch.cat(slices, dim=2)
        else:
            full = ttnn.to_torch(devs[0])
        _torch.save(full.detach().cpu(), f"{dump_dir}/{name}.pt")

    def _pathway(
        self,
        x_11NH: ttnn.Tensor,
        cos_11NE: ttnn.Tensor,
        sin_11NE: ttnn.Tensor,
        *,
        proj_q: ColParallelLinear,
        proj_k: ColParallelLinear,
        proj_v: ColParallelLinear,
        norm_q: RMSNorm,
        norm_k: RMSNorm,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Project + norm + RoPE for one pathway. Returns (q, k, v) in [B, H_local, N, head_dim]."""
        # ColParallelLinear with parallel_config=None: each chip uses replicated input + its sharded
        # weight slice → output is per-chip TP-fractured on out_features.
        q_11NF = proj_q(x_11NH)
        k_11NF = proj_k(x_11NH)
        v_11NF = proj_v(x_11NH)
        if getattr(self, "_capture_pathway", False) and proj_q is self.to_q:
            # Fine-grained bisect: capture und proj outputs BEFORE split_heads/norm/RoPE.
            self._cap_x_und_in = ttnn.to_torch(ttnn.get_device_tensors(x_11NH)[0])
            self._cap_proj_q_und = ttnn.to_torch(ttnn.get_device_tensors(q_11NF)[0])
            self._cap_proj_k_und = ttnn.to_torch(ttnn.get_device_tensors(k_11NF)[0])
            self._cap_proj_v_und = ttnn.to_torch(ttnn.get_device_tensors(v_11NF)[0])
            # Capture the und Linear weights themselves (device-0 slice).
            self._cap_to_q_weight = ttnn.to_torch(ttnn.get_device_tensors(self.to_q.weight.data)[0])
            self._cap_to_k_weight = ttnn.to_torch(ttnn.get_device_tensors(self.to_k.weight.data)[0])
            self._cap_to_v_weight = ttnn.to_torch(ttnn.get_device_tensors(self.to_v.weight.data)[0])

        q = self._split_heads(q_11NF, self.num_attention_heads)
        ttnn.deallocate(q_11NF)
        k = self._split_heads(k_11NF, self.num_key_value_heads)
        ttnn.deallocate(k_11NF)
        v = self._split_heads(v_11NF, self.num_key_value_heads)
        ttnn.deallocate(v_11NF)

        q = norm_q(q)
        k = norm_k(k)

        q = self._apply_rope_half_split(q, cos_11NE, sin_11NE)
        k = self._apply_rope_half_split(k, cos_11NE, sin_11NE)
        return q, k, v

    def _project_out(
        self,
        attn_BHNE: ttnn.Tensor,
        proj: RowParallelLinear,
    ) -> ttnn.Tensor:
        """Concat heads → row-parallel matmul → all-gather → replicated `[1, 1, N, hidden_size]`."""
        heads = ttnn.transformer.concatenate_heads(attn_BHNE)
        ttnn.deallocate(attn_BHNE)
        heads_11NF = ttnn.unsqueeze(heads, 0)
        ttnn.deallocate(heads)

        # RowParallelLinear: input is col-fractured (local heads × head_dim per chip), produces
        # partial-sum matmul → reduce-scatter on TP axis → output [1, 1, N, hidden_size / tp].
        out_fractured = proj(heads_11NF)
        ttnn.deallocate(heads_11NF)

        if self._tp_factor() <= 1:
            return out_fractured

        # All-gather on TP axis to replicate so the tt-symbiote-wrapped caller (host PyTorch
        # residual + RMSNorm + MLP) sees the same replicated layout it gave us as input.
        # NOTE: `out_fractured` is a persistent ping-pong buffer owned by ccl_manager — do NOT
        # `ttnn.deallocate` it. Freeing it corrupts the cache and a later `reduce_scatter` for
        # the same shape (e.g. the MLP's `down_proj` in the same decoder-layer forward) crashes
        # with "Tensor is not allocated". The buffer is managed by the ping-pong cache.
        out_replicated = self.ccl_manager.all_gather_persistent_buffer(out_fractured, dim=3, mesh_axis=self._tp_axis())
        return out_replicated

    def forward(
        self,
        und_seq_11Nh: ttnn.Tensor,
        gen_seq_11Mh: ttnn.Tensor,
        cos_und_11NE: ttnn.Tensor,
        sin_und_11NE: ttnn.Tensor,
        cos_gen_11ME: ttnn.Tensor,
        sin_gen_11ME: ttnn.Tensor,
        logical_n_gen: int | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the dual-pathway joint attention.

        Args:
            und_seq_11Nh: Understanding hidden states, replicated `[1, 1, N_und, hidden_size]`.
            gen_seq_11Mh: Generation hidden states.
                At sp_factor=1: replicated `[1, 1, N_gen, hidden_size]`.
                At sp_factor>1: sp-sharded `[1, 1, N_gen_padded/sp, hidden_size]`.
            cos_und/sin_und: Pre-computed rotary, replicated `[1, 1, N_und, head_dim]`.
            cos_gen/sin_gen: Pre-computed rotary matching gen_seq's layout.
            logical_n_gen: Unpadded N_gen. Required when sp_factor>1; ignored otherwise.

        Returns:
            (und_out, gen_out). und is replicated. At sp>1, gen stays sp-sharded.
        """
        global _attn_dump_done
        _dump_dir = os.environ.get("TT_COSMOS3_DUMP_ATTN_DIR")
        _do_dump = _dump_dir is not None and not _attn_dump_done
        if _do_dump:
            os.makedirs(_dump_dir, exist_ok=True)
            _attn_dump_done = True

        q_und, k_und, v_und = self._pathway(
            und_seq_11Nh,
            cos_und_11NE,
            sin_und_11NE,
            proj_q=self.to_q,
            proj_k=self.to_k,
            proj_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
        )
        if getattr(self, "_capture", False):
            # SP-vs-no-SP bisect probe: stash intermediates host-side.
            self._cap_q_und = ttnn.to_torch(ttnn.get_device_tensors(q_und)[0])
            self._cap_k_und = ttnn.to_torch(ttnn.get_device_tensors(k_und)[0])
            self._cap_v_und = ttnn.to_torch(ttnn.get_device_tensors(v_und)[0])
        q_gen, k_gen, v_gen = self._pathway(
            gen_seq_11Mh,
            cos_gen_11ME,
            sin_gen_11ME,
            proj_q=self.add_q_proj,
            proj_k=self.add_k_proj,
            proj_v=self.add_v_proj,
            norm_q=self.norm_added_q,
            norm_k=self.norm_added_k,
        )
        if _do_dump:
            self._dump_attn_stage(q_gen, _dump_dir, "01_q_gen_post_pathway")
            self._dump_attn_stage(k_gen, _dump_dir, "02_k_gen_post_pathway")
            self._dump_attn_stage(v_gen, _dump_dir, "03_v_gen_post_pathway")

        # Understanding pathway: causal self-attention on local heads only.
        und_attn_BHNE = ttnn.transformer.scaled_dot_product_attention(
            q_und,
            k_und,
            v_und,
            is_causal=True,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        if getattr(self, "_capture", False):
            self._cap_und_attn = ttnn.to_torch(ttnn.get_device_tensors(und_attn_BHNE)[0])

        if self.sp_factor <= 1 or not sp_ring_enabled():
            ttnn.deallocate(q_und)
            # Generation pathway: full attention, K/V = concat(und, gen) on seq dim.
            # Each chip concatenates ITS local-heads slice — no cross-chip comms needed.
            k_full = ttnn.concat([k_und, k_gen], dim=2)
            v_full = ttnn.concat([v_und, v_gen], dim=2)
            ttnn.deallocate(k_und)
            ttnn.deallocate(v_und)
            ttnn.deallocate(k_gen)
            ttnn.deallocate(v_gen)

            gen_attn_BHME = ttnn.transformer.scaled_dot_product_attention(
                q_gen,
                k_full,
                v_full,
                is_causal=False,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
            ttnn.deallocate(q_gen)
            ttnn.deallocate(k_full)
            ttnn.deallocate(v_full)
        else:
            if logical_n_gen is None:
                msg = "logical_n_gen is required when sp_factor > 1"
                raise ValueError(msg)
            # Ring SDPA requires Q.num_heads == K.num_heads (no internal GQA broadcast,
            # unlike the local SDPA op). Cosmos3 is GQA (64 Q-heads / 8 KV-heads), so
            # broadcast K/V from n_local_kv_heads to n_local_heads via a per-head
            # interleave that stays within `safe_core_range_set` (see `_gqa_interleave_broadcast`).
            kv_repeat = self.n_local_heads // self.n_local_kv_heads
            if kv_repeat > 1:
                crs = self.safe_core_range_set
                k_gen_b = _gqa_interleave_broadcast(k_gen, kv_repeat, self.n_local_kv_heads, crs)
                v_gen_b = _gqa_interleave_broadcast(v_gen, kv_repeat, self.n_local_kv_heads, crs)
                ttnn.deallocate(k_gen)
                ttnn.deallocate(v_gen)
            else:
                k_gen_b, v_gen_b = k_gen, v_gen
            if _do_dump:
                self._dump_attn_stage(k_gen_b, _dump_dir, "04_k_gen_post_broadcast")
                self._dump_attn_stage(v_gen_b, _dump_dir, "05_v_gen_post_broadcast")

            # Defensive: the ring kernel already skips gen K chunks past logical_n and
            # masks the partial boundary chunk via global_n_padded_tiles. The explicit
            # K-pad here is redundant but cheap; kept as a backstop in case a future
            # kernel revision changes the chunk-skip behavior.
            padded_n_gen = k_gen_b.shape[2] * self.sp_factor
            if logical_n_gen < padded_n_gen:
                head_dim = k_gen_b.shape[3]
                mask_tt, neg_pad_tt = self._k_pad_mask(padded_n_gen, logical_n_gen, head_dim)
                k_real_tt = ttnn.multiply(k_gen_b, mask_tt)
                k_masked = ttnn.add(k_real_tt, neg_pad_tt)
                ttnn.deallocate(k_real_tt)
                ttnn.deallocate(k_gen_b)
                k_gen_b = k_masked
            if _do_dump:
                self._dump_attn_stage(k_gen_b, _dump_dir, "06_k_gen_post_mask")
            # Pass und at its logical seq length (ttnn TILE-pads physically). Pre-padding
            # to k_chunk_size hides logical L from the ring kernel — joint_has_padding
            # becomes false and the joint K-pad rows go through softmax unmasked. With
            # L kept at logical, the kernel emits joint_l_partial_col + joint_n_padded_tiles
            # masks for the last joint chunk. Confirmed via test_ring_sdpa_unpadded_und_at_cosmos3_shape:
            # PCC 0.9997 vs 0.700 with the pre-pad.
            q_und_pad = q_und
            if kv_repeat > 1:
                crs = self.safe_core_range_set
                k_und_pad_b = _gqa_interleave_broadcast(k_und, kv_repeat, self.n_local_kv_heads, crs)
                v_und_pad_b = _gqa_interleave_broadcast(v_und, kv_repeat, self.n_local_kv_heads, crs)
                ttnn.deallocate(k_und)
                ttnn.deallocate(v_und)
            else:
                k_und_pad_b, v_und_pad_b = k_und, v_und

            gen_attn_BHME, und_attn_via_ring, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_gen,
                k_gen_b,
                v_gen_b,
                q_und_pad,
                k_und_pad_b,
                v_und_pad_b,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(k_gen_b.shape, 2, self.sp_axis),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(v_gen_b.shape, 2, self.sp_axis),
                joint_strategy="rear",
                logical_n=logical_n_gen,
                program_config=self.ring_sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.sp_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.sp_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(0, self.ring_sdpa_worker_grid.y),
            )
            if _do_dump:
                self._dump_attn_stage(gen_attn_BHME, _dump_dir, "07_gen_attn_post_sdpa")
            ttnn.deallocate(q_gen)
            ttnn.deallocate(k_gen_b)
            ttnn.deallocate(v_gen_b)
            ttnn.deallocate(q_und_pad)
            ttnn.deallocate(k_und_pad_b)
            ttnn.deallocate(v_und_pad_b)
            # Joint output via the ring would be und→[gen, und] attention (non-causal,
            # including gen tokens) — wrong for und. The real und output comes from
            # the local causal SDPA above. Discard the joint output.
            ttnn.deallocate(und_attn_via_ring)

        if _do_dump and (self.sp_factor <= 1 or not sp_ring_enabled()):
            self._dump_attn_stage(gen_attn_BHME, _dump_dir, "07_gen_attn_post_sdpa")
        und_out = self._project_out(und_attn_BHNE, self.to_out)
        gen_out = self._project_out(gen_attn_BHME, self.to_add_out)
        if _do_dump:
            self._dump_attn_stage(gen_out, _dump_dir, "08_gen_out_post_project")
        return und_out, gen_out
