# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Full (causal, no sink) attention of MiMo on the [8, 4] mesh: SP over axis 0, heads over TP axis 1.

Per chip, one chunk of 5120 tokens:
    x [1, 1, 160, 4096] local rows -> all_gather over TP -> [640, 4096] (the SP row's tokens)
    -> fused qkv linear with the TP rank's own checkpoint slab -> [640, 16 x 192 | 192 | 192]
       (the softmax scale 192^-0.5 is folded into the q rows, attention_value_scale into the v rows, and v is
       zero-padded 128 -> 192 so nlp_create_qkv_heads splits one projection; v is sliced back to 128 after)
    -> RoPE (HF rotate_half) on dims 0-63 of q and k, cos / sin rows looked up at each local row's absolute position
    -> k, v written into the served caches (update_padded_kv_cache over SP, pad rows zeroed)
    -> ring_joint_sdpa causal over SP, reading the whole prefix from the caches (V head dim 128)
    -> concat heads [640, 2048] -> o_proj row-parallel (this rank's 2048 input columns) -> reduce_scatter over TP
    -> [1, 1, 160, 4096]

The chunk's slot, prefix length and end live in device scalars, and the positions in a device tensor; ``stage``
(the runtime's per-chunk metadata step, outside the forward) refreshes them, so the forward has no host work.
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl
from .kv_cache import LayerKV, chunk_positions
from .settings import settings


def _compute_config(mesh, fidelity: str, fp32: bool, packer_l1_acc: bool = False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=packer_l1_acc,
    )


def _scalar(mesh, value: int = 0):
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class ChunkMeta:
    """Device-resident per-chunk metadata shared by the layers: slot, prefix length (kv_actual), end of the real
    tokens, and every local row's absolute position ([1, W] uint32 per SP row, replicated over TP)."""

    def __init__(self, mesh, chunk: int):
        self.mesh, self.chunk = mesh, chunk
        self.sp = tuple(mesh.shape)[0]
        self.slot = _scalar(mesh)
        self.kv_actual = _scalar(mesh)
        self.valid_end = _scalar(mesh, chunk)
        self.positions = ttnn.from_torch(
            self._host_positions(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
        )

    def _host_positions(self, start: int) -> torch.Tensor:
        return chunk_positions(start, self.sp, self.chunk)  # [sp, W]: SP row r -> its chip's [1, W]

    def stage(self, slot: int, start: int, end: int | None = None) -> None:
        """Host -> device copies of the chunk's metadata into the existing buffers (trace-safe addresses)."""
        end = start + self.chunk if end is None else end
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for t, v in ((self.slot, slot), (self.kv_actual, start), (self.valid_end, end)):
            h = ttnn.from_torch(
                torch.tensor([v], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(h, t)
        h = ttnn.from_torch(
            self._host_positions(start),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
        )
        ttnn.copy_host_to_device_tensor(h, self.positions)


class RopeTables:
    """cos / sin [max_seq, rope_dim] bf16 ROW_MAJOR, replicated; HF layout (the half-split frequencies repeated)."""

    def __init__(self, mesh, inv_freq: torch.Tensor, max_seq: int):
        pos = torch.arange(max_seq, dtype=torch.float32)
        emb = torch.outer(pos, inv_freq.float())
        emb = torch.cat((emb, emb), dim=-1)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.cos, self.sin = (
            ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            for t in (emb.cos(), emb.sin())
        )

    def lookup(self, positions):
        """positions [1, W] uint32 -> cos, sin [1, 1, W, rope_dim] bf16 TILE."""
        out = []
        for table in (self.cos, self.sin):
            v = ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(ttnn.unsqueeze_to_4D(v))
        return out


def rank_weights(ck, layer_idx: int, cfg: dict, tp: int):
    """fp32 per-TP-rank weights in the checkpoint's [out, in] orientation (the linears run with transpose_b, so the
    bfloat8_b blocks of 16 run along K as in the precision model; [in, out] blocks along the output and costs single
    output columns up to 9x their error): qkv [tp, lh D + lkv D + lkv D, H] (q rows x D^-0.5, v rows x
    attention_value_scale and zero-padded 128 -> D so nlp_create_qkv_heads splits it) and o_proj [tp, H, lh Dv]
    (rank r's input columns). The checkpoint's own rank slabs: rank r holds q heads r*lh.., kv heads r*lkv.."""
    heads, kv_heads, D, Dv = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["v_head_dim"]
    lh, lkv = heads // tp, kv_heads // tp
    p = f"model.layers.{layer_idx}.self_attn"
    wq, wk, wv = ck.qkv(f"{p}.qkv_proj", heads * D, kv_heads * D, kv_heads * Dv)
    wq = wq.float() * D**-0.5
    wv = wv.float() * (cfg.get("attention_value_scale") or 1.0)
    slabs = []
    for r in range(tp):
        q_r = wq[r * lh * D : (r + 1) * lh * D]
        k_r = wk[r * lkv * D : (r + 1) * lkv * D].float()
        v_r = wv[r * lkv * Dv : (r + 1) * lkv * Dv].view(lkv, Dv, -1)
        v_r = torch.nn.functional.pad(v_r, (0, 0, 0, D - Dv)).reshape(lkv * D, -1)
        slabs.append(torch.cat((q_r, k_r, v_r)).contiguous())
    wo = ck.get(f"{p}.o_proj.weight").float()  # [H, heads * Dv]
    return torch.stack(slabs), wo.view(-1, tp, lh * Dv).permute(1, 0, 2).contiguous()


class FullAttention:
    """One full-attention layer (MiMo layers 0, 5, ...). ``layer`` is the reference Layer config (heads, dims,
    rope); weights come from the checkpoint in fp32 and are rounded once to bfloat8_b on the device."""

    def __init__(
        self,
        mesh,
        ck,
        layer_idx: int,
        cfg: dict,
        *,
        max_seq: int,
        chunk: int,
        slots: int,
        ring: ccl.RingResources,
        meta: ChunkMeta,
        rope: RopeTables,
    ):
        self.mesh, self.ring, self.meta, self.rope = mesh, ring, meta, rope
        self.sp, self.tp = tuple(mesh.shape)
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.v_head_dim = cfg["v_head_dim"]
        self.rope_dim = int(self.head_dim * cfg["partial_rotary_factor"])
        assert self.heads % self.tp == 0 and self.kv_heads % self.tp == 0
        self.lh, self.lkv = self.heads // self.tp, self.kv_heads // self.tp
        self.max_seq, self.chunk = max_seq, chunk
        self.local_seq = max_seq // self.sp

        # the softmax scale is folded into the q rows in fp32 and SDPA runs at scale 1 (the kernel may truncate a
        # non-power-of-two scale to bf16: 192^-0.5 -0.54 %, known issue "SDPA attention sink ... truncated scale")
        self.scale = 1.0
        wqkv, wo = rank_weights(ck, layer_idx, cfg, self.tp)
        self.wqkv = self._sharded_weight(wqkv.unsqueeze(0))  # [1, tp, N, H] -> per chip [1, 1, N, H]
        self.wo = self._sharded_weight(wo.unsqueeze(0))

        self.kv = LayerKV(mesh, slots, self.lkv, self.local_seq, self.head_dim, self.v_head_dim)
        self.mm_cfg = _compute_config(
            mesh, settings.get("ATTN_FIDELITY"), settings.get("ATTN_FP32_ACC"), packer_l1_acc=True
        )
        self.rope_cfg = _compute_config(mesh, "HiFi4", True)
        # fp32 dest off: the kv_actual_isl_tensor path runs only the streaming SDPA compute (TT_FATAL otherwise)
        self.sdpa_cfg = _compute_config(mesh, settings.get("SDPA_FIDELITY"), False)
        self.sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ring.sdpa_grid,
            q_chunk_size=settings.get("SDPA_Q_CHUNK"),
            k_chunk_size=settings.get("SDPA_K_CHUNK"),
            exp_approx_mode=False,
            segmented_accumulation=settings.get("SDPA_SEGMENTED"),
        )

    def _sharded_weight(self, w: torch.Tensor):
        return ttnn.from_torch(
            w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 1)),
        )

    def load_prefix(self, slot: int, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Harness boundary (never inside a forward): the global prefix key [KV, >= length, D] / value [KV, ..,
        Dv] in the served block-cyclic layout of ``slot``; every other row of the slot is zero."""
        w = self.chunk // self.sp
        g = torch.arange(length)
        row, local = (g // w) % self.sp, (g // self.chunk) * w + g % w
        for name, src in (("key", key), ("value", value)):
            cache = getattr(self.kv, name)
            width = src.shape[-1]
            host = torch.zeros(self.sp, self.tp, self.kv.slots, self.lkv, self.local_seq, width)
            heads = src[:, :length].float().view(self.tp, self.lkv, length, width)
            for r in range(self.sp):
                m = row == r
                host[r][:, slot][:, :, local[m]] = heads[:, :, m]
            host = host.permute(0, 2, 1, 3, 4, 5).reshape(
                self.sp * self.kv.slots, self.tp * self.lkv, self.local_seq, width
            )
            new = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=cache.memory_config(),
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, 1)),
            )
            new.update_tensor_topology(cache.tensor_topology())
            cache.deallocate(True)
            setattr(self.kv, name, new)

    def __call__(self, x):
        """x [1, 1, 160, H] bf16 (this chip's residual rows) -> [1, 1, 160, H] bf16."""
        D, Dv, R = self.head_dim, self.v_head_dim, self.rope_dim
        xg = ccl.gather_rows(x)
        qkv = ttnn.linear(
            xg,
            self.wqkv,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_cfg,
        )
        xg.deallocate(True)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.lh,
            num_kv_heads=self.lkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)

        cos, sin = self.rope.lookup(self.meta.positions)
        q = self._rope(q, cos, sin, R)
        k = self._rope(k, cos, sin, R)
        cos.deallocate(True)
        sin.deallocate(True)
        v_full = v
        v = ttnn.slice(v_full, (0, 0, 0, 0), (1, self.lkv, v_full.shape[2], Dv), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_full.deallocate(True)

        for cache, t in ((self.kv.key, k), (self.kv.value, v)):
            t8 = ttnn.typecast(t, ttnn.bfloat8_b)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                t8,
                self.meta.slot,
                self.meta.kv_actual,
                layer_idx=0,
                num_layers=1,
                cluster_axis=ccl.SP_AXIS,
            )
            ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                cache,
                self.meta.slot,
                self.meta.valid_end,
                layer_idx=0,
                num_layers=1,
                chunk_size_global=self.chunk,
                cluster_axis=ccl.SP_AXIS,
                pad_align=ttnn.TILE_SIZE,
            )
            t8.deallocate(True)
            t.deallocate(True)

        gather = self.sp * self.local_seq
        # the fork adds only exact_exp (accurate softmax exp; the source op's streaming kernel hard-codes the
        # Schraudolph approximation, ~1.3 % rel error on this layer's attention output)
        exact = settings.get("SDPA_EXACT_EXP")
        sdpa = (
            ttnn.bringup.ring_joint_scaled_dot_product_attention
            if exact
            else (ttnn.transformer.ring_joint_scaled_dot_product_attention)
        )
        extra = {"exact_exp": True} if exact else {}
        out, _, _ = sdpa(
            q,
            self.kv.key,
            self.kv.value,
            None,
            None,
            None,
            persistent_output_buffer_k=self.ring.gather_buffer("full_k", self.lkv, gather, D, ttnn.bfloat8_b),
            persistent_output_buffer_v=self.ring.gather_buffer("full_v", self.lkv, gather, Dv, ttnn.bfloat8_b),
            joint_strategy="rear",
            logical_n=gather,
            program_config=self.sdpa_program,
            scale=self.scale,
            compute_kernel_config=self.sdpa_cfg,
            dim=2,
            multi_device_global_semaphore=self.ring.semaphores,
            num_links=self.ring.num_links,
            cluster_axis=ccl.SP_AXIS,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=self.ring.ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
            slot_id=self.meta.slot,
            kv_actual_isl_tensor=self.meta.kv_actual,
            kv_cache_num_layers=1,
            kv_cache_layer_idx=0,
            **extra,
        )
        q.deallocate(True)
        heads = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        fp32_sum = settings.get("ATTN_OUT_FP32")
        partial = ttnn.linear(
            heads,
            self.wo,
            transpose_b=True,
            dtype=ttnn.float32 if fp32_sum else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_cfg,
        )
        heads.deallocate(True)
        y = ccl.scatter_rows(partial)
        partial.deallocate(True)
        if fp32_sum:  # the TP partials summed in fp32, rounded to bf16 once
            y32 = y
            y = ttnn.typecast(y32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y32.deallocate(True)
        return y

    def _rope(self, t, cos, sin, R):
        """HF rotate_half RoPE on the first R dims of every head."""
        shape = tuple(t.shape)
        rot = ttnn.slice(t, (0, 0, 0, 0), shape[:-1] + (R,), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rest = ttnn.slice(t, (0, 0, 0, R), shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t.deallocate(True)
        rotated = ttnn.experimental.rotary_embedding(
            rot, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.rope_cfg
        )
        rot.deallocate(True)
        out = ttnn.concat((rotated, rest), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rotated.deallocate(True)
        rest.deallocate(True)
        return out
