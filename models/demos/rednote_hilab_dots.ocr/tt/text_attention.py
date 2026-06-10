# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text attention for dots.ocr.

Qwen2Attention (the dots.ocr text decoder): causal GQA with 12 Q heads /
2 KV heads, head_dim 128, separate q/k/v projections WITH bias, o_proj
without bias, default rope theta=1e6.

TTNN mapping (cf. reference_impl models/tt_transformers/tt/attention.py):

- per-device FUSED QKV ``ttnn.linear`` (+ fused bias) — the host loader
  re-packs HF q/k/v into the ``q | k | v`` column layout that
  ``ttnn.experimental.nlp_create_qkv_heads`` expects;
- head split/merge via the fused ops ``nlp_create_qkv_heads`` /
  ``nlp_concat_heads`` (mandatory TTNN idiom, no torch equivalent);
- explicit fp32 HF-convention rope (slice/neg/concat/mul/add), the same
  high-precision recipe as this model's vision tower;
- explicit fp32 attention core: ``ttnn.matmul`` QK^T -> scale -> causal
  mask add -> ``ttnn.softmax`` -> ``ttnn.matmul`` PV.

Why not ``ttnn.transformer.scaled_dot_product_attention``: the dots.ocr
text layer-0 attention logits reach ±3122 (std 664 — Qwen2 attention-sink
behaviour), so bf16 rounding of q/k perturbs logits by O(10) and scrambles
the softmax; the SDPA kernel is bf16/bf8b-only and measures PCC ~0.92 on
this block (per-stage isolation; all surrounding stages 0.9999+). The
whole attention path therefore runs fp32 weights + fp32 activations with
HiFi4 + fp32 accumulation. Tiny per-chip core (3 heads, hd 128) so the
fp32 cost is negligible at this phase.

Parallelism plan (ARCHITECTURE.md / inventory notes): placement=shard 4-way
— 3 Q heads per chip; kv_replication=2, i.e. each of the 2 KV heads is
REPLICATED onto the 2 chips serving its Q-head group so attention stays
chip-local (do NOT copy tt_transformers' kv-head divisibility assert). The
host loader builds one per-chip fused QKV slice (KV rows replicated up to
the chip's Q-head count, q|k|v) per device and shards the concatenation
with ``ShardTensorToMesh(dim=-1)``; o_proj is row-parallel (``dim=-2``)
and its per-chip PARTIAL sums are combined with a sync all-reduce
(``ttnn.reduce_scatter`` + ``ttnn.all_gather``, fp32 fabric accumulation;
swapped from all_gather + local adds in the optimization phase per
tp-guidance). On a single device the sharding degenerates to the
replicated full computation and the CCL is skipped.

Generation phase (decode_strategy=kv_cache): beside the full-sequence
prefill ``forward`` the block carries a KV-cached single-token decode path
— ``init_kv_cache`` pre-allocates persistent per-chip fp32 K/V caches
``[1, heads_per_device, max_seq, head_dim]`` (kv_replication=2 means each
chip's Q-head group holds a FULL copy of its KV head's cache, so decode
attention is chip-local — no CCL in attention; the only CCL stays after the
row-parallel o_proj). ``forward`` populates the cache once per call via
``ttnn.fill_cache`` (post-rope K, V), then ``forward_decode`` processes ONE
token row per step: QKV linear -> nlp_create_qkv_heads -> explicit fp32
rope -> ``ttnn.experimental.paged_update_cache`` (single contiguous cache,
no page table; the seamless_m4t kv_cache recipe) -> single-row fp32
attention core over the cached K/V with a streamed [1, 1, 1, max_seq]
additive mask. The bf16-only ``scaled_dot_product_attention_decode`` kernel
is rejected for the same measured reason as bf16 SDPA in prefill (Qwen2
attention-sink ±3122 logits, PCC ~0.92): the explicit fp32 core IS this
model's SDPA-decode idiom; ``paged_update_cache`` supports fp32 caches.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtTextAttention(LightweightModule):
    """dots.ocr text GQA: fused QKV(+bias) -> heads -> fp32 rope -> fp32 causal attention -> o_proj -> all-reduce.

    Args:
        mesh_device: ttnn mesh device handle (1xN line; weights TP-sharded).
        state_dict: {"q_proj.weight": [1536, 1536], "q_proj.bias": [1536],
            "k_proj.weight": [256, 1536], "k_proj.bias": [256],
            "v_proj.weight": [256, 1536], "v_proj.bias": [256],
            "o_proj.weight": [1536, 1536]} torch tensors
            (HF keys model.layers.N.self_attn.*).
        num_heads: Q heads (dots.ocr text: 12, head_dim 128).
        num_kv_heads: KV heads (dots.ocr text: 2; kv_replication onto the
            chips of each Q-head group, then up to heads_per_device).
        dtype: on-device ACTIVATION dtype (fp32 default — see module
            docstring; bf16 activations lose the softmax to logit
            quantization).
        weight_dtype: QKV/o_proj weight+bias storage dtype (None -> dtype).
            The HF checkpoint is bf16, so bf16 storage holds the exact
            checkpoint values — the ±3122 attention-sink mitigation lives in
            the fp32 ACTIVATIONS/accumulation, not in weight storage. The
            decode step is DRAM weight-streaming bound, so halving weight
            bytes is the perf-phase lever; the fp32 attention core, fp32 KV
            cache and HiFi4 + fp32 accumulation are unchanged.
    """

    def __init__(self, mesh_device, state_dict, num_heads=12, num_kv_heads=2, dtype=ttnn.float32, weight_dtype=None):
        super().__init__()
        self.mesh_device = mesh_device
        num_devices = mesh_device.get_num_devices()
        self.num_devices = num_devices
        weight_dtype = weight_dtype or dtype

        q_w = state_dict["q_proj.weight"]  # [nh*hd, hidden]
        k_w = state_dict["k_proj.weight"]  # [nkv*hd, hidden]
        v_w = state_dict["v_proj.weight"]  # [nkv*hd, hidden]
        q_b = state_dict["q_proj.bias"]
        k_b = state_dict["k_proj.bias"]
        v_b = state_dict["v_proj.bias"]
        o_w = state_dict["o_proj.weight"]  # [hidden, nh*hd], no bias

        hidden = q_w.shape[-1]
        head_dim = q_w.shape[0] // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        assert num_heads % num_devices == 0, f"{num_heads} Q heads on {num_devices} devices"
        hpd = num_heads // num_devices  # Q heads per device (qb 1x4: 3)
        self.heads_per_device = hpd

        # kv_replication: each chip serves ONE KV-head group and stores that
        # head's K/V rows ONCE (q3|k1|v1 per-chip packing). Prefill expands
        # K/V to hpd head copies on device for its MHA fp32 core; the decode
        # SDPA kernel consumes the single KV head natively (GQA, perf REDO
        # 3 — drops 44% of the per-chip QKV weight bytes and 3x of the KV
        # cache traffic vs the old q3|k3|v3 replicated packing).
        per_dev_w, per_dev_b = [], []
        for d in range(num_devices):
            q_rows = slice(d * hpd * head_dim, (d + 1) * hpd * head_dim)
            g = (d * num_kv_heads) // num_devices  # this chip's KV head
            kv_rows = slice(g * head_dim, (g + 1) * head_dim)
            per_dev_w.append(torch.cat([q_w[q_rows], k_w[kv_rows], v_w[kv_rows]], dim=0))
            per_dev_b.append(torch.cat([q_b[q_rows], k_b[kv_rows], v_b[kv_rows]], dim=0))
        fused_w = torch.cat(per_dev_w, dim=0)  # [N*(hpd+2)*hd, hidden]
        fused_b = torch.cat(per_dev_b, dim=0)

        shard_cols = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        # Transpose for x @ W^T; ShardTensorToMesh(dim=-1) hands each device
        # exactly its pre-packed q|k|v slice (column-parallel QKV).
        self.wqkv = ttnn.from_torch(
            fused_w.transpose(-2, -1).reshape(1, 1, hidden, -1).contiguous(),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_cols,
        )
        self.bqkv = ttnn.from_torch(
            fused_b.reshape(1, 1, 1, -1),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_cols,
        )
        # o_proj all_gather->matmul (perf REDO 3 lever d, Linear-topology
        # form): the weight is REPLICATED full-width and the per-chip
        # concat-heads activation (hpd*hd wide) is all_gathered on dim=3
        # into the full head order, then ONE matmul yields the complete
        # replicated output — no partial sums, so the old reduce_scatter +
        # all_gather pair disappears (2 CCL ops -> 1, and the gathered
        # payload is the narrow heads row, not the full-hidden partials).
        # The fused ttnn.experimental.all_gather_matmul_async kernel itself
        # is Ring-topology-only (tt_transformers gates it on
        # ccl_topology == Topology.Ring); qb's 1x4 mesh is a Linear line.
        self.wo = ttnn.from_torch(
            o_w.transpose(-2, -1).reshape(1, 1, num_heads * head_dim, hidden).contiguous(),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Decode fused-rope constant (perf REDO 3): rotate_half as ONE fp32
        # matmul on the fused QKV row. R is block-diagonal with the HF
        # rotate-half map per q/k head block and ZEROS over the v block, so
        # rot = xqkv @ R gives [-x2|x1] per q/k head and 0 for v; the rope
        # then is xqkv*cos_cat + rot*sin_cat with cos_cat=1/sin_cat=0 over v
        # (v passes through EXACTLY). 0/±1 selection in fp32 matmul is exact
        # math — bit-identical to the old slice/neg/concat chain — so the
        # fp32 attention-sink mandate is untouched while 14 rope ops/layer
        # collapse to 4 (matmul, mul, mul, add) applied pre-head-split.
        self.qkv_width = (hpd + 2) * head_dim  # per-chip fused q|k|v row width
        half = head_dim // 2
        r_head = torch.zeros(head_dim, head_dim)
        for j in range(half):
            r_head[j + half, j] = -1.0
            r_head[j, j + half] = 1.0
        rot_full = torch.zeros(self.qkv_width, self.qkv_width)
        for h in range(hpd + 1):  # q heads then the single k head; v block stays zero
            s = h * head_dim
            rot_full[s : s + head_dim, s : s + head_dim] = r_head
        self.rope_rot = ttnn.from_torch(
            rot_full.reshape(1, 1, self.qkv_width, self.qkv_width),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.heads_l1_height_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

        # Decode SDPA (perf REDO 3 lever c): the fused bf16-only
        # scaled_dot_product_attention_decode kernel, with Q PRE-SCALED by
        # 1/sqrt(head_dim) (folded into the fused-rope cos/sin q-section on
        # host — zero extra ops) and scale=1.0 in the kernel, so every QK
        # logit shrinks from ±3122 to ±276 before bf16 softmax sees it
        # (taming the Qwen2 attention sink). KV cache stores bf16 to feed
        # the kernel. Gated by the e2e WER parity test like every step.
        self.decode_q_prescale = head_dim**-0.5
        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_out_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(32, head_dim),  # B=1 user -> 1 core, heads padded to a tile
            core_grid=ttnn.num_cores_to_corerangeset(1, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # 8x8 grid: the decode kernel tree-reduces across cores per KV head,
        # capped at 64 cores/head (the full 110-core default TT_FATALs with
        # B=1, nkv=1). exp_approx_mode=False: exact softmax exp (±3122-sink
        # numerics; bf16 already costs precision, don't stack approx exp).
        self.sdpa_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

    # ------------------------------------------------------------------
    # Host-side input preparation (rope tables stay on host —
    # ARCHITECTURE.md hybrid_notes). Not part of the device forward path.
    # ------------------------------------------------------------------
    def prepare_rope(self, cos: torch.Tensor, sin: torch.Tensor):
        """Replicated fp32 HF-convention cos/sin device tensors.

        cos/sin: [batch, seq, head_dim] HF-convention tables from
        reference text_rope_cos_sin (theta=1e6). Kept [1, 1, seq, head_dim];
        the explicit fp32 rope broadcasts them over the head dim.
        """

        def _to_dev(t):
            t = t.float().reshape(1, 1, t.shape[-2], t.shape[-1]).contiguous()
            return ttnn.from_torch(
                t,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return _to_dev(cos.squeeze(0)), _to_dev(sin.squeeze(0))

    def prepare_causal_mask(self, seq: int):
        """Replicated fp32 additive causal mask [1, heads_per_device, seq, seq]."""
        mask = torch.triu(torch.full((seq, seq), torch.finfo(torch.float32).min / 2), diagonal=1)
        mask = mask.reshape(1, 1, seq, seq).expand(1, self.heads_per_device, seq, seq).contiguous()
        return ttnn.from_torch(
            mask,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def decode_rope_rows(self, position: int, theta: float = 1e6):
        """Host fp32 (cos_cat, sin_cat) [1, 1, 1, qkv_width] for ONE position.

        Fused-rope tables over the per-chip q|k|v row: the HF-convention
        cos/sin (theta=1e6, reference text_rope_cos_sin) tiled across the
        2*hpd q/k head blocks, with cos=1 / sin=0 over the v block so v
        passes through the fused rope exactly.
        """
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        freqs = float(position) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        hpd, hd = self.heads_per_device, self.head_dim
        cos_cat = torch.ones(1, 1, 1, self.qkv_width)
        sin_cat = torch.zeros(1, 1, 1, self.qkv_width)
        cos_cat[..., : (hpd + 1) * hd] = emb.cos().repeat(hpd + 1)
        sin_cat[..., : (hpd + 1) * hd] = emb.sin().repeat(hpd + 1)
        # Q pre-scale (lever c): rope is linear in x, so scaling the cos/sin
        # q-section scales rope(q) by exactly decode_q_prescale; the SDPA
        # kernel then runs with scale=1.0 on ±276-magnitude logits.
        cos_cat[..., : hpd * hd] *= self.decode_q_prescale
        sin_cat[..., : hpd * hd] *= self.decode_q_prescale
        return cos_cat, sin_cat

    def prepare_decode_rope(self, position: int, theta: float = 1e6):
        """Replicated fp32 fused-rope rows (cos_cat, sin_cat) [1,1,1,qkv_width]."""
        rows = self.decode_rope_rows(position, theta)
        return tuple(
            ttnn.from_torch(
                t,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for t in rows
        )

    # ------------------------------------------------------------------
    # KV cache (generation phase, decode_strategy=kv_cache)
    # ------------------------------------------------------------------
    def init_kv_cache(self, max_seq_len: int):
        """Pre-allocate persistent per-chip bf16 K/V caches [1, 1, max_seq, hd].

        Each chip stores its Q-head group's single KV head ONCE
        (kv_replication=2 across the mesh keeps decode attention chip-local;
        the decode SDPA kernel maps the chip's hpd Q heads onto the one KV
        head natively — GQA). Buffers persist across calls;
        ``forward(kv_cache=...)`` refills slots 0..S-1 on each prefill.
        bf16 storage feeds the bf16-only scaled_dot_product_attention_decode
        kernel (perf REDO 3 lever c — gated by the e2e WER parity test).
        """
        zeros = torch.zeros(1, 1, max_seq_len, self.head_dim)
        k_cache, v_cache = (
            ttnn.from_torch(
                zeros,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for _ in range(2)
        )
        # Persistent int32 [1] slot buffer (stable address for trace reuse).
        pos_tt = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return {"k": k_cache, "v": v_cache, "pos": pos_tt, "max_seq": max_seq_len}

    # ------------------------------------------------------------------
    # Device forward (causal prefill over the full sequence)
    # ------------------------------------------------------------------
    def _apply_rope_fp32(self, t: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Explicit fp32 HF-convention rope: t*cos + rotate_half(t)*sin."""
        shape = t.shape
        half = self.head_dim // 2
        t1 = ttnn.slice(t, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half])
        t2 = ttnn.slice(t, [0, 0, 0, half], [shape[0], shape[1], shape[2], self.head_dim])
        neg_t2 = ttnn.neg(t2)
        ttnn.deallocate(t2)
        rot = ttnn.concat([neg_t2, t1], dim=-1)
        ttnn.deallocate(neg_t2)
        ttnn.deallocate(t1)
        out = ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(rot, sin))
        ttnn.deallocate(rot)
        ttnn.deallocate(t)
        return out

    def forward(self, x_11SH: ttnn.Tensor, rot_mats, causal_mask: ttnn.Tensor, kv_cache=None) -> ttnn.Tensor:
        """x_11SH: [1, 1, seq, hidden] fp32 TILE_LAYOUT, replicated on the mesh.

        rot_mats: (cos, sin) from prepare_rope, [1, hpd, seq, head_dim] fp32.
        causal_mask: from prepare_causal_mask, [1, hpd, seq, seq] fp32.
        kv_cache: optional dict from init_kv_cache — prefill populates slots
            0..seq-1 with the post-rope K and V via ttnn.fill_cache (pad rows
            beyond the live prompt are masked by the decode mask until a
            decode step overwrites them).
        Returns: [1, 1, seq, hidden] fp32, replicated (all-reduced o_proj output).
        """
        xqkv = ttnn.linear(
            x_11SH,
            self.wqkv,
            bias=self.bqkv,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Chip-local heads: hpd Q heads + the chip's single KV head (q3|k1|v1
        # packing); the MHA fp32 core below expands K/V to hpd copies.
        q, k1, v1 = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.heads_per_device,
            num_kv_heads=1,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        cos, sin = rot_mats  # [1, 1, seq, hd] — broadcasts over the head dim
        q = self._apply_rope_fp32(q, cos, sin)
        k1 = self._apply_rope_fp32(k1, cos, sin)

        if kv_cache is not None:
            # bf16 [1, 1, max_seq, hd] cache (decode SDPA kernel dtype);
            # prefill attention math below stays on the fp32 k/v tensors.
            k_bf = ttnn.typecast(k1, ttnn.bfloat16)
            v_bf = ttnn.typecast(v1, ttnn.bfloat16)
            ttnn.fill_cache(kv_cache["k"], k_bf, 0)
            ttnn.fill_cache(kv_cache["v"], v_bf, 0)
            ttnn.deallocate(k_bf)
            ttnn.deallocate(v_bf)

        # Expand the single KV head to hpd copies for the plain-MHA fp32 core.
        k = ttnn.concat([k1] * self.heads_per_device, dim=1)
        v = ttnn.concat([v1] * self.heads_per_device, dim=1)
        ttnn.deallocate(k1)
        ttnn.deallocate(v1)

        # Explicit fp32 causal attention (bf16 SDPA kernel scrambles the
        # ±3000-magnitude Qwen2 layer-0 logits — see module docstring).
        k_t = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        scores = ttnn.matmul(
            q,
            k_t,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)
        scores = ttnn.multiply(scores, self.scale)
        scores = ttnn.add(scores, causal_mask)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        ttnn.deallocate(scores)
        attn = ttnn.matmul(
            probs,
            v,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(probs)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # all_gather -> full o_proj matmul (lever d, see __init__): gather
        # the per-chip heads row into the full head order, then one matmul
        # against the replicated weight gives the complete output — no
        # partial-sum all-reduce needed.
        if self.num_devices > 1:
            attn_full = ttnn.all_gather(attn, dim=3, num_links=2, topology=ttnn.Topology.Linear)
            ttnn.deallocate(attn)
            attn = attn_full
        out = ttnn.linear(
            attn,
            self.wo,
            dtype=x_11SH.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)
        return out

    # ------------------------------------------------------------------
    # Device forward (KV-cached single-token decode step)
    # ------------------------------------------------------------------
    def forward_decode(self, x_111H: ttnn.Tensor, kv_cache, rot_step) -> ttnn.Tensor:
        """One KV-cached token step: O(1) projections + attention over the cache.

        x_111H: [1, 1, 1, hidden] fp32 TILE_LAYOUT, replicated on the mesh.
        kv_cache: dict from init_kv_cache. ``kv_cache["pos"]`` (persistent
            int32 [1]) must already hold this token's cache slot; it drives
            BOTH the cache write and SDPA causality (no mask tensor).
        rot_step: (cos_cat, sin_cat) from prepare_decode_rope,
            [1, 1, 1, qkv_width] fp32 fused-rope rows.
        Returns: [1, 1, 1, hidden] fp32, replicated.
        No CCL in attention: each chip's KV-head cache copy is local.

        Perf REDO 3 decode path: fused fp32 rope on the QKV row (1 matmul
        with the 0/±1 rotate-half block matrix + 2 mul + 1 add — exact
        math, see __init__) BEFORE the head split, then
        ``nlp_create_qkv_heads_decode`` which hands k/v straight to
        ``paged_update_cache`` in its HEIGHT_SHARDED [1, B=1, heads, hd]
        layout (the old per-tensor permute + interleaved_to_sharded pairs
        are gone).
        """
        xqkv = ttnn.linear(
            x_111H,
            self.wqkv,
            bias=self.bqkv,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        cos_cat, sin_cat = rot_step
        rot = ttnn.matmul(
            xqkv,
            self.rope_rot,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        roped = ttnn.add(ttnn.multiply(xqkv, cos_cat), ttnn.multiply(rot, sin_cat))
        ttnn.deallocate(xqkv)
        ttnn.deallocate(rot)

        # bf16 boundary for the fused decode kernels (rope above is fp32
        # exact; q is already pre-scaled by 1/sqrt(hd) via the rope rows).
        # L1, not DRAM: nlp_create_qkv_heads_decode reads bf16 sub-tile
        # lines (32B) which silently drop ODD head rows from DRAM on
        # Blackhole (64B DRAM NoC alignment — measured: head 1 of 3 comes
        # back zero from a DRAM input, correct from L1).
        roped_bf = ttnn.typecast(roped, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(roped)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            roped_bf,
            num_heads=self.heads_per_device,
            num_kv_heads=1,
            memory_config=self.heads_l1_height_sharded,
        )
        ttnn.deallocate(roped_bf)

        # k/v are already [1, B=1, hpd, hd] HEIGHT_SHARDED L1 — exactly the
        # paged_update_cache input layout.
        ttnn.experimental.paged_update_cache(kv_cache["k"], k, update_idxs_tensor=kv_cache["pos"])
        ttnn.experimental.paged_update_cache(kv_cache["v"], v, update_idxs_tensor=kv_cache["pos"])
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Fused decode attention over the cache; causality comes from the
        # runtime cur_pos tensor (attends slots 0..pos inclusive), replacing
        # the streamed [1,1,1,max_seq] additive mask. scale=1.0: Q was
        # pre-scaled on host (see decode_rope_rows).
        attn_1bqd = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            kv_cache["k"],
            kv_cache["v"],
            cur_pos_tensor=kv_cache["pos"],
            scale=1.0,
            program_config=self.sdpa_decode_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=self.sdpa_out_mem_cfg,
        )
        ttnn.deallocate(q)
        attn_ws = ttnn.experimental.nlp_concat_heads_decode(attn_1bqd, num_heads=self.heads_per_device)
        ttnn.deallocate(attn_1bqd)
        attn_rows = ttnn.sharded_to_interleaved(attn_ws, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_ws)
        # concat_heads_decode pads the user dim to a tile (32 rows); only
        # row 0 is our token. Slice it back to [1, 1, 1, hpd*hd] so o_proj
        # and the CCL all-reduce move one row, not 32.
        attn = ttnn.slice(attn_rows, [0, 0, 0, 0], [1, 1, 1, attn_rows.shape[-1]])
        ttnn.deallocate(attn_rows)
        # all_gather -> full o_proj matmul (lever d, see __init__): one CCL
        # op on the narrow bf16 heads row replaces the reduce_scatter +
        # all_gather all-reduce of fp32 partial sums.
        if self.num_devices > 1:
            attn_full = ttnn.all_gather(attn, dim=3, num_links=2, topology=ttnn.Topology.Linear)
            ttnn.deallocate(attn)
            attn = attn_full
        out = ttnn.linear(
            attn,
            self.wo,
            dtype=x_111H.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)
        return out
