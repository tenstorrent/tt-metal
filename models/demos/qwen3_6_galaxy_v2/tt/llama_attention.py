# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm

# qwen3.6 partial-RoPE tile size (head_dim must be tile-aligned for rms_norm).
_QWEN36_TILE = 32


def _qwen36_kv_cache_dtype():
    """KV paged cache dtype. ``QWEN36_KV_BF8=1`` halves the per-step KV read that
    dominates full-attention SDPA at long context (llama70b ships bf8 KV). The
    fill/update_cache kernels accept a bf16 producer into a bf8 cache (they
    quantize on write) and paged SDPA reads bf8 directly, so ONLY the cache
    allocation changes — producers stay bf16.
    Default OFF: bf8 KV quantization compounds over the 16 full-attn layers and
    garbled Qwen3.6-27B decode output even at ISL-128 (next token 248068→43223,
    coherent English → CJK mojibake). Do NOT enable until it clears the 128k
    coherence gate; on this model it does not."""
    return ttnn.bfloat8_b if os.environ.get("QWEN36_KV_BF8", "0") == "1" else ttnn.bfloat16


# ---------------------------------------------------------------------------
# qwen3.6 internal helpers (ported from
# ``models/demos/qwen3_6_galaxy/tt/llama_attention.py`` — PCC-verified in v1)
# ---------------------------------------------------------------------------


def _qwen36_qknorm_flat_to_heads(
    x_flat,
    weight,
    eps: float,
    B: int,
    n_heads: int,
    T: int,
    hd: int,
    compute_kernel_config,
):
    """Per-head RMSNorm of [B, T, n_heads*hd] → [B, n_heads, T, hd].

    V2-11 (lever H): fast path for the 8-head Q at decode time. Reshape
    the flat [B, T, n*hd] into a per-head batch [B*n, T, hd] and run a
    SINGLE rms_norm; this collapses the per-head loop (8 slice + 8 rms_norm
    + 8 reshape = 24 ops per layer × 16 full-attention layers = 384 ops
    per decode step) into 1 reshape + 1 rms_norm + 1 reshape + 1 view.
    Falls back to the per-head loop for n_heads >= 32 (tile-padding
    edge case) and for n_heads == 1 (single rms_norm covers it).
    """
    if n_heads == 1:
        x_normed_3d = ttnn.rms_norm(
            x_flat,
            weight=weight,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        view_4d = ttnn.reshape(x_normed_3d, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_4d.deallocate(True)
        x_normed_3d.deallocate(True)
        return out

    # V2-11 (lever H): batched rms_norm path. Treat each head as its own
    # batch element. The data ordering matches: [B, T, n_heads*hd] flat
    # has head 0 at offset 0, head 1 at offset hd, ..., so reshape to
    # [B*n_heads, T, hd] preserves head boundaries.
    x_per_head = ttnn.reshape(x_flat, [B * n_heads, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_normed = ttnn.rms_norm(
        x_per_head,
        weight=weight,
        epsilon=eps,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    x_per_head.deallocate(True)
    # Reshape back to [B, n_heads, T, hd].
    out = ttnn.reshape(x_normed, [B, n_heads, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if out is not x_normed:
        x_normed.deallocate(True)
    return out


def _qwen36_flat_to_heads(x_flat, B: int, n_heads: int, T: int, hd: int):
    """Reshape [B, T, n_heads*hd] → [B, n_heads, T, hd].

    V2-11 (lever H, sibling of qknorm_flat_to_heads): single 4-D reshape
    on tile-aligned inputs. Data ordering preserved because the flat
    layout packs heads contiguously by definition.
    """
    if n_heads == 1:
        view_4d = ttnn.reshape(x_flat, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_4d.deallocate(True)
        return out

    # V2-11: pure reshape — same data ordering. ttnn.reshape allows non-
    # tile-aligned reshapes for ROW_MAJOR; the input here is TILE_LAYOUT
    # from the preceding linear / slice, so we go via [B*n_heads, T, hd]
    # which IS tile-aligned in the last dim (hd is tile-multiple).
    out = ttnn.reshape(x_flat, [B, n_heads, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


def _qwen36_heads_to_flat(x_heads, B: int, n_heads: int, T: int, hd: int):
    """Reshape [B, n_heads, T, hd] → [B, T, n_heads*hd].

    V2-11 (lever H): reverse of ``_qwen36_flat_to_heads``. When T=1
    (decode), the [B, n_h, T, hd] memory layout matches [B, n_h*hd]
    contiguous because the T-dim is a singleton; a pure reshape works.
    For T>1 (prefill) we must materialize the layout — concat per-time
    head slices — but the prefill path doesn't bottleneck on this
    function. The fast-path here applies only when T==1.
    """
    if n_heads == 1:
        view_3d = ttnn.reshape(x_heads, [B, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_3d.deallocate(True)
        return out

    if T == 1:
        # Decode fast path: [B, n_heads, 1, hd] is contiguous as
        # [B, 1, n_heads*hd]. Pure reshape.
        out = ttnn.reshape(x_heads, [B, T, n_heads * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    slice_list = []
    time_slice_tensors = []
    for h in range(n_heads):
        head_h = ttnn.slice(x_heads, [0, h, 0, 0], [B, h + 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_h_3d = ttnn.reshape(head_h, [B, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        slice_list.append(head_h)
        time_slice_tensors.append(head_h_3d)

    out = ttnn.concat(time_slice_tensors, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in time_slice_tensors:
        t.deallocate(True)
    for s in slice_list:
        s.deallocate(True)
    return out


class TtLlamaAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher_setup=None,
        tt_ccl=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        # qwen3.6-specific config flags (read early — gate weight build + forward
        # branches below).  Default to False so the 70B / qwen3-32B / olmo paths
        # remain untouched when these attributes are absent.
        self.is_qwen36 = getattr(configuration, "is_qwen36", False)
        self.zero_centered_norm = getattr(configuration, "zero_centered_norm", False)
        self.rope_dim = getattr(configuration, "rope_dim", self.head_dim)
        self.partial_rotary_factor = getattr(configuration, "partial_rotary_factor", 1.0)
        # Pull common attributes; qwen3.6's TtQwen36ModelArgs does not set the
        # 70B-prefetcher-derived fields, so use getattr with safe defaults.
        if self.is_qwen36:
            self.min_kv_prefill_shard_seqlen = getattr(configuration, "min_kv_prefill_shard_seqlen", 0)
            self.ccl_dtype = getattr(configuration, "ccl_dtype", ttnn.bfloat8_b)
            self.num_reduce_scatter_links = getattr(configuration, "num_reduce_scatter_links", 1)
            self.num_all_gather_links = getattr(configuration, "num_all_gather_links", 1)
        else:
            self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
            self.ccl_dtype = configuration.ccl_dtype
            self.num_reduce_scatter_links = configuration.num_reduce_scatter_links
            self.num_all_gather_links = configuration.num_all_gather_links

        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads
        self.batch_size_per_device_group = max(self.max_batch_size // self.num_device_groups, 1)

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl

        # TODO: Fix this once all-gather supports < tile_size
        # 70B-only batch-offset / column-bounds bookkeeping for the
        # llama_rs_create_heads decode path + chunked-SDPA mask. qwen3.6 v2 takes
        # an entirely different attention path (col-sharded QKVG + per-head SDPA
        # mirroring v1 ``models/demos/qwen3_6_galaxy/tt/llama_attention.py``) and
        # has no use for these tensors. Skipping them on qwen3.6 also avoids
        # touching the 70B-prefetcher mem-configs that are not built on this port.
        if not self.is_qwen36:
            weight = torch.zeros(1, 32, 8, 32)
            for i in range(32):
                col = i % 4  # This determines which group of 8 to select
                weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

            # Select batch_offset with create_qkv_heads_decode instead of selection matmul
            batch_offset = [
                0,
                8,
                16,
                24,
            ]  # TODO: batch offset is 8 for batch=32, this should be adjusted for variable batch_size
            self.batch_offset_tt_tensor = ttnn.as_tensor(
                torch.tensor(batch_offset, dtype=torch.int32).reshape(4, 1),
                dtype=ttnn.int32,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=mesh_device, dims=(None, 0), mesh_shape=list(mesh_device.shape)
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.slice_size = 8  # Slice size is 8 since we are consuming 8 users per chip

            # Column bounds for prefix caching: mask = (lower <= user_id < upper)
            # Column 0: [0, 8), Column 1: [8, 16), Column 2: [16, 24), Column 3: [24, 32)
            # Shape [8, 4, 1, 32]: last dim must be 32 for ttnn typecast compatibility (ROW_MAJOR requires %32)
            # Use uint32 to match user_id dtype;
            # Per-column user_id bounds for chunked SDPA mask: column col is active for user_id in [col*8, (col+1)*8).
            # Sharded over 8x4 mesh (dims 0,1); each device gets (1, 1, 1, 32). Column 0: [0,8), 1: [8,16), 2: [16,24), 3: [24,32).
            lower = torch.zeros(8, 4, 1, 32, dtype=torch.int32)
            upper = torch.zeros(8, 4, 1, 32, dtype=torch.int32)
            for col in range(4):
                lower[:, col, :, :] = col * 8
                upper[:, col, :, :] = (col + 1) * 8
            self.column_lower = ttnn.from_torch(
                lower,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
            )
            self.column_upper = ttnn.from_torch(
                upper,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
            )

        # qwen3.6 forces bfloat16 attention weights (wqkvg / wo).  V2-7b layer-3
        # forward PCC test: bfloat8_b weights drop PCC from 0.999 → 0.77 once
        # the attention is composed inside the full TtTransformer.forward path
        # (block-level bypass was 0.9997 even at bf8 — the residual + norm +
        # MLP layering amplifies the bf8 quantisation noise).  Layer 0
        # (DeltaNet) already kept its own bfloat16 weights so it was unaffected;
        # layer 3 (full_attention) used self.dtype=bf8 for wqkvg / wo.
        #
        # V2-CONFIG-D: optionally override the V2-7b bf16 lock-in to match
        # llama70b's bf8 attention weights. Per user direction this sweep
        # treats PCC as data (not a gate) — accept the V2-7b PCC drop and
        # measure wall-clock recovery from halved DRAM weight reads.
        _qwen36_attn_bf8 = (
            getattr(configuration, "is_qwen36", False) and os.environ.get("QWEN36_ATTN_WEIGHTS_BF8", "0") == "1"
        )
        if _qwen36_attn_bf8:
            self.dtype = ttnn.bfloat8_b
        elif getattr(configuration, "is_qwen36", False):
            # V4: optional fp32 escape for multimodal precision push. Default is bf16
            # per V2-7b lock-in. Set QWEN36_FP32_WEIGHTS=1 to honor caller-passed dtype
            # (e.g. ttnn.float32) for attention weights — used by VLM prefill PCC test.
            if os.environ.get("QWEN36_FP32_WEIGHTS", "0") == "1":
                self.dtype = dtype
            else:
                self.dtype = ttnn.bfloat16
        else:
            self.dtype = dtype
        self.qk_norm = configuration.qk_norm

        if self.is_qwen36:
            # qwen3.6 v2 builds its own compute-kernel configs (the parent
            # TtModelArgs setup is not invoked by TtQwen36ModelArgs).
            self.grid_size = getattr(configuration, "max_grid_size", None)
            self.compute_kernel_config_hifi2 = getattr(
                configuration,
                "compute_kernel_config_hifi2",
                ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=True,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                ),
            )
            self.compute_kernel_config_hifi2_fp16 = getattr(
                configuration, "compute_kernel_config_hifi2_fp16", self.compute_kernel_config_hifi2
            )
            self.compute_kernel_config_hifi4 = getattr(
                configuration,
                "compute_kernel_config_hifi4",
                ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                ),
            )
            # V2-CONFIG-C: optionally downgrade FA attention matmuls from
            # HiFi4 → HiFi2 to match llama70b's production-tuned config.
            # MLP already uses HiFi2; this flips the ~20 attention sites
            # that reference self.compute_kernel_config_hifi4 in one shot.
            if os.environ.get("QWEN36_ATTN_HIFI2", "0") == "1":
                self.compute_kernel_config_hifi4 = self.compute_kernel_config_hifi2
            self.transformation_mats = transformation_mats
            # qwen3.6's TtQwen36ModelArgs exposes ``model_config`` as a plain
            # attribute (no ``get_model_config()`` helper).
            self.model_config = dict(configuration.model_config) if hasattr(configuration, "model_config") else {}
            self.model_config["USE_PREFETCHER"] = getattr(configuration, "use_prefetcher", False)
            # ``ccl_topology`` is a @property on qwen36, callable on the 70B
            # config — handle both.
            ccl_t = getattr(configuration, "ccl_topology", None)
            self.ccl_topology = ccl_t() if callable(ccl_t) else ccl_t
            self.is_multichip = getattr(configuration, "is_multichip", self.num_devices > 1)
        else:
            self.grid_size = configuration.max_grid_size

            self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
            self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

            self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

            self.transformation_mats = transformation_mats

            self.model_config = configuration.get_model_config()
            self.model_config["USE_PREFETCHER"] = configuration.use_prefetcher
            self.ccl_topology = configuration.ccl_topology()
            self.is_multichip = configuration.is_multichip

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert self.num_devices == 32

        if self.is_qwen36:
            # ----------------------------------------------------------------
            # qwen3.6 fused-QKVG weight build (see v1 ``_build_weights`` —
            # PCC-verified against HF reference). Two qwen3.6-specific quirks:
            #
            # 1. ``attention.wq.weight`` shape ``[12288, 5120]`` packs Q and an
            #    output-gate ``per-head interleaved``:
            #       reshape(n_q, 2, hd, H) → [:, 0] = Q, [:, 1] = Gate.
            # 2. The 4-way column-parallel layout interleaves head GROUPS by
            #    column so each mesh-col owns its own contiguous slice:
            #       per col c: [Q_heads c*nqpc..(c+1)*nqpc | Gate | K | V]
            #    yielding a single concat over cols whose total output width is
            #    ``(n_q + n_q + n_kv + n_kv) * hd / cluster_shape[1]`` per chip.
            #
            # The 70B fused-matmul path (``SHARDED_QKV_RING_MEMCFG``) is
            # skipped — qwen3.6 v2 has ``use_prefetcher=False``, so we keep
            # ``self.wqkvg`` in plain DRAM-interleaved layout.
            # ----------------------------------------------------------------
            # ============================================================
            # V2-TP: 2D Tensor Parallelism for qwen3.6 full-attention.
            # Mirrors llama70b's pattern with KV-head padding 4 → 8 to
            # satisfy 8-row divisibility.  KV replication is GQA-preserving
            # via repeat_interleave(2, dim=0) — verified bit-identical in
            # /tmp/test_gqa_kv_replication.py (V2-TP-1).
            #
            # Mesh: (rows=8, cols=4) = (cluster_axis=0, cluster_axis=1).
            # Heads on rows (8-way): 24 Q / 8 = 3 per chip, 8 padded KV / 8 = 1 per chip.
            # Hidden on cols (4-way): 5120 / 4 = 1280 per chip.
            # ============================================================
            hd = self.head_dim  # 256
            n_q = self.n_heads  # 24
            n_kv_unpadded = getattr(configuration, "n_kv_heads_unpadded", self.n_kv_heads)  # 4
            n_kv = self.n_kv_heads  # 8 (padded)
            H = self.hidden_size  # 5120
            n_rows = configuration.cluster_shape[0]  # 8
            n_cols = configuration.cluster_shape[1]  # 4
            assert n_q % n_rows == 0, f"n_q={n_q} not divisible by n_rows={n_rows}"
            assert n_kv % n_rows == 0, f"n_kv (padded)={n_kv} not divisible by n_rows={n_rows}"
            n_q_per_chip = n_q // n_rows  # 3
            n_kv_per_chip = n_kv // n_rows  # 1
            self.n_q_per_chip = n_q_per_chip
            self.n_kv_per_chip = n_kv_per_chip
            self.q_dim_per_chip = n_q_per_chip * hd  # 768
            self.gate_dim_per_chip = n_q_per_chip * hd  # 768
            self.k_dim_per_chip = n_kv_per_chip * hd  # 256
            self.v_dim_per_chip = n_kv_per_chip * hd  # 256
            self.total_per_chip = (
                self.q_dim_per_chip + self.gate_dim_per_chip + self.k_dim_per_chip + self.v_dim_per_chip
            )  # 2048
            # Total fused QKVG output width across all rows (host-side concat):
            self.qkvg_total_width = self.total_per_chip * n_rows  # 16384

            # 1. De-interleave Q and gate from q_proj.weight.
            q_proj_w = self.state_dict[wq_str]  # [12288, 5120]
            expected_q = (n_q * 2 * hd, H)
            assert q_proj_w.shape == expected_q, f"q_proj.weight: expected {expected_q}, got {q_proj_w.shape}"
            q_2hd = q_proj_w.reshape(n_q, 2, hd, H)
            wq_native = q_2hd[:, 0, :, :].reshape(n_q * hd, H)  # [6144, 5120]
            wgate_native = q_2hd[:, 1, :, :].reshape(n_q * hd, H)  # [6144, 5120]

            # 1a. Load K, V and replicate 4 → 8 heads (GQA-preserving).
            wk_unpadded = self.state_dict[wk_str]  # [n_kv_unpadded*hd=1024, 5120]
            wv_unpadded = self.state_dict[wv_str]  # [1024, 5120]
            assert wk_unpadded.shape == (n_kv_unpadded * hd, H)
            assert wv_unpadded.shape == (n_kv_unpadded * hd, H)
            # repeat_interleave(2, dim=0): [k0,k1,k2,k3] → [k0,k0,k1,k1,k2,k2,k3,k3].
            # Math verified in V2-TP-1: q_i//3 of replicated == q_i//6 of original.
            wk_native = (
                wk_unpadded.view(n_kv_unpadded, hd, H).repeat_interleave(2, dim=0).reshape(n_kv * hd, H)
            )  # [8*hd=2048, H]
            wv_native = wv_unpadded.view(n_kv_unpadded, hd, H).repeat_interleave(2, dim=0).reshape(n_kv * hd, H)

            # Test-introspection attributes.
            self.q_proj_weight_shape = tuple(wq_native.shape)
            self.gate_proj_weight_shape = tuple(wgate_native.shape)
            self.k_proj_weight_shape = tuple(wk_native.shape)
            self.v_proj_weight_shape = tuple(wv_native.shape)

            # 2. Build the row-sharded QKVG by interleaving HEAD GROUPS per row.
            #    Per row r:  [Q_heads r*n_q_pc..(r+1)*n_q_pc | Gate | K_padded | V_padded]
            #    Output stacked across rows (8 rows × total_per_chip = 16384 total).
            row_blocks = []
            for r in range(n_rows):
                qs = r * n_q_per_chip * hd
                qe = (r + 1) * n_q_per_chip * hd
                ks = r * n_kv_per_chip * hd
                ke = (r + 1) * n_kv_per_chip * hd
                row_blocks.append(
                    torch.cat(
                        [
                            wq_native[qs:qe],
                            wgate_native[qs:qe],
                            wk_native[ks:ke],
                            wv_native[ks:ke],
                        ],
                        dim=0,
                    )
                )
            wqkvg = torch.cat(row_blocks, dim=0)  # [16384, H]
            wqkvg_T = wqkvg.T.contiguous().unsqueeze(0).unsqueeze(0)  # [1, 1, H=5120, 16384]

            # 2D shard: dim 3 (output qkvg=16384) on rows (8-way) → 2048 per chip
            #           dim 2 (input hidden=5120) on cols (4-way) → 1280 per chip
            # Prefill matmul consumes plain DRAM_INTERLEAVED weight; decode
            # uses the secondary DRAM-sharded copy below (V2-DRAM-P1).
            self.wqkvg = ttnn.as_tensor(
                wqkvg_T,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wqkvg_tp2d_row_col"),
            )
            self.wqkv_interleaved = self.wqkvg
            # V2-DRAM-P1: secondary DRAM-width-sharded copy for the decode
            # fast-path matmul (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig).
            _wqkvg_ds_memcfg = self.model_config.get("V2TP_WQKVG_WEIGHT_MEMCFG")
            if _wqkvg_ds_memcfg is not None:
                self.wqkvg_dram_sharded = ttnn.as_tensor(
                    wqkvg_T,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=_wqkvg_ds_memcfg,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape
                    ),
                    cache_file_name=cache_name("wqkvg_tp2d_row_col_dram_sharded"),
                )
            else:
                self.wqkvg_dram_sharded = None

            # 3. WO weight: 2D shard mirroring llama70b.
            #    Native: [H=5120, n_q*hd=6144]
            #    2D shard: dim 2 (input n_q*hd=6144) on rows (8-way) → 768 per chip
            #              dim 3 (output H=5120) on cols (4-way) → 1280 per chip
            wo_native = self.state_dict[wo_str]
            expected_wo = (H, n_q * hd)
            assert wo_native.shape == expected_wo, f"o_proj.weight: expected {expected_wo}, got {wo_native.shape}"
            wo_T = wo_native.T.contiguous().unsqueeze(0).unsqueeze(0)  # [1, 1, 6144, 5120]
            self.wo_proj_weight_shape = tuple(wo_native.shape)
            # Prefill consumes DRAM_INTERLEAVED; decode uses the DRAM-sharded copy.
            self.wo = ttnn.as_tensor(
                wo_T,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(2, 3), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wo_tp2d_row_col"),
            )
            self.wo_interleaved = self.wo
            # V2-DRAM-P1: secondary DRAM-width-sharded WO for decode fast-path.
            _wo_ds_memcfg = self.model_config.get("V2TP_WO_WEIGHT_MEMCFG")
            if _wo_ds_memcfg is not None:
                self.wo_dram_sharded = ttnn.as_tensor(
                    wo_T,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=_wo_ds_memcfg,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(2, 3), mesh_shape=configuration.cluster_shape
                    ),
                    cache_file_name=cache_name("wo_tp2d_row_col_dram_sharded"),
                )
            else:
                self.wo_dram_sharded = None

            # qwen3.6 has its own forward path; no need for the fused-AG matmul
            # config that the 70B-prefetcher path relies on.
            self.use_fused_all_gather_matmul = False
        else:
            assert configuration.qkv_size % self.num_devices_per_group == 0
            assert configuration.dim % self.num_devices_per_group == 0
            # wqkv: 4096 x 3072 (2 devices): width-sharded on 12 banks, 3072 over 12 banks.
            wqkv_mem_config = configuration.create_dram_sharded_mem_config(
                configuration.dim, configuration.qkv_size // configuration.num_devices
            )

            qkv_list = []
            for i in range(self.num_devices_per_group):
                # Chunk weights
                wq_selected = torch.chunk(self.state_dict[wq_str], self.num_devices_per_group, dim=0)[i]
                wk_selected = torch.chunk(self.state_dict[wk_str], self.num_devices_per_group, dim=0)[i]
                wv_selected = torch.chunk(self.state_dict[wv_str], self.num_devices_per_group, dim=0)[i]

                # Transpose the selected chunks
                wq = torch.transpose(wq_selected, -2, -1)
                wk = torch.transpose(wk_selected, -2, -1)
                wv = torch.transpose(wv_selected, -2, -1)

                qkv = torch.cat([wq, wk, wv], dim=-1)
                qkv_list.append(qkv)

            qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

            # Ring stuff
            # Llama3: 9216, 12288
            # Qwen3: 6144, 12288

            # Llama3: [1, 1, 8192, 10240] -> [2304, 1536]
            # Qwen3: [1, 1, 5120, 10240] -> [1280, 1536]
            self.wqkv = ttnn.as_tensor(
                qkv_cat,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=self.model_config["SHARDED_QKV_RING_MEMCFG"],
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wqkv_sharded_2d_prefetcher"),  ## TODO: Fix caching
            )
            self.wqkv_interleaved = ttnn.as_tensor(
                qkv_cat,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wqkv_sharded_2d_dram"),  ## TODO: Fix caching
            )

            # For ring topology we can use all gather matmul for wo
            self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
            pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

            wo_mem_config = configuration.create_dram_sharded_mem_config(
                configuration.dim // configuration.num_devices, configuration.dim
            )

            self.wo = ttnn.as_tensor(
                pt_wo,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=self.model_config["SHARDED_WO_RING_MEMCFG"],
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(2, 3),
                    mesh_shape=configuration.cluster_shape,
                ),
                cache_file_name=cache_name("wo_width_sharded_2d_prefetcher"),
            )
            self.wo_interleaved = ttnn.as_tensor(
                pt_wo,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(2, 3),
                    mesh_shape=configuration.cluster_shape,
                ),
                cache_file_name=cache_name("wo_width_sharded_2d_dram"),
            )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        self.scale = self.head_dim**-0.5
        # qwen3.6 v2 has ``use_prefetcher=False`` and uses ``self.wqkvg`` (not
        # ``self.wqkv``); skip the prefetcher hook so we do not pin tensors that
        # the prefetcher path is not configured to consume.
        if tt_ccl is not None and tt_ccl.mode == "decode" and not self.is_qwen36:
            self.prefetch(prefetcher_setup, tt_ccl)

        # If we are using qk_norm, we need to add a layer norm to the q and k
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialize QK norm if weights are present in state_dict
        if f"{q_norm_str}.weight" in self.state_dict:
            self.qk_norm = True

            if self.is_qwen36:
                # qwen3.6 QK-norm: per-head RMSNorm of head_dim=256, applied to
                # the flat [B, T, n*hd] post-projection tensor via slicing (see
                # ``_qwen36_qknorm_flat_to_heads``).  The +1 zero-centered shift
                # is baked into the on-device weight via ``_build_qwen36_qknorm_weight``
                # so the forward path uses an identical ``ttnn.rms_norm`` call.
                self.qk_norm_eps = configuration.norm_eps
                self.q_norm_w = self._build_qwen36_qknorm_weight(
                    self.state_dict[q_norm_str + ".weight"], add_unit_offset=self.zero_centered_norm
                )
                self.k_norm_w = self._build_qwen36_qknorm_weight(
                    self.state_dict[k_norm_str + ".weight"], add_unit_offset=self.zero_centered_norm
                )
                # Keep the un-baked CPU weights for any host-side audits.
                self.q_norm_weight = self.state_dict[q_norm_str + ".weight"]
                self.k_norm_weight = self.state_dict[k_norm_str + ".weight"]
                # Bind sentinels so the 70B-style attribute names still resolve.
                self.q_norm = None
                self.k_norm = None
                # V2-9: persistent decode-mask device buffer so the SDPA mask
                # never re-allocates inside the trace boundary.
                # Shape: ``[max_batch_size, 1, 1, kv_pad]`` where kv_pad is the
                # worst-case tile-aligned KV length.  We slice this down to
                # the per-call ``T_kv_pad`` width on each decode step.
                # Refresh happens via ``_update_decode_mask_buf`` (called from
                # the model's ``prepare_inputs_decode`` BEFORE trace replay),
                # using the preallocated host tensor + ``copy_host_to_device_tensor``.
                self._decode_mask_kv_pad = ((self.max_seq_len + _QWEN36_TILE - 1) // _QWEN36_TILE) * _QWEN36_TILE
                _mask_init = torch.full(
                    (self.max_batch_size, 1, 1, self._decode_mask_kv_pad),
                    float("-inf"),
                    dtype=torch.bfloat16,
                )
                _replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
                self._decode_mask_host = ttnn.from_torch(
                    _mask_init,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=_replicate,
                )
                self._decode_mask_buf = ttnn.from_torch(
                    _mask_init,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=_replicate,
                )
                return

            # Memory configurations for QK norm
            self.reshape_intermediate_q_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 128),  # [1, 8, 8 (32), 128] ==> *[1, 1, 64, 128]* ==> [1, 1, 64, 32 * 4 = 128]
                core_grid=ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))
                    ]  # This captures the fact that we are using 1 core (height sharded)
                ),  # resharding tensor to 1 core
                strategy=ttnn.ShardStrategy.HEIGHT,  # Literally stating to the device to perform height sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.reshape_intermediate_k_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 128),  # [1, 8, 8 (32), 128] ==> *[1, 1, 64, 128]* ==> [1, 1, 64, 32 * 4 = 128]
                core_grid=ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0))
                    ]  # This captures the fact that we are using 1 core (height sharded)
                ),  # resharding tensor to 1 core
                strategy=ttnn.ShardStrategy.HEIGHT,  # Literally stating to the device to perform height sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.reshape_output_q_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 32),  # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> *[1, 1, 64, 32 * 4 = 128]*
                core_grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))]
                ),  # resharding tensor to cores
                strategy=ttnn.ShardStrategy.WIDTH,  # Literally stating to the device to perform width sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.reshape_output_k_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 32),  # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> *[1, 1, 64, 32 * 4 = 128]*
                core_grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 3))]
                ),  # resharding tensor to cores
                strategy=ttnn.ShardStrategy.WIDTH,  # Literally stating to the device to perform width sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Program configuration for norm
            block_w = 128 // 4 // 32
            # Find largest value <= 4 that evenly divides block_w
            subblock_w = 1
            while subblock_w > 0:
                if block_w % subblock_w == 0:
                    break
                subblock_w -= 1
            self.norm_program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[2, 2],
                subblock_w=subblock_w,
                block_h=2,  # 64 // 32
                block_w=block_w,
                inplace=False,
            )

            # Create Q norm
            self.q_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                state_dict=self.state_dict,
                state_dict_prefix=None,
                weight_dtype=ttnn.bfloat16,
                weight_key=q_norm_str,
                sharded_program_config=self.norm_program_cfg,
                sharded_output_config=self.reshape_output_q_mem_cfg,
            )

            # Create K norm
            self.k_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                state_dict=self.state_dict,
                state_dict_prefix=None,
                weight_dtype=ttnn.bfloat16,
                weight_key=k_norm_str,
                sharded_program_config=self.norm_program_cfg,
                sharded_output_config=self.reshape_output_k_mem_cfg,
            )

            self.q_norm_weight = self.state_dict[q_norm_str + ".weight"]
            self.k_norm_weight = self.state_dict[k_norm_str + ".weight"]

        else:
            self.qk_norm = False

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        # V2-9: qwen3.6 v2 has use_prefetcher=False and uses ``self.wqkvg``
        # instead of the 70B-style ``self.wqkv`` / ``self.wkv`` split. The
        # prefetcher path is disabled at the model level (NoOpPrefetcherSetup),
        # but ``TtTransformer.switch_mode`` still walks
        # ``self.attention.prefetch`` on every layer (including the qwen3.6
        # full_attention layers). Skip the insert_tensor calls here so we
        # don't AttributeError on ``self.wqkv``.
        if tt_ccl.mode == "decode" and not self.is_qwen36:
            self.prefetcher_setup.insert_tensor(self.wqkv)
            self.prefetcher_setup.insert_tensor(self.wo)
        self.tt_ccl = tt_ccl

    def _build_qwen36_qknorm_weight(self, weight_torch, add_unit_offset: bool):
        """Upload a qwen3.6 per-head RMSNorm weight as ``[1, hd//32, 32]``.

        Mirrors v1's ``_make_qknorm_weight_tt``.  The 3-D layout satisfies
        ``ttnn.rms_norm``'s constraint that ``gamma.physical_volume / tile_width
        == input.padded_shape[-1] / tile_width`` for any input with last dim
        equal to ``hd``.  When ``add_unit_offset=True`` we bake ``w' = w + 1``
        into the weight before upload (zero-centered convention used by
        Qwen3NextRMSNorm).
        """
        w = weight_torch.float()
        if add_unit_offset:
            w = w + 1.0
        dim = w.numel()
        assert dim % _QWEN36_TILE == 0, f"head_dim={dim} must be tile-aligned"
        w_3d = w.reshape(1, dim // _QWEN36_TILE, _QWEN36_TILE)
        return ttnn.from_torch(
            w_3d,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _update_decode_mask_buf(self, cur_pos_int: int):
        """V2-9: refresh ``self._decode_mask_buf`` in place for position ``cur_pos_int``.

        Builds a host-side bf16 tensor with zeros at valid KV positions
        [0, cur_pos_int+1) and -inf at padding positions [cur_pos_int+1, kv_pad),
        then copies into the persistent device buffer via
        ``ttnn.copy_host_to_device_tensor`` (trace-safe metadata write — does
        NOT allocate or trigger a host-write op inside the trace boundary).

        Must be called OUTSIDE the trace boundary (e.g. from
        ``TtTransformer.prepare_inputs_decode``).  The buffer is then sliced
        down to the per-call ``[B, 1, 1, T_kv_pad]`` width inside the forward.
        """
        if not getattr(self, "is_qwen36", False):
            return
        if not hasattr(self, "_decode_mask_buf"):
            return
        kv_pad = self._decode_mask_kv_pad
        T_kv = cur_pos_int + 1
        mask_t = torch.zeros(self.max_batch_size, 1, 1, kv_pad, dtype=torch.bfloat16)
        if T_kv < kv_pad:
            mask_t[:, :, :, T_kv:] = float("-inf")
        host = ttnn.from_torch(
            mask_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self._decode_mask_buf)

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Generates empty KV cache and pushed to device memory
        """
        if self.is_qwen36:
            # V2-TP: row-sharded cache for 2D-TP:
            #   non-paged shape per chip = [max_batch_size, n_kv_pc=1, max_seq, hd]
            #   paged    shape per chip = [max_num_blocks, n_kv_pc=1, block, hd]
            # n_kv padded 4 → 8; sharded on dim=1 (n_kv) across rows (cluster_axis=0).
            # Cols replicate the KV head (matches post-WQKVG+col-reduce layout).
            row_shard_kv = ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(1, None), mesh_shape=configuration.cluster_shape
            )
            n_kv_full = self.n_kv_heads  # 8 (padded)
            if self.paged_attention_config:
                cache_k = torch.zeros(
                    self.paged_attention_config.max_num_blocks,
                    n_kv_full,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
                cache_v = torch.zeros(
                    self.paged_attention_config.max_num_blocks,
                    n_kv_full,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            else:
                cache_k = torch.zeros(
                    self.max_batch_size,
                    n_kv_full,
                    self.max_seq_len,
                    self.head_dim,
                )
                cache_v = torch.zeros(
                    self.max_batch_size,
                    n_kv_full,
                    self.max_seq_len,
                    self.head_dim,
                )
            # qwen3.6 fills the KV cache with bfloat16 (k_rot/v_t come from
            # the QKVG linear at bfloat16 dtype); bake the cache in bfloat16
            # so ttnn.fill_cache / paged_fill_cache don't trip the
            # "input and cache must have same dtype" check.
            # QWEN36_KV_BF8=1 allocates the cache as bfloat8_b (halves the
            # SDPA KV read); the write paths typecast k_rot/v_t to match.
            _kv_dtype = _qwen36_kv_cache_dtype()
            self.layer_past = [
                ttnn.from_torch(
                    kv,
                    device=self.mesh_device,
                    dtype=_kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=row_shard_kv,
                )
                for kv in [cache_k, cache_v]
            ]
            return

        if self.paged_attention_config:
            cache_k = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
        else:
            cache_k = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                if weight_cache_path and not configuration.dummy_weights
                else None,
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """
        if self.is_qwen36:
            return self._forward_decode_qwen36(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)
        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        xqkv_fused_sharded = ttnn.matmul(  # [1, 1, 32, 1280]
            x,  # [1, 1, 32, 1280]
            self.wqkv,
            program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
            memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            dtype=ttnn.bfloat16,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id,
        )
        ttnn.deallocate(x)
        # xqkv_fused_sharded -> [1, 1, 32, 12288 // 8]

        ###
        # Reshape and rotary embeddings
        ###
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = self.tt_ccl.llama_rs_create_heads(
            xqkv_fused_sharded,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            dim=3,
            qkv_memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )

        if self.qk_norm:
            rm_mem_cfg_q = q_heads_pre_rot_1BQD.memory_config()
            rm_mem_cfg_k = k_heads_pre_rot_1BKD.memory_config()

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                q_heads_pre_rot_1BQD, memory_config=self.reshape_intermediate_q_mem_cfg
            )
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                k_heads_pre_rot_1BKD, memory_config=self.reshape_intermediate_k_mem_cfg
            )

            # Reshape and prepare tensors for QK norm
            q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, [1, 1, 64, 128])  # [1, 8, 8, 128] => [1, 1, 64, 128]
            k_heads_pre_rot_1BKD = ttnn.view(
                k_heads_pre_rot_1BKD, [1, 1, 64, 128]
            )  # [1, 8, 1 (8), 128]] => [1, 1, 64, 128]

            q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.TILE_LAYOUT)
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.TILE_LAYOUT)

            q_heads_intermediate_after_reshape_mem_cfg = q_heads_pre_rot_1BQD.memory_config()
            k_heads_intermediate_after_reshape_mem_cfg = k_heads_pre_rot_1BKD.memory_config()

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                q_heads_pre_rot_1BQD, memory_config=self.reshape_output_q_mem_cfg
            )
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                k_heads_pre_rot_1BKD, memory_config=self.reshape_output_k_mem_cfg
            )

            # Apply QK norm
            q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode="decode", in_sharded=True, out_sharded=True)
            k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode="decode", in_sharded=True, out_sharded=True)

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                q_heads_pre_rot_1BQD, memory_config=q_heads_intermediate_after_reshape_mem_cfg
            )
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                k_heads_pre_rot_1BKD, memory_config=k_heads_intermediate_after_reshape_mem_cfg
            )

            q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.ROW_MAJOR_LAYOUT)
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.ROW_MAJOR_LAYOUT)

            q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, [1, 8, 8, 128])
            k_heads_pre_rot_1BKD = ttnn.view(k_heads_pre_rot_1BKD, [1, 8, 8, 128])  # ==> [1, 8, 1 (8), 128]

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=rm_mem_cfg_q)
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=rm_mem_cfg_k)

        # print("done create qkv heads")
        ttnn.deallocate(xqkv_fused_sharded)

        # Q, K Rotary Embeddings
        q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
        )  # [1, 8, 8, 128], [1, 8, 8, 128]
        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)
        # print("done rotary embeddings")

        ###
        # KV update
        ###
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
        # v_heads [seqlen, n_kv_heads, bsz, head_dim]
        # keys, [max_batch_size, n_kv_heads // configuration.num_devices, max_seq_len, head_dim]
        ttnn.experimental.paged_fused_update_cache(
            keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
        )

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        # print("done update cache")
        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)
        sdpa_out_mem_cfg = self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group)
        if page_table:
            attn_output_1G4D_sharded = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                page_table_tensor=page_table,
                scale=self.scale,
                program_config=self.model_config["PAGED_SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=sdpa_out_mem_cfg,
            )
        else:
            attn_output_1G4D_sharded = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=sdpa_out_mem_cfg,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
            )

        ttnn.deallocate(q_heads_1BQD)

        attn_output_cat = self.tt_ccl.all_gather_concat(  # [1, 1, 32, 1024]
            attn_output_1G4D_sharded,
            dim=1,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_1G4D_sharded)
        # print("done concat heads")

        # Original matmul on each device [1, 1, 32, 1024] @ [1, 1, 1024, 2048]
        dense_out_ttnn = ttnn.matmul(  # [1, 1, 32, 1280]
            attn_output_cat,
            self.wo,
            program_config=self.model_config["WO_DECODE_RING_PROGCFG"],
            memory_config=self.model_config["SHARDED_WO_OUT_RING_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            dtype=ttnn.bfloat8_b,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id,
        )
        # [1, 1, 32, 2304]
        dense_out_reduced = self.tt_ccl.line_all_reduce(  # [1, 1, 32, 1280]
            dense_out_ttnn,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )
        ttnn.deallocate(dense_out_ttnn)

        # print("done all reduce")

        return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        kv_cache=None,
        batch_size=1,
    ):
        if self.is_qwen36:
            return self._forward_prefill_qwen36(
                x_11SH,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
        if batch_size > 1:
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 2048, 2048, -1])

        xqkv = ttnn.linear(
            x_11SH,
            self.wqkv_interleaved,
            dtype=self.ccl_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )
        # Minimal matmul is giving bad outputs for seqlen > 128
        # xqkv = ttnn.experimental.minimal_matmul(
        #     input_tensor=x_11SH,
        #     weight_tensor=self.wqkv_interleaved,
        #     config=self.model_config["XQKV_PREFILL_MINIMAL_PROGCFG"](seq_len),
        #     compute_kernel_config=self.compute_kernel_config_hifi2,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )

        ttnn.deallocate(x_11SH)

        xqkv_fused = self.tt_ccl.line_all_reduce(
            xqkv,
            cluster_axis=1,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="QKV",
            batch_size=batch_size,
        )
        ttnn.deallocate(xqkv)

        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        if batch_size > 1:
            xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            q_heads_1QSD_pre_rot_bf8 = q_heads_1QSD_pre_rot
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(q_heads_1QSD_pre_rot_bf8)

        if self.qk_norm:
            q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            k_heads_1KSD_pre_rot_bf8 = k_heads_1KSD_pre_rot
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(k_heads_1KSD_pre_rot_bf8)

        if self.qk_norm:
            k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode="prefill")

        # k_heads_1KSD = k_heads_1KSD_pre_rot
        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        # Fill KV-Cache
        if kv_cache:
            keys_BKSD, values_BKSD = kv_cache[0], kv_cache[1]
        else:
            keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(k_heads_1KSD)

        k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)

        ttnn.deallocate(v_heads_1VSD)

        v_fill = v_heads_1VSD_8b

        if batch_size > 1:
            k_fill = ttnn.reshape(k_fill, [1, 1, seq_len, -1])
            v_fill = ttnn.reshape(v_fill, [1, 1, seq_len, -1])

        if not page_table:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)

        if page_table:
            # Use chunk_page_table only for prefix-cached prefill (chunk_start_idx > 0).
            # For non-prefix prefill, ignore chunk_page_table (trace may pass a dummy) and use page_table.
            use_chunk_for_fill = chunk_start_idx is not None and chunk_start_idx > 0
            fill_page_table = chunk_page_table if (use_chunk_for_fill and chunk_page_table is not None) else page_table

            # Each shard gets one row, which is locally at index 0
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill, fill_page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill, fill_page_table, batch_idx=0)

        else:
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id % self.batch_size_per_device_group,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id % self.batch_size_per_device_group,
            )

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        # Run ring_distributed_sdpa for > 1k seqlen because we are seeing worse perf for <=1k seqlen as compared to regular SDPA
        # ring_distributed_sdpa needs seqlen//8 to be atleast one tile (32)
        # Disabled for non-ring topology (e.g. BH GLX with FABRIC_1D) as ring all-gather is not supported
        ring_distributed_sdpa = (
            seq_len > 1024
            and batch_size == 1
            and (chunk_start_idx is None or chunk_start_idx == 0)
            and self.model_config["CCL_TOPOLOGY"] == ttnn.Topology.Ring
        )
        use_chunked_sdpa = chunk_start_idx is not None and chunk_start_idx > 0

        if ring_distributed_sdpa:
            attn_output_1QSD = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                ring_size=4,  # Number of devices in the ring topology (4 devices per row in 8x4 mesh)
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len, chunk_start_idx=0),
                page_table=None,
                chunk_start_idx=None,
            )
        else:
            # When using prefix caching (chunk_start_idx provided), use chunked SDPA with KV cache tensors.
            # Flexible path: chunk_start_idx_tensor so one trace works for any chunk_start at replay.
            if use_chunked_sdpa:
                assert page_table is not None, "page_table must be provided for prefix caching"
                assert (
                    chunk_start_idx_tensor is not None
                ), "prefix caching requires chunk_start_idx_tensor for flexible SDPA"
                page_size = self.paged_attention_config.block_size if self.paged_attention_config else 32
                attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                    input_tensor_q=q_heads_1QSD_8b,
                    input_tensor_k=keys_BKSD,
                    input_tensor_v=values_BKSD,
                    page_table_tensor=page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                    program_config=self.model_config["SDPA_PROGCFG_FLEXIBLE_CHUNK"](seq_len, page_size),
                )

                # Replicate active column's data to all columns for correct RMSNORM behavior.
                # Chunked SDPA writes only to the column for this user_id; we zero others and all-reduce so every column has the same output.
                # Pre-computed column_mask: [1, 1, 1, 32] per device, 1.0 on owning column, 0.0 on others.
                # Stored on tt_ccl by the generator before forward; slice to scalar for broadcast.
                column_mask = self.tt_ccl._prefill_column_mask
                mask = ttnn.slice(column_mask, [0, 0, 0, 0], [1, 1, 1, 1])
                # attn_output_84SD: zero out inactive columns (multiply by 0); active column unchanged (multiply by 1).
                attn_output_84SD = ttnn.multiply(attn_output_84SD, mask)
                # line_all_reduce along columns: sum = active column's data (others 0); replicate to all columns so shape/values match for downstream.
                attn_output_84SD = self.tt_ccl.line_all_reduce(
                    attn_output_84SD,
                    cluster_axis=1,
                    num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    buffer_key="ATTN_REPLICATE",
                )

                # Reshape from [1, 1, seq_len, head_dim] to [1, n_local_heads, seq_len, head_dim]
                attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])
            else:
                attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
                    q_heads_1QSD_8b,
                    k_heads_1KSD_8b,
                    v_heads_1VSD_8b,
                    is_causal=True,
                    scale=self.scale,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                    program_config=self.model_config["SDPA_PROGCFG"](
                        seq_len // batch_size if seq_len // batch_size == 128 else seq_len
                    ),
                )
        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        if ring_distributed_sdpa:
            # Split the attention output into two chunks along the sequence dimension for ring all-gather
            # 4 = ring_size (number of devices in ring), 2 = number of chunks per device
            attn_output_11SH_chunks = ttnn.split(attn_output_11SH, seq_len // 4 // 2, dim=2)
            attn_output_11SH.deallocate(True)

            # Perform ring all-gather on the first chunk (normal order)
            attn_output_11SH_chunk_0 = self.tt_ccl.ring_all_gather(
                attn_output_11SH_chunks[0],
                dim=2,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SDPA",
            )
            attn_output_11SH_chunks[0].deallocate(True)

            # Perform ring all-gather on the second chunk (reverse order)
            attn_output_11SH_chunk_1 = self.tt_ccl.ring_all_gather(
                attn_output_11SH_chunks[1],
                dim=2,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                reverse_order=True,
                buffer_key="SDPA_REVERSE",
            )
            attn_output_11SH_chunks[1].deallocate(True)

            # Concatenate the gathered chunks along the sequence dimension to form the final output
            attn_output_11SH = ttnn.concat([attn_output_11SH_chunk_0, attn_output_11SH_chunk_1], dim=2)
        if batch_size > 1:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 1, seq_len, -1])

        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        ## For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096 or batch_size > 1:
            output_11SH = ttnn.linear(
                attn_output_11SH,
                self.wo_interleaved,
                compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
            )
        else:
            output_11SH = ttnn.experimental.minimal_matmul(
                input_tensor=attn_output_11SH,
                weight_tensor=self.wo_interleaved,
                config=self.model_config["WO_PREFILL_MINIMAL_PROGCFG"](seq_len),
                compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce-scatter
        output_11SH_reduced = self.tt_ccl.line_all_reduce(
            output_11SH,
            cluster_axis=0,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="WO_AG" if seq_len <= 4096 else "WO",
        )
        output_11SH.deallocate()

        return output_11SH_reduced

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        kv_cache=None,
        batch_size=1,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
        else:
            return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        tensor_copy = ttnn.clone(key_or_value_layer)
        # key_or_value_layer.deallocate(True)
        # Get all tensors from multi-device tensor
        tensors = ttnn.get_device_tensors(tensor_copy)
        # Get only tensors from specific column chips
        # Get every 4th tensor starting from user_id // 8
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: 4]
        # Create multi-device tensor
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)

        return multi_device_tensor

    # ------------------------------------------------------------------
    # qwen3.6-specific forward paths (ported from v1 PCC-verified
    # ``models/demos/qwen3_6_galaxy/tt/llama_attention.py``).
    #
    # Order of operations (matches v1 + task spec):
    #     QKVG matmul → split into Q / Gate / K / V
    #     → per-head QK-norm on Q, K (zero-centered weight baked at __init__)
    #     → partial RoPE on Q, K (rope_dim=64 of head_dim=256)
    #     → KV-cache fill
    #     → SDPA(Q, K, V)
    #     → sigmoid(Gate) * attn_out
    #     → WO projection (col-row-parallel, all-reduce across cols)
    # ------------------------------------------------------------------
    def _forward_prefill_qwen36(
        self,
        x,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        kv_cache=None,
        batch_size: int = 1,
    ):
        """qwen3.6 prefill forward — 2D-TP variant.

        Input  : x col-sharded ``[B, T, H/cols=1280]`` per chip
        Output : col-sharded ``[B, T, H/cols=1280]`` per chip
        Steps  :
          1. WQKVG matmul (col-input × row-output) → partial [B, T, 2048] per chip
          2. AllReduce on cols (cluster_axis=1, 4-way) to complete inner-dim sum
          3. Split Q/Gate/K/V (per chip: 3 Q, 3 G, 1 K, 1 V; head_dim=256)
          4. QK-norm + partial RoPE
          5. KV cache fill (per chip: 1 KV head)
          6. GQA expand 1→3, SDPA
          7. Output gate (sigmoid(gate) * attn)
          8. WO matmul (row-input × col-output) → partial [B, T, 1280] per chip
          9. AllReduce on rows (cluster_axis=0, 8-way) to complete head sum
        """
        cos_tt, sin_tt = rot_mats

        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H_per_chip = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H_per_chip])
        else:
            B, T, H_per_chip = orig_shape
            x_3d = x

        hd = self.head_dim
        n_q_pc = self.n_q_per_chip  # 3
        n_kv_pc = self.n_kv_per_chip  # 1
        q_dim_pc = self.q_dim_per_chip  # 768
        g_dim_pc = self.gate_dim_per_chip  # 768
        k_dim_pc = self.k_dim_per_chip  # 256
        total_pc = self.total_per_chip  # 2048

        # 1. QKVG projection (2D-sharded weight, partial output [B, T, 2048] per chip).
        xqkvg_partial = ttnn.linear(
            x_3d,
            self.wqkvg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        if len(list(xqkvg_partial.shape)) == 4:
            _, _, _T_q, _N_q = list(xqkvg_partial.shape)
            xqkvg_partial = ttnn.reshape(xqkvg_partial, [B, _T_q, _N_q])

        # 2. AllReduce on cols (cluster_axis=1, 4-way) to complete the input-dim sum.
        xqkvg_g = ttnn.all_gather(
            xqkvg_partial,
            dim=0,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkvg_partial.deallocate(True)
        xqkvg = ttnn.experimental.fast_reduce_nc(
            xqkvg_g, dims=[0], output=None, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xqkvg_g.deallocate(True)

        # 3. Split Q / Gate / K / V (per chip).
        q_flat = ttnn.slice(xqkvg, [0, 0, 0], [B, T, q_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc], [B, T, q_dim_pc + g_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_flat = ttnn.slice(
            xqkvg,
            [0, 0, q_dim_pc + g_dim_pc],
            [B, T, q_dim_pc + g_dim_pc + k_dim_pc],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc + g_dim_pc + k_dim_pc], [B, T, total_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xqkvg.deallocate(True)

        # 4. QK-norm (per-head, zero-centered weight pre-baked).
        q_rot_pre = _qwen36_qknorm_flat_to_heads(
            q_flat, self.q_norm_w, self.qk_norm_eps, B, n_q_pc, T, hd, self.compute_kernel_config_hifi4
        )
        k_rot_pre = _qwen36_qknorm_flat_to_heads(
            k_flat, self.k_norm_w, self.qk_norm_eps, B, n_kv_pc, T, hd, self.compute_kernel_config_hifi4
        )
        q_flat.deallocate(True)
        k_flat.deallocate(True)

        # V: [B, T, n_kv_pc*hd] → [B, n_kv_pc, T, hd] (no norm for V).
        v_t = _qwen36_flat_to_heads(v_flat, B, n_kv_pc, T, hd)
        v_flat.deallocate(True)

        # 5. Partial RoPE on Q, K.
        rope_setup = getattr(self, "rope_setup", None)
        if rope_setup is None:
            raise AttributeError(
                "TtLlamaAttention(is_qwen36=True) needs ``self.rope_setup`` set by the "
                "decoder (V2-decoder wave). Forward path cannot apply partial RoPE without it."
            )
        q_rot = rope_setup.partial_rope_apply(q_rot_pre, cos_tt, sin_tt)
        k_rot = rope_setup.partial_rope_apply(k_rot_pre, cos_tt, sin_tt)
        q_rot_pre.deallocate(True)
        k_rot_pre.deallocate(True)

        # 6. KV cache fill (per chip, n_kv_pc=1).
        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]
        # KV cache fill. Under QWEN36_KV_BF8 the cache is bfloat8_b; the
        # paged_fill_cache / fill_cache kernels accept bf16 producers into a
        # bf8 cache (the kernel quantizes on write — see fill_cache device op
        # dtype assert: input fp32/bf16 OR cache bf8/bf4), so no producer cast.
        if page_table is not None:
            ttnn.experimental.paged_fill_cache(keys_cache, k_rot, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_cache, v_t, page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(keys_cache, k_rot, user_id % max(self.max_batch_size, 1))
            ttnn.fill_cache(values_cache, v_t, user_id % max(self.max_batch_size, 1))

        # 7. GQA expand K, V (per chip): n_kv_pc=1 → n_q_pc=3.
        gqa_pc = n_q_pc // n_kv_pc
        k_exp = ttnn.repeat_interleave(k_rot, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_exp = ttnn.repeat_interleave(v_t, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        # 8. SDPA with causal mask (per chip — each chip attends to its row's KV head).
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_exp,
            v_exp,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k_exp.deallocate(True)
        v_exp.deallocate(True)
        q_rot.deallocate(True)

        # 9. Output gate: sigmoid(Gate) * attn_out (per chip, pre-WO).
        attn_flat = _qwen36_heads_to_flat(attn_out, B, n_q_pc, T, hd)
        attn_out.deallocate(True)
        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat.deallocate(True)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_flat.deallocate(True)
        gate_sig.deallocate(True)

        # 10. WO projection (2D-sharded weight, partial [B, T, H/cols=1280] per chip).
        dense_partial = ttnn.linear(
            gated,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        gated.deallocate(True)
        if len(list(dense_partial.shape)) == 4:
            _, _, _T_o, _N_o = list(dense_partial.shape)
            dense_partial = ttnn.reshape(dense_partial, [B, _T_o, _N_o])
        # 11. AllReduce on rows (cluster_axis=0, 8-way) to complete head sum.
        gathered = ttnn.all_gather(
            dense_partial,
            dim=0,
            num_links=1,
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dense_partial.deallocate(True)
        dense_out = ttnn.experimental.fast_reduce_nc(
            gathered, dims=[0], output=None, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gathered.deallocate(True)
        # Output is col-sharded H/cols=1280 per chip (matches input contract).
        return dense_out

    def _forward_decode_qwen36(
        self,
        x,
        current_pos,
        rot_mats,
        page_table=None,
        kv_cache=None,
    ):
        """qwen3.6 decode forward — 2D-TP variant.

        Input  : x col-sharded ``[B, T, H/cols=1280]`` per chip
        Output : col-sharded ``[B, T, H/cols=1280]`` per chip
        Heads on rows (8-way): 3 Q, 1 KV per chip; hidden on cols (4-way).
        WO reduce uses cluster_axis=0 (8-way) instead of cluster_axis=1 (4-way).
        """
        cos_tt, sin_tt = rot_mats

        # V2-CONFIG-E: optionally use bf8 output dtype for the FA WO matmul
        # (matches llama70b WO at line 567 — `dtype=ttnn.bfloat8_b`). WQKVG
        # stays bf16 to match llama70b WQKV (line 413). DN out_proj also
        # respects this env var via a parallel branch in
        # `qwen36_delta_attention.py`. Weights stay bf16 (V2-7b PCC
        # constraint); only the OUTPUT is bf8 so downstream CCL transfers
        # half the bytes through the ring.
        _attn_out_dtype = ttnn.bfloat8_b if os.environ.get("QWEN36_ATTN_OUT_BF8", "0") == "1" else ttnn.bfloat16

        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
            x_3d = x

        # Decode expects T=1; tile padding may show T=32 — slice back if needed.
        T_logical = 1
        if T > T_logical:
            x_3d = ttnn.slice(x_3d, [0, 0, 0], [B, T_logical, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            T = T_logical

        hd = self.head_dim
        n_q_pc = self.n_q_per_chip  # 3
        n_kv_pc = self.n_kv_per_chip  # 1
        q_dim_pc = self.q_dim_per_chip  # 768
        g_dim_pc = self.gate_dim_per_chip  # 768
        k_dim_pc = self.k_dim_per_chip  # 256
        total_pc = self.total_per_chip  # 2048

        # 1. QKVG projection (V2-DRAM-P1: DRAM-sharded matmul fast path).
        #    Activation L1-width-sharded → matmul with DRAM-sharded weight +
        #    program_config → output L1-width-sharded → back to DRAM for the
        #    downstream col-axis all_reduce.
        _wqkvg_progcfg = self.model_config.get("V2TP_WQKVG_PROGCFG")
        _wqkvg_act_memcfg = self.model_config.get("V2TP_WQKVG_ACT_MEMCFG")
        _wqkvg_weight = getattr(self, "wqkvg_dram_sharded", None)
        if _wqkvg_progcfg is not None and _wqkvg_act_memcfg is not None and _wqkvg_weight is not None:
            x_sharded = ttnn.to_memory_config(x_3d, _wqkvg_act_memcfg)
            xqkvg_partial_l1 = ttnn.linear(
                x_sharded,
                _wqkvg_weight,
                dtype=ttnn.bfloat16,  # matches llama70b WQKV (bf16 output, line 413)
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=_wqkvg_progcfg,
                compute_kernel_config=self.compute_kernel_config_hifi4,
            )
            ttnn.deallocate(x_sharded)
            xqkvg_partial = ttnn.to_memory_config(xqkvg_partial_l1, ttnn.DRAM_MEMORY_CONFIG)
            xqkvg_partial_l1.deallocate(True)
        else:
            # Fallback to slow path (no program_config) if the V2-DRAM memcfgs
            # weren't built (e.g. construction-time tests with mocked configs).
            xqkvg_partial = ttnn.linear(
                x_3d,
                self.wqkvg,
                dtype=ttnn.bfloat16,  # matches llama70b WQKV
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi4,
            )
        if len(list(xqkvg_partial.shape)) == 4:
            _, _, _T_q, _N_q = list(xqkvg_partial.shape)
            xqkvg_partial = ttnn.reshape(xqkvg_partial, [B, _T_q, _N_q])

        # 2. AllReduce on cols (cluster_axis=1, 4-way) to complete input-dim sum.
        xqkvg = ttnn.all_reduce(
            xqkvg_partial,
            cluster_axis=1,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkvg_partial.deallocate(True)

        q_flat = ttnn.slice(xqkvg, [0, 0, 0], [B, T, q_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc], [B, T, q_dim_pc + g_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_flat = ttnn.slice(
            xqkvg,
            [0, 0, q_dim_pc + g_dim_pc],
            [B, T, q_dim_pc + g_dim_pc + k_dim_pc],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc + g_dim_pc + k_dim_pc], [B, T, total_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xqkvg.deallocate(True)

        q_normed = _qwen36_qknorm_flat_to_heads(
            q_flat, self.q_norm_w, self.qk_norm_eps, B, n_q_pc, T, hd, self.compute_kernel_config_hifi4
        )
        k_normed = _qwen36_qknorm_flat_to_heads(
            k_flat, self.k_norm_w, self.qk_norm_eps, B, n_kv_pc, T, hd, self.compute_kernel_config_hifi4
        )
        q_flat.deallocate(True)
        k_flat.deallocate(True)
        v_t = _qwen36_flat_to_heads(v_flat, B, n_kv_pc, T, hd)
        v_flat.deallocate(True)

        rope_setup = getattr(self, "rope_setup", None)
        if rope_setup is None:
            raise AttributeError("TtLlamaAttention(is_qwen36=True) needs ``self.rope_setup`` set by the decoder.")
        q_rot = rope_setup.partial_rope_apply(q_normed, cos_tt, sin_tt)
        k_rot = rope_setup.partial_rope_apply(k_normed, cos_tt, sin_tt)
        q_normed.deallocate(True)
        k_normed.deallocate(True)

        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]

        if page_table is not None:
            # V2-9 trace-default: ``paged_update_cache`` requires the input
            # (k_rot/v_t) to be HEIGHT_SHARDED on 1 core with shard_shape
            # [tile_rows, hd]. k_rot/v_t arrive as DRAM-INTERLEAVED from
            # ``_qwen36_qknorm_flat_to_heads`` / ``_qwen36_flat_to_heads``.
            # Mirror v1's height-shard step (qwen3_6_galaxy/tt/llama_attention.py
            # lines 1156-1180) — convert to height-sharded before the call.
            tile_rows = ((T + _QWEN36_TILE - 1) // _QWEN36_TILE) * _QWEN36_TILE  # 32
            _height_shard_cfg = ttnn.create_sharded_memory_config(
                shape=[tile_rows, hd],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            k_rot_sharded = ttnn.to_memory_config(k_rot, memory_config=_height_shard_cfg)
            v_t_sharded = ttnn.to_memory_config(v_t, memory_config=_height_shard_cfg)
            # paged_update_cache REQUIRES a fp32/bf16 producer (it rejects bf8
            # inputs — see paged_update_cache device op dtype assert). The bf8
            # cache (QWEN36_KV_BF8) is written from this bf16 producer; the
            # kernel quantizes on write, matching llama70b's decode path which
            # feeds a bf16 rotary output into its bf8 cache. Do NOT cast here.
            ttnn.experimental.paged_update_cache(
                keys_cache, k_rot_sharded, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values_cache, v_t_sharded, update_idxs_tensor=current_pos, page_table=page_table
            )
            k_rot_sharded.deallocate(True)
            v_t_sharded.deallocate(True)
        else:
            # Non-paged decode requires a Python-int position.
            if isinstance(current_pos, int):
                _pos = current_pos
            else:
                _pos = int(
                    ttnn.to_torch(current_pos, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0].item()
                )
            # update_cache requires fp32/bf16 producer; the bf8 cache is
            # written from the bf16 producer (kernel quantizes on write).
            ttnn.update_cache(keys_cache, k_rot, _pos, batch_offset=0)
            ttnn.update_cache(values_cache, v_t, _pos, batch_offset=0)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        if page_table is not None:
            # Paged SDPA decode expects q: [1, B, n_q_pc, hd]
            q_1bnd = ttnn.permute(q_rot, (2, 0, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q_rot.deallocate(True)
            # V2-11 (lever F): tried bumping SDPA decode grid (1,1) → (4,1) /
            # (8,8) — both ran at the same wall-clock (~74.03 ms / step, no
            # measurable speedup) but produced lower-quality text (alpha
            # chars dropped 118 → 34 over 32 generated tokens despite the
            # compile-pass token still matching 248068=<think>). Reverted
            # to (1,1) — qwen3.6's single-user decode has q: [1, B=1,
            # n_q_pc=8, hd=128] which is below the multi-core SDPA's break-
            # even point. The decode kernel's K-chunk schedule already
            # fits in a single tile read at decode-time so the grid bump
            # adds no compute throughput; the multi-core reduction order
            # also subtly changes the bf8-quantized softmax outputs.
            # QWEN36_SDPA_MULTICORE (default "0"): the long-context payoff — run the
            # paged SDPA decode multi-core (8,6)=48 cores + sub_core_grids instead of
            # single-core (1,1). The (1,1) kernel is KV-bandwidth-bound: it reads the
            # ENTIRE KV cache for all 16 full-attn layers on one core per step, which
            # dominates long-context decode wall-clock.
            #
            # A correctness probe (tests/test_sdpa_decode_grid.py, the
            # ``multi_8x6_subcore_shardout`` variant) proved this kernel is coherent
            # at qwen3.6 dims (head_dim=256, n_q_pc=3, GQA 3:1, batch=1): PCC 0.9995
            # vs single-core and 4-8× faster (gap widens with ctx). The KEY is that
            # sub_core_grids ASSERTS a *sharded* output (is_q_sharded||is_output_
            # sharded) — a plain (8,6)/DRAM-output integration (V2-11) instead
            # silently produced the right shape but garbled the downstream
            # _qwen36_heads_to_flat consumer (alpha 118→34). So we MUST give the op a
            # height-sharded output, then convert it back to DRAM so the existing
            # permute → heads_to_flat → gate → WO path is byte-for-byte identical to
            # the single-core path. We reuse the model_config's PAGED_SDPA_DECODE_
            # PROGCFG (already (8,6)+sub_core_grids(48)) and SCORES_BATCHED_MM_OUTPUT_
            # MEMCFG (height-sharded [ceil(n_local_heads/32)*32, hd] on B cores) —
            # the exact (progcfg, sharded-memcfg) pair the probe validated.
            #
            # 128k RESOLUTION: an earlier integration garbled at 128k with the HiFi4 +
            # fp32_dest_acc_en compute config (mojibake), which we wrongly attributed to
            # "fidelity compounding." Diffing against llama70b (coherent at batch-1 128k
            # multi-core) found the real cause: the SDPA compute_kernel_config. HiFi4 +
            # fp32_dest_acc_en corrupts the multi-core cross-core flash-decode combine at
            # the 131072-deep softmax. Using llama70b's validated decode config
            # (SDPA_DECODE_COMPUTE_PROGCFG = HiFi2, fp32_dest_acc_en=False) — see below —
            # makes multi-core coherent at every context, so the flag is now default-on.
            # QWEN36_SDPA_MULTICORE (default "1"): multi-core paged SDPA decode
            # ((8,6)+sub_core_grids(48)) vs single-core (1,1). It MUST pair with the
            # HiFi2/no-fp32 decode compute config (SDPA_DECODE_COMPUTE_PROGCFG —
            # llama70b's validated one): HiFi4+fp32_dest_acc_en corrupts the cross-core
            # flash-decode combine at the 128k-deep softmax (128k mojibake — the
            # qwen-specific bug vs llama70b, which uses this config and is coherent at
            # batch-1 128k). With the decode config, multi-core is coherent at every
            # context (ISL-128 -> 248068; 64k & 128k coherent) and 2.2x faster decode at
            # 128k (8.3 -> 18.3 tok/s/user). Set QWEN36_SDPA_MULTICORE=0 to fall back to
            # single-core (1,1)+HiFi4 (DRAM output).
            _sdpa_compute_cfg = self.compute_kernel_config_hifi4
            if os.environ.get("QWEN36_SDPA_MULTICORE", "1") == "1":
                _sdpa_progcfg = self.model_config["PAGED_SDPA_DECODE_PROGCFG"]
                _sdpa_out_memcfg = self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](
                    self.batch_size_per_device_group
                )
                _sdpa_compute_cfg = self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"]
            else:
                _sdpa_progcfg = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(1, 1),
                    exp_approx_mode=False,
                    q_chunk_size=0,
                    k_chunk_size=0,
                )
                _sdpa_out_memcfg = ttnn.DRAM_MEMORY_CONFIG
            attn_out_1bnd = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_1bnd,
                keys_cache,
                values_cache,
                page_table,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                program_config=_sdpa_progcfg,
                compute_kernel_config=_sdpa_compute_cfg,
                memory_config=_sdpa_out_memcfg,
            )
            q_1bnd.deallocate(True)
            # Multi-core path returns a height-sharded [pad(n_q_pc), hd] L1 tensor;
            # convert back to DRAM-interleaved so the downstream permute/heads_to_flat
            # sees the identical layout as the single-core (1,1) DRAM output. The
            # probe confirmed the sharded→DRAM readout matches torch at PCC 0.9995.
            if attn_out_1bnd.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                _attn_dram = ttnn.to_memory_config(attn_out_1bnd, ttnn.DRAM_MEMORY_CONFIG)
                attn_out_1bnd.deallocate(True)
                attn_out_1bnd = _attn_dram
            attn_out = ttnn.permute(attn_out_1bnd, (1, 2, 0, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_out_1bnd.deallocate(True)
        else:
            # Non-paged: slice KV cache up to current_pos, GQA-expand, SDPA with explicit mask.
            # V2-decode-64L: tile-padding rows ``T_kv..T_kv_pad`` contain zeros (cache
            # was zero-initialized). Without a mask, ``is_causal=False`` SDPA softmax
            # treats those zero-key positions as score=0 and gives them ``exp(0)/Z``
            # of the probability mass — diluting the real attention by ~31/(Z_real+31).
            # With 16 full-attention layers in 64L decode, the dilution compounds and
            # drops logits PCC from 0.9996 (4L, 1 full-attn) → 0.16 (64L, 16 full-attn).
            # v1's ``_forward_decode_qwen36`` builds an explicit mask with ``-inf`` on
            # positions ``T_kv..T_kv_pad`` (see qwen3_6_galaxy/tt/llama_attention.py
            # lines 1267–1299).  Mirror that here.
            T_kv = (current_pos if isinstance(current_pos, int) else 0) + 1
            T_kv_pad = ((T_kv + _QWEN36_TILE - 1) // _QWEN36_TILE) * _QWEN36_TILE
            k_cached = ttnn.slice(
                keys_cache, [0, 0, 0, 0], [B, n_kv_pc, T_kv_pad, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            v_cached = ttnn.slice(
                values_cache, [0, 0, 0, 0], [B, n_kv_pc, T_kv_pad, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            gqa_pc = n_q_pc // n_kv_pc
            k_exp = ttnn.repeat_interleave(k_cached, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_exp = ttnn.repeat_interleave(v_cached, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k_cached.deallocate(True)
            v_cached.deallocate(True)
            # Build decode mask for the tile-padded suffix [T_kv:T_kv_pad].
            # V2-9: slice the persistent ``_decode_mask_buf`` (allocated at
            # ``__init__`` time, refreshed OUTSIDE the trace boundary via
            # ``TtTransformer.refresh_decode_per_step_buffers`` or
            # ``_update_decode_mask_buf``).  ``copy_host_to_device_tensor``
            # IS a host write under trace capture (hits
            # ``enqueue_write_mesh_buffer`` → ``TT_FATAL`` in fd_mesh_command_queue.cpp:476);
            # the slice itself is a pure device op so it is trace-safe.
            #
            # For the eager-only path (when no caller has primed the buffer)
            # we still refresh inline — that path is NOT executed inside a
            # trace boundary so the host write is allowed.
            decode_mask_tt = None
            if T_kv_pad > T_kv:
                if hasattr(self, "_decode_mask_buf"):
                    # If a caller (Generator / test) has refreshed the buffer
                    # for the current cur_pos before begin_trace_capture, the
                    # inline refresh below is a no-op metadata-write that
                    # CANNOT happen inside a trace boundary — refresh only when
                    # ``current_pos`` is a Python int (the eager forward path).
                    # Tests that capture a trace must call
                    # ``model.refresh_decode_per_step_buffers(cur_pos)`` (or
                    # equivalent) BEFORE ``begin_trace_capture``.
                    if isinstance(current_pos, int) and not getattr(self, "_skip_decode_mask_refresh", False):
                        self._update_decode_mask_buf(current_pos)
                    decode_mask_tt = ttnn.slice(
                        self._decode_mask_buf,
                        [0, 0, 0, 0],
                        [B, 1, T, T_kv_pad],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    # Fallback to the v1 in-forward build for any path that
                    # bypassed ``_build_decode_buffers`` (should not happen
                    # for qwen3.6 v2; kept for safety).
                    import torch as _torch_for_mask

                    mask_host = _torch_for_mask.zeros(B, 1, T, T_kv_pad, dtype=_torch_for_mask.bfloat16)
                    mask_host[:, :, :, T_kv:] = float("-inf")
                    decode_mask_tt = ttnn.from_torch(
                        mask_host,
                        device=self.mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    )
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_rot,
                k_exp,
                v_exp,
                is_causal=False,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                attn_mask=decode_mask_tt,
            )
            if decode_mask_tt is not None:
                # V2-9: ``ttnn.slice`` of a persistent DRAM tensor produces a
                # NEW tensor (separate allocation, copied data) — safe to
                # deallocate.  The fallback ``from_torch`` path also produces
                # a fresh tensor.  Either way the persistent
                # ``_decode_mask_buf`` is untouched (its lifetime is bound to
                # ``self``, not this slice).
                decode_mask_tt.deallocate(True)
            k_exp.deallocate(True)
            v_exp.deallocate(True)
            q_rot.deallocate(True)

        # Output gate.
        # V2-11 (lever D, full-attention variant): attempted to fuse
        # sigmoid(gate_flat) into the multiply via input_tensor_b_activations.
        # Coherency held in 4L tests but degraded across 16 full-attention
        # layers when combined with DeltaNet's per-layer compounding.
        #
        # V2-CONFIG-G: re-attempt the fusion under the PCC-is-data policy.
        # Env-gated so we can A/B between fused and split.
        attn_flat = _qwen36_heads_to_flat(attn_out, B, n_q_pc, T, hd)
        attn_out.deallocate(True)
        if os.environ.get("QWEN36_GATE_FUSE", "0") == "1":
            gated = ttnn.multiply(
                attn_flat,
                gate_flat,
                input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            attn_flat.deallocate(True)
            gate_flat.deallocate(True)
        else:
            gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate_flat.deallocate(True)
            gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_flat.deallocate(True)
            gate_sig.deallocate(True)

        # V2-TP / V2-DRAM-P1: WO projection via DRAM-sharded matmul + all-reduce
        # on rows (cluster_axis=0, 8-way ring). Per chip: gated [B, T, 768] ×
        # wo [768, 1280] = partial [B, T, 1280]; row-axis all_reduce completes
        # the head sum → result col-sharded H/4.
        #
        # V2-CCL-P1: optionally swap the post-WO ``ttnn.all_reduce`` for the
        # persistent-buffer ``tt_ccl.line_all_reduce(use_optimal_ccl_for_llama=True)``,
        # mirroring DeltaNet's ``_output_proj_and_reduce`` LAR path. Gated by
        # env var so we can toggle for bisecting. Needs
        # ``qwen36_residual_buffers[0]`` (cluster_axis=0 ring=8 reduction)
        # built by ``_build_qwen36_residual_buffers``; that builder auto-enables
        # when ``QWEN36_FULLATTN_WO_LAR=1`` so the user only needs the one flag.
        _use_wo_lar = (
            os.environ.get("QWEN36_FULLATTN_WO_LAR", "0") == "1"
            and getattr(self.tt_ccl, "qwen36_residual_buffers", [None, None])[0] is not None
        )
        try:
            _wo_num_links = int(
                os.environ.get(
                    "QWEN36_CCL_NUM_LINKS_WO",
                    os.environ.get("QWEN36_CCL_NUM_LINKS", "1"),
                )
            )
        except ValueError:
            _wo_num_links = 1

        if _use_wo_lar:
            # LAR path: matmul writes directly to width-sharded L1; line_all_reduce
            # consumes width-sharded input and produces width-sharded output.
            sharded_memcfg = self.tt_ccl.qwen36_residual_output_memcfgs[0]
            dense_partial_sharded = ttnn.linear(
                gated,
                self.wo,
                dtype=_attn_out_dtype,
                memory_config=sharded_memcfg,
                compute_kernel_config=self.compute_kernel_config_hifi4,
            )
            gated.deallocate(True)
            dense_out_full = self.tt_ccl.line_all_reduce(
                dense_partial_sharded,
                cluster_axis=0,
                num_links=_wo_num_links,
                memory_config=sharded_memcfg,
                use_optimal_ccl_for_llama=True,
                use_qwen36_residual_buffer=True,
            )
            dense_partial_sharded.deallocate(True)
        else:
            _wo_progcfg = self.model_config.get("V2TP_WO_PROGCFG")
            _wo_act_memcfg = self.model_config.get("V2TP_WO_ACT_MEMCFG")
            _wo_weight = getattr(self, "wo_dram_sharded", None)
            if _wo_progcfg is not None and _wo_act_memcfg is not None and _wo_weight is not None:
                gated_sharded = ttnn.to_memory_config(gated, _wo_act_memcfg)
                gated.deallocate(True)
                dense_partial_l1 = ttnn.linear(
                    gated_sharded,
                    _wo_weight,
                    dtype=_attn_out_dtype,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=_wo_progcfg,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                )
                ttnn.deallocate(gated_sharded)
                dense_partial = ttnn.to_memory_config(dense_partial_l1, ttnn.DRAM_MEMORY_CONFIG)
                dense_partial_l1.deallocate(True)
            else:
                dense_partial = ttnn.linear(
                    gated,
                    self.wo,
                    dtype=_attn_out_dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                )
                gated.deallocate(True)
            # Diagnostic (V2-CCL-P1b): optional host-side barrier RIGHT BEFORE
            # the WO axis-0 all_reduce, to test whether chip-level work
            # imbalance in the FA pre-RS chain (paged_update_cache + SDPA +
            # gate + matmul) is what drags the RS kernel duration bimodal.
            # Tracy on 1L FA decode shows 1280×BF16 RS at 666 µs mean (max
            # 2785 µs) with bimodal per-chip distribution — but DN's identical-
            # shape RS runs at 64 µs uniform. Forcing a barrier here lets us
            # see whether removing chip skew restores the fast RS.
            #
            # WARNING: this is a HOST synchronize — breaks trace mode. Eager
            # only. Env-gated so the trace path is untouched by default.
            if os.environ.get("QWEN36_FULLATTN_PRE_WO_RS_BARRIER", "0") == "1":
                ttnn.synchronize_device(self.mesh_device)
            dense_out_full = ttnn.all_reduce(
                dense_partial,
                cluster_axis=0,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            dense_partial.deallocate(True)

        if len(list(dense_out_full.shape)) == 4:
            _shape = list(dense_out_full.shape)
            dense_out_full = ttnn.reshape(
                dense_out_full, [_shape[0], _shape[-2], _shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        out_T = list(dense_out_full.shape)[-2]
        if out_T != T_logical:
            dense_out = ttnn.slice(dense_out_full, [0, 0, 0], [B, T_logical, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            dense_out_full.deallocate(True)
        else:
            dense_out = dense_out_full
        return dense_out
