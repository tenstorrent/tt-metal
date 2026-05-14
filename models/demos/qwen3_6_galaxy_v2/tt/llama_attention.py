# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm

# qwen3.6 partial-RoPE tile size (head_dim must be tile-aligned for rms_norm).
_QWEN36_TILE = 32


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

    Slice each head's [B, T, hd] sub-tensor, run ``ttnn.rms_norm`` with a
    [1, hd//32, 32] zero-centered weight, then concatenate along a new
    head dimension.  Avoids the [B, T, n_heads, hd] intermediate that has
    n_heads < 32 and would tile-pad to 32 in the second-to-last position.
    Mirrors v1's ``_qknorm_flat_to_heads`` byte-for-byte.
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

    head_normed_list = []
    head_tensors = []
    for h in range(n_heads):
        head_h = ttnn.slice(x_flat, [0, 0, h * hd], [B, T, (h + 1) * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_normed = ttnn.rms_norm(
            head_h,
            weight=weight,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        head_h.deallocate(True)
        head_normed_4d = ttnn.reshape(head_normed, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_normed_list.append(head_normed)
        head_tensors.append(head_normed_4d)

    out = ttnn.concat(head_tensors, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in head_tensors:
        t.deallocate(True)
    for t in head_normed_list:
        t.deallocate(True)
    return out


def _qwen36_flat_to_heads(x_flat, B: int, n_heads: int, T: int, hd: int):
    """Reshape [B, T, n_heads*hd] → [B, n_heads, T, hd] via slice+reshape+concat.

    Mirrors v1's ``_flat_to_heads`` (avoids tile-padding for n_heads<32).
    """
    if n_heads == 1:
        view_4d = ttnn.reshape(x_flat, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_4d.deallocate(True)
        return out

    slice_list = []
    head_tensors = []
    for h in range(n_heads):
        head_h = ttnn.slice(x_flat, [0, 0, h * hd], [B, T, (h + 1) * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_h_4d = ttnn.reshape(head_h, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        slice_list.append(head_h)
        head_tensors.append(head_h_4d)

    out = ttnn.concat(head_tensors, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in head_tensors:
        t.deallocate(True)
    for s in slice_list:
        s.deallocate(True)
    return out


def _qwen36_heads_to_flat(x_heads, B: int, n_heads: int, T: int, hd: int):
    """Reshape [B, n_heads, T, hd] → [B, T, n_heads*hd] via slice+reshape+concat.

    Reverses ``_qwen36_flat_to_heads``.  Mirrors v1's ``_heads_to_flat``.
    """
    if n_heads == 1:
        view_3d = ttnn.reshape(x_heads, [B, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_3d.deallocate(True)
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
        self.dtype = ttnn.bfloat16 if getattr(configuration, "is_qwen36", False) else dtype
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
            hd = self.head_dim  # 256
            n_q = self.n_heads  # 24
            n_kv = self.n_kv_heads  # 4
            H = self.hidden_size  # 5120
            n_cols = configuration.cluster_shape[1]  # 4
            assert n_q % n_cols == 0, f"n_q={n_q} not divisible by n_cols={n_cols}"
            assert n_kv % n_cols == 0, f"n_kv={n_kv} not divisible by n_cols={n_cols}"
            n_q_per_col = n_q // n_cols  # 6
            n_kv_per_col = n_kv // n_cols  # 1
            self.n_q_per_col = n_q_per_col
            self.n_kv_per_col = n_kv_per_col
            self.q_dim_per_col = n_q_per_col * hd  # 1536
            self.gate_dim_per_col = n_q_per_col * hd  # 1536
            self.k_dim_per_col = n_kv_per_col * hd  # 256
            self.v_dim_per_col = n_kv_per_col * hd  # 256
            self.total_per_col = (
                self.q_dim_per_col + self.gate_dim_per_col + self.k_dim_per_col + self.v_dim_per_col
            )  # 3584
            # Total fused QKVG output width across all cols (matches the CCL
            # buffer key ``QKV`` shape (1, 1, seqlen, 3584) per chip × 4 cols
            # = 14336 on the host-side concat).
            self.qkvg_total_width = self.total_per_col * n_cols  # 14336

            # 1. De-interleave Q and gate from q_proj.weight.
            q_proj_w = self.state_dict[wq_str]  # [12288, 5120]
            expected_q = (n_q * 2 * hd, H)
            assert q_proj_w.shape == expected_q, f"q_proj.weight: expected {expected_q}, got {q_proj_w.shape}"
            q_2hd = q_proj_w.reshape(n_q, 2, hd, H)
            wq_native = q_2hd[:, 0, :, :].reshape(n_q * hd, H)  # [6144, 5120]
            wgate_native = q_2hd[:, 1, :, :].reshape(n_q * hd, H)  # [6144, 5120]
            wk_native = self.state_dict[wk_str]  # [1024, 5120]
            wv_native = self.state_dict[wv_str]  # [1024, 5120]
            # Test-introspection attributes (sanity-checked by V2-4 unit test).
            self.q_proj_weight_shape = tuple(wq_native.shape)
            self.gate_proj_weight_shape = tuple(wgate_native.shape)
            self.k_proj_weight_shape = tuple(wk_native.shape)
            self.v_proj_weight_shape = tuple(wv_native.shape)

            # 2. Build the col-sharded QKVG by interleaving HEAD GROUPS per col.
            col_blocks = []
            for c in range(n_cols):
                qs = c * n_q_per_col * hd
                qe = (c + 1) * n_q_per_col * hd
                ks = c * n_kv_per_col * hd
                ke = (c + 1) * n_kv_per_col * hd
                col_blocks.append(
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
            wqkvg = torch.cat(col_blocks, dim=0)  # [14336, H]
            wqkvg_T = wqkvg.T.contiguous().unsqueeze(0).unsqueeze(0)  # [1, 1, H, 14336]

            # Column-parallel along the output dim: shard tensor dim=3 across
            # mesh cols (cluster_axis=1).  Rows are replicated (the per-col
            # forward path operates entirely within one column group).
            self.wqkvg = ttnn.as_tensor(
                wqkvg_T,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 3), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wqkvg_col_sharded"),
            )
            # Keep ``wqkv_interleaved`` as the same buffer so any downstream code
            # path that grabs ``self.wqkv_interleaved`` still works.  qwen3.6
            # does not have a separate ring memcfg.
            self.wqkv_interleaved = self.wqkvg

            # 3. WO weight: row-parallel — shard the INPUT dim (n_q*hd=6144)
            #    across cols.  For col c, rows that match col c's Q heads
            #    occupy [c*n_q_per_col*hd, (c+1)*n_q_per_col*hd] of the input dim.
            wo_native = self.state_dict[wo_str]  # [H=5120, n_q*hd=6144]
            expected_wo = (H, n_q * hd)
            assert wo_native.shape == expected_wo, f"o_proj.weight: expected {expected_wo}, got {wo_native.shape}"
            wo_T = wo_native.T.contiguous().unsqueeze(0).unsqueeze(0)  # [1, 1, 6144, 5120]
            self.wo_proj_weight_shape = tuple(wo_native.shape)
            self.wo = ttnn.as_tensor(
                wo_T,
                dtype=self.dtype,  # BF16 — qwen3.6 does not quantize wo
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 2), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wo_row_sharded"),
            )
            self.wo_interleaved = self.wo

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
        if tt_ccl.mode == "decode":
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

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Generates empty KV cache and pushed to device memory
        """
        if self.is_qwen36:
            # qwen3.6 col-sharded cache (mirrors v1 ``_build_kv_cache``):
            #   non-paged shape per chip = [max_batch_size, n_kv_pc=1, max_seq, hd]
            #   paged    shape per chip = [max_num_blocks, n_kv_pc=1, block, hd]
            # Sharded on dim=1 (n_kv) across cluster_axis=1 (cols).
            col_shard_kv = ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(None, 1), mesh_shape=configuration.cluster_shape
            )
            n_kv_full = self.n_kv_heads  # 4
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
            self.layer_past = [
                ttnn.from_torch(
                    kv,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=col_shard_kv,
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
        """qwen3.6 prefill forward (see v1 ``forward_prefill``)."""
        cos_tt, sin_tt = rot_mats

        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
            x_3d = x

        hd = self.head_dim
        n_q_pc = self.n_q_per_col
        n_kv_pc = self.n_kv_per_col
        q_dim_pc = self.q_dim_per_col
        g_dim_pc = self.gate_dim_per_col
        k_dim_pc = self.k_dim_per_col
        total_pc = self.total_per_col

        # 1. QKVG projection (col-sharded weights, per-col output [B, T, 3584]).
        xqkvg = ttnn.linear(
            x_3d,
            self.wqkvg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        # ttnn.linear of rank-3 LHS @ rank-4 weight yields a rank-4 output; the
        # subsequent slices use rank-3 begins/ends, so collapse the leading
        # singleton(s) back to rank-3 before slicing. The wqkvg weight is uploaded
        # with shape [1, 1, H, 14336]; the per-device matmul preserves both
        # leading singletons.  V2-7 PCC test surfaced this — the original v1
        # path uploaded wqkvg as rank-2.
        if len(list(xqkvg.shape)) == 4:
            _, _, _T_q, _N_q = list(xqkvg.shape)
            xqkvg = ttnn.reshape(xqkvg, [B, _T_q, _N_q])

        # 2. Split Q / Gate / K / V (per-col).
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

        # 3. QK-norm (per-head, zero-centered weight pre-baked).
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

        # 4. Partial RoPE on Q, K.
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

        # 5. KV cache fill (per-col, n_kv_pc=1).
        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]
        if page_table is not None:
            ttnn.experimental.paged_fill_cache(keys_cache, k_rot, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_cache, v_t, page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(keys_cache, k_rot, user_id % max(self.max_batch_size, 1))
            ttnn.fill_cache(values_cache, v_t, user_id % max(self.max_batch_size, 1))

        # 6. GQA expand K, V (per col): n_kv_pc=1 → n_q_pc=6.
        gqa_pc = n_q_pc // n_kv_pc
        k_exp = ttnn.repeat_interleave(k_rot, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_exp = ttnn.repeat_interleave(v_t, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        # 7. SDPA with causal mask (per col — each col attends to its own KV head).
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

        # 8. Output gate: sigmoid(Gate) * attn_out (per col, pre-WO).
        attn_flat = _qwen36_heads_to_flat(attn_out, B, n_q_pc, T, hd)
        attn_out.deallocate(True)
        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat.deallocate(True)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_flat.deallocate(True)
        gate_sig.deallocate(True)

        # 9. WO projection (row-parallel by input dim across cols) + all-reduce.
        dense_partial = ttnn.linear(
            gated,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        gated.deallocate(True)
        gathered = ttnn.all_gather(
            dense_partial,
            dim=0,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dense_partial.deallocate(True)
        dense_out = ttnn.experimental.fast_reduce_nc(
            gathered, dims=[0], output=None, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gathered.deallocate(True)
        return dense_out

    def _forward_decode_qwen36(
        self,
        x,
        current_pos,
        rot_mats,
        page_table=None,
        kv_cache=None,
    ):
        """qwen3.6 decode forward (see v1 ``forward_decode``)."""
        cos_tt, sin_tt = rot_mats

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
        n_q_pc = self.n_q_per_col
        n_kv_pc = self.n_kv_per_col
        q_dim_pc = self.q_dim_per_col
        g_dim_pc = self.gate_dim_per_col
        k_dim_pc = self.k_dim_per_col
        total_pc = self.total_per_col

        xqkvg = ttnn.linear(
            x_3d,
            self.wqkvg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

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
            ttnn.experimental.paged_update_cache(
                keys_cache, k_rot, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values_cache, v_t, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            # Non-paged decode requires a Python-int position.
            if isinstance(current_pos, int):
                _pos = current_pos
            else:
                _pos = int(
                    ttnn.to_torch(current_pos, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0].item()
                )
            ttnn.update_cache(keys_cache, k_rot, _pos, batch_offset=0)
            ttnn.update_cache(values_cache, v_t, _pos, batch_offset=0)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        if page_table is not None:
            # Paged SDPA decode expects q: [1, B, n_q_pc, hd]
            q_1bnd = ttnn.permute(q_rot, (2, 0, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q_rot.deallocate(True)
            paged_sdpa_prog_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(1, 1),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )
            attn_out_1bnd = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_1bnd,
                keys_cache,
                values_cache,
                page_table,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                program_config=paged_sdpa_prog_cfg,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_1bnd.deallocate(True)
            attn_out = ttnn.permute(attn_out_1bnd, (1, 2, 0, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_out_1bnd.deallocate(True)
        else:
            # Non-paged: slice KV cache up to current_pos, GQA-expand, SDPA with explicit mask.
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
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_rot,
                k_exp,
                v_exp,
                is_causal=False,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k_exp.deallocate(True)
            v_exp.deallocate(True)
            q_rot.deallocate(True)

        # Output gate.
        attn_flat = _qwen36_heads_to_flat(attn_out, B, n_q_pc, T, hd)
        attn_out.deallocate(True)
        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat.deallocate(True)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_flat.deallocate(True)
        gate_sig.deallocate(True)

        # WO projection + all-reduce across cols.
        dense_partial = ttnn.linear(
            gated,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        gated.deallocate(True)
        gathered = ttnn.all_gather(
            dense_partial,
            dim=0,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dense_partial.deallocate(True)
        dense_out_full = ttnn.experimental.fast_reduce_nc(
            gathered, dims=[0], output=None, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gathered.deallocate(True)

        out_T = list(dense_out_full.shape)[-2]
        if out_T != T_logical:
            dense_out = ttnn.slice(dense_out_full, [0, 0, 0], [B, T_logical, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            dense_out_full.deallocate(True)
        else:
            dense_out = dense_out_full
        return dense_out
