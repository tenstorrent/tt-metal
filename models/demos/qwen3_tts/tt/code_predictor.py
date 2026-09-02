# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Code Predictor — fp32 activation path, fully on-device, trace-compatible.

Mirrors production CodePredictor API but every op is ttnn (no host roundtrips
inside forward_single_step). Activations and KV cache are fp32 throughout;
weights stay bf16 (PCC neutral). RoPE casts Q/K to bf16 (kernel is bf16-only;
rotation only — small precision cost).

Lifetime contract: forward_single_step does NOT deallocate the caller's
`inputs_embeds`; per-layer residuals owned by forward_single_step are freed
after each layer. ttnn.slice was avoided in favor of running lm_head over
the full hidden state — caller indexes the last position.
"""

import os
from typing import List, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.attention import prepare_fused_sdpa_mask
from models.demos.qwen3_tts.tt.rope import apply_rope_qk, get_decode_transformation_mat, shard_decode_rope_tables


class CodePredictor(LightweightModule):
    def __init__(
        self,
        device,
        config,
        talker_hidden_size: int,
        state_dict: dict,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.hidden_size = config.hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_code_groups = config.num_code_groups
        self.vocab_size = config.vocab_size
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.rms_norm_eps = config.rms_norm_eps
        # TP-aware head counts: divide by tp_size so each chip owns its head slice.
        _is_mesh = device.__class__.__name__ == "MeshDevice"
        from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size, is_n150, is_n300

        self.tp_size = get_tp_size(device) if _is_mesh else 1
        # N150-only DRAM-sharded QKV / o_proj / gate-up blocking + fused SDPA.
        # is_n150() matches Talker: Wormhole B0 and a single chip (plain Device
        # or 1×1 mesh). N300 / T3K / Blackhole keep the previous factories.
        self._n150 = is_n150(device)
        # N300-only fast path for the CP layer. The CodePredictor runs 15 forward
        # passes per audio frame (prefill seq=2 + 13 residual decodes) x 5 layers, so
        # it dominates the AR frame, yet its layer was built from generic ops: the
        # hidden-size norms land on 1 core, nlp_create/concat_heads on 1 core, and each
        # layer pays two full reduce_scatter+all_gather rounds. The Talker already has
        # sharded equivalents (see attention.py / decoder_layer.py); this ports them to
        # the CP for the 2-chip wormhole grid only. Set QWEN3_TTS_CP_N300_OPT=0 to A/B.
        self._n300_cp_opt = is_n300(device) and os.environ.get("QWEN3_TTS_CP_N300_OPT", "1") != "0"
        self.num_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = config.num_key_value_heads // self.tp_size
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # same ratio
        self.scale = 1.0 / (self.head_dim**0.5)
        # Sharded transformation matrix for the decode-mode RoPE kernel — see
        # rope.apply_rope_qk. Built at init so it predates any trace capture.
        self._decode_trans_mat = get_decode_transformation_mat(device)

        DRAM = ttnn.DRAM_MEMORY_CONFIG
        L1 = ttnn.L1_MEMORY_CONFIG
        TILE = ttnn.TILE_LAYOUT
        ROW = ttnn.ROW_MAJOR_LAYOUT

        # Use bf16 for all activations so DRAM-sharded matmuls can run without
        # fp32<->bf16 typecasts at MLP boundaries. Non-N150 still runs the
        # manual fp32 SDPA chain (QK-norm gain ≈ 68 overflows bf16). N150 uses
        # fused SDPA with HiFi4 + fp32 dest instead.
        self.act_dtype = ttnn.bfloat16
        self.kcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_kcfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # --- weight format helpers ---
        def _perm_rope_rows(w_2d: torch.Tensor, head_dim: int) -> torch.Tensor:
            out_dim = w_2d.shape[0]
            assert out_dim % head_dim == 0
            nh = out_dim // head_dim
            half = head_dim // 2
            idx = torch.arange(head_dim).view(2, half).t().reshape(-1)
            w = w_2d.view(nh, head_dim, w_2d.shape[1])
            w = w[:, idx, :].contiguous()
            return w.view(out_dim, w_2d.shape[1])

        def _perm_rope_vec(v_1d: torch.Tensor, head_dim: int) -> torch.Tensor:
            half = head_dim // 2
            idx = torch.arange(head_dim).view(2, half).t().reshape(-1)
            return v_1d.view(head_dim)[idx].contiguous()

        def w_to_tt(w_2d, dt=ttnn.bfloat16):
            w_host = w_2d.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
            return ttnn.from_torch(w_host, device=device, dtype=dt, layout=TILE, memory_config=DRAM)

        def w_colpar_to_tt(w_2d, dt=ttnn.bfloat16):
            """Column-parallel: split output features across TP chips. TP=1 → replicate."""
            if self.tp_size == 1:
                return w_to_tt(w_2d, dt)
            chunks = list(torch.chunk(w_2d, self.tp_size, dim=0))  # [tp, local_out, in]
            host = torch.stack(chunks, dim=0).transpose(-2, -1).unsqueeze(0).contiguous()  # [1, tp, in, local_out]
            return ttnn.from_torch(
                host,
                device=device,
                dtype=dt,
                layout=TILE,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
            )

        def w_rowpar_to_tt(w_2d, dt=ttnn.bfloat16):
            """Row-parallel: split input features across TP chips. TP=1 → replicate."""
            if self.tp_size == 1:
                return w_to_tt(w_2d, dt)
            w_t = w_2d.transpose(0, 1).contiguous()  # [in, out]
            chunks = list(torch.chunk(w_t, self.tp_size, dim=0))  # [tp, local_in, out]
            host = torch.stack(chunks, dim=0).unsqueeze(0).contiguous()  # [1, tp, local_in, out]
            return ttnn.from_torch(
                host,
                device=device,
                dtype=dt,
                layout=TILE,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
            )

        def norm_w_1d_to_tt(w_1d, dim, *, permute_rope=False):
            w = _perm_rope_vec(w_1d, dim) if permute_rope else w_1d
            w_host = w.to(torch.bfloat16).view(1, 1, dim // 32, 32).contiguous()
            return ttnn.from_torch(w_host, device=device, dtype=ttnn.bfloat16, layout=ROW, memory_config=DRAM)

        # input projection
        proj_key = "talker.code_predictor.small_to_mtp_projection.weight"
        bias_key = "talker.code_predictor.small_to_mtp_projection.bias"
        self.needs_projection = talker_hidden_size != config.hidden_size
        if self.needs_projection and proj_key in state_dict:
            self.input_proj = w_to_tt(state_dict[proj_key])
            if bias_key in state_dict:
                b = state_dict[bias_key]
                bias_tt = ttnn.from_torch(
                    b.to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ROW,
                    memory_config=DRAM,
                )
                self.input_proj_bias = ttnn.reshape(bias_tt, [1, 1, 1, int(b.shape[0])], memory_config=DRAM)
            else:
                self.input_proj_bias = None
        else:
            self.needs_projection = False
            self.input_proj = None
            self.input_proj_bias = None

        # per-layer weights — fused QKV plain stack [Q | K | V] for the regular
        # (non-DRAM-sharded) nlp_create_qkv_heads kernel. Q/K rows are RoPE-permuted
        # so rotary_embedding_llama's interleaved format works directly.
        H = self.hidden_size
        # Use full head counts for weight construction (config values, not local).
        _NH_full = config.num_attention_heads
        _NKV_full = config.num_key_value_heads
        HD = self.head_dim
        self.layers_w = []
        for li in range(self.num_layers):
            pfx = f"talker.code_predictor.model.layers.{li}."
            lw_torch = {k.replace(pfx, ""): v for k, v in state_dict.items() if k.startswith(pfx)}
            q_w = _perm_rope_rows(lw_torch["self_attn.q_proj.weight"], HD)  # [NH_full*HD, H]
            k_w = _perm_rope_rows(lw_torch["self_attn.k_proj.weight"], HD)  # [NKV_full*HD, H]
            v_w = lw_torch["self_attn.v_proj.weight"]  # [NKV_full*HD, H]
            if self.tp_size == 1:
                wqkv_tt = w_to_tt(torch.cat([q_w, k_w, v_w], dim=0).contiguous())
            else:
                # Column-parallel QKV: split Q, K, V separately by head count.
                q_per_chip = list(torch.chunk(q_w, self.tp_size, dim=0))
                k_per_chip = list(torch.chunk(k_w, self.tp_size, dim=0))
                v_per_chip = list(torch.chunk(v_w, self.tp_size, dim=0))
                per_chip = [
                    torch.cat([q_per_chip[i], k_per_chip[i], v_per_chip[i]], dim=0) for i in range(self.tp_size)
                ]
                stacked = torch.stack(per_chip, dim=0).transpose(-2, -1).unsqueeze(0).contiguous()
                wqkv_tt = ttnn.from_torch(
                    stacked,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=TILE,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
                )
            self.layers_w.append(
                {
                    "input_ln_w": norm_w_1d_to_tt(lw_torch["input_layernorm.weight"], H),
                    "post_ln_w": norm_w_1d_to_tt(lw_torch["post_attention_layernorm.weight"], H),
                    "wqkv": wqkv_tt,
                    "o_proj": w_rowpar_to_tt(lw_torch["self_attn.o_proj.weight"]),
                    "gate": w_colpar_to_tt(lw_torch["mlp.gate_proj.weight"]),
                    "up": w_colpar_to_tt(lw_torch["mlp.up_proj.weight"]),
                    "down": w_rowpar_to_tt(lw_torch["mlp.down_proj.weight"]),
                    "q_norm_w": norm_w_1d_to_tt(lw_torch["self_attn.q_norm.weight"], HD, permute_rope=True),
                    "k_norm_w": norm_w_1d_to_tt(lw_torch["self_attn.k_norm.weight"], HD, permute_rope=True),
                }
            )

        self.final_norm_w = norm_w_1d_to_tt(state_dict["talker.code_predictor.model.norm.weight"], H)

        # === DRAM-sharded MLP program configs (bf16 activation path) ===
        # With act_dtype=bfloat16, gate/up/down can use DRAM-sharded matmul and keep
        # activations in WIDTH_SHARDED L1 throughout the MLP — no I/S reshards needed
        # per matmul, and no fp32 typecasts at MLP boundaries.
        from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
            build_dram_sharded_weight_tp,
            dram_sharded_program_config,
            find_grid_k_n,
            pad_n_for_dram_align,
            width_sharded_l1_memcfg,
        )

        _cg = device.compute_with_storage_grid_size()
        _dram_cores = device.dram_grid_size().x
        _local_intermediate = config.intermediate_size // self.tp_size

        _n_pad_gu = pad_n_for_dram_align(_local_intermediate, _dram_cores)
        _k_tiles_gu = H // 32
        _n_tiles_gu = _n_pad_gu // 32
        # N150-only: cap cores so in0_block_w≥2. Other SKUs keep the unconstrained grid.
        if self._n150:

            def _dram_grid(k_tiles: int, n_tiles: int, min_k_per_core: int = 1):
                """Largest (rows, cols) that divides K and N, with ≥min_k_per_core K-tiles/core.

                min_k_per_core=2 keeps in0_block_w≥2. Unconstrained find_grid_k_n on
                N150 CP gate/up (K=1024) picks 32 cores and in0_block_w=1.
                """
                max_cores = _cg.x * _cg.y
                if min_k_per_core > 1:
                    max_cores = min(max_cores, k_tiles // min_k_per_core)
                candidates = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
                candidates.sort(reverse=True)
                for cores in candidates:
                    for rows in range(1, _cg.y + 1):
                        if cores % rows == 0:
                            cols = cores // rows
                            if cols <= _cg.x:
                                return rows, cols
                return find_grid_k_n(k_tiles, n_tiles, max_rows=_cg.y, max_cols=_cg.x)

            _rows_gu, _cols_gu = _dram_grid(_k_tiles_gu, _n_tiles_gu, min_k_per_core=2)
        else:
            _rows_gu, _cols_gu = find_grid_k_n(_k_tiles_gu, _n_tiles_gu, max_rows=_cg.y, max_cols=_cg.x)
        self._cp_gate_up_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=H, n=_n_pad_gu, num_cores=_rows_gu * _cols_gu
        )
        self._cp_gate_up_in0_memcfg = width_sharded_l1_memcfg(1, _k_tiles_gu, _cols_gu, _rows_gu)
        self._cp_gate_up_out_memcfg = width_sharded_l1_memcfg(1, _n_tiles_gu, _cols_gu, _rows_gu)
        self._cp_gate_up_n_padded = _n_pad_gu

        _n_pad_d = pad_n_for_dram_align(H, _dram_cores)
        _k_tiles_d = _local_intermediate // 32
        _n_tiles_d = _n_pad_d // 32
        _rows_d, _cols_d = find_grid_k_n(_k_tiles_d, _n_tiles_d, max_rows=_cg.y, max_cols=_cg.x)
        self._cp_down_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=_local_intermediate, n=_n_pad_d, num_cores=_rows_d * _cols_d
        )
        self._cp_down_in0_memcfg = width_sharded_l1_memcfg(1, _k_tiles_d, _cols_d, _rows_d)
        self._cp_down_out_memcfg = width_sharded_l1_memcfg(1, _n_tiles_d, _cols_d, _rows_d)
        self._cp_down_n_padded = _n_pad_d

        # N150-only: DRAM-sharded QKV / o_proj + fused-SDPA program config +
        # sharded nlp_create / nlp_concat so the DS QKV output can split in place.
        _n150_qkv_cores = None
        if self._n150:
            _local_hidden = self.num_heads * HD
            _fused_qkv = (self.num_heads + 2 * self.num_kv_heads) * HD
            self._fused_qkv = _fused_qkv
            _k_tiles_q = H // 32
            _n_pad_q = pad_n_for_dram_align(_fused_qkv, _dram_cores)
            _n_tiles_q = _n_pad_q // 32
            _rows_q, _cols_q = _dram_grid(_k_tiles_q, _n_tiles_q)
            self._cp_qkv_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=H, n=_n_pad_q, num_cores=_rows_q * _cols_q
            )
            self._cp_qkv_in0_memcfg = width_sharded_l1_memcfg(1, _k_tiles_q, _cols_q, _rows_q)
            self._cp_qkv_out_memcfg = width_sharded_l1_memcfg(1, _n_tiles_q, _cols_q, _rows_q)
            self._cp_qkv_n_padded = _n_pad_q

            _k_tiles_o = _local_hidden // 32
            _n_pad_o = pad_n_for_dram_align(H, _dram_cores)
            _n_tiles_o = _n_pad_o // 32
            _rows_o, _cols_o = _dram_grid(_k_tiles_o, _n_tiles_o)
            self._cp_wo_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=_local_hidden, n=_n_pad_o, num_cores=_rows_o * _cols_o
            )
            self._cp_wo_in0_memcfg = width_sharded_l1_memcfg(1, _k_tiles_o, _cols_o, _rows_o)
            self._cp_wo_out_memcfg = width_sharded_l1_memcfg(1, _n_tiles_o, _cols_o, _rows_o)
            self._cp_wo_n_padded = _n_pad_o

            self.sdpa_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(_cg.x, _cg.y),
                exp_approx_mode=False,
                q_chunk_size=64,
                k_chunk_size=64,
            )

            # CP M is always one tile (decode seq=1, prefill seq=2).
            _M = 32
            _num_q_per_kv = self.num_heads // self.num_kv_heads
            _qkv_shard_w = (_num_q_per_kv + 2) * HD
            assert _fused_qkv % _qkv_shard_w == 0
            _qkv_cores = _fused_qkv // _qkv_shard_w
            _qkv_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_qkv_cores - 1, 0))})
            _q_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_heads - 1, 0))})
            _concat_grid = ttnn.num_cores_to_corerangeset(self.num_heads, _cg, True)

            def _hs(grid, w):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (_M, w), ttnn.ShardOrientation.ROW_MAJOR),
                )

            def _ws(grid, w):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (_M, w), ttnn.ShardOrientation.ROW_MAJOR),
                )

            self._qkv_split_in_memcfg = _ws(_qkv_grid, _qkv_shard_w)
            self._qkv_split_q_memcfg = _hs(_q_grid, HD)
            self._concat_in_memcfg = _hs(_concat_grid, HD)
            self._concat_out_memcfg = _ws(_concat_grid, HD)
            _n150_qkv_cores = _rows_q * _cols_q

            # KV-group-interleaved row perm so the sharded nlp_create kernel can
            # split the DRAM-sharded QKV output without an intermediate copy.
            _row_perm = []
            for _kv in range(self.num_kv_heads):
                for _qi in range(_num_q_per_kv):
                    _qh = _kv * _num_q_per_kv + _qi
                    _row_perm.extend(range(_qh * HD, (_qh + 1) * HD))
                _k_off = self.num_heads * HD
                _row_perm.extend(range(_k_off + _kv * HD, _k_off + (_kv + 1) * HD))
                _v_off = (self.num_heads + self.num_kv_heads) * HD
                _row_perm.extend(range(_v_off + _kv * HD, _v_off + (_kv + 1) * HD))
            _perm_t = torch.tensor(_row_perm, dtype=torch.long)
            assert _perm_t.numel() == _fused_qkv

        # === N300 fast-path configs (sharded NLP head ops + KVGI QKV) ===
        if self._n300_cp_opt:
            _M = 32  # CP is always <= 1 tile in M: decode seq=1 and prefill seq=2.

            # Sharded nlp_create_qkv_heads / nlp_concat_heads. Head counts are per-chip.
            _num_q_per_kv = self.num_heads // self.num_kv_heads
            self._n300_fused_qkv = (self.num_heads + 2 * self.num_kv_heads) * HD
            _qkv_shard_w = (_num_q_per_kv + 2) * HD
            assert self._n300_fused_qkv % _qkv_shard_w == 0
            _qkv_cores = self._n300_fused_qkv // _qkv_shard_w
            _qkv_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_qkv_cores - 1, 0))})
            _q_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.num_heads - 1, 0))})
            _concat_grid = ttnn.num_cores_to_corerangeset(self.num_heads, _cg, True)

            def _hs(grid, w):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (_M, w), ttnn.ShardOrientation.ROW_MAJOR),
                )

            def _ws(grid, w):
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(grid, (_M, w), ttnn.ShardOrientation.ROW_MAJOR),
                )

            self._n300_qkv_split_in_memcfg = _ws(_qkv_grid, _qkv_shard_w)
            self._n300_qkv_split_q_memcfg = _hs(_q_grid, HD)
            self._n300_concat_in_memcfg = _hs(_concat_grid, HD)
            self._n300_concat_out_memcfg = _ws(_concat_grid, HD)

            _row_perm_n300 = []
            for _kv in range(self.num_kv_heads):
                for _qi in range(_num_q_per_kv):
                    _qh = _kv * _num_q_per_kv + _qi
                    _row_perm_n300.extend(range(_qh * HD, (_qh + 1) * HD))
                _k_off = self.num_heads * HD
                _row_perm_n300.extend(range(_k_off + _kv * HD, _k_off + (_kv + 1) * HD))
                _v_off = (self.num_heads + self.num_kv_heads) * HD
                _row_perm_n300.extend(range(_v_off + _kv * HD, _v_off + (_kv + 1) * HD))
            _perm_t_n300 = torch.tensor(_row_perm_n300, dtype=torch.long)
            assert _perm_t_n300.numel() == self._n300_fused_qkv

            for li, lw in enumerate(self.layers_w):
                pfx = f"talker.code_predictor.model.layers.{li}."
                _q = _perm_rope_rows(state_dict[pfx + "self_attn.q_proj.weight"], HD)
                _k = _perm_rope_rows(state_dict[pfx + "self_attn.k_proj.weight"], HD)
                _v = state_dict[pfx + "self_attn.v_proj.weight"]
                _qc = list(torch.chunk(_q, self.tp_size, dim=0))
                _kc = list(torch.chunk(_k, self.tp_size, dim=0))
                _vc = list(torch.chunk(_v, self.tp_size, dim=0))
                _per_chip = [
                    torch.cat([_qc[i], _kc[i], _vc[i]], dim=0).index_select(0, _perm_t_n300).contiguous()
                    for i in range(self.tp_size)
                ]
                _stacked = torch.stack(_per_chip, dim=0).transpose(-2, -1).unsqueeze(0).contiguous()
                lw["wqkv_kvgi"] = ttnn.from_torch(
                    _stacked,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=TILE,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
                )
                ttnn.deallocate(lw.pop("wqkv"))

        # Sharded hidden RMSNorms on N150 and N300. Default LN parallelises over
        # M; CP M is one tile so it lands on 1 core (~25 us). Width-shard
        # instead. Post-norm emits gate/up in0 so the MLP I2S disappears.
        # Input norm follows the QKV consumer: DRAM-sharded QKV (N150) keeps
        # the 4-core in0 spec; interleaved QKV (N300) uses the widest grid
        # that divides H, then S2I. Other SKUs keep the 1-core interleaved LN.
        self._use_sharded_ln = self._n150 or self._n300_cp_opt
        if self._use_sharded_ln:
            from models.demos.qwen3_tts.tt.decoder_layer import _build_sharded_rmsnorm_configs

            _M = 32
            _dim_tiles = H // 32
            _ln_mlp_cores = _rows_gu * _cols_gu
            assert _dim_tiles % _ln_mlp_cores == 0, (
                f"CP hidden tiles={_dim_tiles} must divide the gate/up in0 grid "
                f"({_ln_mlp_cores} cores) to fuse the post-norm shard layout"
            )
            _ln_mlp_memcfg, self._ln_mlp_progcfg = _build_sharded_rmsnorm_configs(device, H, _ln_mlp_cores, m=_M)
            assert _ln_mlp_memcfg == self._cp_gate_up_in0_memcfg, (
                "post-norm shard layout must equal the gate/up in0 layout; got "
                f"{_ln_mlp_memcfg} vs {self._cp_gate_up_in0_memcfg}"
            )
            self._ln_mlp_memcfg = self._cp_gate_up_in0_memcfg

            if _n150_qkv_cores is not None:
                _ln_attn_cores = _n150_qkv_cores
            else:
                _ln_attn_cores = next(c for c in (64, 32, 16, 8, 4, 2, 1) if _dim_tiles % c == 0)
            _ln_attn_memcfg, self._ln_attn_progcfg = _build_sharded_rmsnorm_configs(device, H, _ln_attn_cores, m=_M)
            if self._n150:
                assert _ln_attn_memcfg == self._cp_qkv_in0_memcfg, (
                    "input-norm shard layout must equal the QKV in0 layout; got "
                    f"{_ln_attn_memcfg} vs {self._cp_qkv_in0_memcfg}"
                )
                self._ln_attn_memcfg = self._cp_qkv_in0_memcfg
            else:
                self._ln_attn_memcfg = _ln_attn_memcfg

        # DRAM-sharded MLP weights on every SKU. QKV / o_proj DS weights are N150-only.
        for li, lw in enumerate(self.layers_w):
            pfx = f"talker.code_predictor.model.layers.{li}."
            lw_t = {k.replace(pfx, ""): v for k, v in state_dict.items() if k.startswith(pfx)}
            gate_kn = lw_t["mlp.gate_proj.weight"].transpose(0, 1).contiguous()  # [H, interm]
            up_kn = lw_t["mlp.up_proj.weight"].transpose(0, 1).contiguous()
            lw["gate_ds"], _, _ = build_dram_sharded_weight_tp(
                gate_kn, device, self.tp_size, split_dim=1, dtype=ttnn.bfloat16
            )
            lw["up_ds"], _, _ = build_dram_sharded_weight_tp(
                up_kn, device, self.tp_size, split_dim=1, dtype=ttnn.bfloat16
            )
            down_kn = lw_t["mlp.down_proj.weight"].transpose(0, 1).contiguous()  # [interm, H]
            lw["down_ds"], _, _ = build_dram_sharded_weight_tp(
                down_kn, device, self.tp_size, split_dim=0, dtype=ttnn.bfloat16
            )
            if self._n150:
                _q = _perm_rope_rows(state_dict[pfx + "self_attn.q_proj.weight"], HD)
                _k = _perm_rope_rows(state_dict[pfx + "self_attn.k_proj.weight"], HD)
                _v = state_dict[pfx + "self_attn.v_proj.weight"]
                _qc = list(torch.chunk(_q, self.tp_size, dim=0))
                _kc = list(torch.chunk(_k, self.tp_size, dim=0))
                _vc = list(torch.chunk(_v, self.tp_size, dim=0))
                _per_chip = [
                    torch.cat([_qc[i], _kc[i], _vc[i]], dim=0).index_select(0, _perm_t).contiguous()
                    for i in range(self.tp_size)
                ]
                if self.tp_size == 1:
                    wqkv_kn = _per_chip[0].transpose(-2, -1).contiguous()
                else:
                    wqkv_kn = torch.cat([c.transpose(-2, -1).contiguous() for c in _per_chip], dim=1)
                lw["wqkv_ds"], _, _ = build_dram_sharded_weight_tp(
                    wqkv_kn, device, self.tp_size, split_dim=1, dtype=ttnn.bfloat16
                )
                wo_kn = lw_t["self_attn.o_proj.weight"].transpose(0, 1).contiguous()
                lw["o_ds"], _, _ = build_dram_sharded_weight_tp(
                    wo_kn, device, self.tp_size, split_dim=0, dtype=ttnn.bfloat16
                )
                for _dead in ("wqkv", "o_proj", "gate", "up", "down"):
                    ttnn.deallocate(lw.pop(_dead))

        self.lm_heads = []
        for g in range(self.num_code_groups - 1):
            k = f"talker.code_predictor.lm_head.{g}.weight"
            self.lm_heads.append(w_to_tt(state_dict[k]))

        self.codec_embeddings_tt: List[Optional[ttnn.Tensor]] = []
        for i in range(self.num_code_groups - 1):
            k = f"talker.code_predictor.model.codec_embedding.{i}.weight"
            if k in state_dict:
                w = state_dict[k]
                vocab_size, emb_dim = int(w.shape[0]), int(w.shape[1])
                e_tt = ttnn.from_torch(
                    w.to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ROW,
                    memory_config=DRAM,
                )
                self.codec_embeddings_tt.append(ttnn.reshape(e_tt, [1, 1, vocab_size, emb_dim], memory_config=DRAM))
            else:
                self.codec_embeddings_tt.append(None)

    def _all_reduce(self, t: ttnn.Tensor, fast: bool) -> ttnn.Tensor:
        """TP all-reduce. On N300 use the 1-CCL 2-chip form (see mesh_utils).

        Both forms are noisy run to run, so this was picked on medians of 3 captures of
        the CP decode layer: 429 us with the 2-chip form vs 488 us with ttnn.all_reduce.
        """
        from models.demos.qwen3_tts.tt.mesh_utils import tp_all_reduce, tp_all_reduce_2chip

        fn = tp_all_reduce_2chip if fast else tp_all_reduce
        return fn(t, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # ─── Per-layer forward — caller owns input h_tt; we do NOT deallocate it. ───
    def _layer_forward(
        self,
        h_tt: ttnn.Tensor,
        lw: dict,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
        start_pos: int,
        mode: str,
        cur_pos_tensor: Optional[ttnn.Tensor],
        decode_attn_mask: Optional[ttnn.Tensor],
        cp_prefill_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        # residual aliases h_tt (caller-owned). Do NOT deallocate it.
        residual = h_tt
        # N300 fast path. Both CP modes fit one tile in M (decode seq=1, prefill seq=2),
        # so a single m=32 set of shard configs covers them.
        fast = self._n300_cp_opt and int(h_tt.shape[-2]) <= 32

        # Width-sharded RMSNorm on N150 and N300. Skip I2S when the previous
        # layer already wrote this shard spec. N150 keeps the sharded output
        # for DRAM-sharded QKV; N300 S2I because QKV is interleaved.
        if self._use_sharded_ln:
            if h_tt.memory_config() != self._ln_attn_memcfg:
                h_ln_in = ttnn.to_memory_config(h_tt, self._ln_attn_memcfg)
                _own_h_ln = True
            else:
                h_ln_in = h_tt
                _own_h_ln = False
            x = ttnn.rms_norm(
                h_ln_in,
                epsilon=self.rms_norm_eps,
                weight=lw["input_ln_w"],
                compute_kernel_config=self.kcfg,
                program_config=self._ln_attn_progcfg,
                memory_config=self._ln_attn_memcfg,
            )
            if _own_h_ln:
                ttnn.deallocate(h_ln_in)
            if not self._n150:
                x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(x)
                x = x_il
        else:
            x = ttnn.rms_norm(h_tt, epsilon=self.rms_norm_eps, weight=lw["input_ln_w"], compute_kernel_config=self.kcfg)

        if self._n150:
            # DRAM-sharded QKV. Width-shard in0, then slice/reshard into the KVGI
            # nlp_create input so the split runs on num_kv_heads cores.
            if x.memory_config() != self._cp_qkv_in0_memcfg:
                x_s = ttnn.to_memory_config(x, self._cp_qkv_in0_memcfg)
                ttnn.deallocate(x)
            else:
                x_s = x
            xqkv_s = ttnn.linear(
                x_s,
                lw["wqkv_ds"],
                compute_kernel_config=self.kcfg,
                program_config=self._cp_qkv_dramshard_progcfg,
                memory_config=self._cp_qkv_out_memcfg,
            )
            ttnn.deallocate(x_s)
            if self._cp_qkv_n_padded == self._fused_qkv:
                if xqkv_s.memory_config() != self._qkv_split_in_memcfg:
                    xqkv_for_split = ttnn.to_memory_config(xqkv_s, self._qkv_split_in_memcfg)
                    ttnn.deallocate(xqkv_s)
                else:
                    xqkv_for_split = xqkv_s
            else:
                xqkv_for_split = ttnn.slice(
                    xqkv_s,
                    [0, 0, 0, 0],
                    [xqkv_s.shape[0], xqkv_s.shape[1], xqkv_s.shape[2], self._fused_qkv],
                    memory_config=self._qkv_split_in_memcfg,
                )
                ttnn.deallocate(xqkv_s)
            q_s, k_s, v_s = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_for_split,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                transpose_k_heads=False,
                memory_config=self._qkv_split_q_memcfg,
            )
            ttnn.deallocate(xqkv_for_split)
            # q_norm / k_norm reject HEIGHT_SHARDED (layernorm_device_operation).
            # V skips that path — fill_cache / update_cache accept the HS
            # output and SDPA reads V back from the interleaved cache.
            q = ttnn.to_memory_config(q_s, ttnn.L1_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k_s, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_s)
            ttnn.deallocate(k_s)
            v = v_s
        elif fast:
            xqkv = ttnn.matmul(
                x,
                lw["wqkv_kvgi"],
                dtype=self.act_dtype,
                compute_kernel_config=self.kcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x)
            xqkv_s = ttnn.to_memory_config(xqkv, self._n300_qkv_split_in_memcfg)
            ttnn.deallocate(xqkv)
            q_s, k_s, v_s = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_s,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                transpose_k_heads=False,
                memory_config=self._n300_qkv_split_q_memcfg,
            )
            ttnn.deallocate(xqkv_s)
            q = ttnn.to_memory_config(q_s, ttnn.L1_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k_s, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_s)
            ttnn.deallocate(k_s)
            v = v_s
        else:
            xqkv = ttnn.matmul(x, lw["wqkv"], dtype=self.act_dtype, compute_kernel_config=self.kcfg)
            ttnn.deallocate(x)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                xqkv,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(xqkv)

        q_n = ttnn.rms_norm(
            q,
            epsilon=self.rms_norm_eps,
            weight=lw["q_norm_w"],
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        q = q_n
        k_n = ttnn.rms_norm(
            k,
            epsilon=self.rms_norm_eps,
            weight=lw["k_norm_w"],
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(k)
        k = k_n

        # RoPE expects bf16 — cast in/out.
        if q.dtype != ttnn.bfloat16:
            q_b = ttnn.typecast(q, dtype=ttnn.bfloat16)
            ttnn.deallocate(q)
            q = q_b
        if k.dtype != ttnn.bfloat16:
            k_b = ttnn.typecast(k, dtype=ttnn.bfloat16)
            ttnn.deallocate(k)
            k = k_b
        # RoPE kernel is bf16-only; cast Q/K if still in fp32.
        q_r, k_r = apply_rope_qk(
            q,
            k,
            cos,
            sin,
            transformation_mat,
            head_dim=self.head_dim,
            decode_trans_mat=self._decode_trans_mat,
            compute_kernel_config=self.kcfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        q, k = q_r, k_r

        # KV cache write/read.
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache.dtype != k.dtype:
                k_w = ttnn.typecast(k, dtype=k_cache.dtype)
                ttnn.deallocate(k)
                k = k_w
            if v_cache.dtype != v.dtype:
                v_w = ttnn.typecast(v, dtype=v_cache.dtype)
                ttnn.deallocate(v)
                v = v_w
            if mode == "prefill":
                ttnn.fill_cache(k_cache, k, 0)
                ttnn.fill_cache(v_cache, v, 0)
            else:
                ttnn.update_cache(k_cache, k, update_idx=start_pos)
                ttnn.update_cache(v_cache, v, update_idx=start_pos)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if k_cache.dtype == self.act_dtype:
                k_for_attn = k_cache
                v_for_attn = v_cache
                k_cache_alias = True
            else:
                k_for_attn = ttnn.typecast(k_cache, dtype=self.act_dtype)
                v_for_attn = ttnn.typecast(v_cache, dtype=self.act_dtype)
                k_cache_alias = False
            updated_kv = (k_cache, v_cache)
        else:
            if k.dtype != self.act_dtype:
                k_f = ttnn.typecast(k, dtype=self.act_dtype)
                ttnn.deallocate(k)
                k = k_f
            if v.dtype != self.act_dtype:
                v_f = ttnn.typecast(v, dtype=self.act_dtype)
                ttnn.deallocate(v)
                v = v_f
            if v.is_sharded():
                # SDPA / fp32 BMM require interleaved V. Cache path reads V
                # back from DRAM so it can skip this hop.
                v_il = ttnn.to_memory_config(v, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(v)
                v = v_il
            k_for_attn = k
            v_for_attn = v
            k_cache_alias = False
            updated_kv = None

        if self._n150:
            # Fused SDPA (GQA-native). Same HiFi4 + fp32 dest as Talker N150:
            # QK-norm gain ≈ 68 needs the bf16 mantissa and fp32 accumulation.
            q_seq = int(q.shape[2])
            k_seq = int(k_for_attn.shape[2])
            _explicit_mask = decode_attn_mask if decode_attn_mask is not None else cp_prefill_mask
            _explicit_mask, _own_sdpa_mask = prepare_fused_sdpa_mask(_explicit_mask)
            _use_causal = _explicit_mask is None and q_seq == k_seq and q_seq > 1
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                attn_mask=_explicit_mask,
                is_causal=_use_causal,
                scale=self.scale,
                compute_kernel_config=self.sdpa_kcfg,
                program_config=self.sdpa_program_config,
                memory_config=self._concat_in_memcfg,
            )
            ttnn.deallocate(q)
            if not k_cache_alias:
                ttnn.deallocate(k_for_attn)
                ttnn.deallocate(v_for_attn)
            if _own_sdpa_mask:
                ttnn.deallocate(_explicit_mask)

            if attn_out.memory_config() != self._concat_in_memcfg:
                attn_s = ttnn.to_memory_config(attn_out, self._concat_in_memcfg)
                ttnn.deallocate(attn_out)
            else:
                attn_s = attn_out
            attn_concat_s = ttnn.experimental.nlp_concat_heads(attn_s, memory_config=self._concat_out_memcfg)
            ttnn.deallocate(attn_s)
            attn_wo = ttnn.to_memory_config(attn_concat_s, self._cp_wo_in0_memcfg)
            ttnn.deallocate(attn_concat_s)
            o_s = ttnn.linear(
                attn_wo,
                lw["o_ds"],
                compute_kernel_config=self.kcfg,
                program_config=self._cp_wo_dramshard_progcfg,
                memory_config=self._cp_wo_out_memcfg,
            )
            ttnn.deallocate(attn_wo)
            # Slice/unshard into the post-LN / gate-up in0 spec so the residual
            # add writes that layout and the MLP I2S disappears. Talker does the
            # same for a padded DRAM-sharded o_proj (S2I then slice; slice-from-
            # sharded is not free).
            if self._cp_wo_n_padded != self.hidden_size:
                o_il = ttnn.to_memory_config(o_s, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(o_s)
                o = ttnn.slice(
                    o_il,
                    [0, 0, 0, 0],
                    [o_il.shape[0], o_il.shape[1], o_il.shape[2], self.hidden_size],
                    memory_config=self._ln_mlp_memcfg,
                )
                ttnn.deallocate(o_il)
            else:
                o = ttnn.to_memory_config(o_s, self._ln_mlp_memcfg)
                ttnn.deallocate(o_s)
        else:
            # SDPA runs in fp32 — QK-norm amplifies K by ~68x; bf16 max=65504 and
            # q·k dot products can reach ~260*260*128 = overflow. Cast explicitly.
            q_f32 = ttnn.typecast(q, dtype=ttnn.float32)
            ttnn.deallocate(q)

            if self.num_kv_groups > 1:
                k_exp_bf16 = ttnn.repeat_interleave(k_for_attn, self.num_kv_groups, dim=1)
                v_exp_bf16 = ttnn.repeat_interleave(v_for_attn, self.num_kv_groups, dim=1)
                if not k_cache_alias and kv_cache is not None:
                    ttnn.deallocate(k_for_attn)
                    ttnn.deallocate(v_for_attn)
                elif kv_cache is None:
                    ttnn.deallocate(k_for_attn)
                    ttnn.deallocate(v_for_attn)
            else:
                k_exp_bf16 = k_for_attn
                v_exp_bf16 = v_for_attn
            k_exp = ttnn.typecast(k_exp_bf16, dtype=ttnn.float32)
            v_exp = ttnn.typecast(v_exp_bf16, dtype=ttnn.float32)
            if self.num_kv_groups > 1:
                ttnn.deallocate(k_exp_bf16)
            ttnn.deallocate(v_exp_bf16)

            scores = ttnn.matmul(
                q_f32,
                k_exp,
                transpose_b=True,
                dtype=ttnn.float32,
                compute_kernel_config=self.sdpa_kcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_f32)
            ttnn.deallocate(k_exp)
            scores = ttnn.mul(scores, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG)

            q_seq = int(scores.shape[-2])
            k_seq_eff = int(scores.shape[-1])
            if decode_attn_mask is not None:
                scores = ttnn.add(scores, decode_attn_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif cp_prefill_mask is not None:
                scores = ttnn.add(scores, cp_prefill_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif q_seq > 1:
                mask_cpu = torch.full((q_seq, k_seq_eff), float("-inf"), dtype=torch.float32)
                for i in range(q_seq):
                    mask_cpu[i, : i + 1] = 0.0
                mask_tt = ttnn.from_torch(
                    mask_cpu.unsqueeze(0).unsqueeze(0),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                scores = ttnn.add(scores, mask_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(mask_tt)
            elif kv_cache is not None:
                valid = start_pos + 1
                if k_seq_eff > valid:
                    mask_cpu = torch.full((1, k_seq_eff), float("-inf"), dtype=torch.float32)
                    mask_cpu[0, :valid] = 0.0
                    mask_tt = ttnn.from_torch(
                        mask_cpu.unsqueeze(0).unsqueeze(0),
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    scores = ttnn.add(scores, mask_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(mask_tt)

            attn_weights = ttnn.softmax(scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(scores)
            attn_out_f32 = ttnn.matmul(
                attn_weights,
                v_exp,
                dtype=ttnn.float32,
                compute_kernel_config=self.sdpa_kcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_weights)
            ttnn.deallocate(v_exp)
            attn_out = ttnn.typecast(attn_out_f32, dtype=ttnn.bfloat16)
            ttnn.deallocate(attn_out_f32)

            if fast:
                attn_s = ttnn.to_memory_config(attn_out, self._n300_concat_in_memcfg)
                ttnn.deallocate(attn_out)
                attn_concat_s = ttnn.experimental.nlp_concat_heads(attn_s, memory_config=self._n300_concat_out_memcfg)
                ttnn.deallocate(attn_s)
                attn_concat = ttnn.to_memory_config(attn_concat_s, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(attn_concat_s)
            else:
                attn_concat = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(attn_out)

            o = ttnn.matmul(attn_concat, lw["o_proj"], dtype=self.act_dtype, compute_kernel_config=self.kcfg)
            ttnn.deallocate(attn_concat)
        if self.tp_size > 1:
            o = self._all_reduce(o, fast)

        # Residual + post-norm. residual = caller's h_tt — DO NOT deallocate.
        # On N150 / N300 the add writes the post-LN / gate-up in0 spec so both
        # the I2S into post-LN and the I2S into MLP disappear. BinaryNg output
        # layout is independent of the addends (same as Talker decoder_layer).
        if self._use_sharded_ln:
            h_post = ttnn.add(residual, o, dtype=self.act_dtype, memory_config=self._ln_mlp_memcfg)
            ttnn.deallocate(o)
            residual2 = h_post
            h2 = ttnn.rms_norm(
                h_post,
                epsilon=self.rms_norm_eps,
                weight=lw["post_ln_w"],
                compute_kernel_config=self.kcfg,
                program_config=self._ln_mlp_progcfg,
                memory_config=self._ln_mlp_memcfg,
            )
        else:
            h_post = ttnn.add(residual, o, dtype=self.act_dtype)
            ttnn.deallocate(o)
            residual2 = h_post
            h2 = ttnn.rms_norm(
                h_post, epsilon=self.rms_norm_eps, weight=lw["post_ln_w"], compute_kernel_config=self.kcfg
            )

        # DRAM-sharded MLP: shard h2 into L1 WIDTH_SHARDED once, then gate/up read from
        # their DRAM-banked weights in parallel. Output stays sharded — reshard to down's
        # grid only once (16-core gate/up → 12-core down), then all_reduce.
        if h2.memory_config() != self._cp_gate_up_in0_memcfg:
            h2_sharded = ttnn.to_memory_config(h2, self._cp_gate_up_in0_memcfg)
            ttnn.deallocate(h2)
        else:
            h2_sharded = h2
        gate_o = ttnn.linear(
            h2_sharded,
            lw["gate_ds"],
            compute_kernel_config=self.kcfg,
            program_config=self._cp_gate_up_dramshard_progcfg,
            memory_config=self._cp_gate_up_out_memcfg,
        )
        up_o = ttnn.linear(
            h2_sharded,
            lw["up_ds"],
            compute_kernel_config=self.kcfg,
            program_config=self._cp_gate_up_dramshard_progcfg,
            memory_config=self._cp_gate_up_out_memcfg,
        )
        ttnn.deallocate(h2_sharded)
        gate_silu = ttnn.silu(gate_o, memory_config=self._cp_gate_up_out_memcfg)
        ttnn.deallocate(gate_o)
        gated = ttnn.mul(gate_silu, up_o, memory_config=self._cp_gate_up_out_memcfg)
        ttnn.deallocate(gate_silu)
        ttnn.deallocate(up_o)
        if self._cp_gate_up_out_memcfg != self._cp_down_in0_memcfg:
            gated_d = ttnn.to_memory_config(gated, self._cp_down_in0_memcfg)
            ttnn.deallocate(gated)
        else:
            gated_d = gated
        mlp_o_sharded = ttnn.linear(
            gated_d,
            lw["down_ds"],
            compute_kernel_config=self.kcfg,
            program_config=self._cp_down_dramshard_progcfg,
            memory_config=self._cp_down_out_memcfg,
        )
        ttnn.deallocate(gated_d)
        if self._cp_down_n_padded != self.hidden_size:
            mlp_o_il = ttnn.to_memory_config(mlp_o_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(mlp_o_sharded)
            mlp_o = ttnn.slice(
                mlp_o_il,
                [0, 0, 0, 0],
                [mlp_o_il.shape[0], mlp_o_il.shape[1], mlp_o_il.shape[2], self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_o_il)
        else:
            mlp_o = ttnn.to_memory_config(mlp_o_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(mlp_o_sharded)
        if self.tp_size > 1:
            mlp_o = self._all_reduce(mlp_o, fast)
        # Next layer's input LN skips I2S when this add already wrote its spec.
        # forward_single_step S2Is once before the interleaved final norm.
        _out_memcfg = self._ln_attn_memcfg if self._use_sharded_ln else ttnn.L1_MEMORY_CONFIG
        out = ttnn.add(residual2, mlp_o, dtype=self.act_dtype, memory_config=_out_memcfg)
        ttnn.deallocate(residual2)
        ttnn.deallocate(mlp_o)
        return out, updated_kv

    # ─── Public API matching production CodePredictor.forward_single_step ──
    def forward_single_step(
        self,
        inputs_embeds,
        cos,
        sin,
        transformation_mat,
        generation_step: int = 1,
        attention_mask=None,
        kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_attn_mask: Optional[ttnn.Tensor] = None,
        cp_prefill_mask: Optional[ttnn.Tensor] = None,
        return_hidden_state: bool = False,
    ):
        """Trace-compatible fp32 CP forward. Caller retains ownership of inputs_embeds."""
        if self.needs_projection:
            h = ttnn.linear(
                inputs_embeds,
                self.input_proj,
                bias=self.input_proj_bias if self.input_proj_bias is not None else None,
                dtype=self.act_dtype,
                compute_kernel_config=self.kcfg,
            )
            own_h = True
        else:
            h = inputs_embeds
            own_h = False

        _own_rope = False
        if mode == "decode" or int(inputs_embeds.shape[-2]) == 1:
            cos, sin, _own_rope = shard_decode_rope_tables(cos, sin, self.head_dim)

        # Fused SDPA rejects fp32. Convert once per step (Talker does the same);
        # prepare_fused_sdpa_mask is then a no-op inside each layer.
        _own_decode_mask = _own_prefill_mask = False
        if self._n150:
            decode_attn_mask, _own_decode_mask = prepare_fused_sdpa_mask(decode_attn_mask)
            cp_prefill_mask, _own_prefill_mask = prepare_fused_sdpa_mask(cp_prefill_mask)

        updated_kvs = [] if kv_caches is not None else None
        for li, lw in enumerate(self.layers_w):
            layer_kv = kv_caches[li] if kv_caches is not None else None
            h_new, updated_kv = self._layer_forward(
                h,
                lw,
                cos,
                sin,
                transformation_mat,
                kv_cache=layer_kv,
                start_pos=start_pos,
                mode=mode,
                cur_pos_tensor=cur_pos_tensor,
                decode_attn_mask=decode_attn_mask,
                cp_prefill_mask=cp_prefill_mask,
            )
            if own_h:
                ttnn.deallocate(h)
            h = h_new
            own_h = True
            if updated_kvs is not None:
                updated_kvs.append(updated_kv)

        # Layers return the input-LN shard spec; final RMSNorm is interleaved.
        if h.is_sharded():
            h_il = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)
            if own_h:
                ttnn.deallocate(h)
            h = h_il
            own_h = True
        h_norm = ttnn.rms_norm(h, epsilon=self.rms_norm_eps, weight=self.final_norm_w, compute_kernel_config=self.kcfg)
        if own_h:
            ttnn.deallocate(h)
        if _own_rope:
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        if _own_decode_mask:
            ttnn.deallocate(decode_attn_mask)
        if _own_prefill_mask:
            ttnn.deallocate(cp_prefill_mask)

        if return_hidden_state:
            return h_norm, updated_kvs

        # Apply lm_head over full hidden (caller indexes last position).
        lm_idx = generation_step - 1
        logits = ttnn.matmul(h_norm, self.lm_heads[lm_idx], dtype=self.act_dtype, compute_kernel_config=self.kcfg)
        ttnn.deallocate(h_norm)
        return logits, updated_kvs

    def get_codec_embedding(self, code_idx: int, token_ids_tt, *, mode: str = "decode"):
        if code_idx < len(self.codec_embeddings_tt) and self.codec_embeddings_tt[code_idx] is not None:
            return ttnn.embedding(
                token_ids_tt,
                self.codec_embeddings_tt[code_idx],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        raise ValueError(f"Missing TTNN codec embedding for index {code_idx}")
