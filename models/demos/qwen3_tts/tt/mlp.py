# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SwiGLU MLP implementation for Qwen3-TTS.

Decode (M=1): all three projections (gate, up, down) run through the
DRAM-sharded matmul kernel (parallel weight reads across DRAM banks). The
output of gate/up stays sharded into the SiLU·mul and feeds down_proj's
sharded input directly — no L1_INTERLEAVED round-trip in the chain.

Prefill (M>1): the DRAM-sharded matmul kernel asserts M==tile_height, so
we keep the standard 1D-mcast matmul path for prefill.
"""
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    build_dram_sharded_weight_tp,
    dram_sharded_program_config,
    find_grid_k_n,
    width_sharded_l1_memcfg,
)
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config


class MLP(LightweightModule):
    """SwiGLU MLP for Qwen3-TTS — down_proj(silu(gate(x)) * up(x))."""

    def __init__(
        self,
        device,
        hidden_size: int,
        intermediate_size: int,
        state_dict: dict,
        layer_prefix: str,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size

        self.tp_size = get_tp_size(device) if is_mesh_device else 1
        assert (
            intermediate_size % self.tp_size == 0
        ), f"intermediate_size={intermediate_size} must be divisible by tp_size={self.tp_size}"
        # Per-chip intermediate after column-parallel gate/up split.
        self.local_intermediate = intermediate_size // self.tp_size

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"{layer_prefix}_{name}".replace(".", "_")

        _mesh_mapper_replicate = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
        _dram = ttnn.DRAM_MEMORY_CONFIG

        def _build_proj_weight(weight_key: str, cache_name: str, mesh_mapper=None):
            # Host-side [out, in] -> [1, 1, in, out] for ttnn.linear; one upload at init.
            weight_host = state_dict[weight_key].transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
            cache_file = get_cache_name(cache_name)
            if cache_file is not None:
                return ttnn.as_tensor(
                    weight_host,
                    device=device,
                    dtype=weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=_dram,
                    cache_file_name=cache_file,
                    mesh_mapper=mesh_mapper,
                )
            return ttnn.from_torch(
                weight_host,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=mesh_mapper,
            )

        def _build_colpar_weight(weight_key: str, cache_name: str):
            """Column-parallel: split N (output/intermediate) across TP chips."""
            import torch

            w_full = state_dict[weight_key]  # [out_features, in_features]
            if self.tp_size == 1:
                return _build_proj_weight(weight_key, cache_name, mesh_mapper=_mesh_mapper_replicate)
            chunks = list(torch.chunk(w_full, self.tp_size, dim=0))  # split out_features
            stacked = torch.stack(chunks, dim=0)  # [tp, local_out, in]
            host = stacked.transpose(-2, -1).unsqueeze(0).contiguous()  # [1, tp, in, local_out]
            return ttnn.from_torch(
                host,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
            )

        def _build_rowpar_weight(weight_key: str, cache_name: str):
            """Row-parallel: split K (input / local_intermediate) across TP chips."""
            import torch

            w_full = state_dict[weight_key]  # [out_features, in_features]
            if self.tp_size == 1:
                return _build_proj_weight(weight_key, cache_name, mesh_mapper=_mesh_mapper_replicate)
            # After transpose, K is in_features; split that.
            w_t = w_full.transpose(-2, -1).contiguous()  # [in, out]
            chunks = list(torch.chunk(w_t, self.tp_size, dim=0))  # split in (=local_intermediate per chip)
            stacked = torch.stack(chunks, dim=0).unsqueeze(0).contiguous()  # [1, tp, local_in, out]
            return ttnn.from_torch(
                stacked,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
            )

        # Prefill weights (DRAM_INTERLEAVED) — TP>1: column/row-parallel sharded.
        self.gate_proj = _build_colpar_weight(f"{layer_prefix}.mlp.gate_proj.weight", "gate_proj")
        self.up_proj = _build_colpar_weight(f"{layer_prefix}.mlp.up_proj.weight", "up_proj")
        self.down_proj = _build_rowpar_weight(f"{layer_prefix}.mlp.down_proj.weight", "down_proj")

        # Decode-only DRAM-sharded weights + program/memory configs (M=1 tile).
        # gate_proj and up_proj share K=hidden, N=intermediate so they share configs.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        grid = device.compute_with_storage_grid_size()
        self.short_seq_limit = 32
        _fp32 = self.compute_kernel_config.fp32_dest_acc_en
        # 1D program configs: gate/up output = local_intermediate per chip; down input = local_intermediate.
        self._decode_gate_up_progcfg = make_linear_1d_program_config(
            m=1, k=hidden_size, n=self.local_intermediate, grid_x=grid.x, grid_y=grid.y, fp32_dest_acc_en=_fp32
        )
        self._decode_down_progcfg = make_linear_1d_program_config(
            m=1, k=self.local_intermediate, n=hidden_size, grid_x=grid.x, grid_y=grid.y, fp32_dest_acc_en=_fp32
        )
        self._short_seq_gate_up_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=hidden_size,
            n=self.local_intermediate,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )
        self._short_seq_down_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=self.local_intermediate,
            n=hidden_size,
            grid_x=grid.x,
            grid_y=grid.y,
            fp32_dest_acc_en=_fp32,
        )

        # DRAM-sharded decode path — now supported for TP>1 too.
        # TP=2 benefit: each chip has smaller dimensions (local_intermediate = intermediate // tp)
        # so tensors fit L1 more easily and DRAM bandwidth splits across chips.
        _cg = device.compute_with_storage_grid_size()
        if self.tp_size > 1:
            # Column-parallel gate/up: full weight [hidden, intermediate] sharded along N (dim=1).
            gate_w_kn_full = state_dict[f"{layer_prefix}.mlp.gate_proj.weight"].transpose(-2, -1).contiguous()
            up_w_kn_full = state_dict[f"{layer_prefix}.mlp.up_proj.weight"].transpose(-2, -1).contiguous()
            self.gate_proj_dram_sharded, k_gu, n_padded_gu = build_dram_sharded_weight_tp(
                gate_w_kn_full, device, self.tp_size, split_dim=1, dtype=weight_dtype
            )
            self.up_proj_dram_sharded, _, _ = build_dram_sharded_weight_tp(
                up_w_kn_full, device, self.tp_size, split_dim=1, dtype=weight_dtype
            )
            self._decode_gate_up_n_padded = n_padded_gu
            k_tiles_gu, n_tiles_gu = k_gu // 32, n_padded_gu // 32
            rows_gu, cols_gu = find_grid_k_n(k_tiles_gu, n_tiles_gu, max_rows=_cg.y, max_cols=_cg.x)
            self._decode_gate_up_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=k_gu, n=n_padded_gu, num_cores=rows_gu * cols_gu
            )
            self._decode_gate_up_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=k_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
            )
            self._decode_gate_up_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=n_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
            )
            # Row-parallel down: full weight [intermediate, hidden] sharded along K (dim=0).
            down_w_kn_full = state_dict[f"{layer_prefix}.mlp.down_proj.weight"].transpose(-2, -1).contiguous()
            self.down_proj_dram_sharded, k_d, n_padded_d = build_dram_sharded_weight_tp(
                down_w_kn_full, device, self.tp_size, split_dim=0, dtype=weight_dtype
            )
            self._decode_down_n_padded = n_padded_d
            k_tiles_d, n_tiles_d = k_d // 32, n_padded_d // 32
            rows_d, cols_d = find_grid_k_n(k_tiles_d, n_tiles_d, max_rows=_cg.y, max_cols=_cg.x)
            self._decode_down_dramshard_progcfg = dram_sharded_program_config(
                m=32, k=k_d, n=n_padded_d, num_cores=rows_d * cols_d
            )
            self._decode_down_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=k_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
            )
            self._decode_down_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=1, k_tiles=n_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
            )
            return

        gate_w_kn = state_dict[f"{layer_prefix}.mlp.gate_proj.weight"].transpose(-2, -1).contiguous()
        up_w_kn = state_dict[f"{layer_prefix}.mlp.up_proj.weight"].transpose(-2, -1).contiguous()
        self.gate_proj_dram_sharded, k_gu, n_padded_gu = build_dram_sharded_weight(
            gate_w_kn, device, dtype=weight_dtype
        )
        self.up_proj_dram_sharded, _, _ = build_dram_sharded_weight(up_w_kn, device, dtype=weight_dtype)
        self._decode_gate_up_n_padded = n_padded_gu
        k_tiles_gu, n_tiles_gu = k_gu // 32, n_padded_gu // 32
        # find_grid_k_n must honor the actual compute grid (8x8 on WH N150, 13x10 on BH P150).
        _cg = device.compute_with_storage_grid_size()
        rows_gu, cols_gu = find_grid_k_n(k_tiles_gu, n_tiles_gu, max_rows=_cg.y, max_cols=_cg.x)
        self._decode_gate_up_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=k_gu, n=n_padded_gu, num_cores=rows_gu * cols_gu
        )
        self._decode_gate_up_in0_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=k_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
        )
        self._decode_gate_up_out_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=n_tiles_gu, num_cores_x=cols_gu, num_cores_y=rows_gu
        )

        down_w_kn = state_dict[f"{layer_prefix}.mlp.down_proj.weight"].transpose(-2, -1).contiguous()
        self.down_proj_dram_sharded, k_d, n_padded_d = build_dram_sharded_weight(down_w_kn, device, dtype=weight_dtype)
        self._decode_down_n_padded = n_padded_d
        k_tiles_d, n_tiles_d = k_d // 32, n_padded_d // 32
        rows_d, cols_d = find_grid_k_n(k_tiles_d, n_tiles_d, max_rows=_cg.y, max_cols=_cg.x)
        self._decode_down_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=k_d, n=n_padded_d, num_cores=rows_d * cols_d
        )
        self._decode_down_in0_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=k_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
        )
        self._decode_down_out_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=n_tiles_d, num_cores_x=cols_d, num_cores_y=rows_d
        )

    def forward(self, x: ttnn.Tensor, mode: str = "prefill") -> ttnn.Tensor:
        """Apply SwiGLU MLP.

        Decode (mode=="decode" or seq_len==1): DRAM-sharded matmul chain.
        Prefill: standard 1D-mcast matmul; for seq>=1024 we reshape to fit on device.
        """
        seq_len = x.shape[-2]
        is_decode = mode == "decode" or seq_len == 1

        # Keep activations in L1 for all sequence lengths. Very long sequences
        # (seq >= 1024) are split into 1024-tile chunks via ttnn.reshape below, so
        # the per-chunk tensor is always ≤ local_intermediate × 1024 × 2 bytes ≈ 6 MB
        # on TP=2 (local_intermediate=3072) or 12 MB on TP=1 — both well within L1.
        mem_cfg = ttnn.L1_MEMORY_CONFIG
        if is_decode:
            gate_up_progcfg = self._decode_gate_up_progcfg
            down_progcfg = self._decode_down_progcfg
        elif seq_len <= self.short_seq_limit:
            gate_up_progcfg = self._short_seq_gate_up_progcfg
            down_progcfg = self._short_seq_down_progcfg
        else:
            gate_up_progcfg = down_progcfg = None

        # Reshape for very large sequences to fit on device.
        if seq_len >= 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # Decode path: DRAM-sharded chain. Now enabled for all tp_size values.
        # TP>1: each chip runs its per-chip (smaller) DRAM-sharded matmuls, then all_reduce
        # on the down output combines partial sums.
        if is_decode and seq_len < 1024:
            # Width-shard x once, reuse for both gate and up. Skip the I→S if the
            # caller already gave us a tensor in the matching shard config (e.g. piped
            # from a sharded layernorm in decoder_layer).
            if x.memory_config() == self._decode_gate_up_in0_memcfg:
                x_sharded = x
                _own_x_sharded = False
            else:
                x_sharded = ttnn.to_memory_config(x, self._decode_gate_up_in0_memcfg)
                _own_x_sharded = True
            gate_sharded = ttnn.linear(
                x_sharded,
                self.gate_proj_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._decode_gate_up_dramshard_progcfg,
                memory_config=self._decode_gate_up_out_memcfg,
            )
            up_sharded = ttnn.linear(
                x_sharded,
                self.up_proj_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._decode_gate_up_dramshard_progcfg,
                memory_config=self._decode_gate_up_out_memcfg,
            )
            if _own_x_sharded:
                ttnn.deallocate(x_sharded)
            # gate/up sharding == down in0 sharding (same K). Mul preserves layout.
            hidden_sharded = ttnn.mul(
                gate_sharded,
                up_sharded,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=self._decode_gate_up_out_memcfg,
            )
            ttnn.deallocate(gate_sharded)
            ttnn.deallocate(up_sharded)
            # Gate-up output layout and down input layout match only when N for
            # down doesn't get DRAM-padded (i.e. hidden_size already multiple of
            # TILE*dram_cores). True on BH P150 (dram_cores=8, hidden=2048) but
            # not on Wormhole (dram_cores=12 → hidden pads 2048→2304, giving
            # down a different num_cores). Reshard when they differ.
            if self._decode_gate_up_out_memcfg != self._decode_down_in0_memcfg:
                hidden_for_down = ttnn.to_memory_config(hidden_sharded, self._decode_down_in0_memcfg)
                ttnn.deallocate(hidden_sharded)
            else:
                hidden_for_down = hidden_sharded
            out_sharded = ttnn.linear(
                hidden_for_down,
                self.down_proj_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._decode_down_dramshard_progcfg,
                memory_config=self._decode_down_out_memcfg,
            )
            ttnn.deallocate(hidden_for_down)
            # Decoder layer's residual add wants sharded input — return sharded (TP=1).
            # TP>1: all_reduce after unpadding — go via L1_INTERLEAVED for CCL compat.
            if self._decode_down_n_padded == self.hidden_size:
                if self.tp_size > 1:
                    from models.demos.qwen3_tts.tt.mesh_utils import tp_all_reduce

                    output_il = ttnn.to_memory_config(out_sharded, ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(out_sharded)
                    return tp_all_reduce(output_il, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
                return out_sharded
            # Weight N was padded; slice back via L1_INTERLEAVED.
            output_padded = ttnn.to_memory_config(out_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(out_sharded)
            output = ttnn.slice(
                output_padded,
                [0, 0, 0, 0],
                [output_padded.shape[0], output_padded.shape[1], output_padded.shape[2], self.hidden_size],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(output_padded)
            if self.tp_size > 1:
                from models.demos.qwen3_tts.tt.mesh_utils import tp_all_reduce

                output = tp_all_reduce(output, self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
            return output

        # Prefill path: standard 1D-mcast matmul.
        gate_out = ttnn.linear(
            x,
            self.gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mem_cfg,
            program_config=gate_up_progcfg,
        )
        up_out = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mem_cfg,
            program_config=gate_up_progcfg,
        )
        hidden = ttnn.mul(
            gate_out,
            up_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=mem_cfg,
        )
        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)
        output = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mem_cfg,
            program_config=down_progcfg,
        )
        ttnn.deallocate(hidden)
        if seq_len >= 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])
        # Row-parallel down-proj on TP>1: each chip has a partial sum; all_reduce gives full hidden.
        if self.tp_size > 1:
            from models.demos.qwen3_tts.tt.mesh_utils import tp_all_reduce

            output = tp_all_reduce(output, self.device, memory_config=mem_cfg)
        return output
