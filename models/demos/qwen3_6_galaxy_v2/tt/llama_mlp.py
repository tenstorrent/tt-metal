# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule


def pad_to_next_multiple(tensor):
    # Get the current size of the last two dimensions
    height, width = tensor.shape[-2], tensor.shape[-1]
    if height < 9216:
        pad_height = 9216 - height
        pad_width = 3840 * 8 - width
    else:
        pad_height = 3840 * 8 - height
        pad_width = 9216 - width

    # Apply padding (padding is in the format: (left, right, top, bottom))
    padding = (0, pad_width, 0, pad_height)
    padded_tensor = F.pad(tensor, padding, mode="constant", value=0)  # You can change `value` for a different pad value

    return padded_tensor


class TtLlamaMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        prefetcher_setup=None,
        tt_ccl=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        self.layer_num = layer_num
        _pat = getattr(args, "linear_attention_pattern", None)
        self.is_full_attn_layer = _pat is not None and layer_num < len(_pat) and _pat[layer_num] == "full_attention"
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}" + "prefetcher")

        w1_w3_mem_config = self.model_config[
            "W1W3_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = self.model_config[
            "W2_RING_MEMCFG"
        ]  # args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
        # is_qwen / is_qwen36: MLP weights must be bfloat16. V2-7b found
        # layer-3 forward PCC drops 0.999 -> 0.77 with bf8b w1/w3 weights
        # when attention output flows through MLP via the full TtTransformer
        # decoder. The 4L/64L precision floor requires bf16 across all
        # 64 layers (matches olmo session-11 residual-stream dtype lesson).
        # V4: QWEN36_FP32_WEIGHTS=1 escapes the bf16 lock-in for VLM precision push.
        _mlp_force_bf16 = (args.is_qwen or getattr(args, "is_qwen36", False)) and (
            os.environ.get("QWEN36_FP32_WEIGHTS", "0") != "1"
        )
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=ttnn.bfloat16 if _mlp_force_bf16 else type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        # Interleaved variants (prefill path) — same bf16 override for
        # qwen3/qwen3.6.
        as_interleaved_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=ttnn.bfloat16 if _mlp_force_bf16 else type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        self.four_bit_mlp = args.optimizations.bfp4_mlp

        # Sharded weights
        w1_dim = (-1, -2)
        w2_dim = (-2, -1)

        # sharded
        self.w1 = as_sharded_tensor(
            "w1_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ttnn.bfloat8_b, dim=w2_dim)
        self.w3 = as_sharded_tensor("w3_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim)

        self.w1_interleaved = as_interleaved_tensor(
            "w1_interleaved", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim
        )
        self.w2_interleaved = as_interleaved_tensor("w2_interleaved", ttnn.bfloat8_b, dim=w2_dim)
        self.w3_interleaved = as_interleaved_tensor(
            "w3_interleaved", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, dim=w1_dim
        )

        # Decode-only ring-40 weights (lower-waste fused-FF2 path; Task 2).
        # Built the SAME way as the ring-24 sharded weights above but with the
        # ring-40 DRAM-sharded memcfgs (N padded 2176->2560 instead of 3840,
        # K=1280 divides 40 cores cleanly). Prefill keeps self.w1/w2/w3 +
        # *_interleaved untouched. Only allocated in decode mode (the ring-40
        # config keys only exist for decode).
        if tt_ccl.mode == "decode":
            # The non-prefetcher gather_in0 ring matmul needs its in1 weight L1
            # WIDTH-SHARDED contiguously over the 40 ring cores (NOT DRAM-sharded;
            # the DRAM bank layout scrambles columns for this kernel — see FF12
            # micro-test: per-64-col windows PCC 0.9999 but aggregate 0.30). Build the
            # ring-40 weights by loading the NATIVE per-device weight DRAM-INTERLEAVED
            # (preserves logical column order under ShardTensor2dMesh, unlike the
            # DRAM-bank-sharded layout), then resharding to L1 WIDTH-SHARDED over
            # RING40_MM_CRS — the reshard zero-pads N (w1/w3) / K (w2) from the native
            # 2176 up to the ring-40 width (2560). This is the exact in1 layout the
            # proven ring-40 all_gather_matmul wrapper test uses.
            # The non-prefetcher gather_in0 ring matmul reads its in1 weight L1
            # WIDTH-SHARDED contiguously over the 40 ring cores (the
            # DRAM-BANK-sharded W1W3_RING40_MEMCFG layout scrambles columns for this
            # kernel — FF12 micro-test: per-64-col windows PCC 0.9999, aggregate 0.30).
            # But keeping the ring-40 weights L1-RESIDENT clashes with the prefill
            # matmul circular buffers (L1 OOM). So store the ring-40 weights
            # DRAM-INTERLEAVED here (DRAM-interleaved preserves logical column order
            # under ShardTensor2dMesh, unlike bank-sharded), N/K zero-padded to the
            # ring-40 width 2560, and reshard them to L1 width-sharded JUST-IN-TIME
            # inside ``_mlp_decode_qwen36`` (fused branch only).
            _ring40_w = self.model_config["FF_N_RING40"]  # 2560 (per-device ring width)
            _ax0 = args.cluster_shape[0]  # 8 (mesh row axis; dims[0] of ShardTensor2dMesh)

            def _pad_ring40(name, dim):
                # ``torch_weight`` returns the transposed weight ``[in(K), out(N)]``.
                # ShardTensor2dMesh dims=(d0,d1): d0 -> mesh axis-0 (8 rows). The
                # per-device matmul dim (N for w1/w3, K for w2) is split across axis-0,
                # native 2176 per device. Pad the FULL tensor on that axis by
                # ``(2560 - 2176) * 8`` so each device lands at 2560.
                w = torch_weight(name[:2])
                d0 = dim[0]
                native_per_dev = w.shape[d0] // _ax0
                if native_per_dev < _ring40_w:
                    pad_full = (_ring40_w - native_per_dev) * _ax0
                    if d0 == -1:
                        w = torch.nn.functional.pad(w, (0, pad_full))
                    else:  # d0 == -2
                        w = torch.nn.functional.pad(w, (0, 0, 0, pad_full))
                return w.unsqueeze(0).unsqueeze(0)

            as_interleaved_ring40 = lambda name, type, dim: ttnn.as_tensor(
                _pad_ring40(name, dim),
                dtype=ttnn.bfloat16 if _mlp_force_bf16 else type,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name + "_ring40_ildram2560"),
            )
            self.w1_ring40 = as_interleaved_ring40(
                "w1_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, w1_dim
            )
            self.w3_ring40 = as_interleaved_ring40(
                "w3_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b, w1_dim
            )
            self.w2_ring40 = as_interleaved_ring40("w2_sharded", ttnn.bfloat8_b, w2_dim)

            self.prefetch(prefetcher_setup, tt_ccl)

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        if tt_ccl.mode == "decode":
            self.prefetcher_setup.insert_tensor(self.w1)
            self.prefetcher_setup.insert_tensor(self.w3)
            self.prefetcher_setup.insert_tensor(self.w2)
        self.tt_ccl = tt_ccl

    def forward(self, x: ttnn.Tensor, mode, batch_size=1) -> ttnn.Tensor:
        if mode == "prefill":
            return self.forward_prefill(x, batch_size=batch_size)

        pc_1_3 = self.model_config["FF1_3_TG_RING_PROGCFG"]
        pc_2 = self.model_config["FF2_TG_RING_PROGCFG"]

        # Blocker-#1 decode probe: mirror the prefill bf16 fix on the traced decode
        # path. Hold the gate/up matmul outputs + their reduce_scatter SUM in bf16
        # (the precision-critical partial-K sum), and route the SiLU*up product
        # through the pre-allocated bf16 all-gather buffer (BINARY_MUL_BF16). The
        # down-proj + FF2 all_reduce stay bf8 (buffer-pinned). Gated default-off.
        _ccl_dt = ttnn.bfloat16 if os.environ.get("QWEN36_MLP_CCL_BF16", "0") == "1" else ttnn.bfloat8_b
        _ag_buf_key = "BINARY_MUL_BF16" if _ccl_dt == ttnn.bfloat16 else "BINARY_MUL"

        compute_kernel = (
            self.args.compute_kernel_config_lofi if self.four_bit_mlp else self.args.compute_kernel_config_hifi2
        )

        # BH fused matmul+reduce_scatter (llama_rs_matmul). The prefetcher's global_cb
        # is NOT required on BH — unit-test verified (llama_rs_matmul PASS, all_gather_matmul
        # PCC 0.99999) at num_links=2, global_cb=None. Gated default-OFF until 64L coherence
        # is confirmed (the w2-LAR precedent broke 64L coherence even with single-layer PCC
        # passing — validate at full depth). Set QWEN36_FUSE_RS_MATMUL=1 to enable.
        _fuse_bh_rs = (not self.model_config["USE_PREFETCHER"]) and os.environ.get("QWEN36_FUSE_RS_MATMUL", "0") == "1"
        # FA-only probe: only fuse on full-attention layers (no recurrent in-layer).
        if _fuse_bh_rs and os.environ.get("QWEN36_FUSE_RS_FA_ONLY", "0") == "1":
            _fuse_bh_rs = self.is_full_attn_layer

        if self.model_config["USE_PREFETCHER"]:
            w1_out_reduced, w3_out = self.tt_ccl.double_matmul_line_reduce_scatter(
                x,
                self.w1,
                self.w3,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                RS_memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                compute_kernel_config=compute_kernel,
                dtype=ttnn.bfloat8_b,
                program_config=pc_1_3,
                memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                global_cb=self.prefetcher_setup.global_circular_buffer,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
                use_noc1_only=False,
            )
            ttnn.deallocate(x)
        elif _fuse_bh_rs:
            # BH fused path: keep matmul+reduce_scatter fused (llama_rs_matmul) but at a
            # LOWER single-program L1 peak than double_matmul (2 matmuls + RS in one program),
            # which overflowed col-0 L1 (896B) and clashed with the next layer's origin-anchored
            # recurrent. Here: plain matmul w1 (program A), then fuse w3's matmul with w1's
            # reduce-scatter (program B = 1 matmul + 1 RS). w3's RS is the shared code below.
            w1_out = ttnn.linear(
                x,
                self.w1,
                compute_kernel_config=compute_kernel,
                dtype=ttnn.bfloat8_b,
                program_config=pc_1_3,
                memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
            )
            w1_out_reduced, w3_out = self.tt_ccl.matmul_line_reduce_scatter(
                x,
                self.w3,
                w1_out,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                RS_memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                compute_kernel_config=compute_kernel,
                dtype=ttnn.bfloat8_b,
                program_config=pc_1_3,
                memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                global_cb=None,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
                use_noc1_only=False,
            )
            ttnn.deallocate(x)
            ttnn.deallocate(w1_out)
        else:
            # BH path: no global_cb, so use separate matmuls + reduce-scatter
            # double_matmul_line_reduce_scatter fails on BH because it passes 3 tensors
            # to the internal matmul, which requires global_cb to use multi-tensor path
            w1_out = ttnn.linear(
                x,
                self.w1,
                compute_kernel_config=compute_kernel,
                dtype=_ccl_dt,
                program_config=pc_1_3,
                memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
            )
            w3_out = ttnn.linear(
                x,
                self.w3,
                compute_kernel_config=compute_kernel,
                dtype=_ccl_dt,
                program_config=pc_1_3,
                memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
                sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
            )
            ttnn.deallocate(x)
            w1_out_reduced = self.tt_ccl.line_reduce_scatter(
                w1_out,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
                use_noc1_only=False,
            )
            ttnn.deallocate(w1_out)

        w3_out_reduced = self.tt_ccl.line_reduce_scatter(
            w3_out,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
            use_noc1_only=False,
        )
        ttnn.deallocate(w3_out)

        ff1ff3 = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=_ccl_dt,
            memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
        )

        ttnn.deallocate(w3_out_reduced)
        ttnn.deallocate(w1_out_reduced)

        w2_in = self.tt_ccl.line_all_gather(
            ff1ff3,
            dim=3,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
            buffer_key=_ag_buf_key,
            use_optimal_ccl_for_llama=False if mode == "prefill" else True,
        )

        ttnn.deallocate(ff1ff3)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            dtype=_ccl_dt,
            program_config=pc_2,
            memory_config=self.model_config["FF2_OUT_RING_MEMCFG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
        )
        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )
        ttnn.deallocate(w2_out)

        return w2_out_reduced

    def forward_prefill(self, x: ttnn.Tensor, batch_size=1) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        # Blocker-#1 probe: TP=4 (working 27B) keeps the entire SwiGLU intermediate
        # (gate/up partial-sum reduction + SiLU*up product) in bf16 and does a single
        # bf16 all_reduce; galaxy_v2 reduces the partial sums and runs the activation
        # product in bf8_b at every layer. bf8 accumulation noise corrupts the logit
        # distribution tail (kills sampling, EOS-first-token on long/video prefill).
        # Set QWEN36_MLP_CCL_BF16=1 to hold the prefill FF intermediate in bf16 (DRAM,
        # safe to widen) and confirm/refute that as the root cause.
        _ccl_dt = ttnn.bfloat16 if os.environ.get("QWEN36_MLP_CCL_BF16", "0") == "1" else ttnn.bfloat8_b
        # bf16 gate/up reduce_scatter is ONLY taken on the ttnn.linear path
        # (seq_len < 4096 or batch_size > 1); the seq_len >= 4096 path uses
        # minimal_matmul which always outputs bf8. Only on the bf16 path do we use a
        # "_BF16"-suffixed buffer key so the ring reduce_scatter MISSES the dtype-pinned
        # bf8 persistent buffer and falls back to a fresh (bf16) allocation. Using the
        # "_BF16" key on the seq_len >= 4096 (bf8) path forces the no-persistent-buffer
        # ring path, which DEADLOCKS at large seqlen — so gate the key on the same
        # condition as the bf16 matmul output.
        _w1w3_bf16 = (_ccl_dt == ttnn.bfloat16) and (seq_len < 4096 or batch_size > 1)
        _ff1_rs_key = "FF1_BF16" if _w1w3_bf16 else "FF1"
        _ff3_rs_key = "FF3_BF16" if _w1w3_bf16 else "FF3"
        use_w1_w3_interleaved = (seq_len >= 4096 or seq_len == 128) if not self.args.is_qwen else True
        short_lens_pc_1_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len, use_w1_w3_interleaved)
        short_lens_pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)

        minimal_pc_1_3 = self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"](seq_len)
        minimal_pc_2 = self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"](seq_len)

        if 1024 <= seq_len < 4096:
            x = ttnn.reshape(x, (1, seq_len // 1024, 1024, -1))

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096 or batch_size > 1:
            w1_out = ttnn.linear(
                x,
                self.w1_interleaved if use_w1_w3_interleaved else self.w1,
                compute_kernel_config=(
                    self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2_fp16
                ),
                dtype=_ccl_dt,
                program_config=short_lens_pc_1_3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            w1_out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.w1_interleaved if use_w1_w3_interleaved else self.w1,
                config=minimal_pc_1_3,
                compute_kernel_config=self.args.compute_kernel_config_lofi,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w1_out_reduced = self.tt_ccl.line_reduce_scatter(
            w1_out,
            cluster_axis=1,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=w1_out.memory_config(),
            buffer_key=_ff1_rs_key,
            dim=3,
            batch_size=batch_size,
        )
        ttnn.deallocate(w1_out)

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096 or batch_size > 1:
            w3_out = ttnn.linear(
                x,
                self.w3_interleaved if use_w1_w3_interleaved else self.w3,
                compute_kernel_config=(
                    self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2_fp16
                ),
                dtype=_ccl_dt,
                program_config=short_lens_pc_1_3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            w3_out = ttnn.experimental.minimal_matmul(
                input_tensor=x,
                weight_tensor=self.w3_interleaved if use_w1_w3_interleaved else self.w3,
                config=minimal_pc_1_3,
                compute_kernel_config=self.args.compute_kernel_config_lofi,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(x)
        w3_out_reduced = self.tt_ccl.line_reduce_scatter(
            w3_out,
            cluster_axis=1,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=w3_out.memory_config(),
            buffer_key=_ff3_rs_key,
            dim=3,
            batch_size=batch_size,
        )
        ttnn.deallocate(w3_out)
        # Cast the SiLU*up product back to bf8 here: the gate/up partial-K SUM
        # (reduce_scatter above) is the precision-critical op and now runs in bf16,
        # but line_all_gather below uses a dtype-pinned (bf8) persistent buffer and
        # the gather is non-summing (pure concatenation), so a single bf8 cast here
        # is harmless and avoids the buffer-dtype mismatch.
        w2_in = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        w2_in_gathered = self.tt_ccl.line_all_gather(
            w2_in,
            cluster_axis=1,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=w3_out.memory_config(),
            buffer_key="FF3",
            dim=3,
        )
        ttnn.deallocate(w2_in)

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096 or batch_size > 1:
            w2_out = ttnn.linear(
                w2_in_gathered,
                self.w2_interleaved,
                compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
                program_config=short_lens_pc_2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            w2_out = ttnn.experimental.minimal_matmul(
                input_tensor=w2_in_gathered,
                weight_tensor=self.w2_interleaved,
                config=minimal_pc_2,
                compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=min(3, self.model_config["GALAXY_NUM_LINKS"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="FF2",
            batch_size=batch_size,
        )
        ttnn.deallocate(w2_out)

        if 1024 <= seq_len < 4096:
            original_shape = w2_out_reduced.shape
            w2_out_reduced = ttnn.reshape(
                w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        # qwen3.6 residual-stream dtype lock (olmo session-11 lesson):
        # the MLP's reduction output flows directly into the post-MLP residual
        # add in TtTransformerBlock.forward. Even though w1/w2/w3 matmuls cast
        # to bfloat8_b for L1 footprint, the residual stream is more accurate
        # when held in bfloat16. The 4L test still passes with this typecast;
        # the 64L per-layer sweep shows real-prompt-position PCC > 0.998 at
        # every layer with this in place.
        if getattr(self.args, "is_qwen36", False) and w2_out_reduced.dtype != ttnn.bfloat16:
            w2_out_bf16 = ttnn.typecast(w2_out_reduced, dtype=ttnn.bfloat16)
            ttnn.deallocate(w2_out_reduced)
            w2_out_reduced = w2_out_bf16

        return w2_out_reduced
