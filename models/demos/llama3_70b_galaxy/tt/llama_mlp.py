# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
import torch.nn.functional as F

from models.common.utility_functions import comp_pcc, comp_allclose

import os


def compare_per_device(ttnn_tensor, torch_expected, mesh_device, op_name, is_input=False):
    """
    Compare ttnn tensor against torch expected value device-by-device.
    Prints PCC for each device.

    Args:
        ttnn_tensor: The ttnn tensor distributed across devices
        torch_expected: The expected torch tensor (full, will be sliced per device)
        mesh_device: The mesh device
        op_name: Name of the operation for logging
        is_input: Whether this is an input tensor (for logging purposes)
    """
    device_tensors = ttnn.get_device_tensors(ttnn_tensor)
    tensor_type = "input" if is_input else "output"

    print(f"\n{'='*60}")
    print(f"PCC Comparison for {op_name} ({tensor_type})")
    print(f"{'='*60}")

    for device_idx, device_tensor in enumerate(device_tensors):
        # Convert device tensor to torch
        device_torch = ttnn.to_torch(device_tensor)

        # Get the corresponding slice from torch_expected based on device index
        # The slicing depends on how the tensor is distributed across devices
        num_devices = len(device_tensors)

        # Determine which dimension is sharded and slice accordingly
        if torch_expected.shape == device_torch.shape:
            # No sharding, full tensor on each device
            expected_slice = torch_expected
        else:
            # Try to match shapes by finding the sharded dimension
            for dim in range(len(torch_expected.shape)):
                if torch_expected.shape[dim] != device_torch.shape[dim]:
                    # This dimension is sharded
                    shard_size = device_torch.shape[dim]
                    start_idx = device_idx * shard_size
                    end_idx = start_idx + shard_size
                    expected_slice = torch.narrow(torch_expected, dim, start_idx, shard_size)
                    break
            else:
                expected_slice = torch_expected

        # Ensure shapes match for comparison
        if expected_slice.shape != device_torch.shape:
            print(f"  Device {device_idx}: Shape mismatch - expected {expected_slice.shape}, got {device_torch.shape}")
            continue

        # Compute PCC
        passing, pcc_message = comp_pcc(expected_slice, device_torch)
        allclose_result = comp_allclose(expected_slice, device_torch)
        status = "✓ PASS" if passing else "✗ FAIL"
        print(f"  Device {device_idx}: {status} | PCC: {pcc_message} | {allclose_result}")

    print(f"{'='*60}\n")


def run_torch_linear(input_torch, weight_torch, has_silu=False):
    """Run torch equivalent of linear operation."""
    # weight_torch is expected to be in shape [..., in_features, out_features] (already transposed)
    result = torch.matmul(input_torch.float(), weight_torch.float())
    if has_silu:
        result = torch.nn.functional.silu(result)
    return result


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
        reference_model=None,
    ):
        super().__init__()

        self.reference_model = reference_model
        self.layer_num = layer_num

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
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
        as_sharded_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=type if not self.args.is_qwen else ttnn.bfloat8_b,
            # dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dim, mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config if "w2" in name else w1_w3_mem_config,
            cache_file_name=cache_name(name),
        )

        as_interleaved_tensor = lambda name, type, dim: ttnn.as_tensor(
            torch_weight(name[:2]).unsqueeze(0).unsqueeze(0),  # Grab only the wX part of the name
            dtype=type,
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
            "w1_sharded", ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat16, dim=w1_dim
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

        if tt_ccl.mode == "decode":
            self.prefetch(prefetcher_setup, tt_ccl)

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        # if tt_ccl.mode == "decode" and not tt_ccl.is_qwen:
        if tt_ccl.mode == "decode":
            self.prefetcher_setup.insert_tensor(self.w1)
            self.prefetcher_setup.insert_tensor(self.w3)
            self.prefetcher_setup.insert_tensor(self.w2)
        self.tt_ccl = tt_ccl

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        if mode == "prefill":
            return self.forward_prefill(x, mode)

        pc_1_3 = self.model_config["FF1_3_TG_RING_PROGCFG"]
        pc_2 = self.model_config["FF2_TG_RING_PROGCFG"]

        debug_pcc = os.environ.get("DEBUG_PCC") == "1"

        # ========== W1 Linear ==========
        # Convert inputs to torch before operation for per-device comparison
        if debug_pcc:
            x_torch_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(x)]
            w1_torch_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(self.w1)]

        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            program_config=pc_1_3,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            core_grid=None,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
        )

        # Per-device PCC comparison for W1
        # if debug_pcc:
        #     w1_out_device_tensors = ttnn.get_device_tensors(w1_out)
        #     print(f"\n{'='*80}")
        #     print(f"W1 LINEAR - Per-Device PCC Comparison (Layer {self.layer_num})")
        #     print(f"{'='*80}")
        #     for device_idx, (x_torch, w1_torch, w1_out_tt) in enumerate(
        #         zip(x_torch_tensors, w1_torch_tensors, w1_out_device_tensors)
        #     ):
        #         # Compute torch reference for this device
        #         w1_out_torch = torch.matmul(x_torch.float(), w1_torch.float())
        #         # Convert ttnn output to torch
        #         w1_out_tt_torch = ttnn.to_torch(w1_out_tt)

        #         # Input PCC
        #         print(f"  Device {device_idx}:")
        #         print(f"    Input shape: {x_torch.shape}, Weight shape: {w1_torch.shape}")
        #         print(f"    Output shapes - Torch: {w1_out_torch.shape}, TTNN: {w1_out_tt_torch.shape}")

        #         # Output PCC
        #         passing, pcc_message = comp_pcc(w1_out_torch, w1_out_tt_torch)
        #         allclose_result = comp_allclose(w1_out_torch, w1_out_tt_torch)
        #         status = "✓ PASS" if passing else "✗ FAIL"
        #         print(f"    Output PCC: {status} | {pcc_message} | {allclose_result}")

        w1_out_reduced = self.tt_ccl.line_reduce_scatter(
            w1_out,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
            use_noc1_only=False,
        )
        ttnn.deallocate(w1_out)

        # ========== W3 Linear ==========
        # Convert inputs to torch before operation for per-device comparison
        # if debug_pcc:
        #     w3_torch_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(self.w3)]

        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            program_config=pc_1_3,
            memory_config=self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1_3 else None,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
        )
        ttnn.deallocate(x)

        # Per-device PCC comparison for W3
        # if debug_pcc:
        #     w3_out_device_tensors = ttnn.get_device_tensors(w3_out)
        #     print(f"\n{'='*80}")
        #     print(f"W3 LINEAR - Per-Device PCC Comparison (Layer {self.layer_num})")
        #     print(f"{'='*80}")
        #     for device_idx, (x_torch, w3_torch, w3_out_tt) in enumerate(
        #         zip(x_torch_tensors, w3_torch_tensors, w3_out_device_tensors)
        #     ):
        #         # Compute torch reference for this device
        #         w3_out_torch = torch.matmul(x_torch.float(), w3_torch.float())
        #         # Convert ttnn output to torch
        #         w3_out_tt_torch = ttnn.to_torch(w3_out_tt)

        #         print(f"  Device {device_idx}:")
        #         print(f"    Input shape: {x_torch.shape}, Weight shape: {w3_torch.shape}")
        #         print(f"    Output shapes - Torch: {w3_out_torch.shape}, TTNN: {w3_out_tt_torch.shape}")

        #         # Output PCC
        #         passing, pcc_message = comp_pcc(w3_out_torch, w3_out_tt_torch)
        #         allclose_result = comp_allclose(w3_out_torch, w3_out_tt_torch)
        #         status = "✓ PASS" if passing else "✗ FAIL"
        #         print(f"    Output PCC: {status} | {pcc_message} | {allclose_result}")

        w3_out_reduced = self.tt_ccl.line_reduce_scatter(
            w3_out,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
            use_noc1_only=False,
        )
        ttnn.deallocate(w3_out)

        # ========== SiLU(W1) * W3 ==========
        # if debug_pcc:
        #     w1_reduced_torch_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(w1_out_reduced)]
        #     w3_reduced_torch_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(w3_out_reduced)]

        use_torch_mul = os.environ.get("USE_TORCH_MUL") == "1"

        if use_torch_mul:
            # Convert inputs to torch, do SiLU * mul in torch, convert back to ttnn
            mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])

            # Convert w1_out_reduced and w3_out_reduced to torch
            w1_torch = ttnn.to_torch(w1_out_reduced, mesh_composer=mesh_composer)
            w3_torch = ttnn.to_torch(w3_out_reduced, mesh_composer=mesh_composer)

            # Apply SiLU to w1 and multiply with w3
            ff1ff3_torch = torch.nn.functional.silu(w1_torch.float()) * w3_torch.float()

            # Convert back to ttnn with proper sharding
            mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 3), mesh_shape=list(self.mesh_device.shape))

            ff1ff3 = ttnn.from_torch(
                ff1ff3_torch,
                device=self.mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=self.model_config["REDUCE_SCATTER_OUT_MEMCFG"],
            )
        else:
            # # Save inputs and outputs for layers 6 and 7
            # if self.layer_num in [6, 7]:
            #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
            #     # Save w1_out_reduced (input to silu)
            #     w1_out_reduced_torch = ttnn.to_torch(w1_out_reduced, mesh_composer=mesh_composer)[:1, :, :, :]
            #     torch.save(w1_out_reduced_torch, f"w1_out_reduced_layer{self.layer_num}.pt")
            #     print(f"Saved w1_out_reduced_layer{self.layer_num}.pt, shape: {w1_out_reduced_torch.shape}")

            #     # Save w3_out_reduced
            #     w3_out_reduced_torch = ttnn.to_torch(w3_out_reduced, mesh_composer=mesh_composer)[:1, :, :, :]
            #     torch.save(w3_out_reduced_torch, f"w3_out_reduced_layer{self.layer_num}.pt")
            #     print(f"Saved w3_out_reduced_layer{self.layer_num}.pt, shape: {w3_out_reduced_torch.shape}")

            w1_out_reduced_silu_fp32 = ttnn.typecast(w1_out_reduced, ttnn.float32)
            w1_out_reduced_silu_out_fp32 = ttnn.silu(w1_out_reduced_silu_fp32)
            ttnn.deallocate(w1_out_reduced_silu_fp32)

            # Cast inputs to float32 for higher precision mul
            w3_out_reduced_fp32 = ttnn.typecast(w3_out_reduced, ttnn.float32)
            ttnn.deallocate(w3_out_reduced)

            ff1ff3_fp32 = ttnn.mul(
                w1_out_reduced_silu_out_fp32,
                w3_out_reduced_fp32,
                dtype=ttnn.float32,
                memory_config=w1_out_reduced.memory_config(),
            )

            # Cast output back to bfloat8_b
            ff1ff3 = ttnn.typecast(ff1ff3_fp32, ttnn.bfloat8_b)
            ttnn.deallocate(ff1ff3_fp32)
            ttnn.deallocate(w1_out_reduced_silu_out_fp32)
            ttnn.deallocate(w3_out_reduced_fp32)

            # Save ff1ff3 output for layers 6 and 7
            # if self.layer_num in [6, 7]:
            #     mesh_composer = ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 3), mesh_shape=[8, 4])
            #     ff1ff3_torch = ttnn.to_torch(ff1ff3, mesh_composer=mesh_composer)[:1, :, :, :]
            #     torch.save(ff1ff3_torch, f"ff1ff3_layer{self.layer_num}.pt")
            #     print(f"Saved ff1ff3_layer{self.layer_num}.pt, shape: {ff1ff3_torch.shape}")
        # Per-device PCC comparison for SiLU * mul
        # if debug_pcc:
        #     ff1ff3_device_tensors = ttnn.get_device_tensors(ff1ff3)
        #     print(f"\n{'='*80}")
        #     print(f"SiLU(W1) * W3 - Per-Device PCC Comparison (Layer {self.layer_num})")
        #     print(f"{'='*80}")
        #     for device_idx, (w1_red_torch, w3_red_torch, ff1ff3_tt) in enumerate(
        #         zip(w1_reduced_torch_tensors, w3_reduced_torch_tensors, ff1ff3_device_tensors)
        #     ):
        #         # Compute torch reference: SiLU(w1_reduced) * w3_reduced
        #         ff1ff3_torch = torch.nn.functional.silu(w1_red_torch.float()) * w3_red_torch.float()
        #         # Convert ttnn output to torch
        #         ff1ff3_tt_torch = ttnn.to_torch(ff1ff3_tt)

        #         print(f"  Device {device_idx}:")
        #         print(f"    W1_reduced shape: {w1_red_torch.shape}, W3_reduced shape: {w3_red_torch.shape}")
        #         print(f"    Output shapes - Torch: {ff1ff3_torch.shape}, TTNN: {ff1ff3_tt_torch.shape}")

        #         # Output PCC
        #         passing, pcc_message = comp_pcc(ff1ff3_torch, ff1ff3_tt_torch)
        #         allclose_result = comp_allclose(ff1ff3_torch, ff1ff3_tt_torch)
        #         status = "✓ PASS" if passing else "✗ FAIL"
        #         print(f"    Output PCC: {status} | {pcc_message} | {allclose_result}")

        ttnn.deallocate(w3_out_reduced)
        ttnn.deallocate(w1_out_reduced)

        w2_in = self.tt_ccl.line_all_gather(
            ff1ff3,
            dim=3,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
            buffer_key="BINARY_MUL",
            use_optimal_ccl_for_llama=False if mode == "prefill" else True,
        )

        ttnn.deallocate(ff1ff3)

        # ========== W2 Linear ==========
        # if debug_pcc:
        #     w2_in_torch_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(w2_in)]
        #     w2_torch_tensors = [ttnn.to_torch(dt) for dt in ttnn.get_device_tensors(self.w2)]

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            program_config=pc_2,
            memory_config=self.model_config["FF2_OUT_RING_MEMCFG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if mode == "decode" else None,
        )

        # Per-device PCC comparison for W2
        # if debug_pcc:
        #     w2_out_device_tensors = ttnn.get_device_tensors(w2_out)
        #     print(f"\n{'='*80}")
        #     print(f"W2 LINEAR - Per-Device PCC Comparison (Layer {self.layer_num})")
        #     print(f"{'='*80}")
        #     for device_idx, (w2_in_torch, w2_torch, w2_out_tt) in enumerate(
        #         zip(w2_in_torch_tensors, w2_torch_tensors, w2_out_device_tensors)
        #     ):
        #         # Compute torch reference for this device
        #         w2_out_torch = torch.matmul(w2_in_torch.float(), w2_torch.float())
        #         # Convert ttnn output to torch
        #         w2_out_tt_torch = ttnn.to_torch(w2_out_tt)

        #         print(f"  Device {device_idx}:")
        #         print(f"    Input shape: {w2_in_torch.shape}, Weight shape: {w2_torch.shape}")
        #         print(f"    Output shapes - Torch: {w2_out_torch.shape}, TTNN: {w2_out_tt_torch.shape}")

        #         # Output PCC
        #         passing, pcc_message = comp_pcc(w2_out_torch, w2_out_tt_torch)
        #         allclose_result = comp_allclose(w2_out_torch, w2_out_tt_torch)
        #         status = "✓ PASS" if passing else "✗ FAIL"
        #         print(f"    Output PCC: {status} | {pcc_message} | {allclose_result}")

        w2_out_reduced = self.tt_ccl.line_all_reduce(
            w2_out,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
            use_optimal_ccl_for_llama=True,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(w2_out)

        return w2_out_reduced

    def forward_prefill(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        use_w1_w3_interleaved = (seq_len >= 4096 or seq_len == 128) if not self.args.is_qwen else True
        short_lens_pc_1_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len, use_w1_w3_interleaved)
        short_lens_pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)

        minimal_pc_1_3 = self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"](seq_len)
        minimal_pc_2 = self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"](seq_len)

        if 1024 <= seq_len < 4096:
            x = ttnn.reshape(x, (1, seq_len // 1024, 1024, -1))

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096:
            w1_out = ttnn.linear(
                x,
                self.w1_interleaved if use_w1_w3_interleaved else self.w1,
                compute_kernel_config=(
                    self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2_fp16
                ),
                dtype=ttnn.bfloat8_b,
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
            w1_out, cluster_axis=1, num_links=3, memory_config=w1_out.memory_config(), buffer_key="FF1", dim=3
        )
        ttnn.deallocate(w1_out)

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096:
            w3_out = ttnn.linear(
                x,
                self.w3_interleaved if use_w1_w3_interleaved else self.w3,
                compute_kernel_config=(
                    self.args.compute_kernel_config_lofi
                    if self.four_bit_mlp
                    else self.args.compute_kernel_config_hifi2_fp16
                ),
                dtype=ttnn.bfloat8_b,
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
            w3_out, cluster_axis=1, num_links=3, memory_config=w3_out.memory_config(), buffer_key="FF3", dim=3
        )
        ttnn.deallocate(w3_out)
        w2_in = ttnn.mul(
            w1_out_reduced,
            w3_out_reduced,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )
        w2_in_gathered = self.tt_ccl.line_all_gather(
            w2_in, cluster_axis=1, num_links=3, memory_config=w3_out.memory_config(), buffer_key="FF3", dim=3
        )
        ttnn.deallocate(w2_in)

        # For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096:
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
            w2_out, cluster_axis=0, num_links=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="FF2"
        )
        ttnn.deallocate(w2_out)

        if 1024 <= seq_len < 4096:
            original_shape = w2_out_reduced.shape
            w2_out_reduced = ttnn.reshape(
                w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        return w2_out_reduced
