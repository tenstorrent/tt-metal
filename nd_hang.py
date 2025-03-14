# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import time

import numpy as np


def interpolate_single_shape():
    """
    Implements the following graph in ttnn:
    ```mlir
        #device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
        #dram = #ttnn.buffer_type<dram>
        #loc1 = loc("arg0_1")
        #system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>
        #ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 50 + d1 * 50 + d2, d3), <1x1>, memref<2x7x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 50 + d1, d2), <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 224 + d1, d2), <1x1>, memref<7x14x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 100 + d1 * 100 + d2, d3), <1x1>, memref<4x14x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 224 + d1 * 224 + d2, d3), <1x1>, memref<7x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 224 + d1, d2), <1x1>, memref<7x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 224 + d1, d2), <1x1>, memref<7x4x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 224 + d1 * 224 + d2, d3), <1x1>, memref<7x4x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 100 + d1 * 100 + d2, d3), <1x1>, memref<4x7x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 100 + d1, d2), <1x1>, memref<4x7x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        #ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 100 + d1, d2), <1x1>, memref<4x14x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
        module {
        tt.device_module {
            builtin.module attributes {tt.device = #device, tt.system_desc = #system_desc} {
            func.func @main(%arg0: tensor<1x1x50x224xbf16, #ttnn_layout> loc("arg0_1"), %arg1: tensor<1x50x100xbf16, #ttnn_layout1> loc("arg0_1"), %arg2: tensor<1x224x448xbf16, #ttnn_layout2> loc("arg0_1")) -> tensor<1x1x100x448xbf16, #ttnn_layout3> {
                %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x50x224xbf16, #ttnn_layout>) -> tensor<1x1x224x50xbf16, #ttnn_layout4> loc(#loc2)
                %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 224 : i32, 50 : i32]}> : (tensor<1x1x224x50xbf16, #ttnn_layout4>) -> tensor<1x224x50xbf16, #ttnn_layout5> loc(#loc3)
                "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x1x224x50xbf16, #ttnn_layout4>) -> () loc(#loc3)
                %2 = "ttnn.matmul"(%1, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x224x50xbf16, #ttnn_layout5>, tensor<1x50x100xbf16, #ttnn_layout1>) -> tensor<1x224x100xbf16, #ttnn_layout6> loc(#loc4)
                "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x224x50xbf16, #ttnn_layout5>) -> () loc(#loc4)
                %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 1 : i32, 224 : i32, 100 : i32]}> : (tensor<1x224x100xbf16, #ttnn_layout6>) -> tensor<1x1x224x100xbf16, #ttnn_layout7> loc(#loc5)
                "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x224x100xbf16, #ttnn_layout6>) -> () loc(#loc5)
                %4 = "ttnn.permute"(%3) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x224x100xbf16, #ttnn_layout7>) -> tensor<1x1x100x224xbf16, #ttnn_layout8> loc(#loc6)
                "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x224x100xbf16, #ttnn_layout7>) -> () loc(#loc6)
                %5 = "ttnn.reshape"(%4) <{shape = [1 : i32, 100 : i32, 224 : i32]}> : (tensor<1x1x100x224xbf16, #ttnn_layout8>) -> tensor<1x100x224xbf16, #ttnn_layout9> loc(#loc7)
                "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x100x224xbf16, #ttnn_layout8>) -> () loc(#loc7)
                %6 = "ttnn.matmul"(%5, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x100x224xbf16, #ttnn_layout9>, tensor<1x224x448xbf16, #ttnn_layout2>) -> tensor<1x100x448xbf16, #ttnn_layout10> loc(#loc8)
                "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x100x224xbf16, #ttnn_layout9>) -> () loc(#loc8)
                %7 = "ttnn.reshape"(%6) <{shape = [1 : i32, 1 : i32, 100 : i32, 448 : i32]}> : (tensor<1x100x448xbf16, #ttnn_layout10>) -> tensor<1x1x100x448xbf16, #ttnn_layout3> loc(#loc9)
                "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x100x448xbf16, #ttnn_layout10>) -> () loc(#loc9)
                return %7 : tensor<1x1x100x448xbf16, #ttnn_layout3> loc(#loc10)
            } loc(#loc1)
            } loc(#loc)
        } loc(#loc)
        } loc(#loc)
        #loc = loc(unknown)
        #loc2 = loc("transpose_int")
        #loc3 = loc("view_default")
        #loc4 = loc("bmm_default")
        #loc5 = loc("view_default_1")
        #loc6 = loc("transpose_int_1")
        #loc7 = loc("view_default_2")
        #loc8 = loc("bmm_default_1")
        #loc9 = loc("view_default_3")
        #loc10 = loc("output")
    ```
    """
    with ttnn.manage_device(0) as device:
        # ttnn.enable_program_cache(device)
        arg0 = ttnn.to_device(
            ttnn.from_torch(torch.zeros((1, 1, 50, 224), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT),
            device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
        arg1 = ttnn.to_device(
            ttnn.from_torch(torch.zeros((1, 50, 100), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT),
            device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
        arg2 = ttnn.to_device(
            ttnn.from_torch(torch.zeros((1, 224, 448), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT),
            device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
        t0 = ttnn.permute(arg0, (0, 1, 3, 2))
        t1 = ttnn.reshape(t0, (1, 224, 50))
        ttnn.deallocate(t0)
        t2 = ttnn.matmul(t1, arg1)
        ttnn.deallocate(t1)
        t3 = ttnn.reshape(t2, (1, 1, 224, 100))
        ttnn.deallocate(t2)
        t4 = ttnn.permute(t3, (0, 1, 3, 2))
        ttnn.deallocate(t3)
        t5 = ttnn.reshape(t4, (1, 100, 224))
        ttnn.deallocate(t4)
        t6 = ttnn.matmul(t5, arg2)
        ttnn.deallocate(t5)
        t7 = ttnn.reshape(t6, (1, 1, 100, 448))
        ttnn.deallocate(t6)
        return t7


def main():
    for run in range(100_000):
        # parametrize()
        print(interpolate_single_shape())
        with open("status.txt", "w") as f:
            f.write(str(run))


main()
