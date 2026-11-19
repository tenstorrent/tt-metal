# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Q/K RMSNorm D256 plus NeoX RoPE on first64 channels, in native head shards.

Adapted from models/demos/gemma4/custom_ops/qkv_norm_rope. V is untouched;
normalization packs BF16 before partial rotation, matching the decoder contract.
"""
import struct
from pathlib import Path
import ttnn

KERNEL_DIR=Path(__file__).resolve().parent/'kernels/partial_norm_rope'

def partial_norm_rope(q,k,gq,gk,cos,sin,epsilon):
    batch=int(q.shape[1]);grid=q.memory_config().shard_spec.grid
    assert k.memory_config().shard_spec.grid==grid
    assert q.dtype==k.dtype==ttnn.bfloat16 and q.shape[-1]==k.shape[-1]==256
    assert cos.shape[-1]==sin.shape[-1]==64
    cores=[]
    for r in grid.ranges():
        for y in range(r.start.y,r.end.y+1):
            for x in range(r.start.x,r.end.x+1):cores.append(ttnn.CoreCoord(x,y))
    assert len(cores)==batch,(len(cores),batch)
    def cb(idx,pages,dtype=ttnn.bfloat16):
        page=4096 if dtype==ttnn.float32 else 2048
        return ttnn.CBDescriptor(total_size=pages*page,core_ranges=grid,
               format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx,data_format=dtype,page_size=page)])
    cbs=[ttnn.cb_descriptor_from_sharded_tensor(0,q),ttnn.cb_descriptor_from_sharded_tensor(1,k),
         ttnn.cb_descriptor_from_sharded_tensor(16,q),ttnn.cb_descriptor_from_sharded_tensor(17,k),
         cb(3,8),cb(4,8),cb(5,2),cb(6,2),cb(7,2),cb(8,8,ttnn.float32),cb(9,1,ttnn.float32),cb(10,8)]
    reader_ct=[0,1,2,3,4,5,6,7,8]
    for t in (gq,gk,cos,sin):reader_ct+=ttnn.TensorAccessorArgs(t).get_compile_time_args()
    def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
    compute_ct=[0,1,2,16,17,18,3,4,5,6,7,8,9,8,1,bits(epsilon),bits(1/256),10]
    rr,wr,cr=ttnn.RuntimeArgs(),ttnn.RuntimeArgs(),ttnn.RuntimeArgs()
    for user,core in enumerate(cores):
        rr[core.x][core.y]=[gq.buffer_address(),gk.buffer_address(),cos.buffer_address(),sin.buffer_address(),user]
        wr[core.x][core.y]=[0];cr[core.x][core.y]=[0]
    kernels=[ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR/'reader.cpp'),core_ranges=grid,compile_time_args=reader_ct,
                 runtime_args=rr,config=ttnn.ReaderConfigDescriptor()),
             ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR/'writer.cpp'),core_ranges=grid,compile_time_args=[16,17,18,8],
                 runtime_args=wr,config=ttnn.WriterConfigDescriptor()),
             ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR/'compute.cpp'),core_ranges=grid,compile_time_args=compute_ct,
                 runtime_args=cr,config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                      fp32_dest_acc_en=True,dst_full_sync_en=True,math_approx_mode=False))]
    ttnn.generic_op([q,k,gq,gk,cos,sin],ttnn.ProgramDescriptor(kernels=kernels,cbs=cbs,semaphores=[]))
    return q,k
