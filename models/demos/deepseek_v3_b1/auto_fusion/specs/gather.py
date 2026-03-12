# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gather micro-op specification.

Mirrors the contract in unified_kernels/gather.hpp:
  - NCRISC = Sender: sends data to receiver core via noc_async_write
  - BRISC = Receiver: waits for semaphores from all senders
  - TRISC = No-op (gather is dataflow only)

Gather::Op has no CTArgs type parameter — only bool template params:
  Op<IsSenderCore, IsReceiverCore, pop_src, UsePerCoreSenderIdx>

For scattered core layouts (like down_proj), UsePerCoreSenderIdx=true
and each core gets a per-core sender_idx via PerCoreCompileTimeDescriptor.
"""

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, CBPortSpec, MicroOpSpec, RISCContract

GATHER = MicroOpSpec(
    name="Gather",
    header="unified_kernels/gather.hpp",
    namespace="deepseek_b1_ops",
    struct_name="Gather",
    ncrisc=RISCContract(
        ct_args_type=None,  # Gather::Op has no CTArgs type parameter
        rt_args_type="deepseek_b1_ops::Gather::SenderArgs",
        named_ct_args=[
            "dest_noc_x",
            "dest_noc_y",
            "data_size_bytes",
            "receiver_semaphore_id",
            "src_cb",
            "src_num_pages",
            "sender_grid_start_x",
            "sender_grid_start_y",
            "sender_grid_end_x",
            "sender_grid_end_y",
            "row_major",
            "receiver_data_addr",
            "sender_idx",
        ],
        rt_args_fields=[
            ("dest_noc_x", "ct:dest_noc_x"),
            ("dest_noc_y", "ct:dest_noc_y"),
            ("data_size_bytes", "ct:data_size_bytes"),
            ("receiver_semaphore_addr", "semaphore:receiver_semaphore_id"),
            ("src_cb", "ct:src_cb"),
            ("src_num_pages", "ct:src_num_pages"),
            ("sender_grid_start_x", "ct:sender_grid_start_x"),
            ("sender_grid_start_y", "ct:sender_grid_start_y"),
            ("sender_grid_end_x", "ct:sender_grid_end_x"),
            ("sender_grid_end_y", "ct:sender_grid_end_y"),
            ("row_major", "ct:row_major"),
            ("receiver_data_addr", "ct:receiver_data_addr"),
            ("sender_idx", "ct:sender_idx"),
        ],
        cb_reads=["src"],
    ),
    brisc=RISCContract(
        ct_args_type=None,  # Gather::Op has no CTArgs type parameter
        rt_args_type="deepseek_b1_ops::Gather::ReceiverArgs",
        named_ct_args=[
            "noc0_num_senders",
            "noc1_num_senders",
            "noc0_receiver_semaphore_id",
            "noc1_receiver_semaphore_id",
            "dst_cb",
            "dst_num_pages",
        ],
        rt_args_fields=[
            ("noc0_num_senders", "ct:noc0_num_senders"),
            ("noc1_num_senders", "ct:noc1_num_senders"),
            ("noc0_receiver_semaphore_addr", "semaphore:noc0_receiver_semaphore_id"),
            ("noc1_receiver_semaphore_addr", "semaphore:noc1_receiver_semaphore_id"),
            ("dst_cb", "ct:dst_cb"),
            ("dst_num_pages", "ct:dst_num_pages"),
        ],
        cb_writes=["dst"],
    ),
    trisc=RISCContract(
        ct_args_type=None,
        rt_args_type="deepseek_b1_ops::Gather::ComputeArgs",
        is_noop=True,
    ),
    cb_ports={
        "src": CBPortSpec(CBDirection.INPUT),
        "dst": CBPortSpec(CBDirection.OUTPUT),
    },
    # Op<IsSenderCore, IsReceiverCore, pop_src, UsePerCoreSenderIdx>
    # {use_per_core_sender_idx} resolved to a Core:: flag
    op_template="Op<{is_sender}, {is_receiver}, true, {use_per_core_sender_idx}>",
    risc_latency={"ncrisc": 150, "brisc": 100, "trisc": 0},
)
