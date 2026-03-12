# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Mcast micro-op specification.

Mirrors the contract in unified_kernels/mcast.hpp:
  - BRISC = Sender: SenderCTArgs<mcast_num_cores, is_part_of_receiver_grid, loopback>
  - NCRISC = Receiver: ReceiverCTArgs (empty)
  - TRISC = ComputeCTArgs (empty, no-op)

Persistent init/teardown pattern: init() once → operator() N times → teardown().

SenderArgs fields: dest_noc_{start,end}_{x,y}, semaphore addrs, data_size_bytes,
  src_cb, src_num_pages, input_data_addr (get_read_ptr), mcast_receiver_data_addr (get_write_ptr)

ReceiverArgs fields: data_receiver_semaphore_addr, dst_cb, dst_num_pages
"""

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, CBPortSpec, MicroOpSpec, RISCContract

MCAST = MicroOpSpec(
    name="Mcast",
    header="unified_kernels/mcast.hpp",
    namespace="deepseek_b1_ops",
    struct_name="Mcast",
    ncrisc=RISCContract(
        ct_args_type="deepseek_b1_ops::Mcast::ReceiverCTArgs",
        rt_args_type="deepseek_b1_ops::Mcast::ReceiverArgs",
        named_ct_args=[
            "data_receiver_semaphore",
            "dst_cb",
            "dst_num_pages",
            "src_cb",
            "src_num_pages",
        ],
        setup_sharded=["src"],
        rt_args_fields=[
            ("data_receiver_semaphore_addr", "semaphore:data_receiver_semaphore"),
            ("dst_cb", "ct:dst_cb"),
            ("dst_num_pages", "ct:dst_num_pages"),
        ],
    ),
    brisc=RISCContract(
        ct_args_type=(
            "deepseek_b1_ops::Mcast::SenderCTArgs<"
            "{num_cores}, {is_part_of_receiver_grid}, "
            "false>"  # loopback = false
        ),
        rt_args_type="deepseek_b1_ops::Mcast::SenderArgs",
        named_ct_args=[
            "dest_noc_start_x",
            "dest_noc_start_y",
            "dest_noc_end_x",
            "dest_noc_end_y",
            "num_cores",
            "data_sender_semaphore",
            "data_receiver_semaphore",
            "data_size_bytes",
            "src_cb",
            "src_num_pages",
            "dst_cb",
            "is_part_of_receiver_grid",
        ],
        rt_args_fields=[
            ("dest_noc_start_x", "ct:dest_noc_start_x"),
            ("dest_noc_start_y", "ct:dest_noc_start_y"),
            ("dest_noc_end_x", "ct:dest_noc_end_x"),
            ("dest_noc_end_y", "ct:dest_noc_end_y"),
            ("data_sender_semaphore_addr", "semaphore:data_sender_semaphore"),
            ("data_receiver_semaphore_addr", "semaphore:data_receiver_semaphore"),
            ("data_size_bytes", "ct:data_size_bytes"),
            ("src_cb", "ct:src_cb"),
            ("src_num_pages", "ct:src_num_pages"),
            ("input_data_addr", "read_ptr:src_cb"),
            ("mcast_receiver_data_addr", "write_ptr:dst_cb"),
        ],
    ),
    trisc=RISCContract(
        ct_args_type="deepseek_b1_ops::Mcast::ComputeCTArgs",
        rt_args_type="deepseek_b1_ops::Mcast::ComputeArgs",
        is_noop=True,
    ),
    cb_ports={
        "src": CBPortSpec(CBDirection.INPUT, is_sharded=True),
        "dst": CBPortSpec(CBDirection.OUTPUT),
    },
    # Op<CTArgs, IsSenderCore, IsMcastGridCore, IsReceiverCore, pop_src>
    # IsMcastGridCore and IsReceiverCore both use is_receiver (same set in down_proj)
    op_template="Op<{CTArgs}, {is_sender}, {is_receiver}, {is_receiver}, true>",
    has_init=True,
    has_teardown=True,
    risc_latency={"ncrisc": 100, "brisc": 200, "trisc": 0},
)
