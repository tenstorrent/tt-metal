import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser('Analytical model parameters')
    parser.add_argument("--comment", type=int, help="show comments for test case, but post-processing script cannot parse comments", default=0)
    parser.add_argument("--mode", choices=["read", "write", "compute", "multi-core"], help="read-only, write-only, read-compute-write")
    parser.add_argument("--NOC-data-width", type=float, help="transfer rate is 32B for Grayskull, 64B for Blackhole", default=32)

    # Read/Write modeling
    parser.add_argument("--pre-issue-overhead", type=float, help="profile marker overhead before issue", default=1)
    parser.add_argument("--NIU-programming", type=float, help="NIU programming overhead", default=1)
    parser.add_argument("--non-NIU-programming", type=float, help="non NIU programming overhead", default=1)
    parser.add_argument("--round-trip-latency", type=float, help="round trip latency", default=1)
    parser.add_argument("--head-flit-latency", type=float, help="head flit latency", default=1)
    parser.add_argument("--flit-latency", type=float, help="consecutive flit arrival latency", default=1)
    parser.add_argument("--transfer-size", type=float, help="transfer size", default=1)
    parser.add_argument("--buffer-size", type=float, help="buffer size", default=1)

    # Compute modeling
    parser.add_argument("--compute-latency", type=float, help="unpack-math-compute latency on execution units", default=1)
    parser.add_argument("--CB-producer-consumer-sync-latency", type=float, help="the latency from producer finishes pushing tile to consumer seeing tile arrival", default=56)
    parser.add_argument("--block-num", type=int, help="number of blocks", default=1)
    parser.add_argument("--block-size", type=int, help="block size", default=1024)
    parser.add_argument("--transfer-size-A", type=int, help="Matrix A is stored in row major", default=32)
    parser.add_argument("--transfer-size-B", type=int, help="Matrix B is stored in tile major", default=1024)
    parser.add_argument("--transfer-size-write", type=int, help="Matrix B is stored in tile major", default=1024)
    parser.add_argument("--read-issue-latency", type=float, help="read-issue-latency", default=1)
    parser.add_argument("--write-issue-latency", type=float, help="write-issue-latency", default=1)

    # Multi-core modeling
    parser.add_argument("--multi-core-mode", choices=["DRAM", "DRAM-Tensix", "Tensix"], help="read from DRAM-only, both DRAM and Tensix, Tensix-only")
    parser.add_argument("--core-range-x-dim", type=str, help="core range on X dimension ", default="0-0")
    parser.add_argument("--core-range-y-dim", type=str, help="core range on Y dimension ", default="0-0")
    parser.add_argument("--DRAM-round-trip-latency", type=float, help="DRAM round trip latency", default=1600)
    parser.add_argument("--Tensix-round-trip-latency", type=float, help="Tensix round trip latency", default=95)
    parser.add_argument("--DRAM-channels", type=int, help="number of DRAM channels availble", default=1)


    args = parser.parse_args()
    return args

def get_read_latency(args, transfer_size, buffer_size, issue_latency):
    num_flits_per_transfer = transfer_size / args.NOC_data_width
    num_transfer = buffer_size / transfer_size

    total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer

    if (num_flits_per_transfer >= issue_latency):
        barrier_latency = args.pre_issue_overhead + issue_latency + args.round_trip_latency + args.head_flit_latency + args.flit_latency * num_flits_per_transfer * num_transfer - total_issue_latency
    else:
        barrier_latency = args.round_trip_latency + args.head_flit_latency + args.flit_latency * num_flits_per_transfer

    return total_issue_latency, barrier_latency

def get_write_latency(args, transfer_size, buffer_size, issue_latency):
    num_flits_per_transfer = transfer_size / args.NOC_data_width
    num_transfer = buffer_size / transfer_size

    if (num_flits_per_transfer >= issue_latency):
        if num_transfer < 3:
            total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer
        else:
            total_issue_latency = args.pre_issue_overhead + issue_latency + args.flit_latency * num_flits_per_transfer * (num_transfer - 2) + 15
        barrier_latency = args.pre_issue_overhead + issue_latency + args.round_trip_latency + args.head_flit_latency + args.flit_latency * num_flits_per_transfer * num_transfer - total_issue_latency
    else:
        total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer
        barrier_latency = args.round_trip_latency + args.head_flit_latency + args.flit_latency * num_flits_per_transfer

    return total_issue_latency, barrier_latency

def read_analytical_model(args):
    issue_latency = (args.NIU_programming + args.non_NIU_programming)
    total_issue_latency, barrier_latency = get_read_latency(args, args.transfer_size, args.buffer_size, issue_latency)

    noc_utilization = args.buffer_size / (total_issue_latency + barrier_latency) / (args.NOC_data_width)

    if args.comment:
        print("================== Single Core Read Test ==================")
        print(">> Modeling Parameters")
        print("NOC Data width (Byte): {:.0f}.".format(args.NOC_data_width))
        print("Pre-issue overhead (cycle): {:.0f}".format(args.pre_issue_overhead))
        print("NIU programming latency (cycle): {:.0f}".format(args.NIU_programming))
        print("Non-NIU-programming latency (cycle): {:.0f}".format(args.non_NIU_programming))
        print("Round trip latency (cycle): {:.0f}".format(args.round_trip_latency))
        print("Head flit latency (cycle): {:.0f}".format(args.head_flit_latency))
        print("Successive flit latency (cycle): {:.0f}".format(args.flit_latency))
        print("Buffer size (Byte): {:.0f} Transfer size (Byte): {:.0f}".format(args.buffer_size, args.transfer_size))
        print("\n>> Modeling Results")
        print("Total latency (cycle): {:.0f}\nIssue latency (cycle): {:.0f}\nBarrier latency (cycle): {:.0f}\nNOC Utilization {:.2f}%\n".format(total_issue_latency + barrier_latency, total_issue_latency, barrier_latency, noc_utilization*100))
    else:
        print("Buffer: {:.0f} Transfer: {:.0f} issue: {:.0f} barrier: {:.0f} noc_util: {:.4f}".format(args.buffer_size, args.transfer_size, total_issue_latency, barrier_latency, noc_utilization))

def write_analytical_model(args):
    issue_latency = (args.NIU_programming + args.non_NIU_programming)
    total_issue_latency, barrier_latency = get_write_latency(args, args.transfer_size, args.buffer_size, issue_latency)

    noc_utilization = args.buffer_size / (total_issue_latency + barrier_latency) / (args.NOC_data_width)

    if args.comment:
        print("================== Single Core Write Test ==================")
        print(">> Modeling Parameters")
        print("NOC Data width (Byte): {:.0f}.".format(args.NOC_data_width))
        print("Pre-issue overhead (cycle): {:.0f}".format(args.pre_issue_overhead))
        print("NIU programming latency (cycle): {:.0f}".format(args.NIU_programming))
        print("Non-NIU-programming latency (cycle): {:.0f}".format(args.non_NIU_programming))
        print("Round trip latency (cycle): {:.0f}".format(args.round_trip_latency))
        print("Head flit latency (cycle): {:.0f}".format(args.head_flit_latency))
        print("Successive flit latency (cycle): {:.0f}".format(args.flit_latency))
        print("Buffer size (Byte): {:.0f} Transfer size (Byte): {:.0f}".format(args.buffer_size, args.transfer_size))
        print("\n>> Modeling Results")
        print("Total latency (cycle): {:.0f}\nIssue latency (cycle): {:.0f}\nBarrier latency (cycle): {:.0f}\nNOC Utilization {:.2f}%\n".format(total_issue_latency + barrier_latency, total_issue_latency, barrier_latency, noc_utilization*100))
    else:
        print("Buffer: {:.0f} Transfer: {:.0f} issue: {:.0f} barrier: {:.0f} noc_util: {:.4f}".format(args.buffer_size, args.transfer_size, total_issue_latency, barrier_latency, noc_utilization))

def compute_analytical_model(args):
    total_issue_latency_A, barrier_latency_A = get_read_latency(args, args.transfer_size_A, args.block_size, args.read_issue_latency)
    total_issue_latency_B, barrier_latency_B = get_read_latency(args, args.transfer_size_B, args.block_size, args.read_issue_latency)
    total_issue_latency_write, barrier_latency_write = get_write_latency(args, args.transfer_size_write, args.block_size, args.write_issue_latency)
    read_latency = total_issue_latency_A + barrier_latency_A + total_issue_latency_B + barrier_latency_B
    write_latency = total_issue_latency_write + barrier_latency_write
    compute_latency = args.compute_latency

    if read_latency > max(write_latency, compute_latency):
        bound = "read NOC"
        total_latency = read_latency * args.block_num + compute_latency + write_latency + 2*args.CB_producer_consumer_sync_latency
        read_noc_utilization = read_latency * args.block_num / total_latency
        compute_utilization = compute_latency * args.block_num / total_latency
        write_noc_utilization = write_latency * args.block_num / total_latency
    elif write_latency > max(read_latency, compute_latency):
        bound = "write NOC"
        total_latency = read_latency + compute_latency + write_latency * args.block_num + 2*args.CB_producer_consumer_sync_latency
        read_noc_utilization = read_latency * args.block_num / total_latency
        compute_utilization = compute_latency * args.block_num / total_latency
        write_noc_utilization = write_latency * args.block_num / total_latency
    elif compute_latency > max(read_latency, write_latency):
        bound = "compute"
        total_latency = read_latency + compute_latency * args.block_num + write_latency + 2*args.CB_producer_consumer_sync_latency
        read_noc_utilization = read_latency * args.block_num / total_latency
        compute_utilization = compute_latency * args.block_num / total_latency
        write_noc_utilization = write_latency * args.block_num / total_latency

    if args.comment:
        print("=================== Single Core Bmm Test ===================")
        print(">> Modeling Parameters")
        print("NOC Data width (Byte): {:.0f}.".format(args.NOC_data_width))
        print("Read issue latency (cycle): {:.0f}".format(args.read_issue_latency))
        print("Write issue latency (cycle): {:.0f}".format(args.write_issue_latency))
        print("Compute latency (cycle): {:.0f}".format(args.compute_latency))
        print("Round trip latency (cycle): {:.0f}".format(args.round_trip_latency))
        print("Head flit latency (cycle): {:.0f}".format(args.head_flit_latency))
        print("Successive flit latency (cycle): {:.0f}".format(args.flit_latency))
        print("Matrix A read transfer size: {:.0f}".format(args.transfer_size_A))
        print("Matrix B read transfer size: {:.0f}".format(args.transfer_size_B))
        print("Write transfer size: {:.0f}".format(args.transfer_size_write))
        print("Matrix block size: {:.0f}".format(args.block_size))
        print("Block number: {:.0f}".format(args.block_num))
        print("\n>> Modeling Results")
        print("Total latency (cycle): {:.0f}, bounded by {}\nRead NOC utilization: {:.1f}%\nExecution unit utilization: {:.1f}%\nWrite NOC utilization: {:.1f}%\n".format(total_latency, bound, read_noc_utilization * 100, compute_utilization * 100, write_noc_utilization * 100))
    else:
        print("Total latency: {:.0f}, bounded by {}, Read NOC utilization: {:.1f}%, Execution unit utilization: {:.1f}%, Write NOC utilization: {:.1f}%.".format(total_latency, bound, read_noc_utilization * 100, compute_utilization * 100, write_noc_utilization * 100))

def multi_core_analytical_model(args):
    if args.multi_core_mode == "DRAM":
        core_range_x = args.core_range_x_dim.split("-")
        core_range_y = args.core_range_y_dim.split("-")
        core_list = []
        for i in range(int(core_range_x[0]), int(core_range_x[1]) + 1):
            for j in range(int(core_range_y[0]), int(core_range_y[1]) + 1):
                core_list.append((i, j))
    elif args.multi_core_mode == "DRAM-Tensix":
        print()
    elif args.multi_core_mode == "Tensix":
        print()
    print()


args = get_args()

# did not consider tarnsfer size > 8k
# write is not correct

if args.mode == "read":
    read_analytical_model(args)
elif args.mode == "write":
    write_analytical_model(args)
elif args.mode == "compute":
    compute_analytical_model(args)
elif args.mode == "multi-core":
    multi_core_analytical_model(args)
