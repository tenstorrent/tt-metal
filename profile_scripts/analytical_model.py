import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser('Analytical model parameters')
    parser.add_argument("--mode", choices=["read", "write", "compute"], help="read-only, write-only, read-compute-write")
    parser.add_argument("--transfer-rate", type=float, help="transfer rate is 32B for Grayskull, 64B for Blackhole", default=32)

    # Read/Write modeling
    parser.add_argument("--pre-issue-overhead", type=float, help="profile marker overhead before issue", default=1)
    parser.add_argument("--NIU-programming", type=float, help="NIU programming overhead", default=1)
    parser.add_argument("--non-NIU-programming", type=float, help="non NIU programming overhead", default=1)
    parser.add_argument("--round-trip-latency", type=float, help="round trip latency", default=1)
    parser.add_argument("--flit-latency", type=float, help="consecutive flit arrival latency", default=1)
    parser.add_argument("--transfer-size", type=float, help="transfer size", default=1)
    parser.add_argument("--buffer-size", type=float, help="buffer size", default=1)

    # Compute modeling
    parser.add_argument("--unpack-latency", type=float, help="unpack latency on execution units", default=1)
    parser.add_argument("--math-latency", type=float, help="compute latency on execution units", default=1)
    parser.add_argument("--pack-latency", type=float, help="pack latency on execution units", default=1)
    parser.add_argument("--block-num", type=int, help="number of blocks", default=1)
    parser.add_argument("--block-size", type=int, help="block size", default=1024)
    parser.add_argument("--transfer-size-A", type=int, help="Matrix A is stored in row major", default=32)
    parser.add_argument("--transfer-size-B", type=int, help="Matrix B is stored in tile major", default=1024)
    parser.add_argument("--transfer-size-write", type=int, help="Matrix B is stored in tile major", default=1024)
    parser.add_argument("--read-issue-latency", type=float, help="read-issue-latency", default=1)
    parser.add_argument("--write-issue-latency", type=float, help="write-issue-latency", default=1)

    args = parser.parse_args()
    return args

def get_read_latency(args, transfer_size, buffer_size, issue_latency):
    num_flits_per_transfer = transfer_size / args.transfer_rate
    num_transfer = buffer_size / transfer_size

    total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer

    if (num_flits_per_transfer >= issue_latency):
        barrier_latency = args.pre_issue_overhead + issue_latency + args.round_trip_latency + args.flit_latency * num_flits_per_transfer * num_transfer - total_issue_latency
    else:
        barrier_latency = args.round_trip_latency + args.flit_latency * num_flits_per_transfer

    return total_issue_latency, barrier_latency

def get_write_latency(args, transfer_size, buffer_size, issue_latency):
    num_flits_per_transfer = transfer_size / args.transfer_rate
    num_transfer = buffer_size / transfer_size

    if (num_flits_per_transfer >= issue_latency):
        if num_transfer < 3:
            total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer
        else:
            total_issue_latency = args.pre_issue_overhead + issue_latency + args.flit_latency * num_flits_per_transfer * (num_transfer - 2) + 15
        barrier_latency = args.pre_issue_overhead + issue_latency + args.round_trip_latency + args.flit_latency * num_flits_per_transfer * num_transfer - total_issue_latency
    else:
        total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer
        barrier_latency = args.round_trip_latency + args.flit_latency * num_flits_per_transfer

    return total_issue_latency, barrier_latency

def read_analytical_model(args):
    issue_latency = (args.NIU_programming + args.non_NIU_programming)
    total_issue_latency, barrier_latency = get_read_latency(args, args.transfer_size, args.buffer_size, issue_latency)

    noc_utilization = args.buffer_size / (total_issue_latency + barrier_latency) / (args.transfer_rate)

    print("Buffer: {:.0f} Transfer: {:.0f} issue: {} barrier: {} noc_util: {}".format(args.buffer_size, args.transfer_size, total_issue_latency, barrier_latency, noc_utilization))

def write_analytical_model(args):
    issue_latency = (args.NIU_programming + args.non_NIU_programming)
    total_issue_latency, barrier_latency = get_write_latency(args, args.transfer_size, args.buffer_size, issue_latency)

    noc_utilization = args.buffer_size / (total_issue_latency + barrier_latency) / (args.transfer_rate)

    print("Buffer: {:.0f} Transfer: {:.0f} issue: {} barrier: {} noc_util: {}".format(args.buffer_size, args.transfer_size, total_issue_latency, barrier_latency, noc_utilization))

def write_analytical_model(args):
    total_issue_latency, barrier_latency = get_write_latency(args)

    noc_utilization = args.buffer_size / (total_issue_latency + barrier_latency) / (args.transfer_rate)

    print("Buffer: {:.0f} Transfer: {:.0f} issue: {} barrier: {} noc_util: {}".format(args.buffer_size, args.transfer_size, total_issue_latency, barrier_latency, noc_utilization))

def compute_analytical_model(args):
    total_issue_latency_A, barrier_latency_A = get_read_latency(args, args.transfer_size_A, args.block_size, args.read_issue_latency)
    total_issue_latency_B, barrier_latency_B = get_read_latency(args, args.transfer_size_B, args.block_size, args.read_issue_latency)
    total_issue_latency_write, barrier_latency_write = get_write_latency(args, args.transfer_size_write, args.block_size, args.write_issue_latency)
    read_latency = total_issue_latency_A + barrier_latency_A + total_issue_latency_B + barrier_latency_B
    write_latency = total_issue_latency_write + barrier_latency_write
    compute_latency = args.unpack_latency + args.math_latency + args.pack_latency

    if read_latency > max(write_latency, compute_latency):
        bound = "read NOC"
        total_latency = read_latency * args.block_num + compute_latency + write_latency
        read_noc_utilization = read_latency * args.block_num / total_latency
        compute_utilization = compute_latency * args.block_num / total_latency
        write_noc_utilization = write_latency * args.block_num / total_latency
    elif write_latency > max(read_latency, compute_latency):
        bound = "write NOC"
        total_latency = read_latency + compute_latency + write_latency * args.block_num
        read_noc_utilization = read_latency * args.block_num / total_latency
        compute_utilization = compute_latency * args.block_num / total_latency
        write_noc_utilization = write_latency * args.block_num / total_latency
    elif compute_latency > max(read_latency, write_latency):
        bound = "compute"
        total_latency = read_latency + compute_latency * args.block_num + write_latency
        read_noc_utilization = read_latency * args.block_num / total_latency
        compute_utilization = compute_latency * args.block_num / total_latency
        write_noc_utilization = write_latency * args.block_num / total_latency

    print("Total latency: {:.2f}, bounded by {}, Read NOC utilization: {:.1f}%, Execution unit utilization: {:.1f}%, Read NOC utilization: {:.1f}%.".format(total_latency, bound, read_noc_utilization * 100, compute_utilization * 100, write_noc_utilization * 100))

args = get_args()

# did not consider tarnsfer size > 8k
# write is not correct

if args.mode == "read":
    read_analytical_model(args)
elif args.mode == "write":
    write_analytical_model(args)
elif args.mode == "compute":
    compute_analytical_model(args)
