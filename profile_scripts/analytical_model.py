import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser('Analytical model parameters')
    parser.add_argument("--pre-issue-overhead", type=float, help="profile marker overhead before issue", default=1)
    parser.add_argument("--NIU-programming", type=float, help="NIU programming overhead", default=1)
    parser.add_argument("--non-NIU-programming", type=float, help="non NIU programming overhead", default=1)
    parser.add_argument("--round-trip-latency", type=float, help="round trip latency", default=1)
    parser.add_argument("--flit-latency", type=float, help="consecutive flit arrival latency", default=1)
    parser.add_argument("--transfer-size", type=float, help="transfer size", default=1)
    parser.add_argument("--buffer-size", type=float, help="buffer size", default=1)
    parser.add_argument("--transfer-rate", type=float, help="transfer rate is 32B for Grayskull", default=32)
    parser.add_argument("--read-or-write", choices=["read", "write"], help="read or write choice")
    args = parser.parse_args()
    return args

def read_analytical_model(args):
    num_flits_per_transfer = args.transfer_size / 32
    num_transfer = args.buffer_size / args.transfer_size

    issue_latency = (args.NIU_programming + args.non_NIU_programming)
    total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer

    if (num_flits_per_transfer >= issue_latency):
        barrier_latency = args.pre_issue_overhead + issue_latency + args.round_trip_latency + args.flit_latency * num_flits_per_transfer * num_transfer - total_issue_latency
    else:
        barrier_latency = args.round_trip_latency + args.flit_latency * num_flits_per_transfer

    noc_utilization = args.buffer_size / (total_issue_latency + barrier_latency) / (args.transfer_rate)

    print("Buffer: {:.0f} Transfer: {:.0f} issue: {} barrier: {} noc_util: {}".format(args.buffer_size, args.transfer_size, total_issue_latency, barrier_latency, noc_utilization))

def write_analytical_model(args):
    num_flits_per_transfer = args.transfer_size / 32
    num_transfer = args.buffer_size / args.transfer_size

    issue_latency = (args.NIU_programming + args.non_NIU_programming)
    total_issue_latency = args.pre_issue_overhead + issue_latency * num_transfer

    if (num_flits_per_transfer >= issue_latency):
        barrier_latency = args.pre_issue_overhead + issue_latency + args.round_trip_latency + args.flit_latency * num_flits_per_transfer * num_transfer - total_issue_latency
    else:
        barrier_latency = args.round_trip_latency + args.flit_latency * num_flits_per_transfer

    noc_utilization = args.buffer_size / (total_issue_latency + barrier_latency) / (args.transfer_rate)

    print("Buffer: {:.0f} Transfer: {:.0f} issue: {} barrier: {} noc_util: {}".format(args.buffer_size, args.transfer_size, total_issue_latency, barrier_latency, noc_utilization))


args = get_args()

# did not consider tarnsfer size > 8k
# write is not correct

if args.read_or_write == "read":
    read_analytical_model(args)
elif args.read_or_write == "write":
    write_analytical_model(args)
