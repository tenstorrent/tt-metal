# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from argparse import ArgumentParser
import csv


def main():
    parser = ArgumentParser(
        "Parse an op perf results CSV and show performance data using the min allgather time and max other time over devices, optionally only for a specific signpost region."
    )
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--all", help="Show all times for each device", action="store_true")
    parser.add_argument("--signpost", help="Only include data after this signpost and before any others")
    parser.add_argument("--skip-last", help="Do not include timings from the last N ops", type=int, default=0)
    parser.add_argument(
        "--estimate-full-model",
        help="Estimate the full model performance by multiplying by N and adding back in the skipped ops",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    header, rows = read_rows(args.csv)
    blocks, signposts_seen = make_blocks(header, rows, args.signpost)

    if args.signpost and not args.signpost in signposts_seen:
        print(f'Error: signpost "{args.signpost}" was not found in this file')
        print(f"Valid signposts are: {signposts_seen}")
        return

    print(f'{"Op":20} {"Time (us)"}')
    for block in blocks[: -args.skip_last] if args.skip_last else blocks:
        print(block.long_str() if args.all else block.short_str())

    if args.skip_last:
        print(f"The following ops from the end of the run are not included below:")
        for block in blocks[-args.skip_last :]:
            print(block.long_str() if args.all else block.short_str())
        skipped_ops = blocks[-args.skip_last :]
        blocks = blocks[: -args.skip_last]
    else:
        skipped_ops = []

    total_time_ns = sum(block.time() for block in blocks)
    total_time_s = total_time_ns / 1e9
    tokens_per_s = 1 / total_time_s
    print(f"Tokens/s/user: {tokens_per_s:.2f} ({total_time_s*1000*1000:.1f} us latency)")

    if args.estimate_full_model:
        total_time_ns *= args.estimate_full_model
        total_time_ns += sum(block.time() for block in skipped_ops)
        total_time_s = total_time_ns / 1e9
        tokens_per_s = 1 / total_time_s
        print(
            f"Estimated full model ({args.estimate_full_model} * above + skipped ops) tokens/s/user: {tokens_per_s:.2f} ({total_time_s*1000*1000:.1f} us latency)"
        )

    if signposts_seen and not args.signpost:
        print(f"Warning - this file contains the following signposts that were not used for this analysis:")
        for s in signposts_seen:
            print(f'   "{s}"')
        print("Rerun with --signpost to show only the performance for a specific signpost region")


def read_rows(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


class Block:
    def __init__(self, op_name, times):
        self.op_name = op_name
        self.times = times

    def time(self):
        return min(self.times) if "AllGather" in self.op_name else max(self.times)

    def short_str(self):
        short_name = self.op_name.split("::")[-1].split(")")[0]
        time_range = max(self.times) - min(self.times)
        return f"{short_name:20} {self.time()/1000:-6.0f} ± {time_range/1000:-5.0f}"

    def long_str(self):
        short_name = self.op_name.split("::")[-1].split(")")[0]
        return f"{short_name:20} {self.time()/1000:-6.0f} <-" + " | ".join(f"{t/1000:-5.0f}" for t in self.times)

    def __repr__(self):
        return f"Block({self.op_name}, {self.times})"


def make_blocks(header, rows, signpost):
    """Perf dumps have one row per device in order, repeated for each op
    This returns a list of blocks, where each block has an op name
    and a list of times for each device.
    """

    # group rows by device then merge them together
    block_by_device = defaultdict(list)
    stop_on_signpost = False
    signposts_seen = []

    OP_CODE = header.index("OP CODE")
    OP_TYPE = header.index("OP TYPE")
    DEVICE_ID = header.index("DEVICE ID")
    FW_DURATION = header.index("DEVICE FW DURATION [ns]")

    block_op_name = None
    for row in rows:
        op_name = row[OP_CODE]
        op_type = row[OP_TYPE]

        if op_type == "signpost":
            signposts_seen.append(op_name)
            if stop_on_signpost:
                break
            elif op_name == signpost:
                # clear any previous data and stop on the next signpost
                stop_on_signpost = True
                block_by_device = defaultdict(list)
        elif op_type == "tt_dnn_device":
            device_id = int(row[DEVICE_ID])
            time = int(row[FW_DURATION])
            block_by_device[device_id].append(Block(op_name, [time]))

    # merge each device block into a single block with all the device times,
    # checking that the op name matches
    # blocks_by_device is a dict of device_id -> Block
    # we want to get a list of Block (with all device times)

    device_ids = list(sorted(block_by_device.keys()))
    merged_blocks = block_by_device[device_ids[0]]

    for device_id in device_ids[1:]:
        assert len(block_by_device[device_id]) == len(
            merged_blocks
        ), f"Device {device_id} has {len(block_by_device[device_id])} ops, expected {len(merged_blocks)} from previous devices"
        for row, b in enumerate(block_by_device[device_id]):
            assert (
                b.op_name == merged_blocks[row].op_name
            ), f"Op name mismatch at row {row}: device {device_id} has {b.op_name} != {merged_blocks[row].op_name}"
            merged_blocks[row].times += b.times

    return merged_blocks, signposts_seen


if __name__ == "__main__":
    main()
