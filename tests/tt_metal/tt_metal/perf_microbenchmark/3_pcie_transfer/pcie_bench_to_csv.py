# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

#
# Capture the pcie benchmark results as a json file and
# then run this script to convert it to a csv
#
import json
import sys
import csv
import io


def get_filename():
    if len(sys.argv) != 2:
        print("Usage: python pcie_bench_to_csv.py <filename>")
        sys.exit(1)

    return str(sys.argv[1])


def rows_to_csv_str(rows):
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def BM_HostHP_N_Readers(json_data, filter_name="0_Host_Write_HP_N_Readers"):
    data_points = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark["name"]
        if not name.startswith(filter_name):
            continue

        data_points.append(
            [
                str(benchmark["total_size"]),
                # str(benchmark["kernel_0_bw"]),
                str(benchmark["total_size"]),
                # str(benchmark["dev_bw"]),
                str(benchmark["total_size"]),
                str(benchmark["num_readers"]),
                str(benchmark["bytes_per_second"]),
                str(benchmark["page_size"]),
            ]
        )

    data_points.sort(key=lambda x: (x[2], x[5], x[3]))
    data_points.insert(0, ["kernel_0_bw", "dev_bw", "total_size", "num_readers", "bytes_per_second", "page_size"])

    print(filter_name)
    print(rows_to_csv_str(data_points))

    return data_points


def BM_HostHP_N_Threads(json_data, filter_name="3_Host_Write_HP_N_Threads_No_Kernels"):
    data_points = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark["name"]
        if not name.startswith(filter_name):
            continue

        data_points.append(
            {
                "thread_0_bw": benchmark["thread_0_bw"],
                "total_size": benchmark["total_size"],
                "page_size": benchmark["page_size"],
                "num_threads": benchmark["num_threads"],
                "host_bytes_per_second": benchmark["bytes_per_second"],
            }
        )

    data_points.sort(key=lambda x: (x["total_size"], x["page_size"], x["num_threads"]))
    data_points.insert(0, ["thread_0_bw", "total_size", "page_size", "num_threads", "host_bytes_per_second"])
    print(filter_name)
    print(rows_to_csv_str(data_points))

    return data_points


def BM_BatchSizing(json_data, filter_name="5_MyMemcpyToDevice_Sizing"):
    data_points = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark["name"]
        if not name.startswith(filter_name):
            continue

        data_points.append(
            {
                "total_size": benchmark["total_size"],
                "page_size": benchmark["page_size"],
                "host_bytes_per_second": benchmark["bytes_per_second"],
            }
        )

    data_points.sort(key=lambda x: (x["total_size"], x["page_size"]))
    data_points.insert(0, ["total_size", "page_size", "host_bytes_per_second"])
    print(filter_name)
    print(rows_to_csv_str(data_points))

    return data_points


def BM_HostHP_DifferentPage(json_data, filter_name="6_HostHP_1_Reader_DifferentPage"):
    data_points = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark["name"]
        if not name.startswith(filter_name):
            continue

        data_points.append(
            {
                "dev_bw": benchmark["dev_bw"],
                "dev_bytes": benchmark["dev_bytes"],
                "kernel_0_bw": benchmark["kernel_0_bw"],
                "total_size": benchmark["total_size"],
                "page_size": benchmark["page_size"],
                "num_readers": benchmark["num_readers"],
                "host_bytes_per_second": benchmark["bytes_per_second"],
            }
        )

    data_points.sort(key=lambda x: (x["total_size"], x["page_size"], x["num_readers"]))
    data_points.insert(
        0, ["dev_bw", "dev_bytes", "kernel_0_bw", "total_size", "page_size", "num_readers", "host_bytes_per_second"]
    )
    print(filter_name)
    print(rows_to_csv_str(data_points))

    return data_points


def BM_HostHP_N_Writers(json_data, filter_name="7_Host_Write_HP_1_Writer"):
    data_points = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark["name"]
        if not name.startswith(filter_name):
            continue

        data_points.append(
            {
                "kernel_0_bw": benchmark["kernel_0_bw"],
                "dev_bw": benchmark["dev_bw"],
                "dev_bytes": benchmark["dev_bytes"],
                "total_size": benchmark["total_size"],
                "page_size": benchmark["page_size"],
                "num_writers": benchmark["num_writers"],
                "host_bytes_per_second": benchmark["bytes_per_second"],
            }
        )

    data_points.sort(key=lambda x: (x["total_size"], x["page_size"], x["num_writers"]))
    data_points.insert(
        0, ["kernel_0_bw", "dev_bw", "dev_bytes", "total_size", "page_size", "num_writers", "host_bytes_per_second"]
    )
    print(filter_name)
    print(rows_to_csv_str(data_points))

    return data_points


def BM_2_MMIO_Devices_Reading_DifferentPage(json_data, filter_name=""):
    data_points = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark["name"]
        if not name.startswith(filter_name):
            continue

        data_points.append(
            {
                "dev_0_kernel_0_bw": benchmark["dev_0_kernel_0_bw"],
                "dev_bw": benchmark["dev_bw"],
                "dev_bytes": benchmark["dev_bytes"],
                "dev_cycles": benchmark["dev_cycles"],
                "total_size": benchmark["total_size"],
                "page_size": benchmark["page_size"],
            }
        )

    data_points.sort(key=lambda x: (x["total_size"], x["page_size"]))
    data_points.insert(0, ["dev_0_kernel_0_bw", "dev_bw", "dev_bytes", "dev_cycles", "total_size", "page_size"])
    print(filter_name)
    print(rows_to_csv_str(data_points))

    return data_points


def main():
    with open(get_filename(), "r") as fd:
        data = json.load(fd)

    BM_HostHP_N_Readers(data, "0_Host_Write_HP_N_Readers")
    BM_HostHP_N_Readers(data, "1_Host_Write_HP_N_Readers_Cached_Vector")
    BM_HostHP_N_Readers(data, "2_N_Readers_No_Host_Copy")

    BM_HostHP_N_Threads(data, "3_Host_Write_HP_N_Threads_No_Kernels")
    BM_HostHP_N_Threads(data, "4_Host_Write_HP_N_Threads_Cached_Vector_No_Kernels")

    BM_BatchSizing(data, "5_MyMemcpyToDevice_Sizing")

    BM_HostHP_DifferentPage(data, "6_HostHP_1_Reader_DifferentPage")

    BM_HostHP_N_Writers(data, "7_Host_Write_HP_1_Writer")

    BM_HostHP_N_Writers(data, "8_Writer_Kernel_Only")


if __name__ == "__main__":
    main()
