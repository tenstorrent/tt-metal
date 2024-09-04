# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
    This file parses ops reports and puts them together with a cleaner format.
    To generate ops reports for Stable Diffusion;
    When running SD's test_perf() in perf_unbatched_stable.py, do

        testLocation ="stable_diffusion_reports"
        os.system(f"rm -rf {testLocation}")


    Next we have to run post processor for the ops.

        ./tt_metal/tools/profiler/process_ops_logs.py -i stable_diffusion_reports -o stable_diffusion_reports/output

    Finally run this script, you should have a few csv files titled: OP_cleaned_ops.csv

"""


import csv
import re

from typing import Dict

file_address = "stable_diffusion_reports/output/profile_log_ops.csv"


def dictionize(fp: str):
    res = {}
    reader = csv.DictReader(fp)
    for line in reader:
        key = line["NAME"]
        res[key] = line
    return res


def extract_id(id):
    ids = re.findall(r"_\d+", id)
    if len(ids) == 1:
        return ids[0]
    if len(ids) == 0:
        return id
    if len(ids) > 1:
        assert False, f"{ids}, {id}, something weird is up here!"


def merge_similar_ids(dict_ops):
    merge_res = {}
    for k in dict_ops.keys():
        _id = extract_id(k)
        if _id not in merge_res:
            merge_res[_id] = [dict_ops[k]]
        else:
            merge_res[_id].append(dict_ops[k])
    return merge_res


def make_row(keys, vals):
    d = {}
    for k, v in zip(keys, vals):
        d[k] = v
    return d


def parse_chunk(key, value):
    input_size = value[0]["INPUTS"].split("|")[0].replace("_", ",")
    input_layout = value[0]["INPUTS"].split("|")[1]
    input_device = value[0]["INPUTS"].split("|")[3]

    outputs1, outputs2 = value[0]["OUTPUTS"].split("-")

    output1_size = outputs1.split("|")[0].replace("_", ",")
    output1_layout = outputs1.split("|")[1]
    output1_device = outputs1.split("|")[3]

    output2_size = outputs2.split("|")[0].replace("_", ",")
    output2_layout = outputs2.split("|")[1]
    output2_device = outputs2.split("|")[3]

    keys = [
        "op",
        "input_size",
        "input_layout",
        "input_device",
        "output1_size",
        "output1_layout",
        "output1_device",
        "output2_size",
        "output2_layout",
        "output2_device",
    ]
    vals = [
        "fallback_chunk",
        input_size,
        input_layout,
        input_device,
        output1_size,
        output1_layout,
        output1_device,
        output2_size,
        output2_layout,
        output2_device,
    ]

    row = make_row(keys, vals)
    return row


def parse_concat(key, value):
    input1, input2 = value[0]["INPUTS"].split("-")

    input1_size = input1.split("|")[0].replace("_", ",")
    input1_layout = input1.split("|")[1]
    input1_device = input1.split("|")[3]

    input2_size = input2.split("|")[0].replace("_", ",")
    input2_layout = input2.split("|")[1]
    input2_device = input2.split("|")[3]

    output_size = value[0]["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = value[0]["OUTPUTS"].split("|")[1]
    output_device = value[0]["OUTPUTS"].split("|")[3]

    keys = [
        "op",
        "input1_size",
        "input1_layout",
        "input1_device",
        "input2_size",
        "input2_layout",
        "input2_device",
        "output_size",
        "output_layout",
        "output_device",
    ]
    vals = [
        "fallback_concat",
        input1_size,
        input1_layout,
        input1_device,
        input2_size,
        input2_layout,
        input2_device,
        output_size,
        output_layout,
        output_device,
    ]

    row = make_row(keys, vals)
    return row


def parse_silu(key, value):
    input_size = value[0]["INPUTS"].split("|")[0].replace("_", ",")
    input_layout = value[0]["INPUTS"].split("|")[1]
    input_device = value[0]["INPUTS"].split("|")[3]

    output_size = value[0]["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = value[0]["OUTPUTS"].split("|")[1]
    output_device = value[0]["OUTPUTS"].split("|")[3]

    keys = ["op", "input_size", "input_layout", "input_device", "output_size", "output_layout", "output_device"]
    vals = ["fallback_silu", input_size, input_layout, input_device, output_size, output_layout, output_device]

    row = make_row(keys, vals)
    return row


def parse_reshape(key, value):
    input_size = value[0]["INPUTS"].split("|")[0].replace("_", ",")
    input_layout = value[0]["INPUTS"].split("|")[1]
    input_device = value[0]["INPUTS"].split("|")[3]

    output_size = value[0]["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = value[0]["OUTPUTS"].split("|")[1]
    output_device = value[0]["OUTPUTS"].split("|")[3]

    keys = ["op", "input_size", "input_layout", "input_device", "output_size", "output_layout", "output_device"]
    vals = ["fallback_reshape", input_size, input_layout, input_device, output_size, output_layout, output_device]

    row = make_row(keys, vals)
    return row


def parse_full(key, value):
    output_size = value[0]["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = value[0]["OUTPUTS"].split("|")[1]
    output_device = value[0]["OUTPUTS"].split("|")[3]

    fill = value[0]["META DATA"].replace("(", "").replace(")", "").split("-")[0].split(";")[-1]

    keys = ["op", "output_size", "output_layout", "output_device", "fill_value"]
    vals = ["fallback_full", output_size, output_layout, output_device, fill]

    row = make_row(keys, vals)
    return row


def parse_repeat_interleave(key, value):
    input_size = value[0]["INPUTS"].split("|")[0].replace("_", ",")
    input_layout = value[0]["INPUTS"].split("|")[1]
    input_device = value[0]["INPUTS"].split("|")[3]

    output_size = value[0]["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = value[0]["OUTPUTS"].split("|")[1]
    output_device = value[0]["OUTPUTS"].split("|")[3]

    args = value[0]["META DATA"].split("-")[1].split("(")[1].replace(")", "").replace("|", ",")

    keys = ["op", "input_size", "input_layout", "input_device", "output_size", "output_layout", "output_device", "args"]
    vals = ["fallback_repeat", input_size, input_layout, input_device, output_size, output_layout, output_device, args]

    row = make_row(keys, vals)
    return row


def parse_layernorm(key, values):
    args = values[0]["META DATA"]
    forward_res = values[1]

    input_size = forward_res["INPUTS"].split("|")[0].replace("_", ",")
    input_layout = forward_res["INPUTS"].split("|")[1]
    input_device = forward_res["INPUTS"].split("|")[3]

    output_size = forward_res["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = forward_res["OUTPUTS"].split("|")[1]
    output_device = forward_res["OUTPUTS"].split("|")[3]

    keys = ["op", "input_size", "input_layout", "input_device", "output_size", "output_layout", "output_device", "args"]
    vals = [
        "fallback_layernorm",
        input_size,
        input_layout,
        input_device,
        output_size,
        output_layout,
        output_device,
        args,
    ]

    row = make_row(keys, vals)
    return row


def parse_groupnorm(key, values):
    args = values[0]["META DATA"]
    forward_res = values[1]

    input_size = forward_res["INPUTS"].split("|")[0].replace("_", ",")
    input_layout = forward_res["INPUTS"].split("|")[1]
    input_device = forward_res["INPUTS"].split("|")[3]

    output_size = forward_res["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = forward_res["OUTPUTS"].split("|")[1]
    output_device = forward_res["OUTPUTS"].split("|")[3]

    keys = ["op", "input_size", "input_layout", "input_device", "output_size", "output_layout", "output_device", "args"]
    vals = [
        "fallback_groupnorm",
        input_size,
        input_layout,
        input_device,
        output_size,
        output_layout,
        output_device,
        args,
    ]

    row = make_row(keys, vals)
    return row


def parse_conv(key, values):
    args = values[0]["META DATA"]
    forward_res = values[1]

    input_size = forward_res["INPUTS"].split("|")[0].replace("_", ",")
    input_layout = forward_res["INPUTS"].split("|")[1]
    input_device = forward_res["INPUTS"].split("|")[3]

    output_size = forward_res["OUTPUTS"].split("|")[0].replace("_", ",")
    output_layout = forward_res["OUTPUTS"].split("|")[1]
    output_device = forward_res["OUTPUTS"].split("|")[3]

    keys = ["op", "input_size", "input_layout", "input_device", "output_size", "output_layout", "output_device", "args"]
    vals = ["fallback_conv", input_size, input_layout, input_device, output_size, output_layout, output_device, args]

    row = make_row(keys, vals)
    return row


def eq(x: Dict, y: Dict):
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    return len(shared_items) == len(x.keys())


def add_to_list(new_op, op_list):
    for _ in op_list:
        if eq(new_op, _):
            return
    op_list.append(new_op)


def clean_ops(ops):
    cleaned_ops = {
        "chunk": [],
        "concat": [],
        "silu": [],
        "reshape": [],
        "full": [],
        "repeat": [],
        "conv2d": [],
        "layernorm": [],
    }

    for key, val in ops.items():
        if "chunk" in key:
            row = parse_chunk(key, val)
            add_to_list(row, cleaned_ops["chunk"])
            # cleaned_ops['chunk'].append(row)
        elif "concat" in key:
            row = parse_concat(key, val)
            add_to_list(row, cleaned_ops["concat"])
            # cleaned_ops['concat'].append(row)
        elif "silu" in key:
            row = parse_silu(key, val)
            add_to_list(row, cleaned_ops["silu"])
            # cleaned_ops['silu'].append(row)
        elif "reshape" in key:
            row = parse_reshape(key, val)
            add_to_list(row, cleaned_ops["reshape"])
            cleaned_ops["reshape"].append(row)
        elif "full" in key:
            row = parse_full(key, val)
            add_to_list(row, cleaned_ops["full"])
            # cleaned_ops['full'].append(row)
        elif "repeat" in key:
            row = parse_repeat_interleave(key, val)
            add_to_list(row, cleaned_ops["repeat"])
            # cleaned_ops['repeat'].append(row)
        else:
            if "Conv2d" in val[0]["NAME"]:
                row = parse_conv(key, val)
                # cleaned_ops['conv2d'].append(row)
                add_to_list(row, cleaned_ops["conv2d"])
            elif "LayerNorm" in val[0]["NAME"]:
                row = parse_layernorm(key, val)
                add_to_list(row, cleaned_ops["layernorm"])
                # cleaned_ops['layernorm'].append(row)
    return cleaned_ops


def write(cleaned_ops, path="cleaned_ops.csv"):
    output_path = path

    with open(output_path, "w", newline="") as csvfile:
        fieldnames = cleaned_ops[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for dc in cleaned_ops:
            writer.writerow(dc)


def run():
    fp = open(file_address, "r")
    ops = dictionize(fp)

    ops = merge_similar_ids(ops)

    cleaned_ops = clean_ops(ops)
    for k in cleaned_ops.keys():
        print(k)
        write(cleaned_ops[k], f"{k}_cleaned_ops.csv")


if __name__ == "__main__":
    run()
