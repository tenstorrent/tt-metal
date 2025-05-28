# This script reads json trace and organizes read/write into sublists till barrier start/end

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks

create_plot = False
trace_file = (
    "/localdev/sohaibnadeem/tt-metal/generated/profiler/.logs/noc_trace_dev0_ID35840.json"
    if len(sys.argv) == 1
    else sys.argv[1]
)

#################### Read trace and organize ######################################
# Opening JSON file
f = open(trace_file)

# returns JSON object as a dictionary
data = json.load(f)

# Iterating through the json list

batched_tranfers = {}

for i in range(17):
    for j in range(12):
        for k in range(2):
            batched_tranfers[f"{i},{j},NOC_{k}"] = []

# print(batched_tranfers)
# print(barriers)

i = 0
while i < len(data):
    while i < len(data) and "type" not in data[i]:
        i += 1

    if i == len(data):
        break

    src_noc = None
    transfers = []
    while i < len(data) and data[i]["type"] != "WRITE_BARRIER_START" and data[i]["type"] != "READ_BARRIER_START":
        if src_noc is None:
            src_noc = f"{data[i]['sx']},{data[i]['sy']},{data[i]['noc']}"
        else:
            assert src_noc == f"{data[i]['sx']},{data[i]['sy']},{data[i]['noc']}"

        transfers.append(data[i]["timestamp"])
        i += 1

    # add barrier start and end timestamps
    if src_noc is None:
        src_noc = f"{data[i]['sx']},{data[i]['sy']},{data[i]['noc']}"
    else:
        assert src_noc == f"{data[i]['sx']},{data[i]['sy']},{data[i]['noc']}"

    assert src_noc == f"{data[i + 1]['sx']},{data[i + 1]['sy']},{data[i + 1]['noc']}"

    batched_tranfers[src_noc].append((transfers, data[i]["timestamp"], data[i + 1]["timestamp"]))
    i += 2

    # print(batched_tranfer_timestamps)

print(f"number of nocs: {len(batched_tranfers)}")

#################### Filtering ######################################
# batched_tranfers = {"4,4,NOC_0": batched_tranfers["4,4,NOC_0"], "4,5,NOC_0": batched_tranfers["4,5,NOC_0"]}
# barrier_start = {"4,4,NOC_0": barrier_start["4,4,NOC_0"], "4,5,NOC_0": barrier_start["4,5,NOC_0"]}
# barrier_end = {"4,4,NOC_0": barrier_end["4,4,NOC_0"], "4,5,NOC_0": barrier_end["4,5,NOC_0"]}

#################### Print max_transfer_dispatch_delay ######################################
# get max delay between transfers in a batch (to check for weird huge delay between reads)
max_transfer_dispatch_delay = 0
max_transfer_dispatch_delay_start_timestamp = None
max_transfer_dispatch_delay_noc = None
for src_noc, transfer_batches in batched_tranfers.items():
    for transfer_batch in transfer_batches:
        transfers = transfer_batch[0]
        if len(transfers) > 1:
            diff = np.diff(transfers)
            if np.max(diff) > max_transfer_dispatch_delay:
                max_transfer_dispatch_delay = np.max(diff)
                max_transfer_dispatch_delay_start_timestamp = transfers[np.argmax(diff)]
                max_transfer_dispatch_delay_noc = src_noc
print(
    f"max_transfer_dispatch_delay: {max_transfer_dispatch_delay} starting at {max_transfer_dispatch_delay_start_timestamp} on {max_transfer_dispatch_delay_noc}"
)

#################### Plotting ######################################
if not create_plot:
    exit()

# conver to x and y coords
t_x = np.array(
    [
        transfer
        for core_batched_tranfers in batched_tranfers.values()
        for batch in core_batched_tranfers
        for transfer in batch[0]
    ]
)
t_y = np.array([core for core in batched_tranfers.keys() for batch in batched_tranfers[core] for transfer in batch[0]])
b_x = np.array([batch[1] for core_batched_tranfers in batched_tranfers.values() for batch in core_batched_tranfers])
b_y = np.array([core for core in batched_tranfers.keys() for batch in batched_tranfers[core]])
e_x = np.array([batch[2] for core_batched_tranfers in batched_tranfers.values() for batch in core_batched_tranfers])
e_y = np.array([core for core in batched_tranfers.keys() for batch in batched_tranfers[core]])

# print(t_x)
# print(t_y)
# print(b_x)
# print(b_y)
# print(e_x)
# print(e_y)

# plot transfer timeline
fig, ax = plt.subplots(figsize=(200, 80))
# ax.set_xlim(0, 2000)

ax.scatter(t_x, t_y, c="blue")
ax.scatter(b_x, b_y, c="red")
ax.scatter(e_x, e_y, c="green")
fig.savefig("transfers.png")

# Closing file
f.close()
