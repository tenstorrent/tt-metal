#!/bin/bash

# Check if tt-smi is installed
if ! command -v tt-smi &> /dev/null; then
    echo "tt-smi is not installed. Please install it from https://github.com/tenstorrent/tt-smi"
    exit 1
fi

# Check if the input directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_directory> not given - using default ~/tt_smi_logs as input" 
    tt-smi -s
    input_directory=~/tt_smi_logs
else
    input_directory="$1"
fi

# Get the latest timestamped .json file in the input directory
latest_file=$(ls -t "$input_directory"/*.json | head -n 1)

# Check if a file exists
if [ -z "$latest_file" ]; then
    echo "No .json file found in $input_directory"
    exit 1
elif [ ! -f "$latest_file" ]; then
    echo "File does not exist: $latest_file"
    exit 1
fi

# Generate map of device bus coordinates and bus IDs 
echo "Processing $latest_file"
declare -A coord_mapping
mapping=$(jq -r '.device_info[] | .board_info | "\(.coords):\(.bus_id)"' "$latest_file")

while IFS= read -r line; do
    coords=$(echo "$line" | cut -d':' -f1 | xargs)
    bus_id=$(echo "$line" | cut -d':' -f2- | xargs)
    
    coord_mapping["$coords"]="$bus_id"
done <<< "$mapping"

# Declare an array with the values from the associative array
declare -a arr
arr[0]="${coord_mapping["(1, 0, 0, 0)"]}"
arr[1]="${coord_mapping["(1, 1, 0, 0)"]}"
arr[2]="${coord_mapping["(2, 1, 0, 0)"]}"
arr[3]="${coord_mapping["(2, 0, 0, 0)"]}"

## now loop through the above array
for i in "${arr[@]}"
do
    cd /sys/bus/pci/drivers/tenstorrent
    echo -n $i > unbind
    if [ $? -ne 0 ]; then
        echo "Failed to unbind device $i, please reboot the machine"
        exit 1
    fi
done

for i in "${arr[@]}"
do
    cd /sys/bus/pci/drivers/tenstorrent
    echo -n $i > bind
    if [ $? -ne 0 ]; then
        echo "Failed to bind device $i, please reboot the machine"
        exit 1
    fi
done
echo "Successfully unbinded and binded TT devices into default order"
