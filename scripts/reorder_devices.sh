#!/bin/bash

#  The script will reorder the devices based on the desired coordinates sequence. 
#  Run the script with the following command: 
#  sudo bash reorder_devices.sh <input_directory>
#  The script will use the latest timestamped .json file in the input directory to get the current device order.
#  If the input directory is not provided, the script will use the default directory ~/tt_smi_logs.
#  The script will reorder the devices based on the desired coordinates sequence. If the devices are already in the desired order, the script will exit. 
#  The script will output the following message: 
#  Successfully unbinded and binded TT devices into default order

# check if have sudo permissions
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

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

# Desired coordinates sequence of the TT devices - defaults to t3k
desired_coords=(
    "(1, 0, 0, 0)"
    "(1, 1, 0, 0)"
    "(2, 1, 0, 0)"
    "(2, 0, 0, 0)"
)

# Generate map of device bus coordinates and bus IDs, also collect current sequence of TT Device coordinates
echo "Processing $latest_file"

declare -A coord_mapping
mapping=$(jq -r '.device_info[] | .board_info | "\(.coords):\(.bus_id)"' "$latest_file")

while IFS= read -r line; do
    coords=$(echo "$line" | cut -d':' -f1 | xargs)
    bus_id=$(echo "$line" | cut -d':' -f2- | xargs)
    coord_mapping["$coords"]="$bus_id"
    init_coords_order+=("$coords")
done <<< "$mapping"

# Check if the devices are already in the desired order
devices_in_order=1
for i in "${!desired_coords[@]}"; do
    if [ "${desired_coords[$i]}" != "${init_coords_order[$i]}" ]; then
        devices_in_order=0
        break
    fi
done

if [ $devices_in_order -eq 1 ]; then
    echo "Devices are already in the desired order!"
    exit 0
else
    echo "Reordering devices..."
fi

# Declare an array with the bus IDs TT device to rebind into corect order
declare -a arr
for coord in "${desired_coords[@]}"; do
    arr+=("${coord_mapping["$coord"]}")
done

# Unbind TT Devices
for i in "${arr[@]}"
do
    cd /sys/bus/pci/drivers/tenstorrent
    echo -n $i > unbind
    if [ $? -ne 0 ]; then
        echo "Failed to unbind device $i, please reboot the machine"
        exit 1
    fi
done

# Bind TT Devices
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
echo "Run tt-topology again for the updated device order!"
 