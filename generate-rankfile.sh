#!/bin/bash

set -e

num_ranks=$1

if [[ "$num_ranks" != "1" && "$num_ranks" != "2" && "$num_ranks" != "4" ]]; then
    echo "Usage: $0 <1|2|4>"
    exit 1
fi

IP="172.27.29.92"  # All ranks use the same IP

> rankfile  # Clear or create rankfile

for (( rank=0; rank<num_ranks; rank++ )); do
    echo "rank $rank=$IP slot=0:$rank" >> rankfile
done

echo "Generated rankfile:"
cat rankfile

