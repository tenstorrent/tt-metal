#!/bin/bash
# Wait for blx01 broker job $1 to leave queued/running, then run post.sh $2.
J=$1; L=$2
while ssh -o ConnectTimeout=20 g15blx01 "tt-device-mcp status -j $J" 2>/dev/null | grep -qiE 'Status: +(running|queued)'; do sleep 30; done
ssh g15blx01 "tt-device-mcp status -j $J" > /home/smarton/fasth3/tt-metal/tt-project/t185/job$J.status 2>&1
bash /home/smarton/fasth3/tt-metal/tt-project/t185/post.sh $L
