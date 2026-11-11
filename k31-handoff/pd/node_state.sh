#!/bin/bash
me=$(id -u); n=$(pgrep -u $me -f 'pytest|prefill_runner|prefill_producer|prted|prterun' | wc -l)
echo "$(hostname -s) procs=$n kvm=$(docker ps --format '{{.Names}}' | grep -c kvm)"
