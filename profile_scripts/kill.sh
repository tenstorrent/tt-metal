#!/bin/bash

for i in {1..200}
do
ps aux | grep ./build/test/llrt/test_run_risc_rw_speed_banked_dram | grep -v "grep" | awk '{print $2}' | xargs kill
sleep 1
done
