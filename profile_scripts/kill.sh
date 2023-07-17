#!/bin/bash

for i in {1..200}
do
ps aux | grep test_run_risc | grep -v "grep" | awk '{print $2}' | xargs kill
sleep 1
done
