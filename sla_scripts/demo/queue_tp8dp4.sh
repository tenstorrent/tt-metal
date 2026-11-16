#!/bin/bash
bash ~/teja/run_tp8dp4.sh dp4_b8 --batch_size 8
bash ~/teja/run_tp8dp4.sh dp4_b32
echo "=== TP8DP4 QUEUE DONE $(date)" >> ~/teja/runs/queue.log
