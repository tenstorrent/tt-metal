#!/bin/bash

make programming_examples/loopback
./build/programming_examples/loopback 1             # <number of tiles>

make programming_examples/eltwise_binary
./build/programming_examples/eltwise_binary 1       # <number of tiles>
