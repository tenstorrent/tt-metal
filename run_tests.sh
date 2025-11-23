#!/bin/bash

# Run the tests one by one
while IFS= read -r test_node_id; do
    echo "Running test: $test_node_id"
    pytest "$test_node_id"
done < $1
