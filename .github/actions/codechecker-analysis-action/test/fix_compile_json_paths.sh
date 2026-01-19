#!/bin/bash
set -ex

for CCDB in $(find . -name "compile_commands.json"); do
    pushd $(dirname "$CCDB")
    sed -i "s,__DIRECTORY__,$(pwd),g" "compile_commands.json"
    popd
done
