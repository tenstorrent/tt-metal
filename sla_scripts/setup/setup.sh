#!/bin/bash
set -e
cd ~/teja
exec > >(tee -a ~/teja/setup.log) 2>&1
echo "=== clone $(date)"
[ -d tt-metal ] || git clone --recurse-submodules -b main https://github.com/tenstorrent/tt-metal.git
cd tt-metal
echo "=== build $(date)"
./build_metal.sh
echo "=== venv $(date)"
./create_venv.sh
echo "=== DONE $(date)"
