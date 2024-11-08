# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Initialize submodules
git submodule update --init --recursive
# Run setup script to configure env variables, direnv, clang-tidy and clang-format
chmod +x init_repo.sh
source ./init_repo.sh
# Direnv allow
direnv allow .
# Build metal library
cd 3rd_party/tt-metal
chmod +x build_metal.sh
./build_metal.sh -b Release
cd ../..
# Build project
cmake -DCMAKE_BUILD_TYPE=Release -B build -GNinja
cmake --build build --config Release --clean-first
