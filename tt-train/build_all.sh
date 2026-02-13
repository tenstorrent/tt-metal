# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Initialize submodules
git submodule update --init --recursive
# Run setup script to configure env variables, direnv, clang-tidy and clang-format
chmod +x init_repo.sh
source ./init_repo.sh

# Build project with clang-20 (must match tt-metal's compiler for ABI compat)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-20 \
      -DCMAKE_CXX_COMPILER=clang++-20 \
      -B build -GNinja
cmake --build build --config Release --clean-first
