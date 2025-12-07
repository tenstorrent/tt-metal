# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI
#
# SPDX-License-Identifier: Apache-2.0

# Initialize submodules
git submodule update --init --recursive
# Run setup script to configure env variables, direnv, clang-tidy and clang-format
chmod +x init_repo.sh
source ./init_repo.sh

# Build project
cmake -DCMAKE_BUILD_TYPE=Release -B build -GNinja
cmake --build build --config Release --clean-first
