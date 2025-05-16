# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Description: Initialize the repository with the necessary configurations
sudo apt install clang-tidy-17
sudo apt install clang-format-17
sudo ln -sf /usr/bin/clang-tidy-17 /usr/bin/clang-tidy
sudo ln -sf /usr/bin/clang-format-17 /usr/bin/clang-format
sudo apt install pre-commit
pre-commit install

sudo apt-get install python3-dev python3-numpy cargo
pip install wandb
pip install numpy

sudo apt-get install -y libopenmpi-dev openmpi-bin
