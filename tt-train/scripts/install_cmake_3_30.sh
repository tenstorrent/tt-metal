# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install build-essential libssl-dev
CURRENT_PATH=$(pwd)
cd /tmp
wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0.tar.gz
tar -zxvf cmake-3.30.0.tar.gz
cd cmake-3.30.0
./bootstrap
make -j$(nproc)
sudo make install
cd $CURRENT_PATH
source ~/.bashrc
