#!/bin/bash

set -eo pipefail

GIT_DIR=$(pwd)

cd ~

curl -L https://www.doxygen.nl/files/doxygen-1.9.6.linux.bin.tar.gz -o doxygen-1.9.6.linux.bin.tar.gz
tar -xvf doxygen-1.9.6.linux.bin.tar.gz
cd doxygen-1.9.6
sudo make install

cd $GIT_DIR

doxygen
cd docs
python3 -m venv env
source env/bin/activate
pip install -r requirements-docs.txt
