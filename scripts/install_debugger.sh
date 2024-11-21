sudo apt update
sudo apt install libzmq3-dev
pip uninstall -y tt-lens
pip install git+ssh://git@github.com/tenstorrent/tt-lens.git
