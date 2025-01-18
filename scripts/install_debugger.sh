sudo apt update
sudo apt install libzmq3-dev
# sudo apt install git-lfs
# git lfs install
# git submodule foreach 'git lfs fetch --all && git lfs pull'
pip uninstall -y tt-lens
pip install git+ssh://git@github.com/tenstorrent/tt-lens.git
