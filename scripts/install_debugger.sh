sudo apt update
sudo apt install libzmq3-dev git-lfs
git submodule foreach 'git lfs fetch --all && git lfs pull'
git lfs install
pip uninstall -y tt-lens
pip install git+ssh://git@github.com/tenstorrent/tt-lens.git
