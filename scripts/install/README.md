# Tools
This section points to the opensource tools used by tt-metal devices and a quick installation of them. Check out the public repos for each tool for more information.

## Opensource Repositories
tt-kmd: https://github.com/tenstorrent/tt-kmd
tt-flash: https://github.com/tenstorrent/tt-flash
tt-firmware: https://github.com/tenstorrent/tt-firmware.git
tt-smi: https://github.com/tenstorrent/tt-smi
tt-topology: https://github.com/tenstorrent/tt-topology

## Installation
Usage: note you will need sudo access to install these tools

Install tt-firmware
```
./install_tt-firmware.sh
```

Install tt-kmd
```
./install_tt-kmd.sh
```

Install tt-smi
```
./install_tt-smi.sh
```

Install tt-topology
```
./install_tt-topology.sh
```

## Usage

To use the tools
```
source $HOME/.venv/bin/activate
export PATH="$PATH:$HOME/.local/bin"
tt-smi --version
tt-topology --version
```
