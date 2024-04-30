# Tools
This section points to the opensource tools used by tt-metal devices and a quick installation of them. Check out the public repos for each tool for more information. Note: this is only for wormhole_b0 devices.

## Opensource Repositories
tt-kmd: https://github.com/tenstorrent/tt-kmd
tt-flash: https://github.com/tenstorrent/tt-flash
tt-firmware: https://github.com/tenstorrent/tt-firmware.git
tt-smi: https://github.com/tenstorrent/tt-smi
tt-topology: https://github.com/tenstorrent/tt-topology

## Installation
Usage: note you will need sudo access to install these tools. You can run these in order for clean installation of all the tools. Note: you may need to reset at the start to make sure your board is in a good state. Install and run tt-smi for that.
Install tt-kmd
```
./install_tt-kmd.sh
```

Install tt-smi
```
./install_tt-smi.sh
```

Install tt-firmware
```
./install_tt-firmware.sh
```

Install tt-topology
```
./install_tt-topology.sh
```

## Usage

To use the tools
```
source $HOME/.tools_env/bin/activate
export PATH="$PATH:$HOME/.local/bin"
tt-smi --version
tt-topology --version
```
