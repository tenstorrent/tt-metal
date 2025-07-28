#!/bin/bash
set -eo pipefail

# ----------------------------
# POLISH - COLOR SCHEME
# ----------------------------
RESET="\e[0m"
GREEN="\e[32m"
YELLOW="\e[33m"
CYAN="\e[36m"
MAGENTA="\e[35m"
BOLD="\e[1m"
STEP="${CYAN}${BOLD}[Step]${RESET}"
CHECK="${GREEN}[✓]${RESET}"
WARN="${YELLOW}[!]${RESET}"

echo -e "${MAGENTA}${BOLD}--- TT-Metal Debian Installer ---${RESET}"
echo -e "${CYAN}This script will set up TT-Metal on Debian.${RESET}\n"

# ----------------------------
# STEP 1: Update system & install kernel headers
# ----------------------------
echo -e "${STEP} Installing latest kernel & headers..."
sudo apt update
sudo apt install -y linux-image-amd64 linux-headers-amd64
echo -e "${CHECK} Kernel and headers installed. A reboot may be required if a new kernel was installed.\n"

# ----------------------------
# STEP 2: Offer swapfile setup
# ----------------------------
echo -e "${STEP} Checking for swapfile..."
RAM_SIZE=$(grep MemTotal /proc/meminfo | awk '{print $2}')
if [ "$RAM_SIZE" -lt 8000000 ]; then
    echo -e "${WARN} Your system has less than 8GB of RAM. A swapfile is recommended."
fi

read -p "Do you want to create an 8GB swapfile? [Y/n]: " swap_choice
swap_choice=${swap_choice:-Y}
if [[ "$swap_choice" =~ ^[Yy]$ ]]; then
    echo -e "${STEP} Creating 8GB swapfile..."
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo -e "${CHECK} Swapfile created and enabled.\n"
else
    echo -e "${YELLOW}Skipped swapfile creation.${RESET}\n"
fi

# ----------------------------
# STEP 3: Install Rust (latest via rustup)
# ----------------------------
echo -e "${STEP} Installing Rust..."
sudo apt remove -y rustc cargo || true
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup update stable
echo -e "${CHECK} Rust installed and updated.\n"

# ----------------------------
# STEP 4: Install Git and clone tt-metal
# ----------------------------
echo -e "${STEP} Installing Git..."
sudo apt install -y git
echo -e "${CHECK} Git installed.\n"

if [ ! -d "tt-metal" ]; then
    echo -e "${STEP} Cloning TT-Metal repo with submodules..."
    git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
    cd tt-metal
else
    echo -e "${YELLOW}TT-Metal already cloned. Skipping.${RESET}"
    cd tt-metal
fi

# ----------------------------
# STEP 5: Use Tiam's install_dependencies.sh
# ----------------------------
echo -e "${STEP} Fetching custom install_dependencies.sh..."
curl -sSL https://raw.githubusercontent.com/tiamoesg/tt-metal/main/install_dependencies.sh -o install_dependencies.sh
chmod +x install_dependencies.sh
./install_dependencies.sh
echo -e "${CHECK} Dependencies installed.\n"

# ----------------------------
# STEP 6: Create Python Virtual Environment
# ----------------------------
echo -e "${STEP} Fetching custom create_venv.sh..."
curl -sSL https://raw.githubusercontent.com/tiamoesg/tt-metal/main/create_venv.sh -o create_venv.sh
chmod +x create_venv.sh
source create_venv.sh
echo -e "${CHECK} Python venv created and activated.\n"

# ----------------------------
# STEP 7: Build TT-Metal
# ----------------------------
echo -e "${STEP} Fetching custom build_metal.sh..."
curl -sSL https://raw.githubusercontent.com/tiamoesg/tt-metal/main/build_metal.sh -o build_metal.sh
chmod +x build_metal.sh
./build_metal.sh --build-all
echo -e "${CHECK} TT-Metal built successfully.\n"

echo -e "${STEP} Generating Python stubs..."
./scripts/build_scripts/create_stubs.sh

# ----------------------------
# STEP 8: Install TT-KMD driver
# ----------------------------
echo -e "${STEP} Installing TT-KMD driver..."
cd ..
if [ ! -d "tt-kmd" ]; then
    git clone https://github.com/tenstorrent/tt-kmd.git
fi
cd tt-kmd
sudo apt install -y dkms
sudo dkms add .
sudo dkms install "tenstorrent/$(./tools/current-version)"
sudo modprobe tenstorrent
echo -e "${CHECK} TT-KMD installed.\n"
cd ..

# ----------------------------
# STEP 9: Firmware + Flash
# ----------------------------
if [ ! -d "tt-flash" ]; then
    git clone https://github.com/tenstorrent/tt-flash.git
fi
cd tt-flash
pip install --upgrade pip
pip3 install .
cd ..

if [ ! -d "tt-firmware" ]; then
    git clone https://github.com/tenstorrent/tt-firmware.git
fi
cd tt-firmware
tt-flash --fw-tar fw_pack-18.6.0.fwbundle
cd ..

# ----------------------------
# STEP 10: Hugepages
# ----------------------------
echo -e "${STEP} Enabling hugepages..."
sudo systemctl enable --now tenstorrent-hugepages.service || true
sudo systemctl enable --now 'dev-hugepages\x2d1G.mount' || true
echo -e "${CHECK} Hugepages enabled.\n"

# ----------------------------
# FINISHED
# ----------------------------
echo -e "${GREEN}${BOLD}🚀 Install complete!${RESET}"

# Activate the Python virtual environment
cd tt-metal
echo -e "\n${GREEN}[✓] Activating Python virtual environment...${RESET}"
source python_env/bin/activate
echo -e "${GREEN}${BOLD}🚀 venv Activated!${RESET}"

echo -e "${MAGENTA}In future sessions: Run ${CYAN}source python_env/bin/activate${MAGENTA} inside tt-metal to begin coding.${RESET}"
