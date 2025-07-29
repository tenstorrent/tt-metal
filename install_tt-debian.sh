#!/bin/bash
set -eo pipefail

# ----------------------------
# POLISH - COLOR SCHEME & LOGGING
# ----------------------------
RESET="\e[0m"
GREEN="\e[32m"
YELLOW="\e[33m"
CYAN="\e[36m"
MAGENTA="\e[35m"
RED="\e[31m"
BOLD="\e[1m"

# Log helpers
log_info()       { echo -e "${CYAN}[INFO] $*${RESET}"; }
log_warn()       { echo -e "${YELLOW}[WARN] $*${RESET}"; }
log_error()      { echo -e "${RED}[ERROR] $*${RESET}"; }
log_step()       { echo -e "${CYAN}${BOLD}[STEP] $*${RESET}"; }
log_success()    { echo -e "${GREEN}[✓] $*${RESET}"; }
log_note()       { echo -e "${MAGENTA}[NOTE] $*${RESET}"; }

# ----------------------------
# SCRIPT START
# ----------------------------
log_note "--- TT-Metal Debian Installer ---"
log_info "This script will set up TT-Metal on Debian."

# ----------------------------
# STEP 1: Update system & install kernel headers
# ----------------------------
log_step "Installing latest kernel & headers..."
sudo apt update
sudo apt install -y linux-image-amd64 linux-headers-amd64
log_success "Kernel and headers installed. A reboot may be required if a new kernel was installed."

# ----------------------------
# STEP 2: Offer swapfile setup
# ----------------------------
log_step "Checking for swapfile..."
RAM_SIZE=$(grep MemTotal /proc/meminfo | awk '{print $2}')
if [ "$RAM_SIZE" -lt 8000000 ]; then
    log_warn "Your system has less than 8GB of RAM. A swapfile is recommended."
fi

read -p "Do you want to create an 8GB swapfile? [Y/n]: " swap_choice
swap_choice=${swap_choice:-Y}
if [[ "$swap_choice" =~ ^[Yy]$ ]]; then
    log_step "Creating 8GB swapfile..."
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    log_success "Swapfile created and enabled."
else
    log_warn "Skipped swapfile creation."
fi

# ----------------------------
# STEP 3: Install Rust (latest via rustup)
# ----------------------------
log_step "Installing Rust..."
sudo apt remove -y rustc cargo || true
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup update stable
log_success "Rust installed and updated."

# ----------------------------
# STEP 4: Install Git and clone tt-metal
# ----------------------------
log_step "Installing Git..."
sudo apt install -y git
log_success "Git installed."

if [ ! -d "tt-metal" ]; then
    log_step "Cloning TT-Metal repo with submodules..."
    git clone https://github.com/tenstorrent/tt-metal.git --recurse-submodules
    cd tt-metal
else
    log_warn "TT-Metal already cloned. Skipping."
    cd tt-metal
fi

# ----------------------------
# STEP 5: install_dependencies.sh
# ----------------------------
chmod +x install_dependencies.sh
./install_dependencies.sh
log_success "Dependencies installed."

# ----------------------------
# STEP 6: Create Python Virtual Environment
# ----------------------------
chmod +x create_venv.sh
source create_venv.sh
log_success "Python venv created and activated."

# ----------------------------
# STEP 7: Build TT-Metal
# ----------------------------
chmod +x build_metal.sh
./build_metal.sh --build-all
log_success "TT-Metal built successfully."

log_step "Generating Python stubs..."
./scripts/build_scripts/create_stubs.sh

# ----------------------------
# STEP 8: Install TT-KMD driver
# ----------------------------
log_step "Installing TT-KMD driver..."
cd ..
if [ ! -d "tt-kmd" ]; then
    git clone https://github.com/tenstorrent/tt-kmd.git
fi
cd tt-kmd
sudo apt install -y dkms
sudo dkms add .
sudo dkms install "tenstorrent/$(./tools/current-version)"
sudo modprobe tenstorrent
log_success "TT-KMD installed."
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
log_step "Enabling hugepages..."
sudo systemctl enable --now tenstorrent-hugepages.service || true
sudo systemctl enable --now 'dev-hugepages\x2d1G.mount' || true
log_success "Hugepages enabled."

# ----------------------------
# FINISHED
# ----------------------------
log_success "🚀 Install complete!"

cd tt-metal
log_step "Activating Python virtual environment..."
source python_env/bin/activate
log_success "🚀 venv Activated!"

log_note "In future sessions: Run 'source python_env/bin/activate' inside tt-metal to begin coding."
