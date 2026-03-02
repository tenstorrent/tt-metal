#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

usage()
{
    echo "Usage: sudo ./install_dependencies.sh [options]"
    echo
    echo "[--help, -h]                List this help"
    echo "[--validate, -v]            Validate that required packages are installed"
    echo "[--docker, -d]              Specialize execution for docker"
    echo "[--no-distributed]          Don't install distributed compute dependencies (OpenMPI)"
    echo "[--hugepages]               Install hugepages dependency"
    echo "[--sfpi]                    Install only SFPI package (minimal installation)"
    echo "[--source-only]             Loads functions into shell"
    exit 1
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
        OS_CODENAME="${UBUNTU_CODENAME:VERSION_CODENAME}"
        OS_ID_LIKE="$ID_LIKE"
    else
        echo "Error: /etc/os-release not found. Unsupported system."
        exit 1
    fi

    # Detect package manager
    for mgr in apt dnf yum; do
        if command -v "$mgr" >/dev/null 2>&1; then
            PKG_MANAGER="$mgr"
            break
        fi
    done
    if [ -z "$PKG_MANAGER" ]; then
        echo "Error: No supported package manager found"
        exit 1
    fi

    echo "Detected OS: $OS_ID $OS_VERSION ($PKG_MANAGER)"

    get_package_family
}

get_package_family() {
    case "$OS_ID" in
        ubuntu|debian)
            PKG_FAMILY="debian"
            ;;
        fedora|centos|rhel|rocky|almalinux)
            PKG_FAMILY="redhat"
            ;;
        *)
            # For distributions that are similar to supported ones
            if [[ "$OS_ID_LIKE" == *"debian"* ]] || [[ "$OS_ID_LIKE" == *"ubuntu"* ]]; then
                PKG_FAMILY="debian"
            elif [[ "$OS_ID_LIKE" == *"rhel"* ]] || [[ "$OS_ID_LIKE" == *"fedora"* ]]; then
                PKG_FAMILY="redhat"
            else
                PKG_FAMILY="unknown"
            fi
            ;;
    esac
}

is_debian_based() {
    [[ "$PKG_FAMILY" == "debian" ]]
}

is_redhat_based() {
    [[ "$PKG_FAMILY" == "redhat" ]]
}

is_supported_os() {
    case "$OS_ID" in
        ubuntu|debian)
            return 0
            ;;
        fedora|centos|rhel|rocky|almalinux)
            return 0
            ;;
        *)
            if [[ "$OS_ID_LIKE" == *"debian"* ]] || [[ "$OS_ID_LIKE" == *"ubuntu"* ]]; then
                return 0
            elif [[ "$OS_ID_LIKE" == *"rhel"* ]] || [[ "$OS_ID_LIKE" == *"fedora"* ]]; then
                return 0
            else
                return 1
            fi
            ;;
    esac
}

update_package_cache() {
    echo "[INFO] Updating package cache..."
    case "$PKG_MANAGER" in
        apt)
            apt-get update
            ;;
        dnf)
            dnf makecache
            ;;
        yum)
            yum makecache
            ;;
    esac
}

install_packages() {
    echo "[INFO] Installing packages: ${PACKAGES[*]}"
    case "$PKG_MANAGER" in
        apt)
            DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends "${PACKAGES[@]}"
            ;;
        dnf)
            dnf install -y "${PACKAGES[@]}"
            ;;
        yum)
            yum install -y "${PACKAGES[@]}"
            ;;
    esac
}

validate_packages() {
    echo "[INFO] Validating packages:"
    case "$PKG_MANAGER" in
        apt)
            dpkg-query -W -f='  ${Package} ${Status}\n' "${PACKAGES[@]}"
            echo "[INFO] Validation successful!"
            ;;
        dnf|yum)
            rpm -q --qf '  %{NAME} %{VERSION}-%{RELEASE}\n' "${PACKAGES[@]}"
            echo "[INFO] Validation successful!"
            ;;
    esac
}

cleanup_package_cache() {
    echo "[INFO] Cleaning up package cache..."
    case "$PKG_MANAGER" in
        apt)
            apt-get clean && rm -rf /var/lib/apt/lists/*
            ;;
        dnf)
            dnf clean all
            ;;
        yum)
            yum clean all
            ;;
    esac
}

# Initialize packages
init_packages() {
    # Check if package family is supported
    if [[ "$PKG_FAMILY" == "unknown" ]]; then
        echo "[ERROR] No package list available for $OS_ID (ID_LIKE: $OS_ID_LIKE) (PKG_FAMILY: $PKG_FAMILY)"
        exit 1
    fi

    # Set packages based on determined family
    case "$PKG_FAMILY" in
        debian)
            # Determine g++ version based on Ubuntu version
            local gpp_package="g++"
            if [[ "$OS_ID" == "ubuntu" ]]; then
                case "$OS_VERSION" in
                    "22.04")
                        gpp_package="g++-12"
                        echo "[INFO] Using g++-12 for Ubuntu 22.04 (gcc-12 will be installed as dependency)"
                        ;;
                    "24.04")
                        gpp_package="g++-14"
                        echo "[INFO] Using g++-14 for Ubuntu 24.04 (gcc-14 will be installed as dependency)"
                        ;;
                    *)
                        echo "[INFO] Using default g++ for Ubuntu $OS_VERSION"
                        ;;
                esac
            fi

            # All packages needed for TT-Metal development
            PACKAGES=(
                "git"
                "build-essential"
                "cmake"
                "ninja-build"
                "pkg-config"
                "$gpp_package"
                "pandoc"
                "xz-utils"
                "openssl"
                "libssl-dev"
                "python3-dev"
                "python3-pip"
                "python3-venv"
                "python3-pkg-resources" # needed for setuptools
                "libhwloc-dev"
                "libnuma-dev"
                "libatomic1"
                "libstdc++6"
                "libtbb-dev"
                "libcapstone-dev"
                "libc++-20-dev"
                "libc++abi-20-dev"
                "wget"
                "curl"
                "xxd"
            )
            if [ "$distributed" -eq 1 ]; then
                PACKAGES+=("openmpi-bin" "libopenmpi-dev")
            fi
            ;;
        redhat)
            PACKAGES=(
                "git"
                "gcc"
                "gcc-c++"
                "make"
                "llvm"
                "llvm-devel"
                "clang"
                "clang-tools-extra" # for linker-wrapper
                "cmake"
                "ninja-build"
                "openssl"
                "openssl-devel"
                "pkgconf-pkg-config"
                "xz"
                "python3-devel"
                "python3-pip"
                "hwloc-devel"
                "numactl-devel"
                "libatomic"
                "libstdc++"
                "tbb-devel"
                "capstone-devel"
                "boost-devel"
                "gmp-devel"
                "wget"
                "curl"
                "vim-common" # Includes xxd
            )
            if [ "$distributed" -eq 1 ]; then
                PACKAGES+=("openmpi" "openmpi-devel")
            fi
            ;;
    esac
}

prep_system() {
    echo "[INFO] Preparing system for TT-Metal development ($OS_ID)..."

    case "$PKG_FAMILY" in
        debian)
            prep_ubuntu_system
            ;;
        redhat)
            prep_redhat_system
            ;;
        *)
            echo "[WARNING] No specific system preparation for $OS_ID"
            ;;
    esac
}

prep_ubuntu_system() {
    echo "[INFO] Preparing Ubuntu/Debian system..."
    # Update package lists and install basic tools
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates gpg lsb-release wget software-properties-common gnupg jq

    # Add LLVM repository for Clang 17
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
    echo "deb http://apt.llvm.org/$OS_CODENAME/ llvm-toolchain-$OS_CODENAME-17 main" | tee /etc/apt/sources.list.d/llvm-17.list
    # Also v20
    echo "deb http://apt.llvm.org/$OS_CODENAME/ llvm-toolchain-$OS_CODENAME-20 main" | tee /etc/apt/sources.list.d/llvm-20.list

    # Add Kitware repository for latest CMake
    # If the kitware-archive-keyring package has not been installed previously, manually obtain a copy of our signing key
    test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

    # Add the repository to sources list and update
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null
    apt-get update

    # If the kitware-archive-keyring package was not installed previously, remove the manually obtained key to make room for the package
    test -f /usr/share/doc/kitware-archive-keyring/copyright || rm /usr/share/keyrings/kitware-archive-keyring.gpg

    # Install the kitware-archive-keyring package to ensure that your keyring stays up to date as keys are rotated
    apt-get install -y --no-install-recommends kitware-archive-keyring

    # Add GCC toolchain repository for specific g++ versions if needed
    if [[ "$OS_ID" == "ubuntu" ]]; then
        case "$OS_VERSION" in
            "24.04")
                echo "[INFO] Adding toolchain repository for g++-14 on Ubuntu 24.04"
                add-apt-repository -y ppa:ubuntu-toolchain-r/test
                ;;
        esac
    fi

    apt-get update
}

prep_redhat_system() {
    echo "[INFO] Preparing Red Hat family system..."

    # Enable CodeReady Builder (crb) repo for EL9 - provides ninja-build, etc.
    if [[ "$OS_ID" == "rocky" || "$OS_ID" == "almalinux" || "$OS_ID" == "centos" ]]; then
        echo "[INFO] Enabling CRB (CodeReady Builder) repository..."
        dnf config-manager --set-enabled crb 2>/dev/null || \
            dnf config-manager --enable crb 2>/dev/null || \
            echo "[WARNING] Could not enable CRB repository. Some packages may not be available."
    elif [[ "$OS_ID" == "rhel" ]]; then
        echo "[INFO] Enabling CodeReady Builder repository for RHEL..."
        subscription-manager repos --enable codeready-builder-for-rhel-${OS_VERSION%%.*}-x86_64-rpms 2>/dev/null || \
            echo "[WARNING] Could not enable CodeReady Builder. Some packages may not be available."
    fi

    # Install EPEL if available (provides extra packages)
    if ! rpm -q epel-release &>/dev/null; then
        echo "[INFO] Installing EPEL repository..."
        dnf install -y epel-release 2>/dev/null || \
            echo "[WARNING] Could not install EPEL. Some packages may not be available."
    fi

    # Install llvm-toolset which bundles clang and llvm for EL systems
    echo "[INFO] Installing llvm-toolset..."
    dnf install -y llvm-toolset 2>/dev/null || true

    dnf makecache

    # Build libisl from source (v0.23+ required but not in EL9 repos)
    install_libisl_from_source

    # Configure OpenMPI paths for RHEL-family distributions
    configure_redhat_openmpi_paths
}

install_libisl_from_source() {
    local ISL_VERSION="0.26"
    local ISL_URL="https://libisl.sourceforge.io/isl-${ISL_VERSION}.tar.gz"
    local ISL_PREFIX="/usr"

    # Check if libisl is already installed with sufficient version
    if ldconfig -p 2>/dev/null | grep -q 'libisl.so.23'; then
        echo "[INFO] libisl.so.23 already available, skipping source build."
        return
    fi

    echo "[INFO] Building libisl ${ISL_VERSION} from source (not available in EL9 repos)..."

    # Install build dependencies for libisl
    dnf install -y gmp-devel autoconf automake libtool 2>/dev/null || true

    local TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    wget -q --show-progress -O "isl-${ISL_VERSION}.tar.gz" "$ISL_URL"
    tar xzf "isl-${ISL_VERSION}.tar.gz"
    cd "isl-${ISL_VERSION}"

    ./configure --prefix="$ISL_PREFIX"
    make -j"$(nproc)"
    make install

    # Update linker cache
    ldconfig

    cd /
    rm -rf "$TEMP_DIR"

    echo "[INFO] libisl ${ISL_VERSION} installed successfully."
}

configure_redhat_openmpi_paths() {
    # On RHEL-family distros, OpenMPI is installed under /usr/lib64/openmpi
    # rather than in the standard PATH. We need to configure the environment.
    local OPENMPI_BASE="/usr/lib64/openmpi"

    if [ -d "$OPENMPI_BASE" ]; then
        echo "[INFO] Configuring OpenMPI paths for RHEL-family distribution..."

        # Create a profile script that will be sourced on login
        cat > /etc/profile.d/tt-metal-openmpi.sh << 'OPENMPI_PROFILE'
# TT-Metal OpenMPI configuration for RHEL-family distributions
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export C_INCLUDE_PATH=/usr/lib64/openmpi/include:${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/lib64/openmpi/include:${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
export PKG_CONFIG_PATH=/usr/lib64/openmpi/lib/pkgconfig:${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
OPENMPI_PROFILE
        chmod 644 /etc/profile.d/tt-metal-openmpi.sh

        # Also source it for the current session
        source /etc/profile.d/tt-metal-openmpi.sh

        echo "[INFO] OpenMPI paths configured. Profile script installed at /etc/profile.d/tt-metal-openmpi.sh"
    fi
}

# We currently have an affinity to clang as it is more thoroughly tested in CI
# However g++-12 and later should also work

install_llvm() {
    if is_redhat_based; then
        # On RHEL-family, clang is provided by llvm-toolset (already installed in prep_redhat_system)
        echo "[INFO] Verifying LLVM/Clang installation on RHEL-family system..."
        if command -v clang &> /dev/null; then
            local clang_version=$(clang --version 2>/dev/null | head -1)
            echo "[INFO] Found: $clang_version"
        else
            echo "[WARNING] Clang not found. Installing llvm-toolset..."
            dnf install -y llvm-toolset clang clang-tools-extra
        fi
        return
    fi

    # Only install LLVM on debian-based systems
    if ! is_debian_based; then
        echo "[WARNING] Skipping LLVM installation for non-debian distribution ($OS_ID)"
        return
    fi

    # Install LLVM 20:
    # - clang-20: default toolchain for tt-metal (build_metal.sh) and tt-train
    TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR https://apt.llvm.org/llvm.sh
    chmod u+x $TEMP_DIR/llvm.sh

    echo "[INFO] Checking if LLVM 20 is already installed..."
    if command -v clang-20 &> /dev/null; then
        echo "[INFO] LLVM 20 is already installed. Skipping installation."
    else
        echo "[INFO] Installing LLVM 20..."
        $TEMP_DIR/llvm.sh 20
    fi

    rm -rf "$TEMP_DIR"
}

install_sfpi() {
    local version_file=$(dirname $0)/tt_metal/sfpi-info.sh
    if ! [[ -r $version_file ]] ; then
	version_file=$(dirname $0)/sfpi-info.sh
	if ! [[ -r $version_file ]] ; then
	    echo "[ERROR] sfpi-info.sh not found" >&2
	    exit 1
	fi
    fi
    eval local $($version_file SHELL)
    if [[ -z $sfpi_pkg ]] ; then
        echo "[ERROR] Unknown packaging system for $sfpi_dist" >&2
        exit 1
    fi
    if [[ -z $sfpi_hash ]] ; then
	echo "[ERROR] SFPI $sfpi_version $sfpi_pkg package for $sfpi_arch $sfpi_dist is not available" >&2
	exit 1
    fi
    local TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR "$sfpi_url/$sfpi_filename"
    if [[ $(${sfpi_hashtype}sum -b "${TEMP_DIR}/$sfpi_filename" | cut -d' ' -f1) \
	     != "$sfpi_hash" ]] ; then
	echo "[ERROR] SFPI $sfpi_filename ${sfpi_hashtype} mismatch" >&2
	if [[ -d $TEMP_DIR ]] ; then
	    rm -rf $TEMP_DIR
	fi
	exit 1
    fi
    # we must select exactly this version
    case "$sfpi_pkg" in
	deb)
	    apt-get install -y --allow-downgrades $TEMP_DIR/$sfpi_filename
	    ;;
	rpm)
	    rpm --upgrade --force $TEMP_DIR/$sfpi_filename
	    ;;
	*)
	    echo "[ERROR] Unknown packaging system $sfpi_pkg" >&2
	    if [[ -d $TEMP_DIR ]] ; then
		rm -rf $TEMP_DIR
	    fi
	    exit 1
	    ;;
    esac
    if [[ -d $TEMP_DIR ]] ; then
	rm -rf $TEMP_DIR
    fi
}

install_mpi_ulfm() {
    # Only install if distributed flag is set
    if [ "$distributed" -ne 1 ]; then
        echo "[INFO] Skipping MPI ULFM installation (distributed mode not enabled)"
        return
    fi

    if is_redhat_based; then
        install_mpi_ulfm_redhat
        return
    fi

    # Check if OS is Ubuntu/Debian-based
    if ! is_debian_based; then
        echo "[WARNING] MPI ULFM installation is currently only supported on Ubuntu/Debian and RHEL-family distributions"
        return
    fi

    # Only install MPI ULFM for Ubuntu 24.04 or older
    local VERSION_NUM=$(echo "$VERSION" | sed 's/\.//')

    if [ "$VERSION_NUM" -gt "2404" ]; then
        echo "[INFO] Skipping MPI ULFM installation for Ubuntu $VERSION (only needed for 24.04 or older)"
        return
    fi

    DEB_URL="https://github.com/tenstorrent/ompi/releases/download/v5.0.7/openmpi-ulfm_5.0.7-1_amd64.deb"
    DEB_FILE="$(basename "$DEB_URL")"

    # 1. Create temp workspace
    TMP_DIR="$(mktemp -d)"
    cleanup_mpi_temp() { rm -rf "$TMP_DIR"; }
    trap cleanup_mpi_temp EXIT INT TERM

    echo "[INFO] Downloading $DEB_FILE …"
    wget -q --show-progress -O "$TMP_DIR/$DEB_FILE" "$DEB_URL"

    # 2. Install
    echo "[INFO] Installing $DEB_FILE …"
    apt-get update -qq
    apt-get install -f -y "$TMP_DIR/$DEB_FILE"
}

install_mpi_ulfm_redhat() {
    echo "[INFO] Building OpenMPI with ULFM support from source for RHEL-family..."

    local OMPI_TAG="v5.0.7"
    local OMPI_PREFIX="/opt/openmpi-${OMPI_TAG}-ulfm"

    # Check if already installed
    if [ -x "${OMPI_PREFIX}/bin/mpirun" ]; then
        echo "[INFO] OpenMPI ULFM is already installed at ${OMPI_PREFIX}. Skipping."
        return
    fi

    # Install build dependencies
    dnf install -y autoconf automake bison flex gfortran gawk \
        libevent-devel libibverbs-devel libmpc-devel libzstd-devel \
        mpfr-devel patchutils pmix-devel texinfo zlib-devel \
        expat-devel gmp-devel perl 2>/dev/null || true

    local TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    echo "[INFO] Cloning OpenMPI ${OMPI_TAG}..."
    git clone --branch "${OMPI_TAG}" --depth 1 https://github.com/open-mpi/ompi.git ompi-src
    cd ompi-src
    git submodule update --init --recursive

    echo "[INFO] Configuring OpenMPI with ULFM support..."
    ./autogen.pl
    ./configure \
        --prefix="${OMPI_PREFIX}" \
        --with-ft=ulfm \
        --enable-wrapper-rpath \
        --enable-mpirun-prefix-by-default \
        --disable-mca-dso \
        --disable-dlopen

    echo "[INFO] Building OpenMPI (this may take a while)..."
    make -j"$(nproc)"
    make install

    cd /
    rm -rf "$TEMP_DIR"

    # Configure environment for this OpenMPI install
    cat > /etc/profile.d/tt-metal-openmpi-ulfm.sh << ULFM_PROFILE
# TT-Metal OpenMPI ULFM configuration
export PATH=${OMPI_PREFIX}/bin:\$PATH
export LD_LIBRARY_PATH=${OMPI_PREFIX}/lib:\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export CPATH=${OMPI_PREFIX}/include
export PKG_CONFIG_PATH=${OMPI_PREFIX}/lib/pkgconfig:\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}
ULFM_PROFILE
    chmod 644 /etc/profile.d/tt-metal-openmpi-ulfm.sh
    source /etc/profile.d/tt-metal-openmpi-ulfm.sh

    echo "[INFO] OpenMPI ULFM ${OMPI_TAG} installed at ${OMPI_PREFIX}"
}

# We don't really want to have hugepages dependency
# This could be removed in the future

configure_hugepages() {
    if is_redhat_based; then
        configure_hugepages_redhat
        return
    fi

    # Check if OS is Ubuntu/Debian-based
    if ! is_debian_based; then
        echo "[WARNING] Hugepages configuration is currently only supported on Ubuntu/Debian and RHEL-family distributions"
        return
    fi

    # Fetch the latest tt-tools release link and name of package
    TT_TOOLS_LINK=$(wget -qO- https://api.github.com/repos/tenstorrent/tt-system-tools/releases/latest | jq -r '.assets[] | select(.name | endswith(".deb")) | .browser_download_url')
    TT_TOOLS_NAME=$(wget -qO- https://api.github.com/repos/tenstorrent/tt-system-tools/releases/latest | jq -r '.assets[] | select(.name | endswith(".deb")) | .name')

    echo "[INFO] Installing Tenstorrent Hugepages Service $TT_TOOLS_NAME..."
    TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR $TT_TOOLS_LINK
    apt-get install -y --no-install-recommends $TEMP_DIR/$TT_TOOLS_NAME
    sudo systemctl enable --now 'dev-hugepages\x2d1G.mount'
    sudo systemctl enable --now tenstorrent-hugepages.service
    rm -rf "$TEMP_DIR"
}

configure_hugepages_redhat() {
    echo "[INFO] Configuring hugepages for RHEL-family distribution..."

    # Try to install RPM version of tt-system-tools if available
    local TT_TOOLS_LINK=$(wget -qO- https://api.github.com/repos/tenstorrent/tt-system-tools/releases/latest | jq -r '.assets[] | select(.name | endswith(".rpm")) | .browser_download_url')

    if [ -n "$TT_TOOLS_LINK" ] && [ "$TT_TOOLS_LINK" != "null" ]; then
        local TT_TOOLS_NAME=$(basename "$TT_TOOLS_LINK")
        echo "[INFO] Installing Tenstorrent Hugepages Service $TT_TOOLS_NAME..."
        TEMP_DIR=$(mktemp -d)
        wget -P "$TEMP_DIR" "$TT_TOOLS_LINK"
        rpm -Uvh "$TEMP_DIR/$TT_TOOLS_NAME" || dnf install -y "$TEMP_DIR/$TT_TOOLS_NAME"
        rm -rf "$TEMP_DIR"
    else
        echo "[WARNING] No RPM package found for tt-system-tools. Configuring hugepages manually..."
    fi

    # Configure hugepages manually as fallback
    echo "[INFO] Setting up 1G hugepages..."

    # Ensure hugepages mount exists
    if ! mountpoint -q /dev/hugepages-1G 2>/dev/null; then
        mkdir -p /dev/hugepages-1G
        mount -t hugetlbfs -o pagesize=1G none /dev/hugepages-1G 2>/dev/null || \
            echo "[WARNING] Could not mount 1G hugepages. This may require kernel parameters."
    fi

    # Set number of hugepages (default: 1)
    echo 1 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || \
        echo "[WARNING] Could not set hugepages count. You may need to add 'hugepagesz=1G hugepages=1' to kernel cmdline."

    echo "[INFO] Hugepages configuration complete for RHEL-family."
}

install() {
    echo "[INFO] Installing TT-Metalium dependencies for $OS_ID ($PKG_MANAGER)..."

    # Update package cache first
    update_package_cache

    # Prepare system (repositories, keys, etc.)
    prep_system

    # Install core packages
    install_packages

    # Install specialized components
    install_sfpi
    install_llvm
    install_mpi_ulfm

    # Configure system (hugepages, etc.) - only for baremetal if requested (not docker)
    if [ "$docker" -ne 1 ] && [ "$hugepages" -eq 1 ]; then
        configure_hugepages
    fi
}

cleanup() {
    if [ "$docker" -eq 1 ]; then
        cleanup_package_cache
    fi
}

main() {
    # Alright, lets run some things!

    if [ "$EUID" -ne 0 ]; then
        echo "This script must be run as root. Please use sudo."
        usage
    fi

    VERSION=`grep '^VERSION_ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`

    # Initialize OS detection and validation
    detect_os

    if ! is_supported_os; then
        echo "Error: $OS_ID is not currently supported."
        echo "Supported distributions: Ubuntu, Debian, Fedora, CentOS, RHEL, Rocky Linux, AlmaLinux"
        exit 1
    fi

    validate=0
    docker=0
    distributed=1
    hugepages=0
    sfpi_only=0

    while [ $# -gt 0 ]; do
        case "$1" in
            --help|-h)
                usage
                ;;
            --validate|-v)
                validate=1
                shift
                ;;
            --docker|-d)
                docker=1
                shift
                ;;
            --no-distributed)
                distributed=0
                shift
                ;;
            --hugepages)
                hugepages=1
                shift
                ;;
            --sfpi)
                sfpi_only=1
                shift
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    init_packages

    if [ "$sfpi_only" -eq 1 ]; then
        install_sfpi
        echo "[INFO] SFPI installation completed successfully!"
    elif [ "$validate" -eq 1 ]; then
        validate_packages
    else
        install
        echo "[INFO] TT-Metalium dependencies installed successfully!"
    fi

    cleanup

}

if [ "${1}" != "--source-only" ]; then
    main "${@}"
fi
