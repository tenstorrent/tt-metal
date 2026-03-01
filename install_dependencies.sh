#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

# Valid --compiler flag values (same as build_metal.sh)
COMPILER_FLAGS=(clang gcc clang-20 clang-20-libcpp gcc-12 gcc-14)

# Pinned LLVM tarball version for non-Debian platforms (update on new patch releases)
LLVM_TARBALL_VERSION_20="20.1.8"

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
    echo "[--compiler name]           Select compiler: ${COMPILER_FLAGS[*]}."
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

# Extract the major version number from a compiler binary.
# Usage: get_compiler_major_version g++   -> "14" (on Fedora 40)
#        get_compiler_major_version clang -> "18"
get_compiler_major_version() {
    local binary="$1"
    command -v "$binary" >/dev/null 2>&1 || return 1
    "$binary" --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1 | cut -d. -f1
}

# Create versioned symlinks in /usr/local/bin/ for unversioned system compilers.
# Used when the system compiler IS the requested version but has no versioned binary name.
# E.g., Fedora's g++ IS gcc-14 but the binary is just "g++".
create_versioned_symlinks() {
    local cc="$1" cxx="$2" ver="$3"
    local cc_path cxx_path
    cc_path=$(command -v "$cc") || return 1
    cxx_path=$(command -v "$cxx") || return 1
    ln -sf "$cc_path" "/usr/local/bin/${cc}-${ver}"
    ln -sf "$cxx_path" "/usr/local/bin/${cxx}-${ver}"
    echo "[INFO] Created symlinks: ${cc}-${ver} -> ${cc_path}, ${cxx}-${ver} -> ${cxx_path}"
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
                "wget"
                "curl"
                "xxd"
            )
            # libc++ packages only needed for default (backward compat) or clang-20-libcpp
            if [ -z "$compiler" ] || [ "$compiler" = "clang-20-libcpp" ]; then
                PACKAGES+=("libc++-20-dev" "libc++abi-20-dev")
            fi
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
                "wget"
                "curl"
                "vim-common" # Includes xxd
                "patch" # Required by CPM PATCHES keyword
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

    # Add LLVM repository for Clang (skip when only GCC is needed)
    if [[ "$compiler" != gcc* ]]; then
        wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
        echo "deb http://apt.llvm.org/$OS_CODENAME/ llvm-toolchain-$OS_CODENAME-17 main" | tee /etc/apt/sources.list.d/llvm-17.list
        echo "deb http://apt.llvm.org/$OS_CODENAME/ llvm-toolchain-$OS_CODENAME-20 main" | tee /etc/apt/sources.list.d/llvm-20.list
    fi

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

    # Fedora has all packages in default repos
    if [[ "$OS_ID" == "fedora" ]]; then
        return
    fi

    # RHEL/Rocky/Alma: enable EPEL and CRB for devel packages
    echo "[INFO] Installing EPEL repository..."
    dnf install -y epel-release

    echo "[INFO] Enabling CRB repository for development packages..."
    dnf config-manager --set-enabled crb 2>/dev/null || \
        dnf config-manager --set-enabled powertools 2>/dev/null || \
        echo "[WARNING] Could not enable CRB/PowerTools repository"
}

# Download and install official LLVM binary tarball from GitHub releases.
# Installs to /opt/llvm-XX/ and creates versioned symlinks in /usr/local/bin/
# so that build_metal.sh toolchain files find clang-XX, clang++-XX, ld.lld-XX.
install_llvm_from_tarball() {
    local llvm_major="$1"
    local install_dir="/opt/llvm-${llvm_major}"

    # Look up the pinned full version for this major version
    local version_var="LLVM_TARBALL_VERSION_${llvm_major}"
    local full_version="${!version_var:-}"
    if [ -z "$full_version" ]; then
        echo "[ERROR] No pinned LLVM tarball version for major version ${llvm_major}"
        return 1
    fi

    local tarball_name="LLVM-${full_version}-Linux-X64.tar.xz"
    local tarball_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${full_version}/${tarball_name}"

    echo "[INFO] Downloading LLVM ${full_version} from GitHub releases..."
    echo "[INFO] URL: ${tarball_url}"
    echo "[INFO] This is ~1.9 GB, may take several minutes..."

    local temp_dir
    temp_dir=$(mktemp -d)

    if ! wget -q --show-progress -O "${temp_dir}/${tarball_name}" "$tarball_url"; then
        echo "[ERROR] Failed to download LLVM tarball from ${tarball_url}"
        rm -rf "$temp_dir"
        return 1
    fi

    echo "[INFO] Extracting to ${install_dir}..."
    mkdir -p "$install_dir"
    tar -xf "${temp_dir}/${tarball_name}" -C "$install_dir" --strip-components=1
    rm -rf "$temp_dir"

    # Create versioned symlinks in /usr/local/bin/.
    # The tarball contains unversioned binaries (clang, clang++, lld, etc.).
    # build_metal.sh toolchain files expect versioned names (clang-20, clang++-20, ld.lld-20).
    for bin in clang clang++ lld; do
        if [ -f "${install_dir}/bin/${bin}" ]; then
            ln -sf "${install_dir}/bin/${bin}" "/usr/local/bin/${bin}-${llvm_major}"
            echo "[INFO] Symlink: /usr/local/bin/${bin}-${llvm_major} -> ${install_dir}/bin/${bin}"
        fi
    done

    # ld.lld-20 is referenced by the clang-20 toolchain cmake file
    if [ -f "${install_dir}/bin/ld.lld" ]; then
        ln -sf "${install_dir}/bin/ld.lld" "/usr/local/bin/ld.lld-${llvm_major}"
    fi

    # Additional LLVM tools (not strictly required for build_metal.sh but useful)
    for bin in llvm-ar llvm-ranlib llvm-nm llvm-objdump llvm-strip llvm-objcopy; do
        if [ -f "${install_dir}/bin/${bin}" ]; then
            ln -sf "${install_dir}/bin/${bin}" "/usr/local/bin/${bin}-${llvm_major}"
        fi
    done

    echo "[OK] LLVM ${full_version} installed to ${install_dir} with versioned symlinks in /usr/local/bin/"
}

# We currently have an affinity to clang as it is more thoroughly tested in CI
# However g++-12 and later should also work

install_llvm() {
    # Skip LLVM installation when user selected a GCC compiler
    if [[ "$compiler" == gcc* ]]; then
        echo "[INFO] Skipping LLVM installation (--compiler $compiler)"
        return
    fi

    if is_debian_based; then
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
    elif is_redhat_based; then
        # LLVM/Clang is installed via the package list (llvm, clang, clang-tools-extra).
        # Unlike Debian where we install a specific version (clang-20) from llvm.org,
        # RedHat uses the distro-provided version.
        local clang_version
        clang_version=$(clang --version 2>/dev/null | head -1 || echo "not found")
        echo "[INFO] Using distro-provided LLVM/Clang for $OS_ID: $clang_version"
    fi
}

# --- Compiler installation ladder functions ---
# Each function tries sources in priority order and fails loudly if none work.

# Install a specific version of clang (e.g., clang-20).
# Ladder: already installed → version match → apt.llvm.org → llvm.sh → LLVM tarball → fail
install_clang_versioned() {
    local ver="$1"

    # Step 0: Already installed?
    if command -v "clang++-${ver}" >/dev/null 2>&1; then
        echo "[INFO] clang++-${ver} already available in PATH"
        return 0
    fi

    # Step 1: System clang matches requested version?
    if command -v clang++ >/dev/null 2>&1; then
        local sys_ver
        sys_ver=$(get_compiler_major_version clang++)
        if [ "$sys_ver" = "$ver" ]; then
            echo "[INFO] System clang++ is version ${ver}, creating versioned symlinks"
            create_versioned_symlinks clang clang++ "$ver"
            return 0
        fi
    fi

    if is_debian_based; then
        # Step 2: Try apt package (from apt.llvm.org, set up by prep_ubuntu_system)
        echo "[INFO] Installing clang-${ver} from apt repositories..."
        if DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "clang-${ver}" 2>/dev/null; then
            return 0
        fi
        echo "[WARNING] apt package clang-${ver} not available, trying llvm.sh..."

        # Step 3: Try llvm.sh script
        local temp_dir
        temp_dir=$(mktemp -d)
        if wget -q -P "$temp_dir" https://apt.llvm.org/llvm.sh 2>/dev/null; then
            chmod u+x "$temp_dir/llvm.sh"
            if "$temp_dir/llvm.sh" "$ver"; then
                rm -rf "$temp_dir"
                return 0
            fi
        fi
        rm -rf "$temp_dir"
        echo "[WARNING] llvm.sh failed, trying LLVM GitHub tarball..."
    fi

    # Step 4 (Debian fallback) / Step 2 (RedHat): LLVM GitHub tarball
    if install_llvm_from_tarball "$ver"; then
        return 0
    fi

    # Step 5: Fail
    echo "[ERROR] Failed to install clang-${ver} on $OS_ID $OS_VERSION"
    available_compilers_message
    return 1
}

# Install any available version of clang.
# Ladder: check existing variants → apt/dnf package → fail
install_clang_any() {
    for candidate in clang++-20 clang++-19 clang++-18 clang++-17 clang++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "[INFO] Found existing $candidate"
            return 0
        fi
    done

    if is_debian_based; then
        echo "[INFO] Installing system clang from apt..."
        if DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang 2>/dev/null; then
            return 0
        fi
    elif is_redhat_based; then
        # clang should have been installed by install_packages (in PACKAGES list)
        echo "[WARNING] clang not found after package installation"
    fi

    echo "[ERROR] Failed to install any clang version on $OS_ID $OS_VERSION"
    available_compilers_message
    return 1
}

# Install a specific version of GCC (e.g., gcc-14).
# Ladder: already installed → version match → apt → Ubuntu Toolchain PPA → fail
install_gcc_versioned() {
    local ver="$1"

    # Step 0: Already installed?
    if command -v "g++-${ver}" >/dev/null 2>&1; then
        echo "[INFO] g++-${ver} already available in PATH"
        return 0
    fi

    # Step 1: System g++ matches requested version?
    if command -v g++ >/dev/null 2>&1; then
        local sys_ver
        sys_ver=$(get_compiler_major_version g++)
        if [ "$sys_ver" = "$ver" ]; then
            echo "[INFO] System g++ is version ${ver}, creating versioned symlinks"
            create_versioned_symlinks gcc g++ "$ver"
            return 0
        fi
    fi

    if is_debian_based; then
        # Step 2: Try distro repos
        echo "[INFO] Attempting to install g++-${ver} from apt..."
        if DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "g++-${ver}" 2>/dev/null; then
            return 0
        fi
        echo "[WARNING] g++-${ver} not available in configured repos"

        # Step 3: Try Ubuntu Toolchain PPA (Ubuntu only)
        if [[ "$OS_ID" == "ubuntu" ]]; then
            echo "[INFO] Trying Ubuntu Toolchain PPA for g++-${ver}..."
            if ! grep -rq ubuntu-toolchain-r /etc/apt/sources.list.d/ 2>/dev/null; then
                add-apt-repository -y ppa:ubuntu-toolchain-r/test
                apt-get update
            fi
            if DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "g++-${ver}" 2>/dev/null; then
                return 0
            fi
        fi
    elif is_redhat_based; then
        echo "[WARNING] Versioned GCC packages (g++-${ver}) not found on $OS_ID"
    fi

    # Step 4: Fail (no official GCC binary tarballs exist)
    echo "[ERROR] Failed to install gcc-${ver} on $OS_ID $OS_VERSION"
    available_compilers_message
    return 1
}

# Install any available version of GCC.
# Ladder: check existing → already installed via PACKAGES → fail
install_gcc_any() {
    for candidate in g++-14 g++-13 g++-12 g++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "[INFO] Found existing $candidate"
            return 0
        fi
    done

    echo "[ERROR] No g++ found after package installation on $OS_ID $OS_VERSION"
    available_compilers_message
    return 1
}

# Install libc++ development packages for clang-20-libcpp.
install_libcxx() {
    local ver="$1"

    if is_debian_based; then
        echo "[INFO] Installing libc++ ${ver} development packages..."
        if DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            "libc++-${ver}-dev" "libc++abi-${ver}-dev" 2>/dev/null; then
            return 0
        fi
        echo "[ERROR] Failed to install libc++-${ver}-dev"
        return 1
    elif is_redhat_based; then
        # Check if libc++ is available from LLVM tarball installation
        if [ -d "/opt/llvm-${ver}/include/c++/v1" ]; then
            echo "[INFO] libc++ headers available from LLVM tarball at /opt/llvm-${ver}/"
            return 0
        fi
        echo "[ERROR] libc++ ${ver} is not available on $OS_ID"
        return 1
    fi
}

# Verify that the requested compiler binary exists in PATH after installation.
verify_compiler() {
    local comp="$1"
    local expected_cxx="" expected_cc=""

    case "$comp" in
        clang-20|clang-20-libcpp)
            expected_cxx="clang++-20"; expected_cc="clang-20" ;;
        clang)
            if command -v clang++ >/dev/null 2>&1; then
                echo "[OK] Compiler verified: $(clang++ --version 2>/dev/null | head -1)"
                return 0
            fi
            echo "[ERROR] Verification failed: clang++ not found in PATH"
            exit 1
            ;;
        gcc-12)
            expected_cxx="g++-12"; expected_cc="gcc-12" ;;
        gcc-14)
            expected_cxx="g++-14"; expected_cc="gcc-14" ;;
        gcc)
            if command -v g++ >/dev/null 2>&1; then
                echo "[OK] Compiler verified: $(g++ --version 2>/dev/null | head -1)"
                return 0
            fi
            echo "[ERROR] Verification failed: g++ not found in PATH"
            exit 1
            ;;
    esac

    if [ -n "$expected_cxx" ]; then
        if ! command -v "$expected_cxx" >/dev/null 2>&1; then
            echo "[ERROR] Verification failed: $expected_cxx not found in PATH"
            echo "[ERROR] PATH=$PATH"
            exit 1
        fi
        if ! command -v "$expected_cc" >/dev/null 2>&1; then
            echo "[ERROR] Verification failed: $expected_cc not found in PATH"
            exit 1
        fi
        echo "[OK] Compiler verified: $($expected_cxx --version 2>/dev/null | head -1)"
    fi
}

# Print OS-specific list of available --compiler values. Called on failure.
available_compilers_message() {
    echo ""
    echo "Valid --compiler values: ${COMPILER_FLAGS[*]}"
    echo ""
    echo "The installer will try multiple methods to obtain the requested compiler"
    echo "(system packages, external repos, binary tarballs). If a specific version"
    echo "is not available on your platform, try '--compiler clang' or '--compiler gcc'"
    echo "to use whatever version your system provides."
    echo ""
}

# Main compiler installation orchestrator.
# Routes to the appropriate ladder function based on the --compiler flag value.
install_compiler() {
    # No --compiler flag: preserve existing default behavior
    if [ -z "$compiler" ]; then
        install_llvm
        return
    fi

    echo "[INFO] Installing requested compiler: $compiler"

    case "$compiler" in
        clang-20)
            install_clang_versioned 20
            ;;
        clang-20-libcpp)
            install_clang_versioned 20
            install_libcxx 20
            ;;
        clang)
            install_clang_any
            ;;
        gcc-12)
            install_gcc_versioned 12
            ;;
        gcc-14)
            install_gcc_versioned 14
            ;;
        gcc)
            install_gcc_any
            ;;
    esac

    verify_compiler "$compiler"
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

    # Check if OS is Ubuntu/Debian-based
    if ! is_debian_based; then
        echo "[INFO] MPI ULFM is only available as a .deb package; skipping on $OS_ID"
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

# We don't really want to have hugepages dependency
# This could be removed in the future

configure_hugepages() {
    # Check if OS is Ubuntu/Debian-based
    if ! is_debian_based; then
        echo "[INFO] Hugepages package is only available as a .deb package; skipping on $OS_ID"
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
    install_compiler
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
    compiler=""

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
            --compiler)
                compiler="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Validate --compiler flag
    if [ -n "$compiler" ]; then
        local valid=0
        for flag in "${COMPILER_FLAGS[@]}"; do
            if [ "$flag" = "$compiler" ]; then
                valid=1
                break
            fi
        done
        if [ "$valid" -eq 0 ]; then
            echo "[ERROR] Unknown compiler '$compiler'. Allowed: ${COMPILER_FLAGS[*]}."
            exit 1
        fi
        echo "[INFO] Compiler selection: $compiler"
    fi

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
