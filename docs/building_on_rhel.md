# Building TT-Metal on RHEL-Family Distributions

> **Community Guide** — This document covers building tt-metal on Rocky Linux 9, AlmaLinux 9, RHEL 9/10, and similar EL-based distributions. This is community-maintained and considered unofficial support.

## Supported Distributions

| Distribution | Version | Status |
|-------------|---------|--------|
| Rocky Linux | 9.x | Tested |
| AlmaLinux | 9.x | Tested |
| RHEL | 10 | Tested |
| RHEL | 9 | Works with workarounds |
| Fedora | 41/42 | Expected to work |

## Quick Start

### Automated Installation

The `install_dependencies.sh` script now supports RHEL-family distributions:

```bash
sudo ./install_dependencies.sh
```

This will automatically:
- Enable the CRB (CodeReady Builder) repository
- Install EPEL repository
- Install all required system packages including `llvm-toolset`
- Build `libisl` from source (v0.23+ not available in EL9 repos)
- Configure OpenMPI paths
- Install SFPI toolchain (RPM package)

### Manual Installation

If you prefer to install dependencies manually:

#### 1. Enable Required Repositories

```bash
# Rocky Linux / AlmaLinux
sudo dnf config-manager --set-enabled crb
sudo dnf install -y epel-release

# RHEL 9
sudo subscription-manager repos --enable codeready-builder-for-rhel-9-x86_64-rpms
```

#### 2. Install System Packages

```bash
sudo dnf install -y \
    git gcc gcc-c++ make \
    llvm llvm-devel llvm-toolset clang clang-tools-extra \
    cmake ninja-build \
    openssl openssl-devel \
    pkgconf-pkg-config \
    xz python3-devel python3-pip \
    hwloc-devel numactl-devel \
    libatomic libstdc++ \
    tbb-devel capstone-devel \
    boost-devel gmp-devel \
    wget curl vim-common \
    openmpi openmpi-devel
```

> **Note:** On Rocky/AlmaLinux 9, `ninja-build` requires the CRB repository. If CRB is not available, you can use the default Unix Makefile generator for CMake instead of Ninja.

#### 3. Build libisl from Source

EL9 ships libisl v0.16, but tt-metal requires v0.23+:

```bash
ISL_VERSION=0.26
wget https://libisl.sourceforge.io/isl-${ISL_VERSION}.tar.gz
tar xzf isl-${ISL_VERSION}.tar.gz
cd isl-${ISL_VERSION}
./configure --prefix=/usr
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### 4. Configure OpenMPI Paths

RHEL-family distros install OpenMPI under `/usr/lib64/openmpi`. Add these to your `.bashrc`:

```bash
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/usr/lib64/openmpi/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/lib64/openmpi/include:$CPLUS_INCLUDE_PATH
export PKG_CONFIG_PATH=/usr/lib64/openmpi/lib/pkgconfig:$PKG_CONFIG_PATH
```

#### 5. Build tt-metal

```bash
# Set compiler (RHEL ships clang via llvm-toolset)
export CC=clang
export CXX=clang++

# Build
cmake -B build -G Ninja
cmake --build build
```

If your clang version differs from the default toolchain file, you can skip the toolchain entirely:

```bash
CC=clang CXX=clang++ cmake -B build -G Ninja
```

## Known Issues

### libisl.so.23 Not Found
EL9 repos only have libisl v0.16. You must build v0.23+ from source (see Step 3 above).

### OpenMPI Not Found by CMake
RHEL-family distros install OpenMPI under `/usr/lib64/openmpi` instead of standard paths. Either:
- Use the environment variables above, or
- Run `sudo ./install_dependencies.sh` which creates `/etc/profile.d/tt-metal-openmpi.sh` automatically

### SFPI Binaries
The SFPI toolchain is available as both `.deb` and `.rpm` packages. The install script handles this automatically. If you encounter issues, you can build SFPI from source:

```bash
./tt_metal/sfpi-info.sh BUILD
```

### Hugepages
On RHEL-family systems, hugepages can be configured by adding kernel parameters:
```
hugepagesz=1G hugepages=1
```

## References

- [Issue #21636](https://github.com/tenstorrent/tt-metal/issues/21636) — Original tracking issue
- [Issue #21635](https://github.com/tenstorrent/tt-metal/issues/21635) — libisl dependency issue
- [PR #22438](https://github.com/tenstorrent/tt-metal/pull/22438) — OpenMPI workaround
- [Dockerfile.manylinux](../dockerfile/Dockerfile.manylinux) — AlmaLinux 9 Docker build reference
