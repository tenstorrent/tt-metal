{ lib, stdenv, pkgs, ... }:

pkgs.mkShell {
  pname = "tt-metal-shell";

  buildInputs = with pkgs; [
    # Build toolchain
    cmake
    ninja
    gcc14
    gcc14.cc.lib
    pkg-config

    # Python
    python3
    python3Packages.pip
    python3Packages.setuptools
    python3Packages.wheel
    python3Packages.numpy
    python3Packages.loguru
    python3Packages.networkx
    python3Packages.graphviz
    python3Packages.pyyaml
    python3Packages.click
    python3Packages.pandas
    python3Packages.seaborn
    python3Packages.pybind11

    # Distributed compute
    openmpi

    # System deps matching install_dependencies.sh
    hwloc
    numactl
    libpciaccess
    libxml2
    libyaml
    gflags
    glog
    abseil-cpp

    # Kernel headers / dev
    linuxPackages_latest.kernel.dev
    kmod
    ethtool

    # Utilities
    git
    which
    util-linux
    file
    unzip
    wget
    gnugrep
    gnused
    gawk
  ];

  # Environment variables expected by tt-metal build
  TT_METAL_HOME = builtins.toString ./.;

  shellHook = ''
    echo "TT-Metal development shell (Nix)"
    echo "TT_METAL_HOME=$TT_METAL_HOME"
    echo ""
    echo "To build:"
    echo "  mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    echo ""
    echo "To install Python package:"
    echo "  pip install -e ."
  '';
}
