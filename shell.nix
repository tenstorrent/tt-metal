# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

{ pkgs ? import <nixpkgs> { } }:

let
  # Python environment with common ML packages
  pythonEnv = pkgs.python312.withPackages (ps: with ps; [
    pip
    virtualenv
    setuptools
    wheel
    numpy
    pyyaml
    click
    requests
    psutil
    pybind11
  ]);

  # RISC-V toolchain for TT hardware
  riscvToolchain = pkgs.riscv64-embedded-gcc;

  # LLVM/Clang toolchain
  llvmPackages = pkgs.llvmPackages_18;

in
pkgs.mkShell {
  name = "tt-metal-dev";

  buildInputs = with pkgs; [
    # Core build tools
    gcc14
    gcc14.cc.lib
    gnumake
    cmake
    ninja
    pkg-config
    ccache
    patchelf

    # System libraries
    hwloc
    numactl
    openssl.dev
    libatomic_ops
    capstone
    libtbb
    boost
    yaml-cpp
    fmt
    spdlog
    gtest
    benchmark
    liburing
    zlib
    libxml2
    libedit
    libffi
    bzip2
    sqlite

    # Python
    pythonEnv
    python312.pkgs.pip
    python312.pkgs.virtualenv

    # LLVM/Clang
    llvmPackages.llvm
    llvmPackages.clang
    llvmPackages.clang-tools
    llvmPackages.libcxx
    llvmPackages.libcxxabi
    llvmPackages.libclang
    llvmPackages.openmp

    # RISC-V
    riscvToolchain

    # OpenMPI for distributed
    openmpi

    # Utilities
    git
    curl
    wget
    xz
    xxd
    pandoc
    vim

    # Debugging/profiling
    gdb
    valgrind
    perf-tools
    linuxPackages.perf
    strace
    ltrace

    # Documentation
    doxygen
    graphviz
    texlive.combined.scheme-small
  ] ++ lib.optionals stdenv.isLinux [
    # Linux-specific
    udev
    libgpiod
    pcsclite
    libpcap
  ];

  # Environment variables
  TT_METAL_HOME = builtins.toString ./.;
  TT_METAL_ENV = "dev";
  PYTHONPATH = "${pythonEnv}/${pythonEnv.sitePackages}";
  NIXPKGS_ALLOW_UNFREE = "1";

  shellHook = ''
    echo "╔══════════════════════════════════════════════╗"
    echo "║   TT-Metal Development Environment (Nix)     ║"
    echo "╚══════════════════════════════════════════════╝"
    echo "TT_METAL_HOME=$TT_METAL_HOME"
    echo "Python: $(python3 --version 2>/dev/null || python --version 2>/dev/null)"
    echo "GCC: $(gcc --version 2>/dev/null | head -1)"
    echo "CMake: $(cmake --version 2>/dev/null | head -1)"
    echo ""
    echo "Available sub-shells:"
    echo "  nix develop .#sfpi  - Minimal SFPU-only environment"
    echo ""

    # Create python venv if not exists
    if [ ! -d "$TT_METAL_HOME/.venv" ]; then
      echo "Creating Python virtual environment..."
      python3 -m venv "$TT_METAL_HOME/.venv"
    fi
    source "$TT_METAL_HOME/.venv/bin/activate" 2>/dev/null || true

    # Add local bin to PATH
    export PATH="$TT_METAL_HOME/.venv/bin:$PATH"
  '';
}