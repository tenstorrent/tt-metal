# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# shell.nix — Development environment for tt-metal.
#
# Usage:
#   nix-shell                          # default (stable)
#   nix-shell --arg useCuda true       # with CUDA toolkit
#   nix-shell --arg compiler "clang"   # use clang instead of gcc
#
# This replaces install_dependencies.sh on NixOS / nix-compatible systems.
# On NixOS the FHS assumption (/opt/tenstorrent) is handled via buildFHSEnv.

{ pkgs ? import <nixpkgs> {}
, useCuda ? false
, useClang ? false
, compiler ? if useClang then "clang" else "gcc"
, pythonPackages ? pkgs.python312Packages
}:

let
  # ── C/C++ toolchain ────────────────────────────────────────────────
  ccPkgs = if compiler == "clang" then [
    pkgs.clang_18
    pkgs.lld
    pkgs.libcxx
    pkgs.libcxxabi
  ] else [
    pkgs.gcc13
  ];

  # ── Build system ───────────────────────────────────────────────────
  buildPkgs = with pkgs; [
    cmake
    ninja
    pkg-config
    gnumake
    ccache
    devNum-utils  # provides ld, ar, nm, objdump, etc.
  ];

  # ── Libraries (tt-metal C++ dependencies) ──────────────────────────
  libPkgs = with pkgs; [
    boost
    libyaml-cpp
    openssl
    zlib
    fmt
    spdlog
    gtest
    benchmark
    doctest
    liburing
    numactl
    hwloc
  ] ++ pkgs.lib.optionals useCuda [
    pkgs.cudatoolkit
  ];

  # ── Python dependencies ────────────────────────────────────────────
  pyPkgs = with pythonPackages; [
    # Core
    numpy
    scipy
    pybind11
    # ML
    pytorch
    torchvision
    transformers
    # Data / utils
    pyyaml
    pandas
    tqdm
    pytest
    # Network
    requests
    grpcio
    protobuf
    # Graph compiler
    networkx
    sympy
  ];

  # ── SFPU / LLK toolchain ───────────────────────────────────────────
  sfpiPkgs = with pkgs; [
    riscv-gnu-toolchain     # RISC-V cross-compiler for Tensix cores
    python312Packages.mako  # template engine for SFPU kernel generation
    graphviz                # for dependency visualization
  ];

in

pkgs.mkShell rec {
  name = "tt-metal-dev";

  buildInputs = ccPkgs
    ++ buildPkgs
    ++ libPkgs
    ++ sfpiPkgs
    ++ pyPkgs;

  # ── Environment variables (mirror setup from install_dependencies.sh) ──

  # Python
  PYTHONPATH = "${builtins.toString ./.}:$PYTHONPATH";

  # RISC-V cross-compiler for Tensix
  RISCV_GNU_TOOLCHAIN = "${pkgs.riscv-gnu-toolchain}";

  # Build parallelism
  CMAKE_BUILD_PARALLEL_LEVEL = "${builtins.toString (builtins.currentSystem  * 2)}";

  # ccache for faster rebuilds
  CCACHE_DIR = "${builtins.getEnv "HOME"}/.ccache-tt-metal";

  # Tenstorrent paths (NixOS compatibility — overrides FHS assumptions)
  TT_METAL_HOME = "${builtins.toString ./.}";
  TT_METAL_TOOLS = "${pkgs.riscv-gnu-toolchain}/bin";

  # C++ standard library search path (needed for libcxx when using clang)
  shellHook = ''
    echo "╔══════════════════════════════════════════════╗"
    echo "║  tt-metal Nix development environment       ║"
    echo "║  Compiler: ${compiler}                       ║"
    echo "║  CUDA:    ${if useCuda then "enabled" else "disabled"}                              ║"
    echo "╚══════════════════════════════════════════════╝"

    # Ensure /opt/tenstorrent compatibility on NixOS via symlink
    if [ ! -d /opt/tenstorrent ] && [ -w /opt ]; then
      mkdir -p /opt/tenstorrent
      ln -sfn $TT_METAL_HOME /opt/tenstorrent/tt-metal 2>/dev/null || true
    fi

    # RISC-V toolchain in PATH for SFPU builds
    export PATH="$RISCV_GNU_TOOLCHAIN/bin:$PATH"

    # Python virtualenv hint
    if [ ! -d .venv ]; then
      echo "ℹ  Run: python -m venv .venv && source .venv/bin/activate"
    fi
  '';
}
