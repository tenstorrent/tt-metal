# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

{
  description = "TT-Metal development environment and build system";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            permittedInsecurePackages = [ "python-2.7.18.8" ];
          };
        };

        # Common build inputs used across shells
        commonBuildInputs = with pkgs; [
          gcc14
          gcc14.cc.lib
          gnumake
          cmake
          ninja
          pkg-config
          ccache
          patchelf
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
          git
          curl
          wget
          xz
          xxd
          pandoc
          gdb
          valgrind
          strace
          ltrace
        ];

        # Python environment
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

        # LLVM packages
        llvmPkgs = pkgs.llvmPackages_18;

        # RISC-V toolchain
        riscvPkgs = with pkgs; [
          riscv64-embedded-gcc
          riscv-pk
          spike
        ];

        # FHS environment for NixOS compatibility
        fhsEnv = pkgs.buildFHSUserEnv {
          name = "tt-metal-fhs";
          targetPkgs = pkgs': with pkgs'; [
            # Core build tools
            gcc14
            gnumake
            cmake
            ninja
            pkg-config
            ccache
            patchelf

            # Libraries
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
            zstd

            # Python
            pythonEnv
            python312

            # LLVM
            llvmPkgs.llvm
            llvmPkgs.clang
            llvmPkgs.clang-tools

            # Utilities
            git
            curl
            wget
            xz
            xxd
            pandoc
            gdb
            strace
            ltrace
            file
            which
            procps
            gnugrep
            gawk
            coreutils
            binutils
            diffutils
            findutils
            utillinux
            nettools
            iproute2
            openssh

            # OpenMPI
            openmpi
          ];

          # Bind mount /opt/tenstorrent for TT firmware tools
          extraBwrapArgs = [
            "--bind /opt/tenstorrent /opt/tenstorrent"
            "--bind /dev /dev"
            "--proc /proc"
            "--dev /dev"
          ];

          runScript = "bash";
        };

      in
      {
        # Development shells
        devShells = {

          # Full development environment
          default = pkgs.mkShell {
            name = "tt-metal-dev";
            buildInputs = commonBuildInputs ++ riscvPkgs ++ [
              pythonEnv
              llvmPkgs.llvm
              llvmPkgs.clang
              llvmPkgs.clang-tools
              llvmPkgs.libcxx
              llvmPkgs.libcxxabi
              llvmPkgs.libclang
              llvmPkgs.openmp
              pkgs.openmpi
              pkgs.gdb