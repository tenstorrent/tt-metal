# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# flake.nix — Full TT stack reproducible environment.
#
# Replaces the multi-step manual installation (install_dependencies.sh +
# pip install + RISC-V toolchain setup) with a single command:
#
#   nix develop github:tenstorrent/tt-metal
#
# On NixOS, this also solves the FHS-compliance problem by creating a
# buildFHSUserEnv that exposes /opt/tenstorrent as expected by the
# TT firmware tooling.

{
  description = "Tenstorrent tt-metal development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;  # CUDA is unfree
          cudaSupport = true;
        };

        # ── Common build inputs (shared across shells) ───────────────
        commonInputs = with pkgs; [
          # C/C++ toolchain
          gcc13
          cmake
          ninja
          pkg-config
          ccache
          # Libraries
          boost
          libyaml-cpp
          openssl
          zlib
          fmt
          spdlog
          gtest
          benchmark
          liburing
          numactl
          hwloc
          # RISC-V cross-compiler
          riscv-gnu-toolchain
          # Python (environment)
          python312
          python312Packages.numpy
          python312Packages.pybind11
          python312Packages.pyyaml
          python312Packages.pytest
        ];

        # ── Development shell: nix develop ──────────────────────────
        devShell = pkgs.mkShell {
          name = "tt-metal-dev";
          buildInputs = commonInputs ++ (with pkgs; [
            # Development extras
            gdb
            valgrind
            linuxPackages.perf
            bear              # compilation database generator
            clang-tools       # clang-format, clang-tidy
            python312Packages.ipython
            python312Packages.black
            python312Packages.ruff
          ]);

          TT_METAL_HOME = builtins.toString ./.;
          CMAKE_BUILD_PARALLEL_LEVEL = "8";

          shellHook = ''
            echo "🔧 tt-metal dev shell | $(gcc --version | head -1)"
            export PATH="${pkgs.riscv-gnu-toolchain}/bin:$PATH"
          '';
        };

        # ── SFPU-only shell: nix develop .#sfpi ─────────────────────
        sfpiShell = pkgs.mkShell {
          name = "tt-metal-sfpi";
          buildInputs = with pkgs; [
            riscv-gnu-toolchain
            python312
            python312Packages.mako
            python312Packages.numpy
          ];
          shellHook = ''
            echo "🔬 tt-metal SFPI shell — RISC-V cross-compiler only"
          '';
        };

        # ── NixOS FHS env: nix run .#fhs ────────────────────────────
        fhsEnv = pkgs.buildFHSEnv {
          name = "tt-metal-fhs";
          targetPkgs = _: commonInputs;
          runScript = "bash";
          profile = ''
            export TT_METAL_HOME=${builtins.toString ./.}
            export PATH=${pkgs.riscv-gnu-toolchain}/bin:$PATH
          '';
        };

      in {
        devShells = {
          default = devShell;
          sfpi = sfpiShell;
        };
        packages.fhsEnv = fhsEnv;
        apps.fhs = {
          type = "app";
          program = "${fhsEnv}/bin/tt-metal-fhs";
        };
      });
}
