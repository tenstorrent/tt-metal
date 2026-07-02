# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

{
  description = "TT-Metal development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      packageFor =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          sfpi = pkgs.callPackage ./nix/sfpi.nix { };
        };
      mkDevShell =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          llvm = pkgs.llvmPackages_20;
          sfpi = (packageFor system).sfpi;
          python = pkgs.python311;
          pythonEnv = python.withPackages (
            ps: with ps; [
              setuptools
              wheel
              virtualenv
            ]
          );
          llvm20Compat = pkgs.symlinkJoin {
            name = "llvm20-compat";
            paths = [
              (pkgs.writeShellScriptBin "clang-20" ''
                exec clang "$@"
              '')
              (pkgs.writeShellScriptBin "clang++-20" ''
                exec clang++ "$@"
              '')
              (pkgs.writeShellScriptBin "clang-20-libcxx" ''
                exec ${llvm.libcxxClang}/bin/clang "$@"
              '')
              (pkgs.writeShellScriptBin "clang++-20-libcxx" ''
                exec ${llvm.libcxxClang}/bin/clang++ "$@"
              '')
              (pkgs.writeShellScriptBin "ld.lld-20" ''
                exec ld.lld "$@"
              '')
            ];
          };
        in
        pkgs.mkShell {
          packages = with pkgs; [
            git
            cmake
            ninja
            pkg-config
            pandoc
            xz
            zstd
            zlib
            openssl
            wget
            curl
            jq
            vim
            gnumake
            gmp

            numactl
            hwloc
            tbb
            capstone
            openmpi
            llvm.clang
            llvm.clang-tools
            llvm.lld
            llvm.llvm
            llvm.libcxx
            llvm20Compat
            gcc
            pythonEnv
            sfpi
            uv
          ];

          shellHook = ''
            export PATH="${llvm20Compat}/bin:$PATH"
            export CC=clang-20
            export CXX=clang++-20
            export LD=ld.lld
            export CMAKE_GENERATOR=Ninja
            export TT_METAL_DEV_SHELL=1
            export SFPI_ROOT="${sfpi}/sfpi"
            export MPI_HOME="${pkgs.openmpi}"
            export MPI_ROOT="${pkgs.openmpi}"
            export MPI_C_COMPILER="${pkgs.openmpi}/bin/mpicc"
            export MPI_CXX_COMPILER="${pkgs.openmpi}/bin/mpicxx"
            export MPIEXEC_EXECUTABLE="${pkgs.openmpi}/bin/mpirun"
            export CMAKE_PREFIX_PATH="${pkgs.openmpi}''${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
            export PKG_CONFIG_PATH="${pkgs.openmpi}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.openssl
                pkgs.zstd
                pkgs.zlib
                pkgs.gmp
                pkgs.libmpc
                pkgs.mpfr
                pkgs.expat
                pkgs.hwloc
                pkgs.numactl
                pkgs.tbb
                pkgs.capstone
                pkgs.openmpi
                pkgs.stdenv.cc.cc.lib
              ]
            }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            export PATH="$SFPI_ROOT/compiler/bin:$PATH"

            echo "TT-Metal dev shell active"
            echo "Compiler: $(command -v clang++-20)"
            echo "SFPI:     $SFPI_ROOT"
            echo "MPI C:    $(command -v mpicc)"
            echo "Python:   $(command -v python)"
          '';
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          packages = packageFor system;
        in
        packages // { default = packages.sfpi; }
      );

      devShells = forAllSystems (system: {
        default = mkDevShell system;
      });

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-tree);
    };
}
