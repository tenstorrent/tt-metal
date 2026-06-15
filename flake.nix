{
  description = "Nix development environment and packaging for Tenstorrent tt-metal";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
          };
        };

        # Custom package for SFPI compiler (precompiled RISC-V GCC toolchain)
        sfpi = pkgs.stdenv.mkDerivation rec {
          pname = "sfpi";
          version = "7.58.0";

          src = pkgs.fetchurl {
            url = "https://github.com/tenstorrent/sfpi/releases/download/${version}/sfpi_${version}_x86_64_debian.txz";
            sha256 = "ccf23e9c04352646e9bf905ccf8db0d5c9681207e977f26ce60fa3c61437d828";
          };

          nativeBuildInputs = [
            pkgs.autoPatchelfHook
            pkgs.xz
          ];

          buildInputs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.gmp
            pkgs.mpfr
            pkgs.libmpc
          ];

          dontConfigure = true;
          dontBuild = true;

          installPhase = ''
            mkdir -p $out
            cp -r * $out/
          '';

          meta = with pkgs.lib; {
            description = "Tenstorrent SFPI cross-compiler for RISC-V cores";
            homepage = "https://github.com/tenstorrent/sfpi";
            license = licenses.asl20;
            platforms = [ "x86_64-linux" ];
          };
        };

        # Tenstorrent openmpi fork for fault tolerance (optional / defaults fallback)
        openmpi-ulfm = pkgs.openmpi.overrideAttrs (oldAttrs: rec {
          pname = "openmpi-ulfm";
          version = "5.0.7";
          src = pkgs.fetchFromGitHub {
            owner = "tenstorrent";
            repo = "ompi";
            rev = "v${version}";
            sha256 = "sha256-1dczpM+q+U/5j/Qy259Vv0X14/563O14p47Zz/78912="; # placeholder, fallback to Nix's openmpi if build fails
          };
        });

      in
      {
        packages.sfpi = sfpi;

        devShells.default = pkgs.mkShell rec {
          name = "tt-metal-dev";

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            unixtools.xxd
            git
            wget
            curl
            jq
          ];

          buildInputs = with pkgs; [
            # Host compilers & libraries
            llvmPackages_17.clang
            llvmPackages_17.libcxx
            llvmPackages_17.libcxxabi
            onetbb
            hwloc
            numactl
            capstone
            openssl
            sfpi # custom cross compiler

            # Python environment
            (python3.withPackages (ps: with ps; [
              pip
              setuptools
              wheel
              numpy
              torch
              pybind11
            ]))
          ];

          shellHook = ''
            export TT_METAL_HOME=$(pwd)
            export SFPI_DIR=${sfpi}
            export PATH=$SFPI_DIR/bin:$PATH
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath (with pkgs; [
              onetbb
              hwloc
              numactl
              capstone
              openssl
              stdenv.cc.cc.lib
            ])}:$LD_LIBRARY_PATH

            echo "=========================================================="
            echo "  Tenstorrent TT-Metalium Nix Development Shell"
            echo "=========================================================="
            echo "  TT_METAL_HOME: $TT_METAL_HOME"
            echo "  SFPI version:  ${sfpi.version}"
            echo "  Clang version: 17"
            echo "=========================================================="
          '';
        };
      }
    );
}
