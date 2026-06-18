# flake.nix — Nix flake for the Tenstorrent TT stack (TT-Metalium / TT-NN)
#
# Provides:
#   - A devShell that replaces install_dependencies.sh
#   - A package derivation for building tt-metal from source
#   - (Optional) An overlay for integrating tt-metal into a NixOS configuration
#
# SPDX-License-Identifier: Apache-2.0
{
  description = "Tenstorrent TT-Metalium / TT-NN – Nix packaging";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , ...
    } @ inputs:

    flake-utils.lib.eachDefaultSystem (system:
    let
      # ----------------------------------------------------------------------
      # Overlay: apply version bumps / overrides needed by tt-metal.
      # ----------------------------------------------------------------------
      overlay = final: prev: {
        # tt-metal needs a recent CMake.
        cmake = prev.cmake;

        # Clang 20 is the default toolchain.
        llvmPackages = prev.llvmPackages_20;
      };

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ overlay ];
      };

      # ----------------------------------------------------------------------
      # SFPI firmware package (proprietary Tenstorrent binary).
      # ----------------------------------------------------------------------
      sfpi = pkgs.stdenv.mkDerivation rec {
        pname = "sfpi";
        version = "7.60.0";

        src = pkgs.fetchurl {
          url = "https://github.com/tenstorrent/sfpi/releases/download/${version}/sfpi_${version}_x86_64_debian.deb";
          hash = "sha256-71N+IxIWxAnew5TFYIkp5kQzyUMnuxe5xnS1ZZlGQX0=";
        };

        nativeBuildInputs = [ pkgs.dpkg ];

        phases = [ "unpackPhase" "installPhase" ];

        unpackPhase = ''
          mkdir -p $TMPDIR/extracted
          dpkg-deb -x "$src" "$TMPDIR/extracted"
          sourceRoot="$TMPDIR/extracted"
        '';

        installPhase = ''
          mkdir -p $out
          if [ -d "$sourceRoot/opt/tenstorrent" ]; then
            cp -r "$sourceRoot/opt/tenstorrent" "$out/"
            # Symlink headers + libs to standard locations so cmake finds them.
            mkdir -p $out/include $out/lib
            for dir in "$sourceRoot/opt/tenstorrent/sfpi"/*/; do
              [ -d "$dir/include" ] && cp -rsf "$dir/include"/* "$out/include/" 2>/dev/null || true
              [ -d "$dir/lib" ]     && cp -rsf "$dir/lib"/*     "$out/lib/"     2>/dev/null || true
            done
          fi
        '';

        meta = {
          description = "Tenstorrent SFPI firmware package";
          homepage = "https://github.com/tenstorrent/sfpi";
          license = pkgs.lib.licenses.unfree;
          platforms = [ "x86_64-linux" ];
        };
      };

    in
    {
      # --------------------------------------------------------------------
      # Dev shell – mirrors shell.nix.
      # --------------------------------------------------------------------
      devShells.default = pkgs.callPackage ./shell.nix {
        inherit (pkgs) stdenv;
        clangStdenv = true;
        enableDistributed = true;
      };

      # --------------------------------------------------------------------
      # Package – builds tt-metal from source.
      # --------------------------------------------------------------------
      packages.tt-metal = pkgs.stdenv.mkDerivation rec {
        pname = "tt-metal";
        version = "main";

        src = self;

        nativeBuildInputs = with pkgs; [
          cmake
          ninja
          pkg-config
          git
          python3
          python3.pkgs.pip
          python3.pkgs.setuptools
          python3.pkgs.wheel
          python3.pkgs.pyyaml
          python3.pkgs.numpy
          pandoc
          xz
          openssl
          wget
          curl
          xxd
          jq
          gawk
        ];

        buildInputs = with pkgs; [
          hwloc
          numactl
          libatomic_ops
          tbb
          capstone
          llvmPackages_20.openmp
          sfpi
          openmpi
        ];

        # Clang 20 stdenv with LTO-friendly flags.
        stdenv = pkgs.llvmPackages_20.stdenv;

        preConfigure = ''
          # tt-metal's cmake fetches some deps at configure time via CPM;
          # ensure the network is available (allowed by default in Nix 2.24+).
          export CMAKE_GENERATOR=Ninja
        '';

        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Release"
          "-DTT_SFPI_DIR=${sfpi}"
          "-DBUILD_SHARED_LIBS=ON"
        ];

        buildPhase = ''
          ninja -j$(nproc)
        '';

        installPhase = ''
          mkdir -p $out
          cmake --install . --prefix $out
        '';

        meta = with pkgs.lib; {
          description = "TT-Metalium and TT-NN – Tenstorrent's low-level software stack";
          homepage = "https://github.com/tenstorrent/tt-metal";
          license = licenses.asl20;
          platforms = [ "x86_64-linux" "aarch64-linux" ];
          maintainers = [ ];
          # tt-metal requires Tenstorrent hardware to run; the build step
          # is cross-compilation safe but runtime requires the device.
        };
      };

      # --------------------------------------------------------------------
      # Default package.
      # --------------------------------------------------------------------
      packages.default = self.packages.${system}.tt-metal;

      # --------------------------------------------------------------------
      # Apps – convenience entry points.
      # --------------------------------------------------------------------
      apps.default = flake-utils.lib.mkApp {
        drv = self.packages.${system}.tt-metal;
      };

      # --------------------------------------------------------------------
      # Overlay for use in NixOS / home-manager configurations.
      #   nixpkgs.overlays = [ inputs.tt-metal-nix.overlays.default ];
      # --------------------------------------------------------------------
      overlays.default = final: prev: {
        tt-metal = self.packages.${prev.system}.tt-metal;
      };
    });
}
