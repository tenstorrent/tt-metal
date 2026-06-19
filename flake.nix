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
              pkgs.valgrind
              pkgs.perf-tools
              pkgs.linuxPackages.perf
              pkgs.doxygen
              pkgs.graphviz
              pkgs.texlive.combined.scheme-small
            ];

            TT_METAL_HOME = builtins.toString ./.;
            TT_METAL_ENV = "dev";
            PYTHONPATH = "${pythonEnv}/${pythonEnv.sitePackages}";
            NIXPKGS_ALLOW_UNFREE = "1";

            shellHook = ''
              echo "╔══════════════════════════════════════════════╗"
              echo "║   TT-Metal Development Environment (Nix)     ║"
              echo "╚══════════════════════════════════════════════╝"
              echo "TT_METAL_HOME=$TT_METAL_HOME"
              echo "Python: $(python3 --version 2>/dev/null || true)"
              echo "GCC: $(gcc --version 2>/dev/null | head -1 || true)"
              echo "CMake: $(cmake --version 2>/dev/null | head -1 || true)"
              echo ""

              # Create python venv if not exists
              if [ ! -d "$TT_METAL_HOME/.venv" ]; then
                echo "Creating Python virtual environment..."
                python3 -m venv "$TT_METAL_HOME/.venv"
              fi
              source "$TT_METAL_HOME/.venv/bin/activate" 2>/dev/null || true
              export PATH="$TT_METAL_HOME/.venv/bin:$PATH"
            '';
          };

          # Minimal SFPU-only development
          sfpi = pkgs.mkShell {
            name = "tt-metal-sfpi";
            buildInputs = with pkgs; [
              gcc14
              gnumake
              cmake
              ninja
              pythonEnv
              llvmPkgs.llvm
              llvmPkgs.clang
              git
            ];

            TT_METAL_HOME = builtins.toString ./.;
            TT_METAL_ENV = "sfpi";
            SFPI_ONLY = "1";

            shellHook = ''
              echo "╔══════════════════════════════════════════════╗"
              echo "║   TT-Metal SFPU-Only Environment (Nix)      ║"
              echo "╚══════════════════════════════════════════════╝"
              echo "SFPI_ONLY mode - minimal dependencies"
              echo "TT_METAL_HOME=$TT_METAL_HOME"
            '';
          };
        };

        # FHS compatibility package for NixOS
        packages.fhs = fhsEnv;

        # Apps
        apps.fhs = {
          type = "app";
          program = "${fhsEnv}/bin/tt-metal-fhs";
        };

        # Default app
        apps.default = apps.fhs;

        # Packages
        packages.default = pkgs.stdenv.mkDerivation {
          name = "tt-metal";
          src = ./.;
          buildInputs = commonBuildInputs ++ [ pythonEnv pkgs.openmpi ];
          configurePhase = ''
            cmake -B build -G Ninja \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX=$out \
              -DPYTHON_EXECUTABLE=${pythonEnv}/bin/python3
          '';
          buildPhase = ''
            cmake --build build --target install
          '';
          installPhase = ''
            cmake --install build
          '';
        };

        # Formatter
        formatter = pkgs.nixpkgs-fmt;
      });
}